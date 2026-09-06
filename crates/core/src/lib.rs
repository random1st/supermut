//! supermut-core — fast local inference for mutation generation.
//!
//! Language-binding-free engine shared by the Python (PyO3) and Node
//! (napi-rs) wrappers. Design: `Engine` owns the backend + model (loaded
//! once, GPU-offloaded where available). A `LlamaContext` borrows the
//! model, so it is created per call rather than stored. The supermut
//! workload — one shared file prefix, many short continuations — is served
//! by `generate_batch`: the prefix is decoded once into sequence 0, its KV
//! cache is shared with every continuation, and continuations decode in
//! parallel waves (one token per live sequence per `decode` call).

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

pub type Error = String;
pub type Result<T> = std::result::Result<T, Error>;

fn err(msg: impl std::fmt::Display) -> Error {
    msg.to_string()
}

/// Sequence id holding the shared prefix KV cache inside `generate_batch`.
const SEQ_PREFIX: i32 = 0;
/// How many continuations decode in parallel per wave.
const WAVE_SIZE: usize = 16;

/// llama.cpp's backend is a process-wide singleton with a deinitializing
/// `Drop`; park it in a static so any number of `Engine`s can share it.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn backend() -> Result<&'static LlamaBackend> {
    if BACKEND.get().is_none() {
        // Bindings construct engines on one thread (GIL / JS main thread),
        // so there is no init race in practice.
        let b = LlamaBackend::init().map_err(err)?;
        let _ = BACKEND.set(b);
    }
    Ok(BACKEND.get().expect("backend just initialized"))
}

/// Sampling parameters for one generation call.
#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.05,
            seed: 42,
        }
    }
}

impl SamplingParams {
    fn sampler(&self, seed_offset: u32) -> LlamaSampler {
        if self.temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::top_p(self.top_p, 1),
                LlamaSampler::min_p(self.min_p, 1),
                LlamaSampler::temp(self.temperature),
                LlamaSampler::dist(self.seed.wrapping_add(seed_offset)),
            ])
        }
    }
}

/// Per-continuation decoding state within a wave.
struct Slot {
    seq: i32,
    pos: i32,
    logits_idx: i32,
    sampler: LlamaSampler,
    decoder: encoding_rs::Decoder,
    out: String,
    finished: bool,
}

/// Local GGUF inference engine (llama.cpp, Metal on macOS / CPU elsewhere).
pub struct Engine {
    backend: &'static LlamaBackend,
    model: LlamaModel,
    n_ctx: u32,
    model_path: PathBuf,
}

impl Engine {
    pub fn load(model_path: &Path, n_ctx: u32, n_gpu_layers: u32) -> Result<Self> {
        if !model_path.exists() {
            return Err(format!("model file not found: {model_path:?}"));
        }
        let backend = backend()?;
        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(backend, model_path, &model_params).map_err(err)?;
        Ok(Self {
            backend,
            model,
            n_ctx,
            model_path: model_path.to_path_buf(),
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    pub fn gpu_offload_supported(&self) -> bool {
        self.backend.supports_gpu_offload()
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(err)?
            .len())
    }

    pub fn describe(&self) -> String {
        format!(
            "Engine(model={:?}, n_ctx={}, gpu={})",
            self.model_path.file_name().unwrap_or_default(),
            self.n_ctx,
            self.backend.supports_gpu_offload()
        )
    }

    pub fn generate(
        &self,
        prompt: &str,
        params: SamplingParams,
        stop: &[String],
    ) -> Result<String> {
        let results = self.generate_batch(prompt, &[String::new()], params, stop)?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    /// Batched generation with a shared prefix: the prefix KV cache is
    /// computed once and shared by every continuation; continuations decode
    /// in parallel waves of up to `WAVE_SIZE` sequences.
    pub fn generate_batch(
        &self,
        prefix: &str,
        continuations: &[String],
        params: SamplingParams,
        stop: &[String],
    ) -> Result<Vec<String>> {
        let prefix_tokens = self
            .model
            .str_to_token(prefix, AddBos::Always)
            .map_err(err)?;
        let prefix_len = prefix_tokens.len();
        if prefix_len as u32 + params.max_tokens as u32 >= self.n_ctx {
            return Err(format!(
                "prefix ({prefix_len} tokens) + max_tokens ({}) exceeds n_ctx ({})",
                params.max_tokens, self.n_ctx
            ));
        }
        let last_prefix_tok = *prefix_tokens.last().ok_or_else(|| err("empty prefix"))?;

        // Size the context for the actual batch: a single generation must
        // not pay for WAVE_SIZE sequence streams.
        let wave_size = continuations.len().clamp(1, WAVE_SIZE);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_n_batch(self.n_ctx.max(512))
            .with_n_seq_max(wave_size as u32 + 1)
            // One shared KV buffer: sequences reference the same prefix
            // cells, so copy_kv_cache_seq marks cells instead of copying.
            .with_kv_unified(true);
        let mut ctx = self
            .model
            .new_context(self.backend, ctx_params)
            .map_err(err)?;
        let mut batch = LlamaBatch::new(ctx.n_batch() as usize, wave_size as i32 + 1);

        let after_prefix =
            Self::decode_tokens(&mut ctx, &mut batch, &prefix_tokens, SEQ_PREFIX, 0)?;

        let mut results = Vec::with_capacity(continuations.len());
        for (wave_no, wave) in continuations.chunks(wave_size).enumerate() {
            let mut slots = Vec::with_capacity(wave.len());
            for (j, _cont) in wave.iter().enumerate() {
                let seq = j as i32 + 1;
                ctx.clear_kv_cache_seq(Some(seq as u32), None, None)
                    .map_err(err)?;
                ctx.copy_kv_cache_seq(SEQ_PREFIX, seq, None, None)
                    .map_err(err)?;
                let global_i = wave_no * wave_size + j;
                slots.push(Slot {
                    seq,
                    pos: after_prefix,
                    logits_idx: 0,
                    sampler: params.sampler(global_i as u32),
                    decoder: encoding_rs::UTF_8.new_decoder(),
                    out: String::new(),
                    finished: false,
                });
            }

            // Decode every slot's continuation tail in one batch. Empty
            // continuations re-decode the last prefix token in their own
            // sequence to obtain fresh logits for sampling.
            batch.clear();
            for (slot, cont) in slots.iter_mut().zip(wave.iter()) {
                if cont.is_empty() {
                    ctx.clear_kv_cache_seq(
                        Some(slot.seq as u32),
                        Some(after_prefix as u32 - 1),
                        None,
                    )
                    .map_err(err)?;
                    batch
                        .add(last_prefix_tok, after_prefix - 1, &[slot.seq], true)
                        .map_err(err)?;
                } else {
                    let cont_tokens = self.model.str_to_token(cont, AddBos::Never).map_err(err)?;
                    if prefix_len + cont_tokens.len() + params.max_tokens >= self.n_ctx as usize {
                        return Err(format!(
                            "prefix + continuation ({} tokens) + max_tokens ({}) exceeds n_ctx ({})",
                            prefix_len + cont_tokens.len(),
                            params.max_tokens,
                            self.n_ctx
                        ));
                    }
                    for (i, tok) in cont_tokens.iter().enumerate() {
                        let is_last = i == cont_tokens.len() - 1;
                        batch
                            .add(*tok, slot.pos, &[slot.seq], is_last)
                            .map_err(err)?;
                        slot.pos += 1;
                    }
                }
                slot.logits_idx = batch.n_tokens() - 1;
            }
            ctx.decode(&mut batch).map_err(err)?;

            self.wave_sample_loop(&mut ctx, &mut batch, &mut slots, params.max_tokens, stop)?;
            results.extend(slots.into_iter().map(|s| s.out));
        }
        Ok(results)
    }

    fn decode_tokens(
        ctx: &mut LlamaContext<'_>,
        batch: &mut LlamaBatch,
        tokens: &[LlamaToken],
        seq: i32,
        start_pos: i32,
    ) -> Result<i32> {
        let n_batch = ctx.n_batch() as usize;
        let mut pos = start_pos;
        for chunk in tokens.chunks(n_batch) {
            batch.clear();
            for (i, tok) in chunk.iter().enumerate() {
                let is_last = i == chunk.len() - 1;
                batch.add(*tok, pos, &[seq], is_last).map_err(err)?;
                pos += 1;
            }
            ctx.decode(batch).map_err(err)?;
        }
        Ok(pos)
    }

    /// Trim `slot.out` at the earliest stop marker; returns true if one hit.
    fn hit_stop(slot: &mut Slot, stop: &[String]) -> bool {
        if stop.is_empty() {
            return false;
        }
        if let Some(idx) = stop.iter().filter_map(|s| slot.out.find(s.as_str())).min() {
            slot.out.truncate(idx);
            return true;
        }
        false
    }

    fn wave_sample_loop(
        &self,
        ctx: &mut LlamaContext<'_>,
        batch: &mut LlamaBatch,
        slots: &mut [Slot],
        max_tokens: usize,
        stop: &[String],
    ) -> Result<()> {
        for _ in 0..max_tokens {
            batch.clear();
            let mut scheduled = 0i32;
            for slot in slots.iter_mut() {
                if slot.finished {
                    continue;
                }
                let tok = slot.sampler.sample(ctx, slot.logits_idx);
                slot.sampler.accept(tok);
                if self.model.is_eog_token(tok) {
                    slot.finished = true;
                    continue;
                }
                if let Ok(piece) = self
                    .model
                    .token_to_piece(tok, &mut slot.decoder, false, None)
                {
                    slot.out.push_str(&piece);
                }
                if Self::hit_stop(slot, stop) {
                    slot.finished = true;
                    continue;
                }
                batch.add(tok, slot.pos, &[slot.seq], true).map_err(err)?;
                slot.pos += 1;
                slot.logits_idx = scheduled;
                scheduled += 1;
            }
            if scheduled == 0 {
                break;
            }
            ctx.decode(batch).map_err(err)?;
        }
        Ok(())
    }
}

//! Thin napi-rs wrapper over supermut-core. All engine logic lives in the
//! core crate, shared with the Python binding.

use std::path::Path;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use supermut_core::{Engine as CoreEngine, SamplingParams};

fn rt(e: supermut_core::Error) -> Error {
    Error::from_reason(e)
}

#[napi(object)]
#[derive(Default)]
pub struct GenerateOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub seed: Option<u32>,
    pub stop: Option<Vec<String>>,
}

impl GenerateOptions {
    fn params(&self) -> SamplingParams {
        let d = SamplingParams::default();
        SamplingParams {
            max_tokens: self.max_tokens.map_or(d.max_tokens, |v| v as usize),
            temperature: self.temperature.map_or(d.temperature, |v| v as f32),
            top_p: self.top_p.map_or(d.top_p, |v| v as f32),
            min_p: self.min_p.map_or(d.min_p, |v| v as f32),
            seed: self.seed.unwrap_or(d.seed),
        }
    }

    fn stop(&self) -> Vec<String> {
        self.stop.clone().unwrap_or_default()
    }
}

#[napi]
pub struct Engine {
    inner: CoreEngine,
}

#[napi]
impl Engine {
    #[napi(constructor)]
    pub fn new(model_path: String, n_ctx: Option<u32>, n_gpu_layers: Option<u32>) -> Result<Self> {
        let inner = CoreEngine::load(
            Path::new(&model_path),
            n_ctx.unwrap_or(4096),
            n_gpu_layers.unwrap_or(1_000_000),
        )
        .map_err(rt)?;
        Ok(Self { inner })
    }

    #[napi]
    pub fn generate(&self, prompt: String, options: Option<GenerateOptions>) -> Result<String> {
        let opts = options.unwrap_or_default();
        self.inner
            .generate(&prompt, opts.params(), &opts.stop())
            .map_err(rt)
    }

    #[napi]
    pub fn generate_batch(
        &self,
        prefix: String,
        continuations: Vec<String>,
        options: Option<GenerateOptions>,
    ) -> Result<Vec<String>> {
        let opts = options.unwrap_or_default();
        self.inner
            .generate_batch(&prefix, &continuations, opts.params(), &opts.stop())
            .map_err(rt)
    }

    #[napi]
    pub fn count_tokens(&self, text: String) -> Result<u32> {
        Ok(self.inner.count_tokens(&text).map_err(rt)? as u32)
    }

    #[napi(getter)]
    pub fn model_path(&self) -> String {
        self.inner.model_path().display().to_string()
    }

    #[napi(getter)]
    pub fn n_ctx(&self) -> u32 {
        self.inner.n_ctx()
    }

    #[napi(getter)]
    pub fn gpu_offload_supported(&self) -> bool {
        self.inner.gpu_offload_supported()
    }
}

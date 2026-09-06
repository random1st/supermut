//! Thin PyO3 wrapper over supermut-core. All engine logic lives in the
//! core crate, shared with the Node binding.

use std::path::PathBuf;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use supermut_core::{Engine as CoreEngine, SamplingParams};

fn rt(e: supermut_core::Error) -> PyErr {
    PyRuntimeError::new_err(e)
}

#[pyclass(unsendable)]
struct Engine {
    inner: CoreEngine,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (model_path, n_ctx=4096, n_gpu_layers=1_000_000, n_threads=None))]
    fn py_new(
        model_path: PathBuf,
        n_ctx: u32,
        n_gpu_layers: u32,
        n_threads: Option<i32>,
    ) -> PyResult<Self> {
        let _ = n_threads; // reserved: applied per-context in a later revision
        Ok(Self {
            inner: CoreEngine::load(&model_path, n_ctx, n_gpu_layers).map_err(rt)?,
        })
    }

    #[pyo3(signature = (prompt, max_tokens=256, temperature=0.8, top_p=0.95, min_p=0.05, seed=42, stop=vec![]))]
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        min_p: f32,
        seed: u32,
        stop: Vec<String>,
    ) -> PyResult<String> {
        let params = SamplingParams {
            max_tokens,
            temperature,
            top_p,
            min_p,
            seed,
        };
        self.inner.generate(prompt, params, &stop).map_err(rt)
    }

    #[pyo3(signature = (prefix, continuations, max_tokens=256, temperature=0.8, top_p=0.95, min_p=0.05, seed=42, stop=vec![]))]
    #[allow(clippy::too_many_arguments)]
    fn generate_batch(
        &self,
        prefix: &str,
        continuations: Vec<String>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        min_p: f32,
        seed: u32,
        stop: Vec<String>,
    ) -> PyResult<Vec<String>> {
        let params = SamplingParams {
            max_tokens,
            temperature,
            top_p,
            min_p,
            seed,
        };
        self.inner
            .generate_batch(prefix, &continuations, params, &stop)
            .map_err(rt)
    }

    fn count_tokens(&self, text: &str) -> PyResult<usize> {
        self.inner.count_tokens(text).map_err(rt)
    }

    #[getter]
    fn model_path(&self) -> String {
        self.inner.model_path().display().to_string()
    }

    #[getter]
    fn n_ctx(&self) -> u32 {
        self.inner.n_ctx()
    }

    #[getter]
    fn gpu_offload_supported(&self) -> bool {
        self.inner.gpu_offload_supported()
    }

    fn __repr__(&self) -> String {
        self.inner.describe()
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    Ok(())
}

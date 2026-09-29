//! `timing-predict` as a wasm32 module, for predicting in a browser.
//!
//! Measured kernel grids arrive as the kernel API's config documents and the
//! files an arch block names (model config, routing artifact) as strings, so
//! nothing reads a file, a database or Python.
//!
//! JS surface (wasm-bindgen; every JSON value is passed as a string):
//! - `version()` -> `{"sim_commit", "kernel_data_format"}`: the checkout this
//!   module was built from and the kernel data it reads.
//! - `new Predictor(config, kernel_data, files)` builds the model (the kernel
//!   cost caches) once. `config` is a predict config (`{"arch", "gpu",
//!   "backends"?}`), `kernel_data` the config documents
//!   (`GET /kernels/{kind}/configs/{hash}` responses, as an array or under
//!   `configs`), `files` `{path: contents}`.
//!   Every selector is accepted: `iter`, `speculative_iter`, `attn`, `ffn`.
//! - `predictor.info()`: the case shape (selector, groups per case, context
//!   bound, draft tokens). `predictor.manifest()`: `{"sections": [{section,
//!   slots, nodes, node_labels}]}`, one cost tree per section.
//! - `predictor.predict(cases)` -> `{"cases": [{"sections": [{section, layer,
//!   total_time_ms, slot_time_ms, node_time_ms}]}]}`: per case, the rows native
//!   `timing-predict` writes to its cost_log (one `iter` row for the iter
//!   selectors, `attn` for attn, `prologue` .. `epilogue` for ffn). An invalid
//!   case throws `case N: <reason>` before any is costed.
//! - `last_panic()`: the message of the panic that trapped the last call. A
//!   panic aborts (wasm32 has no unwinding): the call throws
//!   `RuntimeError: unreachable`, and the instance must be discarded.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use serde_json::Value;
use simulator::common::input_files;
use simulator::timing::bridge::{KernelData, PerfApiBridge, KERNEL_DATA_FORMAT};
use simulator::timing_predict::Predictor as Inner;
use wasm_bindgen::prelude::*;

thread_local! {
    static LAST_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let message = info.to_string();
        LAST_PANIC.with(|p| *p.borrow_mut() = Some(message));
    }));
}

#[wasm_bindgen]
pub fn last_panic() -> Option<String> {
    LAST_PANIC.with(|p| p.borrow().clone())
}

#[wasm_bindgen]
pub fn version() -> String {
    serde_json::json!({
        "sim_commit": option_env!("SERVINGSTUDIO_SIM_COMMIT"),
        "kernel_data_format": KERNEL_DATA_FORMAT,
    })
    .to_string()
}

fn js_err(err: anyhow::Error) -> JsError {
    JsError::new(&format!("{err:#}"))
}

fn to_json(value: &impl serde::Serialize) -> String {
    serde_json::to_string(value).expect("predictor output serializes")
}

#[wasm_bindgen]
pub struct Predictor {
    inner: Inner,
}

#[wasm_bindgen]
impl Predictor {
    #[wasm_bindgen(constructor)]
    pub fn new(config: &str, kernel_data: &str, files: &str) -> Result<Predictor, JsError> {
        install_panic_hook();
        build(config, kernel_data, files).map_err(js_err)
    }

    pub fn info(&self) -> String {
        to_json(&self.inner.info())
    }

    pub fn manifest(&self) -> String {
        to_json(&self.inner.manifest())
    }

    pub fn predict(&mut self, cases: &str) -> Result<String, JsError> {
        let cases: Value = serde_json::from_str(cases)
            .context("parsing cases")
            .map_err(js_err)?;
        let out = self.inner.predict(cases).map_err(js_err)?;
        Ok(to_json(&serde_json::json!({ "cases": out })))
    }
}

fn build(config: &str, kernel_data: &str, files: &str) -> anyhow::Result<Predictor> {
    let mut config: Value = serde_json::from_str(config).context("parsing config")?;
    let config = config.as_object_mut().context("config must be an object")?;
    let arch = config.remove("arch").context("config.arch is required")?;
    let gpu = config
        .remove("gpu")
        .and_then(|g| g.as_str().map(str::to_string))
        .context("config.gpu is required")?;
    let backends = config.remove("backends");
    let data = KernelData::from_json(kernel_data).context("parsing kernel data")?;
    let files: HashMap<PathBuf, String> = serde_json::from_str::<HashMap<String, String>>(files)
        .context("parsing files")?
        .into_iter()
        .map(|(path, text)| (PathBuf::from(path), text))
        .collect();
    let bridge = PerfApiBridge::kernel_data(Arc::new(data));
    let inner = input_files::with_files(files, || Inner::build(arch, &gpu, backends, &bridge))?;
    Ok(Predictor { inner })
}

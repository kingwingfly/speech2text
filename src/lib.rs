//! speech2text — SenseVoice Mandarin transcription.
//!
//! This crate builds two ways:
//! - **ssr** (native): the CLI + realtime web server + the ONNX/ffmpeg pipeline.
//! - **hydrate** (wasm): the browser UI that records the mic and streams it.
//!
//! The native STT modules (`asr`/`audio`/`features`/`server`) are gated behind
//! `ssr` so the wasm build never pulls `ort`/`ffmpeg-next`/`knf-rs`.

pub mod app;
pub mod capture;

#[cfg(feature = "ssr")]
pub mod asr;
#[cfg(feature = "ssr")]
pub mod audio;
#[cfg(feature = "ssr")]
pub mod features;
#[cfg(feature = "ssr")]
pub mod server;

#[cfg(feature = "hydrate")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn hydrate() {
    use crate::app::App;
    console_error_panic_hook::set_once();
    leptos::mount::hydrate_body(App);
}

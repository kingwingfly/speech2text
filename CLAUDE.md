# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A tool that transcribes Mandarin Chinese speech using the **SenseVoiceSmall** ONNX model, run through `ort` with CUDA GPU acceleration. It has two faces:
- a **CLI** (binary named `stt`) for file transcription (`preprocess`, `transcribe`), and
- a **realtime web app** (`stt realtime`) — a **Leptos SSR + Tailwind** front-end that streams mic audio over a WebSocket and streams paced text back, for live reading.

See `README.md` for user-facing docs, model provenance, and the pipeline diagram.

The crate is both a `lib` (`crate-type = ["cdylib", "rlib"]`) and a `bin` (`stt`). `cargo-leptos` compiles it twice: native **`ssr`** (server + CLI + the ONNX/ffmpeg pipeline) and wasm **`hydrate`** (browser UI). The native STT deps (`ort`, `ffmpeg-next`, `knf-rs`, …) are **optional and gated behind `ssr`** so the wasm build never pulls them; likewise `web-sys`/`wasm-bindgen` are `hydrate`-only. When editing, keep this feature split intact — anything touching `asr`/`audio`/`features`/`server` is `#[cfg(feature = "ssr")]`; anything touching `web_sys` is `hydrate`-only (see the stub pattern in `capture.rs`).

## Commands

```sh
# CLI (the plain file-transcription path — no leptos tooling needed)
cargo build --release            # release is what you want; opt-level 3 + thin LTO
cargo run --bin stt -- transcribe --input clean.wav        # run against clean audio
cargo run --bin stt -- transcribe --input lecture.aac --denoise   # denoise + transcribe
cargo run --bin stt -- preprocess --input in.aac --output clean.wav
cargo check --features ssr       # fast type-check of the server/CLI side
cargo clippy --features ssr
RUST_LOG=debug cargo run --bin stt -- ...   # per-module logging; metadata parsing at debug

# Realtime web app (needs cargo-leptos + the wasm32-unknown-unknown target)
cargo leptos watch               # dev server w/ hot reload; bare `stt` defaults to `realtime`
cargo leptos build               # builds wasm + Tailwind + server binary
stt realtime --addr 127.0.0.1:3000   # run the built server explicitly
stt realtime --password hunter2      # gate app+/ws behind HTTP Basic auth (user "stt"); or STT_REALTIME_PASSWORD
```

There is no test suite. Verify the CLI against the sample files in `audio/` (gitignored). Verify the web app with `cargo leptos build` (both feature builds must compile) then `cargo leptos watch` and a browser at http://127.0.0.1:3000 — actual mic streaming needs a real browser + microphone.

## Build prerequisites (will not compile/run without these)

- **FFmpeg dev libraries** (libavcodec/format/filter/util/swresample) — `ffmpeg-next` links against them.
- **C++ toolchain + CMake** — `knf-rs` builds kaldi-native-fbank from source.
- **onnxruntime-gpu shared lib** at runtime. `ort` uses `load-dynamic`; `main.rs` defaults `ORT_DYLIB_PATH` to `/usr/lib/libonnxruntime.so` when unset. On a machine with a different build, set `ORT_DYLIB_PATH` yourself.
- Rust **edition 2024**.
- For the web app only: **`cargo-leptos`** and the **`wasm32-unknown-unknown`** target. cargo-leptos fetches the Tailwind CLI and wasm-bindgen itself (no npm/node). Tailwind here is **v4** — config lives in `style/tailwind.css` (`@import "tailwindcss"; @source "../src";`), not a `tailwind.config.js`. The Tailwind CLI version is pinned in `.env` via `LEPTOS_TAILWIND_VERSION` (cargo-leptos reads dotenvs) because resolving GitHub's "latest" can 404.
- Web mic access needs a **secure context**: over plain HTTP the browser only exposes the mic on `localhost`/`127.0.0.1`. The app serves **plain HTTP only by design** — TLS is terminated by a reverse proxy in front (Pingora/Cloudflare/nginx), which must also upgrade the `/ws` WebSocket. Do **not** add TLS to the app. `capture.rs` degrades gracefully (a status message) when the context is insecure — it must never `.expect()` on the mic path.

## Architecture

The core STT data flow is one direction: audio bytes → f32 PCM → FBank features → ONNX logits → text. Modules split by feature:

**Native / `ssr` (server + CLI):**

- **`lib.rs`** — crate root: declares modules (STT ones gated behind `ssr`), and the `#[wasm_bindgen] hydrate()` entrypoint for the browser build.
- **`main.rs`** — clap CLI (`preprocess`, `transcribe`, `realtime`; a bare invocation defaults to `realtime` so `cargo leptos watch` works). Sets `ORT_DYLIB_PATH` and configures tracing (silences onnxruntime info logs via `ort::logging=warn`). Uses the library crate as `stt::…`.
- **`server/mod.rs`** — the realtime server: loads the `Recognizer` once, shares it as `Arc<Mutex<Recognizer>>` (inference runs via `spawn_blocking`), and serves the Leptos SSR routes + `/ws` on axum. `Engine::infer` wraps `Recognizer::transcribe_once`.
- **`server/ws.rs`** — the WebSocket handler: buffers incoming f32 PCM, runs energy-based VAD to emit live **partials** (~1×/s) and **finalize** a line on a pause (or at the `max_secs` cap), streaming `SttMsg` JSON back.

**Shared (both builds):**

- **`app.rs`** — the Leptos UI (`App`/`Realtime` components, SSR `shell`) and the `SttMsg` wire type. The paced reading pane, font/dwell sliders, backlog badge, and live-preview line live here. Has **no** native deps.
- **`capture.rs`** — browser mic capture + WebSocket streaming + the paced-release pump. All `web_sys` code is behind `hydrate`; an `ssr` stub module mirrors the signatures so `app.rs` compiles both ways. Mic uses a 16 kHz `AudioContext` + `ScriptProcessorNode` (pure web-sys, no JS worklet) and sends raw f32 LE PCM.
- **`audio.rs`** — all ffmpeg work. `decode()` builds a filter graph and returns 16 kHz mono f32 PCM. The user's filter chain is always suffixed with `aresample=16000,aformat=...=flt:mono` because ffmpeg 8's `abuffersink` ignores post-init format setters — the output format is pinned *inside* the graph, not on the sink. `split_segments()` chunks long audio (~15 s windows) by snapping each cut to the lowest-energy point within a search radius so words aren't severed mid-syllable.
- **`features.rs`** — the SenseVoice front-end, which is NOT baked into the model: 80-dim Kaldi FBank (`knf-rs`) → LFR (low-frame-rate stacking, `apply_lfr`) → CMVN (`apply_cmvn`). LFR/CMVN parameters come from model metadata, not constants.
- **`asr.rs`** — `Recognizer`: model acquisition, the `ort` session, chunked inference, and CTC greedy decode (argmax per frame → collapse repeats → drop blank/special tokens). `detokenize()` handles SentencePiece word boundaries (`\u{2581}`) and strips SenseVoice markup tokens (`<|zh|>`, `<|NEUTRAL|>`, etc.).

## Key design point: metadata-driven front-end

The acoustic front-end must stay in sync with how the model was trained. Rather than hardcoding it, `parse_meta()` in `asr.rs` reads everything from the ONNX model's embedded custom metadata at load time: CMVN stats (`neg_mean`/`inv_stddev`), LFR window/shift, per-language ids (`lang_zh`, `lang_en`, …), ITN flags (`with_itn`/`without_itn`), blank id, and the `normalize_samples` flag that decides whether samples are scaled by 1.0 or 32768.0 before FBank. When touching `features.rs` or the tensor inputs in `run_chunk`, keep them consistent with what `Meta` provides — do not reintroduce hardcoded acoustic constants.

The four ONNX inputs are positional-by-index (`input_names[0..4]` = features, lengths, language id, text-norm id); the model is single-output.

## Model files

`model.int8.onnx` (default) / `model.onnx` (`--fp32`) plus `tokens.txt` are downloaded from HF repo `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` into the HF cache on first run, or loaded from `--model-dir`. They are gitignored (`*.onnx`, `*.onnx_data`) and never committed.

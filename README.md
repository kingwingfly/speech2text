# speech2text — Chinese speech transcription CLI

A command-line tool that transcribes long-form **Mandarin Chinese** speech from an
**AAC** audio file using a local **ONNX** model, run through the
[`ort`](https://crates.io/crates/ort) crate with **CUDA GPU acceleration**
(`onnxruntime-gpu`). Built and tuned for general-domain content such as
physics/mechanics lectures (理科), on a 6 GB-VRAM RTX 2060 with 16 GB system RAM.

## Model

**SenseVoiceSmall** (FunAudioLLM), in the ONNX packaging published by the
sherpa-onnx project:

- Repo: `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` on Hugging Face
- Files used:
  - `model.int8.onnx` (~240 MB, quantized) — default, or `model.onnx` (~900 MB, fp32)
  - `tokens.txt` — line-indexed token vocabulary (token id == line number)

### Why this model

- **Non-autoregressive.** One ONNX forward pass per audio chunk produces a full
  token-logits sequence; transcription is a simple **CTC greedy decode** (argmax
  per frame, collapse repeats, drop blanks). No autoregressive decoder loop, no
  beam search, no KV cache to manage.
- **Tiny VRAM footprint.** ~1–1.5 GB at inference — comfortable on a 6 GB 2060,
  leaving room for everything else.
- **Multilingual** (zh / en / ja / ko / yue) with strong Mandarin accuracy; mixed
  Chinese/English technical terms in physics audio are handled well.
- **Ships the exact artifacts requested**: a single ONNX model file plus a plain
  token file.

Trade-off: SenseVoice consumes **80-dim Kaldi FBank features with CMVN
normalization**, so the audio front-end is implemented in this tool rather than
baked into the model.

## Pipeline

```
                        ┌──────────── preprocess (optional) ─────────────┐
AAC / any audio ──▶ ffmpeg-next decode ─▶ denoise + enhance filter graph ─▶ 16 kHz mono s16 WAV
                        └────────────────────────────────────────────────┘
                                                  │
                                                  ▼  (or skip straight to here with clean audio)
clean WAV / AAC
  └─▶ decode + resample         ffmpeg-next ─▶ 16 kHz mono f32 PCM
  └─▶ chunk                     ~15 s windows, cuts snapped to the quietest nearby point
  └─▶ features                  80-dim kaldi FBank (knf-rs) ─▶ LFR (low frame rate) ─▶ CMVN
  └─▶ inference                 ort + CUDAExecutionProvider, one Run per chunk
  └─▶ decode                    CTC greedy: argmax ▸ collapse repeats ▸ strip blanks/special
  └─▶ detokenize                map ids ▸ tokens.txt, join, clean SenseVoice markup
  └─▶ output                    --output <file> or stdout
```

The model also takes auxiliary scalar inputs: a **language id** (`auto`/`zh`/…) and
a **text-normalization flag** (ITN on/off). The tool sets language to Chinese and
ITN on by default (disable ITN with `--no-itn`).

## Preprocessing (denoise + voice enhancement)

Optional, powered by **`ffmpeg-next`**. It decodes the input, runs a speech-focused
filter graph, and writes a **16 kHz mono 16-bit PCM WAV** that the transcription
pipeline can consume directly. Skip it if your audio is already clean.

The default filter chain (FFmpeg `af`), tuned for noisy lecture recordings:

```
highpass=f=80,            # drop sub-bass rumble / handling noise
afftdn=nf=-25,            # FFT spectral denoise (noise floor at -25 dB)
speechnorm=e=12.5:r=0.0001,  # bring up quiet speech without clipping
lowpass=f=7800,           # tame hiss just under the 8 kHz Nyquist
aresample=16000,
aformat=sample_fmts=s16:channel_layouts=mono
```

Two ways to use it:

- **`preprocess` subcommand** — produce a reusable clean WAV explicitly, then
  transcribe it later.
- **`--denoise` flag on `transcribe`** — run the same chain into a temp WAV and feed
  it straight to the model in one shot (use `--keep-clean <FILE>` to also save it).

## Usage

```sh
# 1) Explicit two-step
stt preprocess --input lecture.aac --output clean.wav
stt transcribe --input clean.wav --output out.txt

# 2) One shot: denoise then transcribe
stt transcribe --input lecture.aac --denoise --output out.txt

# 3) Already-clean audio: skip preprocessing entirely
stt transcribe --input clean.wav

# options
stt transcribe --input lecture.aac --language zh   # default; also auto, en, ja, ko, yue
stt transcribe --input lecture.aac --fp32          # full-precision model.onnx
stt transcribe --input lecture.aac --cpu           # force CPU execution provider
```

### `preprocess`

| Flag | Description | Default |
|---|---|---|
| `--input <FILE>` | Input audio (AAC or any ffmpeg-decodable format) | *(required)* |
| `--output <FILE>` | Output clean WAV (16 kHz mono s16) | *(required)* |
| `--filter <STR>` | Override the FFmpeg `af` filter chain | built-in chain above |

### `transcribe`

| Flag | Description | Default |
|---|---|---|
| `--input <FILE>` | Input audio (clean WAV, or AAC/any if `--denoise`) | *(required)* |
| `--output <FILE>` | Output transcript file | stdout |
| `--denoise` | Run the denoise/enhance chain before transcribing | off |
| `--keep-clean <FILE>` | With `--denoise`, also save the cleaned WAV | discarded |
| `--filter <STR>` | FFmpeg `af` chain used by `--denoise` | built-in chain |
| `--language <LANG>` | `auto`, `zh`, `en`, `ja`, `ko`, `yue` | `zh` |
| `--no-itn` | Disable inverse text normalization (keep spoken-form numbers) | off (ITN on) |
| `--fp32` | Use `model.onnx` instead of `model.int8.onnx` | off |
| `--cpu` | Disable CUDA, run on CPU | off |
| `--model-dir <DIR>` | Load `model[.int8].onnx` + `tokens.txt` locally instead of downloading | download |

## Realtime web app (`realtime`)

A browser front-end for **live** transcription, aimed at readers who need large,
paced text (e.g. hard-of-hearing elders). The browser records the mic, streams
16 kHz PCM to the server over a WebSocket, and the server streams text back.

- **Live partials + finalize:** SenseVoice is an offline model, so the server
  re-transcribes the in-progress utterance ~once a second (a "partial") and
  finalizes a line when it detects a pause (energy-based VAD).
- **Paced reading pane:** finalized lines are released one at a time, each held on
  screen for an adjustable **dwell time**, so fast speech doesn't scroll away
  before it can be read. A `⏳ N 等待中` badge shows the backlog. Font size and
  dwell time are adjustable live; a dim "listening…" line previews the current
  partial.

The UI is a **Leptos SSR** app styled with **Tailwind CSS**, built by
[`cargo-leptos`](https://github.com/leptos-rs/cargo-leptos).

```sh
# Dev (rebuilds on change, sets up wasm + Tailwind automatically):
cargo leptos watch                 # bare `stt` binary defaults to `realtime`

# On the same machine, open http://127.0.0.1:3000 and click 开始.

# To use it from another device (phone/tablet on the LAN), bind broadly and put
# an HTTPS-terminating reverse proxy in front (see the note below):
stt realtime --addr 0.0.0.0:3000
```

> **Microphone requires a secure context.** Browsers only expose the mic over
> **HTTPS**, or on `http://localhost` / `127.0.0.1` (the server machine itself).
> This app speaks **plain HTTP only** — terminate TLS at a reverse proxy in front
> of it (**Pingora**, **Cloudflare**, nginx, Caddy, …) and forward to
> `stt realtime`, making sure the proxy also upgrades the `/ws` WebSocket. Without
> a secure context the page shows a clear message instead of recording.

| Flag | Description | Default |
|---|---|---|
| `--addr <HOST:PORT>` | Address to bind | Leptos config (`127.0.0.1:3000`) |
| `--language <LANG>` | `auto`, `zh`, `en`, `ja`, `ko`, `yue` | `zh` |
| `--no-itn` | Disable inverse text normalization | off (ITN on) |
| `--fp32` / `--cpu` / `--model-dir <DIR>` | As in `transcribe` | — |
| `--silence-ms <MS>` | Trailing silence that finalizes an utterance | `600` |
| `--partial-ms <MS>` | How often to emit a live partial | `900` |
| `--max-secs <S>` | Hard cap on an utterance before a forced finalize | `15` |
| `--vad-threshold <F>` | RMS threshold above which audio counts as speech | `0.012` |
| `--password <PW>` | Require this password (HTTP Basic auth, username `stt`); also read from `STT_REALTIME_PASSWORD` | none (open) |

### Password protection

Pass `--password <PW>` (or set `STT_REALTIME_PASSWORD`) to gate the whole app —
the SSR pages **and** the `/ws` audio stream — behind **HTTP Basic auth**. The
browser prompts once when the page loads; that authenticated response hands back
a session cookie, which the browser then sends on the WebSocket handshake, so
pressing **Start** does not ask for the password again and the elderly user only
logs in a single time. The username is fixed to **`stt`**; only the password is
checked. The session lasts until the browser closes or the server restarts.
Leave it unset to run open (e.g. on a trusted LAN). Since the app speaks plain
HTTP, put it behind the HTTPS-terminating reverse proxy above so the password
isn't sent in the clear.

The Tailwind version cargo-leptos downloads is pinned in `.env`
(`LEPTOS_TAILWIND_VERSION`) so the build doesn't depend on GitHub's "latest"
release resolving to a published Linux binary.

## First-run model download

On first run the tool downloads `model.int8.onnx` (or `model.onnx`) and `tokens.txt`
from the Hugging Face repo via `hf-hub`, into the standard HF cache
(`~/.cache/huggingface/hub/`). Subsequent runs load straight from cache. To run
fully offline, point `--model-dir` at a directory holding the two files.

The CMVN statistics, LFR window/shift, language ids, ITN ids, and sample scaling
are all read from the ONNX model's embedded metadata at load time, so the
front-end stays in sync with the model.

## Performance

On an RTX 2060 (6 GB) with the int8 model, a ~37-minute recording transcribes in
about a minute end-to-end (decode + ~150 chunks), i.e. ~35× realtime.

## Requirements

- Rust (edition 2024)
- An ONNX Runtime **GPU** shared library reachable by `ort` (load-dynamic). The
  tool defaults `ORT_DYLIB_PATH` to `/usr/lib/libonnxruntime.so`; set it yourself
  to point at a different `onnxruntime-gpu` build.
- NVIDIA driver + CUDA/cuDNN compatible with the chosen onnxruntime-gpu build.
- **FFmpeg development libraries** (`libavcodec`, `libavformat`, `libavfilter`,
  `libavutil`, `libswresample`) for `ffmpeg-next` to link against — e.g.
  `ffmpeg` on Arch, `libavfilter-dev` & friends on Debian/Ubuntu.
- A **C++ toolchain and CMake** at build time: `knf-rs` compiles
  kaldi-native-fbank from source.
- **For the `realtime` web app only:** [`cargo-leptos`](https://github.com/leptos-rs/cargo-leptos)
  and the `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`).
  cargo-leptos fetches the Tailwind CLI and wasm-bindgen tooling itself — no
  npm/node required. The plain CLI (`preprocess`/`transcribe`) builds without any
  of this via `cargo build --release`.

> Note: `knf-rs` computes FBank with kaldi's default `povey` window (SenseVoice
> was trained with `hamming`), and its high-level helper applies a per-utterance
> mean subtraction on top of the model's own CMVN. In practice SenseVoice is
> robust to both — transcripts are equivalent to a hamming-windowed front-end.

## Crates

| Concern | Crate |
|---|---|
| CLI | `clap` (derive) |
| ONNX inference | `ort` (CUDA execution provider) |
| Audio decode / resample / denoise | `ffmpeg-next` (avcodec, avformat, avfilter, swresample) |
| WAV read/write | `hound` |
| FBank features | `knf-rs` (kaldi-native-fbank C++ bindings) |
| Model download | `hf-hub` (0.5, blocking `ureq` backend) |
| Errors / logging | `anyhow`, `tracing` |
| Realtime web UI | `leptos` (SSR + hydrate), Tailwind CSS, `cargo-leptos` |
| Realtime server / WebSocket | `axum`, `tokio`, `leptos_axum` |
| Browser mic capture | `web-sys` (Web Audio + WebSocket, via wasm) |

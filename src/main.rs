//! speech2text CLI (`stt`): file transcription plus a realtime web server.
//!
//! The transcription pipeline and web server live in the `speech2text` library
//! crate; this binary is the clap front-end.

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::io::Write;
use std::path::PathBuf;
use stt::asr::{ModelSource, Recognizer};
use stt::server::{ServeOpts, VadConfig};
use stt::{audio, server};

#[derive(Parser)]
#[command(
    name = "stt",
    version,
    about = "Chinese speech transcription with SenseVoice ONNX"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Denoise + voice-enhance an audio file into a clean 16 kHz mono 16-bit WAV.
    Preprocess(PreprocessArgs),
    /// Transcribe an audio file to text.
    Transcribe(TranscribeArgs),
    /// Start the realtime STT web server (Leptos SSR UI + mic streaming).
    Realtime(RealtimeArgs),
}

#[derive(Args)]
struct PreprocessArgs {
    /// Input audio (AAC or any ffmpeg-decodable format).
    #[arg(short, long)]
    input: PathBuf,
    /// Output clean WAV (16 kHz mono, 16-bit PCM).
    #[arg(short, long)]
    output: PathBuf,
    /// Override the FFmpeg `af` filter chain.
    #[arg(long, default_value = audio::DENOISE_FILTER)]
    filter: String,
}

#[derive(Args)]
struct TranscribeArgs {
    /// Input audio (clean WAV, or AAC/any when used with --denoise).
    #[arg(short, long)]
    input: PathBuf,
    /// Output transcript file (defaults to stdout).
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Run denoising/voice-enhancement before transcribing.
    #[arg(long)]
    denoise: bool,
    /// With --denoise, also save the cleaned WAV here.
    #[arg(long)]
    keep_clean: Option<PathBuf>,
    /// FFmpeg `af` filter chain used by --denoise.
    #[arg(long, default_value = audio::DENOISE_FILTER)]
    filter: String,
    /// Spoken language: auto, zh, en, ja, ko, yue.
    #[arg(long, default_value = "zh")]
    language: String,
    /// Disable inverse text normalization (keep spoken-form numbers).
    #[arg(long)]
    no_itn: bool,
    /// Use the full-precision model.onnx instead of model.int8.onnx.
    #[arg(long)]
    fp32: bool,
    /// Force CPU execution (disable CUDA).
    #[arg(long)]
    cpu: bool,
    /// Load model.onnx/tokens.txt from a local directory instead of downloading.
    #[arg(long)]
    model_dir: Option<PathBuf>,
}

#[derive(Args)]
struct RealtimeArgs {
    /// Address to bind, e.g. 127.0.0.1:3000 (defaults to the Leptos config).
    #[arg(long)]
    addr: Option<String>,
    /// Spoken language: auto, zh, en, ja, ko, yue.
    #[arg(long, default_value = "zh")]
    language: String,
    /// Disable inverse text normalization (keep spoken-form numbers).
    #[arg(long)]
    no_itn: bool,
    /// Use the full-precision model.onnx instead of model.int8.onnx.
    #[arg(long)]
    fp32: bool,
    /// Force CPU execution (disable CUDA).
    #[arg(long)]
    cpu: bool,
    /// Load model.onnx/tokens.txt from a local directory instead of downloading.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Trailing silence (ms) that finalizes an utterance.
    #[arg(long, default_value_t = 600)]
    silence_ms: u32,
    /// How often (ms) to emit a live partial while speech continues.
    #[arg(long, default_value_t = 900)]
    partial_ms: u32,
    /// Hard cap (seconds) on an utterance before a forced finalize.
    #[arg(long, default_value_t = 15.0)]
    max_secs: f32,
    /// RMS threshold (on [-1, 1] samples) above which audio counts as speech.
    #[arg(long, default_value_t = 0.012)]
    vad_threshold: f32,
}

impl RealtimeArgs {
    /// Defaults used when the binary is launched with no subcommand (e.g. by
    /// `cargo leptos watch`, which runs the server binary bare).
    fn defaults() -> Self {
        Self {
            addr: None,
            language: "zh".to_string(),
            no_itn: false,
            fp32: false,
            cpu: false,
            model_dir: None,
            silence_ms: 600,
            partial_ms: 900,
            max_secs: 15.0,
            vad_threshold: 0.012,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                // Our logs at info; silence onnxruntime's chatty internal info logs.
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort::logging=warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    // ort uses load-dynamic; point it at the system onnxruntime if unset.
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        unsafe { std::env::set_var("ORT_DYLIB_PATH", "/usr/lib/libonnxruntime.so") };
    }

    match Cli::parse().cmd {
        Some(Command::Preprocess(a)) => preprocess(a),
        Some(Command::Transcribe(a)) => transcribe(a),
        Some(Command::Realtime(a)) => realtime(a),
        None => realtime(RealtimeArgs::defaults()),
    }
}

fn preprocess(a: PreprocessArgs) -> Result<()> {
    tracing::info!("preprocessing {}", a.input.display());
    let samples = audio::decode(&a.input, &a.filter)?;
    audio::write_wav(&a.output, &samples)?;
    tracing::info!(
        "wrote {} ({:.1}s) to {}",
        a.output.display(),
        samples.len() as f32 / audio::SAMPLE_RATE as f32,
        a.output.display(),
    );
    Ok(())
}

fn transcribe(a: TranscribeArgs) -> Result<()> {
    let filter = if a.denoise {
        a.filter.as_str()
    } else {
        audio::PASSTHROUGH_FILTER
    };
    tracing::info!("decoding {} (denoise: {})", a.input.display(), a.denoise);
    let samples = audio::decode(&a.input, filter)?;

    if let Some(clean) = &a.keep_clean {
        audio::write_wav(clean, &samples)?;
        tracing::info!("saved cleaned audio to {}", clean.display());
    }

    let source = match a.model_dir {
        Some(dir) => ModelSource::Local(dir),
        None => ModelSource::HuggingFace,
    };
    let mut recognizer = Recognizer::load(source, a.fp32, a.cpu)?;
    let text = recognizer.transcribe(&samples, &a.language, !a.no_itn)?;

    match &a.output {
        Some(path) => {
            std::fs::write(path, format!("{text}\n"))?;
            tracing::info!("transcript written to {}", path.display());
        }
        None => {
            let mut stdout = std::io::stdout().lock();
            writeln!(stdout, "{text}")?;
        }
    }
    Ok(())
}

fn realtime(a: RealtimeArgs) -> Result<()> {
    server::run(ServeOpts {
        addr: a.addr,
        model_dir: a.model_dir,
        fp32: a.fp32,
        cpu: a.cpu,
        language: a.language,
        itn: !a.no_itn,
        vad: VadConfig {
            silence_ms: a.silence_ms,
            partial_ms: a.partial_ms,
            max_secs: a.max_secs,
            threshold: a.vad_threshold,
        },
    })
}

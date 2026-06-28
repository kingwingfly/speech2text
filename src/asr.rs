//! SenseVoice ASR: model/token acquisition, ONNX session (CUDA via `ort`),
//! chunked inference over long audio, and CTC greedy decoding.

use crate::audio;
use crate::features::{self, Fbank};
use anyhow::{Context, Result, anyhow};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use std::path::{Path, PathBuf};

const HF_REPO: &str = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17";

/// Segment length and the search radius used to snap cuts to silence.
const CHUNK_SECS: f32 = 15.0;
const SEARCH_SECS: f32 = 2.0;

/// Where to obtain the model from.
pub enum ModelSource {
    /// Download (and cache) from Hugging Face.
    HuggingFace,
    /// Load `model[.int8].onnx` + `tokens.txt` from a local directory.
    Local(PathBuf),
}

/// Parsed model metadata needed by the front-end and decoder.
struct Meta {
    lfr_m: usize,
    lfr_n: usize,
    /// Sample scaling implied by `normalize_samples` (1.0 or 32768.0).
    sample_scale: f32,
    neg_mean: Vec<f32>,
    inv_stddev: Vec<f32>,
    blank_id: usize,
    with_itn: i32,
    without_itn: i32,
    lang_ids: std::collections::HashMap<String, i32>,
}

pub struct Recognizer {
    session: Session,
    input_names: Vec<String>,
    output_name: String,
    tokens: Vec<String>,
    meta: Meta,
    fbank: Fbank,
}

impl Recognizer {
    pub fn load(source: ModelSource, fp32: bool, use_cpu: bool) -> Result<Self> {
        let (model_path, tokens_path) = resolve_files(source, fp32)?;
        tracing::info!("model: {}", model_path.display());

        // Note: the builder-option methods return `ort::Error<SessionBuilder>`, whose
        // embedded builder is not Send/Sync, so they can't flow through `?` into
        // anyhow directly — flatten them to a message.
        let mut builder = Session::builder()?;
        if !use_cpu {
            // CUDA first, silently falling back to CPU if it can't be registered.
            use ort::ep::CUDAExecutionProvider;
            builder = builder
                .with_execution_providers([CUDAExecutionProvider::default().build()])
                .map_err(|e| anyhow!("registering CUDA execution provider: {e}"))?;
        }
        let session = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("setting optimization level: {e}"))?
            .commit_from_file(&model_path)
            .with_context(|| format!("loading ONNX model {}", model_path.display()))?;

        let input_names = session.inputs().iter().map(|i| i.name().to_string()).collect();
        let output_name = session
            .outputs()
            .first()
            .ok_or_else(|| anyhow!("model has no outputs"))?
            .name()
            .to_string();

        let meta = parse_meta(&session)?;
        let tokens = load_tokens(&tokens_path)?;
        tracing::info!("vocab: {} tokens, lfr {}x{}", tokens.len(), meta.lfr_m, meta.lfr_n);

        Ok(Self { session, input_names, output_name, tokens, meta, fbank: Fbank::new() })
    }

    /// Transcribe a full recording (16 kHz mono f32), chunking long audio.
    pub fn transcribe(&mut self, samples: &[f32], language: &str, itn: bool) -> Result<String> {
        let lang_id = *self
            .meta
            .lang_ids
            .get(language)
            .ok_or_else(|| anyhow!("unsupported language '{language}'; known: {:?}", self.meta.lang_ids.keys()))?;
        let textnorm_id = if itn { self.meta.with_itn } else { self.meta.without_itn };

        let segments = audio::split_segments(samples, CHUNK_SECS, SEARCH_SECS);
        let mut pieces: Vec<String> = Vec::new();
        for (i, &(start, end)) in segments.iter().enumerate() {
            tracing::info!(
                "chunk {}/{} [{:.1}s..{:.1}s]",
                i + 1,
                segments.len(),
                start as f32 / audio::SAMPLE_RATE as f32,
                end as f32 / audio::SAMPLE_RATE as f32,
            );
            let text = self.run_chunk(&samples[start..end], lang_id, textnorm_id)?;
            if !text.is_empty() {
                pieces.push(text);
            }
        }
        Ok(pieces.join(""))
    }

    fn run_chunk(&mut self, samples: &[f32], lang_id: i32, textnorm_id: i32) -> Result<String> {
        let feats = self.fbank.compute(samples, self.meta.sample_scale);
        let mut lfr = features::apply_lfr(&feats, self.meta.lfr_m, self.meta.lfr_n);
        if lfr.is_empty() {
            return Ok(String::new());
        }
        features::apply_cmvn(&mut lfr, &self.meta.neg_mean, &self.meta.inv_stddev);

        let t = lfr.len();
        let dim = lfr[0].len();
        let flat: Vec<f32> = lfr.into_iter().flatten().collect();

        let x = Tensor::from_array((vec![1_i64, t as i64, dim as i64], flat))?;
        let x_len = Tensor::from_array((vec![1_i64], vec![t as i32]))?;
        let lang = Tensor::from_array((vec![1_i64], vec![lang_id]))?;
        let tnorm = Tensor::from_array((vec![1_i64], vec![textnorm_id]))?;

        let blank = self.meta.blank_id;
        // Scope the session borrow so it ends before detokenizing (which borrows &self).
        let ids = {
            let outputs = self.session.run(ort::inputs![
                self.input_names[0].as_str() => x,
                self.input_names[1].as_str() => x_len,
                self.input_names[2].as_str() => lang,
                self.input_names[3].as_str() => tnorm,
            ])?;

            let (shape, logits) = outputs[self.output_name.as_str()].try_extract_tensor::<f32>()?;
            // shape = [1, T', vocab]; greedy argmax per frame, collapse repeats, drop blanks.
            let vocab = shape[2] as usize;
            let frames = shape[1] as usize;
            let mut ids = Vec::with_capacity(frames);
            let mut prev = usize::MAX;
            for t in 0..frames {
                let row = &logits[t * vocab..(t + 1) * vocab];
                let mut best = 0usize;
                let mut best_v = f32::MIN;
                for (i, &v) in row.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        best = i;
                    }
                }
                if best != blank && best != prev {
                    ids.push(best);
                }
                prev = best;
            }
            ids
        };

        Ok(self.detokenize(&ids))
    }

    /// Map token ids to text, stripping SenseVoice special tokens and handling
    /// SentencePiece word boundaries (`\u{2581}`).
    fn detokenize(&self, ids: &[usize]) -> String {
        let mut text = String::new();
        for &id in ids {
            let Some(tok) = self.tokens.get(id) else { continue };
            // Skip SenseVoice special tokens like <|zh|>, <|NEUTRAL|>, <|woitn|>, <s>, </s>.
            if tok.starts_with('<') {
                continue;
            }
            if let Some(rest) = tok.strip_prefix('\u{2581}') {
                // SentencePiece word boundary.
                text.push(' ');
                text.push_str(rest);
            } else {
                text.push_str(tok);
            }
        }
        text.trim().to_string()
    }
}

fn resolve_files(source: ModelSource, fp32: bool) -> Result<(PathBuf, PathBuf)> {
    let model_file = if fp32 { "model.onnx" } else { "model.int8.onnx" };
    match source {
        ModelSource::Local(dir) => {
            let model = dir.join(model_file);
            let tokens = dir.join("tokens.txt");
            if !model.exists() {
                return Err(anyhow!("{} not found", model.display()));
            }
            Ok((model, tokens))
        }
        ModelSource::HuggingFace => {
            use hf_hub::api::sync::Api;
            tracing::info!("fetching {model_file} from {HF_REPO} (first run downloads, then cached)");
            let api = Api::new()?;
            let repo = api.model(HF_REPO.to_string());
            let model = repo.get(model_file).context("downloading model")?;
            let tokens = repo.get("tokens.txt").context("downloading tokens.txt")?;
            Ok((model, tokens))
        }
    }
}

fn load_tokens(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    // Each line: "<token> <id>"; place token at its id index.
    let mut pairs: Vec<(usize, String)> = Vec::new();
    let mut max_id = 0;
    for line in text.lines() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        let (tok, id) = line
            .rsplit_once(' ')
            .ok_or_else(|| anyhow!("malformed tokens.txt line: {line:?}"))?;
        let id: usize = id.trim().parse().with_context(|| format!("bad id in line {line:?}"))?;
        max_id = max_id.max(id);
        pairs.push((id, tok.to_string()));
    }
    let mut tokens = vec![String::new(); max_id + 1];
    for (id, tok) in pairs {
        tokens[id] = tok;
    }
    Ok(tokens)
}

fn parse_meta(session: &Session) -> Result<Meta> {
    let md = session.metadata()?;
    let get = |k: &str| md.custom(k);
    let get_req = |k: &str| md.custom(k).ok_or_else(|| anyhow!("model metadata missing '{k}'"));

    let parse_int = |k: &str| -> Result<i32> {
        get_req(k)?.trim().parse().with_context(|| format!("parsing metadata '{k}'"))
    };
    let parse_floats = |k: &str| -> Result<Vec<f32>> {
        let raw = get_req(k)?;
        tracing::debug!("metadata {k}: len={}, prefix={:?}", raw.len(), &raw[..raw.len().min(80)]);
        raw.split([',', ' ', '\t', '\n', '\r', '[', ']'])
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<f32>().map_err(|e| anyhow!("metadata '{k}' token {s:?}: {e}")))
            .collect()
    };

    let lfr_m = parse_int("lfr_window_size")? as usize;
    let lfr_n = parse_int("lfr_window_shift")? as usize;
    let normalize_samples = parse_int("normalize_samples").unwrap_or(0);
    let sample_scale = if normalize_samples != 0 { 1.0 } else { 32768.0 };
    let blank_id = get("blank_id").and_then(|s| s.trim().parse().ok()).unwrap_or(0);

    let neg_mean = parse_floats("neg_mean")?;
    let inv_stddev = parse_floats("inv_stddev")?;

    let with_itn = parse_int("with_itn")?;
    let without_itn = parse_int("without_itn")?;

    let mut lang_ids = std::collections::HashMap::new();
    for lang in ["auto", "zh", "en", "ja", "ko", "yue"] {
        if let Ok(id) = parse_int(&format!("lang_{lang}")) {
            lang_ids.insert(lang.to_string(), id);
        }
    }
    if lang_ids.is_empty() {
        return Err(anyhow!("model metadata has no language ids (lang_*)"));
    }

    Ok(Meta { lfr_m, lfr_n, sample_scale, neg_mean, inv_stddev, blank_id, with_itn, without_itn, lang_ids })
}

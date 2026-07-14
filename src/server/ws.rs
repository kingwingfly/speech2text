//! WebSocket handler: accepts raw f32 PCM @ 16 kHz, runs energy-based VAD to
//! emit live partials and finalize lines on a pause, streaming JSON text back.

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use std::sync::Arc;

use super::{AppState, Engine, VadConfig};
use crate::app::SttMsg;

const SAMPLE_RATE: f32 = 16_000.0;
/// Fewer samples than one FBank frame produce no features; skip inference.
const MIN_SAMPLES: usize = 400;

pub async fn handler(State(state): State<AppState>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| session(socket, state.engine, state.vad))
}

async fn session(mut socket: WebSocket, engine: Arc<Engine>, vad: VadConfig) {
    let silence_needed = (vad.silence_ms as f32 / 1000.0 * SAMPLE_RATE) as usize;
    let partial_every = (vad.partial_ms as f32 / 1000.0 * SAMPLE_RATE) as usize;
    let max_samples = (vad.max_secs * SAMPLE_RATE) as usize;

    let mut buf: Vec<f32> = Vec::new();
    let mut trailing_silence = 0usize;
    let mut since_partial = 0usize;
    let mut speech = false;

    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Binary(bytes) => {
                let block: Vec<f32> =
                    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                if block.is_empty() {
                    continue;
                }

                if rms(&block) > vad.threshold {
                    speech = true;
                    trailing_silence = 0;
                } else if speech {
                    trailing_silence += block.len();
                }
                buf.extend_from_slice(&block);
                since_partial += block.len();

                let finalize =
                    speech && (trailing_silence >= silence_needed || buf.len() >= max_samples);
                if finalize {
                    emit(&mut socket, &engine, &buf, true).await;
                    buf.clear();
                    trailing_silence = 0;
                    since_partial = 0;
                    speech = false;
                } else if speech && since_partial >= partial_every && buf.len() >= MIN_SAMPLES {
                    since_partial = 0;
                    emit(&mut socket, &engine, &buf, false).await;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    // Flush whatever is left as a final line.
    if speech && buf.len() >= MIN_SAMPLES {
        emit(&mut socket, &engine, &buf, true).await;
    }
}

/// Run inference over `buf` and send the result as a partial or final message.
async fn emit(socket: &mut WebSocket, engine: &Arc<Engine>, buf: &[f32], final_: bool) {
    let text = match engine.infer(buf.to_vec()).await {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("inference failed: {e}");
            return;
        }
    };
    if text.is_empty() {
        return;
    }
    let msg = if final_ { SttMsg::Final { text } } else { SttMsg::Partial { text } };
    if let Ok(json) = serde_json::to_string(&msg) {
        let _ = socket.send(Message::Text(json.into())).await;
    }
}

fn rms(block: &[f32]) -> f32 {
    if block.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = block.iter().map(|x| x * x).sum();
    (sum_sq / block.len() as f32).sqrt()
}

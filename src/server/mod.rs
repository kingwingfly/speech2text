//! Realtime STT web server: loads the recognizer once, serves the Leptos SSR
//! UI, and exposes `/ws` for streaming audio in and text out.

mod ws;

use anyhow::Result;
use axum::Router;
use axum::extract::FromRef;
use axum::http::{HeaderMap, HeaderValue, header};
use axum::response::Response;
use leptos::config::get_configuration;
use leptos::prelude::*;
use leptos_axum::{LeptosRoutes, generate_route_list};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tower_http::validate_request::ValidateRequestHeaderLayer;

use crate::app::{App, shell};
use crate::asr::{ModelSource, Recognizer};

/// Options for the realtime server (populated from CLI flags).
pub struct ServeOpts {
    pub addr: Option<String>,
    pub model_dir: Option<PathBuf>,
    pub fp32: bool,
    pub cpu: bool,
    pub language: String,
    pub itn: bool,
    pub vad: VadConfig,
    /// Optional HTTP Basic-auth password protecting the whole app. `None`
    /// leaves the server open.
    pub password: Option<String>,
}

/// Voice-activity / chunking parameters for the WebSocket handler.
#[derive(Clone, Copy)]
pub struct VadConfig {
    /// Trailing silence that finalizes an utterance.
    pub silence_ms: u32,
    /// How often to re-transcribe the in-progress utterance for a partial.
    pub partial_ms: u32,
    /// Hard cap on utterance length before a forced finalize.
    pub max_secs: f32,
    /// RMS threshold (on [-1, 1] samples) above which a block counts as speech.
    pub threshold: f32,
}

/// Owns the recognizer and runs inference off the async runtime.
pub struct Engine {
    rec: Mutex<Recognizer>,
    language: String,
    itn: bool,
}

impl Engine {
    fn new(rec: Recognizer, language: String, itn: bool) -> Self {
        Self { rec: Mutex::new(rec), language, itn }
    }

    /// Transcribe one buffered segment, blocking on a worker thread.
    pub async fn infer(self: &Arc<Self>, samples: Vec<f32>) -> Result<String> {
        let me = self.clone();
        tokio::task::spawn_blocking(move || {
            let mut rec = me.rec.lock().unwrap();
            rec.transcribe_once(&samples, &me.language, me.itn)
        })
        .await?
    }
}

/// Name of the session cookie minted for clients that pass the Basic-auth gate.
const AUTH_COOKIE: &str = "stt_auth";

/// Bridges the Basic-auth gate to `/ws`.
///
/// Browsers never attach cached HTTP Basic credentials to a WebSocket
/// handshake — the WebSocket API can't set headers, and the HTTP auth cache
/// isn't consulted for `ws://`/`wss://`. So `/ws` can't sit behind the Basic
/// layer: it would 401 and pop a second credential prompt that can't ever
/// succeed. Instead every page response that clears the gate carries this
/// token as a cookie, which the browser *does* send on the same-origin
/// handshake, and `/ws` checks that.
///
/// The token is random per process, so a restart invalidates outstanding
/// sessions.
pub(crate) struct Auth {
    token: String,
}

impl Auth {
    fn new() -> Result<Self> {
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes).map_err(|e| anyhow::anyhow!("no OS randomness: {e}"))?;
        Ok(Self { token: bytes.iter().map(|b| format!("{b:02x}")).collect() })
    }

    /// A session cookie (no `Max-Age`): it dies with the browser session, and
    /// the token dies with the process.
    fn cookie(&self) -> HeaderValue {
        let v = format!("{AUTH_COOKIE}={}; Path=/; HttpOnly; SameSite=Strict", self.token);
        HeaderValue::from_str(&v).expect("hex token is a valid header value")
    }

    /// Does this request carry the token a page load handed out? Compared
    /// naively rather than in constant time: guessing 256 random bits over a
    /// LAN is not a threat this gate needs to model.
    pub(crate) fn cookie_ok(&self, headers: &HeaderMap) -> bool {
        headers
            .get_all(header::COOKIE)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .flat_map(|v| v.split(';'))
            .filter_map(|kv| kv.split_once('='))
            .any(|(k, v)| k.trim() == AUTH_COOKIE && v.trim() == self.token)
    }
}

#[derive(Clone)]
struct AppState {
    leptos_options: LeptosOptions,
    engine: Arc<Engine>,
    vad: VadConfig,
    /// `None` when the server is running without a password.
    auth: Option<Arc<Auth>>,
}

impl FromRef<AppState> for LeptosOptions {
    fn from_ref(s: &AppState) -> Self {
        s.leptos_options.clone()
    }
}

/// Build a Tokio runtime and serve until the process is killed.
pub fn run(opts: ServeOpts) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(serve(opts))
}

async fn serve(opts: ServeOpts) -> Result<()> {
    let conf = get_configuration(None)?;
    let mut leptos_options = conf.leptos_options;
    if let Some(addr) = &opts.addr {
        leptos_options.site_addr = addr.parse()?;
    }

    let source = match opts.model_dir {
        Some(dir) => ModelSource::Local(dir),
        None => ModelSource::HuggingFace,
    };
    let rec = Recognizer::load(source, opts.fp32, opts.cpu)?;
    let engine = Arc::new(Engine::new(rec, opts.language, opts.itn));

    let auth = match opts.password {
        Some(_) => Some(Arc::new(Auth::new()?)),
        None => None,
    };
    let state =
        AppState { leptos_options: leptos_options.clone(), engine, vad: opts.vad, auth: auth.clone() };
    let routes = generate_route_list(App);

    let mut pages = Router::new()
        .leptos_routes(&state, routes, {
            let opts = leptos_options.clone();
            move || shell(opts.clone())
        })
        .fallback(leptos_axum::file_and_error_handler::<AppState, _>(shell));

    // Optional password gate, on the page routes only. `/ws` authenticates off
    // the cookie minted here instead — see `Auth`. Username is fixed to "stt";
    // only the password is checked.
    if let (Some(pw), Some(auth)) = (&opts.password, &auth) {
        let cookie = auth.cookie();
        // The last `.layer` is the outermost, so Basic auth runs first and
        // short-circuits with a 401; the cookie is only minted onto responses
        // it let through.
        pages = pages
            .layer(axum::middleware::map_response(move |mut res: Response| {
                let cookie = cookie.clone();
                async move {
                    res.headers_mut().insert(header::SET_COOKIE, cookie);
                    res
                }
            }))
            // `basic` is deprecated as "too basic for real applications", but a
            // fixed-credential gate is precisely the intent here (single user,
            // LAN, TLS at the reverse proxy). Avoids hand-rolling base64 for a
            // custom validator.
            .layer({
                #[allow(deprecated)]
                ValidateRequestHeaderLayer::basic("stt", pw)
            });
        tracing::info!("password protection enabled (HTTP Basic auth, user \"stt\")");
    }

    let app = Router::new()
        .route("/ws", axum::routing::get(ws::handler))
        .merge(pages)
        .with_state(state);

    let addr = leptos_options.site_addr;
    tracing::info!("realtime STT server listening on http://{addr}");
    // Mic capture needs a secure context: reach this from another device through
    // an HTTPS-terminating reverse proxy (Pingora, Cloudflare, nginx, …), or use
    // http://localhost on the server machine itself.
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

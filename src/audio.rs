//! Audio I/O built on ffmpeg-next: decode any input (AAC, WAV, ...) to 16 kHz
//! mono f32 PCM, optionally through a denoise/voice-enhancement filter graph,
//! plus WAV writing and silence-aware segmentation for long files.

use anyhow::{Context, Result, anyhow};
use ffmpeg_next as ffmpeg;
use ffmpeg::{codec, filter, format, frame, media};
use std::path::Path;

pub const SAMPLE_RATE: u32 = 16_000;

/// Default speech-focused denoise + enhancement chain (FFmpeg `af` syntax).
/// The output sample rate / layout / format are enforced by the sink, so this
/// chain only needs to do the cleanup, not the final resample.
pub const DENOISE_FILTER: &str = "highpass=f=80,afftdn=nf=-25,speechnorm=e=12.5:r=0.0001,lowpass=f=7800";

/// Pass-through filter (just lets the sink resample/downmix to 16 kHz mono).
pub const PASSTHROUGH_FILTER: &str = "anull";

fn ensure_init() -> Result<()> {
    use std::sync::Once;
    static INIT: Once = Once::new();
    static mut OK: bool = true;
    INIT.call_once(|| {
        if ffmpeg::init().is_err() {
            unsafe { OK = false };
        }
    });
    if unsafe { OK } {
        Ok(())
    } else {
        Err(anyhow!("failed to initialize ffmpeg"))
    }
}

/// Decode `path`, run it through `filter_spec`, and return 16 kHz mono f32 samples.
pub fn decode(path: &Path, filter_spec: &str) -> Result<Vec<f32>> {
    ensure_init()?;

    let mut ictx = format::input(&path).with_context(|| format!("opening {}", path.display()))?;
    let stream = ictx
        .streams()
        .best(media::Type::Audio)
        .ok_or_else(|| anyhow!("no audio stream in {}", path.display()))?;
    let stream_index = stream.index();

    let ctx = codec::context::Context::from_parameters(stream.parameters())?;
    let mut decoder = ctx.decoder().audio()?;
    decoder.set_parameters(ictx.stream(stream_index).unwrap().parameters())?;

    let mut graph = build_graph(&decoder, filter_spec)?;

    let mut samples = Vec::new();
    let mut decoded = frame::Audio::empty();

    for (s, packet) in ictx.packets() {
        if s.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet)?;
        while decoder.receive_frame(&mut decoded).is_ok() {
            graph.get("in").unwrap().source().add(&decoded)?;
            drain_sink(&mut graph, &mut samples)?;
        }
    }
    decoder.send_eof()?;
    while decoder.receive_frame(&mut decoded).is_ok() {
        graph.get("in").unwrap().source().add(&decoded)?;
        drain_sink(&mut graph, &mut samples)?;
    }

    // Flush the filter graph.
    graph.get("in").unwrap().source().flush()?;
    drain_sink(&mut graph, &mut samples)?;

    Ok(samples)
}

/// Pull all currently-available filtered frames out of the sink into `out`.
fn drain_sink(graph: &mut filter::Graph, out: &mut Vec<f32>) -> Result<()> {
    let mut filtered = frame::Audio::empty();
    let mut sink = graph.get("out").unwrap();
    while sink.sink().frame(&mut filtered).is_ok() {
        out.extend_from_slice(filtered.plane::<f32>(0));
    }
    Ok(())
}

fn build_graph(decoder: &codec::decoder::Audio, filter_spec: &str) -> Result<filter::Graph> {
    let mut graph = filter::Graph::new();

    // Describe the source layout. When the decoder reports an unspecified layout
    // (common for mono WAV), declare `channels` instead of a layout mask so the
    // abuffer matches the incoming frames exactly.
    let bits = decoder.channel_layout().bits();
    let layout_arg = if bits != 0 {
        format!("channel_layout=0x{bits:x}")
    } else {
        format!("channels={}", decoder.channels().max(1))
    };
    let args = format!(
        "time_base={}:sample_rate={}:sample_fmt={}:{}",
        decoder.time_base(),
        decoder.rate(),
        decoder.format().name(),
        layout_arg,
    );

    graph.add(&filter::find("abuffer").unwrap(), "in", &args)?;
    graph.add(&filter::find("abuffersink").unwrap(), "out", "")?;

    // ffmpeg 8's abuffersink ignores post-init format setters, so we pin the
    // output format/rate/layout inside the graph itself: the user's chain,
    // then resample to 16 kHz and force packed-float mono. This guarantees the
    // frames we read are f32 mono @ 16 kHz regardless of the source.
    let full_spec = format!("{filter_spec},aresample={SAMPLE_RATE},aformat=sample_fmts=flt:channel_layouts=mono");

    graph.output("in", 0)?.input("out", 0)?.parse(&full_spec)?;
    graph.validate()?;
    Ok(graph)
}

/// Write 16 kHz mono f32 samples to a 16-bit PCM WAV file.
pub fn write_wav(path: &Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("creating {}", path.display()))?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Split a long recording into segments near `target_secs`, snapping each cut to
/// the quietest spot within `search_secs` so words are less likely to be severed.
pub fn split_segments(samples: &[f32], target_secs: f32, search_secs: f32) -> Vec<(usize, usize)> {
    let n = samples.len();
    let target = (target_secs * SAMPLE_RATE as f32) as usize;
    let search = (search_secs * SAMPLE_RATE as f32) as usize;
    let win = SAMPLE_RATE as usize / 50; // 20 ms RMS window

    let mut segments = Vec::new();
    let mut pos = 0;
    while pos < n {
        let ideal = pos + target;
        if ideal >= n {
            segments.push((pos, n));
            break;
        }
        // Search [ideal - search, ideal + search] for the lowest-energy window.
        let lo = ideal.saturating_sub(search).max(pos + 1);
        let hi = (ideal + search).min(n - win);
        let mut best = ideal;
        let mut best_energy = f32::MAX;
        let mut c = lo;
        while c <= hi {
            let e: f32 = samples[c..c + win].iter().map(|x| x * x).sum();
            if e < best_energy {
                best_energy = e;
                best = c + win / 2;
            }
            c += win;
        }
        segments.push((pos, best));
        pos = best;
    }
    segments
}

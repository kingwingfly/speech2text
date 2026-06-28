//! Acoustic front-end for SenseVoice: an 80-dim log-mel FBank from `knf-rs`
//! (kaldi-native-fbank), followed by low-frame-rate (LFR) stacking and CMVN.
//!
//! `knf-rs`'s C wrapper computes FBank with 80 mel bins, 16 kHz, 25 ms / 10 ms
//! frames, dither 0 and `snip_edges = true`. The CMVN statistics (`neg_mean`,
//! `inv_stddev`) and LFR window/shift come from the model metadata at runtime.

pub const NUM_MEL: usize = 80;

const FRAME_LENGTH: usize = 400; // 25 ms @ 16 kHz; knf needs at least one frame

/// Thin wrapper around `knf_rs::compute_fbank` exposing per-frame 80-dim vectors.
pub struct Fbank;

impl Fbank {
    pub fn new() -> Self {
        Fbank
    }

    /// Compute log-mel FBank features for `samples`, one 80-dim vector per frame.
    ///
    /// `scale` matches the model's `normalize_samples` flag: pass `1.0` when the
    /// samples are already in [-1, 1] and the model expects normalized input, or
    /// `32768.0` to bring [-1, 1] samples up to the int16 range kaldi/FunASR used
    /// when the CMVN statistics were computed.
    pub fn compute(&self, samples: &[f32], scale: f32) -> Vec<[f32; NUM_MEL]> {
        if samples.len() < FRAME_LENGTH {
            return Vec::new();
        }
        let scaled: Vec<f32> = if scale == 1.0 {
            samples.to_vec()
        } else {
            samples.iter().map(|s| s * scale).collect()
        };

        let feats = match knf_rs::compute_fbank(&scaled) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("knf-rs fbank failed: {e}");
                return Vec::new();
            }
        };

        // feats: Array2<f32> shaped [num_frames, NUM_MEL].
        feats
            .outer_iter()
            .map(|row| {
                let mut a = [0.0f32; NUM_MEL];
                for (i, v) in row.iter().enumerate().take(NUM_MEL) {
                    a[i] = *v;
                }
                a
            })
            .collect()
    }
}

/// FunASR low-frame-rate stacking: pad `(m-1)/2` copies of the first frame at the
/// front, then concatenate `m` consecutive frames with stride `n`, repeating the
/// last frame to fill the final window. Output dim is `m * NUM_MEL`.
pub fn apply_lfr(feats: &[[f32; NUM_MEL]], m: usize, n: usize) -> Vec<Vec<f32>> {
    if feats.is_empty() {
        return Vec::new();
    }
    let pad = (m - 1) / 2;
    let t_lfr = feats.len().div_ceil(n);

    // Logical padded sequence: `pad` copies of feats[0], then feats.
    let get = |idx: usize| -> &[f32; NUM_MEL] {
        if idx < pad {
            &feats[0]
        } else {
            let j = idx - pad;
            &feats[j.min(feats.len() - 1)]
        }
    };

    let mut out = Vec::with_capacity(t_lfr);
    for i in 0..t_lfr {
        let mut v = Vec::with_capacity(m * NUM_MEL);
        for j in 0..m {
            v.extend_from_slice(get(i * n + j));
        }
        out.push(v);
    }
    out
}

/// In-place CMVN: `x = (x + neg_mean) * inv_stddev`, applied per LFR feature dim.
pub fn apply_cmvn(feats: &mut [Vec<f32>], neg_mean: &[f32], inv_stddev: &[f32]) {
    for frame in feats.iter_mut() {
        for (k, x) in frame.iter_mut().enumerate() {
            *x = (*x + neg_mean[k]) * inv_stddev[k];
        }
    }
}

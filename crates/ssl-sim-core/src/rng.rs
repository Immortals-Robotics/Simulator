//! Seeded random streams, one per concern, so changing one feature's use of
//! randomness never perturbs another's.
//!
//! OWNER: general agent A. Replace the `todo!()` bodies; keep the signatures.

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// All random streams used by the core.
#[derive(Debug, Clone)]
pub struct Rngs {
    /// Gaussian measurement noise (positions, angles, area).
    pub vision_noise: Xoshiro256PlusPlus,
    /// Detection dropouts, spurious detections, occlusion sampling jitter.
    pub vision_dropout: Xoshiro256PlusPlus,
    /// Robot command / response loss (used by the net layer through the world).
    pub packet_loss: Xoshiro256PlusPlus,
    /// Ordering shuffles (multiple balls in one frame).
    pub shuffle: Xoshiro256PlusPlus,
}

impl Rngs {
    /// Create every stream from one seed. Streams must be independent:
    /// derive each from `seed` mixed with a distinct stream id.
    pub fn from_seed(seed: u64) -> Self {
        let mk = |stream: u64| {
            let mut s = [0u8; 32];
            s[..8].copy_from_slice(&seed.to_le_bytes());
            s[8..16].copy_from_slice(&stream.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_le_bytes());
            s[16..24].copy_from_slice(&(!seed).to_le_bytes());
            s[24..32].copy_from_slice(&(stream ^ 0xD1B5_4A32_D192_ED03).to_le_bytes());
            Xoshiro256PlusPlus::from_seed(s)
        };
        Self {
            vision_noise: mk(1),
            vision_dropout: mk(2),
            packet_loss: mk(3),
            shuffle: mk(4),
        }
    }
}

/// Draw a standard normal sample (Box-Muller, deterministic given the stream).
///
/// Consumes exactly two uniforms per call (plus retries for the degenerate
/// `u1 == 0` draw, which cannot happen in practice), so a given stream always
/// produces the same sequence for the same sequence of calls.
pub fn normal<R: Rng>(rng: &mut R, stddev: f64) -> f64 {
    if stddev <= 0.0 {
        return 0.0;
    }
    // `random::<f64>()` yields [0, 1); reject 0 so `ln` stays finite.
    let mut u1 = rng.random::<f64>();
    while u1 <= f64::MIN_POSITIVE {
        u1 = rng.random::<f64>();
    }
    let u2 = rng.random::<f64>();
    let r = (-2.0 * u1.ln()).sqrt();
    r * (std::f64::consts::TAU * u2).cos() * stddev
}

/// Draw a 2D vector of independent normal samples (x first, then y).
pub fn normal_vec2<R: Rng>(rng: &mut R, stddev: f64) -> crate::types::Vec2 {
    let x = normal(rng, stddev);
    let y = normal(rng, stddev);
    crate::types::Vec2::new(x, y)
}

/// Bernoulli trial with probability `p` (clamped to 0..=1).
pub fn chance<R: Rng>(rng: &mut R, p: f64) -> bool {
    if p <= 0.0 {
        return false;
    }
    if p >= 1.0 {
        return true;
    }
    rng.random::<f64>() < p
}

/// In-place Fisher-Yates shuffle driven by a seeded stream.
pub fn shuffle<R: Rng, T>(rng: &mut R, items: &mut [T]) {
    if items.len() < 2 {
        return;
    }
    for i in (1..items.len()).rev() {
        let j = rng.random_range(0..=i);
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_normal_matches_requested_moments() {
        let mut rngs = Rngs::from_seed(7);
        let stddev = 0.0137;
        const N: usize = 100_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..N {
            let x = normal(&mut rngs.vision_noise, stddev);
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / N as f64;
        let var = sum_sq / N as f64 - mean * mean;
        let sd = var.sqrt();
        // Mean is 0; compare against the scale of the distribution.
        assert!(
            mean.abs() < 0.02 * stddev,
            "mean {mean} too large for sd {stddev}"
        );
        assert!(
            (sd / stddev - 1.0).abs() < 0.02,
            "sample stddev {sd} not within 2% of {stddev}"
        );
    }

    #[test]
    fn rng_normal_is_deterministic_and_zero_for_zero_stddev() {
        let a: Vec<f64> = {
            let mut r = Rngs::from_seed(3);
            (0..8).map(|_| normal(&mut r.vision_noise, 1.0)).collect()
        };
        let b: Vec<f64> = {
            let mut r = Rngs::from_seed(3);
            (0..8).map(|_| normal(&mut r.vision_noise, 1.0)).collect()
        };
        assert_eq!(a, b);
        let mut r = Rngs::from_seed(3);
        assert_eq!(normal(&mut r.vision_noise, 0.0), 0.0);
    }

    #[test]
    fn rng_streams_are_independent() {
        let mut r = Rngs::from_seed(11);
        let a = normal(&mut r.vision_noise, 1.0);
        let mut r2 = Rngs::from_seed(11);
        // Consuming the dropout stream must not perturb the noise stream.
        let _ = chance(&mut r2.vision_dropout, 0.5);
        let b = normal(&mut r2.vision_noise, 1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn rng_chance_edges() {
        let mut r = Rngs::from_seed(1);
        assert!(!chance(&mut r.vision_dropout, 0.0));
        assert!(chance(&mut r.vision_dropout, 1.0));
    }

    #[test]
    fn rng_shuffle_is_a_permutation() {
        let mut r = Rngs::from_seed(5);
        let mut v: Vec<u32> = (0..16).collect();
        shuffle(&mut r.shuffle, &mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..16).collect::<Vec<_>>());
    }
}

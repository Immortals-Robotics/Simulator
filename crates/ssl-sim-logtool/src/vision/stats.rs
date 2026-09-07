//! Small mergeable accumulators used by the vision analysis.
//!
//! Everything here is streaming and allocation-free in the hot path except the
//! reservoir sampler, which owns one fixed-capacity vector.

use serde::Serialize;

/// Running mean/variance (Welford), mergeable.
#[derive(Debug, Default, Clone)]
pub struct Welford {
    /// Sample count.
    pub n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl Welford {
    /// Add one sample.
    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        if self.n == 0 {
            self.min = x;
            self.max = x;
        } else {
            if x < self.min {
                self.min = x;
            }
            if x > self.max {
                self.max = x;
            }
        }
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }

    /// Fold another accumulator in.
    pub fn merge(&mut self, o: &Self) {
        if o.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = o.clone();
            return;
        }
        let n = (self.n + o.n) as f64;
        let d = o.mean - self.mean;
        self.m2 += o.m2 + d * d * (self.n as f64) * (o.n as f64) / n;
        self.mean += d * (o.n as f64) / n;
        self.n += o.n;
        self.min = self.min.min(o.min);
        self.max = self.max.max(o.max);
    }

    /// Mean, or 0 when empty.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Sample standard deviation (n-1), or 0 when n < 2.
    pub fn std(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            (self.m2 / (self.n as f64 - 1.0)).max(0.0).sqrt()
        }
    }

    /// Serialisable summary, `None` when empty.
    pub fn summary(&self) -> Option<StatSummary> {
        if self.n == 0 {
            return None;
        }
        Some(StatSummary {
            n: self.n,
            mean: r(self.mean),
            std: r(self.std()),
            min: r(self.min),
            max: r(self.max),
        })
    }

    /// Root mean square about zero (the right "noise sigma" when the mean is a
    /// systematic offset that should be counted as error).
    pub fn rms(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        (self.mean * self.mean + self.m2 / self.n as f64).sqrt()
    }
}

/// Serialisable mean/std/min/max.
#[derive(Debug, Clone, Serialize)]
pub struct StatSummary {
    /// Sample count.
    pub n: u64,
    /// Mean.
    pub mean: f64,
    /// Sample standard deviation.
    pub std: f64,
    /// Minimum.
    pub min: f64,
    /// Maximum.
    pub max: f64,
}

/// Reservoir sampler for quantiles. Deterministic (fixed-seed xorshift).
#[derive(Debug, Clone)]
pub struct Quantiles {
    samples: Vec<f32>,
    cap: usize,
    seen: u64,
    rng: u64,
}

impl Quantiles {
    /// New sampler holding at most `cap` samples.
    pub fn new(cap: usize) -> Self {
        Self {
            samples: Vec::new(),
            cap,
            seen: 0,
            rng: 0x2545_F491_4F6C_DD1D,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    /// Add one sample.
    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.seen += 1;
        if self.samples.len() < self.cap {
            self.samples.push(x as f32);
            return;
        }
        let j = (self.next_u64() % self.seen) as usize;
        if j < self.cap {
            self.samples[j] = x as f32;
        }
    }

    /// Fold another sampler in (approximate: both reservoirs are concatenated
    /// and thinned, so the merged set is only weighted by the reservoir sizes).
    pub fn merge(&mut self, o: &Self) {
        for &s in &o.samples {
            self.push(s as f64);
        }
        // `push` counted `o.samples.len()`; correct the total so later pushes
        // keep the right acceptance probability.
        self.seen = self.seen + o.seen - o.samples.len() as u64;
    }

    /// Quantile summary, `None` when empty.
    pub fn summary(&mut self) -> Option<QuantileSummary> {
        if self.samples.is_empty() {
            return None;
        }
        self.samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| -> f64 {
            let i = ((self.samples.len() - 1) as f64 * f).round() as usize;
            self.samples[i] as f64
        };
        Some(QuantileSummary {
            n: self.seen,
            min: r(q(0.0)),
            p1: r(q(0.01)),
            p50: r(q(0.5)),
            p90: r(q(0.90)),
            p99: r(q(0.99)),
            p999: r(q(0.999)),
            max: r(q(1.0)),
        })
    }
}

/// Serialisable quantiles.
#[derive(Debug, Clone, Serialize)]
#[allow(missing_docs)]
pub struct QuantileSummary {
    pub n: u64,
    pub min: f64,
    pub p1: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub p999: f64,
    pub max: f64,
}

/// Fixed-width histogram over `[lo, lo + step * bins)`, with under/overflow.
#[derive(Debug, Clone)]
pub struct Hist {
    lo: f64,
    step: f64,
    bins: Vec<u64>,
    under: u64,
    over: u64,
}

impl Hist {
    /// New histogram.
    pub fn new(lo: f64, step: f64, n: usize) -> Self {
        Self {
            lo,
            step,
            bins: vec![0; n],
            under: 0,
            over: 0,
        }
    }

    /// Add one sample.
    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        let i = ((x - self.lo) / self.step).floor();
        if i < 0.0 {
            self.under += 1;
        } else if i >= self.bins.len() as f64 {
            self.over += 1;
        } else {
            self.bins[i as usize] += 1;
        }
    }

    /// Fold another histogram in (must have the same layout).
    pub fn merge(&mut self, o: &Self) {
        for (a, b) in self.bins.iter_mut().zip(&o.bins) {
            *a += *b;
        }
        self.under += o.under;
        self.over += o.over;
    }

    /// Sum of `bin_centre * count` over the in-range bins.
    pub fn weighted_sum(&self) -> f64 {
        self.bins
            .iter()
            .enumerate()
            .map(|(i, c)| (self.lo + (i as f64 + 0.5) * self.step) * *c as f64)
            .sum()
    }

    /// `(bin_centre, count)` pairs for non-empty bins, plus under/overflow.
    pub fn summary(&self) -> serde_json::Value {
        let pairs: Vec<serde_json::Value> = self
            .bins
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(i, c)| serde_json::json!([r(self.lo + (i as f64 + 0.5) * self.step), c]))
            .collect();
        serde_json::json!({ "bins": pairs, "under": self.under, "over": self.over })
    }
}

/// Per-bin [`Welford`] over an independent variable (e.g. distance).
#[derive(Debug, Clone)]
pub struct Binned {
    lo: f64,
    step: f64,
    bins: Vec<Welford>,
}

impl Binned {
    /// New binned accumulator over `[lo, lo + step * n)`.
    pub fn new(lo: f64, step: f64, n: usize) -> Self {
        Self {
            lo,
            step,
            bins: vec![Welford::default(); n],
        }
    }

    /// Add `value` at coordinate `x` (clamped into the range).
    pub fn push(&mut self, x: f64, value: f64) {
        if !x.is_finite() {
            return;
        }
        let i = (((x - self.lo) / self.step).floor()).clamp(0.0, self.bins.len() as f64 - 1.0);
        self.bins[i as usize].push(value);
    }

    /// Fold another accumulator in (must have the same layout).
    pub fn merge(&mut self, o: &Self) {
        for (a, b) in self.bins.iter_mut().zip(&o.bins) {
            a.merge(b);
        }
    }

    /// `{x, n, mean, std}` rows for non-empty bins.
    pub fn summary(&self) -> Vec<serde_json::Value> {
        self.bins
            .iter()
            .enumerate()
            .filter(|(_, w)| w.n > 0)
            .map(|(i, w)| {
                serde_json::json!({
                    "x": r(self.lo + (i as f64 + 0.5) * self.step),
                    "n": w.n,
                    "mean": r(w.mean()),
                    "std": r(w.std()),
                    "rms": r(w.rms()),
                })
            })
            .collect()
    }
}

/// Success/trial counter.
#[derive(Debug, Default, Clone, Copy)]
pub struct Rate {
    /// Numerator.
    pub hits: u64,
    /// Denominator.
    pub trials: u64,
}

impl Rate {
    /// Record one trial.
    pub fn push(&mut self, hit: bool) {
        self.trials += 1;
        self.hits += u64::from(hit);
    }

    /// Fold another counter in.
    pub fn merge(&mut self, o: &Self) {
        self.hits += o.hits;
        self.trials += o.trials;
    }

    /// Ratio, or 0 with no trials.
    pub fn ratio(&self) -> f64 {
        if self.trials == 0 {
            0.0
        } else {
            self.hits as f64 / self.trials as f64
        }
    }

    /// `{hits, trials, ratio}`.
    pub fn summary(&self) -> serde_json::Value {
        serde_json::json!({ "hits": self.hits, "trials": self.trials, "ratio": r(self.ratio()) })
    }
}

/// Per-bin [`Rate`] over an independent variable.
#[derive(Debug, Clone)]
pub struct BinnedRate {
    lo: f64,
    step: f64,
    bins: Vec<Rate>,
}

impl BinnedRate {
    /// New binned rate over `[lo, lo + step * n)`.
    pub fn new(lo: f64, step: f64, n: usize) -> Self {
        Self {
            lo,
            step,
            bins: vec![Rate::default(); n],
        }
    }

    /// Record one trial at coordinate `x` (clamped into the range).
    pub fn push(&mut self, x: f64, hit: bool) {
        if !x.is_finite() {
            return;
        }
        let i = (((x - self.lo) / self.step).floor()).clamp(0.0, self.bins.len() as f64 - 1.0);
        self.bins[i as usize].push(hit);
    }

    /// Fold another accumulator in (must have the same layout).
    pub fn merge(&mut self, o: &Self) {
        for (a, b) in self.bins.iter_mut().zip(&o.bins) {
            a.merge(b);
        }
    }

    /// `{x, hits, trials, ratio}` rows for non-empty bins.
    pub fn summary(&self) -> Vec<serde_json::Value> {
        self.bins
            .iter()
            .enumerate()
            .filter(|(_, w)| w.trials > 0)
            .map(|(i, w)| {
                serde_json::json!({
                    "x": r(self.lo + (i as f64 + 0.5) * self.step),
                    "trials": w.trials,
                    "hits": w.hits,
                    "ratio": r(w.ratio()),
                })
            })
            .collect()
    }
}

/// Coverage grid over the field: per-cell detection counts for one camera.
#[derive(Debug, Clone)]
pub struct Grid {
    /// Cell size [m].
    pub step: f64,
    /// Cells along x.
    pub nx: usize,
    /// Cells along y.
    pub ny: usize,
    /// Lower-left corner [m].
    pub origin: (f64, f64),
    /// Counts, row-major in y.
    pub cells: Vec<u32>,
}

impl Grid {
    /// New grid covering `[-hx, hx] x [-hy, hy]` with `step` cells.
    pub fn new(hx: f64, hy: f64, step: f64) -> Self {
        let nx = (2.0 * hx / step).ceil() as usize;
        let ny = (2.0 * hy / step).ceil() as usize;
        Self {
            step,
            nx,
            ny,
            origin: (-hx, -hy),
            cells: vec![0; nx * ny],
        }
    }

    /// Increment the cell containing `(x, y)`; out-of-range points are ignored.
    pub fn push(&mut self, x: f64, y: f64) {
        let ix = ((x - self.origin.0) / self.step).floor();
        let iy = ((y - self.origin.1) / self.step).floor();
        if ix < 0.0 || iy < 0.0 || ix >= self.nx as f64 || iy >= self.ny as f64 {
            return;
        }
        self.cells[iy as usize * self.nx + ix as usize] += 1;
    }

    /// Centre of cell `(ix, iy)` [m].
    pub fn centre(&self, ix: usize, iy: usize) -> (f64, f64) {
        (
            self.origin.0 + (ix as f64 + 0.5) * self.step,
            self.origin.1 + (iy as f64 + 0.5) * self.step,
        )
    }

    /// Boolean coverage mask: cells whose count is at least `frac` of the 99th
    /// percentile cell count (robust against a few stray detections).
    pub fn mask(&self, frac: f64) -> Vec<bool> {
        let mut nz: Vec<u32> = self.cells.iter().copied().filter(|c| *c > 0).collect();
        if nz.is_empty() {
            return vec![false; self.cells.len()];
        }
        nz.sort_unstable();
        let p99 = nz[((nz.len() - 1) as f64 * 0.99) as usize] as f64;
        let thr = (p99 * frac).max(1.0);
        self.cells.iter().map(|c| *c as f64 >= thr).collect()
    }
}

/// Round to 6 significant-ish decimals so the JSON stays readable.
pub fn r(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    let m = 1e6;
    (x * m).round() / m
}

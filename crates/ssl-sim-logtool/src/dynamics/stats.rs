//! Small robust-statistics and least-squares helpers (no external crates).

use serde::Serialize;

/// Sorted copy without NaNs.
pub fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s
}

/// Quantile of a sorted slice (linear interpolation).
pub fn quantile_sorted(s: &[f64], q: f64) -> f64 {
    if s.is_empty() {
        return f64::NAN;
    }
    let pos = q.clamp(0.0, 1.0) * (s.len() - 1) as f64;
    let i = pos.floor() as usize;
    let f = pos - i as f64;
    if i + 1 < s.len() {
        s[i] * (1.0 - f) + s[i + 1] * f
    } else {
        s[i]
    }
}

pub fn quantile(v: &[f64], q: f64) -> f64 {
    quantile_sorted(&sorted(v), q)
}

pub fn median(v: &[f64]) -> f64 {
    quantile(v, 0.5)
}

/// Median absolute deviation scaled to a Gaussian sigma (x1.4826).
pub fn mad_sigma(v: &[f64]) -> f64 {
    let s = sorted(v);
    if s.is_empty() {
        return f64::NAN;
    }
    let m = quantile_sorted(&s, 0.5);
    let dev: Vec<f64> = s.iter().map(|x| (x - m).abs()).collect();
    1.4826 * quantile(&dev, 0.5)
}

pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Robust summary of a sample: median, MAD-sigma, standard error of the
/// median (1.2533 * sigma / sqrt(n)), quantiles and n.
#[derive(Debug, Clone, Serialize, Default)]
pub struct Summary {
    pub n: usize,
    pub median: f64,
    /// Robust spread (MAD * 1.4826).
    pub sigma: f64,
    /// Standard error of the median.
    pub se: f64,
    pub mean: f64,
    pub p05: f64,
    pub p25: f64,
    pub p75: f64,
    pub p95: f64,
    pub min: f64,
    pub max: f64,
}

impl Summary {
    pub fn of(v: &[f64]) -> Summary {
        let s = sorted(v);
        if s.is_empty() {
            return Summary::default();
        }
        let sigma = mad_sigma(&s);
        Summary {
            n: s.len(),
            median: quantile_sorted(&s, 0.5),
            sigma,
            se: 1.2533 * sigma / (s.len() as f64).sqrt(),
            mean: mean(&s),
            p05: quantile_sorted(&s, 0.05),
            p25: quantile_sorted(&s, 0.25),
            p75: quantile_sorted(&s, 0.75),
            p95: quantile_sorted(&s, 0.95),
            min: s[0],
            max: s[s.len() - 1],
        }
    }

    /// `median ± se (n)` one-liner.
    pub fn line(&self, unit: &str) -> String {
        format!(
            "{:.4} ± {:.4} {} (σ {:.4}, p05 {:.3}, p95 {:.3}, n {})",
            self.median, self.se, unit, self.sigma, self.p05, self.p95, self.n
        )
    }
}

/// Solve the normal equations of a dense least-squares problem
/// `min |A x - b|` given rows `(a_row, b)`; returns `x` and the residual RMS.
/// `n` = number of unknowns. Uses Gaussian elimination with partial pivoting.
pub fn lstsq(rows: &[(Vec<f64>, f64)], n: usize) -> Option<(Vec<f64>, f64)> {
    lstsq_weighted(rows, None, n)
}

/// Weighted variant: `w[i]` multiplies row i (None = all ones).
pub fn lstsq_weighted(
    rows: &[(Vec<f64>, f64)],
    w: Option<&[f64]>,
    n: usize,
) -> Option<(Vec<f64>, f64)> {
    if rows.len() < n {
        return None;
    }
    let mut ata = vec![vec![0.0; n]; n];
    let mut atb = vec![0.0; n];
    for (i, (a, b)) in rows.iter().enumerate() {
        let wi = w.map_or(1.0, |w| w[i]);
        for j in 0..n {
            atb[j] += wi * a[j] * b;
            for k in 0..n {
                ata[j][k] += wi * a[j] * a[k];
            }
        }
    }
    let x = solve(ata, atb)?;
    let mut ss = 0.0;
    let mut sw = 0.0;
    for (i, (a, b)) in rows.iter().enumerate() {
        let wi = w.map_or(1.0, |w| w[i]);
        let pred: f64 = a.iter().zip(&x).map(|(p, q)| p * q).sum();
        ss += wi * (b - pred).powi(2);
        sw += wi;
    }
    Some((x, (ss / sw.max(1e-12)).sqrt()))
}

/// Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)]
pub fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv =
            (col..n).max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())?;
        if a[piv][col].abs() < 1e-14 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            if f != 0.0 {
                for c in col..n {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for c in r + 1..n {
            s -= a[r][c] * x[c];
        }
        x[r] = s / a[r][r];
    }
    Some(x)
}

/// Ordinary linear regression `y = a + b x`; returns (a, b, se_b, rms, n).
pub fn linreg(x: &[f64], y: &[f64]) -> Option<(f64, f64, f64, f64, usize)> {
    let n = x.len();
    if n < 3 || n != y.len() {
        return None;
    }
    let mx = mean(x);
    let my = mean(y);
    let sxx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum();
    if sxx <= 0.0 {
        return None;
    }
    let sxy: f64 = x.iter().zip(y).map(|(u, v)| (u - mx) * (v - my)).sum();
    let b = sxy / sxx;
    let a = my - b * mx;
    let ss: f64 = x.iter().zip(y).map(|(u, v)| (v - a - b * u).powi(2)).sum();
    let rms = (ss / n as f64).sqrt();
    let se_b = (ss / (n - 2) as f64 / sxx).sqrt();
    Some((a, b, se_b, rms, n))
}

/// Savitzky–Golay style local polynomial fit of degree 2 at the centre of the
/// window `(t_i, y_i)` with the centre time `t0`; returns (value, slope, curvature).
pub fn local_quadratic(t: &[f64], y: &[f64], t0: f64) -> Option<(f64, f64, f64)> {
    if t.len() < 4 {
        return None;
    }
    let rows: Vec<(Vec<f64>, f64)> = t
        .iter()
        .zip(y)
        .map(|(ti, yi)| {
            let d = ti - t0;
            (vec![1.0, d, 0.5 * d * d], *yi)
        })
        .collect();
    let (x, _) = lstsq(&rows, 3)?;
    Some((x[0], x[1], x[2]))
}

/// ASCII histogram: fixed bins in `[lo, hi)`, bar width proportional to count.
#[derive(Debug, Clone, Serialize)]
pub struct Histogram {
    pub lo: f64,
    pub hi: f64,
    pub bin: f64,
    pub counts: Vec<usize>,
    pub below: usize,
    pub above: usize,
}

impl Histogram {
    pub fn new(v: &[f64], lo: f64, hi: f64, bins: usize) -> Histogram {
        let bin = (hi - lo) / bins as f64;
        let mut counts = vec![0; bins];
        let (mut below, mut above) = (0, 0);
        for &x in v {
            if !x.is_finite() {
                continue;
            }
            if x < lo {
                below += 1;
            } else if x >= hi {
                above += 1;
            } else {
                counts[(((x - lo) / bin) as usize).min(bins - 1)] += 1;
            }
        }
        Histogram {
            lo,
            hi,
            bin,
            counts,
            below,
            above,
        }
    }

    pub fn render(&self, label: &str, width: usize) -> String {
        let max = self.counts.iter().copied().max().unwrap_or(1).max(1);
        let mut out = format!("{label} (below {}, above {})\n", self.below, self.above);
        for (i, c) in self.counts.iter().enumerate() {
            let lo = self.lo + i as f64 * self.bin;
            let bar = "#".repeat(c * width / max);
            out += &format!("{:>9.3} .. {:<9.3} {:>6} {}\n", lo, lo + self.bin, c, bar);
        }
        out
    }
}

/// Wrap an angle to (-pi, pi].
pub fn wrap_angle(a: f64) -> f64 {
    let mut a = a % std::f64::consts::TAU;
    if a > std::f64::consts::PI {
        a -= std::f64::consts::TAU;
    } else if a <= -std::f64::consts::PI {
        a += std::f64::consts::TAU;
    }
    a
}

pub fn hypot(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

/// Levenberg–Marquardt least squares with a numerical Jacobian.
/// `residuals(theta)` returns the residual vector. Returns (theta, rms).
pub fn lm_fit(
    mut theta: Vec<f64>,
    residuals: &dyn Fn(&[f64]) -> Vec<f64>,
    max_iter: usize,
) -> Option<(Vec<f64>, f64)> {
    let n = theta.len();
    let mut lambda = 1e-3;
    let mut r = residuals(&theta);
    let mut cost: f64 = r.iter().map(|x| x * x).sum();
    for _ in 0..max_iter {
        let m = r.len();
        if m < n {
            return None;
        }
        let mut jac = vec![vec![0.0; n]; m];
        for j in 0..n {
            let h = 1e-6 * theta[j].abs().max(1e-3);
            let mut tp = theta.clone();
            tp[j] += h;
            let rp = residuals(&tp);
            for i in 0..m {
                jac[i][j] = (rp[i] - r[i]) / h;
            }
        }
        let mut jtj = vec![vec![0.0; n]; n];
        let mut jtr = vec![0.0; n];
        for i in 0..m {
            for a in 0..n {
                jtr[a] += jac[i][a] * r[i];
                for b in 0..n {
                    jtj[a][b] += jac[i][a] * jac[i][b];
                }
            }
        }
        let mut improved = false;
        for _ in 0..12 {
            let mut a = jtj.clone();
            for (k, row) in a.iter_mut().enumerate() {
                row[k] *= 1.0 + lambda;
                row[k] += 1e-12;
            }
            let Some(delta) = solve(a, jtr.iter().map(|x| -x).collect()) else {
                lambda *= 10.0;
                continue;
            };
            let cand: Vec<f64> = theta.iter().zip(&delta).map(|(t, d)| t + d).collect();
            let rc = residuals(&cand);
            let cc: f64 = rc.iter().map(|x| x * x).sum();
            if cc < cost {
                let rel = (cost - cc) / cost.max(1e-30);
                theta = cand;
                r = rc;
                cost = cc;
                lambda = (lambda * 0.3).max(1e-9);
                improved = true;
                if rel < 1e-9 {
                    return Some((theta, (cost / m as f64).sqrt()));
                }
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    let m = r.len().max(1);
    Some((theta, (cost / m as f64).sqrt()))
}

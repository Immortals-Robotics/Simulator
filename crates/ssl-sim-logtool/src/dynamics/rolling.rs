//! Flat rolling: free-ball episodes and the fit of the deceleration law.
//!
//! Every episode is fitted in the position domain along its travel direction
//! (`s(t)`), which is far less noisy than differentiating positions. Three
//! models are compared per episode:
//!
//! * A: single constant deceleration  `s = s0 + v0 t - a t^2 / 2`
//! * B: two-phase (slide a_s then roll a_r) with the switch time on a grid
//! * C: linear drag `dv/dt = -(a + b v)`
//!
//! A per-camera offset column absorbs the calibration offset between the two
//! cameras in the overlap band. Thresholds are constants below.

use serde::Serialize;

use super::kicks::TrackerKick;
use super::stats::{self, lstsq, Histogram, Summary};
use super::track::BallSample;

/// No robot closer than this [m].
pub const FREE_DIST: f64 = 0.25;
/// Tracker height below which the ball counts as on the ground [m].
pub const GROUND_Z: f64 = 0.03;
/// Episode must start at least this fast [m/s].
pub const START_SPEED: f64 = 0.3;
/// Episode ends when the speed falls below this [m/s].
pub const END_SPEED: f64 = 0.08;
/// Split the episode on a per-camera velocity jump larger than this [m/s] (plus 10 % of the speed).
pub const JUMP_SPEED: f64 = 0.45;
/// Split on a direction change larger than this [rad] between consecutive same-camera velocities.
pub const JUMP_ANGLE: f64 = 0.35;
/// Max gap between samples [s].
pub const MAX_GAP: f64 = 0.12;
/// Minimum episode duration [s] and samples.
pub const MIN_DURATION: f64 = 0.4;
pub const MIN_SAMPLES: usize = 20;
/// Minimum path length [m].
pub const MIN_LENGTH: f64 = 0.3;
/// Outlier rejection: drop samples with |residual| > this many MAD sigmas and refit.
const OUTLIER_SIGMA: f64 = 4.0;
/// Half-window (samples) for the pooled local deceleration table.
const POOL_HALF: usize = 8;

/// One fitted episode.
#[derive(Debug, Clone, Serialize)]
pub struct Episode {
    pub game: String,
    pub competition: String,
    pub t0: f64,
    pub duration: f64,
    pub n: usize,
    pub cams: Vec<u32>,
    pub v_start: f64,
    pub v_end: f64,
    /// Travel direction [rad].
    pub dir: f64,
    pub length: f64,
    /// Model A.
    pub a_single: f64,
    pub rms_a: f64,
    /// Model B.
    pub a_slide: f64,
    pub a_roll: f64,
    pub t_switch: f64,
    pub v_switch: f64,
    pub slide_dur: f64,
    pub roll_dur: f64,
    pub rms_b: f64,
    /// Model C: dv/dt = -(c_a + c_b v).
    pub c_a: f64,
    pub c_b: f64,
    pub rms_c: f64,
    /// Fitted v0 of model B at episode start (t0).
    pub v0_fit: f64,
    /// Kick anchoring: tracker kick time relative to t0 (negative), tracker v0, our v0 extrapolated to the kick.
    pub kick_dt: Option<f64>,
    pub kick_v0_tracker: Option<f64>,
    pub kick_v0_extrap: Option<f64>,
    /// Bayesian information criterion differences (positive = second model better).
    pub bic_a_minus_b: f64,
    pub bic_a_minus_c: f64,
    pub bic_c_minus_b: f64,
    /// True when the rolling ball reached rest inside the episode.
    pub to_rest: bool,
}

/// Segment the free-rolling episodes of a ball track.
pub fn segment(track: &[BallSample]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let n = track.len();
    let free = |s: &BallSample| {
        s.near_dist > FREE_DIST && s.z.is_finite() && s.z < GROUND_Z && s.vx.is_finite()
    };
    let mut i = 0;
    while i < n {
        if !(free(&track[i]) && track[i].speed() > START_SPEED) {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i + 1;
        let mut last_v: std::collections::BTreeMap<u32, (f64, f64)> =
            std::collections::BTreeMap::new();
        last_v.insert(track[i].cam, (track[i].vx, track[i].vy));
        let mut to_rest = false;
        while j < n {
            let s = &track[j];
            if s.t - track[j - 1].t > MAX_GAP || !free(s) {
                break;
            }
            if s.speed() < END_SPEED {
                to_rest = true;
                break;
            }
            if let Some(&(pvx, pvy)) = last_v.get(&s.cam) {
                let dv = stats::hypot(s.vx - pvx, s.vy - pvy);
                let sp = stats::hypot(pvx, pvy);
                let ang = ((s.vx * pvx + s.vy * pvy) / (sp * s.speed()).max(1e-9))
                    .clamp(-1.0, 1.0)
                    .acos();
                if dv > JUMP_SPEED + 0.1 * sp || (sp > 0.3 && ang > JUMP_ANGLE) {
                    break;
                }
            }
            last_v.insert(s.cam, (s.vx, s.vy));
            j += 1;
        }
        let end = if to_rest { j + 1 } else { j };
        let end = end.min(n);
        let dur = track[end - 1].t - track[start].t;
        let len = stats::hypot(
            track[end - 1].x - track[start].x,
            track[end - 1].y - track[start].y,
        );
        if end - start >= MIN_SAMPLES && dur >= MIN_DURATION && len >= MIN_LENGTH {
            out.push((start, end));
        }
        i = end.max(start + 1);
    }
    out
}

struct Proj {
    tau: Vec<f64>,
    s: Vec<f64>,
    cam_col: Vec<f64>,
    ncam: usize,
}

fn project(track: &[BallSample]) -> Proj {
    let first = &track[0];
    let last = &track[track.len() - 1];
    let (dx, dy) = (last.x - first.x, last.y - first.y);
    let len = stats::hypot(dx, dy).max(1e-9);
    let (ux, uy) = (dx / len, dy / len);
    let cam0 = first.cam;
    let ncam = if track.iter().any(|s| s.cam != cam0) {
        2
    } else {
        1
    };
    Proj {
        tau: track.iter().map(|s| s.t - first.t).collect(),
        s: track
            .iter()
            .map(|s| (s.x - first.x) * ux + (s.y - first.y) * uy)
            .collect(),
        cam_col: track
            .iter()
            .map(|s| if s.cam != cam0 { 1.0 } else { 0.0 })
            .collect(),
        ncam,
    }
}

/// Fit a row-builder model with outlier rejection; returns (params, rms, n_used).
fn fit_model(
    p: &Proj,
    mask: &[bool],
    nparams: usize,
    row: impl Fn(f64) -> Vec<f64>,
) -> Option<(Vec<f64>, f64, usize)> {
    let build = |use_mask: &[bool]| -> Vec<(Vec<f64>, f64)> {
        p.tau
            .iter()
            .zip(&p.s)
            .zip(&p.cam_col)
            .zip(use_mask)
            .filter(|(_, m)| **m)
            .map(|(((t, s), c), _)| {
                let mut r = row(*t);
                if p.ncam == 2 {
                    r.push(*c);
                }
                (r, *s)
            })
            .collect()
    };
    let ncols = nparams + (p.ncam - 1);
    let rows = build(mask);
    let (x, _) = lstsq(&rows, ncols)?;
    // residuals and outlier rejection
    let res: Vec<f64> = rows
        .iter()
        .map(|(a, b)| b - a.iter().zip(&x).map(|(u, v)| u * v).sum::<f64>())
        .collect();
    let sig = stats::mad_sigma(&res).max(5e-4);
    let mut mask2 = mask.to_vec();
    let mut k = 0;
    for (i, m) in mask2.iter_mut().enumerate() {
        if *m {
            if res[k].abs() > OUTLIER_SIGMA * sig {
                *m = false;
            }
            k += 1;
        }
        let _ = i;
    }
    let rows2 = build(&mask2);
    if rows2.len() < ncols + 3 {
        return None;
    }
    let (x2, rms) = lstsq(&rows2, ncols)?;
    Some((x2, rms, rows2.len()))
}

fn bic(rms: f64, n: usize, k: usize) -> f64 {
    let nf = n as f64;
    nf * (rms * rms).max(1e-12).ln() + k as f64 * nf.ln()
}

/// Fit one episode.
pub fn fit_episode(
    game: &str,
    competition: &str,
    track: &[BallSample],
    kicks: &[TrackerKick],
) -> Option<Episode> {
    let p = project(track);
    let n = track.len();
    let mask = vec![true; n];
    let tau_end = p.tau[n - 1];
    // Model A
    let (xa, rms_a, na) = fit_model(&p, &mask, 3, |t| vec![1.0, t, -0.5 * t * t])?;
    // Model B: grid over switch time
    let mut best_b: Option<(f64, Vec<f64>, f64, usize)> = None;
    let mut grid: Vec<f64> = p.tau.iter().step_by(2).copied().collect();
    grid.push(tau_end);
    for &tsw in &grid {
        let r = fit_model(&p, &mask, 4, |t| {
            let slide = if t <= tsw {
                0.5 * t * t
            } else {
                tsw * t - 0.5 * tsw * tsw
            };
            let roll = if t <= tsw {
                0.0
            } else {
                0.5 * (t - tsw) * (t - tsw)
            };
            vec![1.0, t, -slide, -roll]
        });
        if let Some((x, rms, nu)) = r {
            if best_b.as_ref().is_none_or(|b| rms < b.2) {
                best_b = Some((tsw, x, rms, nu));
            }
        }
    }
    let (tsw, xb, rms_b, nb) = best_b?;
    // Model C: grid over b
    let mut best_c: Option<(f64, Vec<f64>, f64, usize)> = None;
    for k in 0..40 {
        let b = 0.01 * (1.2f64).powi(k); // 0.01 .. ~12
        let r = fit_model(&p, &mask, 3, |t| vec![1.0, (1.0 - (-b * t).exp()) / b, -t]);
        if let Some((x, rms, nu)) = r {
            if best_c.as_ref().is_none_or(|c| rms < c.2) {
                best_c = Some((b, x, rms, nu));
            }
        }
    }
    let (cb, xc, rms_c, nc) = best_c?;
    let c_a = xc[2] * cb;
    let v0_fit = xb[1];
    let v_switch = v0_fit - xb[2] * tsw;
    let dir = (track[n - 1].y - track[0].y).atan2(track[n - 1].x - track[0].x);
    let v_end = track[n - 1].speed();
    // kick anchoring: a tracker kick starting within 0.6 s before t0 and within 1.5 m of the start
    let t0 = track[0].t;
    let kick = kicks
        .iter()
        .filter(|k| k.start <= t0 + 0.02 && t0 - k.start < 0.6)
        .filter(|k| stats::hypot(k.pos[0] - track[0].x, k.pos[1] - track[0].y) < 1.5)
        .min_by(|a, b| (t0 - a.start).partial_cmp(&(t0 - b.start)).unwrap());
    let (kick_dt, kick_v0_tracker, kick_v0_extrap) = match kick {
        Some(k) => {
            let dt = k.start - t0;
            let v_tr = stats::hypot(k.vel[0], k.vel[1]);
            // extrapolate model B back to the kick assuming the slide phase covers it
            let a_back = if tsw > 0.0 { xb[2] } else { xb[3] };
            (Some(dt), Some(v_tr), Some(v0_fit - a_back * dt))
        }
        None => (None, None, None),
    };
    let nn = na.min(nb).min(nc);
    Some(Episode {
        game: game.to_string(),
        competition: competition.to_string(),
        t0,
        duration: tau_end,
        n,
        cams: {
            let mut c: Vec<u32> = track.iter().map(|s| s.cam).collect();
            c.sort();
            c.dedup();
            c
        },
        v_start: track[0].speed(),
        v_end,
        dir,
        length: p.s[n - 1],
        a_single: xa[2],
        rms_a,
        a_slide: xb[2],
        a_roll: xb[3],
        t_switch: tsw,
        v_switch,
        slide_dur: tsw,
        roll_dur: tau_end - tsw,
        rms_b,
        c_a,
        c_b: cb,
        rms_c,
        v0_fit,
        kick_dt,
        kick_v0_tracker,
        kick_v0_extrap,
        bic_a_minus_b: bic(rms_a, nn, 3) - bic(rms_b, nn, 5),
        bic_a_minus_c: bic(rms_a, nn, 3) - bic(rms_c, nn, 4),
        bic_c_minus_b: bic(rms_c, nn, 4) - bic(rms_b, nn, 5),
        to_rest: v_end < END_SPEED,
    })
}

/// Pooled local deceleration samples (speed, decel) from the roll phase of
/// each episode, from a local quadratic fit over same-camera samples.
pub fn pooled_decel(track: &[BallSample], ep: &Episode) -> Vec<(f64, f64, f64, String)> {
    let p = project(track);
    let n = track.len();
    let mut out = Vec::new();
    for i in 0..n {
        if p.tau[i] < ep.t_switch + 0.15 || ep.rms_b > 0.003 {
            continue;
        }
        let cam = track[i].cam;
        let mut ts = Vec::new();
        let mut ss = Vec::new();
        let lo = i.saturating_sub(3 * POOL_HALF);
        for (j, s) in track
            .iter()
            .enumerate()
            .take((i + 3 * POOL_HALF + 1).min(n))
            .skip(lo)
        {
            if s.cam == cam && (p.tau[j] - p.tau[i]).abs() <= (POOL_HALF as f64 + 0.5) * 0.0145 {
                ts.push(p.tau[j]);
                ss.push(p.s[j]);
            }
        }
        if ts.len() < 2 * POOL_HALF - 2 {
            continue;
        }
        if let Some((_, v, a)) = stats::local_quadratic(&ts, &ss, p.tau[i]) {
            out.push((v, -a, ep.dir, ep.competition.clone()));
        }
    }
    out
}

/// Per-competition row: (name, acc_roll, acc_slide_fast, k_switch, pooled fit (a, b, se_b, n), n_episodes).
pub type CompetitionRow = (
    String,
    Summary,
    Summary,
    Summary,
    Option<(f64, f64, f64, usize)>,
    usize,
);

/// Aggregated rolling results.
#[derive(Debug, Clone, Serialize)]
pub struct RollingResult {
    pub thresholds: serde_json::Value,
    pub n_episodes: usize,
    pub n_kick_anchored: usize,
    pub per_game: Vec<(String, usize)>,
    /// acc_roll from model B roll phases (roll_dur >= 0.4 s).
    pub acc_roll: Summary,
    /// acc_roll from model A on slow episodes (v_start < 1 m/s, no slide phase expected).
    pub acc_roll_slow_single: Summary,
    /// acc_slide from model B (slide_dur >= 0.12 s, v_start > 1.5 m/s).
    pub acc_slide: Summary,
    /// acc_slide from fast episodes only (v_start >= 3 m/s, slide_dur >= 0.12 s).
    pub acc_slide_fast: Summary,
    /// Per competition: (name, acc_roll, acc_slide_fast, k_switch, pooled fit a + b v (v<2), n_episodes).
    pub per_competition: Vec<CompetitionRow>,
    /// Pooled quadratic fit decel = a + c v^2 over the whole roll-phase speed range: (a, c, se_c, n).
    pub pooled_quadratic_fit: Option<(f64, f64, f64, usize)>,
    /// k_switch from kick-anchored episodes (v_switch / v0 at kick, our extrapolation).
    pub k_switch: Summary,
    /// k_switch using the tracker's kick v0.
    pub k_switch_tracker_v0: Summary,
    /// Implied inertia distribution p = 1/k - 1.
    pub inertia_p: f64,
    pub inertia_p_se: f64,
    /// Model comparison.
    pub frac_b_beats_a: f64,
    pub frac_c_beats_a: f64,
    pub frac_b_beats_c: f64,
    pub median_bic_a_minus_b: f64,
    pub median_bic_a_minus_c: f64,
    pub median_bic_c_minus_b: f64,
    pub rms_a: Summary,
    pub rms_b: Summary,
    pub rms_c: Summary,
    /// Pooled regression decel = a + b v on roll-phase samples with v < 2 m/s: (a, b, se_b, n).
    pub pooled_decel_fit: Option<(f64, f64, f64, usize)>,
    /// Linear-drag coefficient b [1/s] of model C on pure-roll episodes.
    pub drag_b: Summary,
    pub drag_a: Summary,
    /// Median deceleration vs speed bin (roll phase), bins of 0.25 m/s: (v_lo, median, se, n).
    pub decel_vs_speed: Vec<(f64, f64, f64, usize)>,
    /// Median acc_roll vs travel direction (8 bins of 45 deg): (deg_lo, median, se, n).
    pub roll_vs_direction: Vec<(f64, f64, f64, usize)>,
    /// Slide / roll and a_single vs v_start bins: (v_lo, med a_single, med a_slide, med a_roll, n).
    pub by_start_speed: Vec<(f64, f64, f64, f64, usize)>,
    pub hist_a_roll: Histogram,
    pub hist_a_slide: Histogram,
    pub hist_k: Histogram,
    pub episodes: Vec<Episode>,
    pub text: String,
}

/// Aggregate episodes.
pub fn aggregate(episodes: Vec<Episode>, pooled: &[(f64, f64, f64, String)]) -> RollingResult {
    let roll: Vec<f64> = episodes
        .iter()
        .filter(|e| e.roll_dur >= 0.4 && e.a_roll.abs() < 2.0)
        .map(|e| -e.a_roll)
        .collect();
    let roll_slow: Vec<f64> = episodes
        .iter()
        .filter(|e| e.v_start < 1.0 && e.kick_dt.is_none())
        .map(|e| -e.a_single)
        .collect();
    let slide: Vec<f64> = episodes
        .iter()
        .filter(|e| {
            e.slide_dur >= 0.12 && e.v_start > 1.5 && e.roll_dur > 0.15 && e.a_slide > e.a_roll
        })
        .map(|e| -e.a_slide)
        .collect();
    let slide_fast_sel = |e: &Episode| {
        e.slide_dur >= 0.12 && e.v_start >= 3.0 && e.roll_dur > 0.15 && e.a_slide > e.a_roll
    };
    let slide_fast: Vec<f64> = episodes
        .iter()
        .filter(|e| slide_fast_sel(e))
        .map(|e| -e.a_slide)
        .collect();
    let k_sel = |e: &Episode| -> Option<(f64, f64)> {
        let (vtr, vex) = (e.kick_v0_tracker?, e.kick_v0_extrap?);
        if e.slide_dur >= 0.05
            && e.roll_dur >= 0.2
            && e.a_slide > e.a_roll
            && e.a_slide > 1.0
            && vex > 1.0
        {
            Some((
                e.v_switch / vex,
                if vtr > 1.0 {
                    e.v_switch / vtr
                } else {
                    f64::NAN
                },
            ))
        } else {
            None
        }
    };
    let ks: Vec<f64> = episodes.iter().filter_map(k_sel).map(|k| k.0).collect();
    let ks_tr: Vec<f64> = episodes
        .iter()
        .filter_map(k_sel)
        .map(|k| k.1)
        .filter(|k| k.is_finite())
        .collect();
    let roll_sel = |e: &Episode| e.roll_dur >= 0.4 && e.a_roll.abs() < 2.0;
    let mut comps: Vec<String> = episodes.iter().map(|e| e.competition.clone()).collect();
    comps.sort();
    comps.dedup();
    let per_competition: Vec<CompetitionRow> = comps
        .iter()
        .map(|c| {
            let eps: Vec<&Episode> = episodes.iter().filter(|e| &e.competition == c).collect();
            let (pv, pa): (Vec<f64>, Vec<f64>) = pooled
                .iter()
                .filter(|p| &p.3 == c && p.0 < 2.0 && p.0 > 0.1)
                .map(|p| (p.0, p.1))
                .unzip();
            (
                c.clone(),
                Summary::of(
                    &eps.iter()
                        .filter(|e| roll_sel(e))
                        .map(|e| -e.a_roll)
                        .collect::<Vec<_>>(),
                ),
                Summary::of(
                    &eps.iter()
                        .filter(|e| slide_fast_sel(e))
                        .map(|e| -e.a_slide)
                        .collect::<Vec<_>>(),
                ),
                Summary::of(
                    &eps.iter()
                        .filter_map(|e| k_sel(e))
                        .map(|k| k.0)
                        .collect::<Vec<_>>(),
                ),
                stats::linreg(&pv, &pa).map(|(a, b, se, _, n)| (a, b, se, n)),
                eps.len(),
            )
        })
        .collect();
    // quadratic pooled fit over the full range
    let qrows: Vec<(Vec<f64>, f64)> = pooled
        .iter()
        .filter(|p| p.0 > 0.1 && p.0 < 4.0)
        .map(|p| (vec![1.0, p.0 * p.0], p.1))
        .collect();
    let pooled_quadratic_fit = stats::lstsq(&qrows, 2).map(|(x, rms)| {
        let sxx: f64 = {
            let m = qrows.iter().map(|r| r.0[1]).sum::<f64>() / qrows.len() as f64;
            qrows.iter().map(|r| (r.0[1] - m).powi(2)).sum()
        };
        (x[0], x[1], rms / sxx.max(1e-12).sqrt(), qrows.len())
    });
    let k_sum = Summary::of(&ks);
    let inertia_p = 1.0 / k_sum.median - 1.0;
    let inertia_p_se = k_sum.se / (k_sum.median * k_sum.median);
    let n = episodes.len().max(1) as f64;
    let frac = |f: &dyn Fn(&Episode) -> bool| episodes.iter().filter(|e| f(e)).count() as f64 / n;
    // pooled decel vs speed
    let mut decel_vs_speed = Vec::new();
    for b in 0..16 {
        let lo = b as f64 * 0.25;
        let v: Vec<f64> = pooled
            .iter()
            .filter(|p| p.0 >= lo && p.0 < lo + 0.25)
            .map(|p| p.1)
            .collect();
        if v.len() >= 20 {
            let s = Summary::of(&v);
            decel_vs_speed.push((lo, s.median, s.se, s.n));
        }
    }
    let mut roll_vs_direction = Vec::new();
    for b in 0..8 {
        let lo = -180.0 + b as f64 * 45.0;
        let v: Vec<f64> = episodes
            .iter()
            .filter(|e| e.roll_dur >= 0.4 && e.a_roll.abs() < 2.0)
            .filter(|e| {
                let d = e.dir.to_degrees();
                d >= lo && d < lo + 45.0
            })
            .map(|e| -e.a_roll)
            .collect();
        let s = Summary::of(&v);
        roll_vs_direction.push((lo, s.median, s.se, s.n));
    }
    let mut by_start_speed = Vec::new();
    for b in 0..14 {
        let lo = b as f64 * 0.5;
        let sel: Vec<&Episode> = episodes
            .iter()
            .filter(|e| e.v_start >= lo && e.v_start < lo + 0.5)
            .collect();
        if sel.len() >= 5 {
            by_start_speed.push((
                lo,
                stats::median(&sel.iter().map(|e| -e.a_single).collect::<Vec<_>>()),
                stats::median(&sel.iter().map(|e| -e.a_slide).collect::<Vec<_>>()),
                stats::median(&sel.iter().map(|e| -e.a_roll).collect::<Vec<_>>()),
                sel.len(),
            ));
        }
    }
    let pure_roll: Vec<&Episode> = episodes
        .iter()
        .filter(|e| e.v_start < 1.2 && e.duration > 0.8)
        .collect();
    let drag_b = Summary::of(&pure_roll.iter().map(|e| e.c_b).collect::<Vec<_>>());
    let drag_a = Summary::of(&pure_roll.iter().map(|e| e.c_a).collect::<Vec<_>>());
    let mut per_game: std::collections::BTreeMap<String, usize> = Default::default();
    for e in &episodes {
        *per_game.entry(e.game.clone()).or_default() += 1;
    }
    let acc_roll = Summary::of(&roll);
    let acc_slide = Summary::of(&slide);
    let hist_a_roll = Histogram::new(&roll, -1.0, 0.0, 20);
    let hist_a_slide = Histogram::new(&slide, -8.0, 0.0, 16);
    // pooled linear regression of the roll-phase deceleration on speed (v < 2 m/s)
    let (pv, pa): (Vec<f64>, Vec<f64>) = pooled
        .iter()
        .filter(|p| p.0 < 2.0 && p.0 > 0.1)
        .map(|p| (p.0, p.1))
        .unzip();
    let pooled_fit = stats::linreg(&pv, &pa).map(|(a, b, se, _, n)| (a, b, se, n));
    let hist_k = Histogram::new(&ks, 0.3, 1.0, 14);
    let mut text = String::new();
    text += &format!("episodes {} (kick-anchored {})\n", episodes.len(), ks.len());
    text += &format!(
        "acc_roll (model B roll phase): {}\n",
        acc_roll.line("m/s^2")
    );
    text += &format!(
        "acc_roll (model A, slow episodes): {}\n",
        Summary::of(&roll_slow).line("m/s^2")
    );
    text += &format!("acc_slide (model B): {}\n", acc_slide.line("m/s^2"));
    text += &format!(
        "acc_slide (fast episodes v0 >= 3): {}\n",
        Summary::of(&slide_fast).line("m/s^2")
    );
    for (c, r, s, k, fit, n) in &per_competition {
        text += &format!(
            "  {c}: episodes {n}; acc_roll {:.4} ± {:.4} (n {}); acc_slide_fast {:.3} ± {:.3} (n {}); k {:.3} ± {:.3} (n {}); pooled a(v)={:?}\n",
            -r.median, r.se, r.n, -s.median, s.se, s.n, k.median, k.se, k.n,
            fit.map(|f| ((f.0 * 1e3).round() / 1e3, (f.1 * 1e3).round() / 1e3, (f.2 * 1e3).round() / 1e3, f.3))
        );
    }
    if let Some((a, c, se, n)) = pooled_quadratic_fit {
        text += &format!("pooled roll-phase quadratic fit (0.1<v<4): a(v) = {:.3} + ({:.4} ± {:.4}) v^2 (n {})\n", a, c, se, n);
    }
    text += &format!("k_switch (our v0): {}\n", k_sum.line(""));
    text += &format!("k_switch (tracker v0): {}\n", Summary::of(&ks_tr).line(""));
    text += &format!("implied p = {:.3} ± {:.3}\n", inertia_p, inertia_p_se);
    text += &format!(
        "model comparison: B beats A in {:.0}% (median dBIC {:.1}), C beats A in {:.0}% ({:.1}), B beats C in {:.0}% ({:.1})\n",
        100.0 * frac(&|e| e.bic_a_minus_b > 10.0),
        stats::median(&episodes.iter().map(|e| e.bic_a_minus_b).collect::<Vec<_>>()),
        100.0 * frac(&|e| e.bic_a_minus_c > 10.0),
        stats::median(&episodes.iter().map(|e| e.bic_a_minus_c).collect::<Vec<_>>()),
        100.0 * frac(&|e| e.bic_c_minus_b > 10.0),
        stats::median(&episodes.iter().map(|e| e.bic_c_minus_b).collect::<Vec<_>>()),
    );
    text += &format!(
        "position rms [mm]: A {:.2} B {:.2} C {:.2}\n",
        1e3 * stats::median(&episodes.iter().map(|e| e.rms_a).collect::<Vec<_>>()),
        1e3 * stats::median(&episodes.iter().map(|e| e.rms_b).collect::<Vec<_>>()),
        1e3 * stats::median(&episodes.iter().map(|e| e.rms_c).collect::<Vec<_>>()),
    );
    text += &format!(
        "linear drag (pure roll episodes, n {}): a {:.3} ± {:.3}, b {:.3} ± {:.3} 1/s\n",
        drag_b.n, drag_a.median, drag_a.se, drag_b.median, drag_b.se
    );
    if let Some((a, b, se, n)) = pooled_fit {
        text += &format!(
            "pooled roll-phase decel vs speed (v<2): a(v) = {:.3} + {:.3} ± {:.3} * v (n {})\n",
            a, b, se, n
        );
    }
    text += "decel vs speed (roll phase, pooled local fits):\n";
    for (lo, m, se, n) in &decel_vs_speed {
        text += &format!(
            "  v {:.2}..{:.2}: {:.3} ± {:.3} (n {})\n",
            lo,
            lo + 0.25,
            m,
            se,
            n
        );
    }
    text += "acc_roll vs direction:\n";
    for (lo, m, se, n) in &roll_vs_direction {
        text += &format!(
            "  {:>5.0}..{:>5.0} deg: {:.3} ± {:.3} (n {})\n",
            lo,
            lo + 45.0,
            m,
            se,
            n
        );
    }
    text += "by start speed (v_lo, a_single, a_slide, a_roll, n):\n";
    for (lo, a, s, r, n) in &by_start_speed {
        text += &format!("  {:.1}: {:.3} {:.3} {:.3} ({})\n", lo, a, s, r, n);
    }
    text += &hist_a_roll.render("a_roll histogram", 40);
    text += &hist_a_slide.render("a_slide histogram", 40);
    text += &hist_k.render("k_switch histogram", 40);
    RollingResult {
        thresholds: serde_json::json!({
            "free_dist_m": FREE_DIST, "ground_z_m": GROUND_Z, "start_speed": START_SPEED,
            "end_speed": END_SPEED, "jump_speed": JUMP_SPEED, "jump_angle_rad": JUMP_ANGLE,
            "max_gap_s": MAX_GAP, "min_duration_s": MIN_DURATION, "min_samples": MIN_SAMPLES,
            "min_length_m": MIN_LENGTH, "outlier_sigma": OUTLIER_SIGMA,
            "acc_roll_selection": "roll_dur >= 0.4 s",
            "acc_slide_selection": "slide_dur >= 0.12 s, v_start > 1.5, roll_dur > 0.15",
            "k_selection": "kick-anchored, slide_dur >= 0.05, roll_dur >= 0.2, a_slide > 1",
        }),
        n_episodes: episodes.len(),
        n_kick_anchored: ks.len(),
        per_game: per_game.into_iter().collect(),
        acc_roll,
        acc_roll_slow_single: Summary::of(&roll_slow),
        acc_slide,
        acc_slide_fast: Summary::of(&slide_fast),
        per_competition,
        pooled_quadratic_fit,
        k_switch: k_sum,
        k_switch_tracker_v0: Summary::of(&ks_tr),
        inertia_p,
        inertia_p_se,
        frac_b_beats_a: frac(&|e| e.bic_a_minus_b > 10.0),
        frac_c_beats_a: frac(&|e| e.bic_a_minus_c > 10.0),
        frac_b_beats_c: frac(&|e| e.bic_c_minus_b > 10.0),
        median_bic_a_minus_b: stats::median(
            &episodes.iter().map(|e| e.bic_a_minus_b).collect::<Vec<_>>(),
        ),
        median_bic_a_minus_c: stats::median(
            &episodes.iter().map(|e| e.bic_a_minus_c).collect::<Vec<_>>(),
        ),
        median_bic_c_minus_b: stats::median(
            &episodes.iter().map(|e| e.bic_c_minus_b).collect::<Vec<_>>(),
        ),
        rms_a: Summary::of(&episodes.iter().map(|e| e.rms_a).collect::<Vec<_>>()),
        rms_b: Summary::of(&episodes.iter().map(|e| e.rms_b).collect::<Vec<_>>()),
        rms_c: Summary::of(&episodes.iter().map(|e| e.rms_c).collect::<Vec<_>>()),
        pooled_decel_fit: pooled_fit,
        drag_b,
        drag_a,
        decel_vs_speed,
        roll_vs_direction,
        by_start_speed,
        hist_a_roll,
        hist_a_slide,
        hist_k,
        episodes,
        text,
    }
}

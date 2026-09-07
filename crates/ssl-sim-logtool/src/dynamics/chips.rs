//! Chips: airborne episodes from the tracker height, hop-by-hop kinematics,
//! bounce damping factors and launch angles.
//!
//! Two independent methods per hop:
//!
//! * **T (timing)**: the tracker's bounce instants (local minima of its
//!   ball height) give the flight time `tf` of each hop, hence the vertical
//!   takeoff speed `vz = g tf / 2`; the raw ball positions *at the bounces*
//!   (ball on the floor, so no projection displacement) give the horizontal
//!   speed `vxy = |p_{n+1} - p_n| / tf`. Only tracker timing and grounded raw
//!   positions are used, not the tracker's velocity or bounce model.
//! * **R (raw projection fit)**: raw detections are floor projections through
//!   the camera, `p_obs = c_xy + (p - c_xy) (c_z - r) / (c_z - z)`; each hop is
//!   fitted with `(x0, y0, vx, vy, vz, dt0)` plus a per-camera offset by
//!   Levenberg–Marquardt with outlier rejection.

use serde::Serialize;

use super::kicks::TrackerKick;
use super::load::Game;
use super::stats::{self, hypot, lm_fit, Histogram, Summary};
use super::track::BallSample;

/// Tracker height above which the ball is airborne [m].
pub const AIR_Z: f64 = 0.03;
/// Airborne runs closer than this are one flight (the dips are bounces) [s].
const FLIGHT_GAP: f64 = 0.25;
/// Ball radius [m] (ssl-vision reports the ray intersection with the plane z = r).
const BALL_R: f64 = 0.0215;
const G: f64 = 9.81;
/// Minimum raw samples for a hop fit.
const MIN_HOP_SAMPLES: usize = 6;
/// Max 2D residual RMS for an accepted raw hop fit [m].
const MAX_HOP_RMS: f64 = 0.012;
/// Minimum flight time [s] for an accepted hop (both methods).
const MIN_HOP_T: f64 = 0.10;
/// Soft prior on takeoff/landing times from the tracker bounce times.
const T_PRIOR_SIGMA: f64 = 0.06;
const T_PRIOR_WEIGHT: f64 = 0.05;
/// Raw samples are taken from [ta - margin, tb + margin] for the fit.
const HOP_MARGIN: f64 = 0.02;
/// Outlier rejection in the raw fit (MAD sigmas).
const OUTLIER_SIGMA: f64 = 3.5;
/// Raw ground position at a bounce: median of samples within this of the bounce time [s].
const BOUNCE_POS_WINDOW: f64 = 0.02;
/// Bounce consistency for method R: landing point of hop n vs takeoff of n+1 [m], time gap [s].
const MAX_BOUNCE_POS_GAP: f64 = 0.12;
const MAX_BOUNCE_T_GAP: f64 = 0.06;
/// A hop counts as real only if the tracker saw an apex of at least this [m]
/// and the timing apex agrees with it within a factor of 2.
const MIN_TRACKER_APEX: f64 = 0.04;

/// One hop as seen by both methods.
#[derive(Debug, Clone, Serialize)]
pub struct Hop {
    pub idx: usize,
    /// Method T.
    pub t_takeoff: f64,
    pub t_land: f64,
    pub tf_t: f64,
    pub vz_t: f64,
    pub vxy_t: Option<f64>,
    pub apex_t: f64,
    pub tracker_apex: f64,
    /// Method R (None if the fit failed).
    pub r: Option<HopFit>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HopFit {
    pub t0: f64,
    pub flight_time: f64,
    pub x0: f64,
    pub y0: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub apex: f64,
    pub rms: f64,
    pub n: usize,
    pub cam_offset: Option<(f64, f64)>,
}

impl HopFit {
    fn vxy(&self) -> f64 {
        hypot(self.vx, self.vy)
    }
    fn landing(&self) -> (f64, f64, f64) {
        (
            self.x0 + self.vx * self.flight_time,
            self.y0 + self.vy * self.flight_time,
            self.t0 + self.flight_time,
        )
    }
}

/// One bounce between two hops.
#[derive(Debug, Clone, Serialize)]
pub struct Bounce {
    pub game: String,
    pub competition: String,
    /// 1 = first bounce after the kick.
    pub idx: usize,
    /// Method T.
    pub vz_in_t: f64,
    pub damping_z_t: f64,
    pub damping_xy_t: Option<f64>,
    /// Method R.
    pub damping_xy_r: Option<f64>,
    pub damping_z_r: Option<f64>,
    pub deflection_deg_r: Option<f64>,
    /// Tracker-implied factors (its own velocity before/after the bounce).
    pub tracker_damping_xy: Option<f64>,
    pub tracker_damping_z: Option<f64>,
}

/// One flight (kick until grounded).
#[derive(Debug, Clone, Serialize)]
pub struct Flight {
    pub game: String,
    pub competition: String,
    pub t_start: f64,
    pub n_hops_tracker: usize,
    pub hops: Vec<Hop>,
    pub launch_angle_deg_t: Option<f64>,
    pub launch_speed_t: Option<f64>,
    pub launch_angle_deg_r: Option<f64>,
    pub launch_speed_r: Option<f64>,
    pub kicker: Option<(super::load::Team, u32)>,
    pub kicker_team: Option<String>,
    pub tracker_kick_vz: Option<f64>,
    pub tracker_kick_vxy: Option<f64>,
    pub last_apex_tracker: f64,
}

/// Airborne flights as (start, end) tracker indices plus bounce indices.
fn find_flights(game: &Game) -> Vec<(usize, usize, Vec<usize>)> {
    let tr = &game.tracker;
    let air: Vec<bool> = tr
        .iter()
        .map(|f| f.ball.is_some_and(|b| b.z > AIR_Z))
        .collect();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < tr.len() {
        if !air[i] {
            i += 1;
            continue;
        }
        let s = i;
        while i < tr.len() && air[i] {
            i += 1;
        }
        runs.push((s, i - 1));
    }
    let mut flights: Vec<(usize, usize)> = Vec::new();
    for r in runs {
        match flights.last_mut() {
            Some(last) if tr[r.0].t - tr[last.1].t < FLIGHT_GAP => last.1 = r.1,
            _ => flights.push(r),
        }
    }
    flights
        .into_iter()
        .filter(|(s, e)| e > s && (*e - *s) >= 3)
        .map(|(s, e)| {
            // bounces: local minima of z inside the flight, below 2 * AIR_Z, followed by a rise
            let z = |i: usize| tr[i].ball.map_or(0.0, |b| b.z);
            let mut bounces = Vec::new();
            let mut last_max_ok = true;
            for i in s + 1..e {
                let zi = z(i);
                if zi > AIR_Z {
                    last_max_ok = true;
                }
                if zi < 2.0 * AIR_Z && zi <= z(i - 1) && zi < z(i + 1) && last_max_ok {
                    let rise = (i + 1..(i + 6).min(e + 1)).any(|j| z(j) > zi + 0.02);
                    if rise {
                        bounces.push(i);
                        last_max_ok = false;
                    }
                }
            }
            (s, e, bounces)
        })
        .collect()
}

/// Median raw ball position within ±BOUNCE_POS_WINDOW of `t` (any camera; the ball is on the floor).
fn ground_pos_at(ball: &[BallSample], t: f64) -> Option<(f64, f64)> {
    let a = ball.partition_point(|s| s.t < t - BOUNCE_POS_WINDOW);
    let b = ball.partition_point(|s| s.t < t + BOUNCE_POS_WINDOW);
    if b <= a {
        return None;
    }
    let xs: Vec<f64> = ball[a..b].iter().map(|s| s.x).collect();
    let ys: Vec<f64> = ball[a..b].iter().map(|s| s.y).collect();
    Some((stats::median(&xs), stats::median(&ys)))
}

/// Fit one hop from raw samples in the window [ta, tb] (method R).
fn fit_hop(
    game: &Game,
    ball: &[BallSample],
    ta: f64,
    tb: f64,
    guess: (f64, f64, f64, f64, f64),
) -> Option<HopFit> {
    let cams = &game.cameras;
    let cam_pos = |id: u32| cams.iter().find(|c| c.id == id).map(|c| c.pos);
    let a = ball.partition_point(|s| s.t < ta - HOP_MARGIN);
    let b = ball.partition_point(|s| s.t < tb + HOP_MARGIN);
    let mut cam_ids: Vec<u32> = ball[a..b].iter().map(|s| s.cam).collect();
    cam_ids.sort();
    cam_ids.dedup();
    let two_cams = cam_ids.len() >= 2;
    // (t, x, y, camera position, offset column)
    let mut sel: Vec<(f64, f64, f64, [f64; 3], f64)> = ball[a..b]
        .iter()
        .filter_map(|s| {
            cam_pos(s.cam).map(|c| {
                (
                    s.t,
                    s.x,
                    s.y,
                    c,
                    if two_cams && s.cam != cam_ids[0] {
                        1.0
                    } else {
                        0.0
                    },
                )
            })
        })
        .collect();
    if sel.len() < MIN_HOP_SAMPLES {
        return None;
    }
    let (x0, y0, vx, vy, vz) = guess;
    // theta = (x0, y0, vx, vy, vz, dt0[, ox, oy]) with the takeoff time t0 = ta + dt0
    let mut theta = vec![x0, y0, vx, vy, vz, 0.0];
    if two_cams {
        theta.extend([0.0, 0.0]);
    }
    let model = |th: &[f64], t: f64, c: &[f64; 3], off: f64| -> (f64, f64) {
        let tf = 2.0 * th[4].max(0.0) / G;
        let tau = (t - ta - th[5]).clamp(0.0, tf);
        let px = th[0] + th[2] * tau;
        let py = th[1] + th[3] * tau;
        let z = (BALL_R + th[4] * tau - 0.5 * G * tau * tau).max(BALL_R);
        let f = (c[2] - BALL_R) / (c[2] - z).max(0.1);
        let (ox, oy) = if th.len() > 6 {
            (th[6] * off, th[7] * off)
        } else {
            (0.0, 0.0)
        };
        (c[0] + (px - c[0]) * f + ox, c[1] + (py - c[1]) * f + oy)
    };
    let mut rms = f64::NAN;
    for round in 0..3 {
        let data = sel.clone();
        let resid = |th: &[f64]| -> Vec<f64> {
            let mut r = Vec::with_capacity(data.len() * 2 + 2);
            let tf = 2.0 * th[4].max(0.0) / G;
            for (t, x, y, c, off) in &data {
                let (mx, my) = model(th, *t, c, *off);
                r.push(mx - x);
                r.push(my - y);
            }
            r.push(th[5] / T_PRIOR_SIGMA * T_PRIOR_WEIGHT);
            r.push((ta + th[5] + tf - tb) / T_PRIOR_SIGMA * T_PRIOR_WEIGHT);
            r
        };
        let (th, _) = lm_fit(theta.clone(), &resid, 60)?;
        theta = th;
        let tf = 2.0 * theta[4] / G;
        if !theta[4].is_finite() || theta[4] <= 0.0 {
            return None;
        }
        // residual distances per sample; reselect inside the flight and drop outliers
        let t0 = ta + theta[5];
        let dist: Vec<f64> = sel
            .iter()
            .map(|(t, x, y, c, off)| {
                let (mx, my) = model(&theta, *t, c, *off);
                hypot(mx - x, my - y)
            })
            .collect();
        let sig = stats::mad_sigma(&dist).max(0.002);
        rms = (dist.iter().map(|d| d * d).sum::<f64>() / dist.len().max(1) as f64).sqrt();
        if round < 2 {
            let keep: Vec<bool> = sel
                .iter()
                .zip(&dist)
                .map(|(s, d)| {
                    s.0 >= t0 + 0.004 && s.0 <= t0 + tf - 0.004 && *d < OUTLIER_SIGMA * sig + 0.003
                })
                .collect();
            let mut k = 0;
            sel.retain(|_| {
                k += 1;
                keep[k - 1]
            });
            if sel.len() < MIN_HOP_SAMPLES {
                return None;
            }
        }
    }
    let vz = theta[4];
    let tf = 2.0 * vz / G;
    if tf < MIN_HOP_T || rms > MAX_HOP_RMS {
        return None;
    }
    Some(HopFit {
        t0: ta + theta[5],
        flight_time: tf,
        x0: theta[0],
        y0: theta[1],
        vx: theta[2],
        vy: theta[3],
        vz,
        apex: vz * vz / (2.0 * G),
        rms,
        n: sel.len(),
        cam_offset: if two_cams {
            Some((theta[6], theta[7]))
        } else {
            None
        },
    })
}

/// Analyse all flights of a game.
pub fn analyse(
    game: &Game,
    ball: &[BallSample],
    kicks: &[TrackerKick],
    own_kicks: &[f64],
    verbose: bool,
) -> (Vec<Flight>, Vec<Bounce>) {
    let tr = &game.tracker;
    let mut flights = Vec::new();
    let mut bounces = Vec::new();
    for (s, e, bidx) in find_flights(game) {
        let t_start = tr[s].t;
        let period = if s > 0 { tr[s].t - tr[s - 1].t } else { 0.0125 };
        // takeoff: own kick detector (raw speed jump) preferred, else tracker kick, else first airborne frame
        let kick = kicks
            .iter()
            .filter(|k| k.vel[2] > 0.3 && (k.start - t_start).abs() < 0.25)
            .min_by(|a, b| {
                (a.start - t_start)
                    .abs()
                    .partial_cmp(&(b.start - t_start).abs())
                    .unwrap()
            });
        let own = own_kicks
            .iter()
            .copied()
            .filter(|&t| t <= t_start + 0.02 && t_start - t < 0.25)
            .min_by(|a, b| {
                (a - t_start)
                    .abs()
                    .partial_cmp(&(b - t_start).abs())
                    .unwrap()
            });
        let t_take = own
            .or(kick.map(|k| k.start.min(t_start)))
            .unwrap_or(t_start - period);
        let takeoff_known = own.is_some() || kick.is_some();
        let mut bounds: Vec<f64> = vec![t_take];
        bounds.extend(bidx.iter().map(|&i| tr[i].t));
        // final grounding: first grounded frame after the flight
        bounds.push(tr[e].t + period);
        let n_hops = bounds.len() - 1;
        let mut hops: Vec<Hop> = Vec::new();
        for h in 0..n_hops {
            let (ta, tb) = (bounds[h], bounds[h + 1]);
            let ia = tr.partition_point(|f| f.t < ta);
            let ib = tr.partition_point(|f| f.t < tb);
            if ib <= ia {
                continue;
            }
            let frames = &tr[ia..ib];
            let zmax = frames
                .iter()
                .filter_map(|f| f.ball.map(|b| b.z))
                .fold(0.0, f64::max);
            let (Some(b0), Some(b1)) = (
                frames.first().and_then(|f| f.ball),
                frames.last().and_then(|f| f.ball),
            ) else {
                continue;
            };
            let dt = (tb - ta).max(0.05);
            // method T: valid only when both ends are real bounce/kick instants
            let tf_t = tb - ta;
            let vz_t = 0.5 * G * tf_t;
            let ends_known = (h > 0 || takeoff_known) && h + 1 < n_hops;
            let p0 = if h == 0 {
                kick.map(|k| (k.pos[0], k.pos[1]))
                    .or_else(|| ground_pos_at(ball, ta))
            } else {
                ground_pos_at(ball, ta)
            };
            let p1 = ground_pos_at(ball, tb);
            let vxy_t = match (p0, p1) {
                (Some(a), Some(b)) if ends_known => Some(hypot(b.0 - a.0, b.1 - a.1) / tf_t),
                _ => None,
            };
            // method R
            let guess = (
                b0.x,
                b0.y,
                (b1.x - b0.x) / dt,
                (b1.y - b0.y) / dt,
                0.5 * G * dt,
            );
            let fit = fit_hop(game, ball, ta, tb, guess);
            if verbose {
                let t_line = format!(
                    "T: tf {:.3} vz {:.2} vxy {:?}",
                    tf_t,
                    vz_t,
                    vxy_t.map(|v| (v * 100.0).round() / 100.0)
                );
                match &fit {
                    Some(f) => println!(
                        "  flight {:.2} hop {h} [{:.3},{:.3}] zmax {:.3} {t_line} | R: t0-ta {:+.3} tf {:.3} vxy {:.2} vz {:.2} apex {:.3} rms {:.1}mm n {} off {:?}",
                        t_start, ta - t_start, tb - t_start, zmax, f.t0 - ta, f.flight_time, f.vxy(), f.vz, f.apex, f.rms * 1e3, f.n,
                        f.cam_offset.map(|(x, y)| ((x * 1e3).round(), (y * 1e3).round()))
                    ),
                    None => println!("  flight {:.2} hop {h} [{:.3},{:.3}] zmax {:.3} {t_line} | R: no fit", t_start, ta - t_start, tb - t_start, zmax),
                }
            }
            hops.push(Hop {
                idx: h,
                t_takeoff: ta,
                t_land: tb,
                tf_t,
                vz_t,
                vxy_t,
                apex_t: vz_t * vz_t / (2.0 * G),
                tracker_apex: zmax,
                r: fit,
            });
        }
        // bounces between consecutive hops
        for w in hops.windows(2) {
            let (h0, h1) = (&w[0], &w[1]);
            if h1.idx != h0.idx + 1 {
                continue;
            }
            // T: hop0 fully timed (takeoff known for hop 0) and hop1 ending at a real bounce
            let real = |h: &Hop| {
                h.tracker_apex >= MIN_TRACKER_APEX
                    && (0.5..2.0).contains(&(h.apex_t / h.tracker_apex))
            };
            let t_ok = (h0.idx > 0 || takeoff_known)
                && h1.idx + 1 < n_hops
                && h0.tf_t >= MIN_HOP_T
                && h1.tf_t >= MIN_HOP_T
                && real(h0)
                && real(h1);
            let (damping_z_t, damping_xy_t) = if t_ok {
                (
                    h1.tf_t / h0.tf_t,
                    match (h0.vxy_t, h1.vxy_t) {
                        (Some(a), Some(b)) if a > 0.3 => Some(b / a),
                        _ => None,
                    },
                )
            } else {
                (f64::NAN, None)
            };
            let (mut dxy_r, mut dz_r, mut defl_r) = (None, None, None);
            if let (Some(f0), Some(f1)) = (&h0.r, &h1.r) {
                let (lx, ly, lt) = f0.landing();
                let pos_gap = hypot(f1.x0 - lx, f1.y0 - ly);
                let t_gap = f1.t0 - lt;
                if pos_gap <= MAX_BOUNCE_POS_GAP
                    && t_gap.abs() <= MAX_BOUNCE_T_GAP
                    && f0.vxy() > 0.3
                {
                    dxy_r = Some(f1.vxy() / f0.vxy());
                    dz_r = Some(f1.vz / f0.vz);
                    let dot = (f0.vx * f1.vx + f0.vy * f1.vy) / (f0.vxy() * f1.vxy()).max(1e-9);
                    defl_r = Some(dot.clamp(-1.0, 1.0).acos().to_degrees());
                }
            }
            let (mut tdxy, mut tdz) = (None, None);
            if let Some(&bi) = bidx.get(h0.idx) {
                if bi >= 3 && bi + 3 < tr.len() {
                    if let (Some(a), Some(b)) = (
                        tr[bi - 3].ball.and_then(|b| b.vel),
                        tr[bi + 3].ball.and_then(|b| b.vel),
                    ) {
                        let va = hypot(a[0], a[1]);
                        if va > 0.3 && a[2] < -0.3 {
                            tdxy = Some(hypot(b[0], b[1]) / va);
                            tdz = Some(b[2] / -a[2]);
                        }
                    }
                }
            }
            if t_ok || dxy_r.is_some() {
                bounces.push(Bounce {
                    game: game.name.clone(),
                    competition: game.competition.clone(),
                    idx: h0.idx + 1,
                    vz_in_t: h0.vz_t,
                    damping_z_t,
                    damping_xy_t,
                    damping_xy_r: dxy_r,
                    damping_z_r: dz_r,
                    deflection_deg_r: defl_r,
                    tracker_damping_xy: tdxy,
                    tracker_damping_z: tdz,
                });
            }
        }
        let first = hops.first().filter(|h| h.idx == 0);
        let first_full = first.filter(|_| takeoff_known && n_hops > 1);
        let last_apex_tracker = bidx
            .last()
            .map(|&bi| {
                tr[bi..=e]
                    .iter()
                    .filter_map(|f| f.ball.map(|b| b.z))
                    .fold(0.0, f64::max)
            })
            .unwrap_or(f64::NAN);
        flights.push(Flight {
            game: game.name.clone(),
            competition: game.competition.clone(),
            t_start,
            n_hops_tracker: bidx.len() + 1,
            launch_angle_deg_t: first_full
                .and_then(|h| h.vxy_t.map(|v| h.vz_t.atan2(v).to_degrees())),
            launch_speed_t: first_full.and_then(|h| h.vxy_t.map(|v| hypot(v, h.vz_t))),
            launch_angle_deg_r: first_full
                .and_then(|h| h.r.as_ref().map(|f| f.vz.atan2(f.vxy()).to_degrees())),
            launch_speed_r: first_full.and_then(|h| h.r.as_ref().map(|f| hypot(f.vxy(), f.vz))),
            kicker: kick.and_then(|k| k.robot),
            kicker_team: kick.and_then(|k| k.robot).map(|(t, _)| match t {
                super::load::Team::Yellow => game.team_names.0.clone(),
                super::load::Team::Blue => game.team_names.1.clone(),
            }),
            tracker_kick_vz: kick.map(|k| k.vel[2]),
            tracker_kick_vxy: kick.map(|k| hypot(k.vel[0], k.vel[1])),
            last_apex_tracker,
            hops,
        });
    }
    (flights, bounces)
}

/// Aggregated chip results.
#[derive(Debug, Clone, Serialize)]
pub struct ChipResult {
    pub thresholds: serde_json::Value,
    pub n_flights: usize,
    pub n_bounces: usize,
    /// Method T.
    pub t_damping_z_first: Summary,
    pub t_damping_z_other: Summary,
    pub t_damping_xy_first: Summary,
    pub t_damping_xy_other: Summary,
    pub t_damping_z_all: Summary,
    pub t_damping_xy_all: Summary,
    /// Method R.
    pub r_damping_z_first: Summary,
    pub r_damping_z_other: Summary,
    pub r_damping_xy_first: Summary,
    pub r_damping_xy_other: Summary,
    pub r_deflection_deg: Summary,
    pub per_competition_t: Vec<(String, Summary, Summary, Summary, Summary)>,
    pub tracker_damping_xy_first: Summary,
    pub tracker_damping_z_first: Summary,
    /// Method T damping vs incoming vz bins: (vz_lo, median dz, median dxy, n).
    pub damping_vs_vz_t: Vec<(f64, f64, f64, usize)>,
    pub launch_angle_deg_t: Summary,
    pub launch_speed_t: Summary,
    /// Per kicking team: (team, launch angle T, launch speed T).
    pub launch_per_team: Vec<(String, Summary, Summary)>,
    pub launch_angle_deg_r: Summary,
    pub launch_angle_tracker_deg: Summary,
    pub hops_per_flight: Summary,
    pub last_apex_tracker: Summary,
    pub hop_rms_mm: Summary,
    pub r_apex_over_t_apex: Summary,
    pub t_apex_over_tracker_apex: Summary,
    pub hist_launch_angle_t: Histogram,
    pub hist_damping_xy_t: Histogram,
    pub hist_damping_z_t: Histogram,
    pub flights: Vec<Flight>,
    pub bounces: Vec<Bounce>,
    pub text: String,
}

pub fn aggregate(flights: Vec<Flight>, bounces: Vec<Bounce>) -> ChipResult {
    let first: Vec<&Bounce> = bounces.iter().filter(|b| b.idx == 1).collect();
    let other: Vec<&Bounce> = bounces.iter().filter(|b| b.idx >= 2).collect();
    let all: Vec<&Bounce> = bounces.iter().collect();
    let tz = |v: &[&Bounce]| Summary::of(&v.iter().map(|b| b.damping_z_t).collect::<Vec<_>>());
    let txy =
        |v: &[&Bounce]| Summary::of(&v.iter().filter_map(|b| b.damping_xy_t).collect::<Vec<_>>());
    let rz =
        |v: &[&Bounce]| Summary::of(&v.iter().filter_map(|b| b.damping_z_r).collect::<Vec<_>>());
    let rxy =
        |v: &[&Bounce]| Summary::of(&v.iter().filter_map(|b| b.damping_xy_r).collect::<Vec<_>>());
    let mut comps: Vec<String> = bounces.iter().map(|b| b.competition.clone()).collect();
    comps.sort();
    comps.dedup();
    let per_competition_t = comps
        .iter()
        .map(|c| {
            let f: Vec<&Bounce> = first
                .iter()
                .copied()
                .filter(|b| &b.competition == c)
                .collect();
            let o: Vec<&Bounce> = other
                .iter()
                .copied()
                .filter(|b| &b.competition == c)
                .collect();
            (c.clone(), txy(&f), txy(&o), tz(&f), tz(&o))
        })
        .collect::<Vec<_>>();
    let mut damping_vs_vz_t = Vec::new();
    for b in 0..8 {
        let lo = 0.5 + b as f64 * 0.5;
        let sel: Vec<&Bounce> = all
            .iter()
            .copied()
            .filter(|x| x.vz_in_t >= lo && x.vz_in_t < lo + 0.5 && x.damping_z_t.is_finite())
            .collect();
        if sel.len() >= 4 {
            damping_vs_vz_t.push((lo, tz(&sel).median, txy(&sel).median, sel.len()));
        }
    }
    let hops: Vec<&Hop> = flights.iter().flat_map(|f| f.hops.iter()).collect();
    let launch_angle_deg_t = Summary::of(
        &flights
            .iter()
            .filter_map(|f| f.launch_angle_deg_t)
            .collect::<Vec<_>>(),
    );
    let launch_speed_t = Summary::of(
        &flights
            .iter()
            .filter_map(|f| f.launch_speed_t)
            .collect::<Vec<_>>(),
    );
    let mut teams: Vec<String> = flights
        .iter()
        .filter_map(|f| f.kicker_team.clone())
        .collect();
    teams.sort();
    teams.dedup();
    let launch_per_team: Vec<(String, Summary, Summary)> = teams
        .iter()
        .map(|t| {
            let sel: Vec<&Flight> = flights
                .iter()
                .filter(|f| f.kicker_team.as_deref() == Some(t))
                .collect();
            (
                t.clone(),
                Summary::of(
                    &sel.iter()
                        .filter_map(|f| f.launch_angle_deg_t)
                        .collect::<Vec<_>>(),
                ),
                Summary::of(
                    &sel.iter()
                        .filter_map(|f| f.launch_speed_t)
                        .collect::<Vec<_>>(),
                ),
            )
        })
        .collect();
    let launch_angle_deg_r = Summary::of(
        &flights
            .iter()
            .filter_map(|f| f.launch_angle_deg_r)
            .collect::<Vec<_>>(),
    );
    let launch_angle_tracker_deg = Summary::of(
        &flights
            .iter()
            .filter(|f| f.launch_angle_deg_t.is_some())
            .filter_map(|f| Some(f.tracker_kick_vz?.atan2(f.tracker_kick_vxy?).to_degrees()))
            .collect::<Vec<_>>(),
    );
    let hist_launch_angle_t = Histogram::new(
        &flights
            .iter()
            .filter_map(|f| f.launch_angle_deg_t)
            .collect::<Vec<_>>(),
        20.0,
        70.0,
        10,
    );
    let hist_damping_xy_t = Histogram::new(
        &bounces
            .iter()
            .filter_map(|b| b.damping_xy_t)
            .collect::<Vec<_>>(),
        0.3,
        1.1,
        16,
    );
    let hist_damping_z_t = Histogram::new(
        &bounces.iter().map(|b| b.damping_z_t).collect::<Vec<_>>(),
        0.2,
        0.9,
        14,
    );
    let tracker_damping_xy_first = Summary::of(
        &first
            .iter()
            .filter_map(|b| b.tracker_damping_xy)
            .collect::<Vec<_>>(),
    );
    let tracker_damping_z_first = Summary::of(
        &first
            .iter()
            .filter_map(|b| b.tracker_damping_z)
            .collect::<Vec<_>>(),
    );
    let r_deflection_deg = Summary::of(
        &bounces
            .iter()
            .filter_map(|b| b.deflection_deg_r)
            .collect::<Vec<_>>(),
    );
    let hops_per_flight = Summary::of(
        &flights
            .iter()
            .map(|f| f.n_hops_tracker as f64)
            .collect::<Vec<_>>(),
    );
    let last_apex_tracker = Summary::of(
        &flights
            .iter()
            .map(|f| f.last_apex_tracker)
            .collect::<Vec<_>>(),
    );
    let hop_rms_mm = Summary::of(
        &hops
            .iter()
            .filter_map(|h| h.r.as_ref().map(|r| r.rms * 1e3))
            .collect::<Vec<_>>(),
    );
    let r_apex_over_t_apex = Summary::of(
        &hops
            .iter()
            .filter(|h| h.idx > 0)
            .filter_map(|h| h.r.as_ref().map(|r| r.apex / h.apex_t))
            .collect::<Vec<_>>(),
    );
    let t_apex_over_tracker_apex = Summary::of(
        &hops
            .iter()
            .filter(|h| h.idx > 0 && h.tracker_apex > 0.05)
            .map(|h| h.apex_t / h.tracker_apex)
            .collect::<Vec<_>>(),
    );
    let mut text = String::new();
    text += &format!(
        "flights {}, bounces {} (first {}, later {})\n",
        flights.len(),
        bounces.len(),
        first.len(),
        other.len()
    );
    text += &format!(
        "T damping_z first: {}\nT damping_z other: {}\n",
        tz(&first).line(""),
        tz(&other).line("")
    );
    text += &format!(
        "T damping_xy first: {}\nT damping_xy other: {}\n",
        txy(&first).line(""),
        txy(&other).line("")
    );
    text += &format!(
        "T damping_z all: {}\nT damping_xy all: {}\n",
        tz(&all).line(""),
        txy(&all).line("")
    );
    text += &format!(
        "R damping_z first: {}\nR damping_z other: {}\n",
        rz(&first).line(""),
        rz(&other).line("")
    );
    text += &format!(
        "R damping_xy first: {}\nR damping_xy other: {}\n",
        rxy(&first).line(""),
        rxy(&other).line("")
    );
    text += &format!("R deflection at bounce: {}\n", r_deflection_deg.line("deg"));
    for (c, fx, ox, fz, oz) in &per_competition_t {
        text += &format!(
            "  {c} (T): xy first {:.3}±{:.3} (n{}) other {:.3}±{:.3} (n{}); z first {:.3}±{:.3} (n{}) other {:.3}±{:.3} (n{})\n",
            fx.median, fx.se, fx.n, ox.median, ox.se, ox.n, fz.median, fz.se, fz.n, oz.median, oz.se, oz.n
        );
    }
    text += &format!(
        "tracker-implied first bounce: xy {} z {}\n",
        tracker_damping_xy_first.line(""),
        tracker_damping_z_first.line("")
    );
    text += "T damping vs incoming vz (vz_lo, dz, dxy, n):\n";
    for (lo, dz, dxy, n) in &damping_vs_vz_t {
        text += &format!(
            "  {:.1}..{:.1}: dz {:.3} dxy {:.3} (n {})\n",
            lo,
            lo + 0.5,
            dz,
            dxy,
            n
        );
    }
    text += &format!(
        "launch angle T: {}\nlaunch speed T: {}\n",
        launch_angle_deg_t.line("deg"),
        launch_speed_t.line("m/s")
    );
    text += &format!(
        "launch angle R: {}\nlaunch angle tracker kick: {}\n",
        launch_angle_deg_r.line("deg"),
        launch_angle_tracker_deg.line("deg")
    );
    for (t, a, s) in &launch_per_team {
        text += &format!(
            "  {t}: launch angle {:.1} ± {:.1} deg (σ {:.1}, n {}), speed {:.2} (p95 {:.2})\n",
            a.median, a.se, a.sigma, a.n, s.median, s.p95
        );
    }
    text += &format!(
        "hops per flight (tracker): {}\nlast hop apex (tracker): {}\n",
        hops_per_flight.line(""),
        last_apex_tracker.line("m")
    );
    text += &format!(
        "R hop fit rms: {}\nR apex / T apex: {}\nT apex / tracker apex: {}\n",
        hop_rms_mm.line("mm"),
        r_apex_over_t_apex.line(""),
        t_apex_over_tracker_apex.line("")
    );
    text += &hist_launch_angle_t.render("launch angle histogram (T) [deg]", 40);
    text += &hist_damping_xy_t.render("damping_xy histogram (T, all bounces)", 40);
    text += &hist_damping_z_t.render("damping_z histogram (T, all bounces)", 40);
    ChipResult {
        thresholds: serde_json::json!({
            "air_z_m": AIR_Z, "flight_gap_s": FLIGHT_GAP, "min_hop_samples": MIN_HOP_SAMPLES,
            "max_hop_rms_m": MAX_HOP_RMS, "min_hop_t_s": MIN_HOP_T, "t_prior_sigma_s": T_PRIOR_SIGMA,
            "outlier_sigma": OUTLIER_SIGMA, "bounce_pos_window_s": BOUNCE_POS_WINDOW,
            "max_bounce_pos_gap_m": MAX_BOUNCE_POS_GAP, "max_bounce_t_gap_s": MAX_BOUNCE_T_GAP,
            "min_tracker_apex_m": MIN_TRACKER_APEX,
        }),
        n_flights: flights.len(),
        n_bounces: bounces.len(),
        t_damping_z_first: tz(&first),
        t_damping_z_other: tz(&other),
        t_damping_xy_first: txy(&first),
        t_damping_xy_other: txy(&other),
        t_damping_z_all: tz(&all),
        t_damping_xy_all: txy(&all),
        r_damping_z_first: rz(&first),
        r_damping_z_other: rz(&other),
        r_damping_xy_first: rxy(&first),
        r_damping_xy_other: rxy(&other),
        r_deflection_deg,
        per_competition_t,
        tracker_damping_xy_first,
        tracker_damping_z_first,
        damping_vs_vz_t,
        launch_angle_deg_t,
        launch_speed_t,
        launch_per_team,
        launch_angle_deg_r,
        launch_angle_tracker_deg,
        hops_per_flight,
        last_apex_tracker,
        hop_rms_mm,
        r_apex_over_t_apex,
        t_apex_over_tracker_apex,
        hist_launch_angle_t,
        hist_damping_xy_t,
        hist_damping_z_t,
        flights,
        bounces,
        text,
    }
}

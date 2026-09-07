//! Robot kinematics from raw detections: speed / acceleration / angular
//! percentiles per team, acceleration vs speed (limiter or traction
//! saturation?), and step responses from rest.

use std::collections::BTreeMap;

use serde::Serialize;
use ssl_sim_proto::gc::referee::Command;

use super::load::{Game, Team};
use super::stats::{self, Histogram, Summary};
use super::track::{RobotSample, RobotTracks};

/// Samples slower than this count as "at rest" [m/s].
const REST_SPEED: f64 = 0.12;
/// Rest must last this long before a step [s].
const REST_DURATION: f64 = 0.25;
/// Step window after the rest [s] and minimum plateau speed [m/s].
const STEP_WINDOW: f64 = 1.5;
const STEP_MIN_PEAK: f64 = 1.2;
/// Tangential acceleration only evaluated above this speed [m/s].
const TANGENT_MIN_SPEED: f64 = 0.3;
/// Samples above these are detection glitches, not motion (counted, then dropped).
const GLITCH_SPEED: f64 = 5.0;
const GLITCH_ACCEL: f64 = 25.0;
const GLITCH_ALPHA: f64 = 400.0;

/// One step response from rest.
#[derive(Debug, Clone, Serialize)]
pub struct Step {
    pub team: String,
    pub t: f64,
    pub v_peak: f64,
    pub t10_90: f64,
    pub mean_accel: f64,
    pub peak_accel: f64,
    /// Time from 0.1 m/s to 2 m/s if reached.
    pub t_to_2: Option<f64>,
}

/// Per-team statistics.
#[derive(Debug, Clone, Serialize)]
pub struct TeamStats {
    pub team: String,
    pub n_samples: usize,
    pub n_robots: usize,
    pub speed_p50: f64,
    pub speed_p95: f64,
    pub speed_p99: f64,
    pub speed_max: f64,
    pub tracker_speed_p99: f64,
    pub accel_up_p95: f64,
    pub accel_up_p99: f64,
    pub accel_brake_p95: f64,
    pub accel_brake_p99: f64,
    pub accel_abs_p95: f64,
    pub accel_abs_p99: f64,
    pub omega_p95: f64,
    pub omega_p99: f64,
    pub omega_max: f64,
    pub alpha_p95: f64,
    pub alpha_p99: f64,
    /// Speed bins (lo, p90 speed-up accel, p90 brake accel, n).
    pub accel_vs_speed: Vec<(f64, f64, f64, usize)>,
    pub steps: Summary,
    pub step_t10_90: Summary,
    pub step_mean_accel: Summary,
    pub step_peak_accel: Summary,
    pub step_t_to_2: Summary,
    /// Steps reaching >= 2.5 m/s: t10-90 and mean accel.
    pub fast_step_t10_90: Summary,
    pub fast_step_mean_accel: Summary,
    pub n_steps: usize,
    /// Noise floor of the SG acceleration / angular acceleration on standing robots.
    pub accel_noise_sigma: f64,
    pub alpha_noise_sigma: f64,
    /// Samples dropped as glitches (speed > 5 m/s, |a| > 25, |alpha| > 400).
    pub glitches: usize,
}

#[derive(Default)]
struct Pool {
    speed: Vec<f64>,
    trk_speed: Vec<f64>,
    a_up: Vec<f64>,
    a_brake: Vec<f64>,
    a_abs: Vec<f64>,
    omega: Vec<f64>,
    alpha: Vec<f64>,
    // (speed, a_t)
    a_vs_v: Vec<(f64, f64)>,
    noise_a: Vec<f64>,
    noise_al: Vec<f64>,
    steps: Vec<Step>,
    robots: std::collections::BTreeSet<(String, u32)>,
    glitches: usize,
}

/// Accumulator across games keyed by "team (competition)".
#[derive(Default)]
pub struct RobotAccum {
    pools: BTreeMap<String, Pool>,
}

fn team_key(game: &Game, team: Team) -> String {
    let name = match team {
        Team::Yellow => &game.team_names.0,
        Team::Blue => &game.team_names.1,
    };
    let name = if name.is_empty() {
        team.name().to_string()
    } else {
        name.clone()
    };
    format!("{} ({})", name, game.competition)
}

impl RobotAccum {
    pub fn add_game(&mut self, game: &Game, robots: &RobotTracks) {
        for ((team, id), track) in robots {
            let key = team_key(game, *team);
            let pool = self.pools.entry(key.clone()).or_default();
            pool.robots.insert((game.name.clone(), *id));
            for s in track {
                if !s.vx.is_finite() {
                    continue;
                }
                let cmd = game.command_at(s.t);
                if cmd.is_none_or(|c| c == Command::Halt as i32) {
                    continue;
                }
                let sp = s.speed();
                if sp > GLITCH_SPEED || s.accel() > GLITCH_ACCEL || s.alpha.abs() > GLITCH_ALPHA {
                    pool.glitches += 1;
                    continue;
                }
                pool.speed.push(sp);
                pool.a_abs.push(s.accel());
                pool.omega.push(s.omega.abs());
                pool.alpha.push(s.alpha.abs());
                if sp > TANGENT_MIN_SPEED {
                    let at = s.accel_tangential();
                    pool.a_vs_v.push((sp, at));
                    if at > 0.0 {
                        pool.a_up.push(at);
                    } else {
                        pool.a_brake.push(-at);
                    }
                }
                if sp < 0.03 {
                    pool.noise_a.push(s.ax);
                    pool.noise_al.push(s.alpha);
                }
            }
            for st in steps(track, &key) {
                pool.steps.push(st);
            }
        }
        // tracker speeds
        for f in &game.tracker {
            let cmd = game.command_at(f.t);
            if cmd.is_none_or(|c| c == Command::Halt as i32) {
                continue;
            }
            for r in &f.robots {
                if let Some(v) = r.vel {
                    let key = team_key(game, r.team);
                    self.pools
                        .entry(key)
                        .or_default()
                        .trk_speed
                        .push(stats::hypot(v[0], v[1]));
                }
            }
        }
    }
}

/// Step responses from rest in one track.
fn steps(track: &[RobotSample], team: &str) -> Vec<Step> {
    let mut out = Vec::new();
    let n = track.len();
    let mut i = 0;
    while i < n {
        let s = &track[i];
        if !s.vx.is_finite() || s.speed() >= REST_SPEED {
            i += 1;
            continue;
        }
        // rest run
        let start = i;
        while i < n
            && track[i].vx.is_finite()
            && track[i].speed() < REST_SPEED
            && track[i].t - track[start].t < 10.0
        {
            i += 1;
        }
        if i >= n || track[i - 1].t - track[start].t < REST_DURATION {
            continue;
        }
        let t0 = track[i - 1].t;
        // window
        let mut j = i;
        let mut v_peak = 0.0f64;
        let mut peak_accel = 0.0f64;
        let mut t10 = None;
        let mut t90 = None;
        let mut t_01 = None;
        let mut t_2 = None;
        let mut gap = false;
        // first pass: peak within the window
        let mut k = i;
        while k < n && track[k].t - t0 < STEP_WINDOW {
            if track[k].vx.is_finite() {
                v_peak = v_peak.max(track[k].speed());
            }
            if k > 0 && track[k].t - track[k - 1].t > 0.1 {
                gap = true;
                break;
            }
            k += 1;
        }
        if gap || v_peak < STEP_MIN_PEAK {
            continue;
        }
        while j < n && track[j].t - t0 < STEP_WINDOW {
            let s = &track[j];
            if s.vx.is_finite() {
                let sp = s.speed();
                if t_01.is_none() && sp > 0.1 {
                    t_01 = Some(s.t);
                }
                if t10.is_none() && sp > 0.1 * v_peak {
                    t10 = Some(s.t);
                }
                if t10.is_some() && t90.is_none() && sp > 0.9 * v_peak {
                    t90 = Some(s.t);
                }
                if t_2.is_none() && sp > 2.0 {
                    t_2 = Some(s.t);
                }
                if t90.is_none() && sp > TANGENT_MIN_SPEED {
                    peak_accel = peak_accel.max(s.accel_tangential());
                }
            }
            if t90.is_some() {
                break;
            }
            j += 1;
        }
        if let (Some(a), Some(b)) = (t10, t90) {
            let rise = b - a;
            if rise > 0.05 && peak_accel < 15.0 && v_peak < GLITCH_SPEED {
                out.push(Step {
                    team: team.to_string(),
                    t: t0,
                    v_peak,
                    t10_90: rise,
                    mean_accel: 0.8 * v_peak / rise,
                    peak_accel,
                    t_to_2: match (t_01, t_2) {
                        (Some(x), Some(y)) if y > x => Some(y - x),
                        _ => None,
                    },
                });
            }
        }
        i = j.max(i + 1);
    }
    out
}

#[derive(Debug, Clone, Serialize)]
pub struct RobotResult {
    pub thresholds: serde_json::Value,
    pub teams: Vec<TeamStats>,
    pub hist_speed: Histogram,
    pub hist_accel_up: Histogram,
    pub text: String,
}

pub fn aggregate(acc: RobotAccum) -> RobotResult {
    let mut teams = Vec::new();
    let mut all_speed = Vec::new();
    let mut all_up = Vec::new();
    let mut text = String::new();
    for (key, p) in acc.pools {
        if p.speed.len() < 1000 {
            continue;
        }
        let q = |v: &[f64], x: f64| stats::quantile(v, x);
        let mut accel_vs_speed = Vec::new();
        for b in 0..9 {
            let lo = b as f64 * 0.5;
            let up: Vec<f64> = p
                .a_vs_v
                .iter()
                .filter(|(v, a)| *v >= lo && *v < lo + 0.5 && *a > 0.0)
                .map(|(_, a)| *a)
                .collect();
            let br: Vec<f64> = p
                .a_vs_v
                .iter()
                .filter(|(v, a)| *v >= lo && *v < lo + 0.5 && *a < 0.0)
                .map(|(_, a)| -*a)
                .collect();
            if up.len() + br.len() >= 200 {
                accel_vs_speed.push((lo, q(&up, 0.9), q(&br, 0.9), up.len() + br.len()));
            }
        }
        let ts = TeamStats {
            team: key.clone(),
            n_samples: p.speed.len(),
            n_robots: p.robots.len(),
            speed_p50: q(&p.speed, 0.5),
            speed_p95: q(&p.speed, 0.95),
            speed_p99: q(&p.speed, 0.99),
            speed_max: q(&p.speed, 0.9999),
            tracker_speed_p99: q(&p.trk_speed, 0.99),
            accel_up_p95: q(&p.a_up, 0.95),
            accel_up_p99: q(&p.a_up, 0.99),
            accel_brake_p95: q(&p.a_brake, 0.95),
            accel_brake_p99: q(&p.a_brake, 0.99),
            accel_abs_p95: q(&p.a_abs, 0.95),
            accel_abs_p99: q(&p.a_abs, 0.99),
            omega_p95: q(&p.omega, 0.95),
            omega_p99: q(&p.omega, 0.99),
            omega_max: q(&p.omega, 0.9999),
            alpha_p95: q(&p.alpha, 0.95),
            alpha_p99: q(&p.alpha, 0.99),
            accel_vs_speed,
            steps: Summary::of(&p.steps.iter().map(|s| s.v_peak).collect::<Vec<_>>()),
            step_t10_90: Summary::of(&p.steps.iter().map(|s| s.t10_90).collect::<Vec<_>>()),
            step_mean_accel: Summary::of(&p.steps.iter().map(|s| s.mean_accel).collect::<Vec<_>>()),
            step_peak_accel: Summary::of(&p.steps.iter().map(|s| s.peak_accel).collect::<Vec<_>>()),
            step_t_to_2: Summary::of(&p.steps.iter().filter_map(|s| s.t_to_2).collect::<Vec<_>>()),
            fast_step_t10_90: Summary::of(
                &p.steps
                    .iter()
                    .filter(|s| s.v_peak >= 2.5)
                    .map(|s| s.t10_90)
                    .collect::<Vec<_>>(),
            ),
            fast_step_mean_accel: Summary::of(
                &p.steps
                    .iter()
                    .filter(|s| s.v_peak >= 2.5)
                    .map(|s| s.mean_accel)
                    .collect::<Vec<_>>(),
            ),
            n_steps: p.steps.len(),
            accel_noise_sigma: stats::mad_sigma(&p.noise_a),
            alpha_noise_sigma: stats::mad_sigma(&p.noise_al),
            glitches: p.glitches,
        };
        text += &format!(
            "{}: n {} robots {} (glitches dropped {})\n  speed p50/p95/p99/max {:.2}/{:.2}/{:.2}/{:.2} (tracker p99 {:.2})\n  accel up p95/p99 {:.2}/{:.2}  brake p95/p99 {:.2}/{:.2}  |a| p95/p99 {:.2}/{:.2} (noise σ {:.2})\n  omega p95/p99/max {:.2}/{:.2}/{:.2}  alpha p95/p99 {:.1}/{:.1} (noise σ {:.1})\n",
            ts.team, ts.n_samples, ts.n_robots, ts.glitches, ts.speed_p50, ts.speed_p95, ts.speed_p99, ts.speed_max, ts.tracker_speed_p99,
            ts.accel_up_p95, ts.accel_up_p99, ts.accel_brake_p95, ts.accel_brake_p99, ts.accel_abs_p95, ts.accel_abs_p99, ts.accel_noise_sigma,
            ts.omega_p95, ts.omega_p99, ts.omega_max, ts.alpha_p95, ts.alpha_p99, ts.alpha_noise_sigma
        );
        text += "  accel vs speed (v_lo: p90 up, p90 brake, n): ";
        for (lo, u, b, n) in &ts.accel_vs_speed {
            text += &format!("{:.1}: {:.2}/{:.2} ({}); ", lo, u, b, n);
        }
        text += &format!(
            "\n  steps {}: v_peak {:.2}, t10-90 {:.3} ± {:.3} s, mean accel {:.2} (p95 {:.2}), peak accel {:.2} (p95 {:.2}), t(0.1->2 m/s) {:.3} (n {})\n",
            ts.n_steps, ts.steps.median, ts.step_t10_90.median, ts.step_t10_90.se, ts.step_mean_accel.median, ts.step_mean_accel.p95,
            ts.step_peak_accel.median, ts.step_peak_accel.p95, ts.step_t_to_2.median, ts.step_t_to_2.n
        );
        text += &format!(
            "  fast steps (v_peak >= 2.5, n {}): t10-90 {:.3} (p05 {:.3}), mean accel {:.2} (p95 {:.2}); t(0.1->2) p05 {:.3}\n",
            ts.fast_step_t10_90.n, ts.fast_step_t10_90.median, ts.fast_step_t10_90.p05, ts.fast_step_mean_accel.median,
            ts.fast_step_mean_accel.p95, ts.step_t_to_2.p05
        );
        all_speed.extend(p.speed.iter().copied());
        all_up.extend(p.a_up.iter().copied());
        teams.push(ts);
    }
    let hist_speed = Histogram::new(&all_speed, 0.0, 4.5, 18);
    let hist_accel_up = Histogram::new(&all_up, 0.0, 10.0, 20);
    text += &hist_speed.render("robot speed histogram, all teams [m/s]", 40);
    text += &hist_accel_up.render("robot speed-up acceleration histogram [m/s^2]", 40);
    RobotResult {
        thresholds: serde_json::json!({
            "rest_speed": REST_SPEED, "rest_duration_s": REST_DURATION, "step_window_s": STEP_WINDOW,
            "step_min_peak": STEP_MIN_PEAK, "tangent_min_speed": TANGENT_MIN_SPEED,
            "glitch_speed": GLITCH_SPEED, "glitch_accel": GLITCH_ACCEL, "glitch_alpha": GLITCH_ALPHA,
            "robot_jump_pos_m": super::track::ROBOT_JUMP_POS, "robot_jump_angle_rad": super::track::ROBOT_JUMP_ANGLE,
            "sg_half_window_samples": super::track::ROBOT_SG_HALF, "gating": "referee command != HALT",
        }),
        teams,
        hist_speed,
        hist_accel_up,
        text,
    }
}

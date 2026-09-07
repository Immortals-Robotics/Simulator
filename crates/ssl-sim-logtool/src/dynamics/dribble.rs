//! Dribbling: episodes where the ball sits in front of a robot's kicker while
//! the robot moves. Uses same-frame same-camera ball/robot pairs so that the
//! ball–robot geometry is free of the per-camera calibration offset.

use std::collections::BTreeMap;

use serde::Serialize;

use super::kicks::TrackerKick;
use super::load::{Game, Team};
use super::stats::{self, hypot, Histogram, Summary};
use super::track::{nearest_idx, BallSample, RobotSample, RobotTracks};

/// Mouth zone in the robot frame: ball centre x range and |y| limit [m].
const ZONE_X: (f64, f64) = (0.05, 0.14);
const ZONE_Y: f64 = 0.05;
/// Max gap between in-zone samples inside one episode [s] (occlusion tolerance).
const MAX_GAP: f64 = 0.15;
/// Minimum episode duration [s] and samples.
const MIN_DURATION: f64 = 0.3;
const MIN_SAMPLES: usize = 10;
/// Robot must exceed this speed at some point for the ball to count as carried [m/s].
const CARRY_SPEED: f64 = 0.5;
/// Static hold: robot speed below this and |a| below 1 m/s^2.
const STATIC_SPEED: f64 = 0.3;
/// Loss: robot still faster than this at the end [m/s], no kick within the window, ball seen away.
const LOSS_ROBOT_SPEED: f64 = 0.4;
const LOSS_KICK_WINDOW: (f64, f64) = (-0.25, 0.10);
/// Ball must be seen at least this far outside the zone within 0.4 s after the end [m].
const LOSS_BALL_X: f64 = 0.17;
const LOSS_BALL_Y: f64 = 0.08;
/// Window before the end over which the loss acceleration is taken [s].
const LOSS_ACCEL_WINDOW: f64 = 0.08;

#[derive(Debug, Clone, Copy)]
struct HeldSample {
    t: f64,
    lx: f64,
    ly: f64,
    speed: f64,
    /// Acceleration along the heading (positive forward) and lateral, and magnitude.
    a_h: f64,
    a_l: f64,
    a_abs: f64,
    omega: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DribbleEpisode {
    pub game: String,
    pub team: Team,
    pub robot: u32,
    pub t0: f64,
    pub duration: f64,
    pub n: usize,
    pub max_speed: f64,
    pub max_accel: f64,
    pub max_omega: f64,
    pub median_lx: f64,
    pub median_ly: f64,
    pub end: String,
    /// Acceleration (heading, lateral, abs) and omega over the last window before the end.
    pub end_a_h: f64,
    pub end_a_l: f64,
    pub end_a_abs: f64,
    pub end_omega: f64,
    pub end_speed: f64,
}

fn robot_sample_at(track: &[RobotSample], t: f64, cam: u32) -> Option<&RobotSample> {
    let i = nearest_idx(track, t, |s| s.t)?;
    // prefer exact same-frame sample
    for s in &track[i.saturating_sub(2)..(i + 3).min(track.len())] {
        if s.cam == cam && (s.t - t).abs() < 1e-4 {
            return Some(s);
        }
    }
    let s = &track[i];
    ((s.t - t).abs() < 0.03).then_some(s)
}

/// Extract dribble episodes and held-sample pools of one game.
pub struct GameDribble {
    pub episodes: Vec<DribbleEpisode>,
    /// (lx, ly, speed, a_h, a_abs, omega) of every held sample while carried.
    pub held: Vec<(f64, f64, f64, f64, f64, f64)>,
    /// Team name per held sample (same order as `held`).
    pub held_team: Vec<String>,
}

pub fn analyse(
    game: &Game,
    ball: &[BallSample],
    robots: &RobotTracks,
    kicks: &[TrackerKick],
) -> GameDribble {
    let mut open: BTreeMap<(Team, u32), Vec<HeldSample>> = BTreeMap::new();
    let mut episodes = Vec::new();
    let mut held = Vec::new();
    let mut held_team = Vec::new();
    let team_name = |t: Team| match t {
        Team::Yellow => game.team_names.0.clone(),
        Team::Blue => game.team_names.1.clone(),
    };
    let close = |key: (Team, u32),
                 samples: Vec<HeldSample>,
                 episodes: &mut Vec<DribbleEpisode>,
                 held: &mut Vec<(f64, f64, f64, f64, f64, f64)>,
                 held_team: &mut Vec<String>| {
        if samples.len() < MIN_SAMPLES {
            return;
        }
        let t0 = samples[0].t;
        let t1 = samples[samples.len() - 1].t;
        if t1 - t0 < MIN_DURATION {
            return;
        }
        let max_speed = samples.iter().map(|s| s.speed).fold(0.0, f64::max);
        if max_speed < CARRY_SPEED {
            return;
        }
        for s in &samples {
            held.push((s.lx, s.ly, s.speed, s.a_h, s.a_abs, s.omega));
            held_team.push(team_name(key.0));
        }
        let tail: Vec<&HeldSample> = samples
            .iter()
            .filter(|s| s.t > t1 - LOSS_ACCEL_WINDOW)
            .collect();
        let med = |f: &dyn Fn(&HeldSample) -> f64| {
            stats::median(&tail.iter().map(|s| f(s)).collect::<Vec<_>>())
        };
        let end_speed = med(&|s| s.speed);
        // classify the end
        let kicked = kicks
            .iter()
            .any(|k| k.start - t1 >= LOSS_KICK_WINDOW.0 && k.start - t1 <= LOSS_KICK_WINDOW.1);
        let mut end = "unknown".to_string();
        if kicked {
            end = "kick".to_string();
        } else {
            // ball seen clearly outside the zone within 0.4 s?
            let track = &robots[&key];
            let a = ball.partition_point(|s| s.t < t1);
            let b = ball.partition_point(|s| s.t < t1 + 0.4);
            let mut away = false;
            let mut seen = false;
            for s in &ball[a..b] {
                if let Some(r) = robot_sample_at(track, s.t, s.cam) {
                    seen = true;
                    let (c, sn) = (r.theta.cos(), r.theta.sin());
                    let dx = s.x - r.x;
                    let dy = s.y - r.y;
                    let lx = dx * c + dy * sn;
                    let ly = -dx * sn + dy * c;
                    if lx > LOSS_BALL_X || ly.abs() > LOSS_BALL_Y || lx < 0.0 {
                        away = true;
                        break;
                    }
                }
            }
            if away && end_speed > LOSS_ROBOT_SPEED {
                end = "loss".to_string();
            } else if away {
                end = "release".to_string();
            } else if !seen {
                end = "occluded".to_string();
            }
        }
        episodes.push(DribbleEpisode {
            game: game.name.clone(),
            team: key.0,
            robot: key.1,
            t0,
            duration: t1 - t0,
            n: samples.len(),
            max_speed,
            max_accel: samples.iter().map(|s| s.a_abs).fold(0.0, f64::max),
            max_omega: samples.iter().map(|s| s.omega.abs()).fold(0.0, f64::max),
            median_lx: stats::median(&samples.iter().map(|s| s.lx).collect::<Vec<_>>()),
            median_ly: stats::median(&samples.iter().map(|s| s.ly).collect::<Vec<_>>()),
            end,
            end_a_h: med(&|s| s.a_h),
            end_a_l: med(&|s| s.a_l),
            end_a_abs: med(&|s| s.a_abs),
            end_omega: med(&|s| s.omega.abs()),
            end_speed,
        });
    };
    for s in ball {
        // close stale episodes
        let stale: Vec<(Team, u32)> = open
            .iter()
            .filter(|(_, v)| s.t - v.last().unwrap().t > MAX_GAP)
            .map(|(k, _)| *k)
            .collect();
        for k in stale {
            let v = open.remove(&k).unwrap();
            close(k, v, &mut episodes, &mut held, &mut held_team);
        }
        for (key, track) in robots {
            let Some(r) = robot_sample_at(track, s.t, s.cam) else {
                continue;
            };
            if r.cam != s.cam || (r.t - s.t).abs() > 1e-4 {
                continue;
            }
            let (c, sn) = (r.theta.cos(), r.theta.sin());
            let dx = s.x - r.x;
            let dy = s.y - r.y;
            let lx = dx * c + dy * sn;
            let ly = -dx * sn + dy * c;
            if lx < ZONE_X.0 || lx > ZONE_X.1 || ly.abs() > ZONE_Y {
                continue;
            }
            if !r.vx.is_finite() {
                continue;
            }
            let a_h = r.ax * c + r.ay * sn;
            let a_l = -r.ax * sn + r.ay * c;
            open.entry(*key).or_default().push(HeldSample {
                t: s.t,
                lx,
                ly,
                speed: r.speed(),
                a_h,
                a_l,
                a_abs: hypot(r.ax, r.ay),
                omega: r.omega,
            });
        }
    }
    for (k, v) in open {
        close(k, v, &mut episodes, &mut held, &mut held_team);
    }
    GameDribble {
        episodes,
        held,
        held_team,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DribbleResult {
    pub thresholds: serde_json::Value,
    pub n_episodes: usize,
    pub n_held_samples: usize,
    pub ends: Vec<(String, usize)>,
    /// Ball centre distance along the heading while held: static vs moving.
    pub lx_static: Summary,
    pub lx_moving: Summary,
    pub radial_static: Summary,
    pub ly_all: Summary,
    /// Held lx per team (moving robots).
    pub lx_per_team: Vec<(String, Summary)>,
    /// lx vs heading acceleration: regression slope [m per m/s^2] and intercept.
    pub lx_vs_accel: Option<(f64, f64, f64, usize)>,
    /// lx in heading-acceleration bins: (a_lo, median lx, n).
    pub lx_by_accel: Vec<(f64, f64, usize)>,
    pub held_accel_p95: f64,
    pub held_accel_p99: f64,
    pub held_accel_max: f64,
    pub held_omega_p95: f64,
    pub held_omega_p99: f64,
    pub held_speed_p95: f64,
    pub episode_max_accel: Summary,
    pub loss_accel_abs: Summary,
    pub loss_accel_heading: Summary,
    pub loss_accel_lateral: Summary,
    pub loss_omega: Summary,
    pub loss_speed: Summary,
    /// Fraction of losses where the heading acceleration is negative (braking).
    pub loss_braking_fraction: f64,
    /// Survival table: (accel threshold, held samples above, losses at/above).
    pub survival: Vec<(f64, usize, usize)>,
    /// Same for braking only (heading acceleration <= -threshold): (threshold, held samples, losses).
    pub survival_braking: Vec<(f64, usize, usize)>,
    pub hist_lx: Histogram,
    pub hist_loss_accel: Histogram,
    pub episodes: Vec<DribbleEpisode>,
    pub text: String,
}

pub fn aggregate(
    episodes: Vec<DribbleEpisode>,
    held: Vec<(f64, f64, f64, f64, f64, f64)>,
    held_team: Vec<String>,
) -> DribbleResult {
    // per-team held distance (moving robots only)
    let mut teams: Vec<String> = held_team.clone();
    teams.sort();
    teams.dedup();
    let lx_per_team: Vec<(String, Summary)> = teams
        .iter()
        .map(|t| {
            let v: Vec<f64> = held
                .iter()
                .zip(&held_team)
                .filter(|(h, n)| *n == t && h.2 > 0.3)
                .map(|(h, _)| h.0)
                .collect();
            (t.clone(), Summary::of(&v))
        })
        .collect();
    let stat: Vec<&(f64, f64, f64, f64, f64, f64)> = held
        .iter()
        .filter(|h| h.2 < STATIC_SPEED && h.4 < 1.0)
        .collect();
    let moving: Vec<&(f64, f64, f64, f64, f64, f64)> = held.iter().filter(|h| h.2 > 0.3).collect();
    let lx_static = Summary::of(&stat.iter().map(|h| h.0).collect::<Vec<_>>());
    let lx_moving = Summary::of(&moving.iter().map(|h| h.0).collect::<Vec<_>>());
    let radial_static = Summary::of(&stat.iter().map(|h| hypot(h.0, h.1)).collect::<Vec<_>>());
    let ly_all = Summary::of(&held.iter().map(|h| h.1).collect::<Vec<_>>());
    let (ax, lx): (Vec<f64>, Vec<f64>) = held
        .iter()
        .filter(|h| h.3.abs() < 8.0)
        .map(|h| (h.3, h.0))
        .unzip();
    let lx_vs_accel = stats::linreg(&ax, &lx).map(|(a, b, se, _, n)| (a, b, se, n));
    let mut lx_by_accel = Vec::new();
    for b in -4..4 {
        let lo = b as f64 * 1.0;
        let v: Vec<f64> = held
            .iter()
            .filter(|h| h.3 >= lo && h.3 < lo + 1.0)
            .map(|h| h.0)
            .collect();
        if v.len() >= 30 {
            lx_by_accel.push((lo, stats::median(&v), v.len()));
        }
    }
    let acc: Vec<f64> = held.iter().map(|h| h.4).collect();
    let om: Vec<f64> = held.iter().map(|h| h.5.abs()).collect();
    let sp: Vec<f64> = held.iter().map(|h| h.2).collect();
    let losses: Vec<&DribbleEpisode> = episodes.iter().filter(|e| e.end == "loss").collect();
    let loss_accel_abs = Summary::of(&losses.iter().map(|e| e.end_a_abs).collect::<Vec<_>>());
    let loss_accel_heading = Summary::of(&losses.iter().map(|e| e.end_a_h).collect::<Vec<_>>());
    let loss_accel_lateral =
        Summary::of(&losses.iter().map(|e| e.end_a_l.abs()).collect::<Vec<_>>());
    let loss_omega = Summary::of(&losses.iter().map(|e| e.end_omega).collect::<Vec<_>>());
    let loss_speed = Summary::of(&losses.iter().map(|e| e.end_speed).collect::<Vec<_>>());
    let loss_braking_fraction = if losses.is_empty() {
        f64::NAN
    } else {
        losses.iter().filter(|e| e.end_a_h < 0.0).count() as f64 / losses.len() as f64
    };
    let mut survival = Vec::new();
    for th in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0] {
        survival.push((
            th,
            held.iter().filter(|h| h.4 >= th).count(),
            losses.iter().filter(|e| e.end_a_abs >= th).count(),
        ));
    }
    let mut survival_braking = Vec::new();
    for th in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0] {
        survival_braking.push((
            th,
            held.iter().filter(|h| h.3 <= -th).count(),
            losses.iter().filter(|e| e.end_a_h <= -th).count(),
        ));
    }
    let mut ends: BTreeMap<String, usize> = BTreeMap::new();
    for e in &episodes {
        *ends.entry(e.end.clone()).or_default() += 1;
    }
    let hist_lx = Histogram::new(
        &held.iter().map(|h| h.0).collect::<Vec<_>>(),
        0.05,
        0.14,
        18,
    );
    let hist_loss_accel = Histogram::new(
        &losses.iter().map(|e| e.end_a_abs).collect::<Vec<_>>(),
        0.0,
        10.0,
        20,
    );
    let episode_max_accel = Summary::of(&episodes.iter().map(|e| e.max_accel).collect::<Vec<_>>());
    let mut text = String::new();
    text += &format!(
        "dribble episodes {} held samples {} ends {:?}\n",
        episodes.len(),
        held.len(),
        ends
    );
    text += &format!(
        "lx static: {}\nlx moving: {}\nradial static: {}\nly: {}\n",
        lx_static.line("m"),
        lx_moving.line("m"),
        radial_static.line("m"),
        ly_all.line("m")
    );
    for (t, s) in &lx_per_team {
        text += &format!(
            "  lx moving {t}: {:.4} (p25 {:.4} p75 {:.4}, n {})\n",
            s.median, s.p25, s.p75, s.n
        );
    }
    if let Some((a, b, se, n)) = lx_vs_accel {
        text += &format!(
            "lx = {:.4} + ({:.5} ± {:.5}) * a_heading  (n {})\n",
            a, b, se, n
        );
    }
    text += "lx by heading accel: ";
    for (lo, m, n) in &lx_by_accel {
        text += &format!("[{:+.0},{:+.0}): {:.4} ({}); ", lo, lo + 1.0, m, n);
    }
    text += &format!(
        "\nheld |a| p95/p99/max {:.2}/{:.2}/{:.2}, held |omega| p95/p99 {:.2}/{:.2}, held speed p95 {:.2}\n",
        stats::quantile(&acc, 0.95),
        stats::quantile(&acc, 0.99),
        stats::quantile(&acc, 1.0),
        stats::quantile(&om, 0.95),
        stats::quantile(&om, 0.99),
        stats::quantile(&sp, 0.95)
    );
    text += &format!("episode max |a|: {}\n", episode_max_accel.line("m/s^2"));
    text += &format!(
        "losses {}: |a| {}\n  a_heading {}\n  |a_lateral| {}\n  omega {}\n  speed {}\n  braking fraction {:.2}\n",
        losses.len(),
        loss_accel_abs.line("m/s^2"),
        loss_accel_heading.line("m/s^2"),
        loss_accel_lateral.line("m/s^2"),
        loss_omega.line("rad/s"),
        loss_speed.line("m/s"),
        loss_braking_fraction
    );
    text += "survival (threshold: held samples above, losses above): ";
    for (th, h, l) in &survival {
        text += &format!("{th}: {h}/{l}; ");
    }
    text +=
        "\nsurvival braking (threshold: held samples with a_h <= -th, losses with a_h <= -th): ";
    for (th, h, l) in &survival_braking {
        text += &format!("{th}: {h}/{l}; ");
    }
    text += "\n";
    text += &hist_lx.render("held ball lx histogram [m]", 40);
    text += &hist_loss_accel.render("robot |a| at ball loss [m/s^2]", 40);
    DribbleResult {
        thresholds: serde_json::json!({
            "zone_x_m": ZONE_X, "zone_y_m": ZONE_Y, "max_gap_s": MAX_GAP, "min_duration_s": MIN_DURATION,
            "min_samples": MIN_SAMPLES, "carry_speed": CARRY_SPEED, "static_speed": STATIC_SPEED,
            "loss_robot_speed": LOSS_ROBOT_SPEED, "loss_kick_window_s": LOSS_KICK_WINDOW,
            "loss_ball_x_m": LOSS_BALL_X, "loss_ball_y_m": LOSS_BALL_Y, "loss_accel_window_s": LOSS_ACCEL_WINDOW,
        }),
        n_episodes: episodes.len(),
        n_held_samples: held.len(),
        ends: ends.into_iter().collect(),
        lx_static,
        lx_moving,
        radial_static,
        ly_all,
        lx_per_team,
        lx_vs_accel,
        lx_by_accel,
        held_accel_p95: stats::quantile(&acc, 0.95),
        held_accel_p99: stats::quantile(&acc, 0.99),
        held_accel_max: stats::quantile(&acc, 1.0),
        held_omega_p95: stats::quantile(&om, 0.95),
        held_omega_p99: stats::quantile(&om, 0.99),
        held_speed_p95: stats::quantile(&sp, 0.95),
        episode_max_accel,
        loss_accel_abs,
        loss_accel_heading,
        loss_accel_lateral,
        loss_omega,
        loss_speed,
        loss_braking_fraction,
        survival,
        survival_braking,
        hist_lx,
        hist_loss_accel,
        episodes,
        text,
    }
}

//! Kicks: initial speed, direction error, rise time and the incoming/outgoing
//! speed relation. Kick events come from the tracker (`kicked_ball`) and
//! from an independent speed-jump detector on the raw ball track.

use serde::Serialize;

use super::load::{Game, Team};
use super::stats::{self, hypot, wrap_angle, Histogram, Summary};
use super::track::{robot_pose_at, robot_velocity_at, BallSample, RobotTracks};

/// A distinct tracker kick event (times already on the raw clock).
#[derive(Debug, Clone, Copy)]
pub struct TrackerKick {
    pub start: f64,
    pub pos: [f64; 2],
    pub vel: [f64; 3],
    pub robot: Option<(Team, u32)>,
}

/// Window after the kick in which the peak speed is searched [s].
const PEAK_WINDOW: f64 = 0.35;
/// Window before the kick for the incoming velocity [s].
const INCOMING_WINDOW: (f64, f64) = (-0.30, -0.04);
/// Ball must be this far from the kick position before the direction fit starts [m].
const DIR_MIN_DIST: f64 = 0.10;
/// Direction fit window after the kick [s].
const DIR_WINDOW: f64 = 0.45;
/// Own detector: speed rise over 3 same-camera frames [m/s].
const DETECT_JUMP: f64 = 1.0;
/// Own detector: ball must be in front of a robot: local x range and |y| limit [m].
const DETECT_X: (f64, f64) = (0.03, 0.24);
const DETECT_Y: f64 = 0.09;
/// Distinct-kick separation [s].
const KICK_SEPARATION: f64 = 0.25;

/// Extract distinct tracker kicks (last refinement of each event wins).
pub fn tracker_kicks(game: &Game) -> Vec<TrackerKick> {
    let mut out: Vec<TrackerKick> = Vec::new();
    for f in &game.tracker {
        let Some(k) = f.kick else { continue };
        let tk = TrackerKick {
            start: k.start,
            pos: k.pos,
            vel: k.vel,
            robot: k.robot,
        };
        match out.last_mut() {
            Some(last) if (last.start - k.start).abs() < KICK_SEPARATION => *last = tk,
            _ => out.push(tk),
        }
    }
    out
}

/// One analysed kick.
#[derive(Debug, Clone, Serialize)]
pub struct Kick {
    pub game: String,
    pub t: f64,
    pub team: Option<Team>,
    pub robot: Option<u32>,
    /// Tracker initial speed (2D) and vertical component.
    pub v_tracker: f64,
    pub vz_tracker: f64,
    pub chip: bool,
    /// Raw peak speed after the kick and when it was reached.
    pub v_peak_raw: f64,
    pub t_peak: f64,
    /// Frames (of one camera) from onset (>20 % of peak) to >90 % of peak, and the time.
    pub rise_frames: Option<u32>,
    pub rise_ms: Option<f64>,
    /// Ball direction minus robot orientation [rad].
    pub angle_err: Option<f64>,
    /// Robot orientation at the kick [rad] and its angular rate.
    pub robot_theta: Option<f64>,
    pub robot_omega: Option<f64>,
    /// Robot speed along its heading [m/s].
    pub robot_v_heading: Option<f64>,
    /// Incoming ball velocity component toward the robot (positive = approaching) [m/s].
    pub v_in_normal: Option<f64>,
    pub v_in_speed: Option<f64>,
    /// Ball position in the robot frame at the kick [m] (from the tracker kick position).
    pub ball_local: Option<(f64, f64)>,
    /// Clean kick: still ball seated in front of the kicker, speed > 1 m/s, straight.
    pub clean: bool,
    /// Found by the own detector too.
    pub own_detected: bool,
}

fn ball_window(track: &[BallSample], t0: f64, t1: f64) -> &[BallSample] {
    let a = track.partition_point(|s| s.t < t0);
    let b = track.partition_point(|s| s.t < t1);
    &track[a..b]
}

/// One-sided (backward) per-camera speed profile: speed between a sample and
/// the previous sample of the same camera. Unlike the central-difference
/// velocities this does not smear a step over two frames.
fn speed_profile(win: &[BallSample]) -> Vec<(f64, f64, u32)> {
    let mut out = Vec::new();
    for (i, s) in win.iter().enumerate() {
        let prev = (0..i).rev().take(6).find(|&j| win[j].cam == s.cam);
        let Some(j) = prev else { continue };
        let dt = s.t - win[j].t;
        if dt <= 1e-4 || dt > 0.04 {
            continue;
        }
        out.push((s.t, hypot(s.x - win[j].x, s.y - win[j].y) / dt, s.cam));
    }
    out
}

fn before_profile(prof: &[(f64, f64, u32)], t_kick: f64) -> f64 {
    let v: Vec<f64> = prof
        .iter()
        .filter(|p| p.0 < t_kick - 0.03)
        .map(|p| p.1)
        .collect();
    if v.is_empty() {
        0.0
    } else {
        stats::median(&v)
    }
}

/// Analyse one tracker kick.
pub fn analyse_kick(
    game: &Game,
    k: &TrackerKick,
    ball: &[BallSample],
    robots: &RobotTracks,
    own: &[f64],
) -> Kick {
    let v_tracker = hypot(k.vel[0], k.vel[1]);
    let after = ball_window(ball, k.start - 0.15, k.start + PEAK_WINDOW);
    let prof = speed_profile(after);
    // peak of the median-of-3 (same camera) smoothed profile: robust to single-frame spikes
    let (mut v_peak, mut t_peak) = (0.0, k.start);
    for (i, &(t, v, cam)) in prof.iter().enumerate() {
        if t < k.start - 0.03 {
            continue;
        }
        let mut w = vec![v];
        if let Some(j) = (0..i).rev().find(|&j| prof[j].2 == cam) {
            w.push(prof[j].1);
        }
        if let Some(j) = (i + 1..prof.len()).find(|&j| prof[j].2 == cam) {
            w.push(prof[j].1);
        }
        let vs = stats::median(&w);
        if vs > v_peak {
            v_peak = vs;
            t_peak = t;
        }
    }
    // rise time within the camera that saw the peak
    let mut rise_frames = None;
    let mut rise_ms = None;
    if v_peak > 0.5 {
        let peak_cam = prof.iter().find(|p| p.0 == t_peak).map(|p| p.2);
        if let Some(cam) = peak_cam {
            let cp: Vec<&(f64, f64, u32)> = prof
                .iter()
                .filter(|p| p.2 == cam && p.0 <= t_peak)
                .collect();
            // onset: last sample before the peak with v < 0.2 peak + incoming speed baseline
            let base = before_profile(&prof, k.start).min(0.9 * v_peak);
            let onset = cp.iter().rposition(|p| p.1 < base + 0.2 * (v_peak - base));
            let hi = cp.iter().position(|p| p.1 >= base + 0.9 * (v_peak - base));
            if let (Some(o), Some(h)) = (onset, hi) {
                if h > o {
                    rise_frames = Some((h - o) as u32);
                    rise_ms = Some(1e3 * (cp[h].0 - cp[o].0));
                }
            }
        }
    }
    // direction fit
    let mut angle_err = None;
    let mut robot_theta = None;
    let mut robot_omega = None;
    let mut robot_v_heading = None;
    let mut v_in_normal = None;
    let mut v_in_speed = None;
    let mut ball_local = None;
    let mut dir = None;
    {
        let win = ball_window(ball, k.start, k.start + DIR_WINDOW);
        // pick the camera with most samples
        let mut best_cam = None;
        for cam in [0u32, 1, 2, 3, 4, 5, 6, 7] {
            let n = win
                .iter()
                .filter(|s| s.cam == cam && hypot(s.x - k.pos[0], s.y - k.pos[1]) > DIR_MIN_DIST)
                .count();
            if n >= 4 && best_cam.is_none_or(|(_, bn)| n > bn) {
                best_cam = Some((cam, n));
            }
        }
        if let Some((cam, _)) = best_cam {
            let sel: Vec<&BallSample> = win
                .iter()
                .filter(|s| s.cam == cam && hypot(s.x - k.pos[0], s.y - k.pos[1]) > DIR_MIN_DIST)
                .collect();
            let ts: Vec<f64> = sel.iter().map(|s| s.t).collect();
            let xs: Vec<f64> = sel.iter().map(|s| s.x).collect();
            let ys: Vec<f64> = sel.iter().map(|s| s.y).collect();
            if let (Some(fx), Some(fy)) = (stats::linreg(&ts, &xs), stats::linreg(&ts, &ys)) {
                if hypot(fx.1, fy.1) > 0.3 {
                    dir = Some(fy.1.atan2(fx.1));
                }
            }
        }
    }
    if let Some(rid) = k.robot {
        if let Some(track) = robots.get(&rid) {
            if let Some(p) = robot_pose_at(track, k.start) {
                robot_theta = Some(wrap_angle(p.theta));
                let (c, sn) = (p.theta.cos(), p.theta.sin());
                let (dx, dy) = (k.pos[0] - p.x, k.pos[1] - p.y);
                ball_local = Some((dx * c + dy * sn, -dx * sn + dy * c));
                if let Some(d) = dir {
                    angle_err = Some(wrap_angle(d - p.theta));
                }
                if let Some((vx, vy, om)) = robot_velocity_at(track, k.start, 0.05) {
                    robot_omega = Some(om);
                    robot_v_heading = Some(vx * p.theta.cos() + vy * p.theta.sin());
                }
                // incoming ball velocity
                let before = ball_window(
                    ball,
                    k.start + INCOMING_WINDOW.0,
                    k.start + INCOMING_WINDOW.1,
                );
                let vx: Vec<f64> = before
                    .iter()
                    .filter(|s| s.vx.is_finite())
                    .map(|s| s.vx)
                    .collect();
                let vy: Vec<f64> = before
                    .iter()
                    .filter(|s| s.vx.is_finite())
                    .map(|s| s.vy)
                    .collect();
                if vx.len() >= 3 {
                    let (mx, my) = (stats::median(&vx), stats::median(&vy));
                    v_in_normal = Some(-(mx * p.theta.cos() + my * p.theta.sin()));
                    v_in_speed = Some(hypot(mx, my));
                }
            }
        }
    }
    let own_detected = own.iter().any(|&t| (t - k.start).abs() < 0.12);
    let clean = k.vel[2] < 0.3
        && v_tracker > 1.0
        && v_in_speed.is_some_and(|v| v < 0.2)
        && ball_local.is_some_and(|(x, y)| (0.04..=0.16).contains(&x) && y.abs() < 0.05);
    Kick {
        ball_local,
        clean,
        game: game.name.clone(),
        t: k.start,
        team: k.robot.map(|r| r.0),
        robot: k.robot.map(|r| r.1),
        v_tracker,
        vz_tracker: k.vel[2],
        chip: k.vel[2] > 0.3,
        v_peak_raw: v_peak,
        t_peak,
        rise_frames,
        rise_ms,
        angle_err,
        robot_theta,
        robot_omega,
        robot_v_heading,
        v_in_normal,
        v_in_speed,
        own_detected,
    }
}

/// Independent kick detector: speed rise > DETECT_JUMP over 3 same-camera
/// frames while the ball sits in front of a robot. Returns event times.
pub fn detect_kicks(ball: &[BallSample], robots: &RobotTracks) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::new();
    let n = ball.len();
    for i in 0..n {
        let s = &ball[i];
        if !s.vx.is_finite() || s.near_dist > 0.3 {
            continue;
        }
        // third next same-camera sample
        let mut cnt = 0;
        let mut j_end = None;
        for (j, b) in ball.iter().enumerate().take((i + 12).min(n)).skip(i + 1) {
            if b.cam == s.cam && b.vx.is_finite() {
                cnt += 1;
                if cnt == 3 {
                    j_end = Some(j);
                    break;
                }
            }
        }
        let Some(j) = j_end else { continue };
        if ball[j].t - s.t > 0.08 || ball[j].speed() - s.speed() < DETECT_JUMP {
            continue;
        }
        let Some(rid) = s.near_robot else { continue };
        let Some(track) = robots.get(&rid) else {
            continue;
        };
        let Some(p) = robot_pose_at(track, s.t) else {
            continue;
        };
        let (c, sn) = (p.theta.cos(), p.theta.sin());
        let lx = (s.x - p.x) * c + (s.y - p.y) * sn;
        let ly = -(s.x - p.x) * sn + (s.y - p.y) * c;
        if lx < DETECT_X.0 || lx > DETECT_X.1 || ly.abs() > DETECT_Y {
            continue;
        }
        if out.last().is_some_and(|&t| s.t - t < KICK_SEPARATION) {
            continue;
        }
        out.push(s.t);
    }
    out
}

/// Aggregated kick results.
#[derive(Debug, Clone, Serialize)]
pub struct KickResult {
    pub thresholds: serde_json::Value,
    pub n_tracker: usize,
    pub n_own: usize,
    pub n_tracker_matched_by_own: usize,
    pub n_own_matched_by_tracker: usize,
    pub speed_tracker_all: Summary,
    pub speed_tracker_straight: Summary,
    pub speed_tracker_chip: Summary,
    pub speed_raw_peak_straight: Summary,
    /// Raw peak / tracker v0 for straight kicks.
    pub raw_over_tracker: Summary,
    pub per_team_speed_p95: Vec<(String, f64, f64, usize)>,
    pub angle_err_deg: Summary,
    pub angle_err_deg_fast: Summary,
    pub angle_err_deg_moving_ball: Summary,
    pub angle_err_deg_still_ball: Summary,
    /// Angle error for clean kicks (still ball seated at the kicker, > 1 m/s).
    pub angle_err_deg_clean: Summary,
    pub angle_err_clean_hist: Histogram,
    /// Regression of angle error on robot omega (deg per rad/s) — a latency signature.
    pub angle_err_vs_omega_slope: Option<(f64, f64, usize)>,
    pub rise_frames: Summary,
    pub rise_ms: Summary,
    pub rise_hist: Histogram,
    /// Incoming normal speed bins: (lo, hi, median outgoing raw, median tracker, n).
    pub incoming_vs_outgoing: Vec<(f64, f64, f64, f64, usize)>,
    /// Slope of outgoing (raw peak) on incoming normal speed for straight kicks > 1 m/s.
    pub outgoing_vs_incoming_slope: Option<(f64, f64, usize)>,
    pub speed_hist: Histogram,
    pub angle_hist: Histogram,
    pub kicks: Vec<Kick>,
    pub text: String,
}

pub fn aggregate(
    kicks: Vec<Kick>,
    n_own: usize,
    n_own_matched: usize,
    team_names: &std::collections::BTreeMap<(String, Team), String>,
) -> KickResult {
    let straight: Vec<&Kick> = kicks.iter().filter(|k| !k.chip).collect();
    let chip: Vec<&Kick> = kicks.iter().filter(|k| k.chip).collect();
    let speed_tracker_all = Summary::of(&kicks.iter().map(|k| k.v_tracker).collect::<Vec<_>>());
    let speed_tracker_straight =
        Summary::of(&straight.iter().map(|k| k.v_tracker).collect::<Vec<_>>());
    let speed_tracker_chip = Summary::of(&chip.iter().map(|k| k.v_tracker).collect::<Vec<_>>());
    let speed_raw_peak_straight = Summary::of(
        &straight
            .iter()
            .filter(|k| k.v_peak_raw > 0.3)
            .map(|k| k.v_peak_raw)
            .collect::<Vec<_>>(),
    );
    let raw_over_tracker = Summary::of(
        &straight
            .iter()
            .filter(|k| k.v_peak_raw > 0.5 && k.v_tracker > 0.5)
            .map(|k| k.v_peak_raw / k.v_tracker)
            .collect::<Vec<_>>(),
    );
    // per team
    let mut per_team: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    for k in &straight {
        if let Some(t) = k.team {
            let name = team_names
                .get(&(k.game.clone(), t))
                .cloned()
                .unwrap_or_else(|| t.name().to_string());
            per_team.entry(name).or_default().push(k.v_tracker);
        }
    }
    let per_team_speed_p95: Vec<(String, f64, f64, usize)> = per_team
        .iter()
        .map(|(n, v)| {
            (
                n.clone(),
                stats::quantile(v, 0.95),
                stats::quantile(v, 0.99),
                v.len(),
            )
        })
        .collect();
    let ang = |sel: &[&Kick]| {
        Summary::of(
            &sel.iter()
                .filter_map(|k| k.angle_err)
                .map(|a| a.to_degrees())
                .collect::<Vec<_>>(),
        )
    };
    let angle_err_deg = ang(&straight);
    let fast: Vec<&Kick> = straight
        .iter()
        .copied()
        .filter(|k| k.v_tracker > 2.5)
        .collect();
    let moving: Vec<&Kick> = straight
        .iter()
        .copied()
        .filter(|k| k.v_in_speed.is_some_and(|v| v > 0.5))
        .collect();
    let still: Vec<&Kick> = straight
        .iter()
        .copied()
        .filter(|k| k.v_in_speed.is_some_and(|v| v < 0.15))
        .collect();
    let angle_err_deg_fast = ang(&fast);
    let angle_err_deg_moving_ball = ang(&moving);
    let angle_err_deg_still_ball = ang(&still);
    let clean: Vec<&Kick> = straight.iter().copied().filter(|k| k.clean).collect();
    let angle_err_deg_clean = ang(&clean);
    let angle_err_clean_hist = Histogram::new(
        &clean
            .iter()
            .filter_map(|k| k.angle_err)
            .map(|a| a.to_degrees())
            .collect::<Vec<_>>(),
        -15.0,
        15.0,
        30,
    );
    let (om, ae): (Vec<f64>, Vec<f64>) = clean
        .iter()
        .filter_map(|k| Some((k.robot_omega?, k.angle_err?.to_degrees())))
        .filter(|(o, a)| o.abs() < 10.0 && a.abs() < 30.0)
        .unzip();
    let angle_err_vs_omega_slope = stats::linreg(&om, &ae).map(|(_, b, se, _, n)| (b, se, n));
    let rise_frames = Summary::of(
        &kicks
            .iter()
            .filter_map(|k| k.rise_frames.map(|f| f as f64))
            .collect::<Vec<_>>(),
    );
    let rise_ms = Summary::of(&kicks.iter().filter_map(|k| k.rise_ms).collect::<Vec<_>>());
    let rise_hist = Histogram::new(
        &kicks
            .iter()
            .filter_map(|k| k.rise_frames.map(|f| f as f64))
            .collect::<Vec<_>>(),
        0.0,
        8.0,
        8,
    );
    let mut incoming_vs_outgoing = Vec::new();
    for (lo, hi) in [
        (-3.0, -0.3),
        (-0.3, 0.15),
        (0.15, 0.5),
        (0.5, 1.0),
        (1.0, 1.5),
        (1.5, 2.5),
        (2.5, 5.0),
    ] {
        let sel: Vec<&Kick> = straight
            .iter()
            .copied()
            .filter(|k| k.v_in_normal.is_some_and(|v| v >= lo && v < hi) && k.v_peak_raw > 0.5)
            .collect();
        if sel.len() >= 3 {
            incoming_vs_outgoing.push((
                lo,
                hi,
                stats::median(&sel.iter().map(|k| k.v_peak_raw).collect::<Vec<_>>()),
                stats::median(&sel.iter().map(|k| k.v_tracker).collect::<Vec<_>>()),
                sel.len(),
            ));
        }
    }
    let (vin, vout): (Vec<f64>, Vec<f64>) = straight
        .iter()
        .filter(|k| k.v_peak_raw > 1.0)
        .filter_map(|k| Some((k.v_in_normal?, k.v_peak_raw)))
        .unzip();
    let outgoing_vs_incoming_slope = stats::linreg(&vin, &vout).map(|(_, b, se, _, n)| (b, se, n));
    let speed_hist = Histogram::new(
        &straight.iter().map(|k| k.v_tracker).collect::<Vec<_>>(),
        0.0,
        7.0,
        14,
    );
    let angle_hist = Histogram::new(
        &straight
            .iter()
            .filter_map(|k| k.angle_err)
            .map(|a| a.to_degrees())
            .collect::<Vec<_>>(),
        -20.0,
        20.0,
        20,
    );
    let n_tracker_matched_by_own = kicks.iter().filter(|k| k.own_detected).count();
    let mut text = String::new();
    text += &format!(
        "tracker kicks {} (straight {}, chip {}); own detector {} events, {} tracker kicks confirmed by own detector, {} own events matched a tracker kick\n",
        kicks.len(),
        straight.len(),
        chip.len(),
        n_own,
        n_tracker_matched_by_own,
        n_own_matched
    );
    text += &format!("speed tracker all: {}\n", speed_tracker_all.line("m/s"));
    text += &format!(
        "speed tracker straight: {}\n",
        speed_tracker_straight.line("m/s")
    );
    text += &format!("speed tracker chip: {}\n", speed_tracker_chip.line("m/s"));
    text += &format!(
        "speed raw peak straight: {}\n",
        speed_raw_peak_straight.line("m/s")
    );
    text += &format!("raw peak / tracker v0: {}\n", raw_over_tracker.line(""));
    for (n, p95, max, c) in &per_team_speed_p95 {
        text += &format!("  team {n}: straight p95 {p95:.2} p99 {max:.2} (n {c})\n");
    }
    text += &format!("angle error straight: {}\n", angle_err_deg.line("deg"));
    text += &format!(
        "angle error fast (>2.5 m/s): {}\n",
        angle_err_deg_fast.line("deg")
    );
    text += &format!(
        "angle error moving ball (>0.5 m/s): {}\n",
        angle_err_deg_moving_ball.line("deg")
    );
    text += &format!(
        "angle error still ball (<0.15 m/s): {}\n",
        angle_err_deg_still_ball.line("deg")
    );
    text += &format!(
        "angle error clean kicks: {}\n",
        angle_err_deg_clean.line("deg")
    );
    text += &format!(
        "angle error clean kicks: IQR sigma {:.2} deg, |err| < 2 deg in {:.0}%, < 5 deg in {:.0}%\n",
        (angle_err_deg_clean.p75 - angle_err_deg_clean.p25) / 1.349,
        100.0 * clean.iter().filter(|k| k.angle_err.is_some_and(|a| a.abs().to_degrees() < 2.0)).count() as f64 / clean.len().max(1) as f64,
        100.0 * clean.iter().filter(|k| k.angle_err.is_some_and(|a| a.abs().to_degrees() < 5.0)).count() as f64 / clean.len().max(1) as f64
    );
    text += &angle_err_clean_hist.render("angle error histogram, clean kicks [deg]", 40);
    if let Some((b, se, n)) = angle_err_vs_omega_slope {
        text += &format!("angle error vs robot omega: {:.2} ± {:.2} deg per rad/s (n {}) => latency-like {:.0} ms\n", b, se, n, b.to_radians() * 1e3);
    }
    text += &format!(
        "rise frames: {}\nrise ms: {}\n",
        rise_frames.line("frames"),
        rise_ms.line("ms")
    );
    text += &rise_hist.render("rise time histogram [frames]", 40);
    text += "incoming normal speed -> outgoing (raw peak, tracker v0):\n";
    for (lo, hi, r, t, n) in &incoming_vs_outgoing {
        text += &format!(
            "  {:+.2}..{:+.2}: raw {:.2} tracker {:.2} (n {})\n",
            lo, hi, r, t, n
        );
    }
    if let Some((b, se, n)) = outgoing_vs_incoming_slope {
        text += &format!("slope outgoing/incoming: {:.3} ± {:.3} (n {})\n", b, se, n);
    }
    text += &speed_hist.render("straight kick speed histogram [m/s]", 40);
    text += &angle_hist.render("angle error histogram [deg]", 40);
    KickResult {
        thresholds: serde_json::json!({
            "peak_window_s": PEAK_WINDOW, "incoming_window_s": INCOMING_WINDOW, "dir_min_dist_m": DIR_MIN_DIST,
            "dir_window_s": DIR_WINDOW, "detect_jump_mps": DETECT_JUMP, "detect_x_m": DETECT_X, "detect_y_m": DETECT_Y,
            "kick_separation_s": KICK_SEPARATION, "chip_if_vz_gt": 0.3,
        }),
        n_tracker: kicks.len(),
        n_own,
        n_tracker_matched_by_own,
        n_own_matched_by_tracker: n_own_matched,
        speed_tracker_all,
        speed_tracker_straight,
        speed_tracker_chip,
        speed_raw_peak_straight,
        raw_over_tracker,
        per_team_speed_p95,
        angle_err_deg,
        angle_err_deg_fast,
        angle_err_deg_moving_ball,
        angle_err_deg_still_ball,
        angle_err_deg_clean,
        angle_err_clean_hist,
        angle_err_vs_omega_slope,
        rise_frames,
        rise_ms,
        rise_hist,
        incoming_vs_outgoing,
        outgoing_vs_incoming_slope,
        speed_hist,
        angle_hist,
        kicks,
        text,
    }
}

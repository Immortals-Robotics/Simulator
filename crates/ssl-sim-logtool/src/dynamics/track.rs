//! Build object tracks from the raw per-camera detections.
//!
//! Ball: one merged track (samples keep their camera id so that velocities
//! are always differentiated within one camera, which makes them immune to
//! the per-camera calibration offset in the overlap band). Robots: one track
//! per (team, id) with Savitzky–Golay (local quadratic) velocity and
//! acceleration, again per camera.

use std::collections::BTreeMap;

use super::load::{Game, Team};
use super::stats::{hypot, local_quadratic, wrap_angle};

/// Max gap [s] between neighbouring same-camera samples used for derivatives.
pub const DERIV_MAX_GAP: f64 = 0.06;
/// Half window [samples] for the robot local quadratic fit.
pub const ROBOT_SG_HALF: usize = 4;
/// Robots updated within this many seconds count as "present" for proximity.
const ROBOT_STALE: f64 = 0.15;
/// Same-camera consecutive robot samples further apart than this [m] or [rad]
/// are identity swaps / mis-detections: no derivatives across them.
pub const ROBOT_JUMP_POS: f64 = 0.08;
pub const ROBOT_JUMP_ANGLE: f64 = 0.8;

/// Merged ball sample.
#[derive(Debug, Clone, Copy)]
pub struct BallSample {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    pub cam: u32,
    /// Per-camera finite-difference velocity [m/s] (NaN if unavailable).
    pub vx: f64,
    pub vy: f64,
    /// Distance to the nearest robot centre [m] and that robot.
    pub near_dist: f64,
    pub near_robot: Option<(Team, u32)>,
    /// Tracker ball height [m] at this time (NaN if no tracker frame within 50 ms).
    pub z: f64,
    /// Whether another camera reported the ball within 3 ms (overlap band).
    pub dual: bool,
}

impl BallSample {
    pub fn speed(&self) -> f64 {
        hypot(self.vx, self.vy)
    }
}

/// Robot sample.
#[derive(Debug, Clone, Copy)]
pub struct RobotSample {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    /// Unwrapped orientation [rad].
    pub theta: f64,
    pub cam: u32,
    /// SG velocity [m/s], acceleration [m/s^2], angular velocity/acceleration (NaN if unavailable).
    pub vx: f64,
    pub vy: f64,
    pub ax: f64,
    pub ay: f64,
    pub omega: f64,
    pub alpha: f64,
}

impl RobotSample {
    pub fn speed(&self) -> f64 {
        hypot(self.vx, self.vy)
    }
    pub fn accel(&self) -> f64 {
        hypot(self.ax, self.ay)
    }
    /// Acceleration component along the velocity (positive = speeding up).
    pub fn accel_tangential(&self) -> f64 {
        let s = self.speed();
        if s < 1e-6 {
            f64::NAN
        } else {
            (self.ax * self.vx + self.ay * self.vy) / s
        }
    }
}

/// All robot tracks.
pub type RobotTracks = BTreeMap<(Team, u32), Vec<RobotSample>>;

/// Latest known pose of every robot.
#[derive(Debug, Clone, Copy)]
pub struct RobotPose {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

/// Build the merged ball track.
pub fn ball_track(game: &Game) -> Vec<BallSample> {
    let mut out: Vec<BallSample> = Vec::with_capacity(game.raw.len());
    let mut poses: BTreeMap<(Team, u32), RobotPose> = BTreeMap::new();
    // last accepted sample and a velocity estimate for prediction, per camera
    // (a flying ball projects to different floor positions in different cameras)
    let mut last_cam: BTreeMap<u32, (f64, f64, f64, f64, f64)> = BTreeMap::new();
    let mut last: Option<(f64, f64, f64)> = None;
    let mut trk_idx = 0usize;
    for f in &game.raw {
        for r in &f.robots {
            poses.insert(
                (r.team, r.id),
                RobotPose {
                    t: f.t,
                    x: r.x,
                    y: r.y,
                    theta: r.theta.unwrap_or(0.0),
                },
            );
        }
        if f.balls.is_empty() {
            continue;
        }
        // choose detection: predict from the same camera's last sample when
        // recent, else from the last sample of any camera
        let same = last_cam
            .get(&f.camera)
            .copied()
            .filter(|l| f.t - l.0 < 0.12);
        let choice = match (same, last) {
            (Some((t0, x0, y0, vx, vy)), _) => {
                let dt = f.t - t0;
                let px = x0 + vx * dt;
                let py = y0 + vy * dt;
                let gate = 0.10 + hypot(vx, vy) * dt + 2.0 * dt;
                f.balls
                    .iter()
                    .map(|b| (hypot(b.x - px, b.y - py), b))
                    .filter(|(d, _)| *d < gate)
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .map(|(_, b)| b)
            }
            (None, Some((t0, x0, y0))) if f.t - t0 < 0.5 => {
                // other camera: allow the flight projection offset
                let gate = 0.5 + 8.0 * (f.t - t0);
                f.balls
                    .iter()
                    .map(|b| (hypot(b.x - x0, b.y - y0), b))
                    .filter(|(d, _)| *d < gate)
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .map(|(_, b)| b)
            }
            (None, Some((_, x0, y0))) => f
                .balls
                .iter()
                .map(|b| (hypot(b.x - x0, b.y - y0), b))
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .map(|(_, b)| b),
            (None, None) => f
                .balls
                .iter()
                .max_by(|a, b| a.area.unwrap_or(0).cmp(&b.area.unwrap_or(0))),
        };
        let Some(b) = choice else { continue };
        let (mut vx, mut vy) = (0.0, 0.0);
        if let Some((t0, x0, y0, _, _)) = last_cam.get(&f.camera).copied() {
            let dt = f.t - t0;
            if dt > 0.004 && dt < 0.1 {
                vx = (b.x - x0) / dt;
                vy = (b.y - y0) / dt;
            }
        }
        last_cam.insert(f.camera, (f.t, b.x, b.y, vx, vy));
        last = Some((f.t, b.x, b.y));
        // nearest robot
        let mut near_dist = f64::INFINITY;
        let mut near_robot = None;
        for (k, p) in &poses {
            if f.t - p.t > ROBOT_STALE {
                continue;
            }
            let d = hypot(b.x - p.x, b.y - p.y);
            if d < near_dist {
                near_dist = d;
                near_robot = Some(*k);
            }
        }
        // tracker z
        while trk_idx + 1 < game.tracker.len() && game.tracker[trk_idx + 1].t <= f.t {
            trk_idx += 1;
        }
        let mut z = f64::NAN;
        if !game.tracker.is_empty() {
            let cand = [trk_idx, (trk_idx + 1).min(game.tracker.len() - 1)];
            let mut best = 0.05;
            for c in cand {
                let tf = &game.tracker[c];
                if let Some(bb) = tf.ball {
                    if (tf.t - f.t).abs() < best {
                        best = (tf.t - f.t).abs();
                        z = bb.z;
                    }
                }
            }
        }
        let dual = out
            .last()
            .is_some_and(|p| p.cam != f.camera && (f.t - p.t).abs() < 0.003);
        if dual {
            if let Some(p) = out.last_mut() {
                p.dual = true;
            }
        }
        out.push(BallSample {
            t: f.t,
            x: b.x,
            y: b.y,
            cam: f.camera,
            vx: f64::NAN,
            vy: f64::NAN,
            near_dist,
            near_robot,
            z,
            dual,
        });
    }
    // per-camera velocities: central difference over same-camera neighbours
    let n = out.len();
    for i in 0..n {
        let cam = out[i].cam;
        let prev = (0..i)
            .rev()
            .take(8)
            .find(|&j| out[j].cam == cam && out[i].t - out[j].t < DERIV_MAX_GAP);
        let next = (i + 1..n)
            .take(8)
            .find(|&j| out[j].cam == cam && out[j].t - out[i].t < DERIV_MAX_GAP);
        let (a, b) = match (prev, next) {
            (Some(a), Some(b)) => (a, b),
            (Some(a), None) => (a, i),
            (None, Some(b)) => (i, b),
            (None, None) => continue,
        };
        let dt = out[b].t - out[a].t;
        if dt > 1e-4 {
            out[i].vx = (out[b].x - out[a].x) / dt;
            out[i].vy = (out[b].y - out[a].y) / dt;
        }
    }
    out
}

/// Build robot tracks with SG derivatives.
pub fn robot_tracks(game: &Game) -> RobotTracks {
    let mut tracks: RobotTracks = BTreeMap::new();
    for f in &game.raw {
        for r in &f.robots {
            let Some(theta) = r.theta else { continue };
            tracks.entry((r.team, r.id)).or_default().push(RobotSample {
                t: f.t,
                x: r.x,
                y: r.y,
                theta,
                cam: f.camera,
                vx: f64::NAN,
                vy: f64::NAN,
                ax: f64::NAN,
                ay: f64::NAN,
                omega: f64::NAN,
                alpha: f64::NAN,
            });
        }
    }
    for track in tracks.values_mut() {
        // unwrap orientation per camera in time order
        let mut last_theta: BTreeMap<u32, f64> = BTreeMap::new();
        for s in track.iter_mut() {
            if let Some(&prev) = last_theta.get(&s.cam) {
                s.theta = prev + wrap_angle(s.theta - prev);
            }
            last_theta.insert(s.cam, s.theta);
        }
        sg_derivatives(track);
    }
    tracks
}

/// Local quadratic fit over same-camera neighbours (±ROBOT_SG_HALF samples).
fn sg_derivatives(track: &mut [RobotSample]) {
    let n = track.len();
    let mut ts = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut hs = Vec::new();
    for i in 0..n {
        let cam = track[i].cam;
        let t0 = track[i].t;
        ts.clear();
        xs.clear();
        ys.clear();
        hs.clear();
        let lo = i.saturating_sub(4 * ROBOT_SG_HALF);
        let hi = (i + 4 * ROBOT_SG_HALF + 1).min(n);
        let mut before = 0;
        let mut after = 0;
        for (j, s) in track.iter().enumerate().take(hi).skip(lo) {
            if s.cam != cam || (s.t - t0).abs() > (ROBOT_SG_HALF as f64 + 0.5) * 0.017 {
                continue;
            }
            if j < i {
                before += 1;
            } else if j > i {
                after += 1;
            }
            ts.push(s.t);
            xs.push(s.x);
            ys.push(s.y);
            hs.push(s.theta);
        }
        if before < ROBOT_SG_HALF || after < ROBOT_SG_HALF {
            continue;
        }
        // check the window is contiguous (no gap > 2 frames) and free of jumps
        if ts.windows(2).any(|w| w[1] - w[0] > 0.04) {
            continue;
        }
        let jump = (1..ts.len()).any(|k| {
            hypot(xs[k] - xs[k - 1], ys[k] - ys[k - 1]) > ROBOT_JUMP_POS
                || (hs[k] - hs[k - 1]).abs() > ROBOT_JUMP_ANGLE
        });
        if jump {
            continue;
        }
        if let (Some(fx), Some(fy), Some(fh)) = (
            local_quadratic(&ts, &xs, t0),
            local_quadratic(&ts, &ys, t0),
            local_quadratic(&ts, &hs, t0),
        ) {
            let s = &mut track[i];
            s.vx = fx.1;
            s.vy = fy.1;
            s.ax = fx.2;
            s.ay = fy.2;
            s.omega = fh.1;
            s.alpha = fh.2;
        }
    }
}

/// Index of the sample nearest in time (binary search on `t`).
pub fn nearest_idx<T>(v: &[T], t: f64, time: impl Fn(&T) -> f64) -> Option<usize> {
    if v.is_empty() {
        return None;
    }
    let i = v.partition_point(|s| time(s) < t);
    let cands = [i.saturating_sub(1), i.min(v.len() - 1)];
    cands.into_iter().min_by(|&a, &b| {
        (time(&v[a]) - t)
            .abs()
            .partial_cmp(&(time(&v[b]) - t).abs())
            .unwrap()
    })
}

/// Interpolated robot pose at time `t` (None if no samples within 60 ms).
pub fn robot_pose_at(track: &[RobotSample], t: f64) -> Option<RobotPose> {
    let i = nearest_idx(track, t, |s| s.t)?;
    let s = &track[i];
    if (s.t - t).abs() > 0.06 {
        return None;
    }
    // linear interpolation with the neighbour on the other side
    let j = if s.t <= t { i + 1 } else { i.wrapping_sub(1) };
    if j < track.len() && (track[j].t - t).abs() < 0.06 && track[j].t != s.t {
        let o = &track[j];
        let f = (t - s.t) / (o.t - s.t);
        return Some(RobotPose {
            t,
            x: s.x + (o.x - s.x) * f,
            y: s.y + (o.y - s.y) * f,
            theta: s.theta + wrap_angle(o.theta - s.theta) * f,
        });
    }
    Some(RobotPose {
        t: s.t,
        x: s.x,
        y: s.y,
        theta: s.theta,
    })
}

/// Median SG velocity of the robot within ±window of `t` (NaN if none).
pub fn robot_velocity_at(track: &[RobotSample], t: f64, window: f64) -> Option<(f64, f64, f64)> {
    let i = nearest_idx(track, t, |s| s.t)?;
    let mut vx = Vec::new();
    let mut vy = Vec::new();
    let mut om = Vec::new();
    let lo = i.saturating_sub(12);
    for s in &track[lo..(i + 12).min(track.len())] {
        if (s.t - t).abs() <= window && s.vx.is_finite() {
            vx.push(s.vx);
            vy.push(s.vy);
            om.push(s.omega);
        }
    }
    if vx.is_empty() {
        return None;
    }
    Some((
        super::stats::median(&vx),
        super::stats::median(&vy),
        super::stats::median(&om),
    ))
}

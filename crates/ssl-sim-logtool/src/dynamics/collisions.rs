//! Ball–robot, ball–goal-post and ball–boundary contacts: velocity jumps of a
//! grounded ball at a surface, evaluated with the simulator's collision kernel
//! `v'_n = u_n + (u_n - v_n)(1 - k_n)`, `v'_t = k_t u_t + (1 - k_t) v_t`.

use serde::Serialize;

use super::kicks::TrackerKick;
use super::load::{Game, Team};
use super::stats::{self, hypot, wrap_angle, Histogram, Summary};
use super::track::{robot_pose_at, robot_velocity_at, BallSample, RobotTracks};

/// Velocity change between consecutive same-camera samples that flags an event [m/s].
const JUMP: f64 = 0.5;
/// Robot hull radius, ball radius, centre-to-dribbler [m] (proto defaults).
const ROBOT_R: f64 = 0.09;
const BALL_R: f64 = 0.0215;
const C2D: f64 = 0.075;
/// Slack on the contact distance at the closest approach [m].
const CONTACT_SLACK: f64 = 0.07;
/// Ignore events within this of a tracker kick / own kick [s].
const KICK_EXCLUDE: f64 = 0.20;
/// Velocity windows before/after the event [s].
const V_WINDOW: (f64, f64) = (0.015, 0.11);
/// Minimum normal approach speed [m/s].
const MIN_APPROACH: f64 = 0.3;
/// Minimum tangential relative speed for a k_t estimate [m/s].
const MIN_TANGENT: f64 = 0.4;
/// Goal post / wall proximity [m].
const WALL_NEAR: f64 = 0.06;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Surface {
    Hull,
    Face,
    Post,
    GoalBack,
    Boundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Outcome {
    Rebound,
    Captured,
    Other,
}

/// Closest-approach candidate: robot id, distance, ball x/y, robot x/y.
type Approach = Option<((Team, u32), f64, f64, f64, f64, f64)>;

/// One contact event.
#[derive(Debug, Clone, Serialize)]
pub struct Contact {
    pub game: String,
    pub t: f64,
    pub surface: Surface,
    pub outcome: Outcome,
    pub robot: Option<(Team, u32)>,
    /// Contact angle in the robot frame [deg].
    pub phi_deg: f64,
    /// Relative normal speeds (negative = approaching).
    pub vn_in: f64,
    pub vn_out: f64,
    pub restitution: f64,
    pub k_n: f64,
    pub k_t: Option<f64>,
    pub robot_speed: f64,
    pub ball_speed_in: f64,
    pub ball_speed_out: f64,
}

fn median_velocity(ball: &[BallSample], t0: f64, t1: f64) -> Option<(f64, f64, usize)> {
    let a = ball.partition_point(|s| s.t < t0);
    let b = ball.partition_point(|s| s.t < t1);
    let vx: Vec<f64> = ball[a..b]
        .iter()
        .filter(|s| s.vx.is_finite())
        .map(|s| s.vx)
        .collect();
    let vy: Vec<f64> = ball[a..b]
        .iter()
        .filter(|s| s.vx.is_finite())
        .map(|s| s.vy)
        .collect();
    if vx.len() < 2 {
        return None;
    }
    Some((stats::median(&vx), stats::median(&vy), vx.len()))
}

/// Detect event times: consecutive same-camera velocity jumps, merged within 0.1 s.
fn events(ball: &[BallSample]) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::new();
    let n = ball.len();
    for i in 0..n {
        let s = &ball[i];
        if !s.vx.is_finite() {
            continue;
        }
        let Some(j) =
            (i + 1..(i + 6).min(n)).find(|&j| ball[j].cam == s.cam && ball[j].vx.is_finite())
        else {
            continue;
        };
        let o = &ball[j];
        if o.t - s.t > 0.04 {
            continue;
        }
        let dv = hypot(o.vx - s.vx, o.vy - s.vy);
        if dv > JUMP && s.speed().max(o.speed()) > 0.4 {
            let t = 0.5 * (s.t + o.t);
            if out.last().is_some_and(|&l| t - l < 0.1) {
                continue;
            }
            out.push(t);
        }
    }
    out
}

fn kernel(
    v_in: (f64, f64),
    v_out: (f64, f64),
    u: (f64, f64),
    n: (f64, f64),
) -> Option<(f64, f64, f64, f64, Option<f64>)> {
    let t = (-n.1, n.0);
    let vn_in = (v_in.0 - u.0) * n.0 + (v_in.1 - u.1) * n.1;
    let vn_out = (v_out.0 - u.0) * n.0 + (v_out.1 - u.1) * n.1;
    if vn_in > -MIN_APPROACH {
        return None;
    }
    let e = -vn_out / vn_in;
    let k_n = 1.0 - e;
    let ut = u.0 * t.0 + u.1 * t.1;
    let vt_in = v_in.0 * t.0 + v_in.1 * t.1;
    let vt_out = v_out.0 * t.0 + v_out.1 * t.1;
    let k_t = if (ut - vt_in).abs() > MIN_TANGENT {
        Some((vt_out - vt_in) / (ut - vt_in))
    } else {
        None
    };
    Some((vn_in, vn_out, e, k_n, k_t))
}

/// Analyse the contacts of one game.
pub fn analyse(
    game: &Game,
    ball: &[BallSample],
    robots: &RobotTracks,
    kicks: &[TrackerKick],
    own_kicks: &[f64],
) -> Vec<Contact> {
    let mut out = Vec::new();
    let field = game.field;
    let mouth_half = (C2D / ROBOT_R).acos();
    for t in events(ball) {
        let near_kick = kicks.iter().any(|k| (k.start - t).abs() < KICK_EXCLUDE)
            || own_kicks.iter().any(|&k| (k - t).abs() < KICK_EXCLUDE);
        // grounded?
        let i = ball.partition_point(|s| s.t < t).min(ball.len() - 1);
        let bs = &ball[i];
        if !(bs.z.is_finite() && bs.z < 0.03) {
            continue;
        }
        let Some((vin_x, vin_y, _)) = median_velocity(ball, t - V_WINDOW.1, t - V_WINDOW.0) else {
            continue;
        };
        let Some((vout_x, vout_y, _)) = median_velocity(ball, t + V_WINDOW.0, t + V_WINDOW.1)
        else {
            continue;
        };
        let v_in = (vin_x, vin_y);
        let v_out = (vout_x, vout_y);
        // closest approach to any robot within ±50 ms
        let a = ball.partition_point(|s| s.t < t - 0.05);
        let b = ball.partition_point(|s| s.t < t + 0.05);
        let mut best: Approach = None;
        for s in &ball[a..b] {
            for (id, track) in robots {
                let Some(p) = robot_pose_at(track, s.t) else {
                    continue;
                };
                let d = hypot(s.x - p.x, s.y - p.y);
                if d < ROBOT_R + BALL_R + CONTACT_SLACK && best.is_none_or(|bb| d < bb.1) {
                    best = Some((*id, d, s.x, s.y, p.x, p.y));
                }
            }
        }
        if let Some((id, _d, bx, by, rx, ry)) = best {
            let track = &robots[&id];
            let Some(p) = robot_pose_at(track, t) else {
                continue;
            };
            let Some((rvx, rvy, om)) = robot_velocity_at(track, t, 0.06) else {
                continue;
            };
            let dn = hypot(bx - rx, by - ry).max(1e-6);
            let n_hat = ((bx - rx) / dn, (by - ry) / dn);
            let phi = wrap_angle(n_hat.1.atan2(n_hat.0) - p.theta);
            let (surface, n) = if phi.abs() < mouth_half {
                (Surface::Face, (p.theta.cos(), p.theta.sin()))
            } else {
                (Surface::Hull, n_hat)
            };
            // a kick is a face contact; hull contacts cannot be kicks
            if surface == Surface::Face && near_kick {
                continue;
            }
            // surface velocity at the contact point (robot translation + rotation)
            let r = (n_hat.0 * ROBOT_R, n_hat.1 * ROBOT_R);
            let u = (rvx - om * r.1, rvy + om * r.0);
            let Some((vn_in, vn_out, e, k_n, k_t)) = kernel(v_in, v_out, u, n) else {
                continue;
            };
            let rel_out = hypot(v_out.0 - rvx, v_out.1 - rvy);
            let outcome = if vn_out > 0.15 {
                Outcome::Rebound
            } else if rel_out < 0.3 {
                Outcome::Captured
            } else {
                Outcome::Other
            };
            out.push(Contact {
                game: game.name.clone(),
                t,
                surface,
                outcome,
                robot: Some(id),
                phi_deg: phi.to_degrees(),
                vn_in,
                vn_out,
                restitution: e,
                k_n,
                k_t,
                robot_speed: hypot(rvx, rvy),
                ball_speed_in: hypot(v_in.0, v_in.1),
                ball_speed_out: hypot(v_out.0, v_out.1),
            });
            continue;
        }
        // walls and posts
        if field.length <= 0.0 || near_kick {
            continue;
        }
        let (hl, hw, hg) = (
            field.length / 2.0,
            field.width / 2.0,
            field.goal_width / 2.0,
        );
        let mut wall: Option<(Surface, (f64, f64))> = None;
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                let (px, py) = (sx * hl, sy * hg);
                let d = hypot(bs.x - px, bs.y - py);
                if d < WALL_NEAR + 0.02 {
                    wall = Some((
                        Surface::Post,
                        ((bs.x - px) / d.max(1e-6), (bs.y - py) / d.max(1e-6)),
                    ));
                }
            }
            if wall.is_none()
                && (bs.x.abs() - (hl + field.goal_depth)).abs() < WALL_NEAR
                && bs.y.abs() < hg
                && bs.x.signum() == sx
            {
                wall = Some((Surface::GoalBack, (-sx, 0.0)));
            }
        }
        if wall.is_none() {
            if (bs.x.abs() - (hl + field.boundary_width)).abs() < WALL_NEAR {
                wall = Some((Surface::Boundary, (-bs.x.signum(), 0.0)));
            } else if (bs.y.abs() - (hw + field.boundary_width)).abs() < WALL_NEAR {
                wall = Some((Surface::Boundary, (0.0, -bs.y.signum())));
            }
        }
        let Some((surface, n)) = wall else { continue };
        let Some((vn_in, vn_out, e, k_n, k_t)) = kernel(v_in, v_out, (0.0, 0.0), n) else {
            continue;
        };
        out.push(Contact {
            game: game.name.clone(),
            t,
            surface,
            outcome: if vn_out > 0.15 {
                Outcome::Rebound
            } else {
                Outcome::Other
            },
            robot: None,
            phi_deg: f64::NAN,
            vn_in,
            vn_out,
            restitution: e,
            k_n,
            k_t,
            robot_speed: 0.0,
            ball_speed_in: hypot(v_in.0, v_in.1),
            ball_speed_out: hypot(v_out.0, v_out.1),
        });
    }
    out
}

#[derive(Debug, Clone, Serialize)]
pub struct CollisionResult {
    pub thresholds: serde_json::Value,
    pub n_events: usize,
    pub counts: Vec<(String, String, usize)>,
    pub hull_k_n: Summary,
    pub hull_k_t: Summary,
    pub hull_restitution: Summary,
    pub face_k_n: Summary,
    pub face_k_t: Summary,
    pub face_restitution: Summary,
    pub face_captured_fraction: f64,
    pub hull_captured_fraction: f64,
    pub post_restitution: Summary,
    pub goal_back_restitution: Summary,
    pub boundary_restitution: Summary,
    /// Hull restitution vs normal approach speed bins: (lo, median e, n).
    pub hull_e_vs_speed: Vec<(f64, f64, usize)>,
    pub hist_hull_k_n: Histogram,
    pub hist_face_k_n: Histogram,
    pub contacts: Vec<Contact>,
    pub text: String,
}

pub fn aggregate(contacts: Vec<Contact>) -> CollisionResult {
    let sel = |s: Surface, o: Option<Outcome>| -> Vec<&Contact> {
        contacts
            .iter()
            .filter(|c| c.surface == s && o.is_none_or(|o| c.outcome == o))
            .collect()
    };
    let hull_r = sel(Surface::Hull, Some(Outcome::Rebound));
    let face_r = sel(Surface::Face, Some(Outcome::Rebound));
    let kn = |v: &[&Contact]| Summary::of(&v.iter().map(|c| c.k_n).collect::<Vec<_>>());
    let kt = |v: &[&Contact]| Summary::of(&v.iter().filter_map(|c| c.k_t).collect::<Vec<_>>());
    let e = |v: &[&Contact]| Summary::of(&v.iter().map(|c| c.restitution).collect::<Vec<_>>());
    let mut counts: std::collections::BTreeMap<(String, String), usize> = Default::default();
    for c in &contacts {
        *counts
            .entry((format!("{:?}", c.surface), format!("{:?}", c.outcome)))
            .or_default() += 1;
    }
    let hull_all = sel(Surface::Hull, None);
    let face_all = sel(Surface::Face, None);
    let frac = |all: &[&Contact]| {
        if all.is_empty() {
            f64::NAN
        } else {
            all.iter()
                .filter(|c| c.outcome == Outcome::Captured)
                .count() as f64
                / all.len() as f64
        }
    };
    let mut hull_e_vs_speed = Vec::new();
    for b in 0..6 {
        let lo = 0.3 + b as f64 * 0.5;
        let v: Vec<f64> = hull_r
            .iter()
            .filter(|c| -c.vn_in >= lo && -c.vn_in < lo + 0.5)
            .map(|c| c.restitution)
            .collect();
        if v.len() >= 4 {
            hull_e_vs_speed.push((lo, stats::median(&v), v.len()));
        }
    }
    let hull_k_n = kn(&hull_r);
    let hull_k_t = kt(&hull_r);
    let face_k_n = kn(&face_r);
    let face_k_t = kt(&face_r);
    let post = sel(Surface::Post, Some(Outcome::Rebound));
    let back = sel(Surface::GoalBack, Some(Outcome::Rebound));
    let bound = sel(Surface::Boundary, Some(Outcome::Rebound));
    let hist_hull_k_n = Histogram::new(
        &hull_r.iter().map(|c| c.k_n).collect::<Vec<_>>(),
        0.0,
        1.0,
        20,
    );
    let hist_face_k_n = Histogram::new(
        &face_r.iter().map(|c| c.k_n).collect::<Vec<_>>(),
        0.0,
        1.0,
        20,
    );
    let mut text = String::new();
    text += &format!("contact events {}\n", contacts.len());
    for ((s, o), n) in &counts {
        text += &format!("  {s}/{o}: {n}\n");
    }
    text += &format!(
        "hull rebounds: k_n {}\n              k_t {}\n              e   {}\n",
        hull_k_n.line(""),
        hull_k_t.line(""),
        e(&hull_r).line("")
    );
    text += &format!(
        "face rebounds: k_n {}\n              k_t {}\n              e   {}\n",
        face_k_n.line(""),
        face_k_t.line(""),
        e(&face_r).line("")
    );
    text += &format!(
        "captured fraction: face {:.2} hull {:.2}\n",
        frac(&face_all),
        frac(&hull_all)
    );
    text += &format!(
        "goal post e: {}\ngoal back e: {}\nboundary e: {}\n",
        e(&post).line(""),
        e(&back).line(""),
        e(&bound).line("")
    );
    text += "hull e vs approach speed:\n";
    for (lo, m, n) in &hull_e_vs_speed {
        text += &format!("  {:.1}..{:.1}: e {:.3} (n {})\n", lo, lo + 0.5, m, n);
    }
    text += &hist_hull_k_n.render("hull k_n histogram", 40);
    text += &hist_face_k_n.render("face k_n histogram", 40);
    CollisionResult {
        thresholds: serde_json::json!({
            "jump_mps": JUMP, "robot_r": ROBOT_R, "ball_r": BALL_R, "c2d": C2D, "contact_slack_m": CONTACT_SLACK,
            "kick_exclude_s": KICK_EXCLUDE, "v_window_s": V_WINDOW, "min_approach_mps": MIN_APPROACH,
            "min_tangent_mps": MIN_TANGENT, "wall_near_m": WALL_NEAR,
            "rebound": "relative normal out-speed > 0.15", "captured": "relative speed after < 0.3",
        }),
        n_events: contacts.len(),
        counts: counts.into_iter().map(|((s, o), n)| (s, o, n)).collect(),
        hull_k_n,
        hull_k_t,
        hull_restitution: e(&hull_r),
        face_k_n,
        face_k_t,
        face_restitution: e(&face_r),
        face_captured_fraction: frac(&face_all),
        hull_captured_fraction: frac(&hull_all),
        post_restitution: e(&post),
        goal_back_restitution: e(&back),
        boundary_restitution: e(&bound),
        hull_e_vs_speed,
        hist_hull_k_n,
        hist_face_k_n,
        contacts,
        text,
    }
}

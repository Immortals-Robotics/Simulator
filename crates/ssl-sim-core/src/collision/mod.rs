//! Collision detection and response.
//!
//! OWNER: math agent.
//!
//! Ball contacts: swept-circle tests against robot hulls (disc + flat front
//! chord, height-gated by robot height), wall segments (height-gated) and the
//! room box, ordered by time of impact. Response uses the Sumatra kernel with
//! `damp_normal` (1 = inelastic) and `damp_tangent` (blend toward the surface
//! velocity), keeps a damped spin, and returns the impulse applied to the
//! other body.
//!
//! The robot hull used for the ball is the exact Minkowski sum of
//! `disc(R) ∩ {x <= c2d}` with the ball disc: a circle of radius `R + r`
//! wherever the ball centre lies outside the mouth cone (`|phi| >= theta`,
//! `theta = acos(c2d / R)`), and a capsule of radius `r` around the chord
//! (flat kicker face plus rounded corners) inside the cone.
//!
//! The robot is a cut cylinder of height `h`. Three height regimes for the
//! ball centre `z`:
//! - `z - r >= h`: the ball is above the top. If it descends onto the
//!   footprint it lands on the flat top (vertical normal, `RobotTop`); while
//!   its centre is inside the footprint the top is its floor
//!   (`support_height`, used by `physics::step_ball` to raise the trajectory).
//! - `h <= z < h + r`: the rim band. A ball outside the footprint meets the
//!   top edge: it is swept as a disc of radius `sqrt(r^2 - (z - h)^2)` and the
//!   contact normal is tilted from horizontal toward vertical (sphere vs
//!   edge, `RobotRim`). A ball with its centre inside the footprint is on top.
//! - `z < h`: sideways contact with the hull as before (height gate).
//!
//! Robot contacts: cut disc vs cut disc (when
//! `ContactParams::robot_hull_chord_contacts`, separating-axis test over the
//! centre line, both face normals and the corner-to-centre axes; discs
//! otherwise) and disc vs wall, resolved by mass-weighted positional
//! correction plus restitution/friction impulses over a few iterations.

use std::collections::BTreeMap;

use crate::field::{FieldGeometry, WallSegment};
use crate::params::{BallParams, ContactParams};
use crate::robot::Robot;
use crate::types::{rotate, BallState, Event, RobotId, Vec2, Vec3};
use crate::GRAVITY;

/// Iterations of the robot contact solver.
const ROBOT_ITERATIONS: usize = 4;
/// Max robots the contact solver handles (16 per team).
const MAX_ROBOTS: usize = 32;
/// Max distance [m] the ball is pushed out of a robot per call (Sumatra pushes 1 mm/ms).
const ROBOT_PUSH_CAP: f64 = 0.002;
/// Approach speed [m/s] above which a robot pair overlap counts as a collision event.
const COLLISION_EVENT_SPEED: f64 = 0.02;
/// Tolerance used for "already touching" tests.
const TOUCH_EPS: f64 = 1e-9;
/// Height tolerance [m] for "the ball rests on a robot top" tests.
const TOP_EPS: f64 = 1e-9;

/// What the ball hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Surface {
    /// Round part of a robot hull.
    RobotHull(RobotId),
    /// Flat kicker face of a robot.
    KickerFace(RobotId),
    /// Flat top of a robot (the ball lands on it).
    RobotTop(RobotId),
    /// Top edge of a robot (sphere vs edge, tilted normal).
    RobotRim(RobotId),
    /// Boundary board or goal frame.
    Wall {
        /// True for goal posts / back wall.
        is_goal: bool,
    },
    /// The invisible room box.
    Room,
}

impl Surface {
    /// The robot involved, if any.
    pub fn robot(&self) -> Option<RobotId> {
        match *self {
            Surface::RobotHull(id)
            | Surface::KickerFace(id)
            | Surface::RobotTop(id)
            | Surface::RobotRim(id) => Some(id),
            Surface::Wall { .. } | Surface::Room => None,
        }
    }
}

/// An imminent ball contact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BallContact {
    /// Time from the start of the sweep [s], `0 <= t <= dt`.
    pub time: f64,
    /// Contact normal pointing from the surface toward the ball (unit).
    /// Horizontal for hull, face, wall and room contacts; vertical for a
    /// robot top; tilted for a robot rim.
    pub normal: Vec3,
    /// Velocity of the surface at the contact point [m/s] (includes robot rotation).
    pub surface_velocity: Vec2,
    /// What was hit.
    pub surface: Surface,
}

impl BallContact {
    /// Horizontal part of the normal (zero for a top contact).
    pub fn normal_xy(&self) -> Vec2 {
        Vec2::new(self.normal.x, self.normal.y)
    }
}

// ----------------------------------------------------------------------------
// Primitive helpers
// ----------------------------------------------------------------------------

/// Earliest `t >= 0` at which the point `p + d t` enters the circle
/// `(c, radius)` from outside. `None` if already inside, moving away, or missing.
fn ray_circle_entry(p: Vec2, d: Vec2, c: Vec2, radius: f64) -> Option<f64> {
    let f = p - c;
    let a = d.length_squared();
    if a < 1e-18 {
        return None;
    }
    let b = 2.0 * f.dot(d);
    let cc = f.length_squared() - radius * radius;
    if cc < 0.0 {
        return None;
    }
    let disc = b * b - 4.0 * a * cc;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()) / (2.0 * a);
    (t >= 0.0).then_some(t)
}

/// Time [s] for a ball `dz >= 0` above a level with vertical velocity `vz` to
/// descend to it under gravity (0 if already there and descending).
fn fall_time(dz: f64, vz: f64) -> f64 {
    (vz + (vz * vz + 2.0 * GRAVITY * dz).sqrt()) / GRAVITY
}

/// Which feature of the hull the closest point lies on.
#[derive(Debug, Clone, Copy, PartialEq)]
enum HullFeature {
    Round,
    Face,
}

/// Robot hull geometry in the robot frame.
#[derive(Debug, Clone, Copy)]
struct Hull {
    radius: f64,
    c2d: f64,
    theta: f64,
    half_width: f64,
}

impl Hull {
    fn of(robot: &Robot) -> Hull {
        Hull {
            radius: robot.specs.radius,
            c2d: robot.specs.center_to_dribbler,
            theta: robot.specs.mouth_half_angle(),
            half_width: robot.specs.front_half_width(),
        }
    }

    fn in_mouth_cone(&self, p: Vec2) -> bool {
        p.y.atan2(p.x).abs() < self.theta
    }

    /// Signed distance from a point (robot frame) to the hull, with the
    /// outward normal of the closest feature and the feature kind.
    fn signed_distance(&self, p: Vec2) -> (f64, Vec2, HullFeature) {
        if self.in_mouth_cone(p) {
            if p.x >= self.c2d {
                let q = Vec2::new(self.c2d, p.y.clamp(-self.half_width, self.half_width));
                let delta = p - q;
                let d = delta.length();
                if p.y.abs() <= self.half_width || d < 1e-12 {
                    (p.x - self.c2d, Vec2::X, HullFeature::Face)
                } else {
                    (d, delta / d, HullFeature::Round)
                }
            } else {
                (p.x - self.c2d, Vec2::X, HullFeature::Face)
            }
        } else {
            let d = p.length();
            if d < 1e-12 {
                (-self.radius, Vec2::X, HullFeature::Round)
            } else {
                (d - self.radius, p / d, HullFeature::Round)
            }
        }
    }

    /// True if the point (robot frame) lies inside the footprint.
    fn contains(&self, p: Vec2) -> bool {
        self.signed_distance(p).0 <= 0.0
    }

    /// Sweep a ball of radius `r` (centre `p`, relative velocity `d`) for `dt`
    /// against the hull. Returns `(time, normal, feature)` in the robot frame.
    fn sweep(&self, p: Vec2, d: Vec2, r: f64, dt: f64) -> Option<(f64, Vec2, HullFeature)> {
        let (sd, n, feat) = self.signed_distance(p);
        if sd <= r + TOUCH_EPS {
            // Already touching / inside: contact now if approaching.
            return (d.dot(n) < 0.0).then_some((0.0, n, feat));
        }
        let mut best: Option<(f64, Vec2, HullFeature)> = None;
        let mut consider = |t: f64, n: Vec2, f: HullFeature| {
            if t <= dt && best.is_none_or(|b| t < b.0) {
                best = Some((t, n, f));
            }
        };
        // Big circle, valid outside the mouth cone.
        if let Some(t) = ray_circle_entry(p, d, Vec2::ZERO, self.radius + r) {
            let h = p + d * t;
            if !self.in_mouth_cone(h) {
                consider(t, h.normalize(), HullFeature::Round);
            }
        }
        // Flat face (one-sided plane at x = c2d + r).
        let s0 = p.x - (self.c2d + r);
        if s0 >= 0.0 && d.x < 0.0 {
            let t = s0 / -d.x;
            let h = p + d * t;
            if h.y.abs() <= self.half_width {
                consider(t, Vec2::X, HullFeature::Face);
            }
        }
        // Rounded chord corners.
        for sy in [self.half_width, -self.half_width] {
            let corner = Vec2::new(self.c2d, sy);
            if let Some(t) = ray_circle_entry(p, d, corner, r) {
                let h = p + d * t;
                if h.x >= self.c2d && h.y.abs() >= self.half_width && self.in_mouth_cone(h) {
                    consider(t, (h - corner) / r, HullFeature::Round);
                }
            }
        }
        best
    }
}

/// Sweep against one wall segment (world frame). Returns `(t, normal)`.
fn sweep_wall(p: Vec2, d: Vec2, r: f64, dt: f64, wall: &WallSegment) -> Option<(f64, Vec2)> {
    let n = wall.normal;
    let e = wall.b - wall.a;
    let len = e.length();
    if len < 1e-12 {
        return None;
    }
    let e_hat = e / len;
    let s0 = (p - wall.a).dot(n) - r;
    let vn = d.dot(n);
    let proj0 = (p - wall.a).dot(e_hat);
    // Already touching the face: contact now if approaching.
    if s0 <= TOUCH_EPS && s0 > -2.0 * r && (0.0..=len).contains(&proj0) {
        return (vn < 0.0).then_some((0.0, n));
    }
    let mut best: Option<(f64, Vec2)> = None;
    if s0 > 0.0 && vn < 0.0 {
        let t = s0 / -vn;
        if t <= dt {
            let proj = (p + d * t - wall.a).dot(e_hat);
            if (0.0..=len).contains(&proj) {
                best = Some((t, n));
            }
        }
    }
    for end in [wall.a, wall.b] {
        let f = p - end;
        if f.length_squared() <= (r + TOUCH_EPS) * (r + TOUCH_EPS) {
            let m = if f.length_squared() > 1e-18 {
                f.normalize()
            } else {
                n
            };
            if m.dot(n) >= 0.0 && d.dot(m) < 0.0 && best.is_none_or(|b| b.0 > 0.0) {
                return Some((0.0, m));
            }
            continue;
        }
        if let Some(t) = ray_circle_entry(p, d, end, r) {
            if t <= dt && best.is_none_or(|b| t < b.0) {
                let m = (p + d * t - end) / r;
                if m.dot(n) >= 0.0 {
                    best = Some((t, m));
                }
            }
        }
    }
    best
}

/// Sweep against the room box (inward normals).
fn sweep_room(p: Vec2, d: Vec2, r: f64, dt: f64, half: Vec2) -> Option<(f64, Vec2)> {
    let mut best: Option<(f64, Vec2)> = None;
    let planes = [
        (half.x - r - p.x, d.x, Vec2::NEG_X),
        (half.x - r + p.x, -d.x, Vec2::X),
        (half.y - r - p.y, d.y, Vec2::NEG_Y),
        (half.y - r + p.y, -d.y, Vec2::Y),
    ];
    for (s0, vn, n) in planes {
        if vn <= 0.0 {
            continue;
        }
        let t = if s0 <= TOUCH_EPS { 0.0 } else { s0 / vn };
        if t <= dt && best.is_none_or(|b| t < b.0) {
            best = Some((t, n));
        }
    }
    best
}

fn horizontal(v: Vec2) -> Vec3 {
    Vec3::new(v.x, v.y, 0.0)
}

// ----------------------------------------------------------------------------
// Ball sweep
// ----------------------------------------------------------------------------

/// Earliest contact of the ball moving from `ball` for `dt` seconds, if any.
/// Contacts with the room box are always considered; contacts with walls
/// only when `ball.pos.z - radius < wall.height`; sideways contacts with
/// robots only when `ball.pos.z - radius < robot.specs.height` (the rim band
/// above the top plane uses the tilted edge normal), plus landings on robot tops.
pub fn sweep_ball(
    ball: &BallState,
    dt: f64,
    robots: &BTreeMap<RobotId, Robot>,
    walls: &[WallSegment],
    room_half_extents: Vec2,
    params: &BallParams,
) -> Option<BallContact> {
    sweep_ball_except(ball, dt, robots, walls, room_half_extents, params, None)
}

/// [`sweep_ball`] ignoring one robot (the one whose dribbler holds the ball).
pub fn sweep_ball_except(
    ball: &BallState,
    dt: f64,
    robots: &BTreeMap<RobotId, Robot>,
    walls: &[WallSegment],
    room_half_extents: Vec2,
    params: &BallParams,
    skip: Option<RobotId>,
) -> Option<BallContact> {
    let r = params.radius;
    let p = ball.pos_xy();
    let v = ball.vel_xy();
    let z = ball.pos.z;
    let vz = ball.vel.z;
    let bottom = z - r;
    // Deepest centre height reached during the step (ballistic while airborne).
    let z_end = if z > r + TOP_EPS || vz != 0.0 {
        z + vz * dt - 0.5 * GRAVITY * dt * dt
    } else {
        z
    };
    let z_deep = z.min(z_end);
    let mut best: Option<BallContact> = None;
    let mut consider = |c: BallContact| {
        if best.is_none_or(|b| c.time < b.time) {
            best = Some(c);
        }
    };

    for robot in robots.values() {
        if Some(robot.id) == skip {
            continue;
        }
        let h = robot.specs.height;
        let hull = Hull::of(robot);
        let pl = rotate(p - robot.pos, -robot.orientation);
        let dl = rotate(v - robot.vel, -robot.orientation);
        if bottom >= h - TOP_EPS {
            // Above the top: landing on the flat top when descending onto the footprint.
            let dz = (bottom - h).max(0.0);
            if dz == 0.0 && vz >= 0.0 {
                continue; // resting on or leaving the top
            }
            let t = fall_time(dz, vz);
            if t > dt || !hull.contains(pl + dl * t) {
                continue;
            }
            let hit = p + v * t;
            let lever = hit - (robot.pos + robot.vel * t);
            consider(BallContact {
                time: t,
                normal: Vec3::Z,
                surface_velocity: robot.vel + Vec2::new(-lever.y, lever.x) * robot.omega,
                surface: Surface::RobotTop(robot.id),
            });
            continue;
        }
        if z >= h {
            // Rim band: centre above the top plane, underside below it.
            if hull.contains(pl) {
                continue; // on the top: the top is its floor
            }
            let zc = (z_deep.max(h) - h).min(r);
            let r_eff = (r * r - zc * zc).sqrt();
            if let Some((t, n_local, _)) = hull.sweep(pl, dl, r_eff, dt) {
                let n_h = rotate(n_local, robot.orientation);
                let hit = p + v * t;
                let lever = hit - n_h * r_eff - (robot.pos + robot.vel * t);
                consider(BallContact {
                    time: t,
                    normal: Vec3::new(n_h.x * r_eff / r, n_h.y * r_eff / r, zc / r),
                    surface_velocity: robot.vel + Vec2::new(-lever.y, lever.x) * robot.omega,
                    surface: Surface::RobotRim(robot.id),
                });
            }
            continue;
        }
        if let Some((t, n_local, feat)) = hull.sweep(pl, dl, r, dt) {
            let normal = rotate(n_local, robot.orientation);
            let hit = p + v * t;
            // Contact point relative to where the robot is at the contact time.
            let lever = hit - normal * r - (robot.pos + robot.vel * t);
            let surface_velocity = robot.vel + Vec2::new(-lever.y, lever.x) * robot.omega;
            consider(BallContact {
                time: t,
                normal: horizontal(normal),
                surface_velocity,
                surface: match feat {
                    HullFeature::Face => Surface::KickerFace(robot.id),
                    HullFeature::Round => Surface::RobotHull(robot.id),
                },
            });
        }
    }

    for wall in walls {
        if bottom >= wall.height {
            continue;
        }
        if let Some((t, normal)) = sweep_wall(p, v, r, dt, wall) {
            consider(BallContact {
                time: t,
                normal: horizontal(normal),
                surface_velocity: Vec2::ZERO,
                surface: Surface::Wall {
                    is_goal: wall.is_goal,
                },
            });
        }
    }

    if let Some((t, normal)) = sweep_room(p, v, r, dt, room_half_extents) {
        consider(BallContact {
            time: t,
            normal: horizontal(normal),
            surface_velocity: Vec2::ZERO,
            surface: Surface::Room,
        });
    }

    best
}

/// Height [m] of the surface supporting the ball: the top of the tallest
/// robot whose footprint contains the ball centre while the centre is at or
/// above that top, else 0 (the carpet).
pub fn support_height(ball: &BallState, robots: &BTreeMap<RobotId, Robot>) -> f64 {
    let mut floor = 0.0;
    let p = ball.pos_xy();
    for robot in robots.values() {
        let h = robot.specs.height;
        if h <= floor || ball.pos.z < h {
            continue;
        }
        let pl = rotate(p - robot.pos, -robot.orientation);
        if Hull::of(robot).contains(pl) {
            floor = h;
        }
    }
    floor
}

/// Apply the contact response to `ball` and return the impulse [N s] the
/// ball exerted on the surface (for robots).
pub fn resolve_ball_contact(
    ball: &mut BallState,
    contact: &BallContact,
    params: &BallParams,
    contacts: &ContactParams,
) -> Vec3 {
    let n = contact.normal;
    let u = horizontal(contact.surface_velocity);
    let v = ball.vel;
    let un = n.dot(u);
    let vn = n.dot(v);
    if un <= vn {
        return Vec3::ZERO; // receding
    }
    let (damp_n, damp_t) = match contact.surface {
        Surface::RobotHull(_) | Surface::RobotRim(_) => {
            (contacts.ball_robot_normal, contacts.ball_robot_tangent)
        }
        Surface::KickerFace(_) => (contacts.ball_kicker_normal, contacts.ball_kicker_tangent),
        Surface::RobotTop(_) => (
            contacts.ball_robot_top_normal,
            contacts.ball_robot_top_tangent,
        ),
        Surface::Wall { .. } | Surface::Room => {
            (contacts.ball_wall_normal, contacts.ball_wall_tangent)
        }
    };
    let vn_new = un + (un - vn) * (1.0 - damp_n);
    let v_tan = v - n * vn;
    let u_tan = u - n * un;
    let new_vel = n * vn_new + u_tan * damp_t + v_tan * (1.0 - damp_t);
    let impulse = (ball.vel - new_vel) * params.mass;
    ball.vel = new_vel;
    let s = ball.spin;
    let n_xy = contact.normal_xy();
    ball.spin = if n_xy.length_squared() < 1e-24 {
        // Landed on a flat top: rolling on it, like a touchdown on the carpet.
        Vec2::new(new_vel.x, new_vel.y)
    } else {
        // Spin: mirror about the (horizontal) contact plane and damp.
        let m = n_xy.normalize();
        (s - m * (2.0 * s.dot(m))) * contacts.spin_retention
    };
    impulse
}

/// Push the ball out of any surface it is already inside of (start-of-step
/// penetration), preferring the smallest displacement. Returns true if moved.
pub fn depenetrate_ball(
    ball: &mut BallState,
    robots: &BTreeMap<RobotId, Robot>,
    walls: &[WallSegment],
    room_half_extents: Vec2,
    params: &BallParams,
) -> bool {
    depenetrate_ball_except(ball, robots, walls, room_half_extents, params, None)
}

/// [`depenetrate_ball`] ignoring one robot (the one whose dribbler holds the ball).
/// Robot penetrations are relaxed by at most 2 mm per call so a ball released
/// inside the mouth eases out instead of jumping. A ball whose centre is over
/// a robot top is not pushed (the top becomes its floor instead).
pub fn depenetrate_ball_except(
    ball: &mut BallState,
    robots: &BTreeMap<RobotId, Robot>,
    walls: &[WallSegment],
    room_half_extents: Vec2,
    params: &BallParams,
    skip: Option<RobotId>,
) -> bool {
    let r = params.radius;
    let mut moved = false;
    let z = ball.pos.z;
    let bottom = z - r;

    for robot in robots.values() {
        let h = robot.specs.height;
        if Some(robot.id) == skip || bottom >= h - TOP_EPS {
            continue;
        }
        let hull = Hull::of(robot);
        let pl = rotate(ball.pos_xy() - robot.pos, -robot.orientation);
        let (sd, n_local, _) = hull.signed_distance(pl);
        // Rim band: the edge only reaches out to the reduced radius.
        let r_here = if z >= h {
            if sd <= 0.0 {
                continue;
            }
            let zc = z - h;
            (r * r - zc * zc).max(0.0).sqrt()
        } else {
            r
        };
        if sd < r_here - TOUCH_EPS {
            let push = (r_here - sd).min(ROBOT_PUSH_CAP);
            let n = rotate(n_local, robot.orientation);
            ball.pos.x += n.x * push;
            ball.pos.y += n.y * push;
            moved = true;
        }
    }

    for wall in walls {
        if bottom >= wall.height {
            continue;
        }
        let e = wall.b - wall.a;
        let len = e.length();
        if len < 1e-12 {
            continue;
        }
        let e_hat = e / len;
        let p = ball.pos_xy();
        let rel = p - wall.a;
        let s = rel.dot(wall.normal);
        let proj = rel.dot(e_hat);
        let push = if (0.0..=len).contains(&proj) {
            if s < r - TOUCH_EPS && s > -r {
                Some(wall.normal * (r - s))
            } else {
                None
            }
        } else {
            let end = if proj < 0.0 { wall.a } else { wall.b };
            let f = p - end;
            let d = f.length();
            if d < r - TOUCH_EPS && f.dot(wall.normal) >= 0.0 {
                let m = if d > 1e-12 { f / d } else { wall.normal };
                Some(m * (r - d))
            } else {
                None
            }
        };
        if let Some(push) = push {
            ball.pos.x += push.x;
            ball.pos.y += push.y;
            moved = true;
        }
    }

    let lim = room_half_extents - Vec2::splat(r);
    let clamped = Vec2::new(
        ball.pos.x.clamp(-lim.x, lim.x),
        ball.pos.y.clamp(-lim.y, lim.y),
    );
    if clamped != ball.pos_xy() {
        ball.pos.x = clamped.x;
        ball.pos.y = clamped.y;
        moved = true;
    }
    moved
}

// ----------------------------------------------------------------------------
// Robot contacts
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Body {
    id: RobotId,
    pos: Vec2,
    vel: Vec2,
    radius: f64,
    mass: f64,
    /// Kicker heading (unit).
    heading: Vec2,
    /// Distance from the centre to the flat front chord.
    c2d: f64,
    /// Half width of the chord.
    half_width: f64,
    /// Cosine of the mouth half angle (`c2d / radius`).
    cos_theta: f64,
}

impl Body {
    const EMPTY: Body = Body {
        id: RobotId::new(crate::types::Team::Blue, 0),
        pos: Vec2::ZERO,
        vel: Vec2::ZERO,
        radius: 0.0,
        mass: 1.0,
        heading: Vec2::X,
        c2d: 0.0,
        half_width: 0.0,
        cos_theta: 1.0,
    };

    fn of(r: &Robot) -> Body {
        Body {
            id: r.id,
            pos: r.pos,
            vel: r.vel,
            radius: r.specs.radius,
            mass: r.specs.mass.max(1e-6),
            heading: r.heading(),
            c2d: r.specs.center_to_dribbler,
            half_width: r.specs.front_half_width(),
            cos_theta: (r.specs.center_to_dribbler / r.specs.radius).clamp(-1.0, 1.0),
        }
    }

    /// Support function of the cut disc: `max d . (x - pos)` over the hull
    /// for a unit direction `d`. Inside the mouth cone the extreme point is a
    /// chord corner, elsewhere the arc.
    fn support(&self, d: Vec2) -> f64 {
        let dn = d.dot(self.heading);
        if dn > self.cos_theta {
            let dt = (d.y * self.heading.x - d.x * self.heading.y).abs();
            self.c2d * dn + self.half_width * dt
        } else {
            self.radius
        }
    }

    /// World positions of the two chord corners.
    fn corners(&self) -> [Vec2; 2] {
        let t = Vec2::new(-self.heading.y, self.heading.x);
        let f = self.pos + self.heading * self.c2d;
        [f + t * self.half_width, f - t * self.half_width]
    }
}

/// Overlap of `b` into `a` along the unit axis `d` (pointing from `a` toward
/// `b`); `<= 0` proves the hulls are separated.
fn penetration_along(a: &Body, b: &Body, d: Vec2) -> f64 {
    a.support(d) + b.support(-d) - (b.pos - a.pos).dot(d)
}

/// Minimum translation (unit normal from `a` to `b`, depth) separating two
/// cut discs, or `None` if they do not overlap. Separating-axis test over the
/// centre line, both face normals (both signs) and the corner-to-centre axes,
/// which are the normals of every feature pair except corner-corner.
fn cut_disc_overlap(a: &Body, b: &Body) -> Option<(Vec2, f64)> {
    let delta = b.pos - a.pos;
    let dist = delta.length();
    let mut axes = [Vec2::X; 9];
    let mut n = 0;
    axes[n] = if dist > 1e-9 { delta / dist } else { Vec2::X };
    n += 1;
    for h in [a.heading, -a.heading, b.heading, -b.heading] {
        axes[n] = h;
        n += 1;
    }
    for c in a.corners() {
        let v = b.pos - c;
        if v.length_squared() > 1e-18 {
            axes[n] = v.normalize();
            n += 1;
        }
    }
    for c in b.corners() {
        let v = c - a.pos;
        if v.length_squared() > 1e-18 {
            axes[n] = v.normalize();
            n += 1;
        }
    }
    let mut best: Option<(Vec2, f64)> = None;
    for &d in &axes[..n] {
        let pen = penetration_along(a, b, d);
        if pen <= 0.0 {
            return None;
        }
        if best.is_none_or(|(_, p)| pen < p) {
            best = Some((d, pen));
        }
    }
    best
}

/// Resolve robot-robot and robot-wall overlaps and velocities in place.
/// Emits `Event::RobotCollision` for each new robot pair contact (a pair that
/// overlaps while approaching faster than 2 cm/s), at most once per pair per call.
pub fn resolve_robot_contacts(
    robots: &mut BTreeMap<RobotId, Robot>,
    field: &FieldGeometry,
    walls: &[WallSegment],
    contacts: &ContactParams,
    events: &mut Vec<Event>,
) {
    let mut bodies = [Body::EMPTY; MAX_ROBOTS];
    let mut n = 0usize;
    for r in robots.values() {
        if n == MAX_ROBOTS {
            break;
        }
        bodies[n] = Body::of(r);
        n += 1;
    }
    let mut emitted = [0u32; MAX_ROBOTS];
    let e_rr = contacts.robot_robot_restitution.clamp(0.0, 1.0);
    let k_rr = contacts.robot_robot_friction.clamp(0.0, 1.0);
    let e_rw = contacts.robot_wall_restitution.clamp(0.0, 1.0);
    let chord = contacts.robot_hull_chord_contacts;
    let room = field.room_half_extents();

    for _ in 0..ROBOT_ITERATIONS {
        // robot - robot
        for i in 0..n {
            for j in (i + 1)..n {
                let (bi, bj) = (bodies[i], bodies[j]);
                let delta = bj.pos - bi.pos;
                let dist = delta.length();
                let min_d = bi.radius + bj.radius;
                if dist >= min_d {
                    continue; // the cut hulls lie inside the discs
                }
                let (nrm, pen) = if chord {
                    match cut_disc_overlap(&bi, &bj) {
                        Some(x) => x,
                        None => continue,
                    }
                } else {
                    let nrm = if dist > 1e-9 { delta / dist } else { Vec2::X };
                    (nrm, min_d - dist)
                };
                let inv_mi = 1.0 / bi.mass;
                let inv_mj = 1.0 / bj.mass;
                let inv_sum = inv_mi + inv_mj;
                let v_rel = bj.vel - bi.vel;
                let vn = v_rel.dot(nrm);
                if vn < -COLLISION_EVENT_SPEED && emitted[i] & (1 << j) == 0 {
                    emitted[i] |= 1 << j;
                    events.push(Event::RobotCollision { a: bi.id, b: bj.id });
                }
                // positional correction, mass-weighted
                bodies[i].pos -= nrm * (pen * inv_mi / inv_sum);
                bodies[j].pos += nrm * (pen * inv_mj / inv_sum);
                if vn < 0.0 {
                    let jn = -(1.0 + e_rr) * vn / inv_sum;
                    let tangent = Vec2::new(-nrm.y, nrm.x);
                    let vt = v_rel.dot(tangent);
                    let jt = -k_rr * vt / inv_sum;
                    let imp = nrm * jn + tangent * jt;
                    bodies[i].vel -= imp * inv_mi;
                    bodies[j].vel += imp * inv_mj;
                }
            }
        }
        // robot - wall (goal walls included) and room box
        for b in bodies.iter_mut().take(n) {
            for wall in walls {
                let e = wall.b - wall.a;
                let len = e.length();
                if len < 1e-12 {
                    continue;
                }
                let e_hat = e / len;
                let rel = b.pos - wall.a;
                let proj = rel.dot(e_hat).clamp(0.0, len);
                let q = wall.a + e_hat * proj;
                let delta = b.pos - q;
                let dist = delta.length();
                if dist >= b.radius {
                    continue;
                }
                let dir = if dist > 1e-9 {
                    delta / dist
                } else {
                    wall.normal
                };
                if dir.dot(wall.normal) < 0.0 {
                    continue; // centre on the back side: single-sided
                }
                b.pos += dir * (b.radius - dist);
                let vn = b.vel.dot(dir);
                if vn < 0.0 {
                    let tangent = Vec2::new(-dir.y, dir.x);
                    let vt = b.vel.dot(tangent);
                    b.vel = dir * (-e_rw * vn) + tangent * (vt * (1.0 - k_rr));
                }
            }
            let lim = room - Vec2::splat(b.radius);
            if b.pos.x.abs() > lim.x {
                b.pos.x = b.pos.x.clamp(-lim.x, lim.x);
                b.vel.x = 0.0;
            }
            if b.pos.y.abs() > lim.y {
                b.pos.y = b.pos.y.clamp(-lim.y, lim.y);
                b.vel.y = 0.0;
            }
        }
    }

    for b in bodies.iter().take(n) {
        if let Some(r) = robots.get_mut(&b.id) {
            r.pos = b.pos;
            r.vel = b.vel;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::RobotSpecs;
    use crate::types::Team;
    use std::f64::consts::PI;

    fn params() -> BallParams {
        BallParams::default()
    }

    fn ball_at(p: Vec2, v: Vec2) -> BallState {
        BallState {
            pos: Vec3::new(p.x, p.y, params().radius),
            vel: Vec3::new(v.x, v.y, 0.0),
            spin: Vec2::ZERO,
        }
    }

    fn robot(number: u8, pos: Vec2, orientation: f64) -> Robot {
        Robot::new(
            RobotId::new(Team::Blue, number),
            RobotSpecs::default(),
            pos,
            orientation,
        )
    }

    fn robots(list: Vec<Robot>) -> BTreeMap<RobotId, Robot> {
        list.into_iter().map(|r| (r.id, r)).collect()
    }

    fn wall(a: Vec2, b: Vec2, normal: Vec2, height: f64) -> WallSegment {
        WallSegment {
            a,
            b,
            height,
            normal,
            is_goal: false,
        }
    }

    const ROOM: Vec2 = Vec2::new(10.0, 10.0);

    #[test]
    fn receding_contact_has_no_effect() {
        let p = params();
        let c = ContactParams::default();
        let mut ball = ball_at(Vec2::ZERO, Vec2::new(1.0, 0.3));
        ball.spin = Vec2::new(0.5, 0.0);
        let contact = BallContact {
            time: 0.0,
            normal: Vec3::X,
            surface_velocity: Vec2::ZERO,
            surface: Surface::RobotHull(RobotId::new(Team::Blue, 0)),
        };
        let before = ball;
        let j = resolve_ball_contact(&mut ball, &contact, &p, &c);
        assert_eq!(j, Vec3::ZERO);
        assert_eq!(ball, before);
        // surface moving away faster than the ball approaches: also no effect
        let contact = BallContact {
            surface_velocity: Vec2::new(2.0, 0.0),
            normal: Vec3::NEG_X,
            ..contact
        };
        let mut ball = ball_at(Vec2::ZERO, Vec2::new(1.0, 0.0));
        assert_eq!(
            resolve_ball_contact(&mut ball, &contact, &p, &c),
            Vec3::ZERO
        );
    }

    #[test]
    fn kernel_damping_and_impulse() {
        let p = params();
        let c = ContactParams::default();
        let mut ball = ball_at(Vec2::ZERO, Vec2::new(-2.0, 0.5));
        ball.spin = Vec2::new(-2.0, 0.5);
        let contact = BallContact {
            time: 0.0,
            normal: Vec3::X,
            surface_velocity: Vec2::ZERO,
            surface: Surface::Wall { is_goal: false },
        };
        let j = resolve_ball_contact(&mut ball, &contact, &p, &c);
        assert!((ball.vel.x - 2.0 * (1.0 - c.ball_wall_normal)).abs() < 1e-12);
        assert!((ball.vel.y - 0.5).abs() < 1e-12);
        assert!((j.x - (-2.0 - 1.0) * p.mass).abs() < 1e-12);
        // spin mirrored and damped
        assert!((ball.spin.x - 2.0 * c.spin_retention).abs() < 1e-12);
        assert!((ball.spin.y - 0.5 * c.spin_retention).abs() < 1e-12);
        // kicker face with tangential blend toward the moving surface
        let mut ball = ball_at(Vec2::ZERO, Vec2::new(-1.0, 0.0));
        let contact = BallContact {
            surface_velocity: Vec2::new(0.0, 1.0),
            surface: Surface::KickerFace(RobotId::new(Team::Blue, 1)),
            ..contact
        };
        resolve_ball_contact(&mut ball, &contact, &p, &c);
        assert!((ball.vel.y - c.ball_kicker_tangent).abs() < 1e-12);
        assert!((ball.vel.x - (1.0 - c.ball_kicker_normal)).abs() < 1e-12);
        // vertical wall: the vertical velocity is scaled by the tangential blend
        let mut ball = ball_at(Vec2::ZERO, Vec2::new(-1.0, 0.0));
        ball.vel.z = -3.0;
        let contact = BallContact {
            surface_velocity: Vec2::ZERO,
            surface: Surface::KickerFace(RobotId::new(Team::Blue, 1)),
            ..contact
        };
        resolve_ball_contact(&mut ball, &contact, &p, &c);
        assert!((ball.vel.z - (-3.0) * (1.0 - c.ball_kicker_tangent)).abs() < 1e-12);
    }

    #[test]
    fn top_kernel_uses_top_damping_and_blends_horizontal_velocity() {
        let p = params();
        let c = ContactParams::default();
        let mut ball = ball_at(Vec2::ZERO, Vec2::new(0.8, -0.2));
        ball.pos.z = 0.15 + p.radius;
        ball.vel.z = -2.0;
        ball.spin = Vec2::new(5.0, 5.0);
        let contact = BallContact {
            time: 0.0,
            normal: Vec3::Z,
            surface_velocity: Vec2::new(1.0, 0.5),
            surface: Surface::RobotTop(RobotId::new(Team::Blue, 0)),
        };
        let j = resolve_ball_contact(&mut ball, &contact, &p, &c);
        assert!((ball.vel.z - 2.0 * (1.0 - c.ball_robot_top_normal)).abs() < 1e-12);
        let kt = c.ball_robot_top_tangent;
        assert!((ball.vel.x - (kt * 1.0 + (1.0 - kt) * 0.8)).abs() < 1e-12);
        assert!((ball.vel.y - (kt * 0.5 + (1.0 - kt) * -0.2)).abs() < 1e-12);
        // rolling on the top afterwards
        assert_eq!(ball.spin, ball.vel_xy());
        assert!(j.z < 0.0);
        // a rising ball is receding: nothing happens
        let mut up = ball;
        let before = up;
        assert_eq!(resolve_ball_contact(&mut up, &contact, &p, &c), Vec3::ZERO);
        assert_eq!(up, before);
    }

    #[test]
    fn rolling_ball_bounces_off_stationary_robot() {
        let p = params();
        let c = ContactParams::default();
        let r = robot(0, Vec2::ZERO, PI); // facing -x, ball hits the round back
        let map = robots(vec![r]);
        let mut ball = ball_at(Vec2::new(0.5, 0.0), Vec2::new(-2.0, 0.0));
        ball.spin = ball.vel_xy();
        let dt = 0.001;
        let mut t = 0.0;
        let mut hit = None;
        while t < 0.5 {
            if let Some(ct) = sweep_ball(&ball, dt, &map, &[], ROOM, &p) {
                let st = ball;
                ball.pos += st.vel * ct.time;
                let j = resolve_ball_contact(&mut ball, &ct, &p, &c);
                hit = Some((ct, j));
                break;
            }
            ball.pos += ball.vel * dt;
            t += dt;
        }
        let (ct, j) = hit.expect("ball should hit the robot");
        assert!(matches!(ct.surface, Surface::RobotHull(_)));
        assert!((ct.normal - Vec3::X).length() < 1e-9);
        let expected_x = 0.09 + p.radius;
        assert!((ball.pos.x - expected_x).abs() < 1e-6, "x={}", ball.pos.x);
        assert!((ball.vel.x - 2.0 * (1.0 - c.ball_robot_normal)).abs() < 1e-9);
        assert!(j.x < 0.0);
        // spin mirrored: still roughly rolling in the new direction
        assert!(ball.spin.x > 0.0);
    }

    #[test]
    fn hull_dispatch_face_vs_round() {
        let p = params();
        let r = robot(0, Vec2::ZERO, 0.0);
        let map = robots(vec![r.clone()]);
        // straight into the kicker face
        let ball = ball_at(Vec2::new(0.3, 0.0), Vec2::new(-4.0, 0.0));
        let ct = sweep_ball(&ball, 0.1, &map, &[], ROOM, &p).unwrap();
        assert!(matches!(ct.surface, Surface::KickerFace(_)));
        let x_hit = 0.3 - 4.0 * ct.time;
        assert!((x_hit - (0.075 + p.radius)).abs() < 1e-9);
        // slightly off-centre, still on the chord
        let ball = ball_at(Vec2::new(0.3, 0.03), Vec2::new(-4.0, 0.0));
        let ct = sweep_ball(&ball, 0.1, &map, &[], ROOM, &p).unwrap();
        assert!(matches!(ct.surface, Surface::KickerFace(_)));
        // outside the mouth: round hull, contact at R + r
        let ball = ball_at(Vec2::new(0.3, 0.08), Vec2::new(-4.0, 0.0));
        let ct = sweep_ball(&ball, 0.1, &map, &[], ROOM, &p).unwrap();
        assert!(matches!(ct.surface, Surface::RobotHull(_)));
        let hit = ball.pos_xy() + ball.vel_xy() * ct.time;
        let (sd, _, _) = Hull::of(&r).signed_distance(hit);
        assert!((sd - p.radius).abs() < 1e-9);
        // from behind
        let ball = ball_at(Vec2::new(-0.3, 0.0), Vec2::new(4.0, 0.0));
        let ct = sweep_ball(&ball, 0.1, &map, &[], ROOM, &p).unwrap();
        assert!(matches!(ct.surface, Surface::RobotHull(_)));
        assert!((ct.normal - Vec3::NEG_X).length() < 1e-9);
        // height gate: ball flying above the robot passes (horizontally)
        let mut high = ball_at(Vec2::new(0.3, 0.0), Vec2::new(-4.0, 0.0));
        high.pos.z = 0.3;
        assert!(sweep_ball(&high, 0.1, &map, &[], ROOM, &p).is_none());
        // moving away: nothing
        let ball = ball_at(Vec2::new(0.3, 0.0), Vec2::new(4.0, 0.0));
        assert!(sweep_ball(&ball, 0.1, &map, &[], ROOM, &p).is_none());
    }

    #[test]
    fn top_landing_and_rim_geometry() {
        let p = params();
        let r = robot(0, Vec2::ZERO, 0.0);
        let h = r.specs.height;
        let map = robots(vec![r]);
        // falling straight onto the centre of the top
        let mut ball = ball_at(Vec2::new(0.02, -0.03), Vec2::new(0.1, 0.0));
        ball.pos.z = h + p.radius + 0.01;
        ball.vel.z = -1.0;
        let ct = sweep_ball(&ball, 0.1, &map, &[], ROOM, &p).unwrap();
        assert!(matches!(ct.surface, Surface::RobotTop(_)));
        assert_eq!(ct.normal, Vec3::Z);
        let t_expected = (-1.0 + (1.0f64 + 2.0 * GRAVITY * 0.01).sqrt()) / GRAVITY;
        assert!((ct.time - t_expected).abs() < 1e-12);
        // resting on the top: no contact; rising: no contact
        let mut rest = ball;
        rest.pos.z = h + p.radius;
        rest.vel.z = 0.0;
        assert!(sweep_ball(&rest, 0.001, &map, &[], ROOM, &p).is_none());
        rest.vel.z = 3.0;
        assert!(sweep_ball(&rest, 0.001, &map, &[], ROOM, &p).is_none());
        // descending onto the top but landing beyond the footprint: no top contact
        let mut miss = ball;
        miss.pos.x = 0.2;
        assert!(sweep_ball(&miss, 0.1, &map, &[], ROOM, &p).is_none());
        // support height: over the footprint and above the top plane
        assert_eq!(support_height(&rest, &map), h);
        assert_eq!(support_height(&miss, &map), 0.0);
        let mut low = rest;
        low.pos.z = h - 0.001;
        assert_eq!(support_height(&low, &map), 0.0);

        // rim: ball in the band moving horizontally toward the robot's back,
        // one substep away (the effective radius is taken at the deepest
        // point of the step, a few micrometres lower over 1 ms of free fall)
        let zc = 0.5 * p.radius;
        let r_eff = (p.radius * p.radius - zc * zc).sqrt();
        let x0 = -0.09 - r_eff - 0.001;
        let mut rim = ball_at(Vec2::new(x0, 0.0), Vec2::new(2.0, 0.0));
        rim.pos.z = h + zc;
        let ct = sweep_ball(&rim, 0.001, &map, &[], ROOM, &p).unwrap();
        assert!(matches!(ct.surface, Surface::RobotRim(_)));
        // the horizontal sweep radius is the reduced one
        let x_hit = x0 + 2.0 * ct.time;
        assert!((x_hit - (-0.09 - r_eff)).abs() < 1e-5, "x_hit {x_hit}");
        // tilted normal: unit, pointing away from the robot and upward
        assert!((ct.normal.length() - 1.0).abs() < 1e-12);
        assert!(ct.normal.x < 0.0 && ct.normal.z > 0.0);
        assert!((ct.normal.z - zc / p.radius).abs() < 1e-3);
        // a ball skimming at the top plane level: normal is horizontal (sideways contact)
        let mut level = ball_at(
            Vec2::new(-0.09 - p.radius - 0.001, 0.0),
            Vec2::new(2.0, 0.0),
        );
        level.pos.z = h;
        let ct = sweep_ball(&level, 0.001, &map, &[], ROOM, &p).unwrap();
        assert!((ct.normal.z).abs() < 1e-12);
        assert!(matches!(ct.surface, Surface::RobotRim(_)));
        // the rim bounce never adds energy
        let c = ContactParams::default();
        let mut b = rim;
        b.vel.z = -0.5;
        let ct = sweep_ball(&b, 0.001, &map, &[], ROOM, &p).unwrap();
        let e0 = b.vel.length_squared();
        resolve_ball_contact(&mut b, &ct, &p, &c);
        assert!(b.vel.length_squared() <= e0 + 1e-12);
        assert!(b.vel.dot(ct.normal) >= 0.0);
    }

    #[test]
    fn moving_robot_surface_velocity_and_relative_sweep() {
        let p = params();
        let mut r = robot(0, Vec2::ZERO, 0.0);
        r.vel = Vec2::new(1.0, 0.0);
        r.omega = 2.0;
        let map = robots(vec![r]);
        let ball = ball_at(Vec2::new(0.2, 0.0), Vec2::ZERO); // stationary ball, robot drives into it
        let ct = sweep_ball(&ball, 0.2, &map, &[], ROOM, &p).unwrap();
        assert!(matches!(ct.surface, Surface::KickerFace(_)));
        let expected_t = (0.2 - 0.075 - p.radius) / 1.0;
        assert!((ct.time - expected_t).abs() < 1e-9);
        // surface velocity at the face centre: vel + omega x r
        assert!((ct.surface_velocity.x - 1.0).abs() < 1e-9);
        assert!((ct.surface_velocity.y - 2.0 * 0.075).abs() < 1e-9);
    }

    #[test]
    fn wall_sweep_single_sided_and_height_gated() {
        let p = params();
        let w = wall(Vec2::new(1.0, -1.0), Vec2::new(1.0, 1.0), Vec2::NEG_X, 0.1);
        let map = BTreeMap::new();
        let ball = ball_at(Vec2::new(0.5, 0.2), Vec2::new(2.0, 0.0));
        let ct = sweep_ball(&ball, 1.0, &map, &[w], ROOM, &p).unwrap();
        assert!(matches!(ct.surface, Surface::Wall { is_goal: false }));
        assert!((ct.time - (0.5 - p.radius) / 2.0).abs() < 1e-9);
        assert_eq!(ct.normal, Vec3::NEG_X);
        // beyond the segment: miss
        let ball = ball_at(Vec2::new(0.5, 1.5), Vec2::new(2.0, 0.0));
        assert!(sweep_ball(&ball, 1.0, &map, &[w], ROOM, &p).is_none());
        // endpoint cap
        let ball = ball_at(Vec2::new(0.5, 1.01), Vec2::new(2.0, 0.0));
        let ct = sweep_ball(&ball, 1.0, &map, &[w], ROOM, &p).unwrap();
        assert!(ct.normal.x < 0.0 && ct.normal.y > 0.0);
        // from the back side: ignored
        let ball = ball_at(Vec2::new(1.5, 0.0), Vec2::new(-2.0, 0.0));
        assert!(sweep_ball(&ball, 1.0, &map, &[w], ROOM, &p).is_none());
        // chip above the wall passes
        let mut high = ball_at(Vec2::new(0.5, 0.2), Vec2::new(2.0, 0.0));
        high.pos.z = 0.2;
        assert!(sweep_ball(&high, 1.0, &map, &[w], ROOM, &p).is_none());
        // room box
        let ball = ball_at(Vec2::new(9.5, 0.0), Vec2::new(2.0, 0.0));
        let ct = sweep_ball(&ball, 1.0, &map, &[], ROOM, &p).unwrap();
        assert_eq!(ct.surface, Surface::Room);
        assert_eq!(ct.normal, Vec3::NEG_X);
    }

    #[test]
    fn earliest_contact_wins() {
        let p = params();
        let w = wall(Vec2::new(1.0, -1.0), Vec2::new(1.0, 1.0), Vec2::NEG_X, 0.1);
        // robot facing +x: the ball coming from -x hits its round back before the wall
        let map = robots(vec![robot(0, Vec2::new(0.6, 0.0), 0.0)]);
        let ball = ball_at(Vec2::new(0.0, 0.0), Vec2::new(3.0, 0.0));
        let ct = sweep_ball(&ball, 1.0, &map, &[w], ROOM, &p).unwrap();
        assert!(matches!(ct.surface, Surface::RobotHull(_)));
        assert!((ct.time - (0.6 - 0.09 - p.radius) / 3.0).abs() < 1e-9);
    }

    #[test]
    fn depenetration() {
        let p = params();
        let map = robots(vec![robot(0, Vec2::ZERO, 0.0)]);
        // ball 5 mm inside the kicker face: pushed out by at most 2 mm
        let mut ball = ball_at(Vec2::new(0.075 + p.radius - 0.005, 0.0), Vec2::ZERO);
        assert!(depenetrate_ball(&mut ball, &map, &[], ROOM, &p));
        assert!((ball.pos.x - (0.075 + p.radius - 0.003)).abs() < 1e-12);
        for _ in 0..3 {
            depenetrate_ball(&mut ball, &map, &[], ROOM, &p);
        }
        assert!((ball.pos.x - (0.075 + p.radius)).abs() < 1e-12);
        assert!(!depenetrate_ball(&mut ball, &map, &[], ROOM, &p));
        // skipped robot does not push
        let mut held = ball_at(Vec2::new(0.0885, 0.0), Vec2::ZERO);
        assert!(!depenetrate_ball_except(
            &mut held,
            &map,
            &[],
            ROOM,
            &p,
            Some(RobotId::new(Team::Blue, 0))
        ));
        // rim band: only the reduced radius counts, and a ball over the top is left alone
        let h = map.values().next().unwrap().specs.height;
        let zc = 0.8 * p.radius;
        let r_eff = (p.radius * p.radius - zc * zc).sqrt();
        let mut band = ball_at(Vec2::new(-0.09 - r_eff + 0.001, 0.0), Vec2::ZERO);
        band.pos.z = h + zc;
        assert!(depenetrate_ball(&mut band, &map, &[], ROOM, &p));
        assert!((band.pos.x - (-0.09 - r_eff)).abs() < 1e-12);
        let mut over = ball_at(Vec2::new(0.02, 0.0), Vec2::ZERO);
        over.pos.z = h + zc;
        assert!(!depenetrate_ball(&mut over, &map, &[], ROOM, &p));
        let mut above = ball_at(Vec2::new(0.02, 0.0), Vec2::ZERO);
        above.pos.z = h + p.radius + 0.1;
        assert!(!depenetrate_ball(&mut above, &map, &[], ROOM, &p));
        // wall
        let w = wall(Vec2::new(1.0, -1.0), Vec2::new(1.0, 1.0), Vec2::NEG_X, 0.1);
        let mut ball = ball_at(Vec2::new(0.99, 0.0), Vec2::ZERO);
        assert!(depenetrate_ball(
            &mut ball,
            &BTreeMap::new(),
            &[w],
            ROOM,
            &p
        ));
        assert!((ball.pos.x - (1.0 - p.radius)).abs() < 1e-12);
        // room
        let mut ball = ball_at(Vec2::new(11.0, 0.0), Vec2::ZERO);
        assert!(depenetrate_ball(&mut ball, &BTreeMap::new(), &[], ROOM, &p));
        assert!((ball.pos.x - (10.0 - p.radius)).abs() < 1e-12);
    }

    #[test]
    fn sweep_from_inside_reports_immediate_contact_only_when_approaching() {
        let p = params();
        let map = robots(vec![robot(0, Vec2::ZERO, 0.0)]);
        let inside = ball_at(Vec2::new(0.09, 0.0), Vec2::new(-1.0, 0.0));
        let ct = sweep_ball(&inside, 0.001, &map, &[], ROOM, &p).unwrap();
        assert_eq!(ct.time, 0.0);
        assert!(matches!(ct.surface, Surface::KickerFace(_)));
        let leaving = ball_at(Vec2::new(0.09, 0.0), Vec2::new(1.0, 0.0));
        assert!(sweep_ball(&leaving, 0.001, &map, &[], ROOM, &p).is_none());
    }

    #[test]
    fn robot_overlap_separates_without_energy_gain() {
        let c = ContactParams::default();
        let field = FieldGeometry::default();
        // backs toward each other: round vs round, separation R + R
        let mut a = robot(0, Vec2::new(0.0, 0.0), PI);
        let mut b = robot(1, Vec2::new(0.15, 0.0), 0.0);
        a.vel = Vec2::new(1.0, 0.0);
        b.vel = Vec2::new(-1.0, 0.0);
        let mut map = robots(vec![a, b]);
        let e_before = map
            .values()
            .map(|r| 0.5 * r.specs.mass * r.vel.length_squared())
            .sum::<f64>();
        let mut events = Vec::new();
        resolve_robot_contacts(&mut map, &field, &[], &c, &mut events);
        let pa = map[&RobotId::new(Team::Blue, 0)].pos;
        let pb = map[&RobotId::new(Team::Blue, 1)].pos;
        assert!((pb - pa).length() >= 0.18 - 1e-9);
        assert!((pa.x + 0.015).abs() < 1e-9 && (pb.x - 0.165).abs() < 1e-9);
        let e_after = map
            .values()
            .map(|r| 0.5 * r.specs.mass * r.vel.length_squared())
            .sum::<f64>();
        assert!(e_after <= e_before + 1e-12);
        let va = map[&RobotId::new(Team::Blue, 0)].vel;
        let vb = map[&RobotId::new(Team::Blue, 1)].vel;
        assert!(vb.x - va.x >= 0.0);
        assert!((vb.x - va.x - 2.0 * c.robot_robot_restitution).abs() < 1e-9);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Event::RobotCollision { .. }));
        // second call: separated, no further event
        resolve_robot_contacts(&mut map, &field, &[], &c, &mut events);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn chord_contacts_face_to_face_and_face_to_disc() {
        let c = ContactParams::default();
        assert!(c.robot_hull_chord_contacts);
        let field = FieldGeometry::default();
        let c2d = RobotSpecs::default().center_to_dribbler;
        let r = RobotSpecs::default().radius;
        // face to face: separation 2 c2d
        let a = robot(0, Vec2::ZERO, 0.0);
        let b = robot(1, Vec2::new(0.14, 0.0), PI);
        let mut map = robots(vec![a, b]);
        let mut events = Vec::new();
        resolve_robot_contacts(&mut map, &field, &[], &c, &mut events);
        let pa = map[&RobotId::new(Team::Blue, 0)].pos;
        let pb = map[&RobotId::new(Team::Blue, 1)].pos;
        assert!(
            ((pb - pa).length() - 2.0 * c2d).abs() < 1e-9,
            "{}",
            (pb - pa).length()
        );
        assert!((pa.x + 0.005).abs() < 1e-9 && (pb.x - 0.145).abs() < 1e-9);
        // not touching at 2 c2d + 1 mm even though the discs overlap
        let a = robot(0, Vec2::ZERO, 0.0);
        let b = robot(1, Vec2::new(2.0 * c2d + 0.001, 0.0), PI);
        let mut map = robots(vec![a, b]);
        resolve_robot_contacts(&mut map, &field, &[], &c, &mut events);
        assert!((map[&RobotId::new(Team::Blue, 1)].pos.x - (2.0 * c2d + 0.001)).abs() < 1e-12);
        // face to disc (b faces away): separation c2d + R
        let a = robot(0, Vec2::ZERO, 0.0);
        let b = robot(1, Vec2::new(0.15, 0.0), 0.0);
        let mut map = robots(vec![a, b]);
        resolve_robot_contacts(&mut map, &field, &[], &c, &mut events);
        let pa = map[&RobotId::new(Team::Blue, 0)].pos;
        let pb = map[&RobotId::new(Team::Blue, 1)].pos;
        assert!(
            ((pb - pa).length() - (c2d + r)).abs() < 1e-9,
            "{}",
            (pb - pa).length()
        );
        // corner vs disc: a's corner pokes b's side; separated when the corner
        // is clear. The approach direction lies in the corner's normal cone
        // (between the face normal at 0 deg and the arc normal at 33.6 deg).
        let hw = RobotSpecs::default().front_half_width();
        let corner = Vec2::new(c2d, hw);
        let dir = Vec2::from_angle(15f64.to_radians());
        let a = robot(0, Vec2::ZERO, 0.0);
        let b = robot(1, corner + dir * (r + 0.001), PI / 2.0);
        let mut map = robots(vec![a, b]);
        let before = map[&RobotId::new(Team::Blue, 1)].pos;
        resolve_robot_contacts(&mut map, &field, &[], &c, &mut events);
        assert!((map[&RobotId::new(Team::Blue, 1)].pos - before).length() < 1e-12);
        let a = robot(0, Vec2::ZERO, 0.0);
        let b = robot(1, corner + dir * (r - 0.002), PI / 2.0);
        let mut map = robots(vec![a, b]);
        resolve_robot_contacts(&mut map, &field, &[], &c, &mut events);
        let pb = map[&RobotId::new(Team::Blue, 1)].pos;
        let pa = map[&RobotId::new(Team::Blue, 0)].pos;
        let corner_now = pa + corner;
        assert!(((pb - corner_now).length() - r).abs() < 1e-9);
        // discs only when disabled
        let disc_only = ContactParams {
            robot_hull_chord_contacts: false,
            ..c
        };
        let a = robot(0, Vec2::ZERO, 0.0);
        let b = robot(1, Vec2::new(0.14, 0.0), PI);
        let mut map = robots(vec![a, b]);
        resolve_robot_contacts(&mut map, &field, &[], &disc_only, &mut events);
        let pa = map[&RobotId::new(Team::Blue, 0)].pos;
        let pb = map[&RobotId::new(Team::Blue, 1)].pos;
        assert!(((pb - pa).length() - 2.0 * r).abs() < 1e-9);
    }

    #[test]
    fn robot_vs_wall_and_resting_contact_no_event() {
        let c = ContactParams::default();
        let field = FieldGeometry::default();
        let w = wall(Vec2::new(1.0, -1.0), Vec2::new(1.0, 1.0), Vec2::NEG_X, 0.1);
        let mut a = robot(0, Vec2::new(0.95, 0.0), 0.0);
        a.vel = Vec2::new(1.0, 0.5);
        let mut map = robots(vec![a]);
        let mut events = Vec::new();
        resolve_robot_contacts(&mut map, &field, &[w], &c, &mut events);
        let r = &map[&RobotId::new(Team::Blue, 0)];
        assert!((r.pos.x - (1.0 - 0.09)).abs() < 1e-9);
        assert!(r.vel.x <= 0.0 && r.vel.x >= -c.robot_wall_restitution - 1e-9);
        assert!(r.vel.y < 0.5 && r.vel.y > 0.0);
        assert!(events.is_empty());
        // two robots resting in contact, no relative velocity: no event
        let a = robot(0, Vec2::ZERO, PI);
        let b = robot(1, Vec2::new(0.179, 0.0), 0.0);
        let mut map = robots(vec![a, b]);
        resolve_robot_contacts(&mut map, &field, &[], &c, &mut events);
        assert!(events.is_empty());
    }
}

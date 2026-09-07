//! Closed-form ball trajectories (TIGERs Sumatra model, SI units).
//!
//! OWNER: math agent.
//!
//! Flat motion is the spin-driven two-phase model: from state `(v0, s0)` the
//! ground-contact slip is `c = v0 - s0`; while `|c| > eps` the ball slides
//! with acceleration `acc_slide * c_hat`, its spin accelerates by
//! `acc_slide / p`, and the phase switches after
//! `t_sw = |c| * p / (1 + p) / |acc_slide|`, after which it rolls with the
//! speed-dependent deceleration `a(v) = |acc_roll| + roll_speed_coefficient v`
//! (`BallParams::acc_roll_at`) and `spin == vel_xy`. With `b =
//! roll_speed_coefficient` and `a0 = |acc_roll|` the roll phase is
//! `v(t) = (v0 + a0/b) e^{-bt} - a0/b`, `s(t)` analytic, rest after
//! `ln(1 + b v0/a0) / b`; `b == 0` falls back to the constant-deceleration
//! formulas. Chips are ballistic hops with per-bounce damping
//! `(xy_first | xy_other, z_first | z_other)` selected by whether the ball
//! carries spin (first bounce of a kicked ball: no spin), ending when the next
//! apex would be below `min_hop_height`, then the flat model continues.
//!
//! The floor can be raised (`from_state_on`) so the same model runs on top of
//! a robot (`docs/design.md` §5.2, cut-cylinder hull).
//!
//! The trajectory stores its phase boundaries (hop list, switch time, rest
//! time) at construction so that `state_at` is O(1) for the common case
//! (evaluation inside the first hop or in the flat phase). Hops beyond
//! `MAX_HOPS` are evaluated lazily; this only happens with exotic damping
//! values.

use crate::params::BallParams;
use crate::types::{BallState, Vec2, Vec3};
use crate::GRAVITY;

/// Slip speed [m/s] below which the ball counts as rolling (Sumatra: 0.01 mm/s).
const SLIP_EPS: f64 = 1e-5;
/// Number of hops kept in the precomputed table.
const MAX_HOPS: usize = 16;
/// Hard cap on the hop loop (only reachable with `chip_damping_z >= 1`).
const HOP_LOOP_LIMIT: usize = 4096;
/// Below this `b t` the series expansions of the roll-phase kernels are used.
const SERIES_X: f64 = 0.01;

/// `g1(x) = (1 - e^{-x}) / x`, the velocity kernel of the roll phase (1 at 0).
fn g1(x: f64) -> f64 {
    if x < 1e-12 {
        1.0
    } else {
        -(-x).exp_m1() / x
    }
}

/// `g2(x) = (x - 1 + e^{-x}) / x^2`, the distance kernel of the roll phase (1/2 at 0).
fn g2(x: f64) -> f64 {
    if x < SERIES_X {
        0.5 - x
            * (1.0 / 6.0 - x * (1.0 / 24.0 - x * (1.0 / 120.0 - x * (1.0 / 720.0 - x / 5040.0))))
    } else {
        (x + (-x).exp_m1()) / (x * x)
    }
}

/// `ln(1 + z) / z` (1 at 0).
fn ln1p_over(z: f64) -> f64 {
    if z.abs() < 1e-300 {
        1.0
    } else {
        z.ln_1p() / z
    }
}

/// Rolling deceleration law `a(v) = a0 + b v` (magnitudes) along the direction
/// of motion; all formulas are for the speed magnitude.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RollLaw {
    /// Deceleration at zero speed [m/s^2], >= 0.
    a0: f64,
    /// Speed coefficient [1/s], >= 0.
    b: f64,
}

impl RollLaw {
    fn of(params: &BallParams) -> RollLaw {
        RollLaw {
            a0: params.acc_roll.abs(),
            b: params.roll_speed_coefficient.max(0.0),
        }
    }

    /// Speed after rolling for `u` seconds from `v0` (not clamped at rest).
    fn speed(&self, v0: f64, u: f64) -> f64 {
        if self.b == 0.0 {
            v0 - self.a0 * u
        } else {
            let x = self.b * u;
            v0 * (-x).exp() - self.a0 * u * g1(x)
        }
    }

    /// Distance travelled after rolling for `u` seconds from `v0` (`u` must not
    /// exceed the rest time).
    fn distance(&self, v0: f64, u: f64) -> f64 {
        if self.b == 0.0 {
            v0 * u - 0.5 * self.a0 * u * u
        } else {
            let x = self.b * u;
            v0 * u * g1(x) - self.a0 * u * u * g2(x)
        }
    }

    /// Time until the speed drops from `v0` to `s` (0 if `s >= v0`, infinity
    /// if never).
    fn time_by_speed(&self, v0: f64, s: f64) -> f64 {
        if s >= v0 {
            return 0.0;
        }
        if self.b == 0.0 {
            return if self.a0 > 0.0 {
                (v0 - s) / self.a0
            } else {
                f64::INFINITY
            };
        }
        // ln((v0 + a0/b) / (s + a0/b)) / b, written without cancellation.
        let denom = self.b * s + self.a0;
        if denom <= 0.0 {
            return f64::INFINITY;
        }
        let z = self.b * (v0 - s) / denom;
        (v0 - s) / denom * ln1p_over(z)
    }

    /// Time until the ball rests from `v0`. Without a constant term (`a0 == 0`)
    /// the exponential decay never stops, so rest is declared at `rest_speed`.
    fn rest_time(&self, v0: f64, rest_speed: f64) -> f64 {
        if v0 <= 0.0 {
            0.0
        } else if self.a0 > 0.0 {
            self.time_by_speed(v0, 0.0)
        } else if self.b > 0.0 && rest_speed > 0.0 {
            self.time_by_speed(v0, rest_speed.min(v0))
        } else {
            f64::INFINITY
        }
    }

    /// Distance travelled until rest from `v0`.
    fn stopping_distance(&self, v0: f64, rest_speed: f64) -> f64 {
        let u = self.rest_time(v0, rest_speed);
        if u.is_finite() {
            self.distance(v0, u)
        } else {
            f64::INFINITY
        }
    }

    /// Time to travel `d` from speed `v0` given the rest time `u_rest`, or
    /// `None` if the ball stops first. Newton from below on the concave
    /// distance function (monotone convergence), bisection as a safeguard.
    fn time_by_distance(&self, v0: f64, d: f64, u_rest: f64) -> Option<f64> {
        if d <= 0.0 {
            return Some(0.0);
        }
        if v0 <= 0.0 {
            return None;
        }
        if self.b == 0.0 {
            if self.a0 <= 0.0 {
                return Some(d / v0);
            }
            let disc = v0 * v0 - 2.0 * self.a0 * d;
            if disc < 0.0 {
                return None;
            }
            return Some((v0 - disc.sqrt()) / self.a0);
        }
        let s_max = if u_rest.is_finite() {
            self.distance(v0, u_rest)
        } else {
            v0 / self.b // a0 == 0: the exponential decay converges to v0 / b
        };
        let tol = 1e-13 * d.max(1.0);
        if d >= s_max {
            return (d - s_max <= tol && u_rest.is_finite()).then_some(u_rest);
        }
        let mut lo: f64 = 0.0;
        let mut hi = if u_rest.is_finite() {
            u_rest
        } else {
            // The exponential tail: bound by the time the speed drops to 1e-12 of v0.
            self.time_by_speed(v0, v0 * 1e-12)
        };
        let mut u = d / v0;
        for _ in 0..60 {
            let f = self.distance(v0, u) - d;
            if f.abs() <= tol {
                return Some(u);
            }
            if f < 0.0 {
                lo = lo.max(u);
            } else {
                hi = hi.min(u);
            }
            let v = self.speed(v0, u);
            let next = if v > 0.0 { u - f / v } else { f64::NAN };
            u = if next.is_finite() && next > lo && next < hi {
                next
            } else {
                0.5 * (lo + hi)
            };
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if hi - lo <= 1e-15 * hi.max(1.0) {
                return Some(mid);
            }
            if self.distance(v0, mid) < d {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }
}

/// One ballistic hop: `p(t') = pos + vel t' - g/2 t'^2 z`, `0 <= t' <= duration`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Hop {
    t_start: f64,
    duration: f64,
    pos: Vec3,
    vel: Vec3,
    spin: Vec2,
}

impl Hop {
    const ZERO: Hop = Hop {
        t_start: 0.0,
        duration: 0.0,
        pos: Vec3::ZERO,
        vel: Vec3::ZERO,
        spin: Vec2::ZERO,
    };

    /// Build a hop starting at `t_start` from an airborne state (`pos.z >= ground`,
    /// `ground` = floor height + ball radius).
    fn new(t_start: f64, pos: Vec3, vel: Vec3, spin: Vec2, ground: f64) -> Hop {
        let h = (pos.z - ground).max(0.0);
        let vz = vel.z;
        // z(t) = ground + h + vz t - g/2 t^2 = ground  =>  t = (vz + sqrt(vz^2 + 2 g h)) / g
        let duration = (vz + (vz * vz + 2.0 * GRAVITY * h).sqrt()) / GRAVITY;
        Hop {
            t_start,
            duration: duration.max(0.0),
            pos,
            vel,
            spin,
        }
    }

    fn state_at(&self, t: f64, ground: f64) -> BallState {
        let tau = t.clamp(0.0, self.duration);
        let mut pos = self.pos + self.vel * tau;
        pos.z -= 0.5 * GRAVITY * tau * tau;
        if tau >= self.duration {
            pos.z = ground;
        }
        let vel = Vec3::new(self.vel.x, self.vel.y, self.vel.z - GRAVITY * tau);
        BallState {
            pos,
            vel,
            spin: self.spin,
        }
    }

    /// Touchdown position (horizontal).
    fn touchdown(&self) -> Vec2 {
        Vec2::new(
            self.pos.x + self.vel.x * self.duration,
            self.pos.y + self.vel.y * self.duration,
        )
    }

    /// Vertical speed at touchdown (negative).
    fn vz_touchdown(&self) -> f64 {
        self.vel.z - GRAVITY * self.duration
    }

    /// State right after the bounce at the end of this hop: `Ok(next_hop)` if
    /// the ball leaves the floor again, `Err(grounded_state)` otherwise. A ball
    /// without spin is on its first bounce (kicked ball) and gets the
    /// first-hop damping in both axes.
    fn bounce(&self, params: &BallParams, ground: f64) -> Result<Hop, (f64, Vec3, Vec2)> {
        let td = self.touchdown();
        let t_td = self.t_start + self.duration;
        let first = self.spin.length_squared() < 1e-24;
        let (damp_xy, damp_z) = if first {
            (params.chip_damping_xy_first_hop, params.chip_damping_z)
        } else {
            (
                params.chip_damping_xy_other_hops,
                params.chip_damping_z_other_hops,
            )
        };
        let vz = -self.vz_touchdown() * damp_z;
        let vel = Vec3::new(self.vel.x * damp_xy, self.vel.y * damp_xy, vz.max(0.0));
        let spin = Vec2::new(vel.x, vel.y);
        let pos = Vec3::new(td.x, td.y, ground);
        if vz * vz / (2.0 * GRAVITY) > params.min_hop_height && vz > 0.0 {
            Ok(Hop::new(t_td, pos, vel, spin, ground))
        } else {
            Err((t_td, Vec3::new(vel.x, vel.y, 0.0), spin))
        }
    }
}

/// The two-phase flat model starting at `t_start`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Flat {
    t_start: f64,
    pos: Vec2,
    v0: Vec2,
    s0: Vec2,
    /// Slide acceleration (zero when rolling from the start).
    a_slide: Vec2,
    /// Spin acceleration during the slide (`a_slide / p`), applied as `s = s0 - a_slide_spin t`.
    a_slide_spin: Vec2,
    /// Switch time relative to `t_start`.
    t_switch: f64,
    p_switch: Vec2,
    v_switch: Vec2,
    /// Unit direction of the roll phase (zero when at rest at the switch).
    roll_dir: Vec2,
    /// Speed at the start of the roll phase.
    roll_v0: f64,
    roll: RollLaw,
    /// Rest time relative to `t_start`.
    t_rest: f64,
    /// Rest position (equals `p_switch` when the ball never rests).
    p_rest: Vec2,
}

impl Flat {
    fn new(t_start: f64, pos: Vec2, v0: Vec2, s0: Vec2, params: &BallParams) -> Flat {
        let p = params.inertia_distribution.max(1e-6);
        let a_s = params.acc_slide.abs();
        let roll = RollLaw::of(params);
        let c = v0 - s0;
        let c_len = c.length();

        let (a_slide, a_slide_spin, t_switch, v_switch, p_switch);
        if c_len < SLIP_EPS || a_s <= 0.0 {
            a_slide = Vec2::ZERO;
            a_slide_spin = Vec2::ZERO;
            t_switch = 0.0;
            v_switch = v0;
            p_switch = pos;
        } else {
            let dir = c / c_len;
            a_slide = -dir * a_s;
            a_slide_spin = a_slide / p;
            t_switch = c_len * p / ((1.0 + p) * a_s);
            v_switch = v0 + a_slide * t_switch;
            p_switch = pos + v0 * t_switch + a_slide * (0.5 * t_switch * t_switch);
        }

        let at_rest_now = t_switch == 0.0 && v0.length() < params.rest_speed;
        let v_sw_len = v_switch.length();
        let (roll_dir, roll_v0, t_rest) = if at_rest_now || v_sw_len < 1e-12 {
            (Vec2::ZERO, 0.0, t_switch)
        } else {
            let u_rest = roll.rest_time(v_sw_len, params.rest_speed);
            (v_switch / v_sw_len, v_sw_len, t_switch + u_rest)
        };
        let t_rest = if at_rest_now { 0.0 } else { t_rest };
        let p_rest = if t_rest.is_finite() && roll_v0 > 0.0 {
            p_switch + roll_dir * roll.distance(roll_v0, t_rest - t_switch)
        } else {
            p_switch
        };

        Flat {
            t_start,
            pos,
            v0,
            s0,
            a_slide,
            a_slide_spin,
            t_switch,
            p_switch,
            v_switch,
            roll_dir,
            roll_v0,
            roll,
            t_rest,
            p_rest,
        }
    }

    fn eval(&self, t: f64) -> (Vec2, Vec2, Vec2) {
        let tau = (t - self.t_start).max(0.0);
        if tau >= self.t_rest {
            (self.p_rest, Vec2::ZERO, Vec2::ZERO)
        } else if tau < self.t_switch {
            let p = self.pos + self.v0 * tau + self.a_slide * (0.5 * tau * tau);
            let v = self.v0 + self.a_slide * tau;
            let s = self.s0 - self.a_slide_spin * tau;
            (p, v, s)
        } else {
            let u = tau - self.t_switch;
            let s = self.roll.distance(self.roll_v0, u);
            let v = self.roll.speed(self.roll_v0, u).max(0.0);
            let p = self.p_switch + self.roll_dir * s;
            let vel = self.roll_dir * v;
            (p, vel, vel)
        }
    }

    /// Time (relative to `t_start`) to travel `d` along the path, 1-D
    /// approximation of the slide phase (exact when `v0`, `s0` are collinear).
    fn time_by_distance(&self, d: f64) -> Option<f64> {
        if d <= 0.0 {
            return Some(0.0);
        }
        let d_sw = (self.p_switch - self.pos).length();
        if self.t_switch > 0.0 && d <= d_sw {
            let v = self.v0.length();
            let a = if v > 1e-12 {
                self.a_slide.dot(self.v0) / v
            } else {
                self.a_slide.length()
            };
            if a.abs() < 1e-12 {
                return if v > 1e-12 { Some(d / v) } else { None };
            }
            let disc = v * v + 2.0 * a * d;
            if disc < 0.0 {
                return None;
            }
            let t = (-v + disc.sqrt()) / a;
            return if t >= 0.0 {
                Some(t.min(self.t_switch))
            } else {
                None
            };
        }
        let d2 = d - if self.t_switch > 0.0 { d_sw } else { 0.0 };
        if self.roll_v0 <= 0.0 {
            return (d2 <= 0.0).then_some(self.t_switch);
        }
        self.roll
            .time_by_distance(self.roll_v0, d2, self.t_rest - self.t_switch)
            .map(|u| self.t_switch + u)
    }

    /// Time (relative to `t_start`) at which the speed drops to `s`.
    fn time_by_speed(&self, s: f64) -> Option<f64> {
        if self.v0.length() <= s {
            return None;
        }
        if self.t_switch > 0.0 {
            let a2 = self.a_slide.length_squared();
            let b = 2.0 * self.v0.dot(self.a_slide);
            let c = self.v0.length_squared() - s * s;
            let disc = b * b - 4.0 * a2 * c;
            if disc >= 0.0 && a2 > 0.0 {
                let t = (-b - disc.sqrt()) / (2.0 * a2);
                if (0.0..=self.t_switch).contains(&t) {
                    return Some(t);
                }
            }
        }
        if self.roll_v0 <= s {
            return Some(self.t_switch);
        }
        let u = self.roll.time_by_speed(self.roll_v0, s);
        u.is_finite().then_some(self.t_switch + u)
    }
}

/// A trajectory constructed from a state; evaluable at any `t >= 0`.
#[derive(Debug, Clone)]
pub struct BallTrajectory {
    params: BallParams,
    initial: BallState,
    hops: [Hop; MAX_HOPS],
    n_hops: usize,
    /// Total number of hops (may exceed `MAX_HOPS`; the rest are evaluated lazily).
    total_hops: usize,
    flat: Flat,
    /// Height of the supporting surface [m] (0 = the carpet).
    floor: f64,
    /// Ball centre height when resting on the floor (`floor + radius`).
    ground: f64,
}

impl BallTrajectory {
    /// Build the trajectory starting at `state` on the carpet.
    pub fn from_state(state: &BallState, params: &BallParams) -> Self {
        Self::from_state_on(state, params, 0.0)
    }

    /// Build the trajectory starting at `state` with the supporting surface at
    /// height `floor_z` (a robot top, for a ball inside its footprint).
    pub fn from_state_on(state: &BallState, params: &BallParams, floor_z: f64) -> Self {
        let r = params.radius;
        let floor = if floor_z.is_finite() {
            floor_z.max(0.0)
        } else {
            0.0
        };
        let ground = floor + r;
        let mut st = *state;
        // Negative-z guard: the ball can never be below the floor.
        if st.pos.z < ground {
            st.pos.z = ground;
            if st.vel.z < 0.0 {
                st.vel.z = 0.0;
            }
        }
        for v in [
            &mut st.pos.x,
            &mut st.pos.y,
            &mut st.vel.x,
            &mut st.vel.y,
            &mut st.vel.z,
        ] {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        let mut airborne = st.pos.z > ground + 1e-9 || st.vel.z > 0.0;
        if airborne && st.pos.z <= ground + 1e-9 {
            // Kicked up from the floor: only counts as a hop above min_hop_height.
            let apex = st.vel.z * st.vel.z / (2.0 * GRAVITY);
            if apex <= params.min_hop_height {
                airborne = false;
            }
        }
        if !airborne {
            st.vel.z = 0.0;
            st.pos.z = ground;
        }

        let mut hops = [Hop::ZERO; MAX_HOPS];
        let mut n_hops = 0usize;
        let mut total_hops = 0usize;
        let flat;
        if airborne {
            let mut hop = Hop::new(0.0, st.pos, st.vel, st.spin, ground);
            loop {
                if n_hops < MAX_HOPS {
                    hops[n_hops] = hop;
                    n_hops += 1;
                }
                total_hops += 1;
                match hop.bounce(params, ground) {
                    Ok(next) if total_hops < HOP_LOOP_LIMIT => hop = next,
                    Ok(next) => {
                        // Damping >= 1: force the ball down.
                        let td = next.pos;
                        flat = Flat::new(
                            next.t_start,
                            Vec2::new(td.x, td.y),
                            Vec2::new(next.vel.x, next.vel.y),
                            next.spin,
                            params,
                        );
                        break;
                    }
                    Err((t, vel, spin)) => {
                        let td = hop.touchdown();
                        flat = Flat::new(t, td, Vec2::new(vel.x, vel.y), spin, params);
                        break;
                    }
                }
            }
        } else {
            flat = Flat::new(0.0, st.pos_xy(), st.vel_xy(), st.spin, params);
        }

        Self {
            params: *params,
            initial: st,
            hops,
            n_hops,
            total_hops,
            flat,
            floor,
            ground,
        }
    }

    /// Initial state (after the floor guard).
    pub fn initial(&self) -> &BallState {
        &self.initial
    }

    /// Height of the supporting surface [m].
    pub fn floor(&self) -> f64 {
        self.floor
    }

    /// Hop `i` (lazily recomputed beyond the stored table).
    fn hop(&self, i: usize) -> Option<Hop> {
        if i >= self.total_hops {
            return None;
        }
        if i < self.n_hops {
            return Some(self.hops[i]);
        }
        let mut h = self.hops[self.n_hops - 1];
        for _ in self.n_hops..=i {
            h = h.bounce(&self.params, self.ground).ok()?;
        }
        Some(h)
    }

    /// State at time `t` [s] after the start (clamped to rest).
    pub fn state_at(&self, t: f64) -> BallState {
        let t = if t.is_finite() { t.max(0.0) } else { f64::MAX };
        if t < self.flat.t_start {
            // Find the hop containing t (usually the first).
            let mut i = 0;
            while let Some(h) = self.hop(i) {
                if t < h.t_start + h.duration || i + 1 == self.total_hops {
                    return h.state_at(t - h.t_start, self.ground);
                }
                i += 1;
            }
        }
        let (p, v, s) = self.flat.eval(t);
        BallState {
            pos: Vec3::new(p.x, p.y, self.ground),
            vel: Vec3::new(v.x, v.y, 0.0),
            spin: s,
        }
    }

    /// Time until the ball comes to rest [s].
    pub fn time_to_rest(&self) -> f64 {
        self.flat.t_start + self.flat.t_rest
    }

    /// Time [s] at which the ball touches the floor for the last time and the
    /// flat model takes over (0 for a flat trajectory).
    pub fn time_to_ground(&self) -> f64 {
        self.flat.t_start
    }

    /// True if the trajectory has an airborne phase.
    pub fn is_chipped(&self) -> bool {
        self.total_hops > 0
    }

    /// Time [s] at which the ball has travelled `distance` [m] (horizontal path
    /// length), or `None` if it stops first.
    pub fn time_by_distance(&self, distance: f64) -> Option<f64> {
        if distance <= 0.0 {
            return Some(0.0);
        }
        let mut acc = 0.0;
        let mut i = 0;
        while let Some(h) = self.hop(i) {
            let vxy = Vec2::new(h.vel.x, h.vel.y).length();
            let len = vxy * h.duration;
            if acc + len >= distance {
                if vxy < 1e-12 {
                    return None;
                }
                return Some(h.t_start + (distance - acc) / vxy);
            }
            acc += len;
            i += 1;
        }
        self.flat
            .time_by_distance(distance - acc)
            .map(|t| t + self.flat.t_start)
    }

    /// Time [s] at which the horizontal speed drops to `speed` [m/s], or `None`
    /// if it is already slower.
    pub fn time_by_speed(&self, speed: f64) -> Option<f64> {
        if self.initial.vel_xy().length() <= speed {
            return None;
        }
        let mut i = 0;
        while let Some(h) = self.hop(i) {
            match h.bounce(&self.params, self.ground) {
                Ok(next) => {
                    if Vec2::new(next.vel.x, next.vel.y).length() <= speed {
                        return Some(next.t_start);
                    }
                }
                Err((t, vel, _)) => {
                    if Vec2::new(vel.x, vel.y).length() <= speed {
                        return Some(t);
                    }
                }
            }
            i += 1;
        }
        self.flat
            .time_by_speed(speed)
            .map(|t| t + self.flat.t_start)
    }

    /// Horizontal position of the ball at rest.
    pub fn rest_position(&self) -> Vec2 {
        self.state_at(self.time_to_rest()).pos_xy()
    }

    /// Touchdown points of a chipped trajectory (empty for flat).
    pub fn touchdowns(&self) -> Vec<Vec2> {
        let mut out = Vec::with_capacity(self.total_hops);
        let mut i = 0;
        while let Some(h) = self.hop(i) {
            out.push(h.touchdown());
            i += 1;
        }
        out
    }
}

/// Inverse models used by clients (and by tests).
pub mod inverse {
    use super::{BallParams, RollLaw};
    use crate::GRAVITY;

    /// Slide-phase distance of a straight kick at `v0` without spin.
    fn slide_distance(v0: f64, params: &BallParams) -> f64 {
        let k = params.k_switch();
        let a_s = params.acc_slide.abs().max(1e-9);
        v0 * v0 * (1.0 - k * k) / (2.0 * a_s)
    }

    /// Distance [m] a rolling (no slip) ball at speed `v` travels before it rests.
    pub fn roll_stopping_distance(v: f64, params: &BallParams) -> f64 {
        let law = RollLaw::of(params);
        if law.a0 <= 0.0 && law.b <= 0.0 {
            return f64::INFINITY;
        }
        law.stopping_distance(v.max(0.0), params.rest_speed)
    }

    /// Total distance [m] of a straight kick at `v0` without spin.
    pub fn straight_distance(v0: f64, params: &BallParams) -> f64 {
        let v0 = v0.max(0.0);
        slide_distance(v0, params) + roll_stopping_distance(params.k_switch() * v0, params)
    }

    /// Distance [m] travelled by a straight kick at `v0` until its speed has
    /// dropped to `end_speed` (0 if it starts slower).
    pub fn straight_distance_until_speed(v0: f64, end_speed: f64, params: &BallParams) -> f64 {
        let v0 = v0.max(0.0);
        let e = end_speed.max(0.0);
        if e >= v0 {
            return 0.0;
        }
        let k = params.k_switch();
        if e >= k * v0 {
            let a_s = params.acc_slide.abs().max(1e-9);
            return (v0 * v0 - e * e) / (2.0 * a_s);
        }
        slide_distance(v0, params) + roll_stopping_distance(k * v0, params)
            - roll_stopping_distance(e, params)
    }

    /// `d = v0^2 * K` for the constant-deceleration roll (`roll_speed_coefficient == 0`).
    fn distance_factor(params: &BallParams) -> f64 {
        let k = params.k_switch();
        let a_s = params.acc_slide.abs().max(1e-9);
        let a_r = params.acc_roll.abs().max(1e-9);
        (1.0 - k * k) / (2.0 * a_s) + k * k / (2.0 * a_r)
    }

    /// Smallest `v >= lo` with `f(v) >= target` for a non-decreasing `f`
    /// (bracket by doubling, then bisection to machine precision).
    fn solve_increasing(f: impl Fn(f64) -> f64, target: f64, lo: f64) -> f64 {
        let mut lo = lo.max(0.0);
        if f(lo) >= target {
            return lo;
        }
        let mut hi = lo.max(1.0);
        let mut n = 0;
        while f(hi) < target {
            hi *= 2.0;
            n += 1;
            if n > 64 || !hi.is_finite() {
                return f64::INFINITY;
            }
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if hi - lo <= 1e-15 * hi || mid <= lo || mid >= hi {
                break;
            }
            if f(mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Straight kick speed [m/s] so that the ball stops after `distance` [m].
    pub fn straight_speed_for_distance(distance: f64, params: &BallParams) -> f64 {
        let d = distance.max(0.0);
        if params.roll_speed_coefficient <= 0.0 {
            return (d / distance_factor(params)).sqrt();
        }
        solve_increasing(|v| straight_distance(v, params), d, 0.0)
    }

    /// Straight kick speed [m/s] so that the ball still moves at `end_speed`
    /// after `distance`.
    pub fn straight_speed_for_end_speed(distance: f64, end_speed: f64, params: &BallParams) -> f64 {
        let d = distance.max(0.0);
        let e = end_speed.max(0.0);
        if params.roll_speed_coefficient <= 0.0 {
            let k = params.k_switch();
            let a_s = params.acc_slide.abs().max(1e-9);
            let a_r = params.acc_roll.abs().max(1e-9);
            // Assume the end speed is reached in the roll phase.
            let v0 = ((d + e * e / (2.0 * a_r)) / distance_factor(params)).sqrt();
            return if e <= k * v0 {
                v0
            } else {
                // End speed reached while still sliding.
                (2.0 * a_s * d + e * e).sqrt()
            };
        }
        solve_increasing(|v| straight_distance_until_speed(v, e, params), d, e)
    }

    /// Chip kick speed [m/s] at elevation `angle_rad` so the `n`-th touchdown
    /// (0 = first) lands at `distance` [m]. Ignores the `min_hop_height` cutoff.
    pub fn chip_speed_for_touchdown(
        distance: f64,
        angle_rad: f64,
        n: usize,
        params: &BallParams,
    ) -> f64 {
        let mut f = 1.0;
        let mut dxy = 1.0;
        let mut dz = 1.0;
        for i in 1..=n {
            let (kxy, kz) = if i == 1 {
                (params.chip_damping_xy_first_hop, params.chip_damping_z)
            } else {
                (
                    params.chip_damping_xy_other_hops,
                    params.chip_damping_z_other_hops,
                )
            };
            dxy *= kxy;
            dz *= kz;
            f += dxy * dz;
        }
        let (s, c) = angle_rad.sin_cos();
        let denom = 2.0 * f * s.abs() * c.abs();
        if denom < 1e-12 {
            return 0.0;
        }
        (distance.max(0.0) * GRAVITY / denom).sqrt()
    }

    /// Chip kick speed [m/s] at elevation `angle_rad` reaching apex `height` [m].
    pub fn chip_speed_for_height(height: f64, angle_rad: f64) -> f64 {
        let s = angle_rad.sin().abs();
        if s < 1e-12 {
            return 0.0;
        }
        (2.0 * GRAVITY * height.max(0.0)).sqrt() / s
    }
}

#[cfg(test)]
mod tests {
    use super::inverse;
    use super::*;

    fn params() -> BallParams {
        BallParams::default()
    }

    /// Constant-deceleration variant of the defaults.
    fn params_const() -> BallParams {
        BallParams {
            roll_speed_coefficient: 0.0,
            ..BallParams::default()
        }
    }

    /// Event-located numerical integration of the same ODE: exact slide phase
    /// (piecewise constant acceleration), RK4 on `dv/dt = -(a0 + b v)`,
    /// `ds/dt = v` in the roll phase with the rest instant bisected.
    fn integrate_flat(
        p0: Vec2,
        v0: Vec2,
        s0: Vec2,
        t_end: f64,
        params: &BallParams,
    ) -> (Vec2, Vec2, Vec2) {
        let p_i = params.inertia_distribution;
        let a_s = params.acc_slide.abs();
        let a_r = params.acc_roll.abs();
        let b = params.roll_speed_coefficient.max(0.0);
        let (mut p, mut v, mut s) = (p0, v0, s0);
        let mut t = 0.0;
        let h: f64 = 1e-4;
        let mut rolling = (v - s).length() < SLIP_EPS;
        let mut rest = rolling && v.length() < params.rest_speed;
        // One RK4 step of length `dt` from speed `vl`: (new speed, distance).
        let rk4 = |vl: f64, dt: f64| -> (f64, f64) {
            let f = |x: f64| -(a_r + b * x);
            let k1 = f(vl);
            let k2 = f(vl + 0.5 * dt * k1);
            let k3 = f(vl + 0.5 * dt * k2);
            let k4 = f(vl + dt * k3);
            let v1 = vl + 0.5 * dt * k1;
            let v2 = vl + 0.5 * dt * k2;
            let v3 = vl + dt * k3;
            (
                vl + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
                dt / 6.0 * (vl + 2.0 * v1 + 2.0 * v2 + v3),
            )
        };
        while t < t_end - 1e-15 {
            if rest {
                break;
            }
            let mut dt = h.min(t_end - t);
            if !rolling {
                let c = v - s;
                let dir = c.normalize();
                let a = -dir * a_s;
                let a_spin = -a / p_i; // ds/dt
                                       // slip magnitude decreases at rate a_s (1 + 1/p)
                let t_sw = c.length() / (a_s * (1.0 + 1.0 / p_i));
                let hit = t_sw <= dt;
                if hit {
                    dt = t_sw;
                }
                p += v * dt + a * (0.5 * dt * dt);
                v += a * dt;
                s += a_spin * dt;
                if hit {
                    s = v;
                    rolling = true;
                }
            } else {
                let vl = v.length();
                if vl < 1e-12 {
                    rest = true;
                    continue;
                }
                let dir = v / vl;
                let (mut v_new, mut ds) = rk4(vl, dt);
                if v_new <= 0.0 && a_r > 0.0 {
                    // locate the rest instant inside this step
                    let (mut lo, mut hi) = (0.0, dt);
                    for _ in 0..80 {
                        let mid = 0.5 * (lo + hi);
                        if rk4(vl, mid).0 > 0.0 {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    dt = 0.5 * (lo + hi);
                    let r = rk4(vl, dt);
                    v_new = 0.0;
                    ds = r.1;
                    rest = true;
                }
                p += dir * ds;
                v = dir * v_new;
                s = v;
            }
            t += dt;
        }
        (p, v, s)
    }

    fn flat_state(p: Vec2, v: Vec2, s: Vec2, params: &BallParams) -> BallState {
        BallState {
            pos: Vec3::new(p.x, p.y, params.radius),
            vel: Vec3::new(v.x, v.y, 0.0),
            spin: s,
        }
    }

    fn check_against_integration(params: &BallParams) {
        let cases = [
            (Vec2::new(4.0, 0.0), Vec2::ZERO),
            (Vec2::new(2.0, 1.0), Vec2::ZERO),
            (Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0)), // rolling
            (Vec2::new(3.0, 0.0), Vec2::new(-1.0, 0.0)), // backspin
            (Vec2::new(0.0, 0.0), Vec2::new(1.5, 0.5)), // spinning at rest
            (Vec2::new(1.0, 2.0), Vec2::new(0.5, -0.5)), // non-collinear
            (Vec2::new(0.3, 0.0), Vec2::new(0.3, 0.0)),
        ];
        for (v0, s0) in cases {
            let p0 = Vec2::new(0.1, -0.2);
            let traj = BallTrajectory::from_state(&flat_state(p0, v0, s0, params), params);
            for &t in &[0.001, 0.05, 0.2, 0.5, 1.0, 2.5, 5.0, 12.0] {
                let st = traj.state_at(t);
                let (p, v, s) = integrate_flat(p0, v0, s0, t, params);
                assert!(
                    (st.pos_xy() - p).length() < 1e-6,
                    "b={} v0={v0} s0={s0} t={t}: pos {} vs {p}",
                    params.roll_speed_coefficient,
                    st.pos_xy()
                );
                assert!(
                    (st.vel_xy() - v).length() < 1e-6,
                    "b={} v0={v0} s0={s0} t={t}: vel {} vs {v}",
                    params.roll_speed_coefficient,
                    st.vel_xy()
                );
                assert!((st.spin - s).length() < 1e-6, "v0={v0} s0={s0} t={t}: spin");
            }
        }
    }

    #[test]
    fn flat_matches_numeric_integration() {
        // default (speed-dependent roll), constant fallback, and a strong drag
        check_against_integration(&params());
        check_against_integration(&params_const());
        check_against_integration(&BallParams {
            roll_speed_coefficient: 0.135,
            acc_roll: -0.238,
            ..params()
        });
    }

    #[test]
    fn roll_law_closed_form_identities() {
        let law = RollLaw { a0: 0.22, b: 0.045 };
        let c = law.a0 / law.b;
        for v0 in [0.3, 1.0, 2.5, 4.0] {
            let u_rest = law.rest_time(v0, 0.01);
            assert!((u_rest - (1.0 + law.b * v0 / law.a0).ln() / law.b).abs() < 1e-12);
            assert!(law.speed(v0, u_rest).abs() < 1e-12);
            for u in [0.0, 0.1, 0.5, u_rest * 0.9] {
                let v_ref = (v0 + c) * (-law.b * u).exp() - c;
                let s_ref = (v0 + c) * (1.0 - (-law.b * u).exp()) / law.b - c * u;
                assert!((law.speed(v0, u) - v_ref).abs() < 1e-12);
                assert!((law.distance(v0, u) - s_ref).abs() < 1e-12);
                // time_by_speed inverts speed
                assert!((law.time_by_speed(v0, v_ref) - u).abs() < 1e-10);
                // time_by_distance inverts distance
                let t = law.time_by_distance(v0, s_ref, u_rest).expect("reachable");
                assert!((t - u).abs() < 1e-9, "u={u} t={t}");
            }
            let s_stop = law.stopping_distance(v0, 0.01);
            assert!((s_stop - (v0 / law.b - c / law.b * (1.0 + v0 / c).ln())).abs() < 1e-12);
            assert!(law.time_by_distance(v0, s_stop * 1.001, u_rest).is_none());
            // b -> 0 continuity: tiny b agrees with the constant law
            let tiny = RollLaw { a0: 0.22, b: 1e-9 };
            let cst = RollLaw { a0: 0.22, b: 0.0 };
            assert!((tiny.distance(v0, 1.0) - cst.distance(v0, 1.0)).abs() < 1e-8);
            assert!((tiny.speed(v0, 1.0) - cst.speed(v0, 1.0)).abs() < 1e-8);
            assert!((tiny.rest_time(v0, 0.01) - cst.rest_time(v0, 0.01)).abs() < 1e-6);
        }
        // pure exponential decay rests at rest_speed
        let exp_only = RollLaw { a0: 0.0, b: 0.5 };
        let u = exp_only.rest_time(2.0, 0.01);
        assert!((exp_only.speed(2.0, u) - 0.01).abs() < 1e-12);
        assert!(exp_only.time_by_distance(2.0, 1.0, u).is_some());
        assert!(exp_only.time_by_distance(2.0, 5.0, u).is_none());
    }

    #[test]
    fn zero_velocity_stays_put() {
        let params = params();
        let traj = BallTrajectory::from_state(
            &flat_state(Vec2::new(1.0, 2.0), Vec2::ZERO, Vec2::ZERO, &params),
            &params,
        );
        assert_eq!(traj.time_to_rest(), 0.0);
        let st = traj.state_at(3.0);
        assert_eq!(st.pos, Vec3::new(1.0, 2.0, params.radius));
        assert_eq!(st.vel, Vec3::ZERO);
        assert!(!traj.is_chipped());
    }

    #[test]
    fn rest_clamp_and_rest_position() {
        for params in [params(), params_const()] {
            let v0 = Vec2::new(2.0, 0.0);
            let traj = BallTrajectory::from_state(
                &flat_state(Vec2::ZERO, v0, Vec2::ZERO, &params),
                &params,
            );
            let k = params.k_switch();
            let a_s = params.acc_slide.abs();
            let a0 = params.acc_roll.abs();
            let b = params.roll_speed_coefficient;
            let vs = k * v0.x;
            let roll = if b > 0.0 {
                vs / b - a0 / (b * b) * (1.0 + b * vs / a0).ln()
            } else {
                vs * vs / (2.0 * a0)
            };
            let expected = v0.x * v0.x * (1.0 - k * k) / (2.0 * a_s) + roll;
            assert!(
                (traj.rest_position().x - expected).abs() < 1e-9,
                "b={b}: {} vs {expected}",
                traj.rest_position().x
            );
            assert!((inverse::straight_distance(v0.x, &params) - expected).abs() < 1e-9);
            let after = traj.state_at(traj.time_to_rest() + 10.0);
            assert_eq!(after.vel, Vec3::ZERO);
            assert_eq!(after.spin, Vec2::ZERO);
            assert!((after.pos_xy() - traj.rest_position()).length() < 1e-12);
            // speed is exactly zero at the rest time and monotone before it
            let mut prev = f64::INFINITY;
            let mut t = 0.0;
            while t < traj.time_to_rest() {
                let v = traj.state_at(t).vel_xy().length();
                assert!(v <= prev + 1e-12);
                prev = v;
                t += 0.01;
            }
            // slow ball snaps to rest immediately
            let slow = BallTrajectory::from_state(
                &flat_state(
                    Vec2::ZERO,
                    Vec2::new(0.005, 0.0),
                    Vec2::new(0.005, 0.0),
                    &params,
                ),
                &params,
            );
            assert_eq!(slow.time_to_rest(), 0.0);
        }
    }

    #[test]
    fn time_by_distance_and_speed_consistent() {
        for params in [params(), params_const()] {
            for (v0, s0) in [
                (Vec2::new(3.0, 1.0), Vec2::ZERO),
                (Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0)),
                (Vec2::new(2.0, 0.0), Vec2::new(0.5, 0.0)),
            ] {
                let traj =
                    BallTrajectory::from_state(&flat_state(Vec2::ZERO, v0, s0, &params), &params);
                let total = traj.rest_position().length();
                for frac in [0.05, 0.3, 0.7, 0.99, 0.9999] {
                    let d = total * frac;
                    let t = traj.time_by_distance(d).expect("reachable");
                    let st = traj.state_at(t);
                    assert!(
                        (st.pos_xy().length() - d).abs() < 1e-9,
                        "b={} d={d}: {}",
                        params.roll_speed_coefficient,
                        st.pos_xy().length()
                    );
                }
                assert!(traj.time_by_distance(total + 0.01).is_none());
                for frac in [0.9, 0.5, 0.2, 0.05] {
                    let s = v0.length() * frac;
                    let t = traj.time_by_speed(s).expect("speed reachable");
                    let st = traj.state_at(t);
                    assert!((st.vel_xy().length() - s).abs() < 1e-9, "s={s}");
                }
                assert!(traj.time_by_speed(v0.length() + 0.1).is_none());
            }
        }
    }

    #[test]
    fn chip_touchdown_round_trip() {
        let params = params();
        let angle = 45f64.to_radians();
        for (dist, n) in [(2.0, 0), (3.5, 0), (2.0, 1), (4.0, 2)] {
            let v = inverse::chip_speed_for_touchdown(dist, angle, n, &params);
            let st = BallState {
                pos: Vec3::new(0.0, 0.0, params.radius),
                vel: Vec3::new(v * angle.cos(), 0.0, v * angle.sin()),
                spin: Vec2::ZERO,
            };
            let traj = BallTrajectory::from_state(&st, &params);
            assert!(traj.is_chipped());
            let tds = traj.touchdowns();
            assert!(tds.len() > n, "hops: {}", tds.len());
            assert!(
                (tds[n].x - dist).abs() < 1e-9,
                "n={n}: {} vs {dist}",
                tds[n].x
            );
            // apex height matches the height inverse
            let apex_t = v * angle.sin() / GRAVITY;
            let h = traj.state_at(apex_t).pos.z - params.radius;
            let v_h = inverse::chip_speed_for_height(h, angle);
            assert!((v_h - v).abs() < 1e-9);
        }
    }

    #[test]
    fn chip_phases_and_edges() {
        let params = params();
        // kicked from the floor: first hop uses first-hop damping, later hops other-hop damping
        let st = BallState {
            pos: Vec3::new(0.0, 0.0, params.radius),
            vel: Vec3::new(3.0, 0.0, 3.0),
            spin: Vec2::ZERO,
        };
        let traj = BallTrajectory::from_state(&st, &params);
        let t1 = 2.0 * 3.0 / GRAVITY;
        let just_after = traj.state_at(t1 + 1e-6);
        assert!((just_after.vel.x - 3.0 * params.chip_damping_xy_first_hop).abs() < 1e-6);
        let vz1 = 3.0 * params.chip_damping_z;
        assert!((just_after.vel.z - vz1).abs() < 1e-4);
        assert!((just_after.spin.x - just_after.vel.x).abs() < 1e-9);
        let t2 = t1 + 2.0 * vz1 / GRAVITY;
        let after2 = traj.state_at(t2 + 1e-6);
        let vx2 = 3.0 * params.chip_damping_xy_first_hop * params.chip_damping_xy_other_hops;
        assert!((after2.vel.x - vx2).abs() < 1e-6);
        let vz2 = vz1 * params.chip_damping_z_other_hops;
        assert!((after2.vel.z - vz2).abs() < 1e-4, "vz2 {}", after2.vel.z);
        // z never below the radius
        let mut t = 0.0;
        while t < traj.time_to_rest() {
            assert!(traj.state_at(t).pos.z >= params.radius - 1e-12);
            t += 0.01;
        }
        let end = traj.state_at(traj.time_to_rest() + 1.0);
        assert_eq!(end.vel, Vec3::ZERO);
        assert_eq!(end.pos.z, params.radius);
        // number of hops finite and ends when apex < min_hop_height
        assert!(traj.touchdowns().len() < 20);

        // starting in the air with downward velocity
        let st = BallState {
            pos: Vec3::new(0.0, 0.0, 0.5),
            vel: Vec3::new(1.0, 0.0, -1.0),
            spin: Vec2::ZERO,
        };
        let traj = BallTrajectory::from_state(&st, &params);
        assert!(traj.is_chipped());
        let td = traj.touchdowns()[0];
        let dur = (-1.0 + (1.0f64 + 2.0 * GRAVITY * (0.5 - params.radius)).sqrt()) / GRAVITY;
        assert!((td.x - dur).abs() < 1e-9);
        let mid = traj.state_at(dur * 0.5);
        assert!(mid.pos.z > params.radius && mid.pos.z < 0.5);

        // negative z guard
        let st = BallState {
            pos: Vec3::new(0.0, 0.0, -0.1),
            vel: Vec3::new(1.0, 0.0, -2.0),
            spin: Vec2::ZERO,
        };
        let traj = BallTrajectory::from_state(&st, &params);
        assert!(!traj.is_chipped());
        assert_eq!(traj.state_at(0.0).pos.z, params.radius);
        assert_eq!(traj.state_at(0.1).vel.z, 0.0);

        // tiny upward velocity from the floor is grounded immediately
        let st = BallState {
            pos: Vec3::new(0.0, 0.0, params.radius),
            vel: Vec3::new(1.0, 0.0, 0.1),
            spin: Vec2::ZERO,
        };
        assert!(!BallTrajectory::from_state(&st, &params).is_chipped());

        // spin is kept while airborne
        let st = BallState {
            pos: Vec3::new(0.0, 0.0, 0.3),
            vel: Vec3::new(1.0, 0.0, 0.0),
            spin: Vec2::new(0.7, 0.2),
        };
        let traj = BallTrajectory::from_state(&st, &params);
        assert_eq!(traj.state_at(0.05).spin, Vec2::new(0.7, 0.2));
        // ... and a spinning ball gets other-hop damping (both axes) on its first bounce
        let h0 = traj.hop(0).unwrap();
        let td_t = h0.duration;
        let after = traj.state_at(td_t + 1e-7);
        assert!((after.vel.x - params.chip_damping_xy_other_hops).abs() < 1e-6);
        let vz_in = -h0.vz_touchdown();
        assert!((after.vel.z - vz_in * params.chip_damping_z_other_hops).abs() < 1e-5);
    }

    #[test]
    fn chip_z_damping_per_hop() {
        let params = BallParams {
            chip_damping_z: 0.6,
            chip_damping_z_other_hops: 0.3,
            chip_damping_xy_first_hop: 1.0,
            chip_damping_xy_other_hops: 1.0,
            min_hop_height: 0.001,
            ..params()
        };
        let st = BallState {
            pos: Vec3::new(0.0, 0.0, params.radius),
            vel: Vec3::new(1.0, 0.0, 4.0),
            spin: Vec2::ZERO,
        };
        let traj = BallTrajectory::from_state(&st, &params);
        let hops = traj.touchdowns().len();
        assert!(hops >= 4, "hops {hops}");
        let mut vz = 4.0;
        for i in 0..4 {
            let h = traj.hop(i).unwrap();
            assert!((h.vel.z - vz).abs() < 1e-9, "hop {i}: {} vs {vz}", h.vel.z);
            let damp = if i == 0 {
                params.chip_damping_z
            } else {
                params.chip_damping_z_other_hops
            };
            vz *= damp;
        }
    }

    #[test]
    fn raised_floor_runs_the_same_model_on_a_robot_top() {
        let params = params();
        let floor = 0.15;
        // dropped onto the raised floor: lands at z = floor + r and bounces
        let st = BallState {
            pos: Vec3::new(0.0, 0.0, 0.5),
            vel: Vec3::new(0.5, 0.0, 0.0),
            spin: Vec2::ZERO,
        };
        let traj = BallTrajectory::from_state_on(&st, &params, floor);
        assert_eq!(traj.floor(), floor);
        let dur = (2.0 * (0.5 - floor - params.radius) / GRAVITY).sqrt();
        let td = traj.state_at(dur);
        assert!((td.pos.z - (floor + params.radius)).abs() < 1e-12);
        let after = traj.state_at(dur + 1e-6);
        assert!(after.vel.z > 0.0);
        let mut t = 0.0;
        while t < traj.time_to_rest() {
            assert!(traj.state_at(t).pos.z >= floor + params.radius - 1e-12);
            t += 0.005;
        }
        let end = traj.state_at(traj.time_to_rest() + 1.0);
        assert_eq!(end.pos.z, floor + params.radius);
        assert_eq!(end.vel, Vec3::ZERO);
        // rolling on the raised floor matches rolling on the carpet, shifted in z
        let rolling = BallState {
            pos: Vec3::new(0.0, 0.0, floor + params.radius),
            vel: Vec3::new(1.0, 0.0, 0.0),
            spin: Vec2::new(1.0, 0.0),
        };
        let up = BallTrajectory::from_state_on(&rolling, &params, floor);
        let flat =
            BallTrajectory::from_state(&flat_state(Vec2::ZERO, Vec2::X, Vec2::X, &params), &params);
        assert!(!up.is_chipped());
        let (a, b) = (up.state_at(0.7), flat.state_at(0.7));
        assert_eq!(a.pos_xy(), b.pos_xy());
        assert_eq!(a.vel, b.vel);
        assert_eq!(a.pos.z, floor + params.radius);
        // a ball below the raised floor is lifted onto it
        let below = BallState {
            pos: Vec3::new(0.0, 0.0, floor + 0.5 * params.radius),
            vel: Vec3::new(0.0, 0.0, -1.0),
            spin: Vec2::ZERO,
        };
        let lifted = BallTrajectory::from_state_on(&below, &params, floor);
        assert_eq!(lifted.state_at(0.0).pos.z, floor + params.radius);
        assert_eq!(lifted.state_at(0.0).vel.z, 0.0);
    }

    #[test]
    fn chip_time_by_distance_consistent() {
        let params = params();
        let st = BallState {
            pos: Vec3::new(0.0, 0.0, params.radius),
            vel: Vec3::new(3.0, 0.0, 3.0),
            spin: Vec2::ZERO,
        };
        let traj = BallTrajectory::from_state(&st, &params);
        for d in [0.5, 1.5, 2.2, 3.0] {
            let t = traj.time_by_distance(d).unwrap();
            assert!((traj.state_at(t).pos.x - d).abs() < 1e-9);
        }
        let t = traj.time_by_speed(2.0).unwrap();
        assert!(traj.state_at(t).vel_xy().length() <= 2.0 + 1e-9);
        assert!(traj.state_at(t - 1e-6).vel_xy().length() > 2.0);
    }

    #[test]
    fn straight_inverses_round_trip() {
        for params in [params(), params_const()] {
            for d in [0.5, 2.0, 6.0] {
                let v = inverse::straight_speed_for_distance(d, &params);
                let traj = BallTrajectory::from_state(
                    &flat_state(Vec2::ZERO, Vec2::new(v, 0.0), Vec2::ZERO, &params),
                    &params,
                );
                assert!(
                    (traj.rest_position().x - d).abs() < 1e-9,
                    "b={} d={d}: {}",
                    params.roll_speed_coefficient,
                    traj.rest_position().x
                );
                for e in [0.2, 1.0, 3.0] {
                    let v = inverse::straight_speed_for_end_speed(d, e, &params);
                    let traj = BallTrajectory::from_state(
                        &flat_state(Vec2::ZERO, Vec2::new(v, 0.0), Vec2::ZERO, &params),
                        &params,
                    );
                    let t = traj.time_by_distance(d).unwrap();
                    assert!(
                        (traj.state_at(t).vel.x - e).abs() < 1e-6,
                        "b={} d={d} e={e}: {}",
                        params.roll_speed_coefficient,
                        traj.state_at(t).vel.x
                    );
                }
            }
        }
        // the constant-roll closed form and the generic solver agree when b == 0
        let p = params_const();
        let a = inverse::straight_speed_for_distance(3.0, &p);
        let via_solver = inverse::straight_distance(a, &p);
        assert!((via_solver - 3.0).abs() < 1e-9);
    }

    #[test]
    fn many_hops_overflow_is_lazy_but_consistent() {
        let mut params = params();
        params.chip_damping_z = 0.97;
        params.chip_damping_z_other_hops = 0.97;
        params.chip_damping_xy_other_hops = 0.99;
        let st = BallState {
            pos: Vec3::new(0.0, 0.0, params.radius),
            vel: Vec3::new(1.0, 0.0, 5.0),
            spin: Vec2::ZERO,
        };
        let traj = BallTrajectory::from_state(&st, &params);
        assert!(traj.total_hops > MAX_HOPS);
        let tds = traj.touchdowns();
        assert_eq!(tds.len(), traj.total_hops);
        // evaluate beyond the stored table: continuous in time
        let h = traj.hop(MAX_HOPS + 2).unwrap();
        let a = traj.state_at(h.t_start - 1e-9);
        let b = traj.state_at(h.t_start + 1e-9);
        assert!((a.pos_xy() - b.pos_xy()).length() < 1e-6);
        assert!(traj.time_to_rest().is_finite());
    }
}

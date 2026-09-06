//! Closed-form ball trajectories (TIGERs Sumatra model, SI units).
//!
//! OWNER: math agent.
//!
//! Flat motion is the spin-driven two-phase model: from state `(v0, s0)` the
//! ground-contact slip is `c = v0 - s0`; while `|c| > eps` the ball slides
//! with acceleration `acc_slide * c_hat`, its spin accelerates by
//! `acc_slide / p`, and the phase switches after
//! `t_sw = |c| * p / (1 + p) / |acc_slide|`, after which it rolls with
//! `acc_roll * v_hat` and `spin == vel_xy`. Rest time is analytic. Chips are
//! ballistic hops with per-bounce damping `(xy_first | xy_other, xy, z)`,
//! ending when the next apex would be below `min_hop_height`, then the flat
//! model continues.
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

    /// Build a hop starting at `t_start` from an airborne state (`pos.z >= r`).
    fn new(t_start: f64, pos: Vec3, vel: Vec3, spin: Vec2, radius: f64) -> Hop {
        let h = (pos.z - radius).max(0.0);
        let vz = vel.z;
        // z(t) = r + h + vz t - g/2 t^2 = r  =>  t = (vz + sqrt(vz^2 + 2 g h)) / g
        let duration = (vz + (vz * vz + 2.0 * GRAVITY * h).sqrt()) / GRAVITY;
        Hop {
            t_start,
            duration: duration.max(0.0),
            pos,
            vel,
            spin,
        }
    }

    fn state_at(&self, t: f64, radius: f64) -> BallState {
        let tau = t.clamp(0.0, self.duration);
        let mut pos = self.pos + self.vel * tau;
        pos.z -= 0.5 * GRAVITY * tau * tau;
        if tau >= self.duration {
            pos.z = radius;
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
    /// the ball leaves the floor again, `Err(grounded_state)` otherwise.
    fn bounce(&self, params: &BallParams) -> Result<Hop, (f64, Vec3, Vec2)> {
        let td = self.touchdown();
        let t_td = self.t_start + self.duration;
        let damp_xy = if self.spin.length_squared() < 1e-24 {
            params.chip_damping_xy_first_hop
        } else {
            params.chip_damping_xy_other_hops
        };
        let vz = -self.vz_touchdown() * params.chip_damping_z;
        let vel = Vec3::new(self.vel.x * damp_xy, self.vel.y * damp_xy, vz.max(0.0));
        let spin = Vec2::new(vel.x, vel.y);
        let pos = Vec3::new(td.x, td.y, params.radius);
        if vz * vz / (2.0 * GRAVITY) > params.min_hop_height && vz > 0.0 {
            Ok(Hop::new(t_td, pos, vel, spin, params.radius))
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
    a_roll: Vec2,
    /// Rest time relative to `t_start`.
    t_rest: f64,
}

impl Flat {
    fn new(t_start: f64, pos: Vec2, v0: Vec2, s0: Vec2, params: &BallParams) -> Flat {
        let p = params.inertia_distribution.max(1e-6);
        let a_s = params.acc_slide.abs();
        let a_r = params.acc_roll.abs();
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
        let (a_roll, t_rest) = if at_rest_now || v_sw_len < 1e-12 {
            (Vec2::ZERO, t_switch)
        } else if a_r <= 0.0 {
            (Vec2::ZERO, f64::INFINITY)
        } else {
            (-v_switch / v_sw_len * a_r, t_switch + v_sw_len / a_r)
        };
        let t_rest = if at_rest_now { 0.0 } else { t_rest };

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
            a_roll,
            t_rest,
        }
    }

    fn eval(&self, t: f64) -> (Vec2, Vec2, Vec2) {
        let tau = (t - self.t_start).max(0.0);
        if tau >= self.t_rest {
            let u = (self.t_rest - self.t_switch).max(0.0);
            let p = if self.t_rest.is_finite() {
                self.p_switch + self.v_switch * u + self.a_roll * (0.5 * u * u)
            } else {
                self.p_switch
            };
            (p, Vec2::ZERO, Vec2::ZERO)
        } else if tau < self.t_switch {
            let p = self.pos + self.v0 * tau + self.a_slide * (0.5 * tau * tau);
            let v = self.v0 + self.a_slide * tau;
            let s = self.s0 - self.a_slide_spin * tau;
            (p, v, s)
        } else {
            let u = tau - self.t_switch;
            let p = self.p_switch + self.v_switch * u + self.a_roll * (0.5 * u * u);
            let v = self.v_switch + self.a_roll * u;
            (p, v, v)
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
        let vs = self.v_switch.length();
        let a_r = self.a_roll.length();
        if a_r < 1e-12 {
            return if vs > 1e-12 {
                Some(self.t_switch + d2 / vs)
            } else {
                None
            };
        }
        let disc = vs * vs - 2.0 * a_r * d2;
        if disc < 0.0 {
            return None;
        }
        Some(self.t_switch + (vs - disc.sqrt()) / a_r)
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
        let vs = self.v_switch.length();
        if vs <= s {
            return Some(self.t_switch);
        }
        let a_r = self.a_roll.length();
        if a_r < 1e-12 {
            return None;
        }
        Some(self.t_switch + (vs - s) / a_r)
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
}

impl BallTrajectory {
    /// Build the trajectory starting at `state`.
    pub fn from_state(state: &BallState, params: &BallParams) -> Self {
        let r = params.radius;
        let mut st = *state;
        // Negative-z guard: the ball can never be below the floor.
        if st.pos.z < r {
            st.pos.z = r;
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
        let mut airborne = st.pos.z > r + 1e-9 || st.vel.z > 0.0;
        if airborne && st.pos.z <= r + 1e-9 {
            // Kicked up from the floor: only counts as a hop above min_hop_height.
            let apex = st.vel.z * st.vel.z / (2.0 * GRAVITY);
            if apex <= params.min_hop_height {
                airborne = false;
            }
        }
        if !airborne {
            st.vel.z = 0.0;
            st.pos.z = r;
        }

        let mut hops = [Hop::ZERO; MAX_HOPS];
        let mut n_hops = 0usize;
        let mut total_hops = 0usize;
        let flat;
        if airborne {
            let mut hop = Hop::new(0.0, st.pos, st.vel, st.spin, r);
            loop {
                if n_hops < MAX_HOPS {
                    hops[n_hops] = hop;
                    n_hops += 1;
                }
                total_hops += 1;
                match hop.bounce(params) {
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
        }
    }

    /// Initial state (after the floor guard).
    pub fn initial(&self) -> &BallState {
        &self.initial
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
            h = h.bounce(&self.params).ok()?;
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
                    return h.state_at(t - h.t_start, self.params.radius);
                }
                i += 1;
            }
        }
        let (p, v, s) = self.flat.eval(t);
        BallState {
            pos: Vec3::new(p.x, p.y, self.params.radius),
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
            match h.bounce(&self.params) {
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
    use super::BallParams;
    use crate::GRAVITY;

    /// `d = v0^2 * K` for a straight kick without spin.
    fn distance_factor(params: &BallParams) -> f64 {
        let k = params.k_switch();
        let a_s = params.acc_slide.abs().max(1e-9);
        let a_r = params.acc_roll.abs().max(1e-9);
        (1.0 - k * k) / (2.0 * a_s) + k * k / (2.0 * a_r)
    }

    /// Straight kick speed [m/s] so that the ball stops after `distance` [m].
    pub fn straight_speed_for_distance(distance: f64, params: &BallParams) -> f64 {
        (distance.max(0.0) / distance_factor(params)).sqrt()
    }

    /// Straight kick speed [m/s] so that the ball still moves at `end_speed`
    /// after `distance`.
    pub fn straight_speed_for_end_speed(distance: f64, end_speed: f64, params: &BallParams) -> f64 {
        let k = params.k_switch();
        let a_s = params.acc_slide.abs().max(1e-9);
        let a_r = params.acc_roll.abs().max(1e-9);
        let d = distance.max(0.0);
        let e = end_speed.max(0.0);
        // Assume the end speed is reached in the roll phase.
        let v0 = ((d + e * e / (2.0 * a_r)) / distance_factor(params)).sqrt();
        if e <= k * v0 {
            v0
        } else {
            // End speed reached while still sliding.
            (2.0 * a_s * d + e * e).sqrt()
        }
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
            dxy *= if i == 1 {
                params.chip_damping_xy_first_hop
            } else {
                params.chip_damping_xy_other_hops
            };
            dz *= params.chip_damping_z;
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

    /// Event-located numerical integration of the same ODE (piecewise
    /// constant acceleration, exact switch/rest location).
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
        let (mut p, mut v, mut s) = (p0, v0, s0);
        let mut t = 0.0;
        let h: f64 = 1e-4;
        let mut rolling = (v - s).length() < SLIP_EPS;
        let mut rest = rolling && v.length() < params.rest_speed;
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
                let a = -v / vl * a_r;
                let t_rest = vl / a_r;
                let hit = t_rest <= dt;
                if hit {
                    dt = t_rest;
                }
                p += v * dt + a * (0.5 * dt * dt);
                v += a * dt;
                s = v;
                if hit {
                    v = Vec2::ZERO;
                    s = Vec2::ZERO;
                    rest = true;
                }
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

    #[test]
    fn flat_matches_numeric_integration() {
        let params = params();
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
            let traj = BallTrajectory::from_state(&flat_state(p0, v0, s0, &params), &params);
            for &t in &[0.001, 0.05, 0.2, 0.5, 1.0, 2.5, 5.0] {
                let st = traj.state_at(t);
                let (p, v, s) = integrate_flat(p0, v0, s0, t, &params);
                assert!(
                    (st.pos_xy() - p).length() < 1e-6,
                    "v0={v0} s0={s0} t={t}: pos {} vs {p}",
                    st.pos_xy()
                );
                assert!(
                    (st.vel_xy() - v).length() < 1e-6,
                    "v0={v0} s0={s0} t={t}: vel"
                );
                assert!((st.spin - s).length() < 1e-6, "v0={v0} s0={s0} t={t}: spin");
            }
        }
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
        let params = params();
        let v0 = Vec2::new(2.0, 0.0);
        let traj =
            BallTrajectory::from_state(&flat_state(Vec2::ZERO, v0, Vec2::ZERO, &params), &params);
        let k = params.k_switch();
        let expected = v0.x * v0.x * ((1.0 - k * k) / 6.0 + k * k / 0.6);
        assert!((traj.rest_position().x - expected).abs() < 1e-9);
        let after = traj.state_at(traj.time_to_rest() + 10.0);
        assert_eq!(after.vel, Vec3::ZERO);
        assert_eq!(after.spin, Vec2::ZERO);
        assert!((after.pos_xy() - traj.rest_position()).length() < 1e-12);
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

    #[test]
    fn time_by_distance_and_speed_consistent() {
        let params = params();
        for (v0, s0) in [
            (Vec2::new(3.0, 1.0), Vec2::ZERO),
            (Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0)),
            (Vec2::new(2.0, 0.0), Vec2::new(0.5, 0.0)),
        ] {
            let traj =
                BallTrajectory::from_state(&flat_state(Vec2::ZERO, v0, s0, &params), &params);
            let total = traj.rest_position().length();
            for frac in [0.05, 0.3, 0.7, 0.99] {
                let d = total * frac;
                let t = traj.time_by_distance(d).expect("reachable");
                let st = traj.state_at(t);
                assert!((st.pos_xy().length() - d).abs() < 1e-9, "d={d}");
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
        assert!((just_after.vel.z - 1.5).abs() < 1e-4);
        assert!((just_after.spin.x - just_after.vel.x).abs() < 1e-9);
        let t2 = t1 + 2.0 * 1.5 / GRAVITY;
        let after2 = traj.state_at(t2 + 1e-6);
        let vx2 = 3.0 * params.chip_damping_xy_first_hop * params.chip_damping_xy_other_hops;
        assert!((after2.vel.x - vx2).abs() < 1e-6);
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
        // ... and a spinning ball gets other-hop damping on its first bounce
        let td_t = traj.hop(0).unwrap().duration;
        let after = traj.state_at(td_t + 1e-7);
        assert!((after.vel.x - params.chip_damping_xy_other_hops).abs() < 1e-6);
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
        let params = params();
        for d in [0.5, 2.0, 6.0] {
            let v = inverse::straight_speed_for_distance(d, &params);
            let traj = BallTrajectory::from_state(
                &flat_state(Vec2::ZERO, Vec2::new(v, 0.0), Vec2::ZERO, &params),
                &params,
            );
            assert!((traj.rest_position().x - d).abs() < 1e-9);
            for e in [0.2, 1.0, 3.0] {
                let v = inverse::straight_speed_for_end_speed(d, e, &params);
                let traj = BallTrajectory::from_state(
                    &flat_state(Vec2::ZERO, Vec2::new(v, 0.0), Vec2::ZERO, &params),
                    &params,
                );
                let t = traj.time_by_distance(d).unwrap();
                assert!((traj.state_at(t).vel.x - e).abs() < 1e-6, "d={d} e={e}");
            }
        }
    }

    #[test]
    fn many_hops_overflow_is_lazy_but_consistent() {
        let mut params = params();
        params.chip_damping_z = 0.97;
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

//! Drive model: firmware rate limiter, wheel kinematics, and the wheel force
//! model (`DriveModel::Wheels`) or direct integration (`DriveModel::Ideal`).
//!
//! OWNER: math agent.
//!
//! Wheel i has mounting angle `phi_i` (CCW from robot +x to its radial
//! direction) and drives along the tangent `(-sin phi_i, cos phi_i)`. With
//! mount radius `R`, wheel surface speed for a local twist `(vx, vy, omega)` is
//! `u_i = -vx sin(phi_i) + vy cos(phi_i) + R omega`. Forward kinematics is the
//! least-squares inverse (normal equations).

use crate::params::{DriveModel, RobotSpecs};
use crate::robot::{LocalTwist, Robot};
use crate::types::{wrap_angle, Vec2};
use crate::GRAVITY;

/// Lateral (roller-axis) slip speed [m/s] over which the roller friction
/// saturates; regularises the Coulomb `sign()` so the robot does not chatter at rest.
const LATERAL_REG_SPEED: f64 = 0.05;
/// ER-Force `by_force` robot gain (`1/6` per 5 ms substep) as a continuous stiffness [1/s^2].
const FORCE_MOVER_GAIN: f64 = (1.0 / 6.0) / 0.005;
/// Acceleration cap of the `by_force` mover [m/s^2].
const FORCE_MOVER_MAX_ACCEL: f64 = 20.0;

/// Rate-limit `setpoint` toward `target` by the firmware limits over `dt`,
/// then clamp to the velocity limits. `target == None` means motors off: decay
/// toward zero with the coast decelerations. Returns the new setpoint.
pub fn limit_setpoint(
    setpoint: LocalTwist,
    target: Option<LocalTwist>,
    specs: &RobotSpecs,
    dt: f64,
) -> LocalTwist {
    let lim = &specs.limits;
    let sp_xy = Vec2::new(setpoint.vx, setpoint.vy);
    let (tgt_xy, tgt_w, acc_xy, acc_w) = match target {
        Some(t) => {
            let tgt_xy = Vec2::new(t.vx, t.vy);
            let speeding_up = tgt_xy.length_squared() > sp_xy.length_squared();
            let acc_xy = if speeding_up {
                lim.acc_speedup_absolute_max
            } else {
                lim.acc_brake_absolute_max
            };
            let acc_w = if t.omega.abs() > setpoint.omega.abs() {
                lim.acc_speedup_angular_max
            } else {
                lim.acc_brake_angular_max
            };
            (tgt_xy, t.omega, acc_xy, acc_w)
        }
        None => (
            Vec2::ZERO,
            0.0,
            specs.drive.coast_decel,
            specs.drive.coast_angular_decel,
        ),
    };
    let dv = tgt_xy - sp_xy;
    let max_dv = acc_xy.max(0.0) * dt;
    let mut xy = if dv.length() <= max_dv {
        tgt_xy
    } else {
        sp_xy + dv.normalize() * max_dv
    };
    let dw = tgt_w - setpoint.omega;
    let max_dw = acc_w.max(0.0) * dt;
    let mut w = if dw.abs() <= max_dw {
        tgt_w
    } else {
        setpoint.omega + dw.signum() * max_dw
    };

    if xy.length() > lim.vel_absolute_max {
        xy = xy.normalize() * lim.vel_absolute_max;
    }
    w = w.clamp(-lim.vel_angular_max, lim.vel_angular_max);
    LocalTwist {
        vx: xy.x,
        vy: xy.y,
        omega: w,
    }
}

/// Wheel surface speeds [m/s] for a local twist (protocol wheel order).
pub fn inverse_kinematics(twist: LocalTwist, specs: &RobotSpecs) -> [f64; 4] {
    let r = specs.drive.wheel_mount_radius;
    let mut out = [0.0; 4];
    for (u, phi) in out.iter_mut().zip(specs.wheel_angles.as_array()) {
        let (s, c) = phi.sin_cos();
        *u = -twist.vx * s + twist.vy * c + r * twist.omega;
    }
    out
}

/// Least-squares local twist for wheel surface speeds (protocol wheel order).
pub fn forward_kinematics(wheels: [f64; 4], specs: &RobotSpecs) -> LocalTwist {
    let r = specs.drive.wheel_mount_radius;
    // Normal equations A^T A x = A^T u with rows a_i = (-sin phi, cos phi, R).
    let mut ata = [[0.0f64; 3]; 3];
    let mut atu = [0.0f64; 3];
    for (u, phi) in wheels.iter().zip(specs.wheel_angles.as_array()) {
        let (s, c) = phi.sin_cos();
        let a = [-s, c, r];
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += a[i] * a[j];
            }
            atu[i] += a[i] * u;
        }
    }
    let x = solve3(ata, atu).unwrap_or([0.0; 3]);
    LocalTwist {
        vx: x[0],
        vy: x[1],
        omega: x[2],
    }
}

/// Solve a 3x3 linear system by Gaussian elimination with partial pivoting.
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut piv = col;
        for row in (col + 1)..3 {
            if a[row][col].abs() > a[piv][col].abs() {
                piv = row;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in (col + 1)..3 {
            let f = a[row][col] / a[col][col];
            let pivot_row = a[col];
            for (dst, src) in a[row].iter_mut().zip(pivot_row.iter()).skip(col) {
                *dst -= f * src;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        let mut s = b[i];
        for k in (i + 1)..3 {
            s -= a[i][k] * x[k];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// Body force [N] (world frame) and torque [N m] produced by the wheels when
/// tracking `setpoint` from the robot's current velocity, including motor force
/// caps and anisotropic traction limits.
pub fn wheel_wrench(robot: &Robot, setpoint: LocalTwist) -> (Vec2, f64) {
    let specs = &robot.specs;
    let drive = &specs.drive;
    let actual = robot.local_velocity();
    let u_actual = inverse_kinematics(actual, specs);
    let mut u_target = inverse_kinematics(setpoint, specs);
    let u_max = drive.max_wheel_speed.max(0.0);
    for u in &mut u_target {
        *u = u.clamp(-u_max, u_max);
    }
    let normal = specs.mass * GRAVITY / 4.0;
    let f_traction = drive.mu_drive.max(0.0) * normal;
    let f_lateral_max = drive.mu_lateral.max(0.0) * normal;
    let f_max = drive.max_wheel_force.max(0.0);
    let mount = drive.wheel_mount_radius;

    let mut force = Vec2::ZERO;
    let mut torque = 0.0;
    for (i, phi) in specs.wheel_angles.as_array().into_iter().enumerate() {
        let (s, c) = phi.sin_cos();
        let tangent = Vec2::new(-s, c);
        let radial = Vec2::new(c, s);
        let f_drive = (drive.velocity_gain * (u_target[i] - u_actual[i]))
            .clamp(-f_max, f_max)
            .clamp(-f_traction, f_traction);
        let v_lat = actual.vx * c + actual.vy * s;
        let f_lat = -f_lateral_max * (v_lat / LATERAL_REG_SPEED).clamp(-1.0, 1.0);
        force += tangent * f_drive + radial * f_lat;
        torque += mount * f_drive;
    }
    (robot.to_world(force), torque)
}

/// Acceleration [m/s^2] (world) of the `by_force` mover for a robot, or zero.
pub fn force_mover_accel(robot: &Robot) -> Vec2 {
    match robot.force_target {
        Some(target) => {
            let delta = target - robot.pos;
            let mut a = delta * FORCE_MOVER_GAIN;
            if a.length() > FORCE_MOVER_MAX_ACCEL {
                a = a.normalize() * FORCE_MOVER_MAX_ACCEL;
            }
            // critical damping
            a - robot.vel * (2.0 * FORCE_MOVER_GAIN.sqrt())
        }
        None => Vec2::ZERO,
    }
}

/// Advance the robot's pose and velocity by `dt` using its drive model:
/// `Wheels` integrates the wheel wrench plus any `by_force` mover with
/// semi-implicit Euler; `Ideal` sets the velocity to the setpoint (world
/// frame) and integrates the pose. While a `by_force` target is active the
/// motors are off and only the mover acts (in both modes).
pub fn integrate(robot: &mut Robot, setpoint: LocalTwist, external_force: Vec2, dt: f64) {
    let mass = robot.specs.mass.max(1e-6);
    let inertia = robot.specs.inertia().max(1e-9);
    let mover = force_mover_accel(robot);
    let pushed = robot.force_target.is_some();
    match robot.specs.drive.model {
        DriveModel::Wheels => {
            let (force, torque) = if pushed {
                (Vec2::ZERO, 0.0)
            } else {
                wheel_wrench(robot, setpoint)
            };
            let accel = (force + external_force) / mass + mover;
            robot.vel += accel * dt;
            robot.omega += torque / inertia * dt;
        }
        DriveModel::Ideal => {
            if pushed {
                robot.vel += (external_force / mass + mover) * dt;
            } else {
                robot.vel = robot.to_world(Vec2::new(setpoint.vx, setpoint.vy))
                    + external_force / mass * dt;
                robot.omega = setpoint.omega;
            }
        }
    }
    robot.pos += robot.vel * dt;
    robot.orientation = wrap_angle(robot.orientation + robot.omega * dt);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{RobotId, Team};

    fn robot() -> Robot {
        Robot::new(
            RobotId::new(Team::Blue, 3),
            RobotSpecs::default(),
            Vec2::ZERO,
            0.0,
        )
    }

    #[test]
    fn ik_fk_round_trip() {
        let specs = RobotSpecs::default();
        for twist in [
            LocalTwist {
                vx: 1.0,
                vy: 0.0,
                omega: 0.0,
            },
            LocalTwist {
                vx: 0.0,
                vy: 1.0,
                omega: 0.0,
            },
            LocalTwist {
                vx: 0.0,
                vy: 0.0,
                omega: 3.0,
            },
            LocalTwist {
                vx: 1.2,
                vy: -0.7,
                omega: -2.5,
            },
        ] {
            let wheels = inverse_kinematics(twist, &specs);
            let back = forward_kinematics(wheels, &specs);
            assert!((back.vx - twist.vx).abs() < 1e-9);
            assert!((back.vy - twist.vy).abs() < 1e-9);
            assert!((back.omega - twist.omega).abs() < 1e-9);
        }
        // forward motion spins the front-right wheel positively (CCW tangent convention)
        let w = inverse_kinematics(
            LocalTwist {
                vx: 1.0,
                vy: 0.0,
                omega: 0.0,
            },
            &specs,
        );
        assert!(w[0] > 0.0 && w[3] < 0.0);
        // pure rotation: all wheels equal R * omega
        let w = inverse_kinematics(
            LocalTwist {
                vx: 0.0,
                vy: 0.0,
                omega: 2.0,
            },
            &specs,
        );
        for u in w {
            assert!((u - 2.0 * specs.drive.wheel_mount_radius).abs() < 1e-12);
        }
    }

    #[test]
    fn limiter_obeys_accel_and_velocity_limits() {
        let specs = RobotSpecs::default();
        let dt = 0.001;
        let target = Some(LocalTwist {
            vx: 3.0,
            vy: 0.0,
            omega: 10.0,
        });
        let mut sp = LocalTwist::default();
        let mut t = 0.0;
        while t < 1.0 {
            let next = limit_setpoint(sp, target, &specs, dt);
            let a = (Vec2::new(next.vx, next.vy) - Vec2::new(sp.vx, sp.vy)).length() / dt;
            assert!(a <= specs.limits.acc_speedup_absolute_max + 1e-9);
            let aw = (next.omega - sp.omega).abs() / dt;
            assert!(aw <= specs.limits.acc_speedup_angular_max + 1e-9);
            sp = next;
            t += dt;
        }
        assert!((sp.vx - 3.0).abs() < 1e-9 && (sp.omega - 10.0).abs() < 1e-9);
        // time to reach 3 m/s at 4 m/s^2 is 0.75 s
        let mut sp2 = LocalTwist::default();
        let mut n = 0;
        while (sp2.vx - 3.0).abs() > 1e-9 {
            sp2 = limit_setpoint(sp2, target, &specs, dt);
            n += 1;
        }
        assert!((n as f64 * dt - 0.75).abs() < 2.0 * dt);
        // braking uses the brake limit
        let next = limit_setpoint(sp, Some(LocalTwist::default()), &specs, dt);
        assert!((sp.vx - next.vx - specs.limits.acc_brake_absolute_max * dt).abs() < 1e-9);
        // velocity clamp
        let fast = limit_setpoint(
            LocalTwist {
                vx: 3.4,
                vy: 0.0,
                omega: 19.99,
            },
            Some(LocalTwist {
                vx: 10.0,
                vy: 0.0,
                omega: 50.0,
            }),
            &specs,
            0.1,
        );
        assert!((fast.vx - specs.limits.vel_absolute_max).abs() < 1e-12);
        assert!((fast.omega - specs.limits.vel_angular_max).abs() < 1e-12);
        // coast toward zero with no target
        let coast = limit_setpoint(
            LocalTwist {
                vx: 1.0,
                vy: 0.0,
                omega: 5.0,
            },
            None,
            &specs,
            dt,
        );
        assert!((coast.vx - (1.0 - specs.drive.coast_decel * dt)).abs() < 1e-12);
        assert!((coast.omega - (5.0 - specs.drive.coast_angular_decel * dt)).abs() < 1e-12);
        let stopped = limit_setpoint(
            LocalTwist {
                vx: 0.001,
                vy: 0.0,
                omega: 0.01,
            },
            None,
            &specs,
            dt,
        );
        assert_eq!(stopped, LocalTwist::default());
    }

    /// Drive the robot through the limiter + wheel model for `duration` toward a
    /// local target and return the local velocity history sampled every step.
    fn run_wheels(target: LocalTwist, duration: f64) -> (Robot, Vec<LocalTwist>) {
        let mut r = robot();
        let dt = 0.001;
        let mut hist = Vec::new();
        let mut t = 0.0;
        while t < duration {
            r.setpoint = limit_setpoint(r.setpoint, Some(target), &r.specs, dt);
            let sp = r.setpoint;
            integrate(&mut r, sp, Vec2::ZERO, dt);
            hist.push(r.local_velocity());
            t += dt;
        }
        (r, hist)
    }

    #[test]
    fn wheels_reach_forward_step_without_oscillation() {
        let (r, hist) = run_wheels(
            LocalTwist {
                vx: 1.0,
                vy: 0.0,
                omega: 0.0,
            },
            0.6,
        );
        let at_300 = hist[299];
        assert!(
            (at_300.vx - 1.0).abs() <= 0.02,
            "vx at 0.3 s = {}",
            at_300.vx
        );
        assert!(at_300.vy.abs() < 0.01 && at_300.omega.abs() < 0.05);
        // never overshoots meaningfully and is monotone non-decreasing
        let mut prev = 0.0;
        for h in &hist {
            assert!(h.vx <= 1.005, "overshoot {}", h.vx);
            assert!(h.vx >= prev - 1e-6, "oscillation");
            prev = h.vx;
        }
        assert!((r.local_velocity().vx - 1.0).abs() < 0.005);
        assert!(r.orientation.abs() < 0.02);
    }

    #[test]
    fn wheels_reach_lateral_step() {
        let (_, hist) = run_wheels(
            LocalTwist {
                vx: 0.0,
                vy: 0.5,
                omega: 0.0,
            },
            0.3,
        );
        let at_300 = hist[299];
        assert!(
            (at_300.vy - 0.5).abs() <= 0.02,
            "vy at 0.3 s = {}",
            at_300.vy
        );
        assert!(at_300.vx.abs() < 0.01);
        let (_, hist) = run_wheels(
            LocalTwist {
                vx: 0.0,
                vy: 0.0,
                omega: 5.0,
            },
            0.3,
        );
        assert!((hist[299].omega - 5.0).abs() < 0.1);
    }

    #[test]
    fn large_lateral_step_slips() {
        // Bypass the firmware limiter: 4 m/s lateral setpoint from rest.
        let mut r = robot();
        let dt = 0.001;
        let sp = LocalTwist {
            vx: 0.0,
            vy: 4.0,
            omega: 0.0,
        };
        integrate(&mut r, sp, Vec2::ZERO, dt);
        let a = r.vel.y / dt;
        let commanded = r.specs.drive.velocity_gain * 4.0 * 4.0 / r.specs.mass; // unbounded P demand
        assert!(a > 0.0);
        assert!(a < commanded);
        // traction cap: |F_i| <= mu * m g / 4 per wheel
        let n = r.specs.mass * GRAVITY / 4.0;
        let cap: f64 = r
            .specs
            .wheel_angles
            .as_array()
            .iter()
            .map(|phi| phi.cos().abs())
            .sum::<f64>()
            * r.specs.drive.mu_drive
            * n
            / r.specs.mass;
        assert!(a <= cap + 1e-9, "a={a} cap={cap}");
        assert!(a > 0.5 * cap);
    }

    #[test]
    fn ideal_mode_follows_setpoint_exactly() {
        let mut r = robot();
        r.specs.drive.model = DriveModel::Ideal;
        r.orientation = std::f64::consts::FRAC_PI_2;
        let sp = LocalTwist {
            vx: 1.0,
            vy: 0.0,
            omega: 2.0,
        };
        integrate(&mut r, sp, Vec2::ZERO, 0.001);
        assert!((r.vel - Vec2::new(0.0, 1.0)).length() < 1e-9);
        assert!((r.omega - 2.0).abs() < 1e-12);
        assert!((r.pos.y - 0.001).abs() < 1e-12);
    }

    #[test]
    fn force_mover_converges_without_overshoot() {
        let mut r = robot();
        r.force_target = Some(Vec2::new(1.0, 0.0));
        let dt = 0.001;
        let mut max_x: f64 = 0.0;
        for _ in 0..3000 {
            r.setpoint = limit_setpoint(r.setpoint, None, &r.specs, dt);
            let sp = r.setpoint;
            integrate(&mut r, sp, Vec2::ZERO, dt);
            max_x = max_x.max(r.pos.x);
        }
        assert!((r.pos.x - 1.0).abs() < 0.01, "x={}", r.pos.x);
        assert!(max_x <= 1.001);
        assert!(r.vel.length() < 0.05);
    }
}

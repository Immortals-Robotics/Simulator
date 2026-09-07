//! Dribbler: traction model that pulls the ball to the seated point with a
//! bounded force, imparts back-spin, and releases (slips) when the required
//! force exceeds the budget. Optional "glue" mode snaps the ball instead.
//!
//! OWNER: math agent.
//!
//! Model: the seated point is `shoot_radius` along the heading (default
//! `center_to_dribbler + ball_radius - seat_depth`, i.e. the ball nests
//! `seat_depth` into the kicker face plane; measured `seat_depth = 0`, the
//! ball touches the face). A critically damped PD with a
//! 20 ms horizon computes the acceleration needed to bring the ball to the
//! seat with the surface velocity of the seat. The component that *pushes*
//! the ball forward (along the heading) is provided by the rigid kicker face
//! and is unlimited; the *pulling* component (backward) and the lateral
//! component are traction and are limited to `hold_accel_actual * fraction`
//! (`Robot::hold_accel_actual`, the per-robot draw around `hold_accel`). When
//! the ball was held and the traction demand exceeds the budget, the ball
//! slips: only the budget is applied and `holding` drops. Back-spin is set
//! to `surface_vel - heading * min(fraction * omega_max * roller_radius,
//! MAX_BACKSPIN)`; the cap keeps a released ball from darting back into the
//! robot at unrealistic speed (the slide model turns back-spin into a
//! `acc_slide` pull toward the robot for as long as the slip lasts, which is
//! what keeps a real ball in the mouth).

use crate::params::BallParams;
use crate::robot::Robot;
use crate::types::{BallState, Vec2};

/// Horizon [s] over which the ball is brought to the seated point.
const HORIZON: f64 = 0.02;
/// Cap on the imparted back-spin [m/s of contact-point speed].
const MAX_BACKSPIN: f64 = 1.0;
/// Fraction of the budget below which a slipping ball is re-grabbed.
const REGRAB_FRACTION: f64 = 0.8;
/// Ball centre height limit for mouth-zone / barrier tests [m] above the floor.
const MOUTH_MAX_BOTTOM: f64 = 0.03;

/// Dribbler runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DribblerState {
    /// Dribbler is running (command > 0).
    pub active: bool,
    /// Commanded speed as a fraction of `max_speed_rpm` (0..=1).
    pub fraction: f64,
    /// The ball is currently held (within the mouth zone and not slipping).
    pub holding: bool,
}

/// Ball centre in the robot frame.
fn local_ball(robot: &Robot, ball: &BallState) -> Vec2 {
    robot.to_local(ball.pos_xy() - robot.pos)
}

/// Mouth zone test: ball centre within the region in front of the kicker face
/// where the dribbler can act (robot frame: x in
/// `[center_to_dribbler - seat_depth, center_to_dribbler + ball_radius + 0.015]`,
/// `|y| <= dribbler_width / 2`, `z - ball_radius < 0.03`).
pub fn in_mouth_zone(robot: &Robot, ball: &BallState, params: &BallParams) -> bool {
    let s = &robot.specs;
    let p = local_ball(robot, ball);
    let x_min = s.center_to_dribbler - s.dribbler.seat_depth;
    let x_max = s.center_to_dribbler + params.radius + 0.015;
    p.x >= x_min
        && p.x <= x_max
        && p.y.abs() <= s.dribbler_width * 0.5
        && ball.pos.z - params.radius < MOUTH_MAX_BOTTOM
}

/// Break-beam / infrared test used for `dribbler_ball_contact` feedback:
/// robot frame x in `[center_to_dribbler, center_to_dribbler + 0.0235]`,
/// `|y| <= radius * sin(mouth_half_angle)`, ball centre below 6 cm.
pub fn barrier_interrupted(robot: &Robot, ball: &BallState) -> bool {
    let s = &robot.specs;
    let p = local_ball(robot, ball);
    p.x >= s.center_to_dribbler
        && p.x <= s.center_to_dribbler + 0.0235
        && p.y.abs() <= s.radius * s.mouth_half_angle().sin()
        && ball.pos.z < 0.06
}

/// Outcome of one substep of dribbling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DribbleOutcome {
    /// Ball is held after this substep.
    pub holding: bool,
    /// The ball slipped away this substep (was held, now is not).
    pub slipped: bool,
    /// Force [N] the ball exerts back on the robot (world frame).
    pub reaction_force: Vec2,
}

impl DribbleOutcome {
    const NONE: DribbleOutcome = DribbleOutcome {
        holding: false,
        slipped: false,
        reaction_force: Vec2::ZERO,
    };
}

/// World position of the seated point.
pub fn seat_point(robot: &Robot) -> Vec2 {
    robot.pos + robot.heading() * robot.specs.shoot_radius
}

/// Apply the dribbler to the ball for `dt`. Modifies `ball.vel` / `ball.spin`
/// (and `ball.pos` in glue mode). Does nothing when the dribbler is inactive
/// or the ball is outside the mouth zone.
pub fn apply(
    robot: &mut Robot,
    ball: &mut BallState,
    params: &BallParams,
    dt: f64,
) -> DribbleOutcome {
    let state = robot.dribbler;
    if !state.active || state.fraction <= 0.0 || !in_mouth_zone(robot, ball, params) {
        robot.dribbler.holding = false;
        return DribbleOutcome::NONE;
    }
    let specs = robot.specs;
    let heading = robot.heading();
    let seat = seat_point(robot);
    let u = robot.surface_velocity(seat);
    let omega_max = specs.dribbler.max_speed_rpm * std::f64::consts::TAU / 60.0;
    let backspin = (state.fraction * omega_max * specs.dribbler.roller_radius).min(MAX_BACKSPIN);
    let spin_target = u - heading * backspin;

    if specs.dribbler.glue {
        ball.pos.x = seat.x;
        ball.pos.y = seat.y;
        ball.pos.z = params.radius;
        ball.vel.x = u.x;
        ball.vel.y = u.y;
        ball.vel.z = 0.0;
        ball.spin = spin_target;
        robot.dribbler.holding = true;
        return DribbleOutcome {
            holding: true,
            slipped: false,
            reaction_force: Vec2::ZERO,
        };
    }

    // Critically damped PD toward the seat with the seat's velocity.
    let kp = 1.0 / (HORIZON * HORIZON);
    let kd = 2.0 / HORIZON;
    let demand = (seat - ball.pos_xy()) * kp + (u - ball.vel_xy()) * kd;
    let local = robot.to_local(demand);
    // Forward push is rigid (kicker face); pull and lateral are traction.
    let push = local.x.max(0.0);
    let traction = Vec2::new(local.x.min(0.0), local.y);
    let budget = robot.hold_accel_actual.max(0.0) * state.fraction.clamp(0.0, 1.0);
    let need = traction.length();

    let (applied_traction, holding, slipped) =
        if need <= budget * if state.holding { 1.0 } else { REGRAB_FRACTION } {
            (traction, true, false)
        } else {
            let scaled = if need > 1e-12 {
                traction * (budget / need)
            } else {
                traction
            };
            (scaled, false, state.holding)
        };

    let accel_local = Vec2::new(push + applied_traction.x, applied_traction.y);
    let accel = robot.to_world(accel_local);
    ball.vel.x += accel.x * dt;
    ball.vel.y += accel.y * dt;
    if ball.vel.z < 0.0 {
        ball.vel.z = 0.0;
    }
    // Back-spin from the roller (only while it has grip on the ball).
    if holding {
        ball.spin = spin_target;
    }
    robot.dribbler.holding = holding;
    DribbleOutcome {
        holding,
        slipped,
        reaction_force: -accel * params.mass,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::RobotSpecs;
    use crate::types::{RobotId, Team, Vec3};

    fn robot() -> Robot {
        let mut r = Robot::new(
            RobotId::new(Team::Yellow, 2),
            RobotSpecs::default(),
            Vec2::ZERO,
            0.0,
        );
        r.dribbler = DribblerState {
            active: true,
            fraction: 1.0,
            holding: false,
        };
        r
    }

    fn seated_ball(robot: &Robot, params: &BallParams) -> BallState {
        let p = seat_point(robot);
        BallState {
            pos: Vec3::new(p.x, p.y, params.radius),
            vel: Vec3::new(robot.vel.x, robot.vel.y, 0.0),
            spin: Vec2::ZERO,
        }
    }

    #[test]
    fn zones() {
        let params = BallParams::default();
        let r = robot();
        let mut ball = seated_ball(&r, &params);
        assert!(in_mouth_zone(&r, &ball, &params));
        assert!(barrier_interrupted(&r, &ball));
        ball.pos.x = 0.075 + params.radius + 0.02;
        assert!(!in_mouth_zone(&r, &ball, &params));
        assert!(!barrier_interrupted(&r, &ball));
        ball.pos.x = 0.09;
        ball.pos.y = 0.04;
        assert!(!in_mouth_zone(&r, &ball, &params));
        assert!(barrier_interrupted(&r, &ball)); // barrier is wider (R sin theta ~ 0.0497)
        ball.pos.y = 0.0;
        ball.pos.z = 0.2;
        assert!(!in_mouth_zone(&r, &ball, &params));
        assert!(!barrier_interrupted(&r, &ball));
        // rotated robot
        let mut r2 = robot();
        r2.orientation = std::f64::consts::FRAC_PI_2;
        r2.pos = Vec2::new(1.0, 1.0);
        let b2 = seated_ball(&r2, &params);
        assert!(in_mouth_zone(&r2, &b2, &params) && barrier_interrupted(&r2, &b2));
    }

    /// Robot moves with constant world acceleration `a`; ball follows with the
    /// dribbler only (no ground friction). Returns whether the ball was still
    /// held at the end and whether a slip happened.
    fn run(a: Vec2, duration: f64) -> (bool, bool) {
        let params = BallParams::default();
        let mut r = robot();
        let mut ball = seated_ball(&r, &params);
        let dt = 0.001;
        let mut t = 0.0;
        let mut slipped = false;
        let mut holding = false;
        while t < duration {
            r.vel += a * dt;
            r.pos += r.vel * dt;
            let out = apply(&mut r, &mut ball, &params, dt);
            slipped |= out.slipped;
            holding = out.holding;
            ball.pos += ball.vel * dt;
            t += dt;
        }
        (holding, slipped)
    }

    #[test]
    fn holds_at_2_and_drops_at_5() {
        // pulling (robot reversing); the mean budget is 3 m/s^2
        let (holding, slipped) = run(Vec2::new(-2.0, 0.0), 0.5);
        assert!(holding && !slipped);
        let (holding, slipped) = run(Vec2::new(-5.0, 0.0), 0.5);
        assert!(!holding && slipped);
        // lateral
        let (holding, slipped) = run(Vec2::new(0.0, 2.0), 0.5);
        assert!(holding && !slipped);
        let (holding, slipped) = run(Vec2::new(0.0, 5.0), 0.5);
        assert!(!holding && slipped);
        // pushing forward is rigid: no slip even at 10 m/s^2
        let (holding, slipped) = run(Vec2::new(10.0, 0.0), 0.3);
        assert!(holding && !slipped);
    }

    #[test]
    fn held_ball_tracks_seat_and_gets_backspin() {
        let params = BallParams::default();
        let mut r = robot();
        r.vel = Vec2::new(1.0, 0.0);
        r.omega = 1.0;
        let mut ball = seated_ball(&r, &params);
        let u0 = r.surface_velocity(seat_point(&r));
        ball.vel = Vec3::new(u0.x, u0.y, 0.0);
        let dt = 0.001;
        for i in 0..500 {
            r.pos += r.vel * dt;
            r.orientation += r.omega * dt;
            let out = apply(&mut r, &mut ball, &params, dt);
            assert!(out.holding || i < 5, "lost the ball at step {i}");
            assert!(!out.slipped);
            ball.pos += ball.vel * dt;
        }
        let seat = seat_point(&r);
        assert!((ball.pos_xy() - seat).length() < 0.002);
        assert!(barrier_interrupted(&r, &ball));
        let backspin = ball.spin - r.surface_velocity(seat);
        assert!(backspin.dot(r.heading()) < -0.5);
        assert!(backspin.length() <= MAX_BACKSPIN + 1e-9);
    }

    #[test]
    fn slip_scales_to_budget_and_reaction_force() {
        let params = BallParams::default();
        let mut r = robot();
        r.dribbler.holding = true;
        r.vel = Vec2::new(-1.0, 0.0); // robot reversing fast, ball at rest
        let mut ball = seated_ball(&r, &params);
        ball.vel = Vec3::ZERO;
        let out = apply(&mut r, &mut ball, &params, 0.001);
        assert!(out.slipped && !out.holding);
        let a = ball.vel_xy() / 0.001;
        assert_eq!(r.hold_accel_actual, r.specs.dribbler.hold_accel);
        assert!((a.length() - r.specs.dribbler.hold_accel).abs() < 1e-6);
        assert!((out.reaction_force + a * params.mass).length() < 1e-9);
        // half speed: half budget
        let mut r = robot();
        r.dribbler.fraction = 0.5;
        r.vel = Vec2::new(-1.0, 0.0);
        let mut ball = seated_ball(&r, &params);
        ball.vel = Vec3::ZERO;
        apply(&mut r, &mut ball, &params, 0.001);
        let half = 0.5 * r.specs.dribbler.hold_accel;
        assert!((ball.vel_xy().length() / 0.001 - half).abs() < 1e-6);
        // the per-robot draw is what limits the pull, not the spec mean
        let mut r = robot();
        r.hold_accel_actual = 1.5;
        r.dribbler.holding = true;
        r.vel = Vec2::new(-1.0, 0.0);
        let mut ball = seated_ball(&r, &params);
        ball.vel = Vec3::ZERO;
        let out = apply(&mut r, &mut ball, &params, 0.001);
        assert!(out.slipped);
        assert!((ball.vel_xy().length() / 0.001 - 1.5).abs() < 1e-6);
    }

    #[test]
    fn per_robot_budget_decides_the_slip_threshold() {
        // A generous robot holds at 4 m/s^2 of pull; a weak one drops the ball.
        let run_with = |hold: f64| {
            let params = BallParams::default();
            let mut r = robot();
            r.hold_accel_actual = hold;
            let mut ball = seated_ball(&r, &params);
            let dt = 0.001;
            let mut slipped = false;
            for _ in 0..500 {
                r.vel += Vec2::new(-4.0, 0.0) * dt;
                r.pos += r.vel * dt;
                let out = apply(&mut r, &mut ball, &params, dt);
                slipped |= out.slipped;
                ball.pos += ball.vel * dt;
            }
            slipped
        };
        assert!(!run_with(5.0));
        assert!(run_with(1.5));
    }

    #[test]
    fn inactive_or_outside_does_nothing() {
        let params = BallParams::default();
        let mut r = robot();
        r.dribbler.active = false;
        r.dribbler.holding = true;
        let mut ball = seated_ball(&r, &params);
        let before = ball;
        let out = apply(&mut r, &mut ball, &params, 0.001);
        assert_eq!(out, DribbleOutcome::NONE);
        assert_eq!(ball, before);
        assert!(!r.dribbler.holding);
        let mut r = robot();
        let mut far = seated_ball(&r, &params);
        far.pos.x += 0.5;
        assert_eq!(
            apply(&mut r, &mut far, &params, 0.001),
            DribbleOutcome::NONE
        );
    }

    #[test]
    fn glue_mode_snaps() {
        let params = BallParams::default();
        let mut r = robot();
        r.specs.dribbler.glue = true;
        r.vel = Vec2::new(0.5, 0.2);
        let mut ball = seated_ball(&r, &params);
        ball.pos.x += 0.01;
        ball.vel = Vec3::ZERO;
        let out = apply(&mut r, &mut ball, &params, 0.001);
        assert!(out.holding);
        assert!((ball.pos_xy() - seat_point(&r)).length() < 1e-12);
        assert!((ball.vel_xy() - r.surface_velocity(seat_point(&r))).length() < 1e-12);
    }
}

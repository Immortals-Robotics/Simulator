//! Kicker: charge/cooldown state, kick-zone test, and the impulse applied to
//! the ball.
//!
//! OWNER: general agent A. Replace the `todo!()` bodies; keep the public API.

use crate::params::BallParams;
use crate::robot::Robot;
use crate::types::{BallState, Event, Vec2, Vec3};

/// Slack behind the kicker face for the kick zone [m].
const KICK_ZONE_BACK: f64 = 0.005;
/// Slack in front of the ball for the kick zone [m].
const KICK_ZONE_FRONT: f64 = 0.015;

/// Kicker runtime state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KickerState {
    /// Capacitor charged and ready.
    pub charged: bool,
    /// Seconds since the last kick (charging progress).
    pub since_kick: f64,
}

impl Default for KickerState {
    fn default() -> Self {
        Self {
            charged: true,
            since_kick: f64::INFINITY,
        }
    }
}

/// Advance the charge state by `dt`.
pub fn update_charge(robot: &mut Robot, dt: f64) {
    let charge_time = robot.specs.kicker.charge_time;
    let kicker = &mut robot.kicker;
    if kicker.charged {
        return;
    }
    kicker.since_kick += dt;
    if kicker.since_kick >= charge_time {
        kicker.charged = true;
    }
}

/// Kick-zone test: ball centre in the robot frame within
/// `x in [center_to_dribbler - 0.005, center_to_dribbler + ball_radius + 0.015]`,
/// `|y| <= dribbler_width / 2`, and `z < kicker.max_ball_height`.
pub fn in_kick_zone(robot: &Robot, ball: &BallState, params: &BallParams) -> bool {
    if ball.pos.z >= robot.specs.kicker.max_ball_height {
        return false;
    }
    let local = robot.to_local(ball.pos_xy() - robot.pos);
    let c2d = robot.specs.center_to_dribbler;
    local.x >= c2d - KICK_ZONE_BACK
        && local.x <= c2d + params.radius + KICK_ZONE_FRONT
        && local.y.abs() <= robot.specs.dribbler_width * 0.5
}

/// If the robot's command requests a kick, the kicker is charged and the ball
/// is in the kick zone: set the ball velocity to the kick velocity (robot
/// heading rotated up by `kick_angle_deg`, speed clamped to
/// `[min_speed, max_linear|max_chip]`, plus the robot's own velocity, after
/// cancelling `incoming_damping` of the ball's incoming normal velocity),
/// zero the spin, discharge, and return the event. Chip if angle > 0.
pub fn try_kick(robot: &mut Robot, ball: &mut BallState, params: &BallParams) -> Option<Event> {
    let commanded = robot.command.kick_speed?;
    if commanded <= 0.0 || !robot.kicker.charged {
        return None;
    }
    if !in_kick_zone(robot, ball, params) {
        return None;
    }

    let kicker = robot.specs.kicker;
    let angle_deg = robot.command.kick_angle_deg;
    let max_speed = if angle_deg > 0.0 {
        kicker.max_chip_speed
    } else {
        kicker.max_linear_speed
    };
    let speed = commanded.clamp(kicker.min_speed.min(max_speed), max_speed);

    // The ball's velocity component running into the kicker face is cancelled
    // by `incoming_damping` (1 = fully absorbed, 0 = fully retained).
    let heading = robot.heading();
    let normal = Vec3::new(heading.x, heading.y, 0.0);
    let incoming = ball.vel.dot(normal);
    let retained = if incoming < 0.0 {
        normal * (incoming * (1.0 - kicker.incoming_damping))
    } else {
        Vec3::ZERO
    };

    let robot_vel = Vec3::new(robot.vel.x, robot.vel.y, 0.0);
    ball.vel = kick_velocity(robot.orientation, angle_deg, speed) + robot_vel + retained;
    ball.spin = Vec2::ZERO;

    robot.kicker.charged = false;
    robot.kicker.since_kick = 0.0;

    Some(Event::Kick {
        robot: robot.id,
        speed,
        angle_deg,
    })
}

/// Kick velocity vector [m/s] for a heading angle, elevation and speed.
pub fn kick_velocity(heading_rad: f64, elevation_deg: f64, speed: f64) -> Vec3 {
    let (se, ce) = elevation_deg.to_radians().sin_cos();
    let (sh, ch) = heading_rad.sin_cos();
    Vec3::new(ch * ce * speed, sh * ce * speed, se * speed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{KickerParams, RobotSpecs};
    use crate::types::{RobotCommand, RobotId, Team};

    fn specs_with_max(max_linear: f64) -> RobotSpecs {
        let base = RobotSpecs::default();
        RobotSpecs {
            kicker: KickerParams {
                max_linear_speed: max_linear,
                ..base.kicker
            },
            ..base
        }
    }

    fn robot(specs: RobotSpecs) -> Robot {
        Robot::new(RobotId::new(Team::Blue, 0), specs, Vec2::ZERO, 0.0)
    }

    /// A ball resting against the kicker face of a robot at the origin facing +x.
    fn seated_ball(robot: &Robot, params: &BallParams) -> BallState {
        BallState {
            pos: Vec3::new(
                robot.specs.center_to_dribbler + params.radius,
                0.0,
                params.radius,
            ),
            vel: Vec3::ZERO,
            spin: Vec2::ZERO,
        }
    }

    fn kick_command(speed: f64, angle_deg: f64) -> RobotCommand {
        RobotCommand {
            kick_speed: Some(speed),
            kick_angle_deg: angle_deg,
            ..RobotCommand::default()
        }
    }

    #[test]
    fn kicker_speed_matches_the_command() {
        let params = BallParams::default();
        for commanded in [2.0, 4.0, 6.0, 8.0] {
            let mut r = robot(specs_with_max(10.0));
            r.command = kick_command(commanded, 0.0);
            let mut ball = seated_ball(&r, &params);
            let event = try_kick(&mut r, &mut ball, &params).expect("kick");
            let speed = ball.vel.length();
            assert!(
                (speed - commanded).abs() < 0.10,
                "commanded {commanded}, measured {speed}"
            );
            assert!(ball.vel.z.abs() < 1e-12, "straight kick stays flat");
            assert!((ball.vel.x - commanded).abs() < 1e-12, "kick is along +x");
            match event {
                Event::Kick {
                    robot,
                    speed,
                    angle_deg,
                } => {
                    assert_eq!(robot, RobotId::new(Team::Blue, 0));
                    assert!((speed - commanded).abs() < 1e-12);
                    assert_eq!(angle_deg, 0.0);
                }
                other => panic!("expected a kick event, got {other:?}"),
            }
        }
    }

    #[test]
    fn kicker_clamps_to_the_spec_maximum() {
        let params = BallParams::default();
        let specs = RobotSpecs::default();
        let max = specs.kicker.max_linear_speed;
        let mut r = robot(specs);
        r.command = kick_command(13.0, 0.0);
        let mut ball = seated_ball(&r, &params);
        try_kick(&mut r, &mut ball, &params).expect("kick");
        assert!(
            (ball.vel.length() - max).abs() < 1e-12,
            "13 m/s must clamp to {max}"
        );

        // Chips clamp to the (lower) chip maximum.
        let mut r = robot(RobotSpecs::default());
        let chip_max = r.specs.kicker.max_chip_speed;
        r.command = kick_command(13.0, 45.0);
        let mut ball = seated_ball(&r, &params);
        try_kick(&mut r, &mut ball, &params).expect("chip");
        assert!((ball.vel.length() - chip_max).abs() < 1e-9);

        // Tiny commands are lifted to `min_speed`.
        let mut r = robot(RobotSpecs::default());
        let min = r.specs.kicker.min_speed;
        r.command = kick_command(0.001, 0.0);
        let mut ball = seated_ball(&r, &params);
        try_kick(&mut r, &mut ball, &params).expect("kick");
        assert!((ball.vel.length() - min).abs() < 1e-12);
    }

    #[test]
    fn kicker_chip_leaves_at_the_commanded_elevation() {
        let params = BallParams::default();
        let mut r = robot(RobotSpecs::default());
        let speed = 3.0;
        r.command = kick_command(speed, 45.0);
        let mut ball = seated_ball(&r, &params);
        try_kick(&mut r, &mut ball, &params).expect("chip");
        let expected_z = speed * 45.0f64.to_radians().sin();
        assert!(
            (ball.vel.z - expected_z).abs() < 1e-12,
            "vz {} != {expected_z}",
            ball.vel.z
        );
        assert!((ball.vel.x - speed * 45.0f64.to_radians().cos()).abs() < 1e-12);
        assert!((ball.vel.length() - speed).abs() < 1e-12);
    }

    #[test]
    fn kicker_does_not_fire_when_uncharged_out_of_zone_or_too_high() {
        let params = BallParams::default();

        // Uncharged.
        let mut r = robot(RobotSpecs::default());
        r.kicker.charged = false;
        r.kicker.since_kick = 0.0;
        r.command = kick_command(4.0, 0.0);
        let mut ball = seated_ball(&r, &params);
        assert!(try_kick(&mut r, &mut ball, &params).is_none());
        assert_eq!(ball.vel, Vec3::ZERO);

        // Ball too far in front.
        let mut r = robot(RobotSpecs::default());
        r.command = kick_command(4.0, 0.0);
        let mut ball = seated_ball(&r, &params);
        ball.pos.x = 0.3;
        assert!(try_kick(&mut r, &mut ball, &params).is_none());

        // Ball beside the dribbler.
        let mut ball = seated_ball(&r, &params);
        ball.pos.y = r.specs.dribbler_width;
        assert!(try_kick(&mut r, &mut ball, &params).is_none());

        // Ball behind the robot.
        let mut ball = seated_ball(&r, &params);
        ball.pos.x = -0.2;
        assert!(try_kick(&mut r, &mut ball, &params).is_none());

        // Ball flying above the kicker.
        let mut ball = seated_ball(&r, &params);
        ball.pos.z = r.specs.kicker.max_ball_height + 0.001;
        assert!(try_kick(&mut r, &mut ball, &params).is_none());

        // No kick requested.
        let mut r = robot(RobotSpecs::default());
        r.command = RobotCommand::default();
        let mut ball = seated_ball(&r, &params);
        assert!(try_kick(&mut r, &mut ball, &params).is_none());
        r.command = kick_command(0.0, 0.0);
        assert!(try_kick(&mut r, &mut ball, &params).is_none());
    }

    #[test]
    fn kicker_zone_geometry() {
        let params = BallParams::default();
        let r = robot(RobotSpecs::default());
        let c2d = r.specs.center_to_dribbler;
        let mut ball = seated_ball(&r, &params);
        assert!(in_kick_zone(&r, &ball, &params));

        // Just inside the back and front edges.
        ball.pos.x = c2d - 0.004;
        assert!(in_kick_zone(&r, &ball, &params));
        ball.pos.x = c2d - 0.006;
        assert!(!in_kick_zone(&r, &ball, &params));
        ball.pos.x = c2d + params.radius + 0.014;
        assert!(in_kick_zone(&r, &ball, &params));
        ball.pos.x = c2d + params.radius + 0.016;
        assert!(!in_kick_zone(&r, &ball, &params));

        // Lateral limit is the dribbler width.
        let mut ball = seated_ball(&r, &params);
        ball.pos.y = r.specs.dribbler_width * 0.5 - 1e-6;
        assert!(in_kick_zone(&r, &ball, &params));
        ball.pos.y = r.specs.dribbler_width * 0.5 + 1e-6;
        assert!(!in_kick_zone(&r, &ball, &params));

        // The zone rotates with the robot.
        let rotated = Robot::new(
            RobotId::new(Team::Blue, 1),
            RobotSpecs::default(),
            Vec2::ZERO,
            std::f64::consts::FRAC_PI_2,
        );
        let ball = BallState {
            pos: Vec3::new(
                0.0,
                rotated.specs.center_to_dribbler + params.radius,
                params.radius,
            ),
            ..Default::default()
        };
        assert!(in_kick_zone(&rotated, &ball, &params));
    }

    #[test]
    fn kicker_adds_robot_velocity_and_cancels_the_incoming_ball() {
        let params = BallParams::default();
        let mut r = robot(specs_with_max(10.0));
        r.vel = Vec2::new(1.5, 0.25);
        r.command = kick_command(4.0, 0.0);
        let mut ball = seated_ball(&r, &params);
        // Ball rolling into the kicker at 2 m/s.
        ball.vel = Vec3::new(-2.0, 0.0, 0.0);
        ball.spin = Vec2::new(-2.0, 0.0);
        try_kick(&mut r, &mut ball, &params).expect("kick");
        // incoming_damping defaults to 1.0: the -2 m/s is fully absorbed.
        assert!((ball.vel.x - (4.0 + 1.5)).abs() < 1e-12, "{:?}", ball.vel);
        assert!((ball.vel.y - 0.25).abs() < 1e-12);
        assert_eq!(ball.spin, Vec2::ZERO, "spin is reset by the kick");

        // With no damping the incoming component survives.
        let base = specs_with_max(10.0);
        let specs = RobotSpecs {
            kicker: KickerParams {
                incoming_damping: 0.0,
                ..base.kicker
            },
            ..base
        };
        let mut r = robot(specs);
        r.command = kick_command(4.0, 0.0);
        let mut ball = seated_ball(&r, &params);
        ball.vel = Vec3::new(-2.0, 0.0, 0.0);
        try_kick(&mut r, &mut ball, &params).expect("kick");
        assert!((ball.vel.x - 2.0).abs() < 1e-12, "{:?}", ball.vel);
    }

    #[test]
    fn kicker_discharges_and_recharges() {
        let params = BallParams::default();
        let mut r = robot(specs_with_max(10.0));
        let charge_time = r.specs.kicker.charge_time;
        r.command = kick_command(4.0, 0.0);
        let mut ball = seated_ball(&r, &params);
        assert!(r.kicker.charged);
        try_kick(&mut r, &mut ball, &params).expect("kick");
        assert!(!r.kicker.charged);
        assert_eq!(r.kicker.since_kick, 0.0);

        // A second kick in the same state is refused.
        let mut ball = seated_ball(&r, &params);
        assert!(try_kick(&mut r, &mut ball, &params).is_none());

        // Charging takes `charge_time`.
        let dt = 0.001;
        let steps = (charge_time / dt).round() as usize;
        for _ in 0..steps - 1 {
            update_charge(&mut r, dt);
            assert!(!r.kicker.charged);
        }
        update_charge(&mut r, dt);
        assert!(r.kicker.charged);

        // Charging a charged kicker is a no-op and never overflows.
        let before = r.kicker.since_kick;
        update_charge(&mut r, dt);
        assert!(r.kicker.charged);
        assert_eq!(r.kicker.since_kick, before);

        let mut fresh = robot(RobotSpecs::default());
        assert!(fresh.kicker.charged);
        update_charge(&mut fresh, dt);
        assert!(fresh.kicker.since_kick.is_infinite());
    }
}

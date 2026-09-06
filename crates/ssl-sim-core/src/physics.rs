//! One physics substep, split into the robot phase and the ball phase.
//!
//! OWNER: math agent. `World::step` calls these in order; see
//! `docs/design.md` §5.1.

use std::collections::BTreeMap;

use crate::ball::{Ball, BallTrajectory};
use crate::collision;
use crate::field::{FieldGeometry, WallSegment};
use crate::params::SimConfig;
use crate::robot::{dribbler, drive, kicker, Robot};
use crate::types::{Event, RobotId, SimTime, Vec2};

/// Max collision iterations per substep.
const MAX_CONTACTS: usize = 4;
/// ER-Force `by_force` ball gain (`0.1` per 5 ms substep) as a stiffness [1/s^2].
const BALL_FORCE_GAIN: f64 = 0.1 / 0.005;

/// Robot phase: for every robot, compute the commanded twist (with timeout),
/// rate-limit the setpoint, compute the wheel wrench (or ideal velocity), add
/// the `by_force` mover force, integrate, then resolve robot-robot and
/// robot-wall contacts. Also advances kicker charge and the dribbler command state.
pub fn step_robots(
    robots: &mut BTreeMap<RobotId, Robot>,
    field: &FieldGeometry,
    walls: &[WallSegment],
    config: &SimConfig,
    now: SimTime,
    events: &mut Vec<Event>,
) {
    let dt = config.substep;
    for robot in robots.values_mut() {
        if !robot.kicker.charged {
            kicker::update_charge(robot, dt);
        }
        let timed_out = now.minus(robot.command_time).as_secs_f64() > config.command_timeout;
        let rpm = if timed_out {
            0.0
        } else {
            robot.command.dribbler_rpm.unwrap_or(0.0)
        };
        let active = rpm > 0.0;
        robot.dribbler.active = active;
        robot.dribbler.fraction = if active {
            (rpm / robot.specs.dribbler.max_speed_rpm.max(1e-9)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if !active {
            robot.dribbler.holding = false;
        }
        let target = if robot.force_target.is_some() {
            None
        } else {
            robot.commanded_twist(now, config.command_timeout)
        };
        robot.setpoint = drive::limit_setpoint(robot.setpoint, target, &robot.specs, dt);
        let setpoint = robot.setpoint;
        drive::integrate(robot, setpoint, Vec2::ZERO, dt);
    }
    collision::resolve_robot_contacts(robots, field, walls, &config.contact, events);
}

/// Ball phase: kicks (`kicker::try_kick` for every robot whose command asks),
/// dribbler forces, `by_force` mover, then the collision sweep loop (advance
/// to each contact by time of impact, resolve, repeat up to 4 times), then the
/// free advance of the remaining time. Updates every robot's `ball_contact`.
/// Emits `Kick`, `DribblerSlip`, `Goal` and `BallLeftField` events.
pub fn step_ball(
    ball: &mut Ball,
    robots: &mut BTreeMap<RobotId, Robot>,
    field: &FieldGeometry,
    walls: &[WallSegment],
    config: &SimConfig,
    events: &mut Vec<Event>,
) {
    let dt = config.substep;
    let params = &config.ball;
    let room = field.room_half_extents();

    // 1. Kicks.
    let mut kicked_by: Option<RobotId> = None;
    for robot in robots.values_mut() {
        if robot.command.kick_speed.is_some_and(|s| s > 0.0) && robot.kicker.charged {
            if let Some(ev) = kicker::try_kick(robot, &mut ball.state, params) {
                events.push(ev);
                robot.dribbler.holding = false;
                kicked_by = Some(robot.id);
            }
        }
    }

    // 2. Dribblers.
    let mut held_by: Option<RobotId> = None;
    for robot in robots.values_mut() {
        if Some(robot.id) == kicked_by || !robot.dribbler.active {
            robot.dribbler.holding = false;
            continue;
        }
        let out = dribbler::apply(robot, &mut ball.state, params, dt);
        if out.slipped {
            events.push(Event::DribblerSlip { robot: robot.id });
        }
        if out.holding {
            held_by = Some(robot.id);
        }
        if out.reaction_force != Vec2::ZERO {
            robot.vel += out.reaction_force / robot.specs.mass.max(1e-6) * dt;
        }
    }

    // 3. by_force mover.
    if let Some(target) = ball.force_target {
        let delta = Vec2::new(target.x, target.y) - ball.state.pos_xy();
        let a = delta * BALL_FORCE_GAIN - ball.state.vel_xy() * (2.0 * BALL_FORCE_GAIN.sqrt());
        ball.state.vel.x += a.x * dt;
        ball.state.vel.y += a.y * dt;
        // Dragging: keep the ball rolling so only rolling friction resists.
        ball.state.spin = ball.state.vel_xy();
    }

    // 4. Collision sweep loop.
    collision::depenetrate_ball_except(&mut ball.state, robots, walls, room, params, held_by);
    let mut remaining = dt;
    for _ in 0..MAX_CONTACTS {
        if remaining <= 0.0 {
            break;
        }
        let Some(contact) = collision::sweep_ball_except(
            &ball.state,
            remaining,
            robots,
            walls,
            room,
            params,
            held_by,
        ) else {
            break;
        };
        if contact.time > 0.0 {
            let traj = BallTrajectory::from_state(&ball.state, params);
            ball.state = traj.state_at(contact.time);
            remaining -= contact.time;
        }
        let impulse =
            collision::resolve_ball_contact(&mut ball.state, &contact, params, &config.contact);
        let hit_robot = match contact.surface {
            collision::Surface::RobotHull(id) | collision::Surface::KickerFace(id) => Some(id),
            _ => None,
        };
        if let Some(robot) = hit_robot.and_then(|id| robots.get_mut(&id)) {
            let j = Vec2::new(impulse.x, impulse.y);
            let contact_point = ball.state.pos_xy() - contact.normal * params.radius;
            let r = contact_point - robot.pos;
            robot.vel += j / robot.specs.mass.max(1e-6);
            robot.omega += (r.x * j.y - r.y * j.x) / robot.specs.inertia().max(1e-9);
        }
    }
    if remaining > 0.0 {
        ball.advance(remaining, params);
    }

    // 5. Feedback and events.
    for robot in robots.values_mut() {
        robot.ball_contact = dribbler::barrier_interrupted(robot, &ball.state);
    }
    let p = ball.state.pos_xy();
    let outside = field.is_outside_field(p);
    if outside && !ball.was_outside_field {
        events.push(Event::BallLeftField);
    }
    ball.was_outside_field = outside;
    // A goal is necessarily beyond a goal line, so only test there.
    let goal = if outside {
        field.goal_containing(p)
    } else {
        None
    };
    if let Some(team) = goal {
        if ball.in_goal != Some(team) {
            events.push(Event::Goal { conceding: team });
        }
    }
    ball.in_goal = goal;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::RobotSpecs;
    use crate::types::{BallState, MoveCommand, RobotCommand, Team, Vec3};

    fn config() -> SimConfig {
        SimConfig::default()
    }

    /// A large field so that the test balls never leave it (goal tests need
    /// `FieldGeometry::goal_containing`, owned by another module).
    fn big_field() -> FieldGeometry {
        FieldGeometry {
            length: 40.0,
            width: 40.0,
            ..FieldGeometry::default()
        }
    }

    fn wall_x(x: f64, height: f64) -> WallSegment {
        WallSegment {
            a: Vec2::new(x, -5.0),
            b: Vec2::new(x, 5.0),
            height,
            normal: Vec2::NEG_X,
            is_goal: false,
        }
    }

    fn ball_with(pos: Vec3, vel: Vec3) -> Ball {
        Ball {
            state: BallState {
                pos,
                vel,
                spin: Vec2::ZERO,
            },
            ..Default::default()
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

    #[test]
    fn kicked_ball_reflects_off_wall_with_reduced_speed() {
        let cfg = config();
        let field = big_field();
        let walls = [wall_x(6.3, cfg.wall_height)];
        let mut ball = ball_with(
            Vec3::new(6.0, 0.0, cfg.ball.radius),
            Vec3::new(4.0, 0.0, 0.0),
        );
        let mut robots = BTreeMap::new();
        let mut events = Vec::new();
        let mut min_dist_to_wall = f64::MAX;
        for _ in 0..200 {
            step_ball(&mut ball, &mut robots, &field, &walls, &cfg, &mut events);
            min_dist_to_wall = min_dist_to_wall.min(6.3 - ball.state.pos.x);
            assert!(ball.state.pos.x <= 6.3 - cfg.ball.radius + 1e-9);
        }
        assert!(ball.state.vel.x < 0.0, "vx={}", ball.state.vel.x);
        assert!(ball.state.vel.x.abs() < 4.0 * (1.0 - cfg.contact.ball_wall_normal) + 1e-6);
        assert!(ball.state.vel.x.abs() > 1.0);
        // never penetrates; the closest sampled position is within one substep of travel
        assert!(min_dist_to_wall >= cfg.ball.radius - 1e-9);
        assert!(min_dist_to_wall < cfg.ball.radius + 4.0 * cfg.substep);
        assert!(events.is_empty());
    }

    #[test]
    fn chip_clears_goal_wall_but_flat_ball_bounces() {
        let cfg = config();
        let field = big_field();
        let walls = [wall_x(1.5, 0.155)];
        let mut robots = BTreeMap::new();
        let mut events = Vec::new();
        // chip at 45 deg landing at 3 m: well above 0.155 m over the wall at 1.5 m
        let v = crate::ball::trajectory::inverse::chip_speed_for_touchdown(
            3.0,
            45f64.to_radians(),
            0,
            &cfg.ball,
        );
        let mut chip = ball_with(
            Vec3::new(0.0, 0.0, cfg.ball.radius),
            Vec3::new(
                v * 45f64.to_radians().cos(),
                0.0,
                v * 45f64.to_radians().sin(),
            ),
        );
        for _ in 0..2000 {
            step_ball(&mut chip, &mut robots, &field, &walls, &cfg, &mut events);
        }
        assert!(
            chip.state.pos.x > 2.5,
            "chip stopped at x={}",
            chip.state.pos.x
        );
        let mut flat = ball_with(
            Vec3::new(0.0, 0.0, cfg.ball.radius),
            Vec3::new(3.0, 0.0, 0.0),
        );
        for _ in 0..2000 {
            step_ball(&mut flat, &mut robots, &field, &walls, &cfg, &mut events);
        }
        assert!(flat.state.pos.x < 1.5 - cfg.ball.radius + 1e-9);
        assert!(flat.state.vel.x <= 0.0);
    }

    #[test]
    fn ball_into_robot_transfers_impulse_and_updates_contact_flag() {
        let cfg = config();
        let field = big_field();
        let mut robots: BTreeMap<RobotId, Robot> = BTreeMap::new();
        let r = robot(0, Vec2::ZERO, 0.0);
        robots.insert(r.id, r);
        let mut ball = ball_with(
            Vec3::new(0.4, 0.0, cfg.ball.radius),
            Vec3::new(-2.0, 0.0, 0.0),
        );
        ball.state.spin = ball.state.vel_xy();
        let mut events = Vec::new();
        let mut contact_seen = false;
        for _ in 0..300 {
            step_ball(&mut ball, &mut robots, &field, &[], &cfg, &mut events);
            contact_seen |= robots[&RobotId::new(Team::Blue, 0)].ball_contact;
        }
        assert!(contact_seen);
        assert!(ball.state.vel.x > 0.0);
        assert!(robots[&RobotId::new(Team::Blue, 0)].vel.x < 0.0);
        assert!(!robots[&RobotId::new(Team::Blue, 0)].ball_contact);
    }

    #[test]
    fn robots_step_and_collide() {
        let cfg = config();
        let field = big_field();
        let mut robots: BTreeMap<RobotId, Robot> = BTreeMap::new();
        let mut a = robot(0, Vec2::ZERO, 0.0);
        a.set_command(
            RobotCommand {
                movement: Some(MoveCommand::LocalVelocity {
                    forward: 1.0,
                    left: 0.0,
                    angular: 0.0,
                }),
                dribbler_rpm: Some(5000.0),
                ..Default::default()
            },
            SimTime::ZERO,
        );
        let b = robot(1, Vec2::new(0.6, 0.0), std::f64::consts::PI);
        robots.insert(a.id, a);
        robots.insert(b.id, b);
        let mut events = Vec::new();
        let mut now = SimTime::ZERO;
        for _ in 0..800 {
            // keep the command fresh
            let cmd = robots[&RobotId::new(Team::Blue, 0)].command;
            robots
                .get_mut(&RobotId::new(Team::Blue, 0))
                .unwrap()
                .set_command(cmd, now);
            step_robots(&mut robots, &field, &[], &cfg, now, &mut events);
            now = now.plus(cfg.substep_time());
        }
        let a = &robots[&RobotId::new(Team::Blue, 0)];
        let b = &robots[&RobotId::new(Team::Blue, 1)];
        assert!(a.dribbler.active && (a.dribbler.fraction - 0.5).abs() < 1e-12);
        assert!((b.pos - a.pos).length() >= 0.18 - 1e-6);
        assert!(b.pos.x > 0.6, "pushed robot moved: {}", b.pos.x);
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::RobotCollision { .. })));
        // command timeout: dribbler goes off and the robot coasts
        for _ in 0..500 {
            step_robots(&mut robots, &field, &[], &cfg, now, &mut events);
            now = now.plus(cfg.substep_time());
        }
        let a = &robots[&RobotId::new(Team::Blue, 0)];
        assert!(!a.dribbler.active);
        assert!(a.vel.length() < 0.05);
    }

    #[test]
    fn dribbling_robot_carries_ball_through_full_step() {
        let cfg = config();
        let field = big_field();
        let mut robots: BTreeMap<RobotId, Robot> = BTreeMap::new();
        let mut a = robot(0, Vec2::ZERO, 0.0);
        a.set_command(
            RobotCommand {
                movement: Some(MoveCommand::LocalVelocity {
                    forward: -0.8,
                    left: 0.3,
                    angular: 0.5,
                }),
                dribbler_rpm: Some(10000.0),
                ..Default::default()
            },
            SimTime::ZERO,
        );
        robots.insert(a.id, a);
        let seat = dribbler::seat_point(&robots[&RobotId::new(Team::Blue, 0)]);
        let mut ball = ball_with(Vec3::new(seat.x, seat.y, cfg.ball.radius), Vec3::ZERO);
        let mut events = Vec::new();
        let mut now = SimTime::ZERO;
        for _ in 0..1500 {
            let cmd = robots[&RobotId::new(Team::Blue, 0)].command;
            robots
                .get_mut(&RobotId::new(Team::Blue, 0))
                .unwrap()
                .set_command(cmd, now);
            step_robots(&mut robots, &field, &[], &cfg, now, &mut events);
            step_ball(&mut ball, &mut robots, &field, &[], &cfg, &mut events);
            now = now.plus(cfg.substep_time());
        }
        let r = &robots[&RobotId::new(Team::Blue, 0)];
        assert!(r.vel.length() > 0.5);
        assert!(r.dribbler.holding, "ball not held");
        assert!(r.ball_contact);
        assert!((ball.state.pos_xy() - dribbler::seat_point(r)).length() < 0.005);
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::DribblerSlip { .. })));
    }

    #[test]
    fn by_force_moves_ball_to_target() {
        let cfg = config();
        let field = big_field();
        let mut robots = BTreeMap::new();
        let mut ball = ball_with(Vec3::new(0.0, 0.0, cfg.ball.radius), Vec3::ZERO);
        ball.force_target = Some(Vec3::new(1.0, 0.5, 0.0));
        let mut events = Vec::new();
        for _ in 0..3000 {
            step_ball(&mut ball, &mut robots, &field, &[], &cfg, &mut events);
        }
        assert!(
            (ball.state.pos_xy() - Vec2::new(1.0, 0.5)).length() < 0.02,
            "{}",
            ball.state.pos
        );
        assert!(ball.state.vel.length() < 0.05);
    }

    #[test]
    fn ball_left_field_is_edge_triggered() {
        let cfg = config();
        let field = FieldGeometry {
            length: 2.0,
            width: 40.0,
            goal_width: 0.0,
            ..FieldGeometry::default()
        };
        let mut robots = BTreeMap::new();
        let mut ball = ball_with(
            Vec3::new(0.9, 3.0, cfg.ball.radius),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let mut events = Vec::new();
        for _ in 0..300 {
            step_ball(&mut ball, &mut robots, &field, &[], &cfg, &mut events);
        }
        assert!(ball.state.pos.x > 1.0);
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Event::BallLeftField))
                .count(),
            1
        );
    }
}

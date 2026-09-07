//! One physics substep, split into the robot phase and the ball phase.
//!
//! OWNER: math agent. `World::step` calls these in order; see
//! `docs/design.md` §5.1.

use std::collections::BTreeMap;

use crate::ball::{Ball, BallTrajectory};
use crate::collision;
use crate::field::{FieldGeometry, WallSegment};
use crate::params::SimConfig;
use crate::rng::Rngs;
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

/// Ball phase: kicks (`kicker::try_kick` for every robot whose command asks,
/// with the realism kick errors drawn from `rngs.physics`), dribbler forces,
/// `by_force` mover, then the collision sweep loop (advance to each contact
/// by time of impact, resolve, repeat up to 4 times), then the free advance
/// of the remaining time. A ball whose centre is over a robot top uses that
/// top as its floor (`collision::support_height`). Updates every robot's
/// `ball_contact`. Emits `Kick`, `DribblerSlip`, `Goal` and `BallLeftField`
/// events.
pub fn step_ball(
    ball: &mut Ball,
    robots: &mut BTreeMap<RobotId, Robot>,
    field: &FieldGeometry,
    walls: &[WallSegment],
    config: &SimConfig,
    rngs: &mut Rngs,
    events: &mut Vec<Event>,
) {
    let dt = config.substep;
    let params = &config.ball;
    let room = field.room_half_extents();

    // 1. Kicks.
    let mut kicked_by: Option<RobotId> = None;
    for robot in robots.values_mut() {
        if robot.command.kick_speed.is_some_and(|s| s > 0.0) && robot.kicker.charged {
            if let Some(ev) = kicker::try_kick(
                robot,
                &mut ball.state,
                params,
                &config.realism,
                &mut rngs.physics,
            ) {
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
            let floor = collision::support_height(&ball.state, robots);
            let traj = BallTrajectory::from_state_on(&ball.state, params, floor);
            ball.state = traj.state_at(contact.time);
            remaining -= contact.time;
        }
        let impulse =
            collision::resolve_ball_contact(&mut ball.state, &contact, params, &config.contact);
        if let Some(robot) = contact.surface.robot().and_then(|id| robots.get_mut(&id)) {
            let j = Vec2::new(impulse.x, impulse.y);
            let contact_point = ball.state.pos_xy() - contact.normal_xy() * params.radius;
            let r = contact_point - robot.pos;
            robot.vel += j / robot.specs.mass.max(1e-6);
            robot.omega += (r.x * j.y - r.y * j.x) / robot.specs.inertia().max(1e-9);
        }
    }
    if remaining > 0.0 {
        let floor = collision::support_height(&ball.state, robots);
        ball.advance_on(remaining, params, floor);
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
    use crate::params::{Realism, RobotSpecs};
    use crate::types::{BallState, MoveCommand, RobotCommand, Team, Vec3};

    /// Default config with the exact (noise-free) kicker so tests are analytic.
    fn config() -> SimConfig {
        SimConfig {
            realism: Realism::none(),
            ..SimConfig::default()
        }
    }

    fn rngs() -> Rngs {
        Rngs::from_seed(1)
    }

    /// `step_ball` with a throwaway RNG.
    fn step(
        ball: &mut Ball,
        robots: &mut BTreeMap<RobotId, Robot>,
        field: &FieldGeometry,
        walls: &[WallSegment],
        cfg: &SimConfig,
        events: &mut Vec<Event>,
    ) {
        step_ball(ball, robots, field, walls, cfg, &mut rngs(), events);
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
            step(&mut ball, &mut robots, &field, &walls, &cfg, &mut events);
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
            step(&mut chip, &mut robots, &field, &walls, &cfg, &mut events);
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
            step(&mut flat, &mut robots, &field, &walls, &cfg, &mut events);
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
            step(&mut ball, &mut robots, &field, &[], &cfg, &mut events);
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
        // face to face: the flat chords meet at 2 * center_to_dribbler
        let c2d = a.specs.center_to_dribbler;
        assert!((b.pos - a.pos).length() >= 2.0 * c2d - 1e-6);
        assert!((b.pos - a.pos).length() < 2.0 * c2d + 0.002);
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
            step(&mut ball, &mut robots, &field, &[], &cfg, &mut events);
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
    fn chord_contact_stops_facing_robots_at_twice_center_to_dribbler() {
        // Two robots driving into each other face to face stop chord on chord;
        // with the discs-only option they stop at 2 R.
        let run = |chord: bool| -> f64 {
            let mut cfg = config();
            cfg.contact.robot_hull_chord_contacts = chord;
            let field = big_field();
            let mut robots: BTreeMap<RobotId, Robot> = BTreeMap::new();
            let forward = RobotCommand {
                movement: Some(MoveCommand::LocalVelocity {
                    forward: 0.5,
                    left: 0.0,
                    angular: 0.0,
                }),
                ..Default::default()
            };
            let a = robot(0, Vec2::ZERO, 0.0);
            let b = robot(1, Vec2::new(0.4, 0.0), std::f64::consts::PI);
            robots.insert(a.id, a);
            robots.insert(b.id, b);
            let mut events = Vec::new();
            let mut now = SimTime::ZERO;
            for _ in 0..1500 {
                for id in [RobotId::new(Team::Blue, 0), RobotId::new(Team::Blue, 1)] {
                    robots.get_mut(&id).unwrap().set_command(forward, now);
                }
                step_robots(&mut robots, &field, &[], &cfg, now, &mut events);
                now = now.plus(cfg.substep_time());
            }
            let a = &robots[&RobotId::new(Team::Blue, 0)];
            let b = &robots[&RobotId::new(Team::Blue, 1)];
            assert!(events
                .iter()
                .any(|e| matches!(e, Event::RobotCollision { .. })));
            (b.pos - a.pos).length()
        };
        let specs = RobotSpecs::default();
        let d = run(true);
        assert!(
            (d - 2.0 * specs.center_to_dribbler).abs() < 0.002,
            "chord: {d}"
        );
        let d = run(false);
        assert!((d - 2.0 * specs.radius).abs() < 0.002, "disc: {d}");
    }

    #[test]
    fn ball_lands_on_robot_top_bounces_rolls_off_and_falls() {
        let cfg = config();
        let field = big_field();
        let mut robots: BTreeMap<RobotId, Robot> = BTreeMap::new();
        let r = robot(0, Vec2::ZERO, 0.0);
        let h = r.specs.height;
        let hull_r = r.specs.radius;
        robots.insert(r.id, r);
        let rb = cfg.ball.radius;
        let drop = 0.25;
        let mut ball = ball_with(
            Vec3::new(-0.02, 0.0, h + rb + drop),
            Vec3::new(0.0, 0.0, 0.0),
        );
        let mut events = Vec::new();
        // Phase 1: fall and first bounce on the top with the configured damping.
        let mut first_bounce: Option<f64> = None;
        let mut step_no = 0;
        while first_bounce.is_none() && step_no < 1000 {
            let vz_before = ball.state.vel.z;
            step(&mut ball, &mut robots, &field, &[], &cfg, &mut events);
            step_no += 1;
            if ball.state.vel.z > 0.0 {
                let v_in = (2.0 * crate::GRAVITY * drop).sqrt();
                assert!(vz_before < 0.0);
                let expected = v_in * (1.0 - cfg.contact.ball_robot_top_normal);
                assert!(
                    (ball.state.vel.z - expected).abs() < 0.03,
                    "bounce vz {} vs {expected}",
                    ball.state.vel.z
                );
                first_bounce = Some(ball.state.vel.z);
            }
            assert!(ball.state.pos.z >= h + rb - 1e-9, "sank into the top");
        }
        assert!(first_bounce.is_some(), "never bounced");
        // Phase 2: it comes to rest on the top (its floor is the robot).
        let mut rested = false;
        for _ in 0..2000 {
            step(&mut ball, &mut robots, &field, &[], &cfg, &mut events);
            assert!(ball.state.pos.z >= h + rb - 1e-9);
            if ball.state.pos.z == h + rb && ball.state.vel.z == 0.0 {
                rested = true;
                break;
            }
        }
        assert!(rested, "did not settle on the top: {:?}", ball.state);
        assert!(ball.state.pos_xy().length() < hull_r);
        // Phase 3: push it toward the edge; it rolls on the top, leaves the
        // footprint, falls, and ends on the carpet outside the hull.
        ball.state.vel = Vec3::new(0.6, 0.0, 0.0);
        ball.state.spin = ball.state.vel_xy();
        let mut rolled_on_top = false;
        let mut fell = false;
        let mut max_speed_after_edge: f64 = 0.0;
        for _ in 0..4000 {
            let before = ball.state;
            step(&mut ball, &mut robots, &field, &[], &cfg, &mut events);
            let inside = ball.state.pos_xy().length() < hull_r;
            if inside && ball.state.pos.z == h + rb && ball.state.vel.z == 0.0 {
                rolled_on_top = true;
                assert!(
                    ball.state.vel.x < before.vel.x + 1e-12,
                    "no acceleration on the top"
                );
            }
            if !inside {
                fell = true;
                assert!(ball.state.pos.z <= h + rb + 1e-9);
                max_speed_after_edge = max_speed_after_edge.max(ball.state.vel.length());
            }
            if ball.state.pos.z == rb && ball.state.vel.z == 0.0 && !inside {
                break;
            }
        }
        assert!(rolled_on_top && fell);
        assert_eq!(
            ball.state.pos.z, rb,
            "did not reach the carpet: {:?}",
            ball.state
        );
        assert!(ball.state.pos_xy().length() > hull_r + rb - 1e-9);
        // the drop converts height into speed: at most sqrt(v^2 + 2 g h) (no energy gain)
        let bound = (0.6f64 * 0.6 + 2.0 * crate::GRAVITY * h).sqrt();
        assert!(
            max_speed_after_edge <= bound + 1e-6,
            "{max_speed_after_edge} > {bound}"
        );
        assert!(events.is_empty());
    }

    #[test]
    fn ball_on_a_moving_robot_top_picks_up_surface_velocity_on_landing() {
        let cfg = config();
        let field = big_field();
        let mut robots: BTreeMap<RobotId, Robot> = BTreeMap::new();
        let mut r = robot(0, Vec2::ZERO, 0.0);
        r.vel = Vec2::new(1.0, 0.0);
        let h = r.specs.height;
        robots.insert(r.id, r);
        let rb = cfg.ball.radius;
        let mut ball = ball_with(Vec3::new(0.0, 0.0, h + rb), Vec3::new(0.0, 0.0, -1.0));
        let mut events = Vec::new();
        step(&mut ball, &mut robots, &field, &[], &cfg, &mut events);
        let kt = cfg.contact.ball_robot_top_tangent;
        assert!((ball.state.vel.x - kt * 1.0).abs() < 1e-9);
        // bounce, minus gravity over the rest of the substep
        let vz = 1.0 * (1.0 - cfg.contact.ball_robot_top_normal);
        assert!((ball.state.vel.z - vz).abs() < crate::GRAVITY * cfg.substep + 1e-9);
        assert!(ball.state.vel.z > 0.0);
        // the robot felt the vertical impulse only as a horizontal drag
        let r = &robots[&RobotId::new(Team::Blue, 0)];
        assert!(r.vel.x < 1.0);
    }

    #[test]
    fn slow_ball_at_the_face_of_a_dribbling_robot_is_captured() {
        // A ball arriving at 1-2 m/s at the kicker face of a robot with the
        // dribbler on ends up held at the seat (dynamics.md §6: 48 % of face
        // contacts are captures; §10.5).
        for v_in in [1.0, 1.5, 2.0] {
            let cfg = config();
            assert_eq!(cfg.contact.ball_kicker_normal, 0.8);
            let field = big_field();
            let mut robots: BTreeMap<RobotId, Robot> = BTreeMap::new();
            let mut a = robot(0, Vec2::ZERO, 0.0);
            assert_eq!(a.specs.dribbler.hold_accel, 3.0);
            assert_eq!(a.specs.dribbler.seat_depth, 0.0);
            assert_eq!(a.specs.shoot_radius, 0.0965);
            a.set_command(
                RobotCommand {
                    movement: None,
                    dribbler_rpm: Some(10000.0),
                    ..Default::default()
                },
                SimTime::ZERO,
            );
            robots.insert(a.id, a);
            let mut ball = ball_with(
                Vec3::new(0.4, 0.0, cfg.ball.radius),
                Vec3::new(-v_in, 0.0, 0.0),
            );
            ball.state.spin = ball.state.vel_xy();
            let mut events = Vec::new();
            let mut now = SimTime::ZERO;
            for _ in 0..1500 {
                let cmd = robots[&RobotId::new(Team::Blue, 0)].command;
                robots
                    .get_mut(&RobotId::new(Team::Blue, 0))
                    .unwrap()
                    .set_command(cmd, now);
                step_robots(&mut robots, &field, &[], &cfg, now, &mut events);
                step(&mut ball, &mut robots, &field, &[], &cfg, &mut events);
                now = now.plus(cfg.substep_time());
            }
            let r = &robots[&RobotId::new(Team::Blue, 0)];
            assert!(
                r.dribbler.holding,
                "v_in={v_in}: ball not held, {:?}",
                ball.state
            );
            assert!(r.ball_contact);
            assert!(
                (ball.state.pos_xy() - dribbler::seat_point(r)).length() < 0.003,
                "v_in={v_in}: ball at {:?}",
                ball.state.pos
            );
            assert!(ball.state.vel_xy().length() < 0.05);
        }
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
            step(&mut ball, &mut robots, &field, &[], &cfg, &mut events);
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
            step(&mut ball, &mut robots, &field, &[], &cfg, &mut events);
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

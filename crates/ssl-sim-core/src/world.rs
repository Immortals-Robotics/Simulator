//! The simulation world: owns everything, advances in fixed substeps, applies
//! control requests between substeps, and hands vision output to the caller.
//!
//! OWNER: lead. Integration point for all other modules.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ball::Ball;
use crate::field::{default_formation, Division, FieldGeometry, WallSegment};
use crate::params::{Realism, RobotSpecs, SimConfig};
use crate::physics;
use crate::rng::{normal, Rngs};
use crate::robot::Robot;
use crate::types::{
    BallState, Event, RobotCommand, RobotId, RobotState, SimError, SimTime, Team, TeleportBall,
    TeleportRobot, Vec2, Vec3,
};
use crate::vision::{VisionModel, VisionOutput};

/// Ground truth of the whole world at one instant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldSnapshot {
    /// Sim time.
    pub time: SimTime,
    /// Substep counter.
    pub frame: u64,
    /// Ball.
    pub ball: BallState,
    /// Robots in `RobotId` order.
    pub robots: Vec<RobotState>,
}

/// The simulation.
#[derive(Debug, Clone)]
pub struct World {
    config: SimConfig,
    division: Division,
    field: FieldGeometry,
    walls: Vec<WallSegment>,
    time: SimTime,
    frame: u64,
    ball: Ball,
    robots: BTreeMap<RobotId, Robot>,
    rngs: Rngs,
    vision: VisionModel,
    events: Vec<Event>,
}

impl World {
    /// Create a world for a division with the configured default robots.
    pub fn new(config: SimConfig, division: Division) -> Self {
        let field = FieldGeometry::division(division);
        Self::with_field(config, division, field)
    }

    /// Create a world with explicit field geometry.
    pub fn with_field(config: SimConfig, division: Division, field: FieldGeometry) -> Self {
        let walls = field.walls(config.wall_height);
        let rngs = Rngs::from_seed(config.seed);
        let vision = VisionModel::new(config.vision.clone(), config.realism.clone(), &field);
        let ball = Ball::at_rest(Vec2::ZERO, &config.ball);
        let mut world = Self {
            division,
            field,
            walls,
            time: SimTime::ZERO,
            frame: 0,
            ball,
            robots: BTreeMap::new(),
            rngs,
            vision,
            events: Vec::new(),
            config,
        };
        world.spawn_default_robots();
        world
    }

    /// Place `initial_robots_per_team` robots per team in the default formation.
    pub fn spawn_default_robots(&mut self) {
        let n = self.config.initial_robots_per_team.min(16);
        for number in 0..n {
            for team in [Team::Blue, Team::Yellow] {
                if let Some(p) = default_formation(self.division, number) {
                    let (pos, orientation) = match team {
                        Team::Blue => (Vec2::new(-p.x.abs(), p.y), 0.0),
                        Team::Yellow => (Vec2::new(p.x.abs(), p.y), std::f64::consts::PI),
                    };
                    let id = RobotId::new(team, number);
                    let robot = self.create_robot(id, pos, orientation);
                    self.robots.insert(id, robot);
                }
            }
        }
    }

    /// Default specs for a team.
    pub fn default_specs(&self, team: Team) -> RobotSpecs {
        match team {
            Team::Blue => self.config.blue_specs,
            Team::Yellow => self.config.yellow_specs,
        }
    }

    /// A new robot with the team's default specs and its own dribbler
    /// holding budget drawn from the seeded physics stream:
    /// `max(hold_accel + N(0, hold_accel_stddev), hold_accel_min)`.
    fn create_robot(&mut self, id: RobotId, pos: Vec2, orientation: f64) -> Robot {
        let specs = self.default_specs(id.team);
        let mut robot = Robot::new(id, specs, pos, orientation);
        robot.hold_accel_actual = Self::draw_hold_accel(&mut self.rngs, &specs);
        robot
    }

    fn draw_hold_accel(rngs: &mut Rngs, specs: &RobotSpecs) -> f64 {
        let d = &specs.dribbler;
        (d.hold_accel + normal(&mut rngs.physics, d.hold_accel_stddev)).max(d.hold_accel_min)
    }

    // ----- accessors -----

    /// Sim time.
    pub fn time(&self) -> SimTime {
        self.time
    }

    /// Substep counter.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Config.
    pub fn config(&self) -> &SimConfig {
        &self.config
    }

    /// Field geometry.
    pub fn field(&self) -> &FieldGeometry {
        &self.field
    }

    /// Division.
    pub fn division(&self) -> Division {
        self.division
    }

    /// Ball.
    pub fn ball(&self) -> &Ball {
        &self.ball
    }

    /// Robots.
    pub fn robots(&self) -> &BTreeMap<RobotId, Robot> {
        &self.robots
    }

    /// Mutable robot.
    pub fn robot_mut(&mut self, id: RobotId) -> Option<&mut Robot> {
        self.robots.get_mut(&id)
    }

    /// Vision model.
    pub fn vision(&self) -> &VisionModel {
        &self.vision
    }

    /// Random streams (the net layer uses `packet_loss`).
    pub fn rngs_mut(&mut self) -> &mut Rngs {
        &mut self.rngs
    }

    /// Ground truth.
    pub fn snapshot(&self) -> WorldSnapshot {
        WorldSnapshot {
            time: self.time,
            frame: self.frame,
            ball: self.ball.state,
            robots: self.robots.values().map(Robot::state).collect(),
        }
    }

    /// Events accumulated since the last call; clears the buffer.
    pub fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    // ----- stepping -----

    /// Advance exactly one substep.
    pub fn step(&mut self) {
        let dt = self.config.substep;
        physics::step_robots(
            &mut self.robots,
            &self.field,
            &self.walls,
            &self.config,
            self.time,
            &mut self.events,
        );
        physics::step_ball(
            &mut self.ball,
            &mut self.robots,
            &self.field,
            &self.walls,
            &self.config,
            &mut self.rngs,
            &mut self.events,
        );
        self.time = self.time.plus(SimTime::from_secs_f64(dt));
        self.frame += 1;
        self.vision.maybe_capture(
            self.time,
            &self.ball,
            &self.robots,
            &self.field,
            &self.config.ball,
            &mut self.rngs,
        );
    }

    /// Advance by `duration`, which must be a whole number of substeps.
    pub fn step_for(&mut self, duration: SimTime) -> Result<u64, SimError> {
        let sub = self.config.substep_time().as_nanos().max(1);
        let n = duration.as_nanos() / sub;
        if n * sub != duration.as_nanos() {
            return Err(SimError::InvalidStep(format!(
                "{} ns is not a multiple of the {} ns substep",
                duration.as_nanos(),
                sub
            )));
        }
        for _ in 0..n {
            self.step();
        }
        Ok(n)
    }

    /// Vision outputs whose delay has elapsed as of the current time.
    pub fn drain_vision(&mut self) -> Vec<VisionOutput> {
        self.vision.drain_due(self.time)
    }

    // ----- control -----

    /// Store a robot command (replacement semantics).
    pub fn set_robot_command(
        &mut self,
        id: RobotId,
        command: RobotCommand,
    ) -> Result<(), SimError> {
        let now = self.time;
        self.robots
            .get_mut(&id)
            .map(|r| r.set_command(command, now))
            .ok_or(SimError::UnknownRobot(id))
    }

    /// Teleport / push the ball.
    pub fn teleport_ball(&mut self, req: TeleportBall) -> Result<(), SimError> {
        if req.by_force {
            if req.velocity.is_some_and(|v| v.length_squared() > 0.0) {
                return Err(SimError::VelocityForce("by_force with velocity".into()));
            }
            self.ball.force_target = req.position;
            return Ok(());
        }
        self.ball.force_target = None;
        if req.teleport_safely {
            let Some(p) = req.position else {
                return Err(SimError::TeleportSafelyPartial(
                    "teleport_safely needs x and y".into(),
                ));
            };
            self.evict_robots_from(Vec2::new(p.x, p.y));
        }
        if let Some(p) = req.position {
            self.ball.state.pos = Vec3::new(p.x, p.y, p.z.max(self.config.ball.radius));
        }
        if let Some(v) = req.velocity {
            self.ball.state.vel = v;
            self.ball.state.spin = Vec2::ZERO;
        }
        if req.roll {
            self.ball.state.spin = self.ball.state.vel_xy();
        }
        Ok(())
    }

    /// Teleport / push / add / remove a robot.
    pub fn teleport_robot(&mut self, req: TeleportRobot) -> Result<(), SimError> {
        if req.present == Some(false) {
            self.robots.remove(&req.id);
            return Ok(());
        }
        if !self.robots.contains_key(&req.id) {
            if req.present != Some(true) {
                return Err(SimError::UnknownRobot(req.id));
            }
            let Some(pos) = req.position else {
                return Err(SimError::CreateNoPosRobot(format!(
                    "{} needs x and y",
                    req.id
                )));
            };
            let robot = self.create_robot(req.id, pos, req.orientation.unwrap_or(0.0));
            self.robots.insert(req.id, robot);
        }
        let robot = self.robots.get_mut(&req.id).expect("inserted above");
        if req.by_force {
            if req.velocity.is_some_and(|v| v.length_squared() > 0.0) {
                return Err(SimError::VelocityForce("by_force with velocity".into()));
            }
            robot.force_target = req.position;
            return Ok(());
        }
        robot.force_target = None;
        if let Some(p) = req.position {
            robot.pos = p;
        }
        if let Some(o) = req.orientation {
            robot.orientation = o;
        }
        if let Some(v) = req.velocity {
            robot.vel = v;
            robot.setpoint = robot.local_velocity();
        }
        if let Some(w) = req.angular_velocity {
            robot.omega = w;
            robot.setpoint.omega = w;
        }
        robot.dribbler.holding = false;
        Ok(())
    }

    /// Update specs of one robot (must exist), keeping its state. The robot's
    /// dribbler holding draw keeps its deviation from the (possibly new) mean.
    pub fn set_robot_specs(&mut self, id: RobotId, specs: RobotSpecs) -> Result<(), SimError> {
        validate_specs(&specs)?;
        match self.robots.get_mut(&id) {
            Some(r) => {
                let deviation = r.hold_accel_actual - r.specs.dribbler.hold_accel;
                r.hold_accel_actual =
                    (specs.dribbler.hold_accel + deviation).max(specs.dribbler.hold_accel_min);
                r.specs = specs;
                Ok(())
            }
            None => Err(SimError::UnknownRobot(id)),
        }
    }

    /// Update the default specs for a team (used for robots created later).
    pub fn set_default_specs(&mut self, team: Team, specs: RobotSpecs) -> Result<(), SimError> {
        validate_specs(&specs)?;
        match team {
            Team::Blue => self.config.blue_specs = specs,
            Team::Yellow => self.config.yellow_specs = specs,
        }
        Ok(())
    }

    /// Replace the realism config (vision and dribbler mode).
    pub fn set_realism(&mut self, realism: Realism) {
        let glue = !realism.simulate_dribbling;
        for r in self.robots.values_mut() {
            r.specs.dribbler.glue = glue;
        }
        self.config.blue_specs.dribbler.glue = glue;
        self.config.yellow_specs.dribbler.glue = glue;
        self.config.realism = realism.clone();
        self.vision.set_realism(realism);
    }

    /// Replace the field geometry without touching robots or the ball.
    pub fn set_field(&mut self, field: FieldGeometry) {
        self.walls = field.walls(self.config.wall_height);
        self.field = field;
    }

    /// Replace the camera rig; an empty list restores the default placement.
    pub fn set_cameras(&mut self, cameras: Vec<crate::params::CameraConfig>) {
        self.vision.set_cameras(cameras, &self.field);
    }

    /// Reset the ball to the centre and robots to the default formation.
    pub fn reset(&mut self) {
        self.ball = Ball::at_rest(Vec2::ZERO, &self.config.ball);
        self.robots.clear();
        self.spawn_default_robots();
    }

    /// Hash of the full dynamic state, for determinism tests.
    pub fn state_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.time.hash(&mut h);
        let b = &self.ball.state;
        for v in [
            b.pos.x, b.pos.y, b.pos.z, b.vel.x, b.vel.y, b.vel.z, b.spin.x, b.spin.y,
        ] {
            v.to_bits().hash(&mut h);
        }
        for r in self.robots.values() {
            r.id.hash(&mut h);
            for v in [r.pos.x, r.pos.y, r.orientation, r.vel.x, r.vel.y, r.omega] {
                v.to_bits().hash(&mut h);
            }
        }
        h.finish()
    }

    /// ER-Force `teleport_safely`: robots overlapping the target circle are
    /// pushed away along the ball->robot direction until clear; robots within
    /// 1.5 m have their velocity zeroed.
    fn evict_robots_from(&mut self, target: Vec2) {
        let ball_r = self.config.ball.radius;
        let others: Vec<(RobotId, Vec2, f64)> = self
            .robots
            .values()
            .map(|r| (r.id, r.pos, r.specs.radius))
            .collect();
        for r in self.robots.values_mut() {
            let d = r.pos - target;
            if d.length() < 1.5 {
                r.vel = Vec2::ZERO;
                r.omega = 0.0;
                r.setpoint = Default::default();
            }
            let clearance = r.specs.radius + ball_r;
            if d.length() < clearance {
                let dir = if d.length_squared() > 1e-12 {
                    d.normalize()
                } else {
                    Vec2::X
                };
                let mut candidate = target + dir * clearance;
                // step outward until not overlapping any other robot
                for _ in 0..32 {
                    let overlaps = others.iter().any(|(oid, opos, orad)| {
                        *oid != r.id && (candidate - *opos).length() < orad + r.specs.radius
                    });
                    if !overlaps {
                        break;
                    }
                    candidate += dir * (2.0 * clearance);
                }
                r.pos = candidate;
            }
        }
    }
}

fn validate_specs(s: &RobotSpecs) -> Result<(), SimError> {
    let ok = s.radius > 0.02
        && s.height > 0.02
        && s.mass > 0.1
        && s.center_to_dribbler > 0.0
        && s.center_to_dribbler < s.radius
        && s.dribbler_width > 0.0
        && s.limits.vel_absolute_max > 0.0
        && s.limits.vel_angular_max > 0.0;
    if ok {
        Ok(())
    } else {
        Err(SimError::InvalidSpec(format!("{s:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::DribblerParams;
    use crate::types::MoveCommand;

    fn seeded(seed: u64) -> World {
        let config = SimConfig {
            seed,
            initial_robots_per_team: 6,
            ..SimConfig::default()
        };
        World::new(config, Division::A)
    }

    /// Drive a scripted scenario: blue 0 (moved to open space) carries the
    /// ball on its dribbler while turning, then chips it; the others wander
    /// slowly. Returns the number of kick events seen.
    fn scripted(world: &mut World, substeps: usize) -> usize {
        let carrier = RobotId::new(Team::Blue, 0);
        world
            .teleport_robot(TeleportRobot {
                id: carrier,
                position: Some(Vec2::new(0.0, -3.0)),
                orientation: Some(0.0),
                velocity: Some(Vec2::ZERO),
                angular_velocity: Some(0.0),
                present: None,
                by_force: false,
            })
            .unwrap();
        let pos = world.robots()[&carrier].pos;
        let heading = world.robots()[&carrier].heading();
        let seat = pos + heading * world.robots()[&carrier].specs.shoot_radius;
        world
            .teleport_ball(TeleportBall {
                position: Some(Vec3::new(seat.x, seat.y, 0.0)),
                velocity: Some(Vec3::ZERO),
                ..Default::default()
            })
            .unwrap();
        let mut kicks = 0;
        for i in 0..substeps {
            if i % 10 == 0 {
                let t = i as f64 * world.config().substep;
                for n in 0..6u8 {
                    for team in [Team::Blue, Team::Yellow] {
                        let id = RobotId::new(team, n);
                        let phase = t * 0.9 + n as f64;
                        let cmd = if id == carrier {
                            RobotCommand {
                                movement: Some(MoveCommand::LocalVelocity {
                                    forward: 0.6,
                                    left: 0.0,
                                    angular: 0.8 * (t * 2.0).sin(),
                                }),
                                kick_speed: (4000..4100).contains(&i).then_some(4.0),
                                kick_angle_deg: 45.0,
                                dribbler_rpm: Some(10000.0),
                            }
                        } else {
                            RobotCommand {
                                movement: Some(MoveCommand::LocalVelocity {
                                    forward: 0.5 * phase.cos(),
                                    left: 0.4 * phase.sin(),
                                    angular: 1.5 * (phase * 0.5).sin(),
                                }),
                                dribbler_rpm: Some(3000.0),
                                ..Default::default()
                            }
                        };
                        world.set_robot_command(id, cmd).unwrap();
                    }
                }
            }
            world.step();
            kicks += world
                .take_events()
                .iter()
                .filter(|e| matches!(e, Event::Kick { .. }))
                .count();
            world.drain_vision();
        }
        kicks
    }

    #[test]
    fn same_seed_and_commands_give_identical_state_hashes() {
        assert!(SimConfig::default().realism.kick_direction_stddev > 0.0);
        let mut a = seeded(7);
        let mut b = seeded(7);
        assert_eq!(a.state_hash(), b.state_hash());
        let ka = scripted(&mut a, 10_000);
        let kb = scripted(&mut b, 10_000);
        assert_eq!(ka, 1, "the scripted kick must fire");
        assert_eq!(ka, kb);
        assert_eq!(a.state_hash(), b.state_hash());
        assert_eq!(a.snapshot(), b.snapshot());
        // a different seed perturbs the (noisy) kick and the holding draws
        let mut c = seeded(8);
        scripted(&mut c, 10_000);
        assert_ne!(a.state_hash(), c.state_hash());
    }

    #[test]
    fn per_robot_hold_accel_is_drawn_clamped_and_differs_between_robots() {
        let world = seeded(3);
        let d = DribblerParams::default();
        assert!(d.hold_accel_stddev > 0.0);
        let draws: Vec<f64> = world
            .robots()
            .values()
            .map(|r| r.hold_accel_actual)
            .collect();
        assert_eq!(draws.len(), 12);
        assert!(draws.iter().all(|&h| h >= d.hold_accel_min));
        let distinct = draws
            .iter()
            .filter(|&&h| draws.iter().filter(|&&o| (o - h).abs() < 1e-12).count() == 1)
            .count();
        assert!(distinct >= 10, "draws {draws:?}");
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        assert!((mean - d.hold_accel).abs() < 1.0, "mean {mean}");
        // same seed: same draws; the draws are part of the physics stream
        let again = seeded(3);
        let draws2: Vec<f64> = again
            .robots()
            .values()
            .map(|r| r.hold_accel_actual)
            .collect();
        assert_eq!(draws, draws2);
        // zero spread: everyone gets the mean; a huge spread is clamped from below
        let mut cfg = SimConfig {
            seed: 3,
            initial_robots_per_team: 4,
            ..SimConfig::default()
        };
        cfg.blue_specs.dribbler.hold_accel_stddev = 0.0;
        cfg.yellow_specs.dribbler.hold_accel_stddev = 100.0;
        cfg.yellow_specs.dribbler.hold_accel_min = 2.5;
        let world = World::new(cfg, Division::A);
        for r in world.robots().values() {
            match r.id.team {
                Team::Blue => assert_eq!(r.hold_accel_actual, d.hold_accel),
                Team::Yellow => assert!(r.hold_accel_actual >= 2.5),
            }
        }
        // a robot added later gets its own draw, and spec updates keep the deviation
        let mut world = seeded(5);
        let id = RobotId::new(Team::Yellow, 15);
        world
            .teleport_robot(TeleportRobot {
                id,
                position: Some(Vec2::new(3.0, 3.0)),
                orientation: None,
                velocity: None,
                angular_velocity: None,
                present: Some(true),
                by_force: false,
            })
            .unwrap();
        let drawn = world.robots()[&id].hold_accel_actual;
        assert!(drawn >= d.hold_accel_min);
        let mut specs = world.default_specs(Team::Yellow);
        specs.dribbler.hold_accel = 5.0;
        world.set_robot_specs(id, specs).unwrap();
        let after = world.robots()[&id].hold_accel_actual;
        assert!((after - (drawn - d.hold_accel + 5.0)).abs() < 1e-12 || after == d.hold_accel_min);
    }
}

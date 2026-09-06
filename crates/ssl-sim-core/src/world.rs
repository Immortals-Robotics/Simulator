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
use crate::rng::Rngs;
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
                    self.robots.insert(
                        id,
                        Robot::new(id, self.default_specs(team), pos, orientation),
                    );
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
            let specs = self.default_specs(req.id.team);
            self.robots.insert(
                req.id,
                Robot::new(req.id, specs, pos, req.orientation.unwrap_or(0.0)),
            );
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

    /// Update specs of one robot (must exist), keeping its state.
    pub fn set_robot_specs(&mut self, id: RobotId, specs: RobotSpecs) -> Result<(), SimError> {
        validate_specs(&specs)?;
        match self.robots.get_mut(&id) {
            Some(r) => {
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

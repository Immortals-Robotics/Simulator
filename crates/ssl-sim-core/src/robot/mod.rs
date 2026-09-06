//! Robot state and actuator sub-models.
//!
//! `Robot` (this file) is owned by the lead. `drive` and `dribbler` are owned
//! by the math agent, `kicker` by general agent A.

pub mod dribbler;
pub mod drive;
pub mod kicker;

use crate::params::RobotSpecs;
use crate::types::{rotate, MoveCommand, RobotCommand, RobotId, RobotState, SimTime, Vec2};

/// Firmware-side velocity setpoint after rate limiting, robot frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LocalTwist {
    /// Forward [m/s].
    pub vx: f64,
    /// Left [m/s].
    pub vy: f64,
    /// CCW [rad/s].
    pub omega: f64,
}

/// One simulated robot.
#[derive(Debug, Clone)]
pub struct Robot {
    /// Identity.
    pub id: RobotId,
    /// Specification (live-updatable).
    pub specs: RobotSpecs,
    /// Position of the centre [m].
    pub pos: Vec2,
    /// Orientation [rad] of the kicker direction, CCW from world +x.
    pub orientation: f64,
    /// World velocity [m/s].
    pub vel: Vec2,
    /// Angular velocity [rad/s].
    pub omega: f64,
    /// Last received command.
    pub command: RobotCommand,
    /// Sim time at which `command` was received.
    pub command_time: SimTime,
    /// Rate-limited setpoint the "firmware" is currently tracking.
    pub setpoint: LocalTwist,
    /// Kicker charge/cooldown state.
    pub kicker: kicker::KickerState,
    /// Dribbler state.
    pub dribbler: dribbler::DribblerState,
    /// Active `by_force` target, if any.
    pub force_target: Option<Vec2>,
    /// Ball currently seated on the dribbler (break beam), updated every substep.
    pub ball_contact: bool,
}

impl Robot {
    /// Create a robot at rest.
    pub fn new(id: RobotId, specs: RobotSpecs, pos: Vec2, orientation: f64) -> Self {
        Self {
            id,
            specs,
            pos,
            orientation,
            vel: Vec2::ZERO,
            omega: 0.0,
            command: RobotCommand::default(),
            command_time: SimTime::ZERO,
            setpoint: LocalTwist::default(),
            kicker: kicker::KickerState::default(),
            dribbler: dribbler::DribblerState::default(),
            force_target: None,
            ball_contact: false,
        }
    }

    /// Unit vector of the kicker direction.
    pub fn heading(&self) -> Vec2 {
        Vec2::from_angle(self.orientation)
    }

    /// World -> robot frame.
    pub fn to_local(&self, v: Vec2) -> Vec2 {
        rotate(v, -self.orientation)
    }

    /// Robot -> world frame.
    pub fn to_world(&self, v: Vec2) -> Vec2 {
        rotate(v, self.orientation)
    }

    /// Current velocity in the robot frame.
    pub fn local_velocity(&self) -> LocalTwist {
        let v = self.to_local(self.vel);
        LocalTwist {
            vx: v.x,
            vy: v.y,
            omega: self.omega,
        }
    }

    /// Velocity of the hull surface point `p_world` [m/s] (includes rotation).
    pub fn surface_velocity(&self, p_world: Vec2) -> Vec2 {
        let r = p_world - self.pos;
        self.vel + Vec2::new(-r.y, r.x) * self.omega
    }

    /// World position of the centre of the kicker face.
    pub fn kicker_face_center(&self) -> Vec2 {
        self.pos + self.heading() * self.specs.center_to_dribbler
    }

    /// Store a new command.
    pub fn set_command(&mut self, command: RobotCommand, now: SimTime) {
        self.command = command;
        self.command_time = now;
    }

    /// Commanded local twist, `None` when there is no movement command or the
    /// command has timed out.
    pub fn commanded_twist(&self, now: SimTime, timeout: f64) -> Option<LocalTwist> {
        if now.minus(self.command_time).as_secs_f64() > timeout {
            return None;
        }
        match self.command.movement? {
            MoveCommand::LocalVelocity {
                forward,
                left,
                angular,
            } => Some(LocalTwist {
                vx: forward,
                vy: left,
                omega: angular,
            }),
            MoveCommand::GlobalVelocity { x, y, angular } => {
                let v = self.to_local(Vec2::new(x, y));
                Some(LocalTwist {
                    vx: v.x,
                    vy: v.y,
                    omega: angular,
                })
            }
            MoveCommand::WheelVelocity {
                front_right,
                back_right,
                back_left,
                front_left,
            } => Some(drive::forward_kinematics(
                [front_right, back_right, back_left, front_left],
                &self.specs,
            )),
        }
    }

    /// Ground-truth snapshot.
    pub fn state(&self) -> RobotState {
        RobotState {
            id: self.id,
            pos: self.pos,
            orientation: self.orientation,
            vel: self.vel,
            angular_velocity: self.omega,
            ball_contact: self.ball_contact,
            kicker_charged: self.kicker.charged,
            dribbling: self.dribbler.active,
        }
    }
}

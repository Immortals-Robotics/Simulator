//! Shared value types: identifiers, time, commands, states, events, errors.
//!
//! Everything here is plain data; it is the contract between the physics
//! modules, the vision model, and the network layer.

use std::fmt;

use serde::{Deserialize, Serialize};

/// 2D vector in metres (f64 for closed-form ball math precision).
pub type Vec2 = glam::DVec2;
/// 3D vector in metres, z up.
pub type Vec3 = glam::DVec3;

/// Team colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Team {
    /// Blue team (defends -x by default).
    Blue,
    /// Yellow team (defends +x by default).
    Yellow,
}

impl Team {
    /// The other team.
    pub fn opponent(self) -> Team {
        match self {
            Team::Blue => Team::Yellow,
            Team::Yellow => Team::Blue,
        }
    }
}

/// Identifies one robot. Ordered so `BTreeMap<RobotId, _>` iterates blue 0..15 then yellow 0..15.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RobotId {
    /// Team colour.
    pub team: Team,
    /// Robot number as printed on the pattern (0..=15).
    pub number: u8,
}

impl RobotId {
    /// Construct an id.
    pub const fn new(team: Team, number: u8) -> Self {
        Self { team, number }
    }
}

impl fmt::Display for RobotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let team = match self.team {
            Team::Blue => 'B',
            Team::Yellow => 'Y',
        };
        write!(f, "{team}{}", self.number)
    }
}

/// Simulation time in nanoseconds since the simulation started.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub struct SimTime(pub u64);

impl SimTime {
    /// Zero.
    pub const ZERO: SimTime = SimTime(0);

    /// From seconds (rounded to the nearest nanosecond).
    pub fn from_secs_f64(secs: f64) -> Self {
        SimTime((secs * 1e9).round().max(0.0) as u64)
    }

    /// From a whole number of milliseconds.
    pub const fn from_millis(ms: u64) -> Self {
        SimTime(ms * 1_000_000)
    }

    /// As seconds.
    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 * 1e-9
    }

    /// Nanoseconds.
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// Saturating addition.
    pub fn plus(self, other: SimTime) -> SimTime {
        SimTime(self.0.saturating_add(other.0))
    }

    /// Saturating subtraction.
    pub fn minus(self, other: SimTime) -> SimTime {
        SimTime(self.0.saturating_sub(other.0))
    }
}

/// Velocity command for one robot.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MoveCommand {
    /// Robot-frame velocity: forward (+x), left (+y), angular (CCW).
    LocalVelocity {
        /// Forward speed [m/s].
        forward: f64,
        /// Leftward speed [m/s].
        left: f64,
        /// Angular speed [rad/s], CCW positive.
        angular: f64,
    },
    /// World-frame velocity.
    GlobalVelocity {
        /// World x speed [m/s].
        x: f64,
        /// World y speed [m/s].
        y: f64,
        /// Angular speed [rad/s], CCW positive.
        angular: f64,
    },
    /// Wheel surface speeds [m/s], protocol order.
    WheelVelocity {
        /// Front right wheel.
        front_right: f64,
        /// Back right wheel.
        back_right: f64,
        /// Back left wheel.
        back_left: f64,
        /// Front left wheel.
        front_left: f64,
    },
}

/// One robot control message. Replacement semantics: every message fully
/// replaces the previous command (kick and dribbler included), matching grSim
/// and ER-Force behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct RobotCommand {
    /// Desired motion. `None` = motors off (coast to stop).
    pub movement: Option<MoveCommand>,
    /// Kick speed [m/s]; `<= 0` or `None` = no kick.
    pub kick_speed: Option<f64>,
    /// Kick elevation angle [deg]; 0 = straight, 45 = typical chip.
    pub kick_angle_deg: f64,
    /// Dribbler speed [rpm]; `<= 0` or `None` = off.
    pub dribbler_rpm: Option<f64>,
}

/// Ball teleport request (protocol `TeleportBall`). All positions are metres.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct TeleportBall {
    /// Target position; `z` defaults to the ball radius when only x/y are set.
    pub position: Option<Vec3>,
    /// Target velocity.
    pub velocity: Option<Vec3>,
    /// Move robots out of the way and stop robots nearby.
    pub teleport_safely: bool,
    /// Set the spin so the ball is rolling with its velocity.
    pub roll: bool,
    /// Push the ball with a force instead of teleporting; stays active until
    /// another request with `by_force == false`.
    pub by_force: bool,
}

/// Robot teleport request (protocol `TeleportRobot`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TeleportRobot {
    /// Which robot.
    pub id: RobotId,
    /// Target position [m].
    pub position: Option<Vec2>,
    /// Target orientation [rad].
    pub orientation: Option<f64>,
    /// Target world velocity [m/s].
    pub velocity: Option<Vec2>,
    /// Target angular velocity [rad/s].
    pub angular_velocity: Option<f64>,
    /// `Some(true)` adds the robot if absent, `Some(false)` removes it.
    pub present: Option<bool>,
    /// Push with a force instead of teleporting.
    pub by_force: bool,
}

/// Full kinematic state of the ball.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct BallState {
    /// Centre position [m]; `z` is the height of the centre above the floor.
    pub pos: Vec3,
    /// Velocity [m/s].
    pub vel: Vec3,
    /// Ground-contact spin expressed as a linear velocity [m/s]: the ball is
    /// rolling exactly when `vel.xy() == spin`. Positive spin <=> positive velocity.
    pub spin: Vec2,
}

impl BallState {
    /// Horizontal velocity.
    pub fn vel_xy(&self) -> Vec2 {
        Vec2::new(self.vel.x, self.vel.y)
    }

    /// Horizontal position.
    pub fn pos_xy(&self) -> Vec2 {
        Vec2::new(self.pos.x, self.pos.y)
    }
}

/// Ground-truth state of one robot.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RobotState {
    /// Robot id.
    pub id: RobotId,
    /// Position of the robot centre [m].
    pub pos: Vec2,
    /// Orientation [rad], angle of robot +x (kicker) from world +x, CCW.
    pub orientation: f64,
    /// World velocity [m/s].
    pub vel: Vec2,
    /// Angular velocity [rad/s].
    pub angular_velocity: f64,
    /// Break-beam / infrared sensor: ball seated at the dribbler.
    pub ball_contact: bool,
    /// Kicker capacitor charged.
    pub kicker_charged: bool,
    /// Dribbler currently running.
    pub dribbling: bool,
}

/// Something notable that happened during a substep. Consumed by the network
/// layer (legacy `Robots_Status`) and by tools.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Event {
    /// A robot kicked the ball.
    Kick {
        /// Kicking robot.
        robot: RobotId,
        /// Ball speed right after the kick [m/s].
        speed: f64,
        /// Elevation angle [deg]; 0 = flat.
        angle_deg: f64,
    },
    /// The ball entered a goal (crossed the goal line between the posts, below the crossbar).
    Goal {
        /// Team whose goal the ball entered (i.e. the conceding team).
        conceding: Team,
    },
    /// The ball crossed a field line (touch line or goal line outside the goal).
    BallLeftField,
    /// A robot lost the ball from its dribbler because the holding force was exceeded.
    DribblerSlip {
        /// Robot that lost the ball.
        robot: RobotId,
    },
    /// Two robots collided.
    RobotCollision {
        /// First robot.
        a: RobotId,
        /// Second robot.
        b: RobotId,
    },
}

/// Stable error codes reported over the wire.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SimError {
    /// A teleport specified some but not all required coordinates.
    #[error("PARTIAL_COORD: {0}")]
    PartialCoord(String),
    /// `by_force` combined with a non-zero velocity.
    #[error("VELOCITY_FORCE: {0}")]
    VelocityForce(String),
    /// `teleport_safely` requires both x and y.
    #[error("TELEPORT_SAFELY_PARTIAL: {0}")]
    TeleportSafelyPartial(String),
    /// Creating a robot requires a position.
    #[error("CREATE_NOPOS_ROBOT: {0}")]
    CreateNoPosRobot(String),
    /// Robot specs failed validation.
    #[error("INVALID_SPEC: {0}")]
    InvalidSpec(String),
    /// Feature not supported by this simulator.
    #[error("UNSUPPORTED: {0}")]
    Unsupported(String),
    /// Unknown robot.
    #[error("UNKNOWN_ROBOT: {0}")]
    UnknownRobot(RobotId),
    /// Requested step is not a whole number of substeps.
    #[error("INVALID_STEP: {0}")]
    InvalidStep(String),
}

impl SimError {
    /// The stable wire code (text before the colon).
    pub fn code(&self) -> &'static str {
        match self {
            SimError::PartialCoord(_) => "PARTIAL_COORD",
            SimError::VelocityForce(_) => "VELOCITY_FORCE",
            SimError::TeleportSafelyPartial(_) => "TELEPORT_SAFELY_PARTIAL",
            SimError::CreateNoPosRobot(_) => "CREATE_NOPOS_ROBOT",
            SimError::InvalidSpec(_) => "INVALID_SPEC",
            SimError::Unsupported(_) => "UNSUPPORTED",
            SimError::UnknownRobot(_) => "UNKNOWN_ROBOT",
            SimError::InvalidStep(_) => "INVALID_STEP",
        }
    }
}

/// Wrap an angle to (-pi, pi].
pub fn wrap_angle(a: f64) -> f64 {
    let mut a = a % std::f64::consts::TAU;
    if a > std::f64::consts::PI {
        a -= std::f64::consts::TAU;
    } else if a <= -std::f64::consts::PI {
        a += std::f64::consts::TAU;
    }
    a
}

/// Rotate a 2D vector by `angle` radians CCW.
pub fn rotate(v: Vec2, angle: f64) -> Vec2 {
    let (s, c) = angle.sin_cos();
    Vec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
}

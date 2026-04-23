use rapier3d::prelude::Vector;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Team {
    Blue,
    Yellow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RobotId {
    pub team: Team,
    pub id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MoveCommand {
    LocalVelocity {
        forward: f32,
        left: f32,
        angular: f32,
    },
    GlobalVelocity {
        x: f32,
        y: f32,
        angular: f32,
    },
    WheelVelocity {
        front_right: f32,
        back_right: f32,
        back_left: f32,
        front_left: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobotCommand {
    pub id: RobotId,
    pub movement: Option<MoveCommand>,
    pub kick_speed: Option<f32>,
    pub kick_angle_deg: f32,
    pub dribbler_speed: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TeleportBall {
    pub position: Option<Vector>,
    pub velocity: Option<Vector>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TeleportRobot {
    pub id: RobotId,
    pub present: Option<bool>,
    pub x: Option<f32>,
    pub y: Option<f32>,
    pub orientation: Option<f32>,
    pub vx: Option<f32>,
    pub vy: Option<f32>,
    pub angular: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BallState {
    pub position: Vector,
    pub velocity: Vector,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobotState {
    pub id: RobotId,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub orientation: f32,
    pub vx: f32,
    pub vy: f32,
    pub angular: f32,
    pub dribbler_ball_contact: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    pub time_seconds: f64,
    pub frame_number: u64,
    pub ball: BallState,
    pub robots: Vec<RobotState>,
}

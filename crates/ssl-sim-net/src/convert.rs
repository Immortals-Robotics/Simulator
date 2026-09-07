//! Conversions between the wire protocols and `ssl-sim-core`.
//!
//! # Units
//!
//! | Message | Wire unit | Core unit |
//! |---|---|---|
//! | `TeleportBall`, `TeleportRobot`, `RobotControl`, `RobotSpecs` | metres, m/s, rad | same |
//! | `SSL_DetectionFrame` | millimetres | metres |
//! | `SSL_GeometryData` | millimetres (ball models in m/s²) | metres |
//! | `RealismConfigErForce.vision_delay` / `vision_processing_time` | nanoseconds | seconds |
//! | tracked (`TrackerWrapperPacket`) | metres, m/s | same |
//!
//! Timestamps are seconds everywhere.
//!
//! # Conventions
//!
//! * `RobotWheelAngles` is documented **clockwise** by the protocol; the core
//!   uses CCW mounting angles, so every angle is negated on the way in and out.
//! * Wheel order is the protocol order `(front_right, back_right, back_left,
//!   front_left)` in both directions; see [`crate::legacy`] for the grSim order.

use prost::Message as _;
use prost_types::Any;

use ssl_sim_core::field::{FieldArc, FieldGeometry, FieldLine};
use ssl_sim_core::params::{BallParams, CameraConfig, Realism, RobotSpecs, WheelAngles};
use ssl_sim_core::types::{
    MoveCommand, RobotCommand, RobotId, SimError, Team, TeleportBall, TeleportRobot, Vec2, Vec3,
};
use ssl_sim_core::vision::{
    CameraCalibration, DetectedBall, DetectedRobot, DetectionFrame, GeometryData, VisionOutput,
};
use ssl_sim_core::world::WorldSnapshot;
use ssl_sim_proto::{sim, tracked};

/// Millimetres per metre.
pub const MM_PER_M: f64 = 1000.0;

/// `source_name` advertised on the ground-truth tracker stream.
pub const TRUTH_SOURCE_NAME: &str = "ssl-sim-truth";

/// Fixed UUID advertised on the ground-truth tracker stream. The protocol only
/// requires it to be stable for the lifetime of a source.
pub const TRUTH_UUID: &str = "8f1a4c2e-5b6d-4f70-9a83-1c2d3e4f5a6b";

// --------------------------------------------------------------------------
// errors
// --------------------------------------------------------------------------

/// Build a `SimulatorError` from a stable code and a message.
pub fn error(code: &str, message: impl Into<String>) -> sim::SimulatorError {
    sim::SimulatorError {
        code: Some(code.to_string()),
        message: Some(message.into()),
    }
}

/// Convert a core [`SimError`] into the wire error, keeping the stable code.
pub fn sim_error(err: &SimError) -> sim::SimulatorError {
    error(err.code(), err.to_string())
}

// --------------------------------------------------------------------------
// identity
// --------------------------------------------------------------------------

/// Core team for a protocol team enum value.
pub fn team_from_proto(team: Option<i32>) -> Option<Team> {
    match team.and_then(|t| sim::Team::try_from(t).ok()) {
        Some(sim::Team::Blue) => Some(Team::Blue),
        Some(sim::Team::Yellow) => Some(Team::Yellow),
        _ => None,
    }
}

/// Protocol team enum for a core team.
pub fn team_to_proto(team: Team) -> sim::Team {
    match team {
        Team::Blue => sim::Team::Blue,
        Team::Yellow => sim::Team::Yellow,
    }
}

/// Core robot id for a protocol `RobotId`. Both fields are required in practice.
pub fn robot_id_from_proto(id: &sim::RobotId) -> Result<RobotId, SimError> {
    let team = team_from_proto(id.team)
        .ok_or_else(|| SimError::Unsupported("RobotId without a known team".into()))?;
    let number = id
        .id
        .ok_or_else(|| SimError::Unsupported("RobotId without an id".into()))?;
    if number > u8::MAX as u32 {
        return Err(SimError::Unsupported(format!(
            "robot id {number} out of range"
        )));
    }
    Ok(RobotId::new(team, number as u8))
}

/// Protocol `RobotId` for a core robot id.
pub fn robot_id_to_proto(id: RobotId) -> sim::RobotId {
    sim::RobotId {
        id: Some(id.number as u32),
        team: Some(team_to_proto(id.team) as i32),
    }
}

// --------------------------------------------------------------------------
// robot control
// --------------------------------------------------------------------------

/// Protocol `RobotCommand` (metres, m/s, rad/s, degrees, rpm) to core.
///
/// Wheel velocities are metres per second in this message (unlike the legacy
/// grSim packet, which uses rad/s); the protocol wheel order is preserved.
pub fn robot_command_from_proto(cmd: &sim::RobotCommand) -> RobotCommand {
    use sim::robot_move_command::Command;

    let movement = cmd
        .move_command
        .as_ref()
        .and_then(|m| m.command.as_ref())
        .map(|c| match c {
            Command::LocalVelocity(v) => MoveCommand::LocalVelocity {
                forward: v.forward as f64,
                left: v.left as f64,
                angular: v.angular as f64,
            },
            Command::GlobalVelocity(v) => MoveCommand::GlobalVelocity {
                x: v.x as f64,
                y: v.y as f64,
                angular: v.angular as f64,
            },
            Command::WheelVelocity(v) => MoveCommand::WheelVelocity {
                front_right: v.front_right as f64,
                back_right: v.back_right as f64,
                back_left: v.back_left as f64,
                front_left: v.front_left as f64,
            },
        });

    RobotCommand {
        movement,
        kick_speed: cmd.kick_speed.map(|s| s as f64).filter(|s| *s > 0.0),
        kick_angle_deg: cmd.kick_angle.unwrap_or(0.0) as f64,
        dribbler_rpm: cmd.dribbler_speed.map(|s| s as f64).filter(|s| *s > 0.0),
    }
}

/// Core command back to the protocol message (used by tooling and tests).
pub fn robot_command_to_proto(number: u8, cmd: &RobotCommand) -> sim::RobotCommand {
    use sim::robot_move_command::Command;

    let command = cmd.movement.map(|m| match m {
        MoveCommand::LocalVelocity {
            forward,
            left,
            angular,
        } => Command::LocalVelocity(sim::MoveLocalVelocity {
            forward: forward as f32,
            left: left as f32,
            angular: angular as f32,
        }),
        MoveCommand::GlobalVelocity { x, y, angular } => {
            Command::GlobalVelocity(sim::MoveGlobalVelocity {
                x: x as f32,
                y: y as f32,
                angular: angular as f32,
            })
        }
        MoveCommand::WheelVelocity {
            front_right,
            back_right,
            back_left,
            front_left,
        } => Command::WheelVelocity(sim::MoveWheelVelocity {
            front_right: front_right as f32,
            back_right: back_right as f32,
            back_left: back_left as f32,
            front_left: front_left as f32,
        }),
    });

    sim::RobotCommand {
        id: number as u32,
        move_command: command.map(|c| sim::RobotMoveCommand { command: Some(c) }),
        kick_speed: cmd.kick_speed.map(|s| s as f32),
        kick_angle: Some(cmd.kick_angle_deg as f32),
        dribbler_speed: cmd.dribbler_rpm.map(|s| s as f32),
    }
}

// --------------------------------------------------------------------------
// teleports
// --------------------------------------------------------------------------

/// Protocol `TeleportBall` (metres) to core.
///
/// Partial-coordinate rules: `x` without `y` (or the reverse) is
/// `PARTIAL_COORD`; `z` is only accepted together with `x` and `y`. The same
/// rule applies to the velocity triple.
pub fn teleport_ball_from_proto(t: &sim::TeleportBall) -> Result<TeleportBall, SimError> {
    let position = match (t.x, t.y) {
        (Some(x), Some(y)) => Some(Vec3::new(x as f64, y as f64, t.z.unwrap_or(0.0) as f64)),
        (None, None) => {
            if t.z.is_some() {
                return Err(SimError::PartialCoord(
                    "TeleportBall z without x and y".into(),
                ));
            }
            None
        }
        _ => {
            return Err(SimError::PartialCoord(
                "TeleportBall needs both x and y".into(),
            ))
        }
    };
    let velocity = match (t.vx, t.vy) {
        (Some(vx), Some(vy)) => Some(Vec3::new(vx as f64, vy as f64, t.vz.unwrap_or(0.0) as f64)),
        (None, None) => {
            if t.vz.is_some() {
                return Err(SimError::PartialCoord(
                    "TeleportBall vz without vx and vy".into(),
                ));
            }
            None
        }
        _ => {
            return Err(SimError::PartialCoord(
                "TeleportBall needs both vx and vy".into(),
            ))
        }
    };
    let teleport_safely = t.teleport_safely.unwrap_or(false);
    if teleport_safely && position.is_none() {
        return Err(SimError::TeleportSafelyPartial(
            "teleport_safely needs x and y".into(),
        ));
    }
    Ok(TeleportBall {
        position,
        velocity,
        teleport_safely,
        roll: t.roll.unwrap_or(false),
        by_force: t.by_force.unwrap_or(false),
    })
}

/// Protocol `TeleportRobot` (metres) to core.
///
/// Unlike ER-Force we do **not** require `orientation` alongside a position:
/// the Immortals clients teleport with `x`/`y` only and expect it to work.
pub fn teleport_robot_from_proto(t: &sim::TeleportRobot) -> Result<TeleportRobot, SimError> {
    let id = robot_id_from_proto(&t.id)?;
    let position = match (t.x, t.y) {
        (Some(x), Some(y)) => Some(Vec2::new(x as f64, y as f64)),
        (None, None) => None,
        _ => {
            return Err(SimError::PartialCoord(format!(
                "TeleportRobot {id} needs both x and y"
            )))
        }
    };
    let velocity = match (t.v_x, t.v_y) {
        (Some(vx), Some(vy)) => Some(Vec2::new(vx as f64, vy as f64)),
        (None, None) => None,
        _ => {
            return Err(SimError::PartialCoord(format!(
                "TeleportRobot {id} needs both v_x and v_y"
            )))
        }
    };
    Ok(TeleportRobot {
        id,
        position,
        orientation: t.orientation.map(|o| o as f64),
        velocity,
        angular_velocity: t.v_angular.map(|w| w as f64),
        present: t.present,
        by_force: t.by_force.unwrap_or(false),
    })
}

// --------------------------------------------------------------------------
// robot specs
// --------------------------------------------------------------------------

/// True when `any` carries a message whose (possibly package qualified) name
/// ends in `name`.
fn any_is(any: &Any, name: &str) -> bool {
    let last = any.type_url.rsplit('/').next().unwrap_or("");
    last == name || last.rsplit('.').next() == Some(name)
}

/// Pack a message into an `Any` with the type URL layout the protocol uses.
pub fn pack_any<M: prost::Message>(name: &str, msg: &M) -> Any {
    Any {
        type_url: format!("type.googleapis.com/{name}"),
        value: msg.encode_to_vec(),
    }
}

/// Merge a protocol `RobotSpecs` into the current core specs. Only fields that
/// are actually set on the wire override the current value.
///
/// `custom` entries are searched for `RobotSpecErForce` (`shoot_radius`,
/// `dribbler_width`); unknown `Any` types are ignored.
pub fn merge_robot_specs(mut specs: RobotSpecs, proto: &sim::RobotSpecs) -> RobotSpecs {
    if let Some(v) = proto.radius {
        specs.radius = v as f64;
    }
    if let Some(v) = proto.height {
        specs.height = v as f64;
    }
    if let Some(v) = proto.mass {
        specs.mass = v as f64;
    }
    if let Some(v) = proto.center_to_dribbler {
        specs.center_to_dribbler = v as f64;
    }
    if let Some(v) = proto.max_linear_kick_speed {
        specs.kicker.max_linear_speed = v as f64;
    }
    if let Some(v) = proto.max_chip_kick_speed {
        specs.kicker.max_chip_speed = v as f64;
    }
    if let Some(l) = &proto.limits {
        if let Some(v) = l.acc_speedup_absolute_max {
            specs.limits.acc_speedup_absolute_max = v as f64;
        }
        if let Some(v) = l.acc_speedup_angular_max {
            specs.limits.acc_speedup_angular_max = v as f64;
        }
        if let Some(v) = l.acc_brake_absolute_max {
            specs.limits.acc_brake_absolute_max = v as f64;
        }
        if let Some(v) = l.acc_brake_angular_max {
            specs.limits.acc_brake_angular_max = v as f64;
        }
        if let Some(v) = l.vel_absolute_max {
            specs.limits.vel_absolute_max = v as f64;
        }
        if let Some(v) = l.vel_angular_max {
            specs.limits.vel_angular_max = v as f64;
        }
    }
    if let Some(w) = &proto.wheel_angles {
        specs.wheel_angles = wheel_angles_from_proto(w);
    }
    for any in &proto.custom {
        if any_is(any, "RobotSpecErForce") {
            if let Ok(custom) = sim::RobotSpecErForce::decode(any.value.as_slice()) {
                if let Some(v) = custom.shoot_radius {
                    specs.shoot_radius = v as f64;
                }
                if let Some(v) = custom.dribbler_width {
                    specs.dribbler_width = v as f64;
                }
            }
        }
    }
    specs
}

/// Protocol wheel angles (clockwise) to core wheel angles (CCW): every angle is
/// negated, the order is unchanged.
pub fn wheel_angles_from_proto(w: &sim::RobotWheelAngles) -> WheelAngles {
    WheelAngles {
        front_right: -(w.front_right as f64),
        back_right: -(w.back_right as f64),
        back_left: -(w.back_left as f64),
        front_left: -(w.front_left as f64),
    }
}

/// Core wheel angles (CCW) back to the protocol's clockwise convention.
pub fn wheel_angles_to_proto(w: &WheelAngles) -> sim::RobotWheelAngles {
    sim::RobotWheelAngles {
        front_right: -w.front_right as f32,
        back_right: -w.back_right as f32,
        back_left: -w.back_left as f32,
        front_left: -w.front_left as f32,
    }
}

/// Core specs to the protocol message (used by tooling and round-trip tests).
pub fn robot_specs_to_proto(id: RobotId, specs: &RobotSpecs) -> sim::RobotSpecs {
    sim::RobotSpecs {
        id: robot_id_to_proto(id),
        radius: Some(specs.radius as f32),
        height: Some(specs.height as f32),
        mass: Some(specs.mass as f32),
        max_linear_kick_speed: Some(specs.kicker.max_linear_speed as f32),
        max_chip_kick_speed: Some(specs.kicker.max_chip_speed as f32),
        center_to_dribbler: Some(specs.center_to_dribbler as f32),
        limits: Some(sim::RobotLimits {
            acc_speedup_absolute_max: Some(specs.limits.acc_speedup_absolute_max as f32),
            acc_speedup_angular_max: Some(specs.limits.acc_speedup_angular_max as f32),
            acc_brake_absolute_max: Some(specs.limits.acc_brake_absolute_max as f32),
            acc_brake_angular_max: Some(specs.limits.acc_brake_angular_max as f32),
            vel_absolute_max: Some(specs.limits.vel_absolute_max as f32),
            vel_angular_max: Some(specs.limits.vel_angular_max as f32),
        }),
        wheel_angles: Some(wheel_angles_to_proto(&specs.wheel_angles)),
        custom: vec![pack_any(
            "RobotSpecErForce",
            &sim::RobotSpecErForce {
                shoot_radius: Some(specs.shoot_radius as f32),
                dribbler_width: Some(specs.dribbler_width as f32),
            },
        )],
    }
}

// --------------------------------------------------------------------------
// realism
// --------------------------------------------------------------------------

/// Nanoseconds to seconds, clamped to non-negative (ER-Force clamps too).
fn ns_to_s(ns: i64) -> f64 {
    (ns as f64 * 1e-9).max(0.0)
}

/// Merge every `RealismConfigErForce` packed into `cfg.custom` into `realism`.
/// Unknown `Any` types are ignored. Returns the merged config and the list of
/// type URLs that were ignored (the caller turns those into warnings).
pub fn merge_realism(mut realism: Realism, cfg: &sim::RealismConfig) -> (Realism, Vec<String>) {
    let mut ignored = Vec::new();
    for any in &cfg.custom {
        if !any_is(any, "RealismConfigErForce") {
            ignored.push(any.type_url.clone());
            continue;
        }
        match sim::RealismConfigErForce::decode(any.value.as_slice()) {
            Ok(c) => realism = merge_realism_erforce(realism, &c),
            Err(_) => ignored.push(any.type_url.clone()),
        }
    }
    (realism, ignored)
}

/// Merge a decoded `RealismConfigErForce` into a core [`Realism`].
pub fn merge_realism_erforce(mut r: Realism, c: &sim::RealismConfigErForce) -> Realism {
    if let Some(v) = c.stddev_ball_p {
        r.stddev_ball_p = v as f64;
    }
    if let Some(v) = c.stddev_robot_p {
        r.stddev_robot_p = v as f64;
    }
    if let Some(v) = c.stddev_robot_phi {
        r.stddev_robot_phi = v as f64;
    }
    if let Some(v) = c.stddev_ball_area {
        r.stddev_ball_area = v as f64;
    }
    if let Some(v) = c.enable_invisible_ball {
        r.enable_invisible_ball = v;
    }
    if let Some(v) = c.ball_visibility_threshold {
        r.ball_visibility_threshold = v as f64;
    }
    if let Some(v) = c.camera_overlap {
        r.camera_overlap = v as f64;
    }
    if let Some(v) = c.dribbler_ball_detections {
        r.dribbler_ball_detections = v as f64;
    }
    if let Some(v) = c.camera_position_error {
        r.camera_position_error = v as f64;
    }
    if let Some(v) = c.robot_command_loss {
        r.robot_command_loss = v as f64;
    }
    if let Some(v) = c.robot_response_loss {
        r.robot_response_loss = v as f64;
    }
    if let Some(v) = c.missing_ball_detections {
        r.missing_ball_detections = v as f64;
    }
    if let Some(v) = c.vision_delay {
        r.vision_delay = ns_to_s(v);
    }
    if let Some(v) = c.vision_processing_time {
        r.vision_processing_time = ns_to_s(v);
    }
    if let Some(v) = c.simulate_dribbling {
        r.simulate_dribbling = v;
    }
    if let Some(v) = c.object_position_offset {
        r.object_position_offset = v as f64;
    }
    if let Some(v) = c.missing_robot_detections {
        r.missing_robot_detections = v as f64;
    }
    r
}

/// Core realism back to the ER-Force custom message.
///
/// `command_delay` is part of the core config but not of the ER-Force message,
/// so it is not round-tripped.
pub fn realism_to_erforce(r: &Realism) -> sim::RealismConfigErForce {
    sim::RealismConfigErForce {
        stddev_ball_p: Some(r.stddev_ball_p as f32),
        stddev_robot_p: Some(r.stddev_robot_p as f32),
        stddev_robot_phi: Some(r.stddev_robot_phi as f32),
        stddev_ball_area: Some(r.stddev_ball_area as f32),
        enable_invisible_ball: Some(r.enable_invisible_ball),
        ball_visibility_threshold: Some(r.ball_visibility_threshold as f32),
        camera_overlap: Some(r.camera_overlap as f32),
        dribbler_ball_detections: Some(r.dribbler_ball_detections as f32),
        camera_position_error: Some(r.camera_position_error as f32),
        robot_command_loss: Some(r.robot_command_loss as f32),
        robot_response_loss: Some(r.robot_response_loss as f32),
        missing_ball_detections: Some(r.missing_ball_detections as f32),
        vision_delay: Some((r.vision_delay * 1e9).round() as i64),
        vision_processing_time: Some((r.vision_processing_time * 1e9).round() as i64),
        simulate_dribbling: Some(r.simulate_dribbling),
        object_position_offset: Some(r.object_position_offset as f32),
        missing_robot_detections: Some(r.missing_robot_detections as f32),
    }
}

// --------------------------------------------------------------------------
// field geometry (wire -> core)
// --------------------------------------------------------------------------

fn line_named<'a>(
    g: &'a sim::SslGeometryFieldSize,
    name: &str,
) -> Option<&'a sim::SslFieldLineSegment> {
    g.field_lines.iter().find(|l| l.name == name)
}

/// Merge an `SSL_GeometryData` field description (millimetres) into the current
/// core [`FieldGeometry`]. Fields that are not on the wire keep their value.
///
/// The penalty area is taken from `penalty_area_depth` / `penalty_area_width`
/// when present, else derived from the `LeftPenaltyStretch` /
/// `LeftFieldLeftPenaltyStretch` line segments, else left unchanged.
pub fn field_from_geometry(mut field: FieldGeometry, geo: &sim::SslGeometryData) -> FieldGeometry {
    let g = &geo.field;
    field.length = g.field_length as f64 / MM_PER_M;
    field.width = g.field_width as f64 / MM_PER_M;
    field.goal_width = g.goal_width as f64 / MM_PER_M;
    field.goal_depth = g.goal_depth as f64 / MM_PER_M;
    field.boundary_width = g.boundary_width as f64 / MM_PER_M;

    match (g.penalty_area_depth, g.penalty_area_width) {
        (Some(d), Some(w)) => {
            field.penalty_area_depth = d as f64 / MM_PER_M;
            field.penalty_area_width = w as f64 / MM_PER_M;
        }
        _ => {
            // The "penalty stretch" is the line parallel to the goal line that
            // closes the penalty area; its distance from the goal line is the
            // depth and its length is the width.
            if let Some(l) = line_named(g, "LeftPenaltyStretch") {
                let depth = field.half_length() - (l.p1.x as f64 / MM_PER_M).abs();
                let width = ((l.p2.y - l.p1.y) as f64 / MM_PER_M).abs();
                if depth > 0.0 {
                    field.penalty_area_depth = depth;
                }
                if width > 0.0 {
                    field.penalty_area_width = width;
                }
            } else if let Some(l) = line_named(g, "LeftFieldLeftPenaltyStretch") {
                // Perpendicular stretch: runs from the goal line inward.
                let depth = ((l.p2.x - l.p1.x) as f64 / MM_PER_M).abs();
                let width = 2.0 * (l.p1.y as f64 / MM_PER_M).abs();
                if depth > 0.0 {
                    field.penalty_area_depth = depth;
                }
                if width > 0.0 {
                    field.penalty_area_width = width;
                }
            }
        }
    }

    if let Some(t) = g
        .field_lines
        .iter()
        .map(|l| l.thickness as f64)
        .filter(|t| *t > 0.0)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    {
        field.line_thickness = t / MM_PER_M;
    }
    if let Some(a) = geo
        .field
        .field_arcs
        .iter()
        .find(|a| a.name == "CenterCircle")
    {
        field.center_circle_radius = a.radius as f64 / MM_PER_M;
    }
    field
}

// --------------------------------------------------------------------------
// vision output (core -> wire)
// --------------------------------------------------------------------------

fn detection_ball_to_proto(b: &DetectedBall) -> sim::SslDetectionBall {
    sim::SslDetectionBall {
        confidence: b.confidence as f32,
        // `area` is optional on the wire and a whole tournament in the 2026
        // corpus reported none at all (vision.md §3, §9.9); `report_area = false`
        // reproduces that.
        area: b.area.map(|a| a.max(0.0).round() as u32),
        x: (b.pos.x * MM_PER_M) as f32,
        y: (b.pos.y * MM_PER_M) as f32,
        z: b.z.map(|z| (z * MM_PER_M) as f32),
        pixel_x: 0.0,
        pixel_y: 0.0,
    }
}

fn detection_robot_to_proto(r: &DetectedRobot) -> sim::SslDetectionRobot {
    sim::SslDetectionRobot {
        confidence: r.confidence as f32,
        robot_id: Some(r.number as u32),
        x: (r.pos.x * MM_PER_M) as f32,
        y: (r.pos.y * MM_PER_M) as f32,
        orientation: Some(r.orientation as f32),
        pixel_x: 0.0,
        pixel_y: 0.0,
        height: Some((r.height * MM_PER_M) as f32),
    }
}

/// Core detection frame (metres) to the wire frame (millimetres).
pub fn detection_frame_to_proto(f: &DetectionFrame) -> sim::SslDetectionFrame {
    sim::SslDetectionFrame {
        frame_number: f.frame_number,
        t_capture: f.t_capture,
        t_sent: f.t_sent,
        camera_id: f.camera_id,
        balls: f.balls.iter().map(detection_ball_to_proto).collect(),
        robots_yellow: f
            .robots_yellow
            .iter()
            .map(detection_robot_to_proto)
            .collect(),
        robots_blue: f.robots_blue.iter().map(detection_robot_to_proto).collect(),
    }
}

fn vec2f(p: Vec2) -> sim::Vector2f {
    sim::Vector2f {
        x: (p.x * MM_PER_M) as f32,
        y: (p.y * MM_PER_M) as f32,
    }
}

fn line_to_proto(l: &FieldLine) -> sim::SslFieldLineSegment {
    sim::SslFieldLineSegment {
        name: l.name.clone(),
        p1: vec2f(l.p1),
        p2: vec2f(l.p2),
        thickness: (l.thickness * MM_PER_M) as f32,
        r#type: sim::SslFieldShapeType::from_str_name(&l.name).map(|t| t as i32),
    }
}

fn arc_to_proto(a: &FieldArc) -> sim::SslFieldCircularArc {
    sim::SslFieldCircularArc {
        name: a.name.clone(),
        center: vec2f(a.center),
        radius: (a.radius * MM_PER_M) as f32,
        a1: a.a1 as f32,
        a2: a.a2 as f32,
        thickness: (a.thickness * MM_PER_M) as f32,
        r#type: sim::SslFieldShapeType::from_str_name(&a.name).map(|t| t as i32),
    }
}

/// Build the wire field description from the core geometry plus an explicit
/// line and arc set.
///
/// The line/arc set is passed in rather than read from `field` so that callers
/// (and tests) can supply them directly; [`geometry_to_proto`] uses
/// [`FieldGeometry::lines`] and [`FieldGeometry::arcs`].
pub fn field_size_to_proto(
    field: &FieldGeometry,
    lines: &[FieldLine],
    arcs: &[FieldArc],
) -> sim::SslGeometryFieldSize {
    sim::SslGeometryFieldSize {
        field_length: (field.length * MM_PER_M).round() as i32,
        field_width: (field.width * MM_PER_M).round() as i32,
        goal_width: (field.goal_width * MM_PER_M).round() as i32,
        goal_depth: (field.goal_depth * MM_PER_M).round() as i32,
        boundary_width: (field.boundary_width * MM_PER_M).round() as i32,
        field_lines: lines.iter().map(line_to_proto).collect(),
        field_arcs: arcs.iter().map(arc_to_proto).collect(),
        penalty_area_depth: Some((field.penalty_area_depth * MM_PER_M).round() as i32),
        penalty_area_width: Some((field.penalty_area_width * MM_PER_M).round() as i32),
    }
}

/// Camera rig from a geometry packet's calibrations: one camera per entry,
/// positioned at `derived_camera_world_t{x,y,z}` (mm -> m). Entries without a
/// derived position are skipped.
pub fn cameras_from_geometry(geo: &sim::SslGeometryData) -> Vec<CameraConfig> {
    geo.calib
        .iter()
        .filter_map(|c| {
            let (x, y, z) = (
                c.derived_camera_world_tx?,
                c.derived_camera_world_ty?,
                c.derived_camera_world_tz?,
            );
            Some(CameraConfig {
                id: c.camera_id,
                position: Vec3::new(
                    x as f64 / MM_PER_M,
                    y as f64 / MM_PER_M,
                    z as f64 / MM_PER_M,
                ),
            })
        })
        .collect()
}

/// Camera calibration for the geometry packet.
///
/// The extrinsics are self consistent for a camera looking straight down: the
/// world-to-camera rotation is a 180° turn about x (quaternion `(w,x,y,z) =
/// (0,1,0,0)`), so `t = -R * C` and `derived_camera_world_t = C`. The core
/// stores the quaternion as `(w, x, y, z)`; ssl-vision's `q0..q3` are
/// `(x, y, z, w)`.
pub fn camera_calibration_to_proto(c: &CameraCalibration) -> sim::SslGeometryCameraCalibration {
    let cx = c.position.x * MM_PER_M;
    let cy = c.position.y * MM_PER_M;
    let cz = c.position.z * MM_PER_M;
    let [qw, qx, qy, qz] = c.q;
    sim::SslGeometryCameraCalibration {
        camera_id: c.camera_id,
        focal_length: c.focal_length as f32,
        principal_point_x: c.principal_point.x as f32,
        principal_point_y: c.principal_point.y as f32,
        distortion: c.distortion as f32,
        q0: qx as f32,
        q1: qy as f32,
        q2: qz as f32,
        q3: qw as f32,
        // R = diag(1, -1, -1)  =>  t = -R * C = (-cx, cy, cz)
        tx: -cx as f32,
        ty: cy as f32,
        tz: cz as f32,
        derived_camera_world_tx: Some(cx as f32),
        derived_camera_world_ty: Some(cy as f32),
        derived_camera_world_tz: Some(cz as f32),
        pixel_image_width: None,
        pixel_image_height: None,
    }
}

/// The advertised ball models. These are the very constants the core
/// simulates, with one deliberate translation: the wire model is a *constant*
/// rolling deceleration, while the core rolls with a speed-dependent one, so
/// the packet carries [`BallParams::advertised_acc_roll`] — the value at
/// `advertise_roll_speed` — instead of the zero-speed `acc_roll`
/// (`docs/calibration/dynamics.md`).
///
/// `SSL_BallModelChipFixedLoss` likewise only has room for the **first hop**
/// damping factors; the later-hop constants the core uses are not advertised.
pub fn ball_models_to_proto(ball: &BallParams) -> sim::SslGeometryModels {
    sim::SslGeometryModels {
        straight_two_phase: Some(sim::SslBallModelStraightTwoPhase {
            acc_slide: ball.acc_slide,
            acc_roll: ball.advertised_acc_roll(),
            k_switch: ball.k_switch(),
        }),
        chip_fixed_loss: Some(sim::SslBallModelChipFixedLoss {
            damping_xy_first_hop: ball.chip_damping_xy_first_hop,
            damping_xy_other_hops: ball.chip_damping_xy_other_hops,
            damping_z: ball.chip_damping_z,
        }),
    }
}

/// Core geometry data to the wire geometry packet.
pub fn geometry_to_proto(geo: &GeometryData) -> sim::SslGeometryData {
    let lines = geo.field.lines();
    let arcs = geo.field.arcs();
    sim::SslGeometryData {
        field: field_size_to_proto(&geo.field, &lines, &arcs),
        calib: geo
            .cameras
            .iter()
            .map(camera_calibration_to_proto)
            .collect(),
        models: Some(ball_models_to_proto(&geo.ball)),
    }
}

/// One wrapper packet per camera capture.
///
/// Cameras capture on independent schedules (see
/// [`ssl_sim_core::vision`]), so a [`VisionOutput`] is one camera's frame; the
/// geometry, when the core attached one, rides on it (only camera 0's outputs
/// ever carry geometry).
pub fn vision_output_to_packet(out: &VisionOutput) -> sim::SslWrapperPacket {
    sim::SslWrapperPacket {
        detection: Some(detection_frame_to_proto(&out.frame)),
        geometry: out.geometry.as_ref().map(geometry_to_proto),
        source: Some(sim::SslSource::Other as i32),
    }
}

// --------------------------------------------------------------------------
// ground truth (core -> tracked)
// --------------------------------------------------------------------------

/// Core world snapshot to a `TrackerWrapperPacket`. The tracked protocol is
/// metres and m/s throughout, so no scaling happens here.
pub fn snapshot_to_tracker(
    snap: &WorldSnapshot,
    frame_number: u32,
) -> tracked::TrackerWrapperPacket {
    let ball = tracked::TrackedBall {
        pos: tracked::Vector3 {
            x: snap.ball.pos.x as f32,
            y: snap.ball.pos.y as f32,
            z: snap.ball.pos.z as f32,
        },
        vel: Some(tracked::Vector3 {
            x: snap.ball.vel.x as f32,
            y: snap.ball.vel.y as f32,
            z: snap.ball.vel.z as f32,
        }),
        visibility: Some(1.0),
    };
    let robots = snap
        .robots
        .iter()
        .map(|r| tracked::TrackedRobot {
            robot_id: tracked::RobotId {
                id: r.id.number as u32,
                team_color: match r.id.team {
                    Team::Blue => tracked::TeamColor::Blue as i32,
                    Team::Yellow => tracked::TeamColor::Yellow as i32,
                },
            },
            pos: tracked::Vector2 {
                x: r.pos.x as f32,
                y: r.pos.y as f32,
            },
            orientation: r.orientation as f32,
            vel: Some(tracked::Vector2 {
                x: r.vel.x as f32,
                y: r.vel.y as f32,
            }),
            vel_angular: Some(r.angular_velocity as f32),
            visibility: Some(1.0),
        })
        .collect();

    tracked::TrackerWrapperPacket {
        uuid: TRUTH_UUID.to_string(),
        source_name: Some(TRUTH_SOURCE_NAME.to_string()),
        tracked_frame: Some(tracked::TrackedFrame {
            frame_number,
            timestamp: snap.time.as_secs_f64(),
            balls: vec![ball],
            robots,
            kicked_ball: None,
            capabilities: vec![tracked::Capability::DetectFlyingBalls as i32],
        }),
    }
}

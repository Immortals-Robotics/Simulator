//! Legacy grSim packet adapter: `grSim_Packet` in on UDP 20011,
//! `Robots_Status` out to the sender's address on 30011 (blue) / 30012
//! (yellow).
//!
//! # Units
//!
//! grSim's legacy messages are **metres** for positions and velocities,
//! **degrees** for `grSim_RobotReplacement::dir`, and **rad/s** for
//! `wheel1..4` (the SSL simulation protocol uses m/s instead).
//!
//! # Wheel order
//!
//! grSim builds its wheels in the order of the mounting angles
//! `60°, 135°, 225°, 300°` measured CCW from the robot's forward direction
//! (`robot.cpp`), so
//!
//! | grSim field | angle | wheel |
//! |---|---|---|
//! | `wheel1` | 60° | front left |
//! | `wheel2` | 135° | back left |
//! | `wheel3` | 225° | back right |
//! | `wheel4` | 300° | front right |
//!
//! The SSL simulation protocol orders wheels `(front_right, back_right,
//! back_left, front_left)`, i.e. exactly the reverse, so
//! `front_right = wheel4`, `back_right = wheel3`, `back_left = wheel2`,
//! `front_left = wheel1`.

use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr};

use ssl_sim_core::params::RobotSpecs;
use ssl_sim_core::types::{
    Event, MoveCommand, RobotCommand, RobotId, Team, TeleportBall, TeleportRobot, Vec2, Vec3,
};
use ssl_sim_proto::grsim;

/// Default UDP port the legacy `grSim_Packet` listener binds.
pub const LEGACY_COMMAND_PORT: u16 = 20011;
/// Default `Robots_Status` port for the blue team.
pub const LEGACY_STATUS_PORT_BLUE: u16 = 30011;
/// Default `Robots_Status` port for the yellow team.
pub const LEGACY_STATUS_PORT_YELLOW: u16 = 30012;

/// Fallback dribbler speed [rpm] when the robot's specs are unknown.
pub const DEFAULT_DRIBBLER_RPM: f64 = 10_000.0;

/// A kick counts as "recent" for `Robots_Status` for this many vision frames,
/// matching grSim's `kickstate = 10` frame counter.
pub const KICK_STATUS_FRAMES: u64 = 10;

/// Team addressed by a legacy commands block.
pub fn team_of(cmds: &grsim::GrSimCommands) -> Team {
    if cmds.isteamyellow {
        Team::Yellow
    } else {
        Team::Blue
    }
}

/// Status port for a team.
pub fn status_port(team: Team) -> u16 {
    match team {
        Team::Blue => LEGACY_STATUS_PORT_BLUE,
        Team::Yellow => LEGACY_STATUS_PORT_YELLOW,
    }
}

/// Address to send `Robots_Status` to: the datagram sender's IP with the
/// team's status port. Matches grSim, which unicasts the status back to
/// whoever sent the commands.
pub fn status_address(from: SocketAddr, team: Team) -> SocketAddr {
    SocketAddr::new(from.ip(), status_port(team))
}

/// Convenience: the loopback status address for a team (used by tools).
pub fn loopback_status_address(team: Team) -> SocketAddr {
    SocketAddr::new(IpAddr::from([127, 0, 0, 1]), status_port(team))
}

/// Convert one legacy robot command. `specs` is the target robot's spec, used
/// for the wheel radius (rad/s → m/s) and the dribbler's full-speed rpm.
pub fn robot_command_from_grsim(
    cmd: &grsim::GrSimRobotCommand,
    specs: Option<&RobotSpecs>,
) -> RobotCommand {
    let wheel_radius = specs.map_or(RobotSpecs::default().drive.wheel_radius, |s| {
        s.drive.wheel_radius
    });
    let max_rpm = specs.map_or(DEFAULT_DRIBBLER_RPM, |s| s.dribbler.max_speed_rpm);

    let movement = if cmd.wheelsspeed {
        // grSim order 1..4 = front_left, back_left, back_right, front_right.
        let w = |v: Option<f32>| v.unwrap_or(0.0) as f64 * wheel_radius;
        Some(MoveCommand::WheelVelocity {
            front_right: w(cmd.wheel4),
            back_right: w(cmd.wheel3),
            back_left: w(cmd.wheel2),
            front_left: w(cmd.wheel1),
        })
    } else {
        Some(MoveCommand::LocalVelocity {
            forward: cmd.veltangent as f64,
            left: cmd.velnormal as f64,
            angular: cmd.velangular as f64,
        })
    };

    let kx = cmd.kickspeedx as f64;
    let kz = cmd.kickspeedz as f64;
    let speed = kx.hypot(kz);
    let (kick_speed, kick_angle_deg) = if speed > 1e-6 {
        (Some(speed), kz.atan2(kx).to_degrees())
    } else {
        (None, 0.0)
    };

    RobotCommand {
        movement,
        kick_speed,
        kick_angle_deg,
        dribbler_rpm: if cmd.spinner { Some(max_rpm) } else { None },
    }
}

/// Convert a legacy ball replacement (metres) to a core teleport.
pub fn teleport_ball_from_grsim(b: &grsim::GrSimBallReplacement) -> TeleportBall {
    let position = match (b.x, b.y) {
        (Some(x), Some(y)) => Some(Vec3::new(x, y, 0.0)),
        _ => None,
    };
    let velocity = match (b.vx, b.vy) {
        (None, None) => None,
        (vx, vy) => Some(Vec3::new(vx.unwrap_or(0.0), vy.unwrap_or(0.0), 0.0)),
    };
    TeleportBall {
        position,
        velocity,
        teleport_safely: false,
        roll: false,
        by_force: false,
    }
}

/// Convert a legacy robot replacement (metres, degrees) to a core teleport.
/// `turnon` maps to `present`.
pub fn teleport_robot_from_grsim(r: &grsim::GrSimRobotReplacement) -> TeleportRobot {
    let team = if r.yellowteam {
        Team::Yellow
    } else {
        Team::Blue
    };
    TeleportRobot {
        id: RobotId::new(team, r.id.min(u8::MAX as u32) as u8),
        position: Some(Vec2::new(r.x, r.y)),
        orientation: Some(r.dir.to_radians()),
        velocity: Some(Vec2::ZERO),
        angular_velocity: Some(0.0),
        present: r.turnon.or(Some(true)),
        by_force: false,
    }
}

/// One robot's legacy status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LegacyStatus {
    /// Break beam: a ball is seated at the dribbler.
    pub infrared: bool,
    /// A straight kick happened within the last [`KICK_STATUS_FRAMES`] frames.
    pub flat_kick: bool,
    /// A chip kick happened within the last [`KICK_STATUS_FRAMES`] frames.
    pub chip_kick: bool,
}

/// Tracks per-robot legacy status and reports changes.
///
/// Kick flags stay set for [`KICK_STATUS_FRAMES`] vision frames after the kick
/// (grSim keeps a 10-frame counter for the same purpose).
#[derive(Debug, Clone, Default)]
pub struct StatusTracker {
    /// Vision frames seen so far; kick ages are measured against this.
    frame: u64,
    kicks: BTreeMap<RobotId, (u64, bool)>,
    last_sent: BTreeMap<RobotId, LegacyStatus>,
}

impl StatusTracker {
    /// New empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record simulation events; `Kick` events set the per-robot kick flag.
    pub fn record_events(&mut self, events: &[Event]) {
        for e in events {
            if let Event::Kick {
                robot, angle_deg, ..
            } = e
            {
                self.kicks.insert(*robot, (self.frame, *angle_deg > 1e-3));
            }
        }
    }

    /// Advance the vision-frame counter (call once per published vision frame).
    pub fn advance_frame(&mut self) {
        self.frame += 1;
        let now = self.frame;
        self.kicks
            .retain(|_, (f, _)| now.saturating_sub(*f) < KICK_STATUS_FRAMES);
    }

    /// Current status of one robot given its break-beam state.
    pub fn status_of(&self, id: RobotId, ball_contact: bool) -> LegacyStatus {
        let (flat, chip) = match self.kicks.get(&id) {
            Some((_, true)) => (false, true),
            Some((_, false)) => (true, false),
            None => (false, false),
        };
        LegacyStatus {
            infrared: ball_contact,
            flat_kick: flat,
            chip_kick: chip,
        }
    }

    /// Build a `Robots_Status` message for `team` containing only the robots
    /// whose status changed since the last call, and remember what was sent.
    /// Returns `None` when nothing changed.
    pub fn changed_status(
        &mut self,
        team: Team,
        robots: impl IntoIterator<Item = (RobotId, bool)>,
    ) -> Option<grsim::RobotsStatus> {
        let mut out = Vec::new();
        for (id, ball_contact) in robots {
            if id.team != team {
                continue;
            }
            let status = self.status_of(id, ball_contact);
            if self.last_sent.get(&id) == Some(&status) {
                continue;
            }
            self.last_sent.insert(id, status);
            out.push(grsim::RobotStatus {
                robot_id: id.number as i32,
                infrared: status.infrared,
                flat_kick: status.flat_kick,
                chip_kick: status.chip_kick,
            });
        }
        if out.is_empty() {
            None
        } else {
            Some(grsim::RobotsStatus { robots_status: out })
        }
    }
}

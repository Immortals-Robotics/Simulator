//! Applying a team's `RobotControl` and building the `RobotControlResponse`.
//!
//! Packet loss is drawn from the world's `packet_loss` RNG stream so that a run
//! stays deterministic for a given seed:
//!
//! * `realism.robot_command_loss` drops the **whole datagram** (that is what a
//!   lost UDP packet does); nothing is applied and no response is produced.
//! * `realism.robot_response_loss` drops the response after the commands were
//!   applied.
//!
//! `realism.command_delay` is **not** implemented in this round: commands take
//! effect at the next substep. See the crate README/report for the deferral.

use ssl_sim_core::rng::chance;
use ssl_sim_core::types::{RobotId, Team};
use ssl_sim_core::World;
use ssl_sim_proto::sim;

use crate::convert;

/// What happened to one `RobotControl` datagram.
#[derive(Debug, Clone, PartialEq)]
pub struct RobotControlOutcome {
    /// The response to send back, or `None` when the datagram or the response
    /// was lost.
    pub response: Option<sim::RobotControlResponse>,
    /// True when the incoming datagram itself was dropped.
    pub command_lost: bool,
}

/// Apply a `RobotControl` message for `team`.
pub fn apply_robot_control(
    world: &mut World,
    team: Team,
    control: &sim::RobotControl,
) -> RobotControlOutcome {
    let command_loss = world.config().realism.robot_command_loss;
    if chance(&mut world.rngs_mut().packet_loss, command_loss) {
        tracing::debug!(team = ?team, "dropping robot control datagram (robot_command_loss)");
        return RobotControlOutcome {
            response: None,
            command_lost: true,
        };
    }

    let response = build_response(world, team, control);

    let response_loss = world.config().realism.robot_response_loss;
    if chance(&mut world.rngs_mut().packet_loss, response_loss) {
        tracing::debug!(team = ?team, "dropping robot control response (robot_response_loss)");
        return RobotControlOutcome {
            response: None,
            command_lost: false,
        };
    }

    RobotControlOutcome {
        response: Some(response),
        command_lost: false,
    }
}

/// Apply the commands and build the response without drawing any packet loss.
/// Used by the synchronous path, where loss is drawn by the caller.
pub fn build_response(
    world: &mut World,
    team: Team,
    control: &sim::RobotControl,
) -> sim::RobotControlResponse {
    let mut errors = Vec::new();
    let mut ids = Vec::new();

    for cmd in &control.robot_commands {
        if cmd.id > u8::MAX as u32 {
            errors.push(convert::error(
                "UNKNOWN_ROBOT",
                format!("robot id {} out of range", cmd.id),
            ));
            continue;
        }
        let id = RobotId::new(team, cmd.id as u8);
        let core_cmd = convert::robot_command_from_proto(cmd);
        match world.set_robot_command(id, core_cmd) {
            Ok(()) => ids.push(id),
            // Unknown ids are reported but the rest of the message is applied.
            Err(e) => errors.push(convert::sim_error(&e)),
        }
    }

    let feedback = ids
        .into_iter()
        .map(|id| sim::RobotFeedback {
            id: id.number as u32,
            dribbler_ball_contact: world.robots().get(&id).map(|r| r.ball_contact),
            custom: None,
        })
        .collect();

    sim::RobotControlResponse { errors, feedback }
}

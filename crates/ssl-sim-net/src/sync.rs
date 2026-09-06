//! Synchronous (lock-step) stepping: `SimulationSyncRequest` /
//! `SimulationSyncResponse`.
//!
//! A sync request may arrive on the control port or on a team port. On a team
//! port the embedded `robot_control` is attributed to that team; on the control
//! port there is no team to attribute it to, so it is rejected with
//! `UNSUPPORTED`.

use ssl_sim_core::types::{SimError, SimTime, Team};
use ssl_sim_core::vision::VisionOutput;
use ssl_sim_core::World;
use ssl_sim_proto::sim;

use crate::{control, convert, robot_control};

/// Which socket a sync request arrived on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncSource {
    /// The simulation control port; `robot_control` cannot be attributed.
    Control,
    /// A team port; `robot_control` belongs to this team.
    Team(Team),
}

/// The response plus everything the runner still has to do with the step.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncOutcome {
    /// The message to send back to the requester.
    pub response: sim::SimulationSyncResponse,
    /// Vision outputs released during the step, for the multicast publisher.
    pub vision: Vec<VisionOutput>,
    /// Side effects of the embedded `simulator_command`.
    pub control: control::ControlOutcome,
}

/// Handle one `SimulationSyncRequest`: apply the command, apply the robot
/// control, step, and collect every vision frame released by the step.
pub fn handle_sync_request(
    world: &mut World,
    req: &sim::SimulationSyncRequest,
    source: SyncSource,
) -> SyncOutcome {
    let mut errors = Vec::new();
    let mut control_outcome = control::ControlOutcome::default();

    if let Some(cmd) = &req.simulator_command {
        let outcome = control::apply_simulator_command(world, cmd);
        errors.extend(outcome.errors.iter().cloned());
        control_outcome.merge(control::ControlOutcome {
            errors: Vec::new(),
            ..outcome
        });
    }

    let mut robot_control_response = None;
    if let Some(rc) = &req.robot_control {
        match source {
            SyncSource::Team(team) => {
                let mut resp = robot_control::build_response(world, team, rc);
                resp.errors.append(&mut errors);
                robot_control_response = Some(resp);
            }
            SyncSource::Control => {
                errors.push(convert::error(
                    "UNSUPPORTED",
                    "robot_control in a sync request on the control port has no team; \
                     send the sync request to the blue or yellow port instead",
                ));
            }
        }
    }

    // Step. A zero/absent sim_step is a valid "just poll" request.
    let step = req.sim_step.unwrap_or(0.0);
    if step < 0.0 || !step.is_finite() {
        errors.push(convert::sim_error(&SimError::InvalidStep(format!(
            "sim_step {step} is not a valid duration"
        ))));
    } else if step > 0.0 {
        // `sim_step` is an f32 on the wire (0.016 arrives as 16 000 001 ns), so
        // snap it to the nearest whole substep before stepping; only reject
        // requests that are not a substep multiple beyond f32 precision.
        let substep = world.config().substep_time().as_nanos().max(1);
        let requested = step as f64 * 1e9;
        let n = (requested / substep as f64).round().max(0.0) as u64;
        let snapped = n * substep;
        let tolerance = (requested * 1e-6).max(1000.0);
        if (snapped as f64 - requested).abs() > tolerance {
            errors.push(convert::sim_error(&SimError::InvalidStep(format!(
                "sim_step {step} s is not a multiple of the {substep} ns substep"
            ))));
        } else if let Err(e) = world.step_for(SimTime(snapped)) {
            errors.push(convert::sim_error(&e));
        }
    }

    let vision = world.drain_vision();
    let detection = vision
        .iter()
        .flat_map(|out| out.frames.iter())
        .map(convert::detection_frame_to_proto)
        .collect();

    if !errors.is_empty() {
        robot_control_response
            .get_or_insert_with(Default::default)
            .errors
            .extend(errors);
    }

    SyncOutcome {
        response: sim::SimulationSyncResponse {
            detection,
            robot_control_response,
        },
        vision,
        control: control_outcome,
    }
}

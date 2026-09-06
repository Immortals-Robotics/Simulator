//! Applying `SimulatorCommand` to a [`World`].

use ssl_sim_core::types::SimError;
use ssl_sim_core::World;
use ssl_sim_proto::sim;

use crate::convert;

/// Side effects of a `SimulatorCommand` that the runner (not the world) owns.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ControlOutcome {
    /// Errors to report back to the sender.
    pub errors: Vec<sim::SimulatorError>,
    /// New real-time scaling; `0.0` pauses the simulation.
    pub simulation_speed: Option<f64>,
    /// New vision publish port.
    pub vision_port: Option<u16>,
}

impl ControlOutcome {
    /// Merge another outcome into this one (later values win).
    pub fn merge(&mut self, other: ControlOutcome) {
        self.errors.extend(other.errors);
        if other.simulation_speed.is_some() {
            self.simulation_speed = other.simulation_speed;
        }
        if other.vision_port.is_some() {
            self.vision_port = other.vision_port;
        }
    }

    /// The wire response.
    pub fn response(&self) -> sim::SimulatorResponse {
        sim::SimulatorResponse {
            errors: self.errors.clone(),
        }
    }
}

/// Apply a `SimulatorCommand` to the world.
pub fn apply_simulator_command(world: &mut World, cmd: &sim::SimulatorCommand) -> ControlOutcome {
    let mut out = ControlOutcome::default();
    if let Some(control) = &cmd.control {
        apply_control(world, control, &mut out);
    }
    if let Some(config) = &cmd.config {
        apply_config(world, config, &mut out);
    }
    out
}

fn apply_control(world: &mut World, control: &sim::SimulatorControl, out: &mut ControlOutcome) {
    if let Some(speed) = control.simulation_speed {
        if speed.is_finite() && speed >= 0.0 {
            out.simulation_speed = Some(speed as f64);
        } else {
            out.errors.push(convert::error(
                "UNSUPPORTED",
                format!("simulation_speed {speed} is not a valid scaling"),
            ));
        }
    }
    if let Some(t) = &control.teleport_ball {
        match convert::teleport_ball_from_proto(t) {
            Ok(req) => {
                if let Err(e) = world.teleport_ball(req) {
                    out.errors.push(convert::sim_error(&e));
                }
            }
            Err(e) => out.errors.push(convert::sim_error(&e)),
        }
    }
    for t in &control.teleport_robot {
        match convert::teleport_robot_from_proto(t) {
            Ok(req) => {
                if let Err(e) = world.teleport_robot(req) {
                    out.errors.push(convert::sim_error(&e));
                }
            }
            Err(e) => out.errors.push(convert::sim_error(&e)),
        }
    }
}

fn apply_config(world: &mut World, config: &sim::SimulatorConfig, out: &mut ControlOutcome) {
    if let Some(geo) = &config.geometry {
        // Rebuilds the collision geometry; ball and robot state are untouched.
        let field = convert::field_from_geometry(*world.field(), geo);
        world.set_field(field);
        // Camera calibrations reposition the rig; an empty list keeps it.
        let cameras = convert::cameras_from_geometry(geo);
        if !cameras.is_empty() {
            world.set_cameras(cameras);
        }
    }

    for spec in &config.robot_specs {
        match convert::robot_id_from_proto(&spec.id) {
            Ok(id) => {
                // A spec for robot N of a team applies to that robot and also
                // becomes the team default for robots created later. When the
                // robot does not exist only the default changes.
                if let Some(robot) = world.robots().get(&id) {
                    let merged = convert::merge_robot_specs(robot.specs, spec);
                    if let Err(e) = world.set_robot_specs(id, merged) {
                        out.errors.push(convert::sim_error(&e));
                        continue;
                    }
                }
                let default = convert::merge_robot_specs(world.default_specs(id.team), spec);
                if let Err(e) = world.set_default_specs(id.team, default) {
                    out.errors.push(convert::sim_error(&e));
                }
            }
            Err(e) => out.errors.push(convert::sim_error(&e)),
        }
    }

    if let Some(realism) = &config.realism_config {
        let (merged, ignored) = convert::merge_realism(world.config().realism.clone(), realism);
        for type_url in ignored {
            tracing::warn!(%type_url, "ignoring unknown realism custom message");
            out.errors.push(convert::error(
                "UNSUPPORTED",
                format!("unknown realism custom message {type_url}"),
            ));
        }
        world.set_realism(merged);
    }

    if let Some(port) = config.vision_port {
        match u16::try_from(port) {
            Ok(p) => out.vision_port = Some(p),
            Err(_) => out
                .errors
                .push(convert::sim_error(&SimError::Unsupported(format!(
                    "vision_port {port} out of range"
                )))),
        }
    }
}

//! Control-path tests. These build a `World` but never step it, so they do not
//! depend on the physics modules.

use ssl_sim_core::field::Division;
use ssl_sim_core::params::{Realism, SimConfig};
use ssl_sim_core::types::{RobotId, Team};
use ssl_sim_core::World;
use ssl_sim_net::runner::looks_like_sync;
use ssl_sim_net::sync::SyncSource;
use ssl_sim_net::{control, convert, robot_control, sync};
use ssl_sim_proto::sim;

use prost::Message as _;

fn world() -> World {
    let config = SimConfig {
        // Deterministic: no packet loss, no vision noise.
        realism: Realism::none(),
        initial_robots_per_team: 6,
        ..SimConfig::default()
    };
    World::new(config, Division::A)
}

fn control_command(control: sim::SimulatorControl) -> sim::SimulatorCommand {
    sim::SimulatorCommand {
        control: Some(control),
        config: None,
    }
}

#[test]
fn teleport_ball_moves_the_ball() {
    let mut world = world();
    let cmd = control_command(sim::SimulatorControl {
        teleport_ball: Some(sim::TeleportBall {
            x: Some(1.5),
            y: Some(-2.0),
            z: Some(0.3),
            vx: Some(2.0),
            vy: Some(0.0),
            vz: Some(0.0),
            ..Default::default()
        }),
        teleport_robot: Vec::new(),
        simulation_speed: None,
    });
    let outcome = control::apply_simulator_command(&mut world, &cmd);
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    let ball = world.ball().state;
    assert!((ball.pos.x - 1.5).abs() < 1e-9);
    assert!((ball.pos.y + 2.0).abs() < 1e-9);
    // 0.3 is not exact in f32, so compare at single precision.
    assert!((ball.pos.z - 0.3).abs() < 1e-6);
    assert!((ball.vel.x - 2.0).abs() < 1e-9);
}

#[test]
fn partial_teleport_reports_partial_coord() {
    let mut world = world();
    let cmd = control_command(sim::SimulatorControl {
        teleport_ball: Some(sim::TeleportBall {
            x: Some(1.0),
            ..Default::default()
        }),
        teleport_robot: Vec::new(),
        simulation_speed: None,
    });
    let outcome = control::apply_simulator_command(&mut world, &cmd);
    assert_eq!(outcome.errors.len(), 1);
    assert_eq!(outcome.errors[0].code.as_deref(), Some("PARTIAL_COORD"));
}

#[test]
fn teleport_robot_can_add_and_remove() {
    let mut world = world();
    let id = RobotId::new(Team::Yellow, 9);
    assert!(!world.robots().contains_key(&id));

    let add = control_command(sim::SimulatorControl {
        teleport_ball: None,
        teleport_robot: vec![sim::TeleportRobot {
            id: convert::robot_id_to_proto(id),
            x: Some(2.0),
            y: Some(1.0),
            orientation: Some(0.0),
            present: Some(true),
            ..Default::default()
        }],
        simulation_speed: None,
    });
    let outcome = control::apply_simulator_command(&mut world, &add);
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    assert!(world.robots().contains_key(&id));

    let remove = control_command(sim::SimulatorControl {
        teleport_ball: None,
        teleport_robot: vec![sim::TeleportRobot {
            id: convert::robot_id_to_proto(id),
            present: Some(false),
            ..Default::default()
        }],
        simulation_speed: None,
    });
    control::apply_simulator_command(&mut world, &remove);
    assert!(!world.robots().contains_key(&id));
}

#[test]
fn simulation_speed_and_vision_port_leave_the_world_alone() {
    let mut world = world();
    let cmd = sim::SimulatorCommand {
        control: Some(sim::SimulatorControl {
            teleport_ball: None,
            teleport_robot: Vec::new(),
            simulation_speed: Some(0.0),
        }),
        config: Some(sim::SimulatorConfig {
            vision_port: Some(10021),
            ..Default::default()
        }),
    };
    let outcome = control::apply_simulator_command(&mut world, &cmd);
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    assert_eq!(outcome.simulation_speed, Some(0.0));
    assert_eq!(outcome.vision_port, Some(10021));
}

#[test]
fn geometry_config_resizes_the_field() {
    let mut world = world();
    let cmd = sim::SimulatorCommand {
        control: None,
        config: Some(sim::SimulatorConfig {
            geometry: Some(sim::SslGeometryData {
                field: sim::SslGeometryFieldSize {
                    field_length: 9000,
                    field_width: 6000,
                    goal_width: 1000,
                    goal_depth: 180,
                    boundary_width: 300,
                    field_lines: Vec::new(),
                    field_arcs: Vec::new(),
                    penalty_area_depth: Some(1000),
                    penalty_area_width: Some(2000),
                },
                calib: Vec::new(),
                models: None,
            }),
            ..Default::default()
        }),
    };
    let ball_before = world.ball().state;
    let robots_before = world.robots().len();
    let outcome = control::apply_simulator_command(&mut world, &cmd);
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    assert!((world.field().length - 9.0).abs() < 1e-9);
    assert!((world.field().goal_width - 1.0).abs() < 1e-9);
    // The world state itself is untouched.
    assert_eq!(world.ball().state, ball_before);
    assert_eq!(world.robots().len(), robots_before);
}

#[test]
fn robot_specs_apply_to_the_robot_and_to_the_team_default() {
    let mut world = world();
    let id = RobotId::new(Team::Blue, 2);
    let cmd = sim::SimulatorCommand {
        control: None,
        config: Some(sim::SimulatorConfig {
            robot_specs: vec![sim::RobotSpecs {
                id: convert::robot_id_to_proto(id),
                mass: Some(3.25),
                custom: vec![convert::pack_any(
                    "RobotSpecErForce",
                    &sim::RobotSpecErForce {
                        shoot_radius: Some(0.0655),
                        dribbler_width: Some(0.075),
                    },
                )],
                ..Default::default()
            }],
            ..Default::default()
        }),
    };
    let outcome = control::apply_simulator_command(&mut world, &cmd);
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);

    let specs = world.robots().get(&id).unwrap().specs;
    assert!((specs.mass - 3.25).abs() < 1e-6);
    assert!((specs.shoot_radius - 0.0655).abs() < 1e-6);
    assert!((specs.dribbler_width - 0.075).abs() < 1e-6);

    let default = world.default_specs(Team::Blue);
    assert!((default.mass - 3.25).abs() < 1e-6);
    // The other team is untouched.
    assert!((world.default_specs(Team::Yellow).mass - 2.5).abs() < 1e-6);

    // A spec for a robot that does not exist only moves the default.
    let missing = RobotId::new(Team::Yellow, 15);
    let cmd = sim::SimulatorCommand {
        control: None,
        config: Some(sim::SimulatorConfig {
            robot_specs: vec![sim::RobotSpecs {
                id: convert::robot_id_to_proto(missing),
                mass: Some(4.0),
                ..Default::default()
            }],
            ..Default::default()
        }),
    };
    control::apply_simulator_command(&mut world, &cmd);
    assert!(!world.robots().contains_key(&missing));
    assert!((world.default_specs(Team::Yellow).mass - 4.0).abs() < 1e-6);
}

#[test]
fn invalid_specs_are_rejected_with_invalid_spec() {
    let mut world = world();
    let cmd = sim::SimulatorCommand {
        control: None,
        config: Some(sim::SimulatorConfig {
            robot_specs: vec![sim::RobotSpecs {
                id: convert::robot_id_to_proto(RobotId::new(Team::Blue, 1)),
                // A dribbler further out than the hull is nonsense.
                center_to_dribbler: Some(1.0),
                ..Default::default()
            }],
            ..Default::default()
        }),
    };
    let outcome = control::apply_simulator_command(&mut world, &cmd);
    assert!(outcome
        .errors
        .iter()
        .any(|e| e.code.as_deref() == Some("INVALID_SPEC")));
}

#[test]
fn realism_config_reaches_the_world() {
    let mut world = world();
    let cmd = sim::SimulatorCommand {
        control: None,
        config: Some(sim::SimulatorConfig {
            realism_config: Some(sim::RealismConfig {
                custom: vec![convert::pack_any(
                    "RealismConfigErForce",
                    &sim::RealismConfigErForce {
                        stddev_ball_p: Some(0.005),
                        simulate_dribbling: Some(false),
                        vision_delay: Some(20_000_000),
                        ..Default::default()
                    },
                )],
            }),
            ..Default::default()
        }),
    };
    let outcome = control::apply_simulator_command(&mut world, &cmd);
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    assert!((world.config().realism.stddev_ball_p - 0.005).abs() < 1e-9);
    assert!((world.config().realism.vision_delay - 0.020).abs() < 1e-9);
    // simulate_dribbling = false switches every robot to glue mode.
    assert!(world.robots().values().all(|r| r.specs.dribbler.glue));
}

#[test]
fn unknown_realism_custom_types_are_reported_but_not_fatal() {
    let mut world = world();
    let cmd = sim::SimulatorCommand {
        control: None,
        config: Some(sim::SimulatorConfig {
            realism_config: Some(sim::RealismConfig {
                custom: vec![convert::pack_any(
                    "some.other.SimulatorRealism",
                    &sim::SimulatorResponse { errors: Vec::new() },
                )],
            }),
            ..Default::default()
        }),
    };
    let outcome = control::apply_simulator_command(&mut world, &cmd);
    assert_eq!(outcome.errors.len(), 1);
    assert_eq!(outcome.errors[0].code.as_deref(), Some("UNSUPPORTED"));
}

// --------------------------------------------------------------------------
// robot control
// --------------------------------------------------------------------------

#[test]
fn robot_control_stores_commands_and_answers_with_feedback() {
    let mut world = world();
    let control = sim::RobotControl {
        robot_commands: vec![
            sim::RobotCommand {
                id: 0,
                move_command: Some(sim::RobotMoveCommand {
                    command: Some(sim::robot_move_command::Command::LocalVelocity(
                        sim::MoveLocalVelocity {
                            forward: 1.0,
                            left: 0.0,
                            angular: 0.0,
                        },
                    )),
                }),
                kick_speed: Some(6.0),
                kick_angle: Some(45.0),
                dribbler_speed: Some(9000.0),
            },
            sim::RobotCommand {
                id: 1,
                move_command: None,
                kick_speed: None,
                kick_angle: None,
                dribbler_speed: None,
            },
        ],
    };
    let outcome = robot_control::apply_robot_control(&mut world, Team::Blue, &control);
    assert!(!outcome.command_lost);
    let response = outcome.response.expect("response");
    assert!(response.errors.is_empty(), "{:?}", response.errors);
    assert_eq!(response.feedback.len(), 2);
    assert_eq!(response.feedback[0].id, 0);
    assert_eq!(response.feedback[0].dribbler_ball_contact, Some(false));

    let stored = world.robots()[&RobotId::new(Team::Blue, 0)].command;
    assert_eq!(stored.kick_speed, Some(6.0));
    assert!((stored.kick_angle_deg - 45.0).abs() < 1e-6);
    assert_eq!(stored.dribbler_rpm, Some(9000.0));
    // The command went to the blue robot, not the yellow one with the same number.
    assert_eq!(
        world.robots()[&RobotId::new(Team::Yellow, 0)]
            .command
            .kick_speed,
        None
    );
}

#[test]
fn unknown_robot_ids_error_but_the_rest_is_applied() {
    let mut world = world();
    let control = sim::RobotControl {
        robot_commands: vec![
            sim::RobotCommand {
                id: 15, // only 0..5 exist
                ..Default::default()
            },
            sim::RobotCommand {
                id: 3,
                kick_speed: Some(2.0),
                ..Default::default()
            },
        ],
    };
    let response = robot_control::apply_robot_control(&mut world, Team::Yellow, &control)
        .response
        .expect("response");
    assert_eq!(response.errors.len(), 1);
    assert_eq!(response.errors[0].code.as_deref(), Some("UNKNOWN_ROBOT"));
    assert_eq!(response.feedback.len(), 1);
    assert_eq!(response.feedback[0].id, 3);
    assert_eq!(
        world.robots()[&RobotId::new(Team::Yellow, 3)]
            .command
            .kick_speed,
        Some(2.0)
    );
}

#[test]
fn full_command_loss_drops_the_datagram_and_the_response() {
    let mut config = SimConfig {
        realism: Realism::none(),
        initial_robots_per_team: 6,
        ..SimConfig::default()
    };
    config.realism.robot_command_loss = 1.0;
    let mut world = World::new(config, Division::A);
    let control = sim::RobotControl {
        robot_commands: vec![sim::RobotCommand {
            id: 0,
            kick_speed: Some(5.0),
            ..Default::default()
        }],
    };
    let outcome = robot_control::apply_robot_control(&mut world, Team::Blue, &control);
    assert!(outcome.command_lost);
    assert!(outcome.response.is_none());
    assert_eq!(
        world.robots()[&RobotId::new(Team::Blue, 0)]
            .command
            .kick_speed,
        None,
        "a lost datagram must not reach the world"
    );
}

#[test]
fn full_response_loss_applies_the_command_but_stays_silent() {
    let mut config = SimConfig {
        realism: Realism::none(),
        initial_robots_per_team: 6,
        ..SimConfig::default()
    };
    config.realism.robot_response_loss = 1.0;
    let mut world = World::new(config, Division::A);
    let control = sim::RobotControl {
        robot_commands: vec![sim::RobotCommand {
            id: 0,
            kick_speed: Some(5.0),
            ..Default::default()
        }],
    };
    let outcome = robot_control::apply_robot_control(&mut world, Team::Blue, &control);
    assert!(!outcome.command_lost);
    assert!(outcome.response.is_none());
    assert_eq!(
        world.robots()[&RobotId::new(Team::Blue, 0)]
            .command
            .kick_speed,
        Some(5.0)
    );
}

// --------------------------------------------------------------------------
// sync
// --------------------------------------------------------------------------

#[test]
fn sync_on_the_control_port_rejects_robot_control() {
    let mut world = world();
    let req = sim::SimulationSyncRequest {
        sim_step: Some(0.0),
        simulator_command: None,
        robot_control: Some(sim::RobotControl {
            robot_commands: vec![sim::RobotCommand {
                id: 0,
                ..Default::default()
            }],
        }),
    };
    let outcome = sync::handle_sync_request(&mut world, &req, SyncSource::Control);
    let response = outcome.response.robot_control_response.expect("errors");
    assert!(response
        .errors
        .iter()
        .any(|e| e.code.as_deref() == Some("UNSUPPORTED")));
    assert_eq!(
        world.robots()[&RobotId::new(Team::Blue, 0)]
            .command
            .kick_speed,
        None
    );
}

#[test]
fn sync_on_a_team_port_applies_the_robot_control() {
    let mut world = world();
    let req = sim::SimulationSyncRequest {
        sim_step: Some(0.0),
        simulator_command: Some(control_command(sim::SimulatorControl {
            teleport_ball: Some(sim::TeleportBall {
                x: Some(0.5),
                y: Some(0.25),
                ..Default::default()
            }),
            teleport_robot: Vec::new(),
            simulation_speed: Some(2.0),
        })),
        robot_control: Some(sim::RobotControl {
            robot_commands: vec![sim::RobotCommand {
                id: 1,
                kick_speed: Some(4.0),
                ..Default::default()
            }],
        }),
    };
    let outcome = sync::handle_sync_request(&mut world, &req, SyncSource::Team(Team::Yellow));
    let response = outcome.response.robot_control_response.expect("response");
    assert!(response.errors.is_empty(), "{:?}", response.errors);
    assert_eq!(response.feedback.len(), 1);
    assert_eq!(outcome.control.simulation_speed, Some(2.0));
    assert!((world.ball().state.pos.x - 0.5).abs() < 1e-9);
    assert_eq!(
        world.robots()[&RobotId::new(Team::Yellow, 1)]
            .command
            .kick_speed,
        Some(4.0)
    );
}

#[test]
fn a_non_multiple_sim_step_is_an_invalid_step() {
    let mut world = world();
    // The substep is 1 ms; 0.0005 s is not a whole number of substeps.
    let req = sim::SimulationSyncRequest {
        sim_step: Some(0.0005),
        simulator_command: None,
        robot_control: None,
    };
    let outcome = sync::handle_sync_request(&mut world, &req, SyncSource::Control);
    let response = outcome.response.robot_control_response.expect("errors");
    assert!(response
        .errors
        .iter()
        .any(|e| e.code.as_deref() == Some("INVALID_STEP")));
}

// --------------------------------------------------------------------------
// message disambiguation on shared ports
// --------------------------------------------------------------------------

#[test]
fn sync_requests_are_told_apart_from_the_ports_primary_message() {
    let sync_req = sim::SimulationSyncRequest {
        sim_step: Some(0.016),
        simulator_command: None,
        robot_control: Some(sim::RobotControl {
            robot_commands: vec![sim::RobotCommand {
                id: 0,
                ..Default::default()
            }],
        }),
    }
    .encode_to_vec();
    assert!(looks_like_sync(&sync_req, true), "team port");
    assert!(looks_like_sync(&sync_req, false), "control port");

    let robot_control = sim::RobotControl {
        robot_commands: vec![sim::RobotCommand {
            id: 0,
            ..Default::default()
        }],
    }
    .encode_to_vec();
    assert!(!looks_like_sync(&robot_control, true));

    let simulator_command = control_command(sim::SimulatorControl {
        teleport_ball: None,
        teleport_robot: Vec::new(),
        simulation_speed: Some(1.0),
    })
    .encode_to_vec();
    assert!(!looks_like_sync(&simulator_command, false));

    // A sync request carrying only a simulator_command is indistinguishable by
    // tag alone on the control port; the runner falls back to trying both
    // decoders in that case.
    let ambiguous = sim::SimulationSyncRequest {
        sim_step: None,
        simulator_command: Some(sim::SimulatorCommand::default()),
        robot_control: None,
    }
    .encode_to_vec();
    assert!(!looks_like_sync(&ambiguous, false));
    assert!(
        looks_like_sync(&ambiguous, true),
        "field 2 is decisive vs RobotControl"
    );
}

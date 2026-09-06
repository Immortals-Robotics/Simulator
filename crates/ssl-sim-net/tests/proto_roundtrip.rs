//! Encode/decode round trips for every protobuf message the simulator speaks.
//!
//! These only exercise `ssl-sim-proto`; no `World` is involved.

use prost::Message;
use ssl_sim_proto::{grsim, sim, tracked};

fn round_trip<M>(message: M)
where
    M: Message + PartialEq + Default + std::fmt::Debug,
{
    let bytes = message.encode_to_vec();
    let decoded = M::decode(bytes.as_slice()).expect("decode");
    assert_eq!(message, decoded, "round trip mismatch");
    // Encoding is stable.
    assert_eq!(bytes, decoded.encode_to_vec());
}

fn vector2f(x: f32, y: f32) -> sim::Vector2f {
    sim::Vector2f { x, y }
}

fn line(name: &str) -> sim::SslFieldLineSegment {
    sim::SslFieldLineSegment {
        name: name.to_string(),
        p1: vector2f(-6000.0, 4500.0),
        p2: vector2f(6000.0, 4500.0),
        thickness: 10.0,
        r#type: Some(sim::SslFieldShapeType::TopTouchLine as i32),
    }
}

fn arc() -> sim::SslFieldCircularArc {
    sim::SslFieldCircularArc {
        name: "CenterCircle".to_string(),
        center: vector2f(0.0, 0.0),
        radius: 500.0,
        a1: 0.0,
        a2: std::f32::consts::TAU,
        thickness: 10.0,
        r#type: Some(sim::SslFieldShapeType::CenterCircle as i32),
    }
}

fn field_size() -> sim::SslGeometryFieldSize {
    sim::SslGeometryFieldSize {
        field_length: 12000,
        field_width: 9000,
        goal_width: 1800,
        goal_depth: 180,
        boundary_width: 300,
        field_lines: vec![line("TopTouchLine"), line("BottomTouchLine")],
        field_arcs: vec![arc()],
        penalty_area_depth: Some(1800),
        penalty_area_width: Some(3600),
    }
}

fn calibration() -> sim::SslGeometryCameraCalibration {
    sim::SslGeometryCameraCalibration {
        camera_id: 2,
        focal_length: 390.0,
        principal_point_x: 300.0,
        principal_point_y: 300.0,
        distortion: 0.2,
        q0: 1.0,
        q1: 0.0,
        q2: 0.0,
        q3: 0.0,
        tx: 3000.0,
        ty: -2250.0,
        tz: 4000.0,
        derived_camera_world_tx: Some(-3000.0),
        derived_camera_world_ty: Some(-2250.0),
        derived_camera_world_tz: Some(4000.0),
        pixel_image_width: Some(1280),
        pixel_image_height: Some(1024),
    }
}

fn models() -> sim::SslGeometryModels {
    sim::SslGeometryModels {
        straight_two_phase: Some(sim::SslBallModelStraightTwoPhase {
            acc_slide: -3.0,
            acc_roll: -0.3,
            k_switch: 2.0 / 3.0,
        }),
        chip_fixed_loss: Some(sim::SslBallModelChipFixedLoss {
            damping_xy_first_hop: 0.75,
            damping_xy_other_hops: 0.95,
            damping_z: 0.5,
        }),
    }
}

fn geometry() -> sim::SslGeometryData {
    sim::SslGeometryData {
        field: field_size(),
        calib: vec![calibration()],
        models: Some(models()),
    }
}

fn detection_frame() -> sim::SslDetectionFrame {
    sim::SslDetectionFrame {
        frame_number: 42,
        t_capture: 1.5,
        t_sent: 1.51,
        camera_id: 1,
        balls: vec![sim::SslDetectionBall {
            confidence: 1.0,
            area: Some(120),
            x: 1234.0,
            y: -567.0,
            z: Some(21.5),
            pixel_x: 0.0,
            pixel_y: 0.0,
        }],
        robots_yellow: vec![sim::SslDetectionRobot {
            confidence: 1.0,
            robot_id: Some(3),
            x: 100.0,
            y: 200.0,
            orientation: Some(0.5),
            pixel_x: 0.0,
            pixel_y: 0.0,
            height: Some(150.0),
        }],
        robots_blue: Vec::new(),
    }
}

fn robot_id() -> sim::RobotId {
    sim::RobotId {
        id: Some(7),
        team: Some(sim::Team::Yellow as i32),
    }
}

fn robot_limits() -> sim::RobotLimits {
    sim::RobotLimits {
        acc_speedup_absolute_max: Some(4.0),
        acc_speedup_angular_max: Some(50.0),
        acc_brake_absolute_max: Some(6.0),
        acc_brake_angular_max: Some(50.0),
        vel_absolute_max: Some(3.5),
        vel_angular_max: Some(20.0),
    }
}

fn wheel_angles() -> sim::RobotWheelAngles {
    sim::RobotWheelAngles {
        front_right: 60_f32.to_radians(),
        back_right: 135_f32.to_radians(),
        back_left: 225_f32.to_radians(),
        front_left: 300_f32.to_radians(),
    }
}

fn robot_specs() -> sim::RobotSpecs {
    sim::RobotSpecs {
        id: robot_id(),
        radius: Some(0.09),
        height: Some(0.15),
        mass: Some(2.5),
        max_linear_kick_speed: Some(6.5),
        max_chip_kick_speed: Some(5.5),
        center_to_dribbler: Some(0.075),
        limits: Some(robot_limits()),
        wheel_angles: Some(wheel_angles()),
        custom: vec![ssl_sim_net::convert::pack_any(
            "RobotSpecErForce",
            &sim::RobotSpecErForce {
                shoot_radius: Some(0.067),
                dribbler_width: Some(0.07),
            },
        )],
    }
}

fn robot_control() -> sim::RobotControl {
    sim::RobotControl {
        robot_commands: vec![
            sim::RobotCommand {
                id: 1,
                move_command: Some(sim::RobotMoveCommand {
                    command: Some(sim::robot_move_command::Command::LocalVelocity(
                        sim::MoveLocalVelocity {
                            forward: 1.0,
                            left: -0.5,
                            angular: 2.0,
                        },
                    )),
                }),
                kick_speed: Some(6.0),
                kick_angle: Some(45.0),
                dribbler_speed: Some(5000.0),
            },
            sim::RobotCommand {
                id: 2,
                move_command: Some(sim::RobotMoveCommand {
                    command: Some(sim::robot_move_command::Command::WheelVelocity(
                        sim::MoveWheelVelocity {
                            front_right: 1.0,
                            back_right: 2.0,
                            back_left: 3.0,
                            front_left: 4.0,
                        },
                    )),
                }),
                kick_speed: None,
                kick_angle: None,
                dribbler_speed: None,
            },
        ],
    }
}

fn simulator_command() -> sim::SimulatorCommand {
    sim::SimulatorCommand {
        control: Some(sim::SimulatorControl {
            teleport_ball: Some(sim::TeleportBall {
                x: Some(1.0),
                y: Some(2.0),
                z: Some(0.5),
                vx: Some(0.1),
                vy: Some(0.2),
                vz: Some(0.3),
                teleport_safely: Some(true),
                roll: Some(true),
                by_force: Some(false),
            }),
            teleport_robot: vec![sim::TeleportRobot {
                id: robot_id(),
                x: Some(-1.0),
                y: Some(-2.0),
                orientation: Some(1.57),
                v_x: Some(0.0),
                v_y: Some(0.0),
                v_angular: Some(0.0),
                present: Some(true),
                by_force: Some(true),
            }],
            simulation_speed: Some(2.0),
        }),
        config: Some(sim::SimulatorConfig {
            geometry: Some(geometry()),
            robot_specs: vec![robot_specs()],
            realism_config: Some(sim::RealismConfig {
                custom: vec![ssl_sim_net::convert::pack_any(
                    "RealismConfigErForce",
                    &realism_erforce(),
                )],
            }),
            vision_port: Some(10020),
        }),
    }
}

fn realism_erforce() -> sim::RealismConfigErForce {
    sim::RealismConfigErForce {
        stddev_ball_p: Some(0.0014),
        stddev_robot_p: Some(0.0013),
        stddev_robot_phi: Some(0.01),
        stddev_ball_area: Some(6.5),
        enable_invisible_ball: Some(true),
        ball_visibility_threshold: Some(0.4),
        camera_overlap: Some(1.0),
        dribbler_ball_detections: Some(0.05),
        camera_position_error: Some(0.1),
        robot_command_loss: Some(0.03),
        robot_response_loss: Some(0.1),
        missing_ball_detections: Some(0.05),
        vision_delay: Some(35_000_000),
        vision_processing_time: Some(10_000_000),
        simulate_dribbling: Some(true),
        object_position_offset: Some(0.02),
        missing_robot_detections: Some(0.02),
    }
}

#[test]
fn sim_messages_round_trip() {
    round_trip(robot_id());
    round_trip(vector2f(1.0, 2.0));
    round_trip(line("BottomTouchLine"));
    round_trip(arc());
    round_trip(field_size());
    round_trip(calibration());
    round_trip(models().straight_two_phase.unwrap());
    round_trip(models().chip_fixed_loss.unwrap());
    round_trip(models());
    round_trip(geometry());
    round_trip(detection_frame().balls[0]);
    round_trip(detection_frame().robots_yellow[0]);
    round_trip(detection_frame());
    round_trip(sim::SslWrapperPacket {
        detection: Some(detection_frame()),
        geometry: Some(geometry()),
        source: Some(sim::SslSource::Other as i32),
    });
    round_trip(sim::SimulatorError {
        code: Some("PARTIAL_COORD".into()),
        message: Some("needs both x and y".into()),
    });
    round_trip(robot_control().robot_commands[0]);
    round_trip(robot_control().robot_commands[0].move_command.unwrap());
    round_trip(sim::MoveWheelVelocity {
        front_right: 1.0,
        back_right: 2.0,
        back_left: 3.0,
        front_left: 4.0,
    });
    round_trip(sim::MoveLocalVelocity {
        forward: 1.0,
        left: 2.0,
        angular: 3.0,
    });
    round_trip(sim::MoveGlobalVelocity {
        x: 1.0,
        y: 2.0,
        angular: 3.0,
    });
    round_trip(robot_control());
    round_trip(sim::RobotFeedback {
        id: 3,
        dribbler_ball_contact: Some(true),
        custom: None,
    });
    round_trip(sim::RobotControlResponse {
        errors: vec![sim::SimulatorError {
            code: Some("UNKNOWN_ROBOT".into()),
            message: Some("Y9".into()),
        }],
        feedback: vec![sim::RobotFeedback {
            id: 3,
            dribbler_ball_contact: Some(false),
            custom: None,
        }],
    });
    round_trip(robot_limits());
    round_trip(wheel_angles());
    round_trip(robot_specs());
    round_trip(sim::RealismConfig {
        custom: vec![ssl_sim_net::convert::pack_any(
            "RealismConfigErForce",
            &realism_erforce(),
        )],
    });
    round_trip(simulator_command().config.unwrap());
    round_trip(simulator_command().control.unwrap().teleport_ball.unwrap());
    round_trip(simulator_command().control.unwrap().teleport_robot[0]);
    round_trip(simulator_command().control.unwrap());
    round_trip(simulator_command());
    round_trip(sim::SimulatorResponse {
        errors: vec![sim::SimulatorError {
            code: Some("UNSUPPORTED".into()),
            message: Some("nope".into()),
        }],
    });
    round_trip(sim::SimulationSyncRequest {
        sim_step: Some(0.016),
        simulator_command: Some(simulator_command()),
        robot_control: Some(robot_control()),
    });
    round_trip(sim::SimulationSyncResponse {
        detection: vec![detection_frame()],
        robot_control_response: Some(sim::RobotControlResponse {
            errors: Vec::new(),
            feedback: Vec::new(),
        }),
    });
    round_trip(realism_erforce());
    round_trip(sim::RobotSpecErForce {
        shoot_radius: Some(0.067),
        dribbler_width: Some(0.07),
    });
}

#[test]
fn grsim_messages_round_trip() {
    let command = grsim::GrSimRobotCommand {
        id: 4,
        kickspeedx: 3.0,
        kickspeedz: 3.0,
        veltangent: 1.0,
        velnormal: -1.0,
        velangular: 2.0,
        spinner: true,
        wheelsspeed: true,
        wheel1: Some(10.0),
        wheel2: Some(20.0),
        wheel3: Some(30.0),
        wheel4: Some(40.0),
    };
    round_trip(command);
    let commands = grsim::GrSimCommands {
        timestamp: 12.5,
        isteamyellow: true,
        robot_commands: vec![command],
    };
    round_trip(commands.clone());
    let ball = grsim::GrSimBallReplacement {
        x: Some(1.0),
        y: Some(-2.0),
        vx: Some(0.5),
        vy: Some(0.25),
    };
    round_trip(ball);
    let robot = grsim::GrSimRobotReplacement {
        x: 1.0,
        y: 2.0,
        dir: 90.0,
        id: 5,
        yellowteam: false,
        turnon: Some(true),
    };
    round_trip(robot);
    let replacement = grsim::GrSimReplacement {
        ball: Some(ball),
        robots: vec![robot],
    };
    round_trip(replacement.clone());
    round_trip(grsim::GrSimPacket {
        commands: Some(commands),
        replacement: Some(replacement),
    });
    let status = grsim::RobotStatus {
        robot_id: 3,
        infrared: true,
        flat_kick: false,
        chip_kick: true,
    };
    round_trip(status);
    round_trip(grsim::RobotsStatus {
        robots_status: vec![status],
    });
}

#[test]
fn tracked_messages_round_trip() {
    round_trip(tracked::Vector2 { x: 1.0, y: 2.0 });
    round_trip(tracked::Vector3 {
        x: 1.0,
        y: 2.0,
        z: 3.0,
    });
    let id = tracked::RobotId {
        id: 3,
        team_color: tracked::TeamColor::Blue as i32,
    };
    round_trip(id);
    let ball = tracked::TrackedBall {
        pos: tracked::Vector3 {
            x: 0.1,
            y: 0.2,
            z: 0.3,
        },
        vel: Some(tracked::Vector3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        }),
        visibility: Some(1.0),
    };
    round_trip(ball);
    round_trip(tracked::KickedBall {
        pos: tracked::Vector2 { x: 0.0, y: 0.0 },
        vel: tracked::Vector3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        start_timestamp: 1.0,
        stop_timestamp: Some(3.0),
        stop_pos: Some(tracked::Vector2 { x: 3.0, y: 0.0 }),
        robot_id: Some(id),
    });
    let robot = tracked::TrackedRobot {
        robot_id: id,
        pos: tracked::Vector2 { x: -1.0, y: 0.5 },
        orientation: 1.0,
        vel: Some(tracked::Vector2 { x: 0.0, y: 0.0 }),
        vel_angular: Some(0.0),
        visibility: Some(1.0),
    };
    round_trip(robot);
    let frame = tracked::TrackedFrame {
        frame_number: 9,
        timestamp: 4.5,
        balls: vec![ball],
        robots: vec![robot],
        kicked_ball: None,
        capabilities: vec![tracked::Capability::DetectFlyingBalls as i32],
    };
    round_trip(frame.clone());
    round_trip(tracked::TrackerWrapperPacket {
        uuid: "uuid".into(),
        source_name: Some("ssl-sim-truth".into()),
        tracked_frame: Some(frame),
    });
}

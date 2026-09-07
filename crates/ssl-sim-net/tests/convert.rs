//! Conversion tests. Everything here builds core structs by hand; no `World`
//! is created and no physics runs.

use ssl_sim_core::field::{Division, FieldGeometry};
use ssl_sim_core::params::{BallParams, Realism, RobotSpecs};
use ssl_sim_core::types::{BallState, MoveCommand, RobotId, RobotState, SimTime, Team, Vec2, Vec3};
use ssl_sim_core::vision::{
    CameraCalibration, DetectedBall, DetectedRobot, DetectionFrame, GeometryData, VisionOutput,
};
use ssl_sim_core::world::WorldSnapshot;
use ssl_sim_net::convert;
use ssl_sim_proto::sim;

const EPS: f64 = 1e-6;

// --------------------------------------------------------------------------
// robot control
// --------------------------------------------------------------------------

#[test]
fn local_velocity_command_maps_straight_through() {
    let proto = sim::RobotCommand {
        id: 3,
        move_command: Some(sim::RobotMoveCommand {
            command: Some(sim::robot_move_command::Command::LocalVelocity(
                sim::MoveLocalVelocity {
                    forward: 1.25,
                    left: -0.5,
                    angular: 2.0,
                },
            )),
        }),
        kick_speed: Some(6.0),
        kick_angle: Some(45.0),
        dribbler_speed: Some(5000.0),
    };
    let cmd = convert::robot_command_from_proto(&proto);
    assert_eq!(
        cmd.movement,
        Some(MoveCommand::LocalVelocity {
            forward: 1.25,
            left: -0.5,
            angular: 2.0,
        })
    );
    assert_eq!(cmd.kick_speed, Some(6.0));
    assert!((cmd.kick_angle_deg - 45.0).abs() < EPS);
    assert_eq!(cmd.dribbler_rpm, Some(5000.0));
}

#[test]
fn wheel_velocity_keeps_protocol_order_and_units() {
    let proto = sim::RobotCommand {
        id: 0,
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
    };
    let cmd = convert::robot_command_from_proto(&proto);
    assert_eq!(
        cmd.movement,
        Some(MoveCommand::WheelVelocity {
            front_right: 1.0,
            back_right: 2.0,
            back_left: 3.0,
            front_left: 4.0,
        })
    );
}

#[test]
fn zero_kick_and_dribbler_become_none() {
    let proto = sim::RobotCommand {
        id: 0,
        move_command: None,
        kick_speed: Some(0.0),
        kick_angle: Some(0.0),
        dribbler_speed: Some(0.0),
    };
    let cmd = convert::robot_command_from_proto(&proto);
    assert_eq!(cmd.movement, None);
    assert_eq!(cmd.kick_speed, None);
    assert_eq!(cmd.dribbler_rpm, None);
}

#[test]
fn robot_command_round_trips_through_the_protocol() {
    let original = ssl_sim_core::types::RobotCommand {
        movement: Some(MoveCommand::GlobalVelocity {
            x: 1.0,
            y: -2.0,
            angular: 0.5,
        }),
        kick_speed: Some(3.5),
        kick_angle_deg: 45.0,
        dribbler_rpm: Some(8000.0),
    };
    let proto = convert::robot_command_to_proto(5, &original);
    assert_eq!(proto.id, 5);
    assert_eq!(convert::robot_command_from_proto(&proto), original);
}

// --------------------------------------------------------------------------
// robot specs
// --------------------------------------------------------------------------

#[test]
fn wheel_angles_are_negated_from_clockwise_to_ccw() {
    // The protocol documents wheel angles clockwise; grSim's layout is
    // 60/135/225/300 clockwise, which is -60/-135/+135/+60 in the core's CCW
    // convention (225 deg CW == -225 deg == +135 deg).
    let proto = sim::RobotWheelAngles {
        front_right: 60_f32.to_radians(),
        back_right: 135_f32.to_radians(),
        back_left: 225_f32.to_radians(),
        front_left: 300_f32.to_radians(),
    };
    let angles = convert::wheel_angles_from_proto(&proto);
    assert!((angles.front_right - (-60_f64).to_radians()).abs() < 1e-6);
    assert!((angles.back_right - (-135_f64).to_radians()).abs() < 1e-6);
    assert!((angles.back_left - (-225_f64).to_radians()).abs() < 1e-6);
    assert!((angles.front_left - (-300_f64).to_radians()).abs() < 1e-6);
    // -225 deg and +135 deg name the same direction.
    assert!(
        (ssl_sim_core::types::wrap_angle(angles.back_left) - 135_f64.to_radians()).abs() < 1e-6
    );
    assert!(
        (ssl_sim_core::types::wrap_angle(angles.front_left) - 60_f64.to_radians()).abs() < 1e-6
    );

    // And back out again.
    let back = convert::wheel_angles_to_proto(&angles);
    assert!((back.front_right - proto.front_right).abs() < 1e-6);
    assert!((back.front_left - proto.front_left).abs() < 1e-6);
}

#[test]
fn partial_specs_only_override_set_fields() {
    let current = RobotSpecs {
        mass: 3.0,
        radius: 0.085,
        shoot_radius: 0.06,
        ..RobotSpecs::default()
    };
    let proto = sim::RobotSpecs {
        id: convert::robot_id_to_proto(RobotId::new(Team::Blue, 2)),
        radius: None,
        height: None,
        mass: Some(2.0),
        max_linear_kick_speed: Some(8.0),
        max_chip_kick_speed: None,
        center_to_dribbler: None,
        limits: Some(sim::RobotLimits {
            acc_speedup_absolute_max: None,
            acc_speedup_angular_max: None,
            acc_brake_absolute_max: None,
            acc_brake_angular_max: None,
            vel_absolute_max: Some(2.0),
            vel_angular_max: None,
        }),
        wheel_angles: None,
        custom: Vec::new(),
    };
    let merged = convert::merge_robot_specs(current, &proto);
    assert_eq!(merged.mass, 2.0);
    assert_eq!(merged.radius, 0.085, "unset radius keeps the current value");
    assert_eq!(merged.shoot_radius, 0.06);
    assert_eq!(merged.kicker.max_linear_speed, 8.0);
    assert_eq!(merged.kicker.max_chip_speed, current.kicker.max_chip_speed);
    assert_eq!(merged.limits.vel_absolute_max, 2.0);
    assert_eq!(
        merged.limits.vel_angular_max, current.limits.vel_angular_max,
        "unset limit keeps the current value"
    );
}

#[test]
fn erforce_custom_spec_is_unpacked() {
    let mut proto = sim::RobotSpecs {
        id: convert::robot_id_to_proto(RobotId::new(Team::Yellow, 4)),
        ..Default::default()
    };
    proto.custom.push(convert::pack_any(
        "some.unknown.Message",
        &sim::SimulatorError {
            code: Some("nope".into()),
            message: None,
        },
    ));
    proto.custom.push(convert::pack_any(
        "RobotSpecErForce",
        &sim::RobotSpecErForce {
            shoot_radius: Some(0.0655),
            dribbler_width: Some(0.08),
        },
    ));
    let merged = convert::merge_robot_specs(RobotSpecs::default(), &proto);
    assert!((merged.shoot_radius - 0.0655).abs() < 1e-6);
    assert!((merged.dribbler_width - 0.08).abs() < 1e-6);
}

#[test]
fn specs_survive_a_full_round_trip() {
    let specs = RobotSpecs::default();
    let proto = convert::robot_specs_to_proto(RobotId::new(Team::Blue, 0), &specs);
    let merged = convert::merge_robot_specs(RobotSpecs::default(), &proto);
    assert!((merged.radius - specs.radius).abs() < 1e-6);
    assert!((merged.shoot_radius - specs.shoot_radius).abs() < 1e-6);
    assert!((merged.dribbler_width - specs.dribbler_width).abs() < 1e-6);
    assert!((merged.wheel_angles.front_left - specs.wheel_angles.front_left).abs() < 1e-6);
    assert!((merged.wheel_angles.back_right - specs.wheel_angles.back_right).abs() < 1e-6);
}

// --------------------------------------------------------------------------
// realism
// --------------------------------------------------------------------------

#[test]
fn realism_custom_is_unpacked_with_ns_to_seconds() {
    let cfg = sim::RealismConfig {
        custom: vec![
            convert::pack_any(
                "totally.unrelated.Thing",
                &sim::SimulatorResponse { errors: Vec::new() },
            ),
            convert::pack_any(
                "sslsim.RealismConfigErForce",
                &sim::RealismConfigErForce {
                    stddev_ball_p: Some(0.002),
                    vision_delay: Some(35_000_000),
                    vision_processing_time: Some(10_000_000),
                    simulate_dribbling: Some(false),
                    ..Default::default()
                },
            ),
        ],
    };
    let (merged, ignored) = convert::merge_realism(Realism::none(), &cfg);
    assert!((merged.stddev_ball_p - 0.002).abs() < 1e-9);
    assert!((merged.vision_delay - 0.035).abs() < 1e-9);
    assert!((merged.vision_processing_time - 0.010).abs() < 1e-9);
    assert!(!merged.simulate_dribbling);
    // Untouched fields keep their preset value.
    assert_eq!(merged.camera_overlap, Realism::none().camera_overlap);
    assert_eq!(ignored.len(), 1);
    assert!(ignored[0].ends_with("totally.unrelated.Thing"));
}

#[test]
fn realism_round_trips_through_the_erforce_message() {
    let realism = Realism::realistic();
    let proto = convert::realism_to_erforce(&realism);
    let merged = convert::merge_realism_erforce(Realism::none(), &proto);
    assert!((merged.stddev_ball_p - realism.stddev_ball_p).abs() < 1e-9);
    assert!((merged.vision_delay - realism.vision_delay).abs() < 1e-9);
    assert!((merged.robot_command_loss - realism.robot_command_loss).abs() < 1e-7);
    assert_eq!(merged.simulate_dribbling, realism.simulate_dribbling);
}

// --------------------------------------------------------------------------
// teleports
// --------------------------------------------------------------------------

#[test]
fn teleport_ball_partial_coordinates_are_rejected() {
    let only_x = sim::TeleportBall {
        x: Some(1.0),
        ..Default::default()
    };
    let err = convert::teleport_ball_from_proto(&only_x).unwrap_err();
    assert_eq!(err.code(), "PARTIAL_COORD");

    let only_z = sim::TeleportBall {
        z: Some(0.5),
        ..Default::default()
    };
    assert_eq!(
        convert::teleport_ball_from_proto(&only_z)
            .unwrap_err()
            .code(),
        "PARTIAL_COORD"
    );

    let safely = sim::TeleportBall {
        teleport_safely: Some(true),
        ..Default::default()
    };
    assert_eq!(
        convert::teleport_ball_from_proto(&safely)
            .unwrap_err()
            .code(),
        "TELEPORT_SAFELY_PARTIAL"
    );
}

#[test]
fn teleport_ball_full_coordinates_convert() {
    let proto = sim::TeleportBall {
        x: Some(1.0),
        y: Some(-2.0),
        z: Some(0.25),
        vx: Some(3.0),
        vy: Some(4.0),
        vz: Some(0.0),
        teleport_safely: Some(true),
        roll: Some(true),
        by_force: Some(false),
    };
    let req = convert::teleport_ball_from_proto(&proto).unwrap();
    assert_eq!(req.position, Some(Vec3::new(1.0, -2.0, 0.25)));
    assert_eq!(req.velocity, Some(Vec3::new(3.0, 4.0, 0.0)));
    assert!(req.teleport_safely && req.roll && !req.by_force);
}

#[test]
fn teleport_robot_accepts_position_without_orientation() {
    // The Immortals `Software` client sends x/y and by_force without an
    // orientation; ER-Force would reject that, we must not.
    let proto = sim::TeleportRobot {
        id: convert::robot_id_to_proto(RobotId::new(Team::Blue, 4)),
        x: Some(1.5),
        y: Some(-0.5),
        orientation: None,
        v_x: None,
        v_y: None,
        v_angular: None,
        present: None,
        by_force: Some(true),
    };
    let req = convert::teleport_robot_from_proto(&proto).unwrap();
    assert_eq!(req.id, RobotId::new(Team::Blue, 4));
    assert_eq!(req.position, Some(Vec2::new(1.5, -0.5)));
    assert_eq!(req.orientation, None);
    assert!(req.by_force);

    let only_x = sim::TeleportRobot {
        id: convert::robot_id_to_proto(RobotId::new(Team::Blue, 4)),
        x: Some(1.5),
        ..Default::default()
    };
    assert_eq!(
        convert::teleport_robot_from_proto(&only_x)
            .unwrap_err()
            .code(),
        "PARTIAL_COORD"
    );
}

// --------------------------------------------------------------------------
// geometry: wire -> core
// --------------------------------------------------------------------------

#[test]
fn geometry_sets_field_dimensions_and_penalty_area() {
    let field = FieldGeometry::division(Division::A);
    let proto = sim::SslGeometryData {
        field: sim::SslGeometryFieldSize {
            field_length: 9000,
            field_width: 6000,
            goal_width: 1000,
            goal_depth: 180,
            boundary_width: 250,
            field_lines: Vec::new(),
            field_arcs: Vec::new(),
            penalty_area_depth: Some(1000),
            penalty_area_width: Some(2000),
        },
        calib: Vec::new(),
        models: None,
    };
    let merged = convert::field_from_geometry(field, &proto);
    assert!((merged.length - 9.0).abs() < EPS);
    assert!((merged.width - 6.0).abs() < EPS);
    assert!((merged.goal_width - 1.0).abs() < EPS);
    assert!((merged.boundary_width - 0.25).abs() < EPS);
    assert!((merged.penalty_area_depth - 1.0).abs() < EPS);
    assert!((merged.penalty_area_width - 2.0).abs() < EPS);
    // Not on the wire; unchanged.
    assert_eq!(
        merged.goal_height,
        FieldGeometry::division(Division::A).goal_height
    );
}

#[test]
fn penalty_area_is_derived_from_the_stretch_lines_when_absent() {
    // Emit a Division B field through the core and strip the explicit penalty
    // area fields; the stretch lines must be enough to recover it.
    let source = FieldGeometry::division(Division::B);
    let lines = source.lines();
    let arcs = source.arcs();
    let mut field_size = convert::field_size_to_proto(&source, &lines, &arcs);
    field_size.penalty_area_depth = None;
    field_size.penalty_area_width = None;

    let proto = sim::SslGeometryData {
        field: field_size,
        calib: Vec::new(),
        models: None,
    };
    let merged = convert::field_from_geometry(FieldGeometry::division(Division::A), &proto);
    assert!((merged.penalty_area_depth - source.penalty_area_depth).abs() < 1e-3);
    assert!((merged.penalty_area_width - source.penalty_area_width).abs() < 1e-3);
    assert!((merged.center_circle_radius - source.center_circle_radius).abs() < 1e-6);
    assert!((merged.line_thickness - source.line_thickness).abs() < 1e-6);
}

// --------------------------------------------------------------------------
// vision output: core -> wire
// --------------------------------------------------------------------------

/// One camera's capture. Cameras have independent capture instants, so a
/// `VisionOutput` carries exactly one frame; only camera 0's outputs ever get
/// a geometry packet attached.
fn sample_vision_output(camera_id: u32, with_geometry: bool) -> VisionOutput {
    let frame = |camera_id: u32| DetectionFrame {
        camera_id,
        frame_number: 7 + camera_id,
        t_capture: 1.230,
        t_sent: 1.240,
        balls: vec![DetectedBall {
            pos: Vec2::new(1.5, -2.25),
            z: Some(0.1),
            area: Some(123.4),
            confidence: 0.95,
        }],
        robots_blue: vec![DetectedRobot {
            number: 3,
            pos: Vec2::new(-1.0, 0.5),
            orientation: 0.75,
            height: 0.15,
            confidence: 1.0,
        }],
        robots_yellow: vec![DetectedRobot {
            number: 9,
            pos: Vec2::new(2.0, 0.0),
            orientation: -1.5,
            height: 0.15,
            confidence: 1.0,
        }],
    };
    let field = FieldGeometry::division(Division::A);
    VisionOutput {
        capture_time: SimTime::from_millis(1230),
        camera_index: camera_id as usize,
        frame: frame(camera_id),
        geometry: with_geometry.then(|| GeometryData {
            field,
            cameras: vec![CameraCalibration {
                camera_id: 0,
                position: Vec3::new(-3.0, -2.25, 4.0),
                focal_length: 390.0,
                principal_point: Vec2::new(300.0, 300.0),
                distortion: 0.2,
                q: [0.0, 1.0, 0.0, 0.0],
            }],
            ball: BallParams::default(),
        }),
    }
}

#[test]
fn detections_are_converted_to_millimetres() {
    let out = sample_vision_output(0, false);
    let packet = convert::vision_output_to_packet(&out);

    let detection = packet.detection.as_ref().unwrap();
    assert_eq!(detection.camera_id, 0);
    assert_eq!(detection.frame_number, 7);
    assert!((detection.t_capture - 1.230).abs() < 1e-9);
    assert!((detection.t_sent - 1.240).abs() < 1e-9);

    let ball = &detection.balls[0];
    assert!((ball.x - 1500.0).abs() < 1e-3, "x in mm, got {}", ball.x);
    assert!((ball.y + 2250.0).abs() < 1e-3);
    assert!((ball.z.unwrap() - 100.0).abs() < 1e-3, "z in mm");
    assert_eq!(ball.area, Some(123));
    assert!((ball.confidence - 0.95).abs() < 1e-6);

    let blue = &detection.robots_blue[0];
    assert_eq!(blue.robot_id, Some(3));
    assert!((blue.x + 1000.0).abs() < 1e-3);
    assert!((blue.y - 500.0).abs() < 1e-3);
    assert!(
        (blue.orientation.unwrap() - 0.75).abs() < 1e-6,
        "orientation stays in radians"
    );
    assert!((blue.height.unwrap() - 150.0).abs() < 1e-3, "height in mm");

    let yellow = &detection.robots_yellow[0];
    assert_eq!(yellow.robot_id, Some(9));
    assert!((yellow.x - 2000.0).abs() < 1e-3);

    // No geometry was requested, so the packet carries none.
    assert!(packet.geometry.is_none());
}

#[test]
fn one_packet_per_camera_capture_and_geometry_rides_along() {
    // Camera 1 captured on its own phase: one packet, no geometry.
    let out = sample_vision_output(1, false);
    let packet = convert::vision_output_to_packet(&out);
    assert_eq!(packet.detection.as_ref().unwrap().camera_id, 1);
    assert!(packet.geometry.is_none());

    // Camera 0's capture is the one the core attaches geometry to.
    let out = sample_vision_output(0, true);
    let packet = convert::vision_output_to_packet(&out);
    assert_eq!(packet.detection.as_ref().unwrap().camera_id, 0);
    assert!(packet.geometry.is_some());
}

#[test]
fn ball_area_is_omitted_when_the_core_reports_none() {
    let mut out = sample_vision_output(0, false);
    out.frame.balls[0].area = None;
    let packet = convert::vision_output_to_packet(&out);
    let ball = &packet.detection.as_ref().unwrap().balls[0];
    assert_eq!(ball.area, None, "a rig without `area` sends no area field");
}

#[test]
fn geometry_packet_carries_field_lines_and_ball_models() {
    let out = sample_vision_output(0, true);
    let packet = convert::vision_output_to_packet(&out);
    let geo = packet.geometry.as_ref().unwrap();

    assert_eq!(geo.field.field_length, 12_000);
    assert_eq!(geo.field.field_width, 9_000);
    assert_eq!(geo.field.goal_width, 1_800);
    assert_eq!(geo.field.goal_depth, 180);
    assert_eq!(geo.field.boundary_width, 300);
    assert_eq!(geo.field.penalty_area_depth, Some(1_800));
    assert_eq!(geo.field.penalty_area_width, Some(3_600));

    // The 2018+ line set, with the SSL_FieldShapeType enum filled in.
    let names: Vec<&str> = geo
        .field
        .field_lines
        .iter()
        .map(|l| l.name.as_str())
        .collect();
    for expected in [
        "TopTouchLine",
        "BottomTouchLine",
        "LeftGoalLine",
        "RightGoalLine",
        "HalfwayLine",
        "CenterLine",
        "LeftPenaltyStretch",
        "RightPenaltyStretch",
        "LeftFieldLeftPenaltyStretch",
        "LeftFieldRightPenaltyStretch",
        "RightFieldLeftPenaltyStretch",
        "RightFieldRightPenaltyStretch",
    ] {
        assert!(names.contains(&expected), "missing line {expected}");
        let line = geo
            .field
            .field_lines
            .iter()
            .find(|l| l.name == expected)
            .unwrap();
        assert_eq!(
            line.r#type,
            sim::SslFieldShapeType::from_str_name(expected).map(|t| t as i32),
            "shape type for {expected}"
        );
        assert!((line.thickness - 10.0).abs() < 1e-3, "line thickness in mm");
    }
    let touch = geo
        .field
        .field_lines
        .iter()
        .find(|l| l.name == "TopTouchLine")
        .unwrap();
    assert!((touch.p1.x + 6000.0).abs() < 1e-3);
    assert!((touch.p1.y - 4500.0).abs() < 1e-3);

    assert_eq!(geo.field.field_arcs.len(), 1);
    let circle = &geo.field.field_arcs[0];
    assert_eq!(circle.name, "CenterCircle");
    assert!((circle.radius - 500.0).abs() < 1e-3);
    assert_eq!(
        circle.r#type,
        Some(sim::SslFieldShapeType::CenterCircle as i32)
    );

    // Ball models are the constants the core simulates, except that the wire
    // model has only a constant rolling deceleration: the packet advertises the
    // value at `advertise_roll_speed`, not the zero-speed `acc_roll`.
    let params = BallParams::default();
    let models = geo.models.as_ref().unwrap();
    let straight = models.straight_two_phase.as_ref().unwrap();
    assert!((straight.acc_slide - params.acc_slide).abs() < 1e-12);
    assert!(
        (straight.acc_roll - params.advertised_acc_roll()).abs() < 1e-12,
        "advertised acc_roll {} should be the value at {} m/s",
        straight.acc_roll,
        params.advertise_roll_speed
    );
    assert!(
        straight.acc_roll < params.acc_roll,
        "the speed term makes the advertised value more negative than acc_roll"
    );
    assert!((straight.k_switch - params.k_switch()).abs() < 1e-12);
    let chip = models.chip_fixed_loss.as_ref().unwrap();
    assert!((chip.damping_xy_first_hop - params.chip_damping_xy_first_hop).abs() < 1e-12);
    assert!((chip.damping_xy_other_hops - params.chip_damping_xy_other_hops).abs() < 1e-12);
    assert!((chip.damping_z - params.chip_damping_z).abs() < 1e-12);
}

#[test]
fn camera_calibration_extrinsics_are_self_consistent() {
    let out = sample_vision_output(0, true);
    let packet = convert::vision_output_to_packet(&out);
    let calib = &packet.geometry.as_ref().unwrap().calib[0];

    // Camera at (-3, -2.25, 4) m => (-3000, -2250, 4000) mm.
    assert_eq!(calib.derived_camera_world_tx, Some(-3000.0));
    assert_eq!(calib.derived_camera_world_ty, Some(-2250.0));
    assert_eq!(calib.derived_camera_world_tz, Some(4000.0));
    // R = diag(1, -1, -1) for a downward-looking camera => t = -R C.
    assert!((calib.tx - 3000.0).abs() < 1e-3);
    assert!((calib.ty + 2250.0).abs() < 1e-3);
    assert!((calib.tz - 4000.0).abs() < 1e-3);
    // Core stores (w, x, y, z) = (0, 1, 0, 0); the wire is q0..q3 = (x, y, z, w).
    assert!((calib.q0 - 1.0).abs() < 1e-6);
    assert!(calib.q1.abs() < 1e-6);
    assert!(calib.q2.abs() < 1e-6);
    assert!(calib.q3.abs() < 1e-6);
    assert!((calib.focal_length - 390.0).abs() < 1e-6);
}

// --------------------------------------------------------------------------
// ground truth
// --------------------------------------------------------------------------

#[test]
fn snapshot_becomes_a_tracked_frame_in_metres() {
    let snapshot = WorldSnapshot {
        time: SimTime::from_millis(2500),
        frame: 2500,
        ball: BallState {
            pos: Vec3::new(1.0, -2.0, 0.35),
            vel: Vec3::new(3.0, 0.0, 1.5),
            spin: Vec2::new(3.0, 0.0),
        },
        robots: vec![
            RobotState {
                id: RobotId::new(Team::Blue, 1),
                pos: Vec2::new(-1.0, 0.0),
                orientation: 0.5,
                vel: Vec2::new(0.1, 0.2),
                angular_velocity: 1.0,
                ball_contact: true,
                kicker_charged: true,
                dribbling: true,
            },
            RobotState {
                id: RobotId::new(Team::Yellow, 2),
                pos: Vec2::new(1.0, 0.0),
                orientation: -0.5,
                vel: Vec2::ZERO,
                angular_velocity: 0.0,
                ball_contact: false,
                kicker_charged: true,
                dribbling: false,
            },
        ],
    };
    let packet = convert::snapshot_to_tracker(&snapshot, 11);
    assert_eq!(packet.source_name.as_deref(), Some("ssl-sim-truth"));
    assert_eq!(packet.uuid, convert::TRUTH_UUID);

    let frame = packet.tracked_frame.unwrap();
    assert_eq!(frame.frame_number, 11);
    assert!((frame.timestamp - 2.5).abs() < 1e-9);

    // Tracked protos are metres; nothing is scaled.
    let ball = frame.balls[0];
    assert!((ball.pos.x - 1.0).abs() < 1e-6);
    assert!((ball.pos.z - 0.35).abs() < 1e-6);
    assert!((ball.vel.unwrap().z - 1.5).abs() < 1e-6);
    assert_eq!(ball.visibility, Some(1.0));

    assert_eq!(frame.robots.len(), 2);
    assert_eq!(frame.robots[0].robot_id.id, 1);
    assert_eq!(
        frame.robots[0].robot_id.team_color,
        ssl_sim_proto::tracked::TeamColor::Blue as i32
    );
    assert!((frame.robots[0].pos.x + 1.0).abs() < 1e-6);
    assert!((frame.robots[0].orientation - 0.5).abs() < 1e-6);
    assert_eq!(frame.robots[0].visibility, Some(1.0));
    assert_eq!(
        frame.robots[1].robot_id.team_color,
        ssl_sim_proto::tracked::TeamColor::Yellow as i32
    );
}

// --------------------------------------------------------------------------
// errors
// --------------------------------------------------------------------------

#[test]
fn core_errors_keep_their_stable_wire_code() {
    use ssl_sim_core::types::SimError;
    for err in [
        SimError::PartialCoord("x".into()),
        SimError::VelocityForce("x".into()),
        SimError::TeleportSafelyPartial("x".into()),
        SimError::CreateNoPosRobot("x".into()),
        SimError::InvalidSpec("x".into()),
        SimError::Unsupported("x".into()),
        SimError::UnknownRobot(RobotId::new(Team::Blue, 1)),
        SimError::InvalidStep("x".into()),
    ] {
        let proto = convert::sim_error(&err);
        assert_eq!(proto.code.as_deref(), Some(err.code()));
        assert!(proto.message.is_some());
    }
}

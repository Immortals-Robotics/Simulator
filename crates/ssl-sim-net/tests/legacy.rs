//! Legacy grSim adapter: command mapping, replacement units, status tracking.

use std::net::{IpAddr, SocketAddr};

use ssl_sim_core::params::RobotSpecs;
use ssl_sim_core::types::{Event, MoveCommand, RobotId, Team, Vec2};
use ssl_sim_net::legacy;
use ssl_sim_proto::grsim;

fn base_command() -> grsim::GrSimRobotCommand {
    grsim::GrSimRobotCommand {
        id: 3,
        kickspeedx: 0.0,
        kickspeedz: 0.0,
        veltangent: 0.0,
        velnormal: 0.0,
        velangular: 0.0,
        spinner: false,
        wheelsspeed: false,
        wheel1: None,
        wheel2: None,
        wheel3: None,
        wheel4: None,
    }
}

#[test]
fn velocities_map_tangent_forward_normal_left() {
    let cmd = grsim::GrSimRobotCommand {
        veltangent: 1.5,
        velnormal: -0.25,
        velangular: 2.0,
        ..base_command()
    };
    let core = legacy::robot_command_from_grsim(&cmd, None);
    assert_eq!(
        core.movement,
        Some(MoveCommand::LocalVelocity {
            forward: 1.5,
            left: -0.25,
            angular: 2.0,
        })
    );
    assert_eq!(core.kick_speed, None);
    assert_eq!(core.dribbler_rpm, None);
}

#[test]
fn wheel_speeds_convert_rad_per_second_to_metres_and_reverse_the_order() {
    // grSim wheel1..4 are front_left, back_left, back_right, front_right;
    // the protocol order is front_right, back_right, back_left, front_left.
    let specs = RobotSpecs::default();
    let r = specs.drive.wheel_radius;
    let cmd = grsim::GrSimRobotCommand {
        wheelsspeed: true,
        wheel1: Some(1.0),
        wheel2: Some(2.0),
        wheel3: Some(3.0),
        wheel4: Some(4.0),
        ..base_command()
    };
    let core = legacy::robot_command_from_grsim(&cmd, Some(&specs));
    assert_eq!(
        core.movement,
        Some(MoveCommand::WheelVelocity {
            front_right: 4.0 * r,
            back_right: 3.0 * r,
            back_left: 2.0 * r,
            front_left: 1.0 * r,
        })
    );
}

#[test]
fn straight_kick_uses_the_hypotenuse_and_zero_angle() {
    let cmd = grsim::GrSimRobotCommand {
        kickspeedx: 6.0,
        kickspeedz: 0.0,
        ..base_command()
    };
    let core = legacy::robot_command_from_grsim(&cmd, None);
    assert_eq!(core.kick_speed, Some(6.0));
    assert!(core.kick_angle_deg.abs() < 1e-9);
}

#[test]
fn chip_kick_becomes_speed_and_elevation_in_degrees() {
    // Tyr's grsim sender decomposes a chip magnitude along 45 degrees.
    let magnitude = 4.0_f32;
    let cmd = grsim::GrSimRobotCommand {
        kickspeedx: magnitude * 45_f32.to_radians().cos(),
        kickspeedz: magnitude * 45_f32.to_radians().sin(),
        ..base_command()
    };
    let core = legacy::robot_command_from_grsim(&cmd, None);
    assert!((core.kick_speed.unwrap() - 4.0).abs() < 1e-5);
    assert!((core.kick_angle_deg - 45.0).abs() < 1e-4);
}

#[test]
fn no_kick_when_both_components_are_zero() {
    let core = legacy::robot_command_from_grsim(&base_command(), None);
    assert_eq!(core.kick_speed, None);
    assert_eq!(core.kick_angle_deg, 0.0);
}

#[test]
fn spinner_maps_to_the_specs_full_dribbler_speed() {
    let cmd = grsim::GrSimRobotCommand {
        spinner: true,
        ..base_command()
    };
    assert_eq!(
        legacy::robot_command_from_grsim(&cmd, None).dribbler_rpm,
        Some(legacy::DEFAULT_DRIBBLER_RPM)
    );

    let specs = RobotSpecs {
        dribbler: ssl_sim_core::params::DribblerParams {
            max_speed_rpm: 12_500.0,
            ..Default::default()
        },
        ..RobotSpecs::default()
    };
    assert_eq!(
        legacy::robot_command_from_grsim(&cmd, Some(&specs)).dribbler_rpm,
        Some(12_500.0)
    );
}

#[test]
fn team_comes_from_isteamyellow() {
    let mut cmds = grsim::GrSimCommands {
        timestamp: 0.0,
        isteamyellow: true,
        robot_commands: Vec::new(),
    };
    assert_eq!(legacy::team_of(&cmds), Team::Yellow);
    cmds.isteamyellow = false;
    assert_eq!(legacy::team_of(&cmds), Team::Blue);
}

#[test]
fn ball_replacement_is_metres() {
    let req = legacy::teleport_ball_from_grsim(&grsim::GrSimBallReplacement {
        x: Some(1.25),
        y: Some(-2.5),
        vx: Some(3.0),
        vy: Some(0.0),
    });
    let pos = req.position.unwrap();
    assert!((pos.x - 1.25).abs() < 1e-9);
    assert!((pos.y + 2.5).abs() < 1e-9);
    let vel = req.velocity.unwrap();
    assert!((vel.x - 3.0).abs() < 1e-9);
    assert!(!req.by_force);
}

#[test]
fn robot_replacement_is_metres_and_degrees() {
    let req = legacy::teleport_robot_from_grsim(&grsim::GrSimRobotReplacement {
        x: -1.0,
        y: 2.0,
        dir: 90.0,
        id: 5,
        yellowteam: true,
        turnon: Some(false),
    });
    assert_eq!(req.id, RobotId::new(Team::Yellow, 5));
    assert_eq!(req.position, Some(Vec2::new(-1.0, 2.0)));
    assert!((req.orientation.unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    assert_eq!(req.present, Some(false));

    // `turnon` unset means "leave it on the field".
    let req = legacy::teleport_robot_from_grsim(&grsim::GrSimRobotReplacement {
        x: 0.0,
        y: 0.0,
        dir: 0.0,
        id: 0,
        yellowteam: false,
        turnon: None,
    });
    assert_eq!(req.present, Some(true));
}

#[test]
fn status_ports_and_addresses() {
    assert_eq!(legacy::status_port(Team::Blue), 30011);
    assert_eq!(legacy::status_port(Team::Yellow), 30012);
    let from: SocketAddr = "192.168.1.20:54321".parse().unwrap();
    let to = legacy::status_address(from, Team::Yellow);
    assert_eq!(to.ip(), IpAddr::from([192, 168, 1, 20]));
    assert_eq!(to.port(), 30012);
}

#[test]
fn status_is_only_sent_when_it_changes() {
    let mut tracker = legacy::StatusTracker::new();
    let b0 = RobotId::new(Team::Blue, 0);
    let b1 = RobotId::new(Team::Blue, 1);
    let y0 = RobotId::new(Team::Yellow, 0);
    let robots = [(b0, false), (b1, false), (y0, true)];

    // First call reports everything (nothing has been sent yet).
    let first = tracker
        .changed_status(Team::Blue, robots)
        .expect("initial status");
    assert_eq!(first.robots_status.len(), 2);
    assert!(first.robots_status.iter().all(|s| !s.infrared));

    // Nothing changed.
    assert!(tracker.changed_status(Team::Blue, robots).is_none());

    // Break beam on robot 1 changes.
    let robots = [(b0, false), (b1, true), (y0, true)];
    let changed = tracker.changed_status(Team::Blue, robots).unwrap();
    assert_eq!(changed.robots_status.len(), 1);
    assert_eq!(changed.robots_status[0].robot_id, 1);
    assert!(changed.robots_status[0].infrared);

    // Yellow is tracked separately and still unreported.
    let yellow = tracker.changed_status(Team::Yellow, robots).unwrap();
    assert_eq!(yellow.robots_status.len(), 1);
    assert_eq!(yellow.robots_status[0].robot_id, 0);
    assert!(yellow.robots_status[0].infrared);
}

#[test]
fn kick_events_stay_visible_for_ten_vision_frames() {
    let mut tracker = legacy::StatusTracker::new();
    let b0 = RobotId::new(Team::Blue, 0);

    tracker.record_events(&[Event::Kick {
        robot: b0,
        speed: 5.0,
        angle_deg: 0.0,
    }]);
    let status = tracker.status_of(b0, false);
    assert!(status.flat_kick && !status.chip_kick);

    for _ in 0..(legacy::KICK_STATUS_FRAMES - 1) {
        tracker.advance_frame();
    }
    assert!(
        tracker.status_of(b0, false).flat_kick,
        "still within the window"
    );

    tracker.advance_frame();
    assert!(
        !tracker.status_of(b0, false).flat_kick,
        "expired after 10 frames"
    );

    // A chip sets the other flag.
    tracker.record_events(&[Event::Kick {
        robot: b0,
        speed: 4.0,
        angle_deg: 45.0,
    }]);
    let status = tracker.status_of(b0, true);
    assert!(status.chip_kick && !status.flat_kick && status.infrared);
}

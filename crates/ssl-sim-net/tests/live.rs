//! Tests that need a stepping `World`.
//!
//! Every test in this file is `#[ignore]`d: enable after core physics lands
//! (`ssl_sim_core::physics::step_robots` / `step_ball` and the ball trajectory,
//! collision and drive modules are still `todo!()`, so `World::step` panics).
//!
//! Run them with `cargo test -p ssl-sim-net --test live -- --ignored`.

use std::net::UdpSocket;
use std::thread;
use std::time::{Duration, Instant};

use prost::Message as _;
use ssl_sim_core::field::Division;
use ssl_sim_core::params::{Realism, SimConfig};
use ssl_sim_core::types::Team;
use ssl_sim_core::World;
use ssl_sim_net::endpoints::EndpointPorts;
use ssl_sim_net::sync::SyncSource;
use ssl_sim_net::{sync, Mode, RunOptions, Runner};
use ssl_sim_proto::sim;

fn free_port() -> u16 {
    UdpSocket::bind("127.0.0.1:0")
        .expect("bind")
        .local_addr()
        .expect("local_addr")
        .port()
}

fn config() -> SimConfig {
    SimConfig {
        realism: Realism::none(),
        initial_robots_per_team: 6,
        ..SimConfig::default()
    }
}

#[test]
fn sync_step_returns_one_detection_frame_per_camera() {
    let mut world = World::new(config(), Division::A);
    let cameras = world.vision().cameras().len();
    assert_eq!(cameras, 4, "the default rig has four cameras");

    // Warm up past the vision delay so the queue is in steady state.
    let warmup = sim::SimulationSyncRequest {
        sim_step: Some(0.1),
        simulator_command: None,
        robot_control: None,
    };
    sync::handle_sync_request(&mut world, &warmup, SyncSource::Control);

    // 0.016 s is slightly shorter than the 1/60 s frame period, so an
    // individual step returns either no captures or exactly one set of four
    // frames; over 60 steps the count must match the configured frame rate.
    let request = sim::SimulationSyncRequest {
        sim_step: Some(0.016),
        simulator_command: None,
        robot_control: None,
    };
    let mut total = 0usize;
    let mut steps_with_a_full_set = 0usize;
    for _ in 0..60 {
        let outcome = sync::handle_sync_request(&mut world, &request, SyncSource::Control);
        let n = outcome.response.detection.len();
        assert_eq!(
            n % cameras,
            0,
            "detections come one set per capture, got {n}"
        );
        if n == cameras {
            steps_with_a_full_set += 1;
        }
        total += n;
    }
    assert!(
        steps_with_a_full_set > 0,
        "no step produced a full camera set"
    );

    let captures = total / cameras;
    let expected = 60.0 * 0.016 * world.config().vision.frame_rate;
    assert!(
        (captures as f64 - expected).abs() <= expected * 0.05,
        "expected ~{expected:.1} captures over 0.96 s, got {captures}"
    );
}

#[test]
fn realtime_run_publishes_vision_at_sixty_hertz() {
    let vision_port = free_port();
    let receiver = UdpSocket::bind(("127.0.0.1", vision_port)).expect("vision receiver");
    receiver
        .set_read_timeout(Some(Duration::from_millis(500)))
        .expect("timeout");

    let options = RunOptions {
        mode: Mode::Realtime,
        speed: 1.0,
        division: Division::A,
        ports: EndpointPorts {
            control: free_port(),
            blue: free_port(),
            yellow: free_port(),
            legacy: Some(free_port()),
            localhost: true,
        },
        vision_addr: format!("127.0.0.1:{vision_port}").parse().unwrap(),
        truth: false,
        max_duration: Some(Duration::from_secs(3)),
    };
    let handle = thread::spawn(move || Runner::run(config(), options));

    // Discard the first 300 ms so the run is in steady state, then count the
    // camera-0 detection frames arriving over exactly 2 s.
    let mut buf = vec![0u8; 65_536];
    let settle = Instant::now() + Duration::from_millis(300);
    while Instant::now() < settle {
        let _ = receiver.recv_from(&mut buf);
    }

    let start = Instant::now();
    let window = Duration::from_secs(2);
    let mut frames = 0usize;
    while start.elapsed() < window {
        let Ok((n, _)) = receiver.recv_from(&mut buf) else {
            continue;
        };
        let packet = sim::SslWrapperPacket::decode(&buf[..n]).expect("wrapper packet");
        if let Some(detection) = packet.detection {
            if detection.camera_id == 0 {
                frames += 1;
            }
        }
    }

    let expected = 2.0 * 60.0;
    assert!(
        (frames as f64 - expected).abs() <= expected * 0.05,
        "expected {expected} frames +-5%, got {frames}"
    );

    let _ = handle.join();
}

#[test]
fn kick_events_reach_the_legacy_status_tracker() {
    use ssl_sim_core::types::{MoveCommand, RobotCommand, RobotId, TeleportBall, Vec3};
    use ssl_sim_net::legacy::StatusTracker;

    let mut world = World::new(config(), Division::A);
    let id = RobotId::new(Team::Blue, 0);

    // Put the ball on robot 0's dribbler and ask for a straight kick.
    let robot = world.robots()[&id].clone();
    let seat = robot.pos + robot.heading() * robot.specs.shoot_radius;
    world
        .teleport_ball(TeleportBall {
            position: Some(Vec3::new(seat.x, seat.y, world.config().ball.radius)),
            ..Default::default()
        })
        .expect("teleport");
    world
        .set_robot_command(
            id,
            RobotCommand {
                movement: Some(MoveCommand::LocalVelocity {
                    forward: 0.0,
                    left: 0.0,
                    angular: 0.0,
                }),
                kick_speed: Some(5.0),
                kick_angle_deg: 0.0,
                dribbler_rpm: None,
            },
        )
        .expect("command");

    let mut tracker = StatusTracker::new();
    for _ in 0..10 {
        world.step();
        tracker.record_events(&world.take_events());
    }
    assert!(
        tracker.status_of(id, false).flat_kick,
        "a straight kick must show up as flat_kick"
    );
}

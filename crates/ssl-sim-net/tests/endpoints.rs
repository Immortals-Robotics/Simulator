//! End-to-end UDP tests against a running [`Runner`].
//!
//! The runner is started in `sync` mode, which never advances the world on its
//! own, and every request uses `sim_step = 0`, so no physics runs. That keeps
//! these tests usable while `ssl-sim-core`'s physics modules are still stubs.

use std::net::UdpSocket;
use std::thread;
use std::time::Duration;

use prost::Message as _;
use ssl_sim_core::field::Division;
use ssl_sim_core::params::{Realism, SimConfig};
use ssl_sim_net::endpoints::EndpointPorts;
use ssl_sim_net::{Mode, RunOptions, Runner};
use ssl_sim_proto::{grsim, sim};

/// Ask the OS for a free UDP port, then release it.
fn free_port() -> u16 {
    let socket = UdpSocket::bind("127.0.0.1:0").expect("bind");
    socket.local_addr().expect("local_addr").port()
}

struct Harness {
    ports: EndpointPorts,
    client: UdpSocket,
}

impl Harness {
    fn start() -> Self {
        let ports = EndpointPorts {
            control: free_port(),
            blue: free_port(),
            yellow: free_port(),
            legacy: Some(free_port()),
            localhost: true,
        };
        let vision_port = free_port();
        let options = RunOptions {
            mode: Mode::Sync,
            speed: 1.0,
            division: Division::A,
            ports,
            vision_addr: format!("127.0.0.1:{vision_port}").parse().unwrap(),
            truth: false,
            max_duration: Some(Duration::from_secs(20)),
        };
        let config = SimConfig {
            realism: Realism::none(),
            initial_robots_per_team: 6,
            ..SimConfig::default()
        };
        thread::spawn(move || {
            let _ = Runner::run(config, options);
        });

        let client = UdpSocket::bind("127.0.0.1:0").expect("client bind");
        client
            .set_read_timeout(Some(Duration::from_millis(250)))
            .expect("timeout");
        Self { ports, client }
    }

    /// Send `msg` to `port` and decode the first reply, retrying until the
    /// runner's sockets are up.
    fn request<Req, Res>(&self, port: u16, msg: &Req) -> Res
    where
        Req: prost::Message,
        Res: prost::Message + Default,
    {
        let bytes = msg.encode_to_vec();
        let mut buf = vec![0u8; 65_536];
        for _ in 0..40 {
            if self.client.send_to(&bytes, ("127.0.0.1", port)).is_err() {
                thread::sleep(Duration::from_millis(25));
                continue;
            }
            if let Ok((n, _)) = self.client.recv_from(&mut buf) {
                return Res::decode(&buf[..n]).expect("decode reply");
            }
        }
        panic!("no reply from port {port}");
    }
}

#[test]
fn simulator_command_gets_a_simulator_response() {
    let h = Harness::start();
    let command = sim::SimulatorCommand {
        control: Some(sim::SimulatorControl {
            teleport_ball: Some(sim::TeleportBall {
                x: Some(1.0),
                ..Default::default()
            }),
            teleport_robot: Vec::new(),
            simulation_speed: None,
        }),
        config: None,
    };
    let response: sim::SimulatorResponse = h.request(h.ports.control, &command);
    assert_eq!(response.errors.len(), 1, "x without y is PARTIAL_COORD");
    assert_eq!(response.errors[0].code.as_deref(), Some("PARTIAL_COORD"));
}

#[test]
fn unreadable_control_datagram_gets_unreadable() {
    let h = Harness::start();
    let mut buf = vec![0u8; 4096];
    // 0xFF is not a valid protobuf tag byte start for these messages.
    let garbage = vec![0xFFu8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
    for _ in 0..40 {
        if h.client
            .send_to(&garbage, ("127.0.0.1", h.ports.control))
            .is_err()
        {
            thread::sleep(Duration::from_millis(25));
            continue;
        }
        if let Ok((n, _)) = h.client.recv_from(&mut buf) {
            let response = sim::SimulatorResponse::decode(&buf[..n]).expect("decode");
            assert_eq!(response.errors[0].code.as_deref(), Some("UNREADABLE"));
            return;
        }
    }
    panic!("no reply to the garbage datagram");
}

#[test]
fn robot_control_gets_feedback_on_the_team_port() {
    let h = Harness::start();
    let control = sim::RobotControl {
        robot_commands: vec![sim::RobotCommand {
            id: 2,
            move_command: Some(sim::RobotMoveCommand {
                command: Some(sim::robot_move_command::Command::LocalVelocity(
                    sim::MoveLocalVelocity {
                        forward: 1.0,
                        left: 0.0,
                        angular: 0.0,
                    },
                )),
            }),
            kick_speed: Some(4.0),
            kick_angle: Some(45.0),
            dribbler_speed: Some(1000.0),
        }],
    };
    let response: sim::RobotControlResponse = h.request(h.ports.blue, &control);
    assert!(response.errors.is_empty(), "{:?}", response.errors);
    assert_eq!(response.feedback.len(), 1);
    assert_eq!(response.feedback[0].id, 2);
    assert_eq!(response.feedback[0].dribbler_ball_contact, Some(false));
}

#[test]
fn sync_request_on_a_team_port_gets_a_sync_response() {
    let h = Harness::start();
    let request = sim::SimulationSyncRequest {
        sim_step: Some(0.0),
        simulator_command: None,
        robot_control: Some(sim::RobotControl {
            robot_commands: vec![sim::RobotCommand {
                id: 1,
                kick_speed: Some(3.0),
                ..Default::default()
            }],
        }),
    };
    let response: sim::SimulationSyncResponse = h.request(h.ports.yellow, &request);
    let rcr = response
        .robot_control_response
        .expect("robot control response");
    assert!(rcr.errors.is_empty(), "{:?}", rcr.errors);
    assert_eq!(rcr.feedback.len(), 1);
    assert_eq!(rcr.feedback[0].id, 1);
}

#[test]
fn legacy_grsim_packet_is_accepted_without_a_reply() {
    let h = Harness::start();
    let packet = grsim::GrSimPacket {
        commands: Some(grsim::GrSimCommands {
            timestamp: 0.0,
            isteamyellow: false,
            robot_commands: vec![grsim::GrSimRobotCommand {
                id: 0,
                kickspeedx: 5.0,
                kickspeedz: 0.0,
                veltangent: 1.0,
                velnormal: 0.0,
                velangular: 0.0,
                spinner: true,
                wheelsspeed: false,
                wheel1: None,
                wheel2: None,
                wheel3: None,
                wheel4: None,
            }],
        }),
        replacement: Some(grsim::GrSimReplacement {
            ball: Some(grsim::GrSimBallReplacement {
                x: Some(0.5),
                y: Some(0.5),
                vx: Some(0.0),
                vy: Some(0.0),
            }),
            robots: Vec::new(),
        }),
    };
    let bytes = packet.encode_to_vec();
    let legacy_port = h.ports.legacy.unwrap();
    // The legacy port answers only with Robots_Status, and only once vision
    // frames start flowing, so all we assert here is that the datagram is
    // accepted and the runner keeps serving other ports afterwards.
    for _ in 0..40 {
        if h.client.send_to(&bytes, ("127.0.0.1", legacy_port)).is_ok() {
            break;
        }
        thread::sleep(Duration::from_millis(25));
    }
    let response: sim::SimulatorResponse = h.request(
        h.ports.control,
        &sim::SimulatorCommand {
            control: Some(sim::SimulatorControl::default()),
            config: None,
        },
    );
    assert!(response.errors.is_empty(), "{:?}", response.errors);
}

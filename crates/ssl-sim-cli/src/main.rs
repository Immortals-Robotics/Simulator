use std::{
    net::{SocketAddr, UdpSocket},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use prost::Message;
use rapier3d::prelude::Vector;
use ssl_sim_core::{
    MoveCommand, RobotCommand, RobotId, Simulator, SimulatorConfig, Snapshot, Team, TeleportBall,
    TeleportRobot,
};
use ssl_sim_proto::sim;

const MAX_DATAGRAM_SIZE: usize = 8192;

#[derive(Debug, Parser)]
#[command(name = "ssl-sim")]
#[command(about = "Headless RoboCup SSL simulator")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Run(RunArgs),
}

#[derive(Debug, Parser)]
struct RunArgs {
    #[arg(long, default_value = "0.0.0.0:10300")]
    control_addr: SocketAddr,
    #[arg(long, default_value = "0.0.0.0:10301")]
    blue_addr: SocketAddr,
    #[arg(long, default_value = "0.0.0.0:10302")]
    yellow_addr: SocketAddr,
    #[arg(long, default_value = "224.5.23.2:10020")]
    vision_addr: SocketAddr,
    #[arg(long, default_value_t = 60.0)]
    vision_rate_hz: f32,
    #[arg(long)]
    no_vision: bool,
    #[arg(long)]
    log_commands: bool,
    #[arg(long, value_enum, default_value_t = RunMode::Realtime)]
    mode: RunMode,
    #[arg(long, default_value_t = 2.0)]
    step_ms: f32,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RunMode {
    Realtime,
    Fast,
}

struct Endpoint {
    socket: UdpSocket,
    last_peer: Option<SocketAddr>,
}

impl Endpoint {
    fn bind(addr: SocketAddr) -> Result<Self> {
        let socket = UdpSocket::bind(addr).with_context(|| format!("bind UDP socket {addr}"))?;
        socket
            .set_nonblocking(true)
            .with_context(|| format!("set UDP socket {addr} nonblocking"))?;
        Ok(Self {
            socket,
            last_peer: None,
        })
    }

    fn local_addr(&self) -> Result<SocketAddr> {
        self.socket.local_addr().context("read UDP local address")
    }

    fn send<M: Message>(&self, peer: SocketAddr, message: &M) -> Result<()> {
        let mut buf = Vec::with_capacity(message.encoded_len());
        message.encode(&mut buf)?;
        self.socket.send_to(&buf, peer)?;
        Ok(())
    }
}

struct VisionPublisher {
    socket: UdpSocket,
    target: SocketAddr,
    interval: Duration,
    last_sent: Instant,
}

impl VisionPublisher {
    fn bind(target: SocketAddr, rate_hz: f32) -> Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0").context("bind vision UDP socket")?;
        socket
            .set_multicast_loop_v4(true)
            .context("enable multicast loopback for vision UDP socket")?;
        Ok(Self {
            socket,
            target,
            interval: Duration::from_secs_f32(1.0 / rate_hz.max(1.0)),
            last_sent: Instant::now(),
        })
    }

    fn set_port(&mut self, port: u16) {
        self.target.set_port(port);
    }

    fn maybe_publish(&mut self, snapshot: &Snapshot) -> Result<()> {
        if self.last_sent.elapsed() < self.interval {
            return Ok(());
        }

        self.publish(snapshot)?;
        self.last_sent = Instant::now();
        Ok(())
    }

    fn publish(&self, snapshot: &Snapshot) -> Result<()> {
        let timestamp = unix_now_seconds();
        let packet = sim::SslWrapperPacket {
            detection: Some(detection_from_snapshot(snapshot, timestamp)),
            geometry: None,
            source: Some(sim::SslSource::Other as i32),
        };
        let mut buf = Vec::with_capacity(packet.encoded_len());
        packet.encode(&mut buf)?;
        self.socket.send_to(&buf, self.target)?;
        Ok(())
    }
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Run(args) => run(args),
    }
}

fn run(args: RunArgs) -> Result<()> {
    let fixed_step_seconds = (args.step_ms / 1000.0).max(0.000_1);
    let mut sim = Simulator::new(SimulatorConfig {
        fixed_step_seconds,
        ..SimulatorConfig::default()
    });

    let mut control = Endpoint::bind(args.control_addr)?;
    let mut blue = Endpoint::bind(args.blue_addr)?;
    let mut yellow = Endpoint::bind(args.yellow_addr)?;
    let mut vision = if args.no_vision {
        None
    } else {
        Some(VisionPublisher::bind(
            args.vision_addr,
            args.vision_rate_hz,
        )?)
    };

    eprintln!("simulation control: {}", control.local_addr()?);
    eprintln!("blue robot control: {}", blue.local_addr()?);
    eprintln!("yellow robot control: {}", yellow.local_addr()?);
    if let Some(vision) = &vision {
        eprintln!("vision multicast: {}", vision.target);
    } else {
        eprintln!("vision multicast: disabled");
    }
    eprintln!(
        "mode: {:?}, fixed step: {:.4} s",
        args.mode, fixed_step_seconds
    );

    let mut last = Instant::now();
    let mut accumulator = 0.0_f32;
    let mut last_log = Instant::now();
    let mut speed = 1.0_f32;

    loop {
        speed = handle_control(&mut control, &mut sim, speed, vision.as_mut())?;
        handle_robot_control(&mut blue, &mut sim, Team::Blue, args.log_commands)?;
        handle_robot_control(&mut yellow, &mut sim, Team::Yellow, args.log_commands)?;

        match args.mode {
            RunMode::Realtime => {
                let now = Instant::now();
                accumulator += (now - last).as_secs_f32() * speed;
                last = now;
                while accumulator >= fixed_step_seconds {
                    sim.fixed_step();
                    accumulator -= fixed_step_seconds;
                }
                thread::sleep(Duration::from_millis(1));
            }
            RunMode::Fast => {
                sim.fixed_step();
            }
        }

        let snapshot = sim.snapshot();
        if let Some(vision) = &mut vision {
            vision.maybe_publish(&snapshot)?;
        }

        if last_log.elapsed() >= Duration::from_secs(1) {
            eprintln!(
                "t={:.3}s frame={} ball=({:.3}, {:.3}, {:.3}) robots={}",
                snapshot.time_seconds,
                snapshot.frame_number,
                snapshot.ball.position.x,
                snapshot.ball.position.y,
                snapshot.ball.position.z,
                snapshot.robots.len()
            );
            last_log = Instant::now();
        }
    }
}

fn handle_control(
    endpoint: &mut Endpoint,
    sim_core: &mut Simulator,
    speed: f32,
    mut vision: Option<&mut VisionPublisher>,
) -> Result<f32> {
    let mut current_speed = speed;
    let mut buf = [0_u8; MAX_DATAGRAM_SIZE];

    loop {
        match endpoint.socket.recv_from(&mut buf) {
            Ok((len, peer)) => {
                endpoint.last_peer = Some(peer);
                let response = match sim::SimulatorCommand::decode(&buf[..len]) {
                    Ok(command) => apply_simulator_command(
                        sim_core,
                        command,
                        &mut current_speed,
                        vision.as_deref_mut(),
                    ),
                    Err(err) => response_with_error("decode.simulator_command", err.to_string()),
                };
                endpoint.send(peer, &response)?;
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(err) => return Err(err).context("receive simulator command"),
        }
    }

    Ok(current_speed)
}

fn handle_robot_control(
    endpoint: &mut Endpoint,
    sim_core: &mut Simulator,
    team: Team,
    log_commands: bool,
) -> Result<()> {
    let mut buf = [0_u8; MAX_DATAGRAM_SIZE];

    loop {
        match endpoint.socket.recv_from(&mut buf) {
            Ok((len, peer)) => {
                endpoint.last_peer = Some(peer);
                let response = match sim::RobotControl::decode(&buf[..len]) {
                    Ok(command) => apply_robot_control(sim_core, team, command, log_commands),
                    Err(err) => robot_response_with_error("decode.robot_control", err.to_string()),
                };
                endpoint.send(peer, &response)?;
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(err) => return Err(err).context("receive robot control"),
        }
    }

    Ok(())
}

fn apply_simulator_command(
    sim_core: &mut Simulator,
    command: sim::SimulatorCommand,
    speed: &mut f32,
    mut vision: Option<&mut VisionPublisher>,
) -> sim::SimulatorResponse {
    let mut errors = Vec::new();

    if let Some(config) = command.config {
        if config.geometry.is_some() {
            errors.push(error(
                "config.geometry.unsupported",
                "geometry updates are parsed but not applied yet",
            ));
        }
        if !config.robot_specs.is_empty() {
            errors.push(error(
                "config.robot_specs.unsupported",
                "robot spec updates are parsed but not applied yet",
            ));
        }
        if config.realism_config.is_some() {
            errors.push(error(
                "config.realism.unsupported",
                "custom realism config is not supported yet",
            ));
        }
        if let Some(vision_port) = config.vision_port {
            match (u16::try_from(vision_port), vision.as_deref_mut()) {
                (Ok(port), Some(vision)) => vision.set_port(port),
                (Ok(_), None) => errors.push(error(
                    "config.vision_port.disabled",
                    "vision publishing is disabled",
                )),
                (Err(_), _) => errors.push(error(
                    "config.vision_port.invalid",
                    "vision port must fit into a u16",
                )),
            }
        }
    }

    if let Some(control) = command.control {
        if let Some(simulation_speed) = control.simulation_speed {
            *speed = simulation_speed.max(0.0);
        }

        if let Some(ball) = control.teleport_ball {
            if ball.teleport_safely.unwrap_or(false) {
                errors.push(error(
                    "control.teleport_ball.safely.unsupported",
                    "safe teleport is not implemented yet",
                ));
            }
            if ball.by_force.unwrap_or(false) {
                errors.push(error(
                    "control.teleport_ball.by_force.unsupported",
                    "force teleport is not implemented yet",
                ));
            }
            sim_core.teleport_ball(TeleportBall {
                position: any_ball_position(&ball),
                velocity: any_ball_velocity(&ball),
            });
        }

        for robot in control.teleport_robot {
            if robot.by_force.unwrap_or(false) {
                errors.push(error(
                    "control.teleport_robot.by_force.unsupported",
                    "force robot teleport is not implemented yet",
                ));
            }

            let Some(id) = robot_id_from_proto(robot.id) else {
                errors.push(error(
                    "control.teleport_robot.id.missing",
                    "teleport robot requires a valid RobotId",
                ));
                continue;
            };

            sim_core.teleport_robot(TeleportRobot {
                id,
                present: robot.present,
                x: robot.x,
                y: robot.y,
                orientation: robot.orientation,
                vx: robot.v_x,
                vy: robot.v_y,
                angular: robot.v_angular,
            });
        }
    }

    sim::SimulatorResponse { errors }
}

fn apply_robot_control(
    sim_core: &mut Simulator,
    team: Team,
    control: sim::RobotControl,
    log_commands: bool,
) -> sim::RobotControlResponse {
    let errors = Vec::new();

    for command in control.robot_commands {
        let id = RobotId {
            team,
            id: command.id,
        };

        let movement = command
            .move_command
            .and_then(|move_command| move_command.command)
            .map(move_command_from_proto);
        if log_commands {
            eprintln!(
                "robot command: team={team:?} id={} movement={movement:?} kick={:?}",
                command.id, command.kick_speed
            );
        }

        sim_core.apply_robot_command(RobotCommand {
            id,
            movement,
            kick_speed: command.kick_speed,
            kick_angle_deg: command.kick_angle.unwrap_or(0.0),
            dribbler_speed: command.dribbler_speed,
        });
    }

    let feedback = sim_core
        .snapshot()
        .robots
        .into_iter()
        .filter(|robot| robot.id.team == team)
        .map(|robot| sim::RobotFeedback {
            id: robot.id.id,
            dribbler_ball_contact: Some(robot.dribbler_ball_contact),
            custom: None,
        })
        .collect();

    sim::RobotControlResponse { errors, feedback }
}

fn detection_from_snapshot(snapshot: &Snapshot, timestamp: f64) -> sim::SslDetectionFrame {
    let mut robots_blue = Vec::new();
    let mut robots_yellow = Vec::new();

    for robot in &snapshot.robots {
        let detected = sim::SslDetectionRobot {
            confidence: 1.0,
            robot_id: Some(robot.id.id),
            x: meters_to_millimeters(robot.x),
            y: meters_to_millimeters(robot.y),
            orientation: Some(robot.orientation),
            pixel_x: 0.0,
            pixel_y: 0.0,
            height: Some(meters_to_millimeters(robot.z * 2.0)),
        };

        match robot.id.team {
            Team::Blue => robots_blue.push(detected),
            Team::Yellow => robots_yellow.push(detected),
        }
    }

    sim::SslDetectionFrame {
        frame_number: u32::try_from(snapshot.frame_number).unwrap_or(u32::MAX),
        t_capture: timestamp,
        t_sent: timestamp,
        camera_id: 0,
        balls: vec![sim::SslDetectionBall {
            confidence: 1.0,
            area: None,
            x: meters_to_millimeters(snapshot.ball.position.x),
            y: meters_to_millimeters(snapshot.ball.position.y),
            z: Some(meters_to_millimeters(snapshot.ball.position.z)),
            pixel_x: 0.0,
            pixel_y: 0.0,
        }],
        robots_yellow,
        robots_blue,
    }
}

fn meters_to_millimeters(value: f32) -> f32 {
    value * 1000.0
}

fn unix_now_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn move_command_from_proto(command: sim::robot_move_command::Command) -> MoveCommand {
    match command {
        sim::robot_move_command::Command::LocalVelocity(local) => MoveCommand::LocalVelocity {
            forward: local.forward,
            left: local.left,
            angular: local.angular,
        },
        sim::robot_move_command::Command::GlobalVelocity(global) => MoveCommand::GlobalVelocity {
            x: global.x,
            y: global.y,
            angular: global.angular,
        },
        sim::robot_move_command::Command::WheelVelocity(wheel) => MoveCommand::WheelVelocity {
            front_right: wheel.front_right,
            back_right: wheel.back_right,
            back_left: wheel.back_left,
            front_left: wheel.front_left,
        },
    }
}

fn robot_id_from_proto(id: sim::RobotId) -> Option<RobotId> {
    let team = match sim::Team::try_from(id.team?).ok()? {
        sim::Team::Blue => Team::Blue,
        sim::Team::Yellow => Team::Yellow,
        sim::Team::Unknown => return None,
    };

    Some(RobotId { team, id: id.id? })
}

fn any_ball_position(ball: &sim::TeleportBall) -> Option<Vector> {
    if ball.x.is_none() && ball.y.is_none() && ball.z.is_none() {
        return None;
    }

    Some(Vector::new(
        ball.x.unwrap_or(0.0),
        ball.y.unwrap_or(0.0),
        ball.z.unwrap_or(0.0215),
    ))
}

fn any_ball_velocity(ball: &sim::TeleportBall) -> Option<Vector> {
    if ball.vx.is_none() && ball.vy.is_none() && ball.vz.is_none() {
        return None;
    }

    Some(Vector::new(
        ball.vx.unwrap_or(0.0),
        ball.vy.unwrap_or(0.0),
        ball.vz.unwrap_or(0.0),
    ))
}

fn response_with_error(code: &'static str, message: String) -> sim::SimulatorResponse {
    sim::SimulatorResponse {
        errors: vec![error(code, message)],
    }
}

fn robot_response_with_error(code: &'static str, message: String) -> sim::RobotControlResponse {
    sim::RobotControlResponse {
        errors: vec![error(code, message)],
        feedback: Vec::new(),
    }
}

fn error(code: impl Into<String>, message: impl Into<String>) -> sim::SimulatorError {
    sim::SimulatorError {
        code: Some(code.into()),
        message: Some(message.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protobuf_robot_control_moves_robot() {
        let mut sim_core = Simulator::default();
        let before = sim_core
            .snapshot()
            .robots
            .into_iter()
            .find(|robot| {
                robot.id
                    == RobotId {
                        team: Team::Blue,
                        id: 0,
                    }
            })
            .expect("default blue robot 0 should exist");

        let control = sim::RobotControl {
            robot_commands: vec![sim::RobotCommand {
                id: 0,
                move_command: Some(sim::RobotMoveCommand {
                    command: Some(sim::robot_move_command::Command::GlobalVelocity(
                        sim::MoveGlobalVelocity {
                            x: 1.0,
                            y: 0.0,
                            angular: 0.0,
                        },
                    )),
                }),
                kick_speed: None,
                kick_angle: None,
                dribbler_speed: None,
            }],
        };
        let mut bytes = Vec::new();
        control.encode(&mut bytes).expect("encode robot control");
        let decoded = sim::RobotControl::decode(bytes.as_slice()).expect("decode robot control");

        let response = apply_robot_control(&mut sim_core, Team::Blue, decoded, false);
        sim_core.step(0.1);

        let after = sim_core
            .snapshot()
            .robots
            .into_iter()
            .find(|robot| {
                robot.id
                    == RobotId {
                        team: Team::Blue,
                        id: 0,
                    }
            })
            .expect("default blue robot 0 should exist");

        assert!(response.errors.is_empty());
        assert!(after.x > before.x + 0.05);
    }
}

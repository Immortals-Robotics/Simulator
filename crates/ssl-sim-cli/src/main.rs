use std::{
    net::{SocketAddr, UdpSocket},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use prost::Message;
use rapier3d::prelude::Vector;
use ssl_sim_core::{
    MoveCommand, RobotCommand, RobotId, Simulator, SimulatorConfig, Team, TeleportBall,
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

    eprintln!("simulation control: {}", control.local_addr()?);
    eprintln!("blue robot control: {}", blue.local_addr()?);
    eprintln!("yellow robot control: {}", yellow.local_addr()?);
    eprintln!(
        "mode: {:?}, fixed step: {:.4} s",
        args.mode, fixed_step_seconds
    );

    let mut last = Instant::now();
    let mut accumulator = 0.0_f32;
    let mut last_log = Instant::now();
    let mut speed = 1.0_f32;

    loop {
        speed = handle_control(&mut control, &mut sim, speed)?;
        handle_robot_control(&mut blue, &mut sim, Team::Blue)?;
        handle_robot_control(&mut yellow, &mut sim, Team::Yellow)?;

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

        if last_log.elapsed() >= Duration::from_secs(1) {
            let snapshot = sim.snapshot();
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

fn handle_control(endpoint: &mut Endpoint, sim_core: &mut Simulator, speed: f32) -> Result<f32> {
    let mut current_speed = speed;
    let mut buf = [0_u8; MAX_DATAGRAM_SIZE];

    loop {
        match endpoint.socket.recv_from(&mut buf) {
            Ok((len, peer)) => {
                endpoint.last_peer = Some(peer);
                let response = match sim::SimulatorCommand::decode(&buf[..len]) {
                    Ok(command) => apply_simulator_command(sim_core, command, &mut current_speed),
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
) -> Result<()> {
    let mut buf = [0_u8; MAX_DATAGRAM_SIZE];

    loop {
        match endpoint.socket.recv_from(&mut buf) {
            Ok((len, peer)) => {
                endpoint.last_peer = Some(peer);
                let response = match sim::RobotControl::decode(&buf[..len]) {
                    Ok(command) => apply_robot_control(sim_core, team, command),
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
        if config.vision_port.is_some() {
            errors.push(error(
                "config.vision_port.unsupported",
                "vision publishing is not implemented yet",
            ));
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

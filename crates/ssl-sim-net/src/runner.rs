//! The simulation thread: owns the [`World`], applies incoming datagrams
//! between substeps, publishes vision and answers every request on the socket
//! it arrived on.
//!
//! Three modes:
//!
//! * [`Mode::Realtime`] — accumulate wall time × speed, run whole substeps,
//!   publish due vision packets, then sleep the remainder (spin-waiting the
//!   last ~0.5 ms, because Windows timers are only millisecond-accurate).
//! * [`Mode::Fast`] — never sleep; vision is still generated on sim time.
//! * [`Mode::Sync`] — the world only advances on `SimulationSyncRequest`.
//!
//! In realtime and fast mode the free-running clock is paused while a sync
//! request has been seen within the last [`SYNC_HOLD`].

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError};
use prost::Message as _;

use ssl_sim_core::field::Division;
use ssl_sim_core::params::SimConfig;
use ssl_sim_core::types::{RobotId, Team};
use ssl_sim_core::vision::VisionOutput;
use ssl_sim_core::World;
use ssl_sim_proto::{grsim, sim};

use crate::endpoints::{Datagram, EndpointKind, EndpointPorts, Endpoints};
use crate::legacy::{self, StatusTracker};
use crate::sync::{self, SyncSource};
use crate::vision::VisionPublisher;
use crate::{control, convert, robot_control};

/// While a sync request has been seen this recently, the free-running clock is
/// paused (design §6).
pub const SYNC_HOLD: Duration = Duration::from_secs(1);

/// Never try to catch up more than this much simulation time in one iteration;
/// prevents a spiral of death after a long stall.
const MAX_CATCHUP: f64 = 0.25;

/// Longest single sleep in the realtime loop, so incoming datagrams stay responsive.
const MAX_SLEEP: Duration = Duration::from_millis(5);

/// Spin-wait the last part of a sleep for timer accuracy.
const SPIN_MARGIN: Duration = Duration::from_micros(500);

/// How the runner drives simulation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Follow the wall clock, scaled by `speed`.
    #[default]
    Realtime,
    /// Run as fast as the machine allows.
    Fast,
    /// Only advance on `SimulationSyncRequest`.
    Sync,
}

impl std::str::FromStr for Mode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "realtime" | "real-time" => Ok(Mode::Realtime),
            "fast" => Ok(Mode::Fast),
            "sync" => Ok(Mode::Sync),
            other => Err(format!("unknown mode `{other}` (realtime|fast|sync)")),
        }
    }
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::Realtime => "realtime",
            Mode::Fast => "fast",
            Mode::Sync => "sync",
        })
    }
}

/// Everything the runner needs beyond the [`SimConfig`].
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Stepping mode.
    pub mode: Mode,
    /// Real-time scaling; `0.0` pauses.
    pub speed: f64,
    /// Field preset used to build the world.
    pub division: Division,
    /// UDP ports to bind.
    pub ports: EndpointPorts,
    /// Vision destination.
    pub vision_addr: SocketAddr,
    /// Also publish the ground-truth tracker stream.
    pub truth: bool,
    /// Stop after this long (used by tests and bounded runs).
    pub max_duration: Option<Duration>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            mode: Mode::Realtime,
            speed: 1.0,
            division: Division::A,
            ports: EndpointPorts::default(),
            vision_addr: crate::vision::default_vision_addr(),
            truth: false,
            max_duration: None,
        }
    }
}

/// Owns the world and drives it. Construct through [`Runner::run`].
pub struct Runner {
    world: World,
    publisher: VisionPublisher,
    endpoints: Endpoints,
    status: StatusTracker,
    legacy_peers: BTreeMap<Team, SocketAddr>,
    substep: f64,
    speed: f64,
    last_sync: Option<Instant>,
    // status-line accounting
    steps_since_report: u64,
    substep_nanos: u64,
}

impl Runner {
    /// Build the world, bind the sockets and run until `max_duration` elapses
    /// (or forever when it is `None`).
    pub fn run(config: SimConfig, options: RunOptions) -> anyhow::Result<()> {
        let (tx, rx) = crossbeam_channel::unbounded();
        let endpoints = Endpoints::bind(options.ports, tx)?;
        let publisher = VisionPublisher::new(options.vision_addr, options.truth)?;
        let substep = config.substep;
        let world = World::new(config, options.division);

        tracing::info!(
            mode = %options.mode,
            speed = options.speed,
            control = options.ports.control,
            blue = options.ports.blue,
            yellow = options.ports.yellow,
            legacy = ?options.ports.legacy,
            vision = %options.vision_addr,
            truth = options.truth,
            "simulator listening"
        );

        let mut runner = Runner {
            world,
            publisher,
            endpoints,
            status: StatusTracker::new(),
            legacy_peers: BTreeMap::new(),
            substep,
            speed: options.speed,
            last_sync: None,
            steps_since_report: 0,
            substep_nanos: 0,
        };
        runner.main_loop(&rx, &options)
    }

    fn main_loop(&mut self, rx: &Receiver<Datagram>, options: &RunOptions) -> anyhow::Result<()> {
        let start = Instant::now();
        let mut last_wall = start;
        let mut last_report = start;
        let mut last_report_sim = self.world.time().as_secs_f64();
        let mut accumulator = 0.0f64;

        loop {
            if let Some(limit) = options.max_duration {
                if start.elapsed() >= limit {
                    tracing::info!("run duration reached, stopping");
                    return Ok(());
                }
            }

            match options.mode {
                Mode::Sync => {
                    // Only advance on requests; block until one arrives.
                    match rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(d) => {
                            self.handle_datagram(d);
                            self.drain(rx);
                        }
                        Err(RecvTimeoutError::Timeout) => {}
                        Err(RecvTimeoutError::Disconnected) => return Ok(()),
                    }
                }
                Mode::Fast | Mode::Realtime => {
                    self.drain(rx);
                    let now = Instant::now();
                    let wall_dt = now.duration_since(last_wall).as_secs_f64();
                    last_wall = now;

                    if self.free_running() {
                        if options.mode == Mode::Fast {
                            // No wall-clock coupling: always one substep per iteration.
                            self.drain(rx);
                            self.step_once();
                        } else {
                            accumulator = (accumulator + wall_dt * self.speed).min(MAX_CATCHUP);
                            while accumulator >= self.substep {
                                self.drain(rx);
                                self.step_once();
                                accumulator -= self.substep;
                            }
                        }
                    } else {
                        accumulator = 0.0;
                    }

                    if options.mode == Mode::Realtime {
                        let wait = if self.speed > 0.0 {
                            Duration::from_secs_f64(
                                ((self.substep - accumulator).max(0.0) / self.speed).min(1.0),
                            )
                        } else {
                            Duration::from_millis(2)
                        };
                        sleep_precise(wait.min(MAX_SLEEP));
                    }
                }
            }

            let now = Instant::now();
            if now.duration_since(last_report) >= Duration::from_secs(1) {
                let wall = now.duration_since(last_report).as_secs_f64();
                let sim_now = self.world.time().as_secs_f64();
                let rtf = (sim_now - last_report_sim) / wall.max(1e-9);
                let mean_us = if self.steps_since_report > 0 {
                    self.substep_nanos as f64 / self.steps_since_report as f64 / 1000.0
                } else {
                    0.0
                };
                tracing::info!(
                    "sim {:.2}s | rtf {:.2}x | substep {:.1}us | robots {} | mode {}",
                    sim_now,
                    rtf,
                    mean_us,
                    self.world.robots().len(),
                    options.mode
                );
                last_report = now;
                last_report_sim = sim_now;
                self.steps_since_report = 0;
                self.substep_nanos = 0;
            }
        }
    }

    /// True when the free-running clock should advance.
    fn free_running(&self) -> bool {
        if self.speed <= 0.0 {
            return false;
        }
        match self.last_sync {
            Some(t) => t.elapsed() >= SYNC_HOLD,
            None => true,
        }
    }

    fn drain(&mut self, rx: &Receiver<Datagram>) {
        while let Ok(d) = rx.try_recv() {
            self.handle_datagram(d);
        }
    }

    /// One substep plus everything that hangs off it.
    fn step_once(&mut self) {
        let t0 = Instant::now();
        self.world.step();
        self.substep_nanos += t0.elapsed().as_nanos() as u64;
        self.steps_since_report += 1;

        let events = self.world.take_events();
        if !events.is_empty() {
            self.status.record_events(&events);
        }

        let outputs = self.world.drain_vision();
        if !outputs.is_empty() {
            self.publish_vision(&outputs);
        }
    }

    fn publish_vision(&mut self, outputs: &[VisionOutput]) {
        for out in outputs {
            if let Err(e) = self.publisher.publish(out) {
                tracing::warn!(error = %e, "vision publish failed");
            }
            self.status.advance_frame();
        }
        if self.publisher.truth_enabled() {
            let snapshot = self.world.snapshot();
            if let Err(e) = self.publisher.publish_truth(&snapshot) {
                tracing::warn!(error = %e, "truth publish failed");
            }
        }
        self.send_legacy_status();
    }

    /// Send `Robots_Status` to every known legacy peer whose robots changed.
    fn send_legacy_status(&mut self) {
        if self.legacy_peers.is_empty() {
            return;
        }
        let contacts: Vec<(RobotId, bool)> = self
            .world
            .robots()
            .iter()
            .map(|(id, r)| (*id, r.ball_contact))
            .collect();
        let peers: Vec<(Team, SocketAddr)> =
            self.legacy_peers.iter().map(|(t, a)| (*t, *a)).collect();
        for (team, peer) in peers {
            let Some(msg) = self.status.changed_status(team, contacts.iter().copied()) else {
                continue;
            };
            let to = legacy::status_address(peer, team);
            if let Err(e) = self
                .endpoints
                .reply(EndpointKind::Legacy, to, &msg.encode_to_vec())
            {
                tracing::warn!(error = %e, %to, "legacy status send failed");
            }
        }
    }

    // ----------------------------------------------------------------------
    // datagram dispatch
    // ----------------------------------------------------------------------

    fn handle_datagram(&mut self, d: Datagram) {
        match d.kind {
            EndpointKind::Control => self.handle_control(&d),
            EndpointKind::Blue => self.handle_team(&d, Team::Blue),
            EndpointKind::Yellow => self.handle_team(&d, Team::Yellow),
            EndpointKind::Legacy => self.handle_legacy(&d),
        }
    }

    fn handle_control(&mut self, d: &Datagram) {
        if looks_like_sync(&d.bytes, /* primary_is_robot_control */ false) {
            if let Ok(req) = sim::SimulationSyncRequest::decode(d.bytes.as_slice()) {
                self.handle_sync(d, &req, SyncSource::Control);
                return;
            }
        }
        match sim::SimulatorCommand::decode(d.bytes.as_slice()) {
            Ok(cmd) => {
                let outcome = control::apply_simulator_command(&mut self.world, &cmd);
                self.apply_side_effects(&outcome);
                self.send(EndpointKind::Control, d.from, &outcome.response());
            }
            Err(_) => match sim::SimulationSyncRequest::decode(d.bytes.as_slice()) {
                Ok(req) => self.handle_sync(d, &req, SyncSource::Control),
                Err(e) => {
                    tracing::debug!(error = %e, from = %d.from, "unreadable control datagram");
                    self.send(
                        EndpointKind::Control,
                        d.from,
                        &sim::SimulatorResponse {
                            errors: vec![convert::error(
                                "UNREADABLE",
                                "the received message was unreadable as SimulatorCommand or \
                                 SimulationSyncRequest",
                            )],
                        },
                    );
                }
            },
        }
    }

    fn handle_team(&mut self, d: &Datagram, team: Team) {
        let kind = match team {
            Team::Blue => EndpointKind::Blue,
            Team::Yellow => EndpointKind::Yellow,
        };
        if looks_like_sync(&d.bytes, /* primary_is_robot_control */ true) {
            if let Ok(req) = sim::SimulationSyncRequest::decode(d.bytes.as_slice()) {
                tracing::debug!(
                    from = %d.from,
                    ?team,
                    len = d.bytes.len(),
                    hex = %hex_prefix(&d.bytes, 48),
                    "team datagram classified as SimulationSyncRequest"
                );
                self.handle_sync(d, &req, SyncSource::Team(team));
                return;
            }
        }
        match sim::RobotControl::decode(d.bytes.as_slice()) {
            Ok(rc) => {
                let outcome = robot_control::apply_robot_control(&mut self.world, team, &rc);
                if let Some(response) = outcome.response {
                    self.send(kind, d.from, &response);
                }
            }
            Err(e) => {
                tracing::debug!(error = %e, from = %d.from, ?team, "unreadable robot control");
                self.send(
                    kind,
                    d.from,
                    &sim::RobotControlResponse {
                        errors: vec![convert::error(
                            "UNREADABLE",
                            "the received message was unreadable as RobotControl or \
                             SimulationSyncRequest",
                        )],
                        feedback: Vec::new(),
                    },
                );
            }
        }
    }

    fn handle_sync(&mut self, d: &Datagram, req: &sim::SimulationSyncRequest, src: SyncSource) {
        self.last_sync = Some(Instant::now());
        let outcome = sync::handle_sync_request(&mut self.world, req, src);
        self.apply_side_effects(&outcome.control);
        if !outcome.vision.is_empty() {
            self.publish_vision(&outcome.vision);
        }
        self.send(d.kind, d.from, &outcome.response);
    }

    fn handle_legacy(&mut self, d: &Datagram) {
        let packet = match grsim::GrSimPacket::decode(d.bytes.as_slice()) {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!(error = %e, from = %d.from, "unreadable grSim packet");
                return;
            }
        };

        if let Some(cmds) = &packet.commands {
            let team = legacy::team_of(cmds);
            self.legacy_peers.insert(team, d.from);
            for cmd in &cmds.robot_commands {
                let id = RobotId::new(team, cmd.id.min(u8::MAX as u32) as u8);
                let specs = self.world.robots().get(&id).map(|r| r.specs);
                let core_cmd = legacy::robot_command_from_grsim(cmd, specs.as_ref());
                if let Err(e) = self.world.set_robot_command(id, core_cmd) {
                    tracing::debug!(error = %e, "legacy command for unknown robot");
                }
            }
        }

        if let Some(replacement) = &packet.replacement {
            if let Some(ball) = &replacement.ball {
                if let Err(e) = self
                    .world
                    .teleport_ball(legacy::teleport_ball_from_grsim(ball))
                {
                    tracing::debug!(error = %e, "legacy ball replacement failed");
                }
            }
            for r in &replacement.robots {
                if let Err(e) = self
                    .world
                    .teleport_robot(legacy::teleport_robot_from_grsim(r))
                {
                    tracing::debug!(error = %e, "legacy robot replacement failed");
                }
            }
        }
    }

    fn apply_side_effects(&mut self, outcome: &control::ControlOutcome) {
        if let Some(speed) = outcome.simulation_speed {
            if speed != self.speed {
                tracing::info!(speed, "simulation speed changed");
            }
            self.speed = speed;
        }
        if let Some(port) = outcome.vision_port {
            self.publisher.set_port(port);
        }
    }

    fn send<M: prost::Message>(&self, kind: EndpointKind, to: SocketAddr, msg: &M) {
        if let Err(e) = self.endpoints.reply(kind, to, &msg.encode_to_vec()) {
            tracing::warn!(error = %e, %to, endpoint = kind.name(), "reply failed");
        }
    }
}

/// Sleep for `d`, spin-waiting the last [`SPIN_MARGIN`] for timer accuracy.
pub fn sleep_precise(d: Duration) {
    if d.is_zero() {
        return;
    }
    let target = Instant::now() + d;
    if d > SPIN_MARGIN {
        std::thread::sleep(d - SPIN_MARGIN);
    }
    while Instant::now() < target {
        std::hint::spin_loop();
    }
}

// --------------------------------------------------------------------------
// message disambiguation
// --------------------------------------------------------------------------

/// Decide whether a datagram is a `SimulationSyncRequest` rather than the
/// port's primary message, by scanning top-level protobuf tags.
///
/// `SimulationSyncRequest` is `{1: float sim_step, 2: SimulatorCommand,
/// 3: RobotControl}`. On a team port the primary message is `RobotControl`
/// (`{1: repeated RobotCommand}`), so anything but field 1 with wire type 2 is
/// a sync request. On the control port the primary message is
/// `SimulatorCommand` (`{1: SimulatorControl, 2: SimulatorConfig}`), which
/// overlaps on field 2, so only a fixed32 field 1 or a field 3 are decisive and
/// the caller falls back to trying both decoders.
/// Hex dump of the first `n` bytes (for debug logging).
fn hex_prefix(bytes: &[u8], n: usize) -> String {
    let mut s = String::with_capacity(n * 2 + 3);
    for b in bytes.iter().take(n) {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    if bytes.len() > n {
        s.push_str("...");
    }
    s
}

pub fn looks_like_sync(bytes: &[u8], primary_is_robot_control: bool) -> bool {
    let mut decisive = false;
    for (field, wire) in TagScan::new(bytes) {
        match (field, wire) {
            // sim_step is a float (fixed32); no other message has that here.
            (1, 5) => return true,
            (3, _) => return true,
            (2, _) if primary_is_robot_control => decisive = true,
            _ => {}
        }
    }
    decisive
}

/// Iterator over the `(field_number, wire_type)` pairs of a protobuf message's
/// top-level fields. Stops at the first malformed byte.
struct TagScan<'a> {
    buf: &'a [u8],
}

impl<'a> TagScan<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }

    fn varint(&mut self) -> Option<u64> {
        let mut value = 0u64;
        for i in 0..10 {
            let byte = *self.buf.first()?;
            self.buf = &self.buf[1..];
            value |= u64::from(byte & 0x7f) << (7 * i);
            if byte & 0x80 == 0 {
                return Some(value);
            }
        }
        None
    }

    fn skip(&mut self, wire: u8) -> Option<()> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                if self.buf.len() < 8 {
                    return None;
                }
                self.buf = &self.buf[8..];
            }
            2 => {
                let len = self.varint()? as usize;
                if self.buf.len() < len {
                    return None;
                }
                self.buf = &self.buf[len..];
            }
            5 => {
                if self.buf.len() < 4 {
                    return None;
                }
                self.buf = &self.buf[4..];
            }
            _ => return None,
        }
        Some(())
    }
}

impl Iterator for TagScan<'_> {
    type Item = (u32, u8);

    fn next(&mut self) -> Option<Self::Item> {
        if self.buf.is_empty() {
            return None;
        }
        let tag = self.varint()?;
        let field = u32::try_from(tag >> 3).ok()?;
        let wire = (tag & 0x7) as u8;
        if field == 0 {
            return None;
        }
        self.skip(wire)?;
        Some((field, wire))
    }
}

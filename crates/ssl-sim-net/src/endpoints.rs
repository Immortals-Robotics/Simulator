//! UDP endpoints: one blocking receiver thread per socket, all forwarding
//! `(bytes, from, kind)` into a single crossbeam channel consumed by the sim
//! thread. Replies always go to the exact `from` address of the datagram that
//! caused them, never to a "last sender".

use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::Sender;

/// Default control port (`SimulatorCommand` / `SimulationSyncRequest`).
pub const CONTROL_PORT: u16 = 10300;
/// Default blue `RobotControl` port.
pub const BLUE_PORT: u16 = 10301;
/// Default yellow `RobotControl` port.
pub const YELLOW_PORT: u16 = 10302;

/// Maximum UDP payload we are willing to receive.
const RECV_BUF: usize = 65_536;

/// Poll interval for the shutdown flag; also the socket read timeout.
const POLL: Duration = Duration::from_millis(200);

/// Which socket a datagram arrived on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EndpointKind {
    /// Simulation control port.
    Control,
    /// Blue team robot control port.
    Blue,
    /// Yellow team robot control port.
    Yellow,
    /// Legacy grSim packet port.
    Legacy,
}

impl EndpointKind {
    /// Human readable name used in logs.
    pub fn name(self) -> &'static str {
        match self {
            EndpointKind::Control => "control",
            EndpointKind::Blue => "blue",
            EndpointKind::Yellow => "yellow",
            EndpointKind::Legacy => "legacy",
        }
    }
}

/// One received datagram.
#[derive(Debug, Clone)]
pub struct Datagram {
    /// Raw payload.
    pub bytes: Vec<u8>,
    /// Exact sender address; replies go here.
    pub from: SocketAddr,
    /// Socket the datagram arrived on.
    pub kind: EndpointKind,
}

/// Ports to bind. `legacy` is `None` when the grSim adapter is disabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndpointPorts {
    /// Control port.
    pub control: u16,
    /// Blue robot control port.
    pub blue: u16,
    /// Yellow robot control port.
    pub yellow: u16,
    /// Legacy grSim port, or `None` to disable it.
    pub legacy: Option<u16>,
    /// Bind to `127.0.0.1` instead of `0.0.0.0`.
    pub localhost: bool,
}

impl Default for EndpointPorts {
    fn default() -> Self {
        Self {
            control: CONTROL_PORT,
            blue: BLUE_PORT,
            yellow: YELLOW_PORT,
            legacy: Some(crate::legacy::LEGACY_COMMAND_PORT),
            localhost: false,
        }
    }
}

/// The bound sockets and their receiver threads.
#[derive(Debug)]
pub struct Endpoints {
    sockets: Vec<(EndpointKind, Arc<UdpSocket>)>,
    running: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl Endpoints {
    /// Bind every configured port and spawn one receiver thread each.
    pub fn bind(ports: EndpointPorts, tx: Sender<Datagram>) -> std::io::Result<Self> {
        let host = if ports.localhost {
            "127.0.0.1"
        } else {
            "0.0.0.0"
        };
        let running = Arc::new(AtomicBool::new(true));
        let mut sockets = Vec::new();
        let mut threads = Vec::new();

        let mut wanted = vec![
            (EndpointKind::Control, ports.control),
            (EndpointKind::Blue, ports.blue),
            (EndpointKind::Yellow, ports.yellow),
        ];
        if let Some(p) = ports.legacy {
            wanted.push((EndpointKind::Legacy, p));
        }

        for (kind, port) in wanted {
            let socket = Arc::new(UdpSocket::bind((host, port))?);
            socket.set_read_timeout(Some(POLL))?;
            sockets.push((kind, Arc::clone(&socket)));
            let tx = tx.clone();
            let running = Arc::clone(&running);
            let handle = thread::Builder::new()
                .name(format!("ssl-sim-{}", kind.name()))
                .spawn(move || receive_loop(kind, socket, tx, running))?;
            threads.push(handle);
        }

        Ok(Self {
            sockets,
            running,
            threads,
        })
    }

    /// The local address a socket is bound to (useful when port 0 was asked for).
    pub fn local_addr(&self, kind: EndpointKind) -> Option<SocketAddr> {
        self.socket(kind).and_then(|s| s.local_addr().ok())
    }

    fn socket(&self, kind: EndpointKind) -> Option<&Arc<UdpSocket>> {
        self.sockets
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, s)| s)
    }

    /// Send `bytes` from the socket of `kind` to the exact address `to`.
    pub fn reply(&self, kind: EndpointKind, to: SocketAddr, bytes: &[u8]) -> std::io::Result<()> {
        match self.socket(kind) {
            Some(s) => s.send_to(bytes, to).map(|_| ()),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no {} socket bound", kind.name()),
            )),
        }
    }

    /// Stop the receiver threads and join them.
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for Endpoints {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn receive_loop(
    kind: EndpointKind,
    socket: Arc<UdpSocket>,
    tx: Sender<Datagram>,
    running: Arc<AtomicBool>,
) {
    let mut buf = vec![0u8; RECV_BUF];
    while running.load(Ordering::Relaxed) {
        match socket.recv_from(&mut buf) {
            Ok((n, from)) => {
                let datagram = Datagram {
                    bytes: buf[..n].to_vec(),
                    from,
                    kind,
                };
                if tx.send(datagram).is_err() {
                    return;
                }
            }
            Err(e) => match e.kind() {
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => continue,
                // Windows reports ICMP port-unreachable from a previous send as
                // ConnectionReset on the next recv; it is not fatal.
                std::io::ErrorKind::ConnectionReset => continue,
                _ => {
                    tracing::warn!(endpoint = kind.name(), error = %e, "recv failed");
                    return;
                }
            },
        }
    }
}

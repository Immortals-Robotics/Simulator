//! Blocking UDP receivers (one thread per stream) and the control sender.
//!
//! No async runtime: each stream owns a thread that blocks in `recv_from`,
//! decodes with prost and folds the result into the shared [`State`].

use std::{
    io,
    net::{Ipv4Addr, SocketAddr, UdpSocket},
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use anyhow::{Context, Result};
use prost::Message as _;
use socket2::{Domain, Protocol, Socket, Type};
use ssl_sim_proto::{sim::SslWrapperPacket, tracked::TrackerWrapperPacket};

use crate::state::State;

/// Largest SSL vision datagram we expect; wrapper packets are far smaller.
const MAX_DATAGRAM: usize = 65_536;

/// Bind a receiving socket for `addr`.
///
/// For a multicast address the socket binds `0.0.0.0:port` with
/// `SO_REUSEADDR` so it coexists with the team software already listening on
/// the same group, then joins the group on the default (`0.0.0.0`) interface.
pub fn bind_receiver(addr: SocketAddr) -> Result<UdpSocket> {
    let SocketAddr::V4(v4) = addr else {
        anyhow::bail!("only IPv4 vision/tracker addresses are supported, got {addr}");
    };
    let socket =
        Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP)).context("create UDP socket")?;
    socket.set_reuse_address(true).context("SO_REUSEADDR")?;
    let bind_ip = if v4.ip().is_multicast() {
        Ipv4Addr::UNSPECIFIED
    } else {
        *v4.ip()
    };
    socket
        .bind(&SocketAddr::from((bind_ip, v4.port())).into())
        .with_context(|| format!("bind {bind_ip}:{}", v4.port()))?;
    if v4.ip().is_multicast() {
        socket
            .join_multicast_v4(v4.ip(), &Ipv4Addr::UNSPECIFIED)
            .with_context(|| format!("join multicast group {}", v4.ip()))?;
        socket.set_multicast_loop_v4(true).ok();
    }
    Ok(socket.into())
}

/// Spawn the vision receiver thread.
///
/// `wake` is called after every accepted packet so the UI repaints promptly
/// even when it is otherwise idle.
pub fn spawn_vision(
    addr: SocketAddr,
    state: Arc<Mutex<State>>,
    wake: impl Fn() + Send + 'static,
) -> Result<()> {
    let socket = bind_receiver(addr).with_context(|| format!("vision stream on {addr}"))?;
    spawn_loop("vision", socket, move |bytes, now| {
        let mut state = state.lock().expect("state mutex poisoned");
        match SslWrapperPacket::decode(bytes) {
            Ok(packet) => state.apply_wrapper(packet, now),
            Err(err) => {
                state.vision_decode_errors += 1;
                state.last_error = Some(format!("vision decode: {err}"));
            }
        }
        drop(state);
        wake();
    });
    Ok(())
}

/// Spawn the ground-truth (tracker) receiver thread.
pub fn spawn_truth(
    addr: SocketAddr,
    state: Arc<Mutex<State>>,
    wake: impl Fn() + Send + 'static,
) -> Result<()> {
    let socket = bind_receiver(addr).with_context(|| format!("truth stream on {addr}"))?;
    spawn_loop("truth", socket, move |bytes, now| {
        let mut state = state.lock().expect("state mutex poisoned");
        match TrackerWrapperPacket::decode(bytes) {
            Ok(packet) => state.apply_tracker(packet, now),
            Err(err) => {
                state.truth_decode_errors += 1;
                state.last_error = Some(format!("truth decode: {err}"));
            }
        }
        drop(state);
        wake();
    });
    Ok(())
}

fn spawn_loop(
    name: &'static str,
    socket: UdpSocket,
    mut on_packet: impl FnMut(&[u8], Instant) + Send + 'static,
) {
    thread::Builder::new()
        .name(format!("ssl-sim-viewer-{name}"))
        .spawn(move || {
            let mut buf = vec![0u8; MAX_DATAGRAM];
            loop {
                match socket.recv_from(&mut buf) {
                    Ok((len, _from)) => on_packet(&buf[..len], Instant::now()),
                    Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                    Err(err) => {
                        eprintln!("ssl-sim-viewer: {name} receive failed: {err}");
                        return;
                    }
                }
            }
        })
        .expect("spawn receiver thread");
}

/// Fire-and-forget sender for `SimulatorCommand`s on the control port.
#[derive(Debug)]
pub struct ControlSender {
    socket: UdpSocket,
    target: SocketAddr,
    /// Last error from `send_to`, surfaced in the status panel.
    pub last_error: Option<String>,
    /// Number of commands sent.
    pub sent: u64,
}

impl ControlSender {
    /// Bind an ephemeral local port for talking to `target`.
    pub fn new(target: SocketAddr) -> Result<Self> {
        let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).context("bind control socket")?;
        Ok(Self {
            socket,
            target,
            last_error: None,
            sent: 0,
        })
    }

    /// The address commands are sent to.
    pub fn target(&self) -> SocketAddr {
        self.target
    }

    /// Encode and send one command. Errors are recorded, not propagated: a
    /// simulator that is not listening yet must not take the viewer down.
    pub fn send(&mut self, command: &ssl_sim_proto::sim::SimulatorCommand) {
        let bytes = command.encode_to_vec();
        match self.socket.send_to(&bytes, self.target) {
            Ok(_) => {
                self.sent += 1;
                self.last_error = None;
            }
            Err(err) => self.last_error = Some(format!("control send: {err}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ssl_sim_proto::sim::{SslDetectionFrame, SslWrapperPacket};

    /// Opt-in smoke test (`cargo test -p ssl-sim-viewer -- --ignored`): joins
    /// the real vision group and checks a locally sent packet comes back.
    /// Ignored by default because it needs a multicast-capable interface.
    #[test]
    #[ignore = "requires multicast networking"]
    fn receives_a_multicast_wrapper_packet() {
        let group: SocketAddr = "224.5.23.2:10020".parse().unwrap();
        let state = Arc::new(Mutex::new(State::default()));
        spawn_vision(group, Arc::clone(&state), || {}).expect("join vision group");

        let sender = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).unwrap();
        sender.set_multicast_loop_v4(true).unwrap();
        let packet = SslWrapperPacket {
            detection: Some(SslDetectionFrame {
                frame_number: 7,
                t_capture: 0.0,
                t_sent: 0.0,
                camera_id: 1,
                balls: vec![],
                robots_blue: vec![],
                robots_yellow: vec![],
            }),
            geometry: None,
            source: None,
        };

        for _ in 0..50 {
            sender.send_to(&packet.encode_to_vec(), group).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(20));
            if state.lock().unwrap().cameras.contains_key(&1) {
                return;
            }
        }
        panic!("no multicast packet arrived within 1 s");
    }
}

//! Vision publisher: SSL wrapper packets on the vision multicast group and an
//! optional ground-truth `TrackerWrapperPacket` stream.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};

use prost::Message as _;
use ssl_sim_core::vision::VisionOutput;
use ssl_sim_core::world::WorldSnapshot;
use ssl_sim_proto::sim;

use crate::convert;

/// Default SSL vision multicast group.
pub const VISION_GROUP: Ipv4Addr = Ipv4Addr::new(224, 5, 23, 2);
/// Default vision port.
pub const VISION_PORT: u16 = 10020;
/// Default ground-truth (tracked) port.
pub const TRUTH_PORT: u16 = 10010;
/// Multicast TTL: 1 = stay on the local link, like grSim and ER-Force.
pub const MULTICAST_TTL: u32 = 1;

/// The default vision address, `224.5.23.2:10020`.
pub fn default_vision_addr() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(VISION_GROUP), VISION_PORT)
}

/// The default ground-truth address, `224.5.23.2:10010`.
pub fn default_truth_addr() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(VISION_GROUP), TRUTH_PORT)
}

/// Publishes vision (and optionally ground-truth) packets.
#[derive(Debug)]
pub struct VisionPublisher {
    socket: UdpSocket,
    addr: SocketAddr,
    truth_addr: Option<SocketAddr>,
    truth_frame: u32,
    buf: Vec<u8>,
}

impl VisionPublisher {
    /// Open the publisher socket.
    ///
    /// `addr` is the vision destination; when it is a multicast group the TTL
    /// is set to [`MULTICAST_TTL`] and loopback delivery is enabled so tools on
    /// the same machine see the traffic. `truth` enables the ground-truth
    /// stream on the same host at [`TRUTH_PORT`].
    pub fn new(addr: SocketAddr, truth: bool) -> std::io::Result<Self> {
        let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))?;
        if let IpAddr::V4(ip) = addr.ip() {
            if ip.is_multicast() {
                socket.set_multicast_ttl_v4(MULTICAST_TTL)?;
                socket.set_multicast_loop_v4(true)?;
            }
        }
        let truth_addr = truth.then(|| SocketAddr::new(addr.ip(), TRUTH_PORT));
        Ok(Self {
            socket,
            addr,
            truth_addr,
            truth_frame: 0,
            buf: Vec::with_capacity(8192),
        })
    }

    /// Current vision destination.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Change the vision port (`SimulatorConfig.vision_port`).
    pub fn set_port(&mut self, port: u16) {
        self.addr.set_port(port);
        tracing::info!(addr = %self.addr, "vision port changed");
    }

    /// Whether the ground-truth stream is enabled.
    pub fn truth_enabled(&self) -> bool {
        self.truth_addr.is_some()
    }

    /// Send one wrapper packet.
    pub fn publish_packet(&mut self, packet: &sim::SslWrapperPacket) -> std::io::Result<()> {
        self.buf.clear();
        packet
            .encode(&mut self.buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.socket.send_to(&self.buf, self.addr).map(|_| ())
    }

    /// Send every wrapper packet of one vision output (one per camera).
    pub fn publish(&mut self, out: &VisionOutput) -> std::io::Result<()> {
        for packet in convert::vision_output_to_packets(out) {
            self.publish_packet(&packet)?;
        }
        Ok(())
    }

    /// Send a ground-truth frame if the truth stream is enabled.
    pub fn publish_truth(&mut self, snapshot: &WorldSnapshot) -> std::io::Result<()> {
        let Some(addr) = self.truth_addr else {
            return Ok(());
        };
        let packet = convert::snapshot_to_tracker(snapshot, self.truth_frame);
        self.truth_frame = self.truth_frame.wrapping_add(1);
        self.buf.clear();
        packet
            .encode(&mut self.buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.socket.send_to(&self.buf, addr).map(|_| ())
    }
}

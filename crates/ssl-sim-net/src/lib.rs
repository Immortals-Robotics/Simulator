//! Network layer: translates between the wire protocols and `ssl-sim-core`,
//! and owns the UDP endpoints.
//!
//! ```text
//! control 10300  SimulatorCommand      -> SimulatorResponse
//!                SimulationSyncRequest -> SimulationSyncResponse
//! blue    10301  RobotControl          -> RobotControlResponse
//! yellow  10302  (and sync requests carrying robot_control for that team)
//! legacy  20011  grSim_Packet          -> Robots_Status to 30011 / 30012
//! vision         224.5.23.2:10020 (SSL_WrapperPacket), optional truth on :10010
//! ```
//!
//! Modules:
//!
//! * [`convert`] — proto ↔ core, all unit conversions (mm, ns, clockwise wheel
//!   angles) live here and nowhere else.
//! * [`legacy`] — grSim packet adapter and `Robots_Status` tracking.
//! * [`endpoints`] — UDP sockets, one receiver thread each, replies to the
//!   exact sender address.
//! * [`control`] — `SimulatorCommand` application.
//! * [`robot_control`] — `RobotControl` application, feedback and packet loss.
//! * [`sync`] — lock-step `SimulationSyncRequest` handling.
//! * [`vision`] — multicast vision and ground-truth publisher.
//! * [`runner`] — the sim thread that owns the [`ssl_sim_core::World`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod control;
pub mod convert;
pub mod endpoints;
pub mod legacy;
pub mod robot_control;
pub mod runner;
pub mod sync;
pub mod vision;

pub use endpoints::{Datagram, EndpointKind, EndpointPorts, Endpoints};
pub use runner::{Mode, RunOptions, Runner};
pub use vision::{default_truth_addr, default_vision_addr, VisionPublisher};

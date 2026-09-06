//! Deterministic RoboCup Small Size League simulation core.
//!
//! The core has no I/O, no threads and no protobuf. It owns a [`World`] that is
//! advanced in fixed substeps of [`SimConfig::substep`] and produces
//! [`vision::DetectionFrame`]s and ground-truth [`WorldSnapshot`]s. All units
//! are SI (m, s, rad, kg); wire conversions live in `ssl-sim-net`.
//!
//! Module ownership during the initial build-out (see `docs/design.md` §9):
//! - `types`, `params`, `robot` (state struct), `world`: lead
//! - `ball`, `collision`, `physics`, `robot::drive`, `robot::dribbler`: math agent
//! - `field`, `vision`, `rng`, `robot::kicker`, presets/formations: general agent

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod ball;
pub mod collision;
pub mod field;
pub mod params;
pub mod physics;
pub mod rng;
pub mod robot;
pub mod types;
pub mod vision;
pub mod world;

pub use ball::Ball;
pub use field::FieldGeometry;
pub use params::{BallParams, Realism, RobotSpecs, SimConfig, VisionConfig};
pub use robot::Robot;
pub use types::*;
pub use world::{World, WorldSnapshot};

/// Gravitational acceleration [m/s²].
pub const GRAVITY: f64 = 9.81;

//! Generated protobuf bindings.
//!
//! - [`sim`]: RoboCup SSL simulation protocol, SSL vision wrapper/detection/
//!   geometry, and the ER-Force custom realism / robot-spec messages.
//! - [`grsim`]: legacy grSim command / replacement / status packets.
//! - [`tracked`]: ssl-vision tracked-frame packets (used for ground truth).

#[allow(clippy::all, missing_docs)]
pub mod sim {
    include!(concat!(env!("OUT_DIR"), "/sim.rs"));
}

#[allow(clippy::all, missing_docs)]
pub mod grsim {
    include!(concat!(env!("OUT_DIR"), "/grsim.rs"));
}

#[allow(clippy::all, missing_docs)]
pub mod tracked {
    include!(concat!(env!("OUT_DIR"), "/tracked.rs"));
}

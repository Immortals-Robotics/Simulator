//! Generates Rust bindings for every protobuf family the simulator speaks.
//!
//! None of the upstream `.proto` files declare a `package`, so families whose
//! message names collide (`RobotId` exists in both `ssl_gc_common.proto` and
//! `messages_robocup_ssl_detection_tracked.proto`) are compiled in separate
//! invocations into separate modules.
use std::{env, path::PathBuf};

use anyhow::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-env-changed=SSL_SIMULATION_PROTO_DIR");

    let protocol_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("protocol");
    let sim_dir = env::var_os("SSL_SIMULATION_PROTO_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| protocol_dir.join("ssl-simulation-protocol/proto"));
    let erforce_dir = protocol_dir.join("erforce");
    let grsim_dir = protocol_dir.join("grsim");
    let vision_dir = protocol_dir.join("ssl-vision");

    // Family 1: ssl-simulation-protocol + ssl-vision wrapper + ER-Force custom messages.
    let sim_files = [
        sim_dir.join("ssl_gc_common.proto"),
        sim_dir.join("ssl_vision_geometry.proto"),
        sim_dir.join("ssl_vision_detection.proto"),
        sim_dir.join("ssl_vision_wrapper.proto"),
        sim_dir.join("ssl_simulation_error.proto"),
        sim_dir.join("ssl_simulation_robot_control.proto"),
        sim_dir.join("ssl_simulation_robot_feedback.proto"),
        sim_dir.join("ssl_simulation_config.proto"),
        sim_dir.join("ssl_simulation_control.proto"),
        sim_dir.join("ssl_simulation_synchronous.proto"),
        erforce_dir.join("ssl_simulation_custom_erforce_realism.proto"),
        erforce_dir.join("ssl_simulation_custom_erforce_robot_spec.proto"),
    ];
    compile("sim", &sim_files, &[sim_dir.clone(), erforce_dir])?;

    // Family 2: legacy grSim packets.
    let grsim_files = [
        grsim_dir.join("grSim_Commands.proto"),
        grsim_dir.join("grSim_Replacement.proto"),
        grsim_dir.join("grSim_Packet.proto"),
        grsim_dir.join("grSim_Robotstatus.proto"),
    ];
    compile("grsim", &grsim_files, &[grsim_dir])?;

    // Family 3: ssl-vision tracked (ground truth) packets.
    let tracked_files = [
        vision_dir.join("messages_robocup_ssl_detection_tracked.proto"),
        vision_dir.join("messages_robocup_ssl_wrapper_tracked.proto"),
    ];
    compile("tracked", &tracked_files, &[vision_dir])?;

    // Family 4: game-controller referee messages (for reading game logs).
    let gc_dir = protocol_dir.join("ssl-game-controller");
    let gc_files = [
        gc_dir.join("state/ssl_gc_common.proto"),
        gc_dir.join("geom/ssl_gc_geometry.proto"),
        gc_dir.join("state/ssl_gc_game_event.proto"),
        gc_dir.join("state/ssl_gc_referee_message.proto"),
    ];
    compile("gc", &gc_files, &[gc_dir])?;

    Ok(())
}

fn compile(module: &str, files: &[PathBuf], includes: &[PathBuf]) -> Result<()> {
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    let mut config = prost_build::Config::new();
    config.default_package_filename(module);
    config.compile_protos(files, includes)?;
    Ok(())
}

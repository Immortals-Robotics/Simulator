use std::{env, path::PathBuf};

use anyhow::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-env-changed=SSL_SIMULATION_PROTO_DIR");

    let workspace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let proto_dir = env::var_os("SSL_SIMULATION_PROTO_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_dir.join("protocol/ssl-simulation-protocol/proto"));

    let files = [
        "ssl_gc_common.proto",
        "ssl_vision_geometry.proto",
        "ssl_vision_detection.proto",
        "ssl_simulation_error.proto",
        "ssl_simulation_robot_control.proto",
        "ssl_simulation_robot_feedback.proto",
        "ssl_simulation_config.proto",
        "ssl_simulation_control.proto",
        "ssl_simulation_synchronous.proto",
    ]
    .into_iter()
    .map(|file| proto_dir.join(file))
    .collect::<Vec<_>>();

    for file in &files {
        println!("cargo:rerun-if-changed={}", file.display());
    }

    let mut config = prost_build::Config::new();
    config.default_package_filename("sim");
    config.compile_protos(&files, &[proto_dir])?;

    Ok(())
}

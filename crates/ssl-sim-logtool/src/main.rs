//! `ssl-logtool`: analyse recorded SSL game logs to calibrate the simulator.
//!
//! Subcommands are added per analysis; `inventory` is the smoke test.

#![forbid(unsafe_code)]

mod dynamics;
mod reader;
mod vision;

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use reader::{LogReader, Record};

#[derive(Debug, Parser)]
#[command(
    name = "ssl-logtool",
    about = "SSL game log analysis for simulator calibration"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print record counts, duration, cameras and frame rates.
    Inventory {
        /// Log files (`.log` or `.log.gz`).
        files: Vec<PathBuf>,
        /// Stop after this many records (0 = whole file).
        #[arg(long, default_value_t = 0)]
        limit: u64,
    },
    /// Vision imperfection statistics (noise, dropouts, latency, cameras, spurious balls).
    Vision(vision::Args),
    /// Ball and robot dynamics fitting (rolling, chips, kicks, collisions, robot accelerations).
    Dynamics(dynamics::Args),
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Inventory { files, limit } => {
            for f in files {
                inventory(&f, limit)?;
            }
        }
        Command::Vision(args) => vision::run(args)?,
        Command::Dynamics(args) => dynamics::run(args)?,
    }
    Ok(())
}

fn inventory(path: &std::path::Path, limit: u64) -> Result<()> {
    let mut reader = LogReader::open(path)?;
    let mut kinds: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut per_camera: BTreeMap<u32, (u64, f64, f64)> = BTreeMap::new(); // frames, first, last t_capture
    let mut geometry = 0u64;
    let mut first_ns = None;
    let mut last_ns = 0i64;
    let mut ref_commands: BTreeMap<i32, u64> = BTreeMap::new();
    while let Some(entry) = reader.next_entry()? {
        first_ns.get_or_insert(entry.time_ns);
        last_ns = entry.time_ns;
        let name = match &entry.record {
            Record::Vision(p) => {
                if let Some(d) = &p.detection {
                    let e = per_camera
                        .entry(d.camera_id)
                        .or_insert((0, d.t_capture, d.t_capture));
                    e.0 += 1;
                    e.2 = d.t_capture;
                }
                if p.geometry.is_some() {
                    geometry += 1;
                }
                "vision"
            }
            Record::Referee(r) => {
                *ref_commands.entry(r.command).or_default() += 1;
                "referee"
            }
            Record::Tracker(_) => "tracker",
            Record::Other { .. } => "other",
        };
        *kinds.entry(name).or_default() += 1;
        if limit > 0 && reader.count >= limit {
            break;
        }
    }
    let span = (last_ns - first_ns.unwrap_or(last_ns)) as f64 * 1e-9;
    println!("{}", path.display());
    println!(
        "  records {} span {:.1} s decode_errors {} kinds {:?}",
        reader.count, span, reader.decode_errors, kinds
    );
    println!("  geometry packets {}", geometry);
    for (cam, (n, t0, t1)) in &per_camera {
        let dt = t1 - t0;
        let hz = if dt > 0.0 {
            (*n as f64 - 1.0) / dt
        } else {
            0.0
        };
        println!("  camera {cam}: {n} frames, {hz:.2} Hz over {dt:.1} s");
    }
    println!("  referee commands {:?}", ref_commands);
    Ok(())
}

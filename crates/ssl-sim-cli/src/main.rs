//! `ssl-sim` command line: config file + flags, run loop, logging.
//!
//! See `docs/design.md` §7 and the repository README.

use std::net::{SocketAddr, ToSocketAddrs};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};

use ssl_sim_core::field::Division;
use ssl_sim_core::params::{Realism, SimConfig};
use ssl_sim_net::endpoints::{EndpointPorts, BLUE_PORT, CONTROL_PORT, YELLOW_PORT};
use ssl_sim_net::legacy::LEGACY_COMMAND_PORT;
use ssl_sim_net::runner::{Mode, RunOptions, Runner};

/// Headless RoboCup SSL simulator speaking the standard simulation protocol.
#[derive(Debug, Parser)]
#[command(name = "ssl-sim", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the simulator.
    Run(RunArgs),
    /// Print the built-in defaults (field, robot specs, realism) as TOML.
    Presets(PresetArgs),
}

#[derive(Debug, Args)]
struct RunArgs {
    /// TOML file with a `SimConfig`; missing keys fall back to the defaults.
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Field size preset.
    #[arg(long, value_name = "a|b")]
    division: Option<String>,

    /// Robots per team at start (0..=16).
    #[arg(long, value_name = "N")]
    robots: Option<u8>,

    /// Stepping mode.
    #[arg(long, value_name = "realtime|fast|sync")]
    mode: Option<Mode>,

    /// Real-time scaling; 0 pauses the simulation.
    #[arg(long, value_name = "X")]
    speed: Option<f64>,

    /// Physics substep in milliseconds.
    #[arg(long = "step-ms", value_name = "MS")]
    step_ms: Option<f64>,

    /// Realism preset name or a TOML file with a `Realism` table.
    #[arg(long, value_name = "none|friendly|realistic|rc2021|FILE")]
    realism: Option<String>,

    /// RNG seed, or `random` for a wall-clock derived seed.
    #[arg(long, value_name = "N|random")]
    seed: Option<String>,

    /// Vision multicast destination.
    #[arg(long = "vision-addr", value_name = "HOST:PORT")]
    vision_addr: Option<String>,

    /// Publish vision to 127.0.0.1 instead of the multicast group.
    #[arg(long)]
    localhost: bool,

    /// Also publish the ground-truth tracker stream on port 10010.
    #[arg(long)]
    truth: bool,

    /// Do not listen for legacy grSim packets on 20011.
    #[arg(long = "no-legacy-grsim")]
    no_legacy_grsim: bool,

    /// Stop after this many seconds (default: run forever).
    #[arg(long, value_name = "SECS")]
    duration: Option<f64>,

    /// Log level (`error`, `warn`, `info`, `debug`, `trace`, or an
    /// `RUST_LOG`-style filter).
    #[arg(long = "log-level", default_value = "info")]
    log_level: String,
}

#[derive(Debug, Args)]
struct PresetArgs {
    /// Log level for the presets command.
    #[arg(long = "log-level", default_value = "warn")]
    log_level: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(args),
        Command::Presets(args) => presets(args),
    }
}

fn init_logging(filter: &str) {
    use tracing_subscriber::EnvFilter;
    let env = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(filter))
        .unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env)
        .with_target(false)
        .try_init();
}

fn run(args: RunArgs) -> Result<()> {
    init_logging(&args.log_level);

    let mut config = match &args.config {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading config {}", path.display()))?;
            toml::from_str::<SimConfig>(&text)
                .with_context(|| format!("parsing config {}", path.display()))?
        }
        None => SimConfig::default(),
    };

    // Flags override the file.
    let division = match args.division.as_deref() {
        None => Division::A,
        Some(d) => parse_division(d)?,
    };
    if let Some(n) = args.robots {
        if n > 16 {
            bail!("--robots must be 0..=16");
        }
        config.initial_robots_per_team = n;
    }
    if let Some(ms) = args.step_ms {
        if !(ms.is_finite() && ms > 0.0) {
            bail!("--step-ms must be a positive number of milliseconds");
        }
        config.substep = ms / 1000.0;
    }
    if let Some(spec) = &args.realism {
        config.realism = load_realism(spec)?;
    }
    if let Some(seed) = &args.seed {
        config.seed = parse_seed(seed)?;
    }

    let mode = args.mode.unwrap_or_default();
    let speed = args.speed.unwrap_or(1.0);
    if !(speed.is_finite() && speed >= 0.0) {
        bail!("--speed must be >= 0");
    }

    let mut vision_addr = match &args.vision_addr {
        Some(a) => parse_addr(a)?,
        None => ssl_sim_net::default_vision_addr(),
    };
    if args.localhost {
        vision_addr.set_ip(std::net::IpAddr::from([127, 0, 0, 1]));
    }

    let ports = EndpointPorts {
        control: CONTROL_PORT,
        blue: BLUE_PORT,
        yellow: YELLOW_PORT,
        legacy: (!args.no_legacy_grsim).then_some(LEGACY_COMMAND_PORT),
        localhost: false,
    };

    let options = RunOptions {
        mode,
        speed,
        division,
        ports,
        vision_addr,
        truth: args.truth,
        max_duration: args.duration.map(Duration::from_secs_f64),
    };

    tracing::info!(
        division = ?division,
        robots = config.initial_robots_per_team,
        substep_ms = config.substep * 1000.0,
        seed = config.seed,
        "starting ssl-sim"
    );

    Runner::run(config, options)
}

fn presets(args: PresetArgs) -> Result<()> {
    init_logging(&args.log_level);
    let config = SimConfig::default();
    let text = toml::to_string_pretty(&config).context("serialising the default SimConfig")?;
    println!("# ssl-sim default configuration (`ssl-sim presets`)");
    println!("# Save this to a file and pass it with `ssl-sim run --config <file>`.");
    println!("#");
    println!("# Realism presets available to `--realism`: none, friendly, realistic, rc2021.");
    println!("# Field presets available to `--division`: a (12x9 m), b (9x6 m).");
    println!();
    print!("{text}");
    Ok(())
}

fn parse_division(s: &str) -> Result<Division> {
    match s.to_ascii_lowercase().as_str() {
        "a" | "diva" | "div_a" => Ok(Division::A),
        "b" | "divb" | "div_b" => Ok(Division::B),
        other => bail!("unknown division `{other}` (expected `a` or `b`)"),
    }
}

fn parse_seed(s: &str) -> Result<u64> {
    if s.eq_ignore_ascii_case("random") {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        // Cheap avalanche so nearby launch times give unrelated seeds.
        let mut x = nanos ^ 0x9E37_79B9_7F4A_7C15;
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        return Ok(x);
    }
    s.parse::<u64>()
        .with_context(|| format!("`{s}` is not a seed (expected a number or `random`)"))
}

fn load_realism(spec: &str) -> Result<Realism> {
    if let Some(preset) = Realism::preset(spec) {
        return Ok(preset);
    }
    let path = PathBuf::from(spec);
    if !path.exists() {
        bail!(
            "`{spec}` is neither a realism preset (none|friendly|realistic|rc2021) \
             nor an existing file"
        );
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading realism file {}", path.display()))?;
    toml::from_str::<Realism>(&text)
        .with_context(|| format!("parsing realism file {}", path.display()))
}

fn parse_addr(s: &str) -> Result<SocketAddr> {
    if let Ok(addr) = s.parse::<SocketAddr>() {
        return Ok(addr);
    }
    s.to_socket_addrs()
        .with_context(|| format!("resolving `{s}`"))?
        .next()
        .with_context(|| format!("`{s}` did not resolve to an address"))
}

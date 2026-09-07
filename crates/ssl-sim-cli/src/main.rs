//! `ssl-sim` command line: config file + flags, run loop, logging.
//!
//! See `docs/design.md` §7 and the repository README.

use std::net::{SocketAddr, ToSocketAddrs};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};

use ssl_sim_core::field::Division;
use ssl_sim_core::params::{BallParams, Realism, RobotLimits, SimConfig};
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
    // Boxed: `RunArgs` dwarfs the other variants.
    Run(Box<RunArgs>),
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

    /// Realism preset name or a TOML file with a `Realism` table. `realistic`
    /// (the default) is the pooled 2026 measurement; `go26` / `rc26` are the
    /// two venues it pools; the `erforce_*` presets are ER-Force's own values.
    #[arg(
        long,
        value_name = "none|realistic|go26|rc26|erforce_friendly|erforce_realistic|erforce_rc2021|FILE"
    )]
    realism: Option<String>,

    /// Ball physics preset (`ssl-sim presets` lists them).
    #[arg(
        long = "ball-preset",
        value_name = "default|go26|rc26|erforce|tigers|grsim"
    )]
    ball_preset: Option<String>,

    /// Firmware movement limits applied to both teams' default robot specs.
    #[arg(
        long = "robot-limits",
        value_name = "default|tigers|erforce|kiks|fast|grsim"
    )]
    robot_limits: Option<String>,

    /// Cameras to auto-place (1, 2 or 4). Overrides any `[[vision.cameras]]`
    /// in the config file. Real division-A rigs used 2 (`docs/calibration/vision.md` §2).
    #[arg(long, value_name = "N")]
    cameras: Option<u32>,

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
        Command::Run(args) => run(*args),
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
    if let Some(name) = &args.ball_preset {
        config.ball = BallParams::preset(name)
            .with_context(|| format!("unknown --ball-preset `{name}` ({})", preset_names(BALL)))?;
    }
    if let Some(name) = &args.robot_limits {
        let limits = RobotLimits::preset(name).with_context(|| {
            format!("unknown --robot-limits `{name}` ({})", preset_names(LIMITS))
        })?;
        config.blue_specs.limits = limits;
        config.yellow_specs.limits = limits;
    }
    if let Some(n) = args.cameras {
        if ![1, 2, 4].contains(&n) {
            bail!("--cameras must be 1, 2 or 4");
        }
        // An explicit rig in the config file would win over the count, so drop
        // it and let the auto-placement run.
        config.vision.cameras.clear();
        config.vision.default_camera_count = n;
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

/// `(name, one-line description)` for every `--realism` preset.
const REALISM: &[(&str, &str)] = &[
    ("none", "no noise, no loss, no delay beyond the fixed 35 ms"),
    (
        "realistic",
        "DEFAULT: measured from ten 2026 games (vision.md §8)",
    ),
    (
        "go26",
        "German Open 2026 venue: no `area`, many dribbler-LED false balls",
    ),
    (
        "rc26",
        "RoboCup 2026 venue: `area` reported, 2-2.6 cm constant camera offset",
    ),
    ("erforce_friendly", "ER-Force \"Friendly\" (alias friendly)"),
    ("erforce_realistic", "ER-Force \"Realistic\""),
    (
        "erforce_rc2021",
        "ER-Force \"RC2021\", glued dribbler (alias rc2021)",
    ),
];

/// `(name, one-line description)` for every `--ball-preset`.
const BALL: &[(&str, &str)] = &[
    (
        "default",
        "pooled 2026 fit, speed-dependent rolling (alias measured)",
    ),
    ("go26", "German Open 2026 carpet"),
    ("rc26", "RoboCup 2026 carpet"),
    ("erforce", "the constants ER-Force's simulator advertises"),
    ("tigers", "the constants TIGERs advertise"),
    ("grsim", "grSim's ball"),
];

/// `(name, one-line description)` for every `--robot-limits`.
const LIMITS: &[(&str, &str)] = &[
    ("default", "fitted from 2026 game logs (alias measured)"),
    ("tigers", "TIGERs Mannheim envelope"),
    ("erforce", "ER-Force envelope"),
    ("kiks", "KIKS envelope"),
    ("fast", "upper envelope seen in the logs"),
    ("grsim", "grSim's unconstrained defaults"),
];

fn preset_names(list: &[(&str, &str)]) -> String {
    let names: Vec<&str> = list.iter().map(|(n, _)| *n).collect();
    format!("known: {}", names.join(", "))
}

fn print_preset_table(title: &str, flag: &str, list: &[(&str, &str)]) {
    println!("# {title} (`{flag}`):");
    let width = list.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
    for (name, description) in list {
        println!("#   {name:<width$}  {description}");
    }
    println!("#");
}

fn presets(args: PresetArgs) -> Result<()> {
    init_logging(&args.log_level);
    let config = SimConfig::default();
    let text = toml::to_string_pretty(&config).context("serialising the default SimConfig")?;
    println!("# ssl-sim default configuration (`ssl-sim presets`)");
    println!("# Save this to a file and pass it with `ssl-sim run --config <file>`.");
    println!("#");
    print_preset_table("Realism presets", "--realism", REALISM);
    print_preset_table("Ball presets", "--ball-preset", BALL);
    print_preset_table("Robot limit presets", "--robot-limits", LIMITS);
    println!("# Field presets (`--division`):");
    println!("#   a  12 x 9 m, division A");
    println!("#   b  9 x 6 m, division B");
    println!("#");
    println!("# Camera rigs (`--cameras`): 1 (centre), 2 (measured default), 4 (quadrants).");
    println!("# Every preset is also expressible in this file; the flags only swap");
    println!("# whole tables. Measurements and their evidence: docs/calibration/.");
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
            "`{spec}` is neither a realism preset ({}) nor an existing file",
            preset_names(REALISM)
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory as _;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    /// Every name the `presets` command advertises must actually resolve, and
    /// every preset the core knows must be advertised.
    #[test]
    fn advertised_presets_all_resolve() {
        for (name, description) in REALISM {
            assert!(Realism::preset(name).is_some(), "realism preset {name}");
            assert!(!description.is_empty());
        }
        for (name, _) in BALL {
            assert!(BallParams::preset(name).is_some(), "ball preset {name}");
        }
        for (name, _) in LIMITS {
            assert!(RobotLimits::preset(name).is_some(), "limits preset {name}");
        }
        // Aliases resolve too but are not listed separately.
        for alias in ["friendly", "rc2021", "measured", "default"] {
            assert!(Realism::preset(alias).is_some(), "alias {alias}");
        }
        assert!(Realism::preset("nope").is_none());
        assert!(BallParams::preset("nope").is_none());
        assert!(RobotLimits::preset("nope").is_none());
    }

    #[test]
    fn realistic_is_the_default_realism() {
        assert_eq!(SimConfig::default().realism, Realism::realistic());
        assert_eq!(load_realism("realistic").unwrap(), Realism::realistic());
        assert!(load_realism("no-such-preset-or-file").is_err());
    }

    /// The measured 2026 rig: two cameras, 73.3 Hz.
    #[test]
    fn default_vision_config_is_the_measured_rig() {
        let v = SimConfig::default().vision;
        assert_eq!(v.default_camera_count, 2);
        assert!((v.frame_rate - 73.3).abs() < 1e-9);
        assert!(v.cameras.is_empty(), "auto-placed by default");
    }

    #[test]
    fn camera_flag_overrides_an_explicit_rig() {
        let cli = Cli::try_parse_from(["ssl-sim", "run", "--cameras", "4"]).expect("parse");
        let Command::Run(args) = cli.command else {
            panic!("expected run")
        };
        assert_eq!(args.cameras, Some(4));
        assert!(Cli::try_parse_from(["ssl-sim", "run", "--cameras", "x"]).is_err());
    }

    #[test]
    fn division_and_seed_parsing() {
        assert!(matches!(parse_division("A").unwrap(), Division::A));
        assert!(matches!(parse_division("div_b").unwrap(), Division::B));
        assert!(parse_division("c").is_err());
        assert_eq!(parse_seed("42").unwrap(), 42);
        assert!(parse_seed("random").is_ok());
        assert!(parse_seed("nope").is_err());
    }

    /// `config/example.toml` must stay loadable, and its documented vision and
    /// realism tables must still be the built-in defaults.
    #[test]
    fn example_config_matches_the_defaults() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/example.toml");
        let text = std::fs::read_to_string(&path).expect("read config/example.toml");
        let config: SimConfig = toml::from_str(&text).expect("parse config/example.toml");
        let default = SimConfig::default();
        assert_eq!(
            config.vision, default.vision,
            "[vision] drifted from the defaults"
        );
        assert_eq!(
            config.realism, default.realism,
            "[realism] drifted from the defaults"
        );
        assert_eq!(
            config.ball, default.ball,
            "[ball] drifted from the defaults"
        );
    }
}

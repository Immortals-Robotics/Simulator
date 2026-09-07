//! Ball and robot dynamics fitting. OWNER: dynamics (math) agent.
//!
//! `ssl-logtool dynamics <files...> --out docs/calibration/dynamics.json`
//! fits the ball rolling law, kicks, chips, ball–robot collisions, robot
//! accelerations and dribbling from recorded games. Everything is SI; all
//! thresholds are named constants in the submodules and echoed into the JSON.

mod chips;
mod collisions;
mod dribble;
mod kicks;
mod load;
mod robots;
mod rolling;
mod stats;
mod track;

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use load::Team;

/// Command-line arguments.
#[derive(Debug, Parser)]
pub struct Args {
    /// Log files (`.log` or `.log.gz`).
    pub files: Vec<PathBuf>,
    /// Write the JSON report here (default: stdout summary only).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Prefer the tracker source whose name contains this string (default: the one with kick events).
    #[arg(long)]
    pub source: Option<String>,
    /// Only print data-inventory information about each log.
    #[arg(long)]
    pub probe: bool,
    /// Print verbose per-episode diagnostics.
    #[arg(long)]
    pub verbose: bool,
}

/// Per-game inventory written to the JSON.
#[derive(Debug, serde::Serialize)]
struct GameInfo {
    name: String,
    competition: String,
    teams: (String, String),
    tracker_source: String,
    tracker_offset_s: f64,
    cameras: Vec<load::Camera>,
    field: load::Field,
    advertised_models: load::AdvertisedModels,
    raw_frames: usize,
    tracker_frames: usize,
    ball_samples: usize,
    rolling_episodes: usize,
    tracker_kicks: usize,
    flights: usize,
    bounces: usize,
    contacts: usize,
    dribble_episodes: usize,
}

/// Everything accumulated across games.
#[derive(Default)]
struct Accum {
    games: Vec<GameInfo>,
    episodes: Vec<rolling::Episode>,
    pooled_decel: Vec<(f64, f64, f64, String)>,
    kicks: Vec<kicks::Kick>,
    n_own_kicks: usize,
    n_own_matched: usize,
    team_names: BTreeMap<(String, Team), String>,
    flights: Vec<chips::Flight>,
    bounces: Vec<chips::Bounce>,
    contacts: Vec<collisions::Contact>,
    robots: robots::RobotAccum,
    dribble_episodes: Vec<dribble::DribbleEpisode>,
    dribble_held: Vec<(f64, f64, f64, f64, f64, f64)>,
    dribble_held_team: Vec<String>,
}

/// Entry point.
pub fn run(args: Args) -> Result<()> {
    let mut acc = Accum::default();
    for f in &args.files {
        let game = load::load(f, args.source.as_deref())?;
        if args.probe {
            probe(&game);
            continue;
        }
        process_game(&game, &mut acc, args.verbose);
    }
    if args.probe {
        return Ok(());
    }
    let rolling = rolling::aggregate(std::mem::take(&mut acc.episodes), &acc.pooled_decel);
    let kicks = kicks::aggregate(
        std::mem::take(&mut acc.kicks),
        acc.n_own_kicks,
        acc.n_own_matched,
        &acc.team_names,
    );
    let chips = chips::aggregate(
        std::mem::take(&mut acc.flights),
        std::mem::take(&mut acc.bounces),
    );
    let collisions = collisions::aggregate(std::mem::take(&mut acc.contacts));
    let robots = robots::aggregate(std::mem::take(&mut acc.robots));
    let dribble = dribble::aggregate(
        std::mem::take(&mut acc.dribble_episodes),
        std::mem::take(&mut acc.dribble_held),
        std::mem::take(&mut acc.dribble_held_team),
    );
    println!("=== ROLLING ===\n{}", rolling.text);
    println!("=== KICKS ===\n{}", kicks.text);
    println!("=== CHIPS ===\n{}", chips.text);
    println!("=== COLLISIONS ===\n{}", collisions.text);
    println!("=== ROBOTS ===\n{}", robots.text);
    println!("=== DRIBBLING ===\n{}", dribble.text);
    if let Some(out) = &args.out {
        let report = serde_json::json!({
            "generated_by": "ssl-logtool dynamics",
            "games": acc.games,
            "rolling": rolling,
            "kicks": kicks,
            "chips": chips,
            "collisions": collisions,
            "robots": robots,
            "dribbling": dribble,
        });
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(out, serde_json::to_string(&report)?)?;
        println!("wrote {}", out.display());
    }
    Ok(())
}

fn process_game(game: &load::Game, acc: &mut Accum, verbose: bool) {
    let ball = track::ball_track(game);
    let robots = track::robot_tracks(game);
    let tkicks = kicks::tracker_kicks(game);
    acc.team_names
        .insert((game.name.clone(), Team::Yellow), game.team_names.0.clone());
    acc.team_names
        .insert((game.name.clone(), Team::Blue), game.team_names.1.clone());
    // rolling
    let segs = rolling::segment(&ball);
    let mut n_ep = 0;
    for (a, b) in &segs {
        if let Some(ep) =
            rolling::fit_episode(&game.name, &game.competition, &ball[*a..*b], &tkicks)
        {
            if verbose {
                println!(
                    "ep t {:.2} dur {:.2} n {} v0 {:.2} A {:.2} B {:.2}/{:.2} tsw {:.2} rmsA {:.1} rmsB {:.1} rmsC {:.1} k {:?}",
                    ep.t0, ep.duration, ep.n, ep.v_start, ep.a_single, ep.a_slide, ep.a_roll, ep.t_switch,
                    ep.rms_a * 1e3, ep.rms_b * 1e3, ep.rms_c * 1e3,
                    ep.kick_v0_extrap.map(|v| ep.v_switch / v)
                );
            }
            acc.pooled_decel
                .extend(rolling::pooled_decel(&ball[*a..*b], &ep));
            acc.episodes.push(ep);
            n_ep += 1;
        }
    }
    // kicks
    let own = kicks::detect_kicks(&ball, &robots);
    acc.n_own_kicks += own.len();
    acc.n_own_matched += own
        .iter()
        .filter(|t| tkicks.iter().any(|k| (k.start - **t).abs() < 0.12))
        .count();
    for k in &tkicks {
        acc.kicks
            .push(kicks::analyse_kick(game, k, &ball, &robots, &own));
    }
    // chips
    let (flights, bounces) = chips::analyse(game, &ball, &tkicks, &own, verbose);
    let n_flights = flights.len();
    let n_bounces = bounces.len();
    acc.flights.extend(flights);
    acc.bounces.extend(bounces);
    // collisions
    let contacts = collisions::analyse(game, &ball, &robots, &tkicks, &own);
    let n_contacts = contacts.len();
    acc.contacts.extend(contacts);
    // robots
    acc.robots.add_game(game, &robots);
    // dribbling
    let d = dribble::analyse(game, &ball, &robots, &tkicks);
    let n_dribble = d.episodes.len();
    acc.dribble_episodes.extend(d.episodes);
    acc.dribble_held.extend(d.held);
    acc.dribble_held_team.extend(d.held_team);
    acc.games.push(GameInfo {
        name: game.name.clone(),
        competition: game.competition.clone(),
        teams: game.team_names.clone(),
        tracker_source: game.tracker_source.clone(),
        tracker_offset_s: game.tracker_offset,
        cameras: game.cameras.clone(),
        field: game.field,
        advertised_models: game.models,
        raw_frames: game.raw.len(),
        tracker_frames: game.tracker.len(),
        ball_samples: ball.len(),
        rolling_episodes: n_ep,
        tracker_kicks: tkicks.len(),
        flights: n_flights,
        bounces: n_bounces,
        contacts: n_contacts,
        dribble_episodes: n_dribble,
    });
    eprintln!(
        "{}: raw {} tracker {} ({}) offset {:.4} ball samples {} episodes {} kicks {} own-kicks {} flights {} bounces {} contacts {} dribbles {}",
        game.name,
        game.raw.len(),
        game.tracker.len(),
        game.tracker_source,
        game.tracker_offset,
        ball.len(),
        n_ep,
        tkicks.len(),
        own.len(),
        n_flights,
        n_bounces,
        n_contacts,
        n_dribble
    );
}

fn probe(g: &load::Game) {
    println!("== {} ({}) teams {:?}", g.name, g.competition, g.team_names);
    println!(
        "field {:?} cameras {:?}\nmodels {:?}",
        g.field, g.cameras, g.models
    );
    for s in &g.sources {
        println!(
            "  source {:?}: frames {} z {} bvel {} rvel {} kick {} caps {:?} span {:.1}s rate {:.1}Hz",
            s.name,
            s.frames,
            s.frames_with_ball_z,
            s.frames_with_ball_vel,
            s.frames_with_robot_vel,
            s.frames_with_kick,
            s.capabilities,
            s.t_last - s.t_first,
            s.frames as f64 / (s.t_last - s.t_first).max(1e-9)
        );
    }
    println!(
        "  chosen {:?} tracker_offset {:.4} log_offset {:.4}",
        g.tracker_source, g.tracker_offset, g.log_offset
    );
    let mut cams = BTreeMap::<u32, (u64, f64, f64, u64, u64)>::new();
    for f in &g.raw {
        let e = cams
            .entry(f.camera)
            .or_insert((0, f64::MAX, f64::MIN, 0, 0));
        e.0 += 1;
        for r in &f.robots {
            e.1 = e.1.min(r.x);
            e.2 = e.2.max(r.x);
        }
        e.3 += f.balls.len() as u64;
        if f.balls.len() > 1 {
            e.4 += 1;
        }
    }
    for (c, e) in cams {
        println!(
            "  cam {c}: frames {} robot x [{:.2},{:.2}] balls {} multi-ball frames {}",
            e.0, e.1, e.2, e.3, e.4
        );
    }
    let mut zh = [0u64; 8];
    for f in &g.tracker {
        if let Some(b) = &f.ball {
            zh[((b.z / 0.05).floor() as usize).min(7)] += 1;
        }
    }
    let kicks = kicks::tracker_kicks(g);
    println!(
        "  tracker frames {} z hist (50mm bins) {:?} distinct kicks {}",
        g.tracker.len(),
        zh,
        kicks.len()
    );
    let n = g.raw.len().max(1);
    let running = g.raw.iter().filter(|f| g.running_at(f.t)).count();
    println!(
        "  running fraction of raw frames {:.2}",
        running as f64 / n as f64
    );
}

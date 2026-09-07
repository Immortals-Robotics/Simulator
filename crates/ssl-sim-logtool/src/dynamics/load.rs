//! Load one SSL log into memory in SI units: raw detections per camera,
//! one tracker source, referee state timeline and geometry.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use ssl_sim_proto::gc;

use crate::reader::{LogReader, Record};

/// Raw ball detection (one camera, one frame).
#[derive(Debug, Clone, Copy)]
pub struct RawBall {
    /// Position [m].
    pub x: f64,
    pub y: f64,
    /// Blob area [px] if present.
    pub area: Option<u32>,
}

/// Raw robot detection.
#[derive(Debug, Clone, Copy)]
pub struct RawRobot {
    pub team: Team,
    pub id: u32,
    pub x: f64,
    pub y: f64,
    /// Orientation [rad] (None if the camera did not resolve it).
    pub theta: Option<f64>,
}

/// Team colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Team {
    Yellow,
    Blue,
}

impl Team {
    pub fn name(self) -> &'static str {
        match self {
            Team::Yellow => "yellow",
            Team::Blue => "blue",
        }
    }
}

/// One raw detection frame.
#[derive(Debug, Clone)]
pub struct RawFrame {
    pub camera: u32,
    /// Capture time [s] (vision clock).
    pub t: f64,
    pub balls: Vec<RawBall>,
    pub robots: Vec<RawRobot>,
}

/// Tracked ball state.
#[derive(Debug, Clone, Copy)]
pub struct TrkBall {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vel: Option<[f64; 3]>,
}

/// Tracked robot state.
#[derive(Debug, Clone, Copy)]
pub struct TrkRobot {
    pub team: Team,
    pub id: u32,
    pub x: f64,
    pub y: f64,
    pub vel: Option<[f64; 2]>,
}

/// Tracker `kicked_ball` info.
#[derive(Debug, Clone, Copy)]
pub struct TrkKick {
    pub pos: [f64; 2],
    pub vel: [f64; 3],
    pub start: f64,
    pub robot: Option<(Team, u32)>,
}

/// One tracker frame from the chosen source.
#[derive(Debug, Clone)]
pub struct TrkFrame {
    /// Tracker timestamp [s] (source clock).
    pub t: f64,
    pub ball: Option<TrkBall>,
    pub robots: Vec<TrkRobot>,
    pub kick: Option<TrkKick>,
}

/// Referee command sample.
#[derive(Debug, Clone, Copy)]
pub struct RefSample {
    /// Log receive time [s] (wall clock of the logger).
    pub t_log: f64,
    pub command: i32,
}

/// Camera calibration.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Camera {
    pub id: u32,
    /// World position of the optical centre [m].
    pub pos: [f64; 3],
}

/// Ball model as advertised by the vision geometry packet.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct AdvertisedModels {
    pub acc_slide: Option<f64>,
    pub acc_roll: Option<f64>,
    pub k_switch: Option<f64>,
    pub damping_xy_first_hop: Option<f64>,
    pub damping_xy_other_hops: Option<f64>,
    pub damping_z: Option<f64>,
}

/// Field geometry [m].
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct Field {
    pub length: f64,
    pub width: f64,
    pub goal_width: f64,
    pub goal_depth: f64,
    pub boundary_width: f64,
}

/// Summary of a tracker source seen in the log.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SourceInfo {
    pub name: String,
    pub frames: u64,
    pub frames_with_ball_z: u64,
    pub frames_with_ball_vel: u64,
    pub frames_with_robot_vel: u64,
    pub frames_with_kick: u64,
    pub capabilities: Vec<i32>,
    pub t_first: f64,
    pub t_last: f64,
}

/// One game.
#[derive(Debug)]
pub struct Game {
    pub name: String,
    pub raw: Vec<RawFrame>,
    pub tracker: Vec<TrkFrame>,
    pub tracker_source: String,
    pub sources: Vec<SourceInfo>,
    pub referee: Vec<RefSample>,
    pub cameras: Vec<Camera>,
    pub field: Field,
    pub models: AdvertisedModels,
    /// Offset such that `tracker.t + tracker_offset ~= raw.t` (seconds).
    pub tracker_offset: f64,
    /// Offset such that `referee.t_log + log_offset ~= raw.t`.
    pub log_offset: f64,
    /// Team names from the referee (yellow, blue).
    pub team_names: (String, String),
    /// Competition tag derived from the file name/date.
    pub competition: String,
}

fn team_of(color: i32) -> Team {
    if color == 2 {
        Team::Blue
    } else {
        Team::Yellow
    }
}

/// Load a log. `prefer_source` picks the tracker source by substring, else
/// the source with the most frames having a 3D ball and velocities wins.
pub fn load(path: &Path, prefer_source: Option<&str>) -> Result<Game> {
    let mut reader = LogReader::open(path)?;
    let mut raw = Vec::new();
    let mut per_source: BTreeMap<String, Vec<TrkFrame>> = BTreeMap::new();
    let mut infos: BTreeMap<String, SourceInfo> = BTreeMap::new();
    let mut referee = Vec::new();
    let mut team_names = (String::new(), String::new());
    let mut cameras: BTreeMap<u32, Camera> = BTreeMap::new();
    let mut field = Field::default();
    let mut models = AdvertisedModels::default();
    // pairs of (raw t_capture, log time) to estimate the log clock offset
    let mut raw_vs_log: Vec<f64> = Vec::new();
    let mut trk_vs_log: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    while let Some(entry) = reader.next_entry()? {
        let t_log = entry.time_ns as f64 * 1e-9;
        match entry.record {
            Record::Vision(p) => {
                if let Some(d) = p.detection {
                    if raw_vs_log.len() < 200_000 {
                        raw_vs_log.push(d.t_capture - t_log);
                    }
                    let mut robots = Vec::with_capacity(22);
                    for (team, list) in [
                        (Team::Yellow, &d.robots_yellow),
                        (Team::Blue, &d.robots_blue),
                    ] {
                        for r in list {
                            if let Some(id) = r.robot_id {
                                robots.push(RawRobot {
                                    team,
                                    id,
                                    x: r.x as f64 * 1e-3,
                                    y: r.y as f64 * 1e-3,
                                    theta: r.orientation.map(|o| o as f64),
                                });
                            }
                        }
                    }
                    let balls = d
                        .balls
                        .iter()
                        .map(|b| RawBall {
                            x: b.x as f64 * 1e-3,
                            y: b.y as f64 * 1e-3,
                            area: b.area,
                        })
                        .collect();
                    raw.push(RawFrame {
                        camera: d.camera_id,
                        t: d.t_capture,
                        balls,
                        robots,
                    });
                }
                if let Some(g) = p.geometry {
                    field = Field {
                        length: g.field.field_length as f64 * 1e-3,
                        width: g.field.field_width as f64 * 1e-3,
                        goal_width: g.field.goal_width as f64 * 1e-3,
                        goal_depth: g.field.goal_depth as f64 * 1e-3,
                        boundary_width: g.field.boundary_width as f64 * 1e-3,
                    };
                    for c in &g.calib {
                        if let (Some(x), Some(y), Some(z)) = (
                            c.derived_camera_world_tx,
                            c.derived_camera_world_ty,
                            c.derived_camera_world_tz,
                        ) {
                            cameras.insert(
                                c.camera_id,
                                Camera {
                                    id: c.camera_id,
                                    pos: [x as f64 * 1e-3, y as f64 * 1e-3, z as f64 * 1e-3],
                                },
                            );
                        }
                    }
                    if let Some(m) = g.models {
                        if let Some(s) = m.straight_two_phase {
                            models.acc_slide = Some(s.acc_slide);
                            models.acc_roll = Some(s.acc_roll);
                            models.k_switch = Some(s.k_switch);
                        }
                        if let Some(c) = m.chip_fixed_loss {
                            models.damping_xy_first_hop = Some(c.damping_xy_first_hop);
                            models.damping_xy_other_hops = Some(c.damping_xy_other_hops);
                            models.damping_z = Some(c.damping_z);
                        }
                    }
                }
            }
            Record::Tracker(p) => {
                let name = p.source_name.clone().unwrap_or_else(|| p.uuid.clone());
                let Some(f) = p.tracked_frame else { continue };
                let info = infos.entry(name.clone()).or_insert_with(|| SourceInfo {
                    name: name.clone(),
                    t_first: f.timestamp,
                    ..Default::default()
                });
                info.frames += 1;
                info.t_last = f.timestamp;
                if info.capabilities.is_empty() {
                    info.capabilities = f.capabilities.clone();
                }
                let ball = f.balls.first().map(|b| TrkBall {
                    x: b.pos.x as f64,
                    y: b.pos.y as f64,
                    z: b.pos.z as f64,
                    vel: b.vel.map(|v| [v.x as f64, v.y as f64, v.z as f64]),
                });
                if let Some(b) = ball {
                    if b.z != 0.0 {
                        info.frames_with_ball_z += 1;
                    }
                    if b.vel.is_some() {
                        info.frames_with_ball_vel += 1;
                    }
                }
                if f.robots.iter().any(|r| r.vel.is_some()) {
                    info.frames_with_robot_vel += 1;
                }
                if f.kicked_ball.is_some() {
                    info.frames_with_kick += 1;
                }
                let robots = f
                    .robots
                    .iter()
                    .map(|r| TrkRobot {
                        team: team_of(r.robot_id.team_color),
                        id: r.robot_id.id,
                        x: r.pos.x as f64,
                        y: r.pos.y as f64,
                        vel: r.vel.map(|v| [v.x as f64, v.y as f64]),
                    })
                    .collect();
                let kick = f.kicked_ball.map(|k| TrkKick {
                    pos: [k.pos.x as f64, k.pos.y as f64],
                    vel: [k.vel.x as f64, k.vel.y as f64, k.vel.z as f64],
                    start: k.start_timestamp,
                    robot: k.robot_id.map(|r| (team_of(r.team_color), r.id)),
                });
                let v = trk_vs_log.entry(name.clone()).or_default();
                if v.len() < 200_000 {
                    v.push(f.timestamp - t_log);
                }
                per_source.entry(name).or_default().push(TrkFrame {
                    t: f.timestamp,
                    ball,
                    robots,
                    kick,
                });
            }
            Record::Referee(r) => {
                if team_names.0.is_empty() && !r.yellow.name.is_empty() {
                    team_names = (r.yellow.name.clone(), r.blue.name.clone());
                }
                referee.push(RefSample {
                    t_log,
                    command: r.command,
                });
            }
            Record::Other { .. } => {}
        }
    }
    raw.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    let sources: Vec<SourceInfo> = infos.values().cloned().collect();
    let chosen = match prefer_source {
        Some(p) => sources
            .iter()
            .find(|s| s.name.contains(p))
            .map(|s| s.name.clone()),
        None => None,
    }
    .or_else(|| {
        sources
            .iter()
            .max_by_key(|s| (s.frames_with_kick > 0, s.frames_with_ball_z, s.frames))
            .map(|s| s.name.clone())
    })
    .unwrap_or_default();
    let mut tracker = per_source.remove(&chosen).unwrap_or_default();
    tracker.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    tracker.dedup_by(|a, b| a.t == b.t);
    // Clock offsets. The raw capture clock is the reference. Logger receive
    // time ~= t_capture + delay, so use the median difference.
    let log_offset = median_of(&mut raw_vs_log);
    // tracker.t - t_log  ->  tracker.t + tracker_offset = raw.t
    let tracker_offset = match trk_vs_log.get_mut(&chosen) {
        Some(v) if !v.is_empty() => {
            let trk_minus_log = median_of(v);
            // raw.t - log = log_offset; trk.t - log = trk_minus_log
            // raw.t - trk.t = log_offset - trk_minus_log
            log_offset - trk_minus_log
        }
        _ => 0.0,
    };
    let refine = refine_offset(&raw, &tracker, tracker_offset);
    let tracker_offset = tracker_offset + refine;
    for f in &mut tracker {
        f.t += tracker_offset;
        if let Some(k) = &mut f.kick {
            k.start += tracker_offset;
        }
    }
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().trim_end_matches(".log").to_string())
        .unwrap_or_default();
    let competition = if name.starts_with("2026-03") {
        "GO26".to_string()
    } else if name.starts_with("2026-07") {
        "RC26".to_string()
    } else {
        "other".to_string()
    };
    Ok(Game {
        name,
        team_names,
        competition,
        raw,
        tracker,
        tracker_source: chosen,
        sources,
        referee,
        cameras: cameras.into_values().collect(),
        field,
        models,
        tracker_offset,
        log_offset,
    })
}

/// Refine the tracker clock offset by matching tracked robot positions to raw
/// detections of moving robots: scan ±120 ms in 2 ms steps for the offset with
/// the smallest median position error, then parabolic refinement.
fn refine_offset(raw: &[RawFrame], tracker: &[TrkFrame], base: f64) -> f64 {
    // per-robot raw series (t, x, y)
    type Series = BTreeMap<(Team, u32), Vec<(f64, f64, f64)>>;
    let mut series: Series = BTreeMap::new();
    for f in raw {
        for r in &f.robots {
            series
                .entry((r.team, r.id))
                .or_default()
                .push((f.t, r.x, r.y));
        }
    }
    let cost = |off: f64| -> f64 {
        let mut d = Vec::new();
        for f in tracker.iter().step_by(7) {
            let t = f.t + base + off;
            for r in &f.robots {
                let Some(v) = r.vel else { continue };
                let sp = (v[0] * v[0] + v[1] * v[1]).sqrt();
                if sp < 0.8 {
                    continue;
                }
                let Some(s) = series.get(&(r.team, r.id)) else {
                    continue;
                };
                let i = s.partition_point(|x| x.0 < t);
                if i == 0 || i >= s.len() {
                    continue;
                }
                let (a, b) = (s[i - 1], s[i]);
                if b.0 - a.0 > 0.05 {
                    continue;
                }
                let fr = (t - a.0) / (b.0 - a.0);
                let x = a.1 + (b.1 - a.1) * fr;
                let y = a.2 + (b.2 - a.2) * fr;
                d.push(((r.x - x).powi(2) + (r.y - y).powi(2)).sqrt());
                if d.len() > 20_000 {
                    break;
                }
            }
        }
        if d.len() < 200 {
            return f64::NAN;
        }
        median_of(&mut d)
    };
    let mut best = (0.0, f64::INFINITY);
    let mut curve = Vec::new();
    for k in -60..=60 {
        let off = k as f64 * 0.002;
        let c = cost(off);
        curve.push((off, c));
        if c < best.1 {
            best = (off, c);
        }
    }
    if !best.1.is_finite() {
        return 0.0;
    }
    // parabolic refinement around the best grid point
    let i = curve.iter().position(|c| c.0 == best.0).unwrap();
    if i > 0 && i + 1 < curve.len() {
        let (y0, y1, y2) = (curve[i - 1].1, curve[i].1, curve[i + 1].1);
        let den = y0 - 2.0 * y1 + y2;
        if den > 0.0 {
            return best.0 + 0.002 * 0.5 * (y0 - y2) / den;
        }
    }
    best.0
}

fn median_of(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

impl Game {
    /// Referee command in force at raw time `t` (None before the first sample).
    pub fn command_at(&self, t: f64) -> Option<i32> {
        let t_log = t - self.log_offset;
        let idx = self.referee.partition_point(|r| r.t_log <= t_log);
        if idx == 0 {
            None
        } else {
            Some(self.referee[idx - 1].command)
        }
    }

    /// Whether the game is running (not HALT/STOP/timeout/ball placement) at raw time `t`.
    pub fn running_at(&self, t: f64) -> bool {
        use gc::referee::Command as C;
        match self.command_at(t) {
            Some(c) => !matches!(
                C::try_from(c).unwrap_or(C::Halt),
                C::Halt
                    | C::Stop
                    | C::TimeoutYellow
                    | C::TimeoutBlue
                    | C::BallPlacementYellow
                    | C::BallPlacementBlue
            ),
            None => false,
        }
    }
}

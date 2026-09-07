//! Vision imperfection analysis. OWNER: vision agent.
//!
//! One streaming pass per log measures, per camera where it matters:
//! frame timing, camera geometry and coverage, robot/ball detection noise,
//! dropouts, multi-camera disagreement, spurious "dribbler" balls, the ball
//! `area` model, and ball visibility while dribbled.
//!
//! Everything is computed inside a short sliding window (a few hundred ms of
//! detection frames per camera plus the tracker stream), so memory is constant
//! and a 400 MB log takes a couple of seconds.

mod stats;

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{json, Value};

use crate::reader::{LogReader, Record};
use stats::{r, Binned, BinnedRate, Grid, Hist, Quantiles, Rate, Welford};

/// Command-line arguments.
#[derive(Debug, Parser)]
pub struct Args {
    /// Log files (`.log` or `.log.gz`).
    pub files: Vec<PathBuf>,
    /// Write the JSON report here (default: stdout summary only).
    #[arg(long)]
    pub out: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// tuning constants
// ---------------------------------------------------------------------------

/// Process a frame once it is this far behind the newest capture time [s].
const SETTLE: f64 = 0.35;
/// Drop a frame from the window once it is this far behind [s].
const RETAIN: f64 = 0.80;
/// Length of the detrending window used for the noise estimate [s].
const NOISE_WIN: f64 = 1.0;
/// Maximum gap inside a noise window before it is restarted [s].
const NOISE_MAX_GAP: f64 = 0.10;
/// Minimum samples in a noise window.
const NOISE_MIN_N: usize = 20;
/// Maximum fitted speed for a window to count as "slow" [m/s].
const SLOW_SPEED: f64 = 0.03;
/// Maximum end-to-end displacement for a window to count as "slow" [m].
const SLOW_DISP: f64 = 0.02;
/// Maximum end-to-end displacement for a "strictly static" window [m].
const STATIC_DISP: f64 = 0.005;
/// Referee command must have been static for this long [s].
const REF_STATIC_SETTLE: f64 = 2.0;
/// Centre-to-kicker-face distance assumed [m].
const CENTER_TO_DRIBBLER: f64 = 0.075;
/// Ball radius [m].
const BALL_RADIUS: f64 = 0.0215;
/// A raw ball detection is "the tracked ball" within this distance [m].
const BALL_MATCH: f64 = 0.15;
/// Tracker ball speed below which the ball counts as resting [m/s].
const BALL_REST_SPEED: f64 = 0.03;
/// Ball centre within this distance of a robot centre counts as dribbled [m].
const DRIBBLE_RADIUS: f64 = 0.11;
/// Distance from a camera's nadir inside which it certainly has the object in
/// view; used to separate genuine misses from field-of-view edge effects [m].
const CORE_NADIR: f64 = 5.0;
/// Camera exposure/readout + LAN latency that a log cannot observe [s].
const UNOBSERVED_CAMERA_LATENCY: f64 = 0.015;
/// Position differences beyond this are treated as identity mismatches [m].
const MATCH_OUTLIER: f64 = 0.15;
/// `PIXEL_PER_AREA` in `ssl-sim-core`; used to invert the fitted area constant.
const PIXEL_PER_AREA: f64 = 10.0;

// ---------------------------------------------------------------------------
// window records
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct RawBall {
    x: f64,
    y: f64,
    z: Option<f64>,
    area: Option<u32>,
    conf: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct RawRobot {
    slot: usize, // team * 16 + id
    x: f64,
    y: f64,
    phi: f64,
    conf: f64,
    height: f64,
}

#[derive(Debug, Default)]
struct RawFrame {
    t_capture: f64,
    t_sent: f64,
    recv_s: f64,
    frame_number: u32,
    balls: Vec<RawBall>,
    robots: Vec<RawRobot>,
    /// Bit i set when slot i was detected in this frame.
    mask: u32,
    ref_static: bool,
}

impl RawFrame {
    fn clear(&mut self) {
        self.balls.clear();
        self.robots.clear();
        self.mask = 0;
    }
}

#[derive(Debug, Clone, Copy)]
struct TrkRobot {
    slot: usize,
    x: f64,
    y: f64,
    phi: f64,
    vx: f64,
    vy: f64,
}

#[derive(Debug, Default)]
struct TrkFrame {
    t: f64,
    bx: f64,
    by: f64,
    bz: f64,
    bv: f64,
    has_ball: bool,
    robots: Vec<TrkRobot>,
}

// ---------------------------------------------------------------------------
// accumulators
// ---------------------------------------------------------------------------

/// One rolling window of samples for the detrended noise estimate.
#[derive(Debug, Default, Clone)]
struct NoiseWin {
    t: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    p: Vec<f64>,
    a: Vec<f64>,
    nadir: f64,
    last_t: f64,
    all_static: bool,
}

impl NoiseWin {
    fn reset(&mut self) {
        self.t.clear();
        self.x.clear();
        self.y.clear();
        self.p.clear();
        self.a.clear();
        self.nadir = 0.0;
        self.all_static = true;
    }
}

/// Linear least squares: returns `(slope, residual_std)`.
fn detrend(t: &[f64], v: &[f64]) -> Option<(f64, f64)> {
    let n = t.len();
    if n < 6 {
        return None;
    }
    let nf = n as f64;
    let mt = t.iter().sum::<f64>() / nf;
    let mv = v.iter().sum::<f64>() / nf;
    let mut stt = 0.0;
    let mut stv = 0.0;
    for i in 0..n {
        let dt = t[i] - mt;
        stt += dt * dt;
        stv += dt * (v[i] - mv);
    }
    if stt <= 1e-12 {
        return None;
    }
    let slope = stv / stt;
    let mut ss = 0.0;
    for i in 0..n {
        let e = v[i] - mv - slope * (t[i] - mt);
        ss += e * e;
    }
    Some((slope, (ss / (nf - 2.0)).sqrt()))
}

/// Per-camera accumulators. Mergeable so the aggregate is just a fold.
#[derive(Debug, Clone)]
struct CamAcc {
    id: u32,
    frames: u64,
    t_first: f64,
    t_last: f64,
    // timing
    gap: Welford,
    gap_clean: Welford,
    gap_q: Quantiles,
    gap_neg: u64,
    gap_zero: u64,
    gap_long: u64, // > 1.8 nominal periods
    proc: Welford,
    proc_q: Quantiles,
    transport: Welford,
    transport_q: Quantiles,
    fn_step: Hist,
    // detections
    empty_frames: u64,
    robot_dets: u64,
    ball_dets: u64,
    robot_conf: Welford,
    robot_conf_h: Hist,
    ball_conf: Welford,
    ball_conf_h: Hist,
    robot_height: Welford,
    balls_per_frame: Hist,
    ball_z_reported: u64,
    ball_area_present: u64,
    ball_area_missing: u64,
    // noise
    noise_x: Welford,
    noise_y: Welford,
    noise_phi: Welford,
    noise_x_static: Welford,
    noise_y_static: Welford,
    noise_phi_static: Welford,
    noise_vs_nadir_p: Binned,
    noise_vs_nadir_phi: Binned,
    ball_noise: Welford,
    ball_noise_vs_nadir: Binned,
    ball_area_noise: Welford,
    ball_area_rel_noise: Welford,
    // dropouts
    robot_single_drop: Rate,
    robot_gap: Hist,
    robot_dupes: u64,
    ball_single_drop: Rate,
    ball_gap: Hist,
    ball_drop_vs_robot_dist: BinnedRate,
    // ball visibility
    ball_seen_vs_robot_dist: BinnedRate,
    dribbled_ball_seen: Rate,
    free_ball_seen: Rate,
    // region-gated completeness (tracker says the object is in this camera's
    // Manhattan region; did the camera report it this frame?)
    robot_seen_in_region: Rate,
    robot_seen_core: Rate,
    robot_seen_vs_nadir: BinnedRate,
    ball_seen_in_region: Rate,
    ball_seen_core: Rate,
    ball_seen_vs_nadir: BinnedRate,
    ball_seen_vs_speed: BinnedRate,
    ball_seen_free_at_rest: Rate,
    // systematic per-camera offset relative to the fused tracker estimate
    bias_x: Welford,
    bias_y: Welford,
    bias_radial: Welford,
    bias_tangential: Welford,
    bias_radial_vs_nadir: Binned,
    bias_phi: Welford,
    // area model
    area: Welford,
    area_k: Welford,
    area_vs_nadir: Binned,
    area_k_vs_z: Binned,
    area_vs_z: Binned,
    // coverage
    grid: Grid,
}

impl CamAcc {
    fn new(id: u32) -> Self {
        Self {
            id,
            frames: 0,
            t_first: 0.0,
            t_last: 0.0,
            gap: Welford::default(),
            gap_clean: Welford::default(),
            gap_q: Quantiles::new(200_000),
            gap_neg: 0,
            gap_zero: 0,
            gap_long: 0,
            proc: Welford::default(),
            proc_q: Quantiles::new(200_000),
            transport: Welford::default(),
            transport_q: Quantiles::new(200_000),
            fn_step: Hist::new(-0.5, 1.0, 12),
            empty_frames: 0,
            robot_dets: 0,
            ball_dets: 0,
            robot_conf: Welford::default(),
            robot_conf_h: Hist::new(0.0, 0.05, 21),
            ball_conf: Welford::default(),
            ball_conf_h: Hist::new(0.0, 0.05, 21),
            robot_height: Welford::default(),
            balls_per_frame: Hist::new(-0.5, 1.0, 10),
            ball_z_reported: 0,
            ball_area_present: 0,
            ball_area_missing: 0,
            noise_x: Welford::default(),
            noise_y: Welford::default(),
            noise_phi: Welford::default(),
            noise_x_static: Welford::default(),
            noise_y_static: Welford::default(),
            noise_phi_static: Welford::default(),
            noise_vs_nadir_p: Binned::new(0.0, 0.5, 20),
            noise_vs_nadir_phi: Binned::new(0.0, 0.5, 20),
            ball_noise: Welford::default(),
            ball_noise_vs_nadir: Binned::new(0.0, 0.5, 20),
            ball_area_noise: Welford::default(),
            ball_area_rel_noise: Welford::default(),
            robot_single_drop: Rate::default(),
            robot_gap: Hist::new(0.5, 1.0, 60),
            robot_dupes: 0,
            ball_single_drop: Rate::default(),
            ball_gap: Hist::new(0.5, 1.0, 60),
            ball_drop_vs_robot_dist: BinnedRate::new(0.0, 0.05, 24),
            ball_seen_vs_robot_dist: BinnedRate::new(0.0, 0.02, 30),
            dribbled_ball_seen: Rate::default(),
            free_ball_seen: Rate::default(),
            robot_seen_in_region: Rate::default(),
            robot_seen_core: Rate::default(),
            robot_seen_vs_nadir: BinnedRate::new(0.0, 0.5, 20),
            ball_seen_in_region: Rate::default(),
            ball_seen_core: Rate::default(),
            ball_seen_vs_nadir: BinnedRate::new(0.0, 0.5, 20),
            ball_seen_vs_speed: BinnedRate::new(0.0, 0.25, 32),
            ball_seen_free_at_rest: Rate::default(),
            bias_x: Welford::default(),
            bias_y: Welford::default(),
            bias_radial: Welford::default(),
            bias_tangential: Welford::default(),
            bias_radial_vs_nadir: Binned::new(0.0, 0.5, 20),
            bias_phi: Welford::default(),
            area: Welford::default(),
            area_k: Welford::default(),
            area_vs_nadir: Binned::new(0.0, 0.5, 20),
            area_k_vs_z: Binned::new(0.0, 0.05, 20),
            area_vs_z: Binned::new(0.0, 0.05, 20),
            grid: Grid::new(8.0, 6.0, 0.1),
        }
    }

    fn merge(&mut self, o: &Self) {
        self.frames += o.frames;
        self.gap.merge(&o.gap);
        self.gap_clean.merge(&o.gap_clean);
        self.gap_q.merge(&o.gap_q);
        self.gap_neg += o.gap_neg;
        self.gap_zero += o.gap_zero;
        self.gap_long += o.gap_long;
        self.proc.merge(&o.proc);
        self.proc_q.merge(&o.proc_q);
        self.transport.merge(&o.transport);
        self.transport_q.merge(&o.transport_q);
        self.fn_step.merge(&o.fn_step);
        self.empty_frames += o.empty_frames;
        self.robot_dets += o.robot_dets;
        self.ball_dets += o.ball_dets;
        self.robot_conf.merge(&o.robot_conf);
        self.robot_conf_h.merge(&o.robot_conf_h);
        self.ball_conf.merge(&o.ball_conf);
        self.ball_conf_h.merge(&o.ball_conf_h);
        self.robot_height.merge(&o.robot_height);
        self.balls_per_frame.merge(&o.balls_per_frame);
        self.ball_z_reported += o.ball_z_reported;
        self.ball_area_present += o.ball_area_present;
        self.ball_area_missing += o.ball_area_missing;
        self.noise_x.merge(&o.noise_x);
        self.noise_y.merge(&o.noise_y);
        self.noise_phi.merge(&o.noise_phi);
        self.noise_x_static.merge(&o.noise_x_static);
        self.noise_y_static.merge(&o.noise_y_static);
        self.noise_phi_static.merge(&o.noise_phi_static);
        self.noise_vs_nadir_p.merge(&o.noise_vs_nadir_p);
        self.noise_vs_nadir_phi.merge(&o.noise_vs_nadir_phi);
        self.ball_noise.merge(&o.ball_noise);
        self.ball_noise_vs_nadir.merge(&o.ball_noise_vs_nadir);
        self.ball_area_noise.merge(&o.ball_area_noise);
        self.ball_area_rel_noise.merge(&o.ball_area_rel_noise);
        self.robot_single_drop.merge(&o.robot_single_drop);
        self.robot_gap.merge(&o.robot_gap);
        self.robot_dupes += o.robot_dupes;
        self.ball_single_drop.merge(&o.ball_single_drop);
        self.ball_gap.merge(&o.ball_gap);
        self.ball_drop_vs_robot_dist
            .merge(&o.ball_drop_vs_robot_dist);
        self.ball_seen_vs_robot_dist
            .merge(&o.ball_seen_vs_robot_dist);
        self.dribbled_ball_seen.merge(&o.dribbled_ball_seen);
        self.free_ball_seen.merge(&o.free_ball_seen);
        self.robot_seen_in_region.merge(&o.robot_seen_in_region);
        self.robot_seen_core.merge(&o.robot_seen_core);
        self.robot_seen_vs_nadir.merge(&o.robot_seen_vs_nadir);
        self.ball_seen_in_region.merge(&o.ball_seen_in_region);
        self.ball_seen_core.merge(&o.ball_seen_core);
        self.ball_seen_vs_nadir.merge(&o.ball_seen_vs_nadir);
        self.ball_seen_vs_speed.merge(&o.ball_seen_vs_speed);
        self.ball_seen_free_at_rest.merge(&o.ball_seen_free_at_rest);
        self.bias_x.merge(&o.bias_x);
        self.bias_y.merge(&o.bias_y);
        self.bias_radial.merge(&o.bias_radial);
        self.bias_tangential.merge(&o.bias_tangential);
        self.bias_radial_vs_nadir.merge(&o.bias_radial_vs_nadir);
        self.bias_phi.merge(&o.bias_phi);
        self.area.merge(&o.area);
        self.area_k.merge(&o.area_k);
        self.area_vs_nadir.merge(&o.area_vs_nadir);
        self.area_k_vs_z.merge(&o.area_k_vs_z);
        self.area_vs_z.merge(&o.area_vs_z);
    }

    fn hz(&self) -> f64 {
        let dt = self.t_last - self.t_first;
        if dt > 0.0 && self.frames > 1 {
            (self.frames as f64 - 1.0) / dt
        } else {
            0.0
        }
    }

    fn json(&mut self, with_coverage: bool) -> Value {
        let mut v = json!({
            "camera_id": self.id,
            "frames": self.frames,
            "frame_rate_hz": r(self.hz()),
            "frame_period": {
                "stats": self.gap.summary(),
                "stats_excluding_gaps_over_3_periods": self.gap_clean.summary(),
                "quantiles": self.gap_q.summary(),
                "negative_gaps": self.gap_neg,
                "zero_gaps": self.gap_zero,
                "long_gaps_gt_1p8_period": self.gap_long,
            },
            "t_sent_minus_t_capture": {
                "stats": self.proc.summary(),
                "quantiles": self.proc_q.summary(),
            },
            "logger_recv_minus_t_sent": {
                "stats": self.transport.summary(),
                "quantiles": self.transport_q.summary(),
                "note": "logger clock has an arbitrary offset per log; only the spread is meaningful",
            },
            "frame_number_step": self.fn_step.summary(),
            "detections": {
                "empty_frames": self.empty_frames,
                "empty_frame_ratio": r(self.empty_frames as f64 / self.frames.max(1) as f64),
                "robot_detections": self.robot_dets,
                "ball_detections": self.ball_dets,
                "robot_confidence": self.robot_conf.summary(),
                "robot_confidence_hist": self.robot_conf_h.summary(),
                "ball_confidence": self.ball_conf.summary(),
                "ball_confidence_hist": self.ball_conf_h.summary(),
                "robot_height_mm": self.robot_height.summary(),
                "balls_per_frame_hist": self.balls_per_frame.summary(),
                "ball_z_reported": self.ball_z_reported,
                "ball_area_present": self.ball_area_present,
                "ball_area_missing": self.ball_area_missing,
            },
            "noise": {
                "robot_x_slow": self.noise_x.summary(),
                "robot_y_slow": self.noise_y.summary(),
                "robot_phi_slow": self.noise_phi.summary(),
                "robot_x_static": self.noise_x_static.summary(),
                "robot_y_static": self.noise_y_static.summary(),
                "robot_phi_static": self.noise_phi_static.summary(),
                "robot_pos_vs_nadir_dist": self.noise_vs_nadir_p.summary(),
                "robot_phi_vs_nadir_dist": self.noise_vs_nadir_phi.summary(),
                "ball_at_rest": self.ball_noise.summary(),
                "ball_vs_nadir_dist": self.ball_noise_vs_nadir.summary(),
                "ball_area_at_rest_px": self.ball_area_noise.summary(),
                "ball_area_at_rest_relative": self.ball_area_rel_noise.summary(),
            },
            "dropouts": {
                "robot_single_frame": self.robot_single_drop.summary(),
                "robot_gap_frames_hist": self.robot_gap.summary(),
                "robot_dropout_rate_incl_multi_frame_gaps": r(gap_rate(
                    self.robot_gap.weighted_sum(), self.robot_dets)),
                "robot_duplicate_ids_in_frame": self.robot_dupes,
                "ball_single_frame": self.ball_single_drop.summary(),
                "ball_gap_frames_hist": self.ball_gap.summary(),
                "ball_dropout_rate_incl_multi_frame_gaps": r(gap_rate(
                    self.ball_gap.weighted_sum(), self.ball_dets)),
                "ball_drop_vs_nearest_robot_dist": self.ball_drop_vs_robot_dist.summary(),
            },
            "ball_visibility": {
                "seen_vs_nearest_robot_dist": self.ball_seen_vs_robot_dist.summary(),
                "dribbled": self.dribbled_ball_seen.summary(),
                "free_gt_0p3m": self.free_ball_seen.summary(),
                "in_region": self.ball_seen_in_region.summary(),
                "in_core_region": self.ball_seen_core.summary(),
                "seen_vs_nadir_dist": self.ball_seen_vs_nadir.summary(),
                "seen_vs_ball_speed": self.ball_seen_vs_speed.summary(),
                "free_and_slow": self.ball_seen_free_at_rest.summary(),
            },
            "completeness": {
                "robot_seen_in_region": self.robot_seen_in_region.summary(),
                "robot_seen_in_core_region": self.robot_seen_core.summary(),
                "robot_seen_vs_nadir_dist": self.robot_seen_vs_nadir.summary(),
            },
            "systematic_offset_vs_tracker": {
                "dx": self.bias_x.summary(),
                "dy": self.bias_y.summary(),
                "radial_outward": self.bias_radial.summary(),
                "tangential": self.bias_tangential.summary(),
                "radial_vs_nadir_dist": self.bias_radial_vs_nadir.summary(),
                "dphi": self.bias_phi.summary(),
            },
            "ball_area": {
                "area_px": self.area.summary(),
                "k_px_m2": self.area_k.summary(),
                "implied_focal_px_at_pixel_per_area_10": r(implied_focal(self.area_k.mean())),
                "area_vs_nadir_dist": self.area_vs_nadir.summary(),
                "k_vs_tracker_ball_z": self.area_k_vs_z.summary(),
                "area_vs_tracker_ball_z": self.area_vs_z.summary(),
            },
        });
        if with_coverage {
            v["coverage"] = coverage_json(&self.grid);
        }
        v
    }
}

/// Missed frames / (missed + seen) for a per-object detection sequence.
fn gap_rate(missed: f64, seen: u64) -> f64 {
    if seen == 0 {
        0.0
    } else {
        missed / (missed + seen as f64)
    }
}

fn implied_focal(k: f64) -> f64 {
    // area = PIXEL_PER_AREA * pi * r_mm^2 * f^2 / d_mm^2, k = area * d_m^2.
    let r_mm = BALL_RADIUS * 1000.0;
    let denom = PIXEL_PER_AREA * std::f64::consts::PI * r_mm * r_mm / 1e6;
    if denom <= 0.0 || k <= 0.0 {
        0.0
    } else {
        (k / denom).sqrt()
    }
}

fn coverage_json(grid: &Grid) -> Value {
    let mask = grid.mask(0.02);
    let mut minx = f64::INFINITY;
    let mut maxx = f64::NEG_INFINITY;
    let mut miny = f64::INFINITY;
    let mut maxy = f64::NEG_INFINITY;
    let mut cells = 0u64;
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            if mask[iy * grid.nx + ix] {
                cells += 1;
                let (cx, cy) = grid.centre(ix, iy);
                minx = minx.min(cx);
                maxx = maxx.max(cx);
                miny = miny.min(cy);
                maxy = maxy.max(cy);
            }
        }
    }
    if cells == 0 {
        return json!({ "cells": 0 });
    }
    json!({
        "cells": cells,
        "area_m2": r(cells as f64 * grid.step * grid.step),
        "bbox": [r(minx), r(miny), r(maxx), r(maxy)],
    })
}

/// Statistics for one ordered camera pair.
#[derive(Debug, Clone)]
struct PairAcc {
    dx: Welford,
    dy: Welford,
    dr: Welford,
    dphi: Welford,
    dr_q: Quantiles,
    dphi_q: Quantiles,
    dt: Welford,
    dt_q: Quantiles,
    dt_phase: Hist,
    dr_static: Welford,
    dphi_static: Welford,
    dx_static: Welford,
    dy_static: Welford,
    outliers: u64,
    gross_phi: u64,
}

impl PairAcc {
    fn new() -> Self {
        Self {
            dx: Welford::default(),
            dy: Welford::default(),
            dr: Welford::default(),
            dphi: Welford::default(),
            dr_q: Quantiles::new(200_000),
            dphi_q: Quantiles::new(200_000),
            dt: Welford::default(),
            dt_q: Quantiles::new(200_000),
            dt_phase: Hist::new(-0.5, 0.05, 20),
            dr_static: Welford::default(),
            dphi_static: Welford::default(),
            dx_static: Welford::default(),
            dy_static: Welford::default(),
            outliers: 0,
            gross_phi: 0,
        }
    }

    fn merge(&mut self, o: &Self) {
        self.dx.merge(&o.dx);
        self.dy.merge(&o.dy);
        self.dr.merge(&o.dr);
        self.dphi.merge(&o.dphi);
        self.dr_q.merge(&o.dr_q);
        self.dphi_q.merge(&o.dphi_q);
        self.dt.merge(&o.dt);
        self.dt_q.merge(&o.dt_q);
        self.dt_phase.merge(&o.dt_phase);
        self.dr_static.merge(&o.dr_static);
        self.dphi_static.merge(&o.dphi_static);
        self.dx_static.merge(&o.dx_static);
        self.dy_static.merge(&o.dy_static);
        self.outliers += o.outliers;
        self.gross_phi += o.gross_phi;
    }

    fn json(&mut self) -> Value {
        json!({
            "capture_offset_s": {
                "stats": self.dt.summary(),
                "quantiles": self.dt_q.summary(),
                "phase_hist_periods": self.dt_phase.summary(),
            },
            "robot_pos_offset_m": {
                "dx": self.dx.summary(),
                "dy": self.dy.summary(),
                "distance": self.dr.summary(),
                "distance_quantiles": self.dr_q.summary(),
            },
            "robot_phi_offset_rad": { "stats": self.dphi.summary(), "quantiles": self.dphi_q.summary() },
            "static_only": {
                "dx": self.dx_static.summary(),
                "dy": self.dy_static.summary(),
                "distance": self.dr_static.summary(),
                "dphi": self.dphi_static.summary(),
            },
            "rejected_as_mismatch": self.outliers,
            "gross_orientation_disagreements_gt_0p5rad": self.gross_phi,
        })
    }
}

/// Log-wide accumulators that are not per camera.
#[derive(Debug, Clone)]
struct GlobalAcc {
    frames_multi_ball: u64,
    multi_ball_unattributed: u64,
    extra_balls: u64,
    extra_near_robot: u64,
    extra_dist_hist: Hist,
    extra_fwd: Welford,
    extra_lat: Welford,
    extra_fwd_hist: Hist,
    extra_lat_hist: Hist,
    extra_area: Welford,
    real_ball_area: Welford,
    robot_frames: u64,
    robot_seconds: f64,
    ref_time: BTreeMap<i32, f64>,
    ref_switches: u64,
    static_frames: u64,
    processed_frames: u64,
    tracker_lag: Welford,
    tracker_lag_q: Quantiles,
    geometry_packets: u64,
    geometry_changes: u64,
    geometry_calibs_per_packet: Hist,
    geometry_interval: Welford,
    last_geometry_t: f64,
}

impl GlobalAcc {
    fn new() -> Self {
        Self {
            frames_multi_ball: 0,
            multi_ball_unattributed: 0,
            extra_balls: 0,
            extra_near_robot: 0,
            extra_dist_hist: Hist::new(0.0, 0.02, 40),
            extra_fwd: Welford::default(),
            extra_lat: Welford::default(),
            extra_fwd_hist: Hist::new(-0.10, 0.01, 40),
            extra_lat_hist: Hist::new(-0.20, 0.01, 40),
            extra_area: Welford::default(),
            real_ball_area: Welford::default(),
            robot_frames: 0,
            robot_seconds: 0.0,
            ref_time: BTreeMap::new(),
            ref_switches: 0,
            static_frames: 0,
            processed_frames: 0,
            tracker_lag: Welford::default(),
            tracker_lag_q: Quantiles::new(200_000),
            geometry_packets: 0,
            geometry_changes: 0,
            geometry_calibs_per_packet: Hist::new(-0.5, 1.0, 10),
            geometry_interval: Welford::default(),
            last_geometry_t: 0.0,
        }
    }

    fn merge(&mut self, o: &Self) {
        self.frames_multi_ball += o.frames_multi_ball;
        self.multi_ball_unattributed += o.multi_ball_unattributed;
        self.extra_balls += o.extra_balls;
        self.extra_near_robot += o.extra_near_robot;
        self.extra_dist_hist.merge(&o.extra_dist_hist);
        self.extra_fwd.merge(&o.extra_fwd);
        self.extra_lat.merge(&o.extra_lat);
        self.extra_fwd_hist.merge(&o.extra_fwd_hist);
        self.extra_lat_hist.merge(&o.extra_lat_hist);
        self.extra_area.merge(&o.extra_area);
        self.real_ball_area.merge(&o.real_ball_area);
        self.robot_frames += o.robot_frames;
        self.robot_seconds += o.robot_seconds;
        for (k, v) in &o.ref_time {
            *self.ref_time.entry(*k).or_default() += v;
        }
        self.ref_switches += o.ref_switches;
        self.static_frames += o.static_frames;
        self.processed_frames += o.processed_frames;
        self.tracker_lag.merge(&o.tracker_lag);
        self.tracker_lag_q.merge(&o.tracker_lag_q);
        self.geometry_packets += o.geometry_packets;
        self.geometry_changes += o.geometry_changes;
        self.geometry_calibs_per_packet
            .merge(&o.geometry_calibs_per_packet);
        self.geometry_interval.merge(&o.geometry_interval);
    }

    fn json(&mut self) -> Value {
        let rate = if self.robot_seconds > 0.0 {
            self.extra_near_robot as f64 / self.robot_seconds
        } else {
            0.0
        };
        json!({
            "frames_with_multiple_balls": self.frames_multi_ball,
            "multi_ball_frames_without_tracker_match": self.multi_ball_unattributed,
            "extra_ball_detections": self.extra_balls,
            "extra_balls_within_0p3m_of_a_robot": self.extra_near_robot,
            "extra_ball_dist_to_kicker_face_hist": self.extra_dist_hist.summary(),
            "extra_ball_robot_frame_forward_m": self.extra_fwd.summary(),
            "extra_ball_robot_frame_lateral_m": self.extra_lat.summary(),
            "extra_ball_forward_hist": self.extra_fwd_hist.summary(),
            "extra_ball_lateral_hist": self.extra_lat_hist.summary(),
            "extra_ball_area_px": self.extra_area.summary(),
            "tracked_ball_area_px": self.real_ball_area.summary(),
            "robot_camera_frames": self.robot_frames,
            "robot_seconds_observed": r(self.robot_seconds),
            "spurious_ball_rate_per_robot_per_s": r(rate),
            "referee_seconds_per_command": self.ref_time.iter().map(|(k, v)| (k.to_string(), r(*v))).collect::<BTreeMap<_, _>>(),
            "referee_command_switches": self.ref_switches,
            "processed_camera_frames": self.processed_frames,
            "frames_under_static_referee": self.static_frames,
            "tracker_timestamp_minus_t_capture": {
                "stats": self.tracker_lag.summary(),
                "quantiles": self.tracker_lag_q.summary(),
            },
            "geometry_packets": self.geometry_packets,
            "geometry_changes": self.geometry_changes,
            "geometry_calibs_per_packet": self.geometry_calibs_per_packet.summary(),
            "geometry_interval_s": self.geometry_interval.summary(),
        })
    }
}

/// Camera calibration as read from the geometry packet (metres).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Calib {
    id: u32,
    focal: f64,
    ppx: f64,
    ppy: f64,
    distortion: f64,
    q: [f64; 4],
    t: [f64; 3],
    world: Option<[f64; 3]>,
    img: (u32, u32),
}

impl Calib {
    fn nadir(&self) -> Option<(f64, f64, f64)> {
        self.world.map(|w| (w[0], w[1], w[2]))
    }

    fn json(&self) -> Value {
        json!({
            "camera_id": self.id,
            "focal_length_px": r(self.focal),
            "principal_point_px": [r(self.ppx), r(self.ppy)],
            "distortion": r(self.distortion),
            "q": self.q.map(r),
            "t_m": self.t.map(r),
            "derived_camera_world_m": self.world.map(|w| w.map(r)),
            "image_px": [self.img.0, self.img.1],
        })
    }
}

// ---------------------------------------------------------------------------
// per-log analyzer
// ---------------------------------------------------------------------------

/// Everything measured for one log.
struct LogAcc {
    file: String,
    cams: Vec<CamAcc>,
    cam_ids: Vec<u32>,
    pairs: BTreeMap<(usize, usize), PairAcc>,
    global: GlobalAcc,
    calib: BTreeMap<u32, Calib>,
    field: Option<(i32, i32, i32, i32, i32)>,
    tracker_sources: BTreeMap<String, u64>,
    primary_tracker: Option<String>,
    span: f64,
    records: u64,
}

struct Analyzer {
    acc: LogAcc,
    windows: Vec<VecDeque<RawFrame>>,
    wnext: Vec<usize>,
    pool: Vec<RawFrame>,
    trk: VecDeque<TrkFrame>,
    trk_pool: Vec<TrkFrame>,
    now: f64,
    /// Per camera, per slot: last processed-frame index where the robot was seen.
    last_seen: Vec<[u64; 32]>,
    /// Per camera: last processed-frame index where the ball was seen.
    ball_last_seen: Vec<u64>,
    proc_idx: Vec<u64>,
    /// Per camera, per slot: rolling noise window.
    nwin: Vec<Vec<NoiseWin>>,
    /// Per camera: ball noise window.
    bwin: Vec<NoiseWin>,
    ref_cmd: i32,
    ref_since: f64,
    ref_last_t: f64,
    nominal_period: f64,
    tracker_probe: BTreeMap<String, (u64, u64, bool)>,
    tracker_decided: bool,
}

impl Analyzer {
    fn new(file: String) -> Self {
        Self {
            acc: LogAcc {
                file,
                cams: Vec::new(),
                cam_ids: Vec::new(),
                pairs: BTreeMap::new(),
                global: GlobalAcc::new(),
                calib: BTreeMap::new(),
                field: None,
                tracker_sources: BTreeMap::new(),
                primary_tracker: None,
                span: 0.0,
                records: 0,
            },
            windows: Vec::new(),
            wnext: Vec::new(),
            pool: Vec::new(),
            trk: VecDeque::new(),
            trk_pool: Vec::new(),
            now: f64::NEG_INFINITY,
            last_seen: Vec::new(),
            ball_last_seen: Vec::new(),
            proc_idx: Vec::new(),
            nwin: Vec::new(),
            bwin: Vec::new(),
            ref_cmd: -1,
            ref_since: 0.0,
            ref_last_t: 0.0,
            nominal_period: 1.0 / 63.0,
            tracker_probe: BTreeMap::new(),
            tracker_decided: false,
        }
    }

    fn cam_index(&mut self, id: u32) -> usize {
        if let Some(i) = self.acc.cam_ids.iter().position(|c| *c == id) {
            return i;
        }
        self.acc.cam_ids.push(id);
        self.acc.cams.push(CamAcc::new(id));
        self.windows.push(VecDeque::new());
        self.wnext.push(0);
        self.last_seen.push([u64::MAX; 32]);
        self.ball_last_seen.push(u64::MAX);
        self.proc_idx.push(0);
        self.nwin.push(vec![NoiseWin::default(); 32]);
        self.bwin.push(NoiseWin::default());
        self.acc.cam_ids.len() - 1
    }

    fn take_frame(&mut self) -> RawFrame {
        match self.pool.pop() {
            Some(mut f) => {
                f.clear();
                f
            }
            None => RawFrame::default(),
        }
    }
}

/// Analyse one log file.
fn analyse(path: &Path) -> Result<LogAcc> {
    let mut reader = LogReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut a = Analyzer::new(path.display().to_string());
    let mut first_ns: Option<i64> = None;
    let mut last_ns = 0i64;

    while let Some(entry) = reader.next_entry()? {
        first_ns.get_or_insert(entry.time_ns);
        last_ns = entry.time_ns;
        let recv_s = entry.time_ns as f64 * 1e-9;
        match entry.record {
            Record::Referee(rf) => {
                if a.ref_cmd < 0 {
                    a.ref_cmd = rf.command;
                    a.ref_since = recv_s;
                    a.ref_last_t = recv_s;
                }
                let dt = (recv_s - a.ref_last_t).clamp(0.0, 1.0);
                *a.acc.global.ref_time.entry(a.ref_cmd).or_default() += dt;
                a.ref_last_t = recv_s;
                if rf.command != a.ref_cmd {
                    a.ref_cmd = rf.command;
                    a.ref_since = recv_s;
                    a.acc.global.ref_switches += 1;
                }
            }
            Record::Tracker(t) => {
                let key = format!(
                    "{} [{}]",
                    t.source_name.clone().unwrap_or_else(|| "?".into()),
                    &t.uuid[..8.min(t.uuid.len())]
                );
                let Some(f) = t.tracked_frame else { continue };
                *a.acc.tracker_sources.entry(key.clone()).or_default() += 1;
                if !a.tracker_decided {
                    // Some sources timestamp with a monotonic uptime clock instead
                    // of the vision PC's unix clock; only a source on the same
                    // clock as `t_capture` can be used as a time reference.
                    let e = a.tracker_probe.entry(key.clone()).or_insert((0, 0, false));
                    e.0 += 1;
                    if a.now.is_finite() && (f.timestamp - a.now).abs() < 5.0 {
                        e.1 += 1;
                    }
                    e.2 |= f.capabilities.contains(&1);
                    if a.tracker_probe.values().map(|v| v.0).sum::<u64>() >= 400 {
                        let best = a
                            .tracker_probe
                            .iter()
                            .max_by_key(|(_, v)| (v.1 * 2 >= v.0, v.2, v.0))
                            .map(|(k, _)| k.clone());
                        a.acc.primary_tracker = best;
                        a.tracker_decided = true;
                    }
                    continue;
                }
                if a.acc.primary_tracker.as_deref() != Some(key.as_str()) {
                    continue;
                }
                let mut tf = a.trk_pool.pop().unwrap_or_default();
                tf.robots.clear();
                tf.t = f.timestamp;
                tf.has_ball = false;
                tf.bx = 0.0;
                tf.by = 0.0;
                tf.bz = 0.0;
                tf.bv = 0.0;
                if let Some(b) = f.balls.first() {
                    tf.has_ball = true;
                    tf.bx = b.pos.x as f64;
                    tf.by = b.pos.y as f64;
                    tf.bz = b.pos.z as f64;
                    tf.bv = b
                        .vel
                        .as_ref()
                        .map(|v| ((v.x * v.x + v.y * v.y) as f64).sqrt())
                        .unwrap_or(f64::NAN);
                }
                for rb in &f.robots {
                    let team = if rb.robot_id.team_color == 2 { 0 } else { 1 };
                    let id = rb.robot_id.id;
                    if id >= 16 {
                        continue;
                    }
                    tf.robots.push(TrkRobot {
                        slot: team * 16 + id as usize,
                        x: rb.pos.x as f64,
                        y: rb.pos.y as f64,
                        phi: rb.orientation as f64,
                        vx: rb.vel.as_ref().map(|v| v.x as f64).unwrap_or(0.0),
                        vy: rb.vel.as_ref().map(|v| v.y as f64).unwrap_or(0.0),
                    });
                }
                a.trk.push_back(tf);
                while a.trk.len() > 2 && a.trk[0].t < a.now - 1.2 {
                    let old = a.trk.pop_front().unwrap();
                    a.trk_pool.push(old);
                }
            }
            Record::Vision(p) => {
                if let Some(g) = p.geometry {
                    a.acc.global.geometry_packets += 1;
                    a.acc
                        .global
                        .geometry_calibs_per_packet
                        .push(g.calib.len() as f64);
                    if a.acc.global.last_geometry_t > 0.0 {
                        let dt = recv_s - a.acc.global.last_geometry_t;
                        if (0.0..60.0).contains(&dt) {
                            a.acc.global.geometry_interval.push(dt);
                        }
                    }
                    a.acc.global.last_geometry_t = recv_s;
                    a.acc.field = Some((
                        g.field.field_length,
                        g.field.field_width,
                        g.field.goal_width,
                        g.field.boundary_width,
                        g.field.penalty_area_depth.unwrap_or(0),
                    ));
                    for c in &g.calib {
                        let cal = Calib {
                            id: c.camera_id,
                            focal: c.focal_length as f64,
                            ppx: c.principal_point_x as f64,
                            ppy: c.principal_point_y as f64,
                            distortion: c.distortion as f64,
                            q: [c.q0 as f64, c.q1 as f64, c.q2 as f64, c.q3 as f64],
                            t: [c.tx as f64 * 1e-3, c.ty as f64 * 1e-3, c.tz as f64 * 1e-3],
                            world: match (
                                c.derived_camera_world_tx,
                                c.derived_camera_world_ty,
                                c.derived_camera_world_tz,
                            ) {
                                (Some(x), Some(y), Some(z)) => {
                                    Some([x as f64 * 1e-3, y as f64 * 1e-3, z as f64 * 1e-3])
                                }
                                _ => None,
                            },
                            img: (
                                c.pixel_image_width.unwrap_or(0),
                                c.pixel_image_height.unwrap_or(0),
                            ),
                        };
                        match a.acc.calib.get(&cal.id) {
                            Some(old) if *old != cal => {
                                a.acc.global.geometry_changes += 1;
                                a.acc.calib.insert(cal.id, cal);
                            }
                            None => {
                                a.acc.calib.insert(cal.id, cal);
                            }
                            _ => {}
                        }
                    }
                }
                let Some(d) = p.detection else { continue };
                let ci = a.cam_index(d.camera_id);
                let mut f = a.take_frame();
                f.t_capture = d.t_capture;
                f.t_sent = d.t_sent;
                f.recv_s = recv_s;
                f.frame_number = d.frame_number;
                f.ref_static =
                    matches!(a.ref_cmd, 0 | 12 | 13) && recv_s - a.ref_since > REF_STATIC_SETTLE;
                for b in &d.balls {
                    f.balls.push(RawBall {
                        x: b.x as f64 * 1e-3,
                        y: b.y as f64 * 1e-3,
                        z: b.z.map(|z| z as f64 * 1e-3),
                        area: b.area,
                        conf: b.confidence as f64,
                    });
                }
                let mut dupes = 0u64;
                for (team, list) in [(0usize, &d.robots_blue), (1usize, &d.robots_yellow)] {
                    for rb in list.iter() {
                        let Some(id) = rb.robot_id else { continue };
                        if id >= 16 {
                            continue;
                        }
                        let slot = team * 16 + id as usize;
                        if f.mask & (1 << slot) != 0 {
                            dupes += 1;
                        }
                        f.mask |= 1 << slot;
                        f.robots.push(RawRobot {
                            slot,
                            x: rb.x as f64 * 1e-3,
                            y: rb.y as f64 * 1e-3,
                            phi: rb.orientation.unwrap_or(0.0) as f64,
                            conf: rb.confidence as f64,
                            height: rb.height.unwrap_or(0.0) as f64,
                        });
                    }
                }
                a.acc.cams[ci].robot_dupes += dupes;
                a.now = a.now.max(d.t_capture);
                a.windows[ci].push_back(f);
                drive(&mut a, false);
            }
            Record::Other { .. } => {}
        }
    }
    // flush
    a.now = f64::INFINITY;
    drive(&mut a, true);

    a.acc.span = (last_ns - first_ns.unwrap_or(last_ns)) as f64 * 1e-9;
    a.acc.records = reader.count;
    Ok(a.acc)
}

/// Process every frame that has settled.
fn drive(a: &mut Analyzer, flush: bool) {
    loop {
        let mut did = false;
        for ci in 0..a.windows.len() {
            loop {
                let i = a.wnext[ci];
                let w = &a.windows[ci];
                if i + 1 >= w.len() {
                    break;
                }
                if !flush && w[i].t_capture > a.now - SETTLE {
                    break;
                }
                process_frame(a, ci, i);
                a.wnext[ci] += 1;
                did = true;
            }
            // retire old frames
            while a.wnext[ci] > 1 && (flush || a.windows[ci][0].t_capture < a.now - RETAIN) {
                let old = a.windows[ci].pop_front().unwrap();
                a.pool.push(old);
                a.wnext[ci] -= 1;
            }
        }
        if !did {
            break;
        }
    }
}

#[allow(clippy::too_many_lines)]
fn process_frame(a: &mut Analyzer, ci: usize, i: usize) {
    let Analyzer {
        acc,
        windows,
        trk,
        last_seen,
        ball_last_seen,
        proc_idx,
        nwin,
        bwin,
        nominal_period,
        ..
    } = a;
    let w = &windows[ci];
    let cur = &w[i];
    let prev = if i > 0 { Some(&w[i - 1]) } else { None };
    let next = &w[i + 1];
    let cam = &mut acc.cams[ci];
    let idx = proc_idx[ci];
    proc_idx[ci] = idx + 1;

    // --- timing ---
    if cam.frames == 0 {
        cam.t_first = cur.t_capture;
    }
    cam.t_last = cur.t_capture;
    cam.frames += 1;
    acc.global.processed_frames += 1;
    if cur.ref_static {
        acc.global.static_frames += 1;
    }
    if let Some(p) = prev {
        let gap = cur.t_capture - p.t_capture;
        cam.gap.push(gap);
        cam.gap_q.push(gap);
        if gap < 0.0 {
            cam.gap_neg += 1;
        } else if gap == 0.0 {
            cam.gap_zero += 1;
        }
        if cam.gap.n > 200 {
            *nominal_period = cam.gap.mean().clamp(1.0 / 200.0, 1.0 / 20.0);
        }
        if gap > 1.8 * *nominal_period {
            cam.gap_long += 1;
        }
        if gap > 0.0 && gap < 3.0 * *nominal_period {
            cam.gap_clean.push(gap);
        }
        cam.fn_step
            .push(cur.frame_number as f64 - p.frame_number as f64);
    }
    cam.proc.push(cur.t_sent - cur.t_capture);
    cam.proc_q.push(cur.t_sent - cur.t_capture);
    cam.transport.push(cur.recv_s - cur.t_sent);
    cam.transport_q.push(cur.recv_s - cur.t_sent);

    // --- detections ---
    if cur.balls.is_empty() && cur.robots.is_empty() {
        cam.empty_frames += 1;
    }
    cam.robot_dets += cur.robots.len() as u64;
    cam.ball_dets += cur.balls.len() as u64;
    cam.balls_per_frame.push(cur.balls.len() as f64);
    acc.global.robot_frames += cur.robots.len() as u64;
    acc.global.robot_seconds += cur.robots.len() as f64 * *nominal_period;
    for rb in &cur.robots {
        cam.robot_conf.push(rb.conf);
        cam.robot_conf_h.push(rb.conf);
        cam.robot_height.push(rb.height);
        cam.grid.push(rb.x, rb.y);
    }
    for b in &cur.balls {
        cam.ball_conf.push(b.conf);
        cam.ball_conf_h.push(b.conf);
        if b.z.is_some() {
            cam.ball_z_reported += 1;
        }
        if b.area.is_some() {
            cam.ball_area_present += 1;
        } else {
            cam.ball_area_missing += 1;
        }
    }

    // --- tracker frame nearest this capture instant ---
    let tk = nearest_tracker(trk, cur.t_capture);
    if let Some(t) = tk {
        let lag = t.t - cur.t_capture;
        acc.global.tracker_lag.push(lag);
        acc.global.tracker_lag_q.push(lag);
    }
    let nadir = acc.calib.get(&cam.id).and_then(Calib::nadir);

    // --- robot dropouts (present in prev and next, missing in cur) ---
    if let Some(p) = prev {
        let both = p.mask & next.mask;
        let missing = both & !cur.mask;
        for slot in 0..32u32 {
            if both & (1 << slot) != 0 {
                cam.robot_single_drop.push(missing & (1 << slot) != 0);
            }
        }
        // ball
        let ball_both = !p.balls.is_empty() && !next.balls.is_empty();
        if ball_both {
            let miss = cur.balls.is_empty();
            cam.ball_single_drop.push(miss);
            if let Some(t) = tk {
                if t.has_ball {
                    let d = nearest_robot_dist(t, t.bx, t.by);
                    cam.ball_drop_vs_robot_dist.push(d, miss);
                }
            }
        }
    }
    // gap runs
    for rb in &cur.robots {
        let last = last_seen[ci][rb.slot];
        if last != u64::MAX && idx > last + 1 {
            cam.robot_gap.push((idx - last - 1) as f64);
        }
        last_seen[ci][rb.slot] = idx;
    }
    if !cur.balls.is_empty() {
        let last = ball_last_seen[ci];
        if last != u64::MAX && idx > last + 1 && idx - last - 1 < 60 {
            cam.ball_gap.push((idx - last - 1) as f64);
        }
        ball_last_seen[ci] = idx;
    }

    // --- robot noise windows ---
    for rb in &cur.robots {
        let win = &mut nwin[ci][rb.slot];
        if !win.t.is_empty() && cur.t_capture - win.last_t > NOISE_MAX_GAP {
            win.reset();
        }
        if !win.t.is_empty() && cur.t_capture - win.t[0] >= NOISE_WIN {
            flush_robot_window(win, cam, nadir);
            win.reset();
        }
        if win.t.is_empty() {
            win.all_static = true;
        }
        let phi = match win.p.last() {
            Some(prev_phi) => unwrap_angle(*prev_phi, rb.phi),
            None => rb.phi,
        };
        win.t.push(cur.t_capture);
        win.x.push(rb.x);
        win.y.push(rb.y);
        win.p.push(phi);
        win.last_t = cur.t_capture;
        win.all_static &= cur.ref_static;
        if let Some(n) = nadir {
            win.nadir += (((rb.x - n.0).powi(2) + (rb.y - n.1).powi(2)).sqrt() - win.nadir)
                / win.t.len() as f64;
        }
    }

    // --- ball: match to the tracked ball, noise, area, extras ---
    let mut real_ball: Option<usize> = None;
    if let Some(t) = tk {
        if t.has_ball && !cur.balls.is_empty() {
            let mut best = (f64::INFINITY, 0usize);
            for (bi, b) in cur.balls.iter().enumerate() {
                let d = ((b.x - t.bx).powi(2) + (b.y - t.by).powi(2)).sqrt();
                if d < best.0 {
                    best = (d, bi);
                }
            }
            if best.0 < BALL_MATCH {
                real_ball = Some(best.1);
            }
        }
    }

    if let (Some(bi), Some(t)) = (real_ball, tk) {
        let b = cur.balls[bi];
        if let Some(area) = b.area {
            let area = f64::from(area);
            acc.global.real_ball_area.push(area);
            cam.area.push(area);
            if let Some(n) = nadir {
                let dxy = ((b.x - n.0).powi(2) + (b.y - n.1).powi(2)).sqrt();
                let z = t.bz.max(0.0) + BALL_RADIUS;
                let d = (dxy * dxy + (n.2 - z).powi(2)).sqrt();
                cam.area_k.push(area * d * d);
                cam.area_vs_nadir.push(dxy, area);
                cam.area_k_vs_z.push(t.bz.max(0.0), area * d * d);
                cam.area_vs_z.push(t.bz.max(0.0), area);
            }
        }
        // ball at rest: detrended noise window
        let at_rest = t.bv.is_finite() && t.bv < BALL_REST_SPEED && t.bz.abs() < 0.02;
        let win = &mut bwin[ci];
        if !at_rest || (!win.t.is_empty() && cur.t_capture - win.last_t > NOISE_MAX_GAP) {
            if !win.t.is_empty() {
                flush_ball_window(win, cam, nadir);
            }
            win.reset();
        }
        if at_rest {
            if !win.t.is_empty() && cur.t_capture - win.t[0] >= NOISE_WIN {
                flush_ball_window(win, cam, nadir);
                win.reset();
            }
            win.t.push(cur.t_capture);
            win.x.push(b.x);
            win.y.push(b.y);
            if let Some(area) = b.area {
                win.a.push(f64::from(area));
            }
            win.last_t = cur.t_capture;
            if let Some(n) = nadir {
                win.nadir += (((b.x - n.0).powi(2) + (b.y - n.1).powi(2)).sqrt() - win.nadir)
                    / win.t.len() as f64;
            }
        }
    }

    // extra balls (spurious detections)
    if cur.balls.len() > 1 {
        acc.global.frames_multi_ball += 1;
        if real_ball.is_none() {
            acc.global.multi_ball_unattributed += 1;
        }
        for (bi, b) in cur.balls.iter().enumerate() {
            if real_ball.is_none() {
                break;
            }
            if Some(bi) == real_ball {
                continue;
            }
            acc.global.extra_balls += 1;
            if let Some(area) = b.area {
                acc.global.extra_area.push(f64::from(area));
            }
            // nearest robot in this same raw frame
            let mut best = (f64::INFINITY, 0usize);
            for (ri, rb) in cur.robots.iter().enumerate() {
                let d = ((b.x - rb.x).powi(2) + (b.y - rb.y).powi(2)).sqrt();
                if d < best.0 {
                    best = (d, ri);
                }
            }
            if best.0 < 0.3 && !cur.robots.is_empty() {
                let rb = cur.robots[best.1];
                let (s, c) = rb.phi.sin_cos();
                let dx = b.x - rb.x;
                let dy = b.y - rb.y;
                let fwd = dx * c + dy * s;
                let lat = -dx * s + dy * c;
                acc.global.extra_near_robot += 1;
                acc.global.extra_fwd.push(fwd);
                acc.global.extra_lat.push(lat);
                acc.global.extra_fwd_hist.push(fwd);
                acc.global.extra_lat_hist.push(lat);
                // distance to the kicker-face centre
                let fx = rb.x + c * CENTER_TO_DRIBBLER;
                let fy = rb.y + s * CENTER_TO_DRIBBLER;
                acc.global
                    .extra_dist_hist
                    .push(((b.x - fx).powi(2) + (b.y - fy).powi(2)).sqrt());
            }
        }
    }

    // --- region-gated completeness and per-camera systematic offset ---
    if let Some(t) = tk {
        let dt = cur.t_capture - t.t;
        for tr in &t.robots {
            let px = tr.x + tr.vx * dt;
            let py = tr.y + tr.vy * dt;
            if owns_point(&acc.calib, &acc.cam_ids, ci, px, py) == Some(true) {
                let seen = cur.mask & (1 << tr.slot) != 0;
                cam.robot_seen_in_region.push(seen);
                if let Some(n) = nadir {
                    let dn = ((px - n.0).powi(2) + (py - n.1).powi(2)).sqrt();
                    cam.robot_seen_vs_nadir.push(dn, seen);
                    if dn < CORE_NADIR {
                        cam.robot_seen_core.push(seen);
                    }
                }
            }
            if let Some(rb) = cur.robots.iter().find(|r| r.slot == tr.slot) {
                let bx = rb.x - px;
                let by = rb.y - py;
                if bx.hypot(by) > MATCH_OUTLIER {
                    continue; // tracker/detection mismatch, not a vision offset
                }
                cam.bias_x.push(bx);
                cam.bias_y.push(by);
                cam.bias_phi.push(wrap_pi(rb.phi - tr.phi));
                if let Some(n) = nadir {
                    let rx = px - n.0;
                    let ry = py - n.1;
                    let rl = (rx * rx + ry * ry).sqrt();
                    if rl > 0.2 {
                        let rad = (bx * rx + by * ry) / rl;
                        cam.bias_radial.push(rad);
                        cam.bias_tangential.push((-bx * ry + by * rx) / rl);
                        cam.bias_radial_vs_nadir.push(rl, rad);
                    }
                }
            }
        }
        if t.has_ball
            && t.bz.abs() < 0.05
            && owns_point(&acc.calib, &acc.cam_ids, ci, t.bx, t.by) == Some(true)
        {
            let seen = cur
                .balls
                .iter()
                .any(|b| (b.x - t.bx).powi(2) + (b.y - t.by).powi(2) < BALL_MATCH * BALL_MATCH);
            cam.ball_seen_in_region.push(seen);
            if let Some(n) = nadir {
                let dn = ((t.bx - n.0).powi(2) + (t.by - n.1).powi(2)).sqrt();
                cam.ball_seen_vs_nadir.push(dn, seen);
                if dn < CORE_NADIR {
                    cam.ball_seen_core.push(seen);
                }
            }
        }
    }

    // --- ball visibility vs occlusion ---
    if let Some(t) = tk {
        if t.has_ball && t.bz.abs() < 0.05 {
            // Is this camera looking at that spot? It must own the region (by the
            // Manhattan rule on the calibrated nadirs) and have reported a robot
            // near the ball this frame, so a miss really is an occlusion.
            let owns = owns_point(&acc.calib, &acc.cam_ids, ci, t.bx, t.by);
            let covered = match (owns, nadir) {
                // calibration known: the camera owns the region and the ball is
                // well inside the field of view, so a miss is a real miss
                (Some(o), Some(n)) => {
                    o && ((t.bx - n.0).powi(2) + (t.by - n.1).powi(2)).sqrt() < CORE_NADIR
                }
                // no calibration: fall back to "this camera reported a robot nearby"
                _ => cur
                    .robots
                    .iter()
                    .any(|rb| (rb.x - t.bx).powi(2) + (rb.y - t.by).powi(2) < 0.6 * 0.6),
            };
            if covered {
                let seen = cur
                    .balls
                    .iter()
                    .any(|b| (b.x - t.bx).powi(2) + (b.y - t.by).powi(2) < BALL_MATCH * BALL_MATCH);
                let d = nearest_robot_dist(t, t.bx, t.by);
                cam.ball_seen_vs_robot_dist.push(d, seen);
                if t.bv.is_finite() {
                    cam.ball_seen_vs_speed.push(t.bv, seen);
                }
                if d > 0.3 {
                    cam.free_ball_seen.push(seen);
                    if t.bv.is_finite() && t.bv < 0.5 {
                        cam.ball_seen_free_at_rest.push(seen);
                    }
                }
                // dribbled: within DRIBBLE_RADIUS of a robot centre and in front of it
                for tr in &t.robots {
                    let dx = t.bx - tr.x;
                    let dy = t.by - tr.y;
                    let dd = (dx * dx + dy * dy).sqrt();
                    if dd > DRIBBLE_RADIUS {
                        continue;
                    }
                    let (s, c) = tr.phi.sin_cos();
                    if dx * c + dy * s > 0.6 * dd && cur.mask & (1 << tr.slot) != 0 {
                        cam.dribbled_ball_seen.push(seen);
                        break;
                    }
                }
            }
        }
    }

    // --- multi-camera disagreement ---
    // handle each pair from the lower-index camera only
    for (cj, other_window) in windows.iter().enumerate().skip(ci + 1) {
        let key = (ci, cj);
        let Some(other) = nearest_frame(other_window, cur.t_capture) else {
            continue;
        };
        let dt = other.t_capture - cur.t_capture;
        let pair = acc.pairs.entry(key).or_insert_with(PairAcc::new);
        pair.dt.push(dt);
        pair.dt_q.push(dt);
        pair.dt_phase.push(dt / *nominal_period);
        if dt.abs() > 0.6 * *nominal_period {
            continue;
        }
        let shared = cur.mask & other.mask;
        if shared == 0 {
            continue;
        }
        for rb in &cur.robots {
            if shared & (1 << rb.slot) == 0 {
                continue;
            }
            let Some(ob) = other.robots.iter().find(|o| o.slot == rb.slot) else {
                continue;
            };
            // compensate for the capture-time offset with the tracked velocity
            let (mut vx, mut vy) = (0.0, 0.0);
            if let Some(t) = tk {
                if let Some(tr) = t.robots.iter().find(|r| r.slot == rb.slot) {
                    vx = tr.vx;
                    vy = tr.vy;
                }
            }
            let dx = ob.x - rb.x - vx * dt;
            let dy = ob.y - rb.y - vy * dt;
            let dphi = wrap_pi(ob.phi - rb.phi);
            if dx.hypot(dy) > MATCH_OUTLIER {
                pair.outliers += 1;
                continue;
            }
            pair.dx.push(dx);
            pair.dy.push(dy);
            let dr = (dx * dx + dy * dy).sqrt();
            pair.dr.push(dr);
            pair.dr_q.push(dr);
            pair.dphi.push(dphi);
            pair.dphi_q.push(dphi);
            if dphi.abs() > 0.5 {
                pair.gross_phi += 1;
            }
            if cur.ref_static {
                pair.dx_static.push(dx);
                pair.dy_static.push(dy);
                pair.dr_static.push(dr);
                pair.dphi_static.push(dphi);
            }
        }
    }
}

fn flush_robot_window(win: &mut NoiseWin, cam: &mut CamAcc, nadir: Option<(f64, f64, f64)>) {
    if win.t.len() < NOISE_MIN_N {
        return;
    }
    let disp = ((win.x[win.x.len() - 1] - win.x[0]).powi(2)
        + (win.y[win.y.len() - 1] - win.y[0]).powi(2))
    .sqrt();
    if disp > SLOW_DISP {
        return;
    }
    let (Some((sx, rx)), Some((sy, ry)), Some((_, rp))) = (
        detrend(&win.t, &win.x),
        detrend(&win.t, &win.y),
        detrend(&win.t, &win.p),
    ) else {
        return;
    };
    if sx.abs() > SLOW_SPEED || sy.abs() > SLOW_SPEED {
        return;
    }
    cam.noise_x.push(rx);
    cam.noise_y.push(ry);
    cam.noise_phi.push(rp);
    if nadir.is_some() {
        cam.noise_vs_nadir_p
            .push(win.nadir, (rx * rx + ry * ry).sqrt() / 2f64.sqrt());
        cam.noise_vs_nadir_phi.push(win.nadir, rp);
    }
    if win.all_static && disp < STATIC_DISP {
        cam.noise_x_static.push(rx);
        cam.noise_y_static.push(ry);
        cam.noise_phi_static.push(rp);
    }
}

fn flush_ball_window(win: &mut NoiseWin, cam: &mut CamAcc, nadir: Option<(f64, f64, f64)>) {
    if win.t.len() < NOISE_MIN_N {
        return;
    }
    let disp = ((win.x[win.x.len() - 1] - win.x[0]).powi(2)
        + (win.y[win.y.len() - 1] - win.y[0]).powi(2))
    .sqrt();
    if disp > SLOW_DISP {
        return;
    }
    let (Some((sx, rx)), Some((sy, ry))) = (detrend(&win.t, &win.x), detrend(&win.t, &win.y))
    else {
        return;
    };
    if sx.abs() > SLOW_SPEED || sy.abs() > SLOW_SPEED {
        return;
    }
    let sigma = ((rx * rx + ry * ry) / 2.0).sqrt();
    cam.ball_noise.push(sigma);
    if win.a.len() == win.t.len() {
        if let Some((_, ra)) = detrend(&win.t, &win.a) {
            cam.ball_area_noise.push(ra);
            let mean = win.a.iter().sum::<f64>() / win.a.len() as f64;
            if mean > 1.0 {
                cam.ball_area_rel_noise.push(ra / mean);
            }
        }
    }
    if nadir.is_some() {
        cam.ball_noise_vs_nadir.push(win.nadir, sigma);
    }
}

fn nearest_tracker(trk: &VecDeque<TrkFrame>, t: f64) -> Option<&TrkFrame> {
    let mut best: Option<(f64, &TrkFrame)> = None;
    for f in trk {
        let d = (f.t - t).abs();
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, f));
        }
    }
    best.filter(|(d, _)| *d < 0.02).map(|(_, f)| f)
}

fn nearest_frame(w: &VecDeque<RawFrame>, t: f64) -> Option<&RawFrame> {
    let mut best: Option<(f64, &RawFrame)> = None;
    for f in w {
        let d = (f.t_capture - t).abs();
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, f));
        }
    }
    best.map(|(_, f)| f)
}

/// Is camera `ci` the Manhattan-nearest camera to `(x, y)`? `None` when any
/// camera's calibrated nadir is not known yet.
fn owns_point(
    calib: &BTreeMap<u32, Calib>,
    cam_ids: &[u32],
    ci: usize,
    x: f64,
    y: f64,
) -> Option<bool> {
    let mut own = f64::INFINITY;
    let mut best = f64::INFINITY;
    for (i, id) in cam_ids.iter().enumerate() {
        let n = calib.get(id).and_then(Calib::nadir)?;
        let d = (n.0 - x).abs() + (n.1 - y).abs();
        if i == ci {
            own = d;
        }
        best = best.min(d);
    }
    Some(own <= best)
}

fn nearest_robot_dist(t: &TrkFrame, x: f64, y: f64) -> f64 {
    let mut best = f64::INFINITY;
    for r in &t.robots {
        let d = ((r.x - x).powi(2) + (r.y - y).powi(2)).sqrt();
        if d < best {
            best = d;
        }
    }
    if best.is_finite() {
        best
    } else {
        10.0
    }
}

fn unwrap_angle(prev: f64, x: f64) -> f64 {
    prev + wrap_pi(x - prev)
}

fn wrap_pi(mut a: f64) -> f64 {
    use std::f64::consts::PI;
    while a > PI {
        a -= 2.0 * PI;
    }
    while a < -PI {
        a += 2.0 * PI;
    }
    a
}

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

/// Overlap analysis derived from the per-camera coverage grids.
fn overlap_json(acc: &LogAcc) -> Value {
    if acc.cams.len() < 2 {
        return json!({ "cameras": acc.cams.len(), "note": "single camera, no overlap" });
    }
    let masks: Vec<Vec<bool>> = acc.cams.iter().map(|c| c.grid.mask(0.02)).collect();
    let grid = &acc.cams[0].grid;
    let mut covered = 0u64;
    let mut multi = 0u64;
    // Manhattan-excess: for cells covered by camera c that are not the
    // Manhattan-nearest camera's cells, how far past the nearest camera are they?
    let mut excess: Vec<f64> = Vec::new();
    let mut band_x: Vec<f64> = Vec::new();
    let nadirs: Vec<Option<(f64, f64, f64)>> = acc
        .cams
        .iter()
        .map(|c| acc.calib.get(&c.id).and_then(Calib::nadir))
        .collect();
    for iy in 0..grid.ny {
        let mut row_multi = 0usize;
        for ix in 0..grid.nx {
            let k = iy * grid.nx + ix;
            let n = masks.iter().filter(|m| m[k]).count();
            if n > 0 {
                covered += 1;
            }
            if n > 1 {
                multi += 1;
                row_multi += 1;
            }
            if n > 0 {
                let (cx, cy) = grid.centre(ix, iy);
                let mut man: Vec<f64> = Vec::with_capacity(nadirs.len());
                let mut ok = true;
                for nd in &nadirs {
                    match nd {
                        Some(p) => man.push((p.0 - cx).abs() + (p.1 - cy).abs()),
                        None => ok = false,
                    }
                }
                if ok {
                    let min = man.iter().cloned().fold(f64::INFINITY, f64::min);
                    for (c, m) in masks.iter().enumerate() {
                        if m[k] {
                            excess.push(man[c] - min);
                        }
                    }
                }
            }
        }
        if row_multi > 0 {
            band_x.push(row_multi as f64 * grid.step);
        }
    }
    excess.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| -> f64 {
        if excess.is_empty() {
            0.0
        } else {
            excess[((excess.len() - 1) as f64 * f) as usize]
        }
    };
    band_x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let bq = |f: f64| -> f64 {
        if band_x.is_empty() {
            0.0
        } else {
            band_x[((band_x.len() - 1) as f64 * f) as usize]
        }
    };
    json!({
        "cameras": acc.cams.len(),
        "covered_cells": covered,
        "cells_covered_by_2plus": multi,
        "overlap_area_fraction": r(multi as f64 / covered.max(1) as f64),
        "overlap_band_width_x_m": {
            "p10": r(bq(0.10)), "p50": r(bq(0.5)), "p90": r(bq(0.9)), "max": r(bq(1.0)),
            "rows": band_x.len(),
        },
        "manhattan_excess_m": {
            "p50": r(q(0.5)), "p90": r(q(0.9)), "p95": r(q(0.95)), "p99": r(q(0.99)), "max": r(q(1.0)),
        },
        "implied_camera_overlap_m_from_band_p50": r(bq(0.5) / 2.0),
        "implied_camera_overlap_m_from_band_p90": r(bq(0.9) / 2.0),
        "implied_camera_overlap_m_from_excess_p95": r(q(0.95) / 2.0),
    })
}

fn log_json(acc: &mut LogAcc) -> Value {
    let overlap = overlap_json(acc);
    let cameras: Vec<Value> = acc.cams.iter_mut().map(|c| c.json(true)).collect();
    let calib: Vec<Value> = acc.calib.values().map(Calib::json).collect();
    let pairs: Vec<Value> = acc
        .pairs
        .iter_mut()
        .map(|((i, j), p)| {
            let mut v = p.json();
            v["cameras"] = json!([acc.cam_ids[*i], acc.cam_ids[*j]]);
            v
        })
        .collect();
    json!({
        "file": acc.file,
        "records": acc.records,
        "logger_span_s": r(acc.span),
        "field_mm": acc.field.map(|f| json!({
            "length": f.0, "width": f.1, "goal_width": f.2,
            "boundary_width": f.3, "penalty_area_depth": f.4,
        })),
        "calibration": calib,
        "tracker_sources": acc.tracker_sources,
        "primary_tracker": acc.primary_tracker,
        "cameras": cameras,
        "camera_pairs": pairs,
        "coverage_overlap": overlap,
        "global": acc.global.json(),
    })
}

/// Fold all logs into one all-cameras accumulator plus one pair accumulator.
fn aggregate(logs: &[LogAcc]) -> (CamAcc, PairAcc, GlobalAcc, Value) {
    let mut cam = CamAcc::new(u32::MAX);
    let mut pair = PairAcc::new();
    let mut global = GlobalAcc::new();
    let mut per_cam_rows = Vec::new();
    for l in logs {
        for c in &l.cams {
            cam.merge(c);
            per_cam_rows.push(json!({
                "file": l.file,
                "camera_id": c.id,
                "frames": c.frames,
                "hz": r(c.hz()),
                "robot_sigma_xy_m": r((c.noise_x.mean().powi(2) + c.noise_y.mean().powi(2)).sqrt() / 2f64.sqrt()),
                "robot_sigma_phi_rad": r(c.noise_phi.mean()),
                "ball_sigma_m": r(c.ball_noise.mean()),
                "robot_drop_rate": r(c.robot_single_drop.ratio()),
                "ball_drop_rate": r(c.ball_single_drop.ratio()),
                "dribbled_ball_seen": r(c.dribbled_ball_seen.ratio()),
                "free_ball_seen": r(c.free_ball_seen.ratio()),
                "area_k_px_m2": r(c.area_k.mean()),
            }));
        }
        for p in l.pairs.values() {
            pair.merge(p);
        }
        global.merge(&l.global);
    }
    (cam, pair, global, json!(per_cam_rows))
}

fn recommended(cam: &mut CamAcc, pair: &mut PairAcc, global: &GlobalAcc, logs: &[LogAcc]) -> Value {
    let sigma_p = ((cam.noise_x.mean().powi(2) + cam.noise_y.mean().powi(2)) / 2.0).sqrt();
    let sigma_phi = cam.noise_phi.mean();
    let sigma_ball = cam.ball_noise.mean();
    let hz = {
        let mut w = Welford::default();
        for l in logs {
            for c in &l.cams {
                if c.frames > 100 {
                    w.push(c.hz());
                }
            }
        }
        w.mean()
    };
    let ncams = {
        let mut w = Welford::default();
        for l in logs {
            w.push(l.cams.len() as f64);
        }
        w.mean()
    };
    let overlap = {
        let mut w = Welford::default();
        for l in logs {
            if l.cams.len() < 2 {
                continue;
            }
            let v = overlap_json(l);
            if let Some(x) = v["implied_camera_overlap_m_from_band_p90"].as_f64() {
                if x > 0.0 {
                    w.push(x);
                }
            }
        }
        w.mean()
    };
    // pairwise disagreement -> per-camera systematic offset (two independent
    // cameras each with offset s give a pairwise rms of sqrt(2) * s)
    let pair_rms = (pair.dx.rms().powi(2) + pair.dy.rms().powi(2)).sqrt();
    let per_cam_offset = pair_rms / 2f64.sqrt();
    let area_k = cam.area_k.mean();
    json!({
        "stddev_robot_p": r(sigma_p),
        "stddev_robot_p_static_referee": r(
            ((cam.noise_x_static.mean().powi(2) + cam.noise_y_static.mean().powi(2)) / 2.0).sqrt()),
        "stddev_robot_phi": r(sigma_phi),
        "stddev_robot_phi_static_referee": r(cam.noise_phi_static.mean()),
        "stddev_ball_p": r(sigma_ball),
        "stddev_ball_area": r(cam.ball_area_noise.mean()),
        "stddev_ball_area_relative": r(cam.ball_area_rel_noise.mean()),
        "mean_ball_area_px": r(cam.area.mean()),
        // The simulator draws an independent Bernoulli per object per frame, so
        // the matching estimator is P(missing | the same camera saw the object in
        // both the previous and the next frame). The remaining rows are the same
        // quantity under looser gates, listed so the choice can be second-guessed.
        "missing_robot_detections": r(cam.robot_single_drop.ratio()),
        "missing_robot_detections_incl_multi_frame_gaps": r(gap_rate(
            cam.robot_gap.weighted_sum(), cam.robot_dets)),
        "missing_robot_detections_core_region_tracker_gated": r(
            1.0 - cam.robot_seen_core.ratio()),
        "missing_robot_detections_whole_region": r(1.0 - cam.robot_seen_in_region.ratio()),
        "missing_ball_detections": r(cam.ball_single_drop.ratio()),
        "missing_ball_detections_incl_multi_frame_gaps": r(gap_rate(
            cam.ball_gap.weighted_sum(), cam.ball_dets)),
        "missing_ball_detections_free_slow_tracker_gated": r(
            1.0 - cam.ball_seen_free_at_rest.ratio()),
        "missing_ball_detections_core_region_any_situation": r(1.0 - cam.ball_seen_core.ratio()),
        "missing_ball_detections_whole_region": r(1.0 - cam.ball_seen_in_region.ratio()),
        "dribbler_ball_detections": r(if global.robot_seconds > 0.0 {
            global.extra_near_robot as f64 / global.robot_seconds
        } else { 0.0 }),
        "camera_overlap": r(overlap),
        "object_position_offset": r(per_cam_offset),
        "object_position_offset_systematic_only": r(
            (pair.dx.mean().powi(2) + pair.dy.mean().powi(2)).sqrt() / 2f64.sqrt()),
        "per_camera_radial_bias_vs_tracker_m": r(cam.bias_radial.mean()),
        "camera_position_error": r(per_cam_offset),
        "vision_delay": r(cam.proc.mean() + UNOBSERVED_CAMERA_LATENCY),
        "vision_delay_note": "only t_sent - t_capture is observable in a log; the camera \
    exposure and readout latency ahead of t_capture and the LAN hop are not, so a conventional \
    15 ms is added",
        "vision_processing_time": r(cam.proc.mean()),
        "frame_rate": r(hz),
        "cameras": r(ncams),
        "ball_visibility_threshold_hint": {
            "dribbled_seen_ratio": r(cam.dribbled_ball_seen.ratio()),
            "free_seen_ratio": r(cam.free_ball_seen.ratio()),
        },
        "focal_length_px_from_area": r(implied_focal(area_k)),
        "area_k_px_m2": r(area_k),
    })
}

/// Entry point.
pub fn run(args: Args) -> Result<()> {
    if args.files.is_empty() {
        anyhow::bail!("no log files given");
    }
    let mut logs = Vec::new();
    for f in &args.files {
        let t0 = std::time::Instant::now();
        let acc = analyse(f)?;
        eprintln!(
            "{}: {} records, {} cameras, {:.1} s, in {:.1} s",
            f.display(),
            acc.records,
            acc.cams.len(),
            acc.span,
            t0.elapsed().as_secs_f64()
        );
        logs.push(acc);
    }

    let (mut cam, mut pair, global, per_cam_rows) = aggregate(&logs);
    let rec = recommended(&mut cam, &mut pair, &global, &logs);

    // stdout summary
    println!("\n=== vision calibration summary ({} logs) ===", logs.len());
    for l in &logs {
        println!(
            "{}: {} cameras, {:.0} s, {} geometry packets ({} changes)",
            Path::new(&l.file)
                .file_name()
                .map_or_else(|| l.file.clone().into(), |s| s.to_owned())
                .to_string_lossy(),
            l.cams.len(),
            l.span,
            l.global.geometry_packets,
            l.global.geometry_changes,
        );
        for c in &l.cams {
            println!(
                "  cam {}: {:.2} Hz  proc {:.2} ms  sigma_p {:.2} mm  sigma_phi {:.2} mrad  \
                 sigma_ball {:.2} mm  drop robot {:.3}% ball {:.3}%  dribbled ball seen {:.1}%",
                c.id,
                c.hz(),
                c.proc.mean() * 1e3,
                ((c.noise_x.mean().powi(2) + c.noise_y.mean().powi(2)) / 2.0).sqrt() * 1e3,
                c.noise_phi.mean() * 1e3,
                c.ball_noise.mean() * 1e3,
                c.robot_single_drop.ratio() * 100.0,
                c.ball_single_drop.ratio() * 100.0,
                c.dribbled_ball_seen.ratio() * 100.0,
            );
        }
    }
    println!("\nrecommended: {}", serde_json::to_string_pretty(&rec)?);

    if let Some(out) = &args.out {
        let per_log: Vec<Value> = logs.iter_mut().map(log_json).collect();
        let doc = json!({
            "tool": "ssl-logtool vision",
            "note": "units: metres, seconds, radians, pixels (raw wire units are mm/s and converted on read)",
            "method": {
                "noise": "per (camera, object) 1 s windows of consecutive detections, \
                          linear-detrended; residual std is the per-frame noise. \
                          'slow' = fitted speed < 30 mm/s and < 20 mm end-to-end travel; \
                          'static' additionally requires referee HALT/TIMEOUT for > 2 s and < 5 mm travel.",
                "dropouts": "a robot/ball detected by a camera in both the previous and the next \
                             frame of that same camera but missing in between.",
                "reference": "the tracker stream (autoref/team tracker) is used only to identify \
                              the real ball, robot velocities and ball height; it is itself a \
                              filtered estimate, so it is never used as a position ground truth.",
            },
            "logs": per_log,
            "aggregate": {
                "all_cameras": cam.json(false),
                "all_camera_pairs": pair.json(),
                "global": { "merged": GlobalAcc::json(&mut global.clone()) },
                "per_camera_rows": per_cam_rows,
            },
            "recommended": rec,
        });
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(out, serde_json::to_string_pretty(&doc)?)
            .with_context(|| format!("write {}", out.display()))?;
        println!("\nwrote {}", out.display());
    }
    Ok(())
}

//! Vision model: camera regions, floor projection of the ball, occlusion,
//! static calibration warp, noise, dropouts, spurious detections, outages and
//! the delay queue.
//!
//! The model is calibrated against ten 2026 division-A/B game logs; every
//! number it uses comes from [`VisionConfig`] or [`Realism`], and the evidence
//! for each default is in `docs/calibration/vision.md` (section references are
//! quoted at each item below). See `docs/design.md` §5.5.
//!
//! # Per-camera capture instants
//!
//! Real rigs are either hardware-locked with a small per-camera offset or
//! completely free-running (vision.md §1, §9.1), so cameras do **not** capture
//! at one instant here either. Each camera keeps its own schedule and each
//! capture produces its own [`VisionOutput`] carrying exactly one
//! [`DetectionFrame`]; the geometry packet rides on camera 0's outputs.
//! Capture instants are quantised to the substep the world is stepped with,
//! which at the 1 ms design substep resolves a 13.6 ms frame period to ~7 %.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::ball::Ball;
use crate::field::FieldGeometry;
use crate::params::{BallParams, CameraConfig, CameraPhase, Realism, VisionConfig};
use crate::rng::{chance, normal, normal_vec2, uniform, unit_vec2, Rngs};
use crate::robot::Robot;
use crate::types::{RobotId, SimTime, Team, Vec2, Vec3};

/// Pixel-area scale factor. **1**: the reported `area` is the raw blob pixel
/// area, so `focal_length_px` can be the physically calibrated focal length
/// instead of a fudge (vision.md §6, §9.3 — ER-Force's 10 forced a 390 px
/// "focal length" onto rigs that calibrate at 1416–1474 px).
pub const PIXEL_PER_AREA: f64 = 1.0;

/// Fraction of the camera height above which a flying ball is no longer
/// projected (the projection would diverge at the camera plane).
pub const PROJECTION_SCALING_LIMIT: f64 = 0.9;

/// Number of samples per axis of the occlusion grid (ER-Force uses 15x15).
pub const OCCLUSION_SAMPLE_RADIUS: i32 = 7;

/// Number of plane waves summed into one smooth calibration warp field. Three
/// is enough for the field to look aperiodic over a field-sized region while
/// keeping its spatial standard deviation predictable (see [`WarpBasis`]).
pub const WARP_HARMONICS: usize = 3;

/// Fixed direction of the reported-calibration position error (ER-Force uses a
/// deterministic `normalize(0.3, 0.7, 0.05)` offset, not a random one).
const CAMERA_ERROR_DIRECTION: Vec3 = Vec3::new(0.3, 0.7, 0.05);

/// A detected ball.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DetectedBall {
    /// Reported horizontal position [m] (floor-projected, warped, noisy).
    pub pos: Vec2,
    /// Reported height [m], only when `report_ball_z`.
    pub z: Option<f64>,
    /// Reported area [px]. `None` when `report_area` is off — a whole
    /// tournament in the corpus reported no `area` at all (vision.md §3, §9.9).
    pub area: Option<f64>,
    /// Confidence 0..1.
    pub confidence: f64,
}

/// A detected robot.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DetectedRobot {
    /// Robot number.
    pub number: u8,
    /// Reported position [m].
    pub pos: Vec2,
    /// Reported orientation [rad].
    pub orientation: f64,
    /// Reported height [m] (from `RobotSpecs::height`; vision.md §9.12).
    pub height: f64,
    /// Confidence 0..1.
    pub confidence: f64,
}

/// One camera's detection frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionFrame {
    /// Camera id.
    pub camera_id: u32,
    /// Per-camera frame counter.
    pub frame_number: u32,
    /// Capture timestamp [s] (sim time + delay - processing + epoch offset).
    pub t_capture: f64,
    /// Send timestamp [s].
    pub t_sent: f64,
    /// Balls.
    pub balls: Vec<DetectedBall>,
    /// Blue robots.
    pub robots_blue: Vec<DetectedRobot>,
    /// Yellow robots.
    pub robots_yellow: Vec<DetectedRobot>,
}

/// Camera calibration as advertised in the geometry packet.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraCalibration {
    /// Camera id.
    pub camera_id: u32,
    /// Reported world position [m] (true position plus `camera_position_error`).
    pub position: Vec3,
    /// Focal length [px].
    pub focal_length: f64,
    /// Principal point [px].
    pub principal_point: Vec2,
    /// Radial distortion coefficient.
    pub distortion: f64,
    /// Orientation quaternion (w, x, y, z) of the camera looking straight down.
    pub q: [f64; 4],
}

impl CameraCalibration {
    /// Translation part of the world-to-camera transform, i.e. the SSL
    /// `tx`/`ty`/`tz` fields [m], consistent with [`Self::q`] and
    /// [`Self::position`]: `t = -R * c` with `R = diag(1, -1, -1)` for a camera
    /// looking straight down.
    pub fn translation(&self) -> Vec3 {
        Vec3::new(-self.position.x, self.position.y, self.position.z)
    }
}

/// Geometry data for the geometry packet (field + cameras + ball models).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryData {
    /// Field dimensions and lines.
    pub field: FieldGeometry,
    /// Camera calibrations.
    pub cameras: Vec<CameraCalibration>,
    /// Ball model constants. The net layer advertises
    /// [`BallParams::advertised_acc_roll`], not the raw `acc_roll`.
    pub ball: BallParams,
}

/// One camera's capture: exactly one detection frame plus an optional geometry
/// packet (attached to camera 0's captures only).
///
/// Cameras have independent capture instants (see the module docs), so an
/// output is per camera, not per rig.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisionOutput {
    /// Sim time of capture.
    pub capture_time: SimTime,
    /// Index of the capturing camera in [`VisionModel::cameras`].
    pub camera_index: usize,
    /// The frame.
    pub frame: DetectionFrame,
    /// Geometry to attach, if due on this camera-0 frame.
    pub geometry: Option<GeometryData>,
}

/// Points per axis of the grid the warp is normalised over.
const WARP_NORMALISATION_SAMPLES: usize = 24;

/// A sum of [`WARP_HARMONICS`] plane waves of wavelength
/// `calibration_warp_length` with random directions and phases, drawn once per
/// camera: `gain * sqrt(2/K) * sum_k sin(2 pi (p . u_k) / length + phi_k)`,
/// scaled by the configured standard deviation at sample time.
///
/// `gain` is fitted at draw time so that the field's standard deviation over
/// the playing area is exactly the configured value: with only a handful of
/// waves and a field spanning a few wavelengths, a single realisation's spread
/// is otherwise up to ~30 % away from the `sqrt(K/2)` ensemble value. It is
/// fitted with the warp length in force at draw time, so a *later* change of
/// `calibration_warp_length` re-shapes the field without re-fitting the gain.
///
/// Only the directions, phases and gain are drawn; the amplitude is read from
/// [`Realism`] at sample time, so a live realism update takes effect at once.
#[derive(Debug, Clone, Copy, PartialEq)]
struct WarpBasis {
    terms: [(Vec2, f64); WARP_HARMONICS],
    gain: f64,
}

impl WarpBasis {
    /// Draw a basis and normalise it over the `extent` (half length, half
    /// width) of the playing area.
    fn draw(rng: &mut impl rand::Rng, extent: Vec2, length: f64) -> Self {
        let mut terms = [(Vec2::X, 0.0); WARP_HARMONICS];
        for term in terms.iter_mut() {
            let dir = unit_vec2(rng);
            let phase = uniform(rng, 0.0, std::f64::consts::TAU);
            *term = (dir, phase);
        }
        let mut basis = Self { terms, gain: 1.0 };
        basis.gain = basis.fit_gain(extent, length);
        basis
    }

    /// Unit-standard-deviation field before the gain fit.
    fn raw(&self, p: Vec2, length: f64) -> f64 {
        let amp = (2.0 / WARP_HARMONICS as f64).sqrt();
        let k = std::f64::consts::TAU / length;
        amp * self
            .terms
            .iter()
            .map(|(dir, phase)| (p.dot(*dir) * k + phase).sin())
            .sum::<f64>()
    }

    fn fit_gain(&self, extent: Vec2, length: f64) -> f64 {
        if length <= 0.0 || extent.x <= 0.0 || extent.y <= 0.0 {
            return 1.0;
        }
        let n = WARP_NORMALISATION_SAMPLES;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..n {
            for j in 0..n {
                let f = |t: usize, half: f64| -half + 2.0 * half * (t as f64 + 0.5) / n as f64;
                let v = self.raw(Vec2::new(f(i, extent.x), f(j, extent.y)), length);
                sum += v;
                sum_sq += v * v;
            }
        }
        let count = (n * n) as f64;
        let mean = sum / count;
        let var = (sum_sq / count - mean * mean).max(0.0);
        if var > 1e-18 {
            1.0 / var.sqrt()
        } else {
            0.0
        }
    }

    fn sample(&self, p: Vec2, stddev: f64, length: f64) -> f64 {
        if stddev == 0.0 || length <= 0.0 {
            return 0.0;
        }
        stddev * self.gain * self.raw(p, length)
    }
}

/// Static per-camera state drawn once from `rngs.calibration`: the capture
/// phase and the calibration error field.
#[derive(Debug, Clone, Copy, PartialEq)]
struct CameraState {
    /// Capture offset within the frame period, as a fraction of the period.
    phase_fraction: f64,
    /// Unit direction of the constant position offset (vision.md §5: RoboCup
    /// 2026 shows a 2.0-2.6 cm constant inter-camera offset along x).
    offset_direction: Vec2,
    /// Sign of the constant orientation offset.
    orientation_sign: f64,
    /// Smooth position warp, one basis per axis.
    warp_x: WarpBasis,
    /// Smooth position warp, y axis.
    warp_y: WarpBasis,
    /// Smooth orientation warp.
    warp_phi: WarpBasis,
}

impl CameraState {
    fn draw(
        rng: &mut impl rand::Rng,
        phase: CameraPhase,
        locked_fraction: f64,
        extent: Vec2,
        warp_length: f64,
    ) -> Self {
        // Draw in a fixed order so a camera's state depends only on the seed
        // and its index, not on the phase mode.
        let free_fraction = uniform(rng, 0.0, 1.0);
        let direction = unit_vec2(rng);
        let sign = if uniform(rng, 0.0, 1.0) < 0.5 {
            -1.0
        } else {
            1.0
        };
        Self {
            phase_fraction: match phase {
                CameraPhase::Locked => locked_fraction,
                CameraPhase::FreeRunning => free_fraction,
            },
            offset_direction: direction,
            orientation_sign: sign,
            warp_x: WarpBasis::draw(rng, extent, warp_length),
            warp_y: WarpBasis::draw(rng, extent, warp_length),
            warp_phi: WarpBasis::draw(rng, extent, warp_length),
        }
    }
}

/// The vision generator: owns per-camera schedules, the static calibration
/// state and the delay queue.
#[derive(Debug, Clone)]
pub struct VisionModel {
    config: VisionConfig,
    realism: Realism,
    cameras: Vec<CameraConfig>,
    /// Half length / half width of the playing area, used to normalise the warp.
    field_extent: Vec2,
    /// Drawn lazily on the first capture (the constructor has no RNG).
    states: Vec<CameraState>,
    frame_numbers: Vec<u32>,
    next_frame_times: Vec<SimTime>,
    frames_since_geometry: u32,
    queue: VecDeque<(SimTime, VisionOutput)>,
}

impl VisionModel {
    /// Create the model; auto-places cameras when `config.cameras` is empty.
    ///
    /// The per-camera calibration state (phase, offset direction, warp) is
    /// drawn from `rngs.calibration` on the first [`Self::maybe_capture`],
    /// which is the first point at which the model sees the seeded streams.
    pub fn new(config: VisionConfig, realism: Realism, field: &FieldGeometry) -> Self {
        let cameras = if config.cameras.is_empty() {
            auto_cameras(
                field,
                config.default_camera_count,
                config.default_camera_height,
                config.default_camera_x_fraction,
            )
        } else {
            config.cameras.clone()
        };
        let n = cameras.len();
        Self {
            config,
            realism,
            cameras,
            field_extent: Vec2::new(field.length * 0.5, field.width * 0.5),
            states: Vec::new(),
            frame_numbers: vec![0; n],
            next_frame_times: Vec::new(),
            frames_since_geometry: u32::MAX,
            queue: VecDeque::new(),
        }
    }

    /// Replace the realism config. Amplitudes and lengths of the calibration
    /// warp are read from it live, so the change takes effect immediately
    /// without redrawing (and therefore without changing) the warp shape.
    pub fn set_realism(&mut self, realism: Realism) {
        self.realism = realism;
    }

    /// Replace the camera rig (e.g. from a `SimulatorConfig.geometry.calib`
    /// update). Per-camera frame counters restart, the calibration state is
    /// re-drawn on the next capture and the geometry packet is re-sent; queued
    /// outputs are kept. An empty list re-derives the default rig.
    pub fn set_cameras(&mut self, cameras: Vec<CameraConfig>, field: &FieldGeometry) {
        self.cameras = if cameras.is_empty() {
            auto_cameras(
                field,
                self.config.default_camera_count,
                self.config.default_camera_height,
                self.config.default_camera_x_fraction,
            )
        } else {
            cameras
        };
        self.field_extent = Vec2::new(field.length * 0.5, field.width * 0.5);
        self.frame_numbers = vec![0; self.cameras.len()];
        self.states.clear();
        self.next_frame_times.clear();
        self.frames_since_geometry = u32::MAX;
    }

    /// Current realism config.
    pub fn realism(&self) -> &Realism {
        &self.realism
    }

    /// Cameras in use.
    pub fn cameras(&self) -> &[CameraConfig] {
        &self.cameras
    }

    /// Vision config.
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Sim time of the next capture of camera `cam`, or [`SimTime::ZERO`]
    /// before the schedule has been drawn.
    pub fn next_capture_time_of(&self, cam: usize) -> SimTime {
        self.next_frame_times
            .get(cam)
            .copied()
            .unwrap_or(SimTime::ZERO)
    }

    /// Sim time of the next capture of any camera.
    pub fn next_capture_time(&self) -> SimTime {
        self.next_frame_times
            .iter()
            .copied()
            .min()
            .unwrap_or(SimTime::ZERO)
    }

    /// Interval between captures of one camera, from `config.frame_rate`
    /// (measured 73.2-73.3 Hz on all 19 streams of the corpus, vision.md §1).
    pub fn frame_period(&self) -> SimTime {
        let hz = if self.config.frame_rate > 0.0 {
            self.config.frame_rate
        } else {
            60.0
        };
        SimTime::from_secs_f64(1.0 / hz).max(SimTime(1))
    }

    /// Draw the static per-camera calibration state and the capture schedule if
    /// that has not happened yet. Idempotent.
    pub fn ensure_calibration(&mut self, rngs: &mut Rngs) {
        if self.states.len() == self.cameras.len() {
            return;
        }
        let period = self.frame_period();
        let period_secs = period.as_secs_f64();
        self.states = (0..self.cameras.len())
            .map(|i| {
                let locked = self
                    .config
                    .phase_offsets
                    .get(i)
                    .copied()
                    .unwrap_or(0.0)
                    .rem_euclid(period_secs.max(f64::MIN_POSITIVE))
                    / period_secs;
                CameraState::draw(
                    &mut rngs.calibration,
                    self.config.camera_phase,
                    locked,
                    self.field_extent,
                    self.realism.calibration_warp_length,
                )
            })
            .collect();
        self.next_frame_times = self
            .states
            .iter()
            .map(|s| SimTime::from_secs_f64(s.phase_fraction * period_secs))
            .collect();
        self.frame_numbers = vec![0; self.cameras.len()];
    }

    /// Capture offset of camera `cam` within the frame period [s]; 0 before the
    /// schedule has been drawn.
    pub fn camera_phase_offset(&self, cam: usize) -> f64 {
        self.states
            .get(cam)
            .map(|s| s.phase_fraction * self.frame_period().as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Constant per-camera position offset [m]: `object_position_offset` in a
    /// direction drawn once per camera (vision.md §5, §9.5). Zero before the
    /// calibration state has been drawn.
    pub fn position_offset(&self, cam: usize) -> Vec2 {
        match self.states.get(cam) {
            Some(s) => s.offset_direction * self.realism.object_position_offset,
            None => Vec2::ZERO,
        }
    }

    /// Total static position error this camera adds to an object reported at
    /// `p`: the constant offset plus the smooth spatial warp. This is the
    /// dominant vision error in real logs (20 mm rms between two cameras versus
    /// 0.4 mm of white noise, vision.md §5).
    pub fn calibration_offset(&self, cam: usize, p: Vec2) -> Vec2 {
        let Some(s) = self.states.get(cam) else {
            return Vec2::ZERO;
        };
        let (sd, len) = (
            self.realism.calibration_warp_stddev,
            self.realism.calibration_warp_length,
        );
        s.offset_direction * self.realism.object_position_offset
            + Vec2::new(s.warp_x.sample(p, sd, len), s.warp_y.sample(p, sd, len))
    }

    /// Total static orientation error this camera adds to a robot at `p` [rad]
    /// (25-32 mrad rms between cameras in the corpus, vision.md §5).
    pub fn calibration_orientation(&self, cam: usize, p: Vec2) -> f64 {
        let Some(s) = self.states.get(cam) else {
            return 0.0;
        };
        s.orientation_sign * self.realism.calibration_orientation_offset
            + s.warp_phi.sample(
                p,
                self.realism.calibration_orientation_warp,
                self.realism.calibration_warp_length,
            )
    }

    /// True while `now` falls inside a configured vision outage. Nine of ten
    /// games in the corpus contain a 303-421 s gap (vision.md §1, §9.13).
    pub fn in_outage(&self, now: SimTime) -> bool {
        let t = now.as_secs_f64();
        self.config
            .outages
            .iter()
            .any(|o| o.duration > 0.0 && t >= o.start && t < o.start + o.duration)
    }

    /// Called once per substep after physics. Every camera whose own capture
    /// time has arrived generates a frame, which is enqueued with due time
    /// `now + vision_delay`. Returns true if at least one camera captured.
    pub fn maybe_capture(
        &mut self,
        now: SimTime,
        ball: &Ball,
        robots: &BTreeMap<RobotId, Robot>,
        field: &FieldGeometry,
        ball_params: &BallParams,
        rngs: &mut Rngs,
    ) -> bool {
        if self.cameras.is_empty() {
            return false;
        }
        self.ensure_calibration(rngs);
        let period = self.frame_period();
        let suppressed = self.in_outage(now);
        let mut captured = false;

        for ci in 0..self.cameras.len() {
            if now < self.next_frame_times[ci] {
                continue;
            }
            // Advance the schedule past `now` even when the capture is
            // suppressed, and keep the frame counter running (real outages do
            // not reset the camera's frame numbering, vision.md §1).
            self.next_frame_times[ci] = self.next_frame_times[ci].plus(period);
            while self.next_frame_times[ci] <= now {
                self.next_frame_times[ci] = self.next_frame_times[ci].plus(period);
            }
            let frame_number = self.frame_numbers[ci];
            self.frame_numbers[ci] = frame_number.wrapping_add(1);
            if suppressed {
                // The geometry cadence keeps running too, so a geometry packet
                // that fell inside the outage goes out on the first frame after
                // it rather than a full period later.
                if ci == 0 {
                    self.frames_since_geometry = self.frames_since_geometry.saturating_add(1);
                }
                continue;
            }

            let frame = self.capture_camera(ci, frame_number, now, ball, robots, ball_params, rngs);
            let geometry = if ci == 0 {
                let every = self.config.geometry_every_n_frames.max(1);
                if self.frames_since_geometry >= every {
                    self.frames_since_geometry = 1;
                    Some(self.geometry(field, ball_params, rngs))
                } else {
                    self.frames_since_geometry = self.frames_since_geometry.saturating_add(1);
                    None
                }
            } else {
                None
            };

            let due = now.plus(SimTime::from_secs_f64(self.realism.vision_delay));
            self.queue.push_back((
                due,
                VisionOutput {
                    capture_time: now,
                    camera_index: ci,
                    frame,
                    geometry,
                },
            ));
            captured = true;
        }
        captured
    }

    /// Build one camera's detection frame.
    #[allow(clippy::too_many_arguments)]
    fn capture_camera(
        &self,
        ci: usize,
        frame_number: u32,
        now: SimTime,
        ball: &Ball,
        robots: &BTreeMap<RobotId, Robot>,
        ball_params: &BallParams,
        rngs: &mut Rngs,
    ) -> DetectionFrame {
        let camera = self.cameras[ci];
        let period_secs = self.frame_period().as_secs_f64();
        let epoch = self.config.timestamp_epoch_offset;
        let t_sent = now.as_secs_f64() + self.realism.vision_delay + epoch;
        let t_capture = t_sent - self.realism.vision_processing_time;

        let mut frame = DetectionFrame {
            camera_id: camera.id,
            frame_number,
            t_capture,
            t_sent,
            balls: Vec::new(),
            robots_blue: Vec::new(),
            robots_yellow: Vec::new(),
        };

        // --- robots ---
        for (id, robot) in robots.iter() {
            if !self.visible_in_camera(ci, robot.pos) {
                continue;
            }
            if chance(
                &mut rngs.vision_dropout,
                self.realism.missing_robot_detections,
            ) {
                continue;
            }
            let warped = robot.pos + self.calibration_offset(ci, robot.pos);
            let warped_phi = robot.orientation + self.calibration_orientation(ci, robot.pos);
            let mut push = |rngs: &mut Rngs| {
                let noise = normal_vec2(&mut rngs.vision_noise, self.realism.stddev_robot_p);
                let phi = normal(&mut rngs.vision_noise, self.realism.stddev_robot_phi);
                let det = DetectedRobot {
                    number: id.number,
                    pos: warped + noise,
                    orientation: warped_phi + phi,
                    height: robot.specs.height,
                    confidence: self.draw_confidence(self.config.robot_confidence_mean, rngs),
                };
                match id.team {
                    Team::Blue => frame.robots_blue.push(det),
                    Team::Yellow => frame.robots_yellow.push(det),
                }
            };
            push(rngs);
            // Duplicate ids in one frame are rare but real: 932 in 31.4 M
            // detections (vision.md §4, §9.10).
            if chance(&mut rngs.vision_dropout, self.config.duplicate_robot_rate) {
                push(rngs);
            }
        }

        // --- the real ball ---
        let ball_pos = ball.state.pos;
        let ball_xy = Vec2::new(ball_pos.x, ball_pos.y);
        if self.visible_in_camera(ci, ball_xy)
            && !chance(
                &mut rngs.vision_dropout,
                self.realism.missing_ball_detections,
            )
        {
            let (visibility, centroid) = if self.realism.enable_invisible_ball {
                ball_visibility(ball_pos, ball_params.radius, camera.position, robots)
            } else {
                (1.0, ball_pos)
            };
            if visibility >= self.realism.ball_visibility_threshold {
                let projected = project_to_floor(centroid, ball_params.radius, camera.position);
                let area_noise = normal(&mut rngs.vision_noise, self.realism.stddev_ball_area);
                let area = self.report_area(ball_area(
                    centroid,
                    ball_params.radius,
                    camera.position,
                    visibility,
                    self.config.focal_length_px,
                    self.config.area_at_nadir_px,
                    area_noise,
                ));
                let noise = normal_vec2(&mut rngs.vision_noise, self.realism.stddev_ball_p);
                frame.balls.push(DetectedBall {
                    pos: projected + self.calibration_offset(ci, projected) + noise,
                    z: self.config.report_ball_z.then_some(ball_pos.z),
                    area,
                    confidence: self.draw_confidence(self.config.ball_confidence_mean, rngs),
                });
            }
        }

        // --- spurious "dribbler balls" (an IR break beam firing into the
        // camera): on the robot's centreline, ~0.13 m ahead of its centre
        // (vision.md §6, §9.7). ---
        let spurious_p = self.realism.dribbler_ball_detections * period_secs;
        if spurious_p > 0.0 {
            for robot in robots.values() {
                if !chance(&mut rngs.vision_dropout, spurious_p) {
                    continue;
                }
                let lateral = normal(
                    &mut rngs.vision_noise,
                    self.config.spurious_ball_lateral_stddev,
                );
                let pos = spurious_ball_pos(robot, self.config.spurious_ball_forward, lateral);
                if !self.visible_in_camera(ci, pos) {
                    continue;
                }
                frame.balls.push(DetectedBall {
                    pos: pos + self.calibration_offset(ci, pos),
                    z: self.config.report_ball_z.then_some(ball_params.radius),
                    area: self.report_area(self.config.spurious_ball_area_px),
                    confidence: self.draw_confidence(self.config.ball_confidence_mean, rngs),
                });
            }
        }

        // Trackers can have systematic errors depending on the ball order.
        if frame.balls.len() > 1 {
            crate::rng::shuffle(&mut rngs.shuffle, &mut frame.balls);
        }
        frame
    }

    /// `Some(area)` unless the rig is configured not to report `area` at all.
    fn report_area(&self, area: f64) -> Option<f64> {
        self.config.report_area.then_some(area)
    }

    /// Draw a reported confidence: `N(mean, confidence_stddev)` clamped to
    /// `(0, 1]`. Robots average 0.896 and balls 0.913 in the corpus, not the
    /// constant 1.0 we used to emit (vision.md §3, §9.11).
    fn draw_confidence(&self, mean: f64, rngs: &mut Rngs) -> f64 {
        let v = mean + normal(&mut rngs.vision_noise, self.config.confidence_stddev);
        v.clamp(f64::MIN_POSITIVE, 1.0)
    }

    /// Pop every queued output whose due time is `<= now`, oldest first.
    pub fn drain_due(&mut self, now: SimTime) -> Vec<VisionOutput> {
        let mut out = Vec::new();
        while let Some((due, _)) = self.queue.front() {
            if *due <= now {
                out.push(self.queue.pop_front().unwrap().1);
            } else {
                break;
            }
        }
        out
    }

    /// Number of queued outputs not yet due.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// Build the geometry packet content (used on demand as well as periodically).
    ///
    /// The reported camera positions carry a deterministic
    /// `camera_position_error` offset along `normalize(0.3, 0.7, 0.05)` (the
    /// true positions used for the ball projection stay correct, so this models
    /// *calibration* error, exactly like ER-Force). The much larger, visible
    /// error lives in [`Self::calibration_offset`]. `rngs` is unused.
    pub fn geometry(
        &self,
        field: &FieldGeometry,
        ball: &BallParams,
        rngs: &mut Rngs,
    ) -> GeometryData {
        let _ = rngs;
        let error = CAMERA_ERROR_DIRECTION.normalize() * self.realism.camera_position_error;
        let cameras = self
            .cameras
            .iter()
            .map(|c| CameraCalibration {
                camera_id: c.id,
                position: c.position + error,
                focal_length: self.config.focal_length_px,
                principal_point: Vec2::new(300.0, 300.0),
                distortion: 0.0,
                // Looking straight down: rotate pi about world +x, so camera +z
                // points at the floor and camera +y points at world -y.
                q: [0.0, 1.0, 0.0, 0.0],
            })
            .collect();
        GeometryData {
            field: *field,
            cameras,
            ball: *ball,
        }
    }

    /// True if `p` is within `fov_radius` of camera `cam`'s nadir. Real cameras
    /// see 98.8-99.8 % of robots out to 6.5 m and almost nothing past 7.25 m
    /// (vision.md §4, §9.6); `fov_radius = 0` disables the test.
    pub fn within_fov(&self, cam: usize, p: Vec2) -> bool {
        let Some(c) = self.cameras.get(cam) else {
            return false;
        };
        let r = self.config.fov_radius;
        r <= 0.0 || Vec2::new(c.position.x, c.position.y).distance(p) <= r
    }

    /// A point is visible to camera `cam` if it is inside that camera's
    /// field-of-view disc **and** its Manhattan distance to that camera is
    /// within `2 * camera_overlap` of the minimum over all cameras (the
    /// ER-Force region rule).
    pub fn visible_in_camera(&self, cam: usize, p: Vec2) -> bool {
        let Some(own) = self.cameras.get(cam) else {
            return false;
        };
        if !self.within_fov(cam, p) {
            return false;
        }
        let manhattan = |c: &CameraConfig| (c.position.x - p.x).abs() + (c.position.y - p.y).abs();
        let own_distance = manhattan(own);
        let mut min_distance = f64::INFINITY;
        for c in &self.cameras {
            min_distance = min_distance.min(manhattan(c));
        }
        own_distance <= min_distance + 2.0 * self.realism.camera_overlap
    }
}

/// Place `count` cameras: 4 at `(+-x_fraction*L, +-W/4)`, 2 at
/// `(+-x_fraction*L, 0)`, 1 at the origin, all at `height`.
///
/// Ids are `0..count` in the order returned: for four cameras that is
/// `(+,+), (-,+), (-,-), (+,-)`; for two `(+, 0), (-, 0)`. Counts other than
/// 1, 2 and 4 are rounded down to the nearest supported layout (0 becomes 1).
///
/// Defaults come from the 2026 corpus (vision.md §2): two-camera rigs split
/// along x only at x ~ +-2.4 m on a 12 m field (0.20 L, not the classic
/// 0.25 L) and hang 5.97-6.52 m above the floor.
pub fn auto_cameras(
    field: &FieldGeometry,
    count: u32,
    height: f64,
    x_fraction: f64,
) -> Vec<CameraConfig> {
    let qx = field.length * x_fraction;
    let qy = field.width * 0.25;
    let xy: Vec<(f64, f64)> = match count {
        0 | 1 => vec![(0.0, 0.0)],
        2 | 3 => vec![(qx, 0.0), (-qx, 0.0)],
        _ => vec![(qx, qy), (-qx, qy), (-qx, -qy), (qx, -qy)],
    };
    xy.into_iter()
        .enumerate()
        .map(|(i, (x, y))| CameraConfig {
            id: i as u32,
            position: Vec3::new(x, y, height),
        })
        .collect()
}

/// Project a ball at `pos` (centre) to the floor as seen from `camera`
/// (ER-Force: `z' = min(0.9 cz, max(0, z - r))`, scale `cz / (cz - z')`).
pub fn project_to_floor(pos: Vec3, radius: f64, camera: Vec3) -> Vec2 {
    let cam_xy = Vec2::new(camera.x, camera.y);
    let p_xy = Vec2::new(pos.x, pos.y);
    if camera.z <= 0.0 {
        return p_xy;
    }
    let z = (pos.z - radius)
        .max(0.0)
        .min(PROJECTION_SCALING_LIMIT * camera.z);
    let scale = camera.z / (camera.z - z);
    cam_xy + (p_xy - cam_xy) * scale
}

/// Reported ball `area` [px]:
/// `area_at_nadir_px * visibility * height_term`, plus `area_noise`.
///
/// The height term is the pinhole disc area at the ball's height relative to
/// the same disc on the floor, `((f(H) / f(H - z'))^2` with ER-Force's
/// `f(d) = 1000 d / focal_length_px - 1`; at `focal_length_px = 1420` and a
/// 6.4 m camera that runs ~15 % steeper than an ideal pinhole, which is what
/// the logs show (vision.md §6).
///
/// There is deliberately **no horizontal distance term**: measured `area` is
/// flat to within +-8 % from the nadir out to 5 m where a pinhole 1/d^2 would
/// predict a 37 % fall, because a flat sensor's 1/cos^3 obliquity nearly
/// cancels it (vision.md §6, §9.2).
pub fn ball_area(
    ball_pos: Vec3,
    radius: f64,
    camera: Vec3,
    visibility: f64,
    focal_length_px: f64,
    area_at_nadir_px: f64,
    area_noise: f64,
) -> f64 {
    let height = camera.z.max(0.0);
    let z = (ball_pos.z - radius)
        .max(0.0)
        .min(PROJECTION_SCALING_LIMIT * height);
    let f = |d: f64| d * 1000.0 / focal_length_px.max(1e-9) - 1.0;
    let (nadir, at_z) = (f(height), f(height - z));
    let term = if nadir > 1e-9 && at_z > 1e-9 {
        (nadir / at_z) * (nadir / at_z)
    } else {
        1.0
    };
    let area = visibility * (area_at_nadir_px * term + area_noise).max(0.0);
    area * PIXEL_PER_AREA
}

/// World position of a spurious "dribbler ball": on the robot's centreline,
/// `forward` metres ahead of the robot *centre*, displaced `lateral` metres to
/// the left (a pre-drawn `N(0, spurious_ball_lateral_stddev)` sample).
///
/// The German Open logs put these at +0.119..+0.141 m forward and
/// 0.000 +- 0.07 m lateral: on the centreline, a few cm in front of the kicker
/// face — not at the dribbler corner our old model used (vision.md §6, §9.7).
pub fn spurious_ball_pos(robot: &Robot, forward: f64, lateral: f64) -> Vec2 {
    let heading = robot.heading();
    let left = Vec2::new(-heading.y, heading.x);
    robot.pos + heading * forward + left * lateral
}

/// True if the segment from `from` to `to` passes through the vertical cylinder
/// of radius `radius` and height `height` standing at `center` on the floor.
fn segment_hits_cylinder(from: Vec3, to: Vec3, center: Vec2, radius: f64, height: f64) -> bool {
    let d = to - from;
    let dxy = Vec2::new(d.x, d.y);
    let m = Vec2::new(from.x - center.x, from.y - center.y);
    let a = dxy.length_squared();
    let c = m.length_squared() - radius * radius;

    // Parameter interval [t0, t1] (subset of [0, 1]) spent inside the disc.
    let (t0, t1) = if a <= 1e-18 {
        // Purely vertical ray: either always inside the disc or never.
        if c > 0.0 {
            return false;
        }
        (0.0, 1.0)
    } else {
        let b = 2.0 * m.dot(dxy);
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return false;
        }
        let sq = disc.sqrt();
        let r0 = (-b - sq) / (2.0 * a);
        let r1 = (-b + sq) / (2.0 * a);
        (r0.max(0.0), r1.min(1.0))
    };
    if t0 > t1 {
        return false;
    }

    // z is linear in t, so its range over [t0, t1] is between the endpoints.
    let z0 = from.z + t0 * d.z;
    let z1 = from.z + t1 * d.z;
    let (zlo, zhi) = if z0 <= z1 { (z0, z1) } else { (z1, z0) };
    zlo <= height && zhi >= 0.0
}

/// Fraction of the ball disc (facing the camera) visible past the robot
/// cylinders, and the centroid of the visible samples (floor-projected
/// afterwards by the caller). 15x15 grid, analytic ray-vs-cylinder tests.
///
/// The samples are laid out on a horizontal grid, tilted to face the camera for
/// the ray test, then counted at their untilted position so the centroid keeps
/// the ball's height (ER-Force does the same). Samples outside the disc are
/// excluded from both the numerator and the denominator. With nothing visible
/// the result is `(0.0, ball_pos)`.
pub fn ball_visibility(
    ball_pos: Vec3,
    radius: f64,
    camera: Vec3,
    robots: &BTreeMap<RobotId, Robot>,
) -> (f64, Vec3) {
    let to_camera = camera - ball_pos;
    let len = to_camera.length();
    if len <= 1e-9 {
        return (1.0, ball_pos);
    }
    let dir = to_camera / len;
    // Rotation taking +z onto `dir` (about axis = up x dir).
    let axis = Vec3::Z.cross(dir);
    let axis_len = axis.length();
    let (axis, angle) = if axis_len <= 1e-12 {
        (Vec3::X, 0.0)
    } else {
        (axis / axis_len, Vec3::Z.dot(dir).clamp(-1.0, 1.0).acos())
    };
    let (sin_a, cos_a) = angle.sin_cos();

    let n = OCCLUSION_SAMPLE_RADIUS;
    let step = radius / n as f64;
    let mut total = 0u32;
    let mut visible = 0u32;
    let mut sum = Vec3::ZERO;

    for ix in -n..=n {
        for iy in -n..=n {
            let flat = Vec3::new(ix as f64 * step, iy as f64 * step, 0.0);
            if flat.length_squared() >= radius * radius {
                continue;
            }
            total += 1;
            // Rodrigues' rotation of `flat` about `axis` by `angle`.
            let rotated =
                flat * cos_a + axis.cross(flat) * sin_a + axis * axis.dot(flat) * (1.0 - cos_a);
            let sample = ball_pos + rotated;
            let blocked = robots.values().any(|r| {
                segment_hits_cylinder(sample, camera, r.pos, r.specs.radius, r.specs.height)
            });
            if !blocked {
                visible += 1;
                sum += ball_pos + flat;
            }
        }
    }

    if total == 0 || visible == 0 {
        return (0.0, ball_pos);
    }
    (visible as f64 / total as f64, sum / visible as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Division;
    use crate::params::{RobotSpecs, VisionOutage};
    use crate::types::RobotId;

    fn field() -> FieldGeometry {
        FieldGeometry::division(Division::A)
    }

    fn model(cameras: Vec<CameraConfig>, realism: Realism) -> VisionModel {
        let config = VisionConfig {
            cameras,
            ..VisionConfig::default()
        };
        VisionModel::new(config, realism, &field())
    }

    fn cam(id: u32, x: f64, y: f64) -> CameraConfig {
        CameraConfig {
            id,
            position: Vec3::new(x, y, 4.0),
        }
    }

    fn robot_at(id: RobotId, x: f64, y: f64) -> Robot {
        Robot::new(id, RobotSpecs::default(), Vec2::new(x, y), 0.0)
    }

    fn robots(list: Vec<Robot>) -> BTreeMap<RobotId, Robot> {
        list.into_iter().map(|r| (r.id, r)).collect()
    }

    /// Step a model for `ms` milliseconds and return everything it produced.
    fn run(m: &mut VisionModel, ms: u64, rngs: &mut Rngs) -> Vec<VisionOutput> {
        let bp = BallParams::default();
        let ball = Ball::at_rest(Vec2::ZERO, &bp);
        let rs = BTreeMap::new();
        let f = field();
        for t in 1..=ms {
            m.maybe_capture(SimTime::from_millis(t), &ball, &rs, &f, &bp, rngs);
        }
        m.drain_due(SimTime::from_millis(ms + 10_000))
    }

    #[test]
    fn vision_auto_camera_placement() {
        let f = field();
        // The measured rig: two cameras at +-0.20 L, 6.4 m up (vision.md §2).
        let at = |c: &CameraConfig, x: f64, y: f64, z: f64| {
            assert!(
                (c.position - Vec3::new(x, y, z)).length() < 1e-9,
                "{:?} should be ({x}, {y}, {z})",
                c.position
            );
        };
        let two = auto_cameras(&f, 2, 6.4, 0.20);
        assert_eq!(two.len(), 2);
        at(&two[0], 2.4, 0.0, 6.4);
        at(&two[1], -2.4, 0.0, 6.4);

        let four = auto_cameras(&f, 4, 6.4, 0.20);
        assert_eq!(four.len(), 4);
        assert_eq!(
            four.iter().map(|c| c.id).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        at(&four[0], 2.4, 2.25, 6.4);
        at(&four[1], -2.4, 2.25, 6.4);
        at(&four[2], -2.4, -2.25, 6.4);
        at(&four[3], 2.4, -2.25, 6.4);

        let one = auto_cameras(&f, 1, 6.4, 0.20);
        assert_eq!(one.len(), 1);
        at(&one[0], 0.0, 0.0, 6.4);

        // The default config is the measured 2-camera rig.
        let m = VisionModel::new(VisionConfig::default(), Realism::none(), &f);
        assert_eq!(m.cameras().len(), 2);
        at(&m.cameras()[0], 2.4, 0.0, 6.4);
    }

    /// The 17 ER-Force `CameraOverlap` assertions: four cameras at (+-3, +-3)
    /// with `camera_overlap = 0.5`. Our stable id order for this layout is
    /// 0 = (+3, +3), 1 = (-3, +3), 2 = (+3, -3), 3 = (-3, -3).
    #[test]
    fn vision_camera_overlap_regions() {
        let realism = Realism {
            camera_overlap: 0.5,
            ..Realism::none()
        };
        let m = model(
            vec![
                cam(0, 3.0, 3.0),
                cam(1, -3.0, 3.0),
                cam(2, 3.0, -3.0),
                cam(3, -3.0, -3.0),
            ],
            realism,
        );
        let seen = |x: f64, y: f64| -> Vec<usize> {
            (0..4)
                .filter(|&i| m.visible_in_camera(i, Vec2::new(x, y)))
                .collect()
        };

        assert_eq!(seen(0.0, 0.0), vec![0, 1, 2, 3]);
        assert_eq!(seen(2.0, 0.0), vec![0, 2]);
        assert_eq!(seen(2.0, 0.51), vec![0]);
        assert_eq!(seen(2.0, 0.49), vec![0, 2]);
        assert_eq!(seen(2.0, -0.51), vec![2]);
        assert_eq!(seen(2.0, -0.49), vec![0, 2]);
        assert_eq!(seen(0.0, 2.0), vec![0, 1]);
        assert_eq!(seen(0.51, 2.0), vec![0]);
        assert_eq!(seen(0.49, 2.0), vec![0, 1]);
        assert_eq!(seen(-0.51, 2.0), vec![1]);
        assert_eq!(seen(3.0, 3.0), vec![0]);
        assert_eq!(seen(-3.0, -3.0), vec![3]);
        assert_eq!(seen(0.0, 0.4), vec![0, 1, 2, 3]);
        assert_eq!(seen(0.0, 0.51), vec![0, 1]);
        assert_eq!(seen(0.51, 0.51), vec![0]);
        assert_eq!(seen(0.2, 0.2), vec![0, 1, 2, 3]);
        assert_eq!(seen(5.0, 0.0), vec![0, 2]);
    }

    #[test]
    fn vision_field_of_view_clips_the_manhattan_region() {
        // One camera, huge overlap: only the FOV disc can clip anything.
        let config = VisionConfig {
            cameras: vec![cam(0, 0.0, 0.0)],
            fov_radius: 2.0,
            ..VisionConfig::default()
        };
        let m = VisionModel::new(
            config,
            Realism {
                camera_overlap: 100.0,
                ..Realism::none()
            },
            &field(),
        );
        assert!(m.visible_in_camera(0, Vec2::new(1.9, 0.0)));
        assert!(m.visible_in_camera(0, Vec2::new(0.0, -1.99)));
        assert!(!m.visible_in_camera(0, Vec2::new(2.01, 0.0)));
        assert!(!m.visible_in_camera(0, Vec2::new(1.5, 1.5)), "diagonal");
        assert!(!m.visible_in_camera(9, Vec2::ZERO), "unknown camera");

        // fov_radius = 0 disables the disc entirely.
        let config = VisionConfig {
            cameras: vec![cam(0, 0.0, 0.0)],
            fov_radius: 0.0,
            ..VisionConfig::default()
        };
        let m = VisionModel::new(
            config,
            Realism {
                camera_overlap: 100.0,
                ..Realism::none()
            },
            &field(),
        );
        assert!(m.visible_in_camera(0, Vec2::new(5.9, 4.4)));

        // And a robot beyond the edge is simply not in the frame.
        let config = VisionConfig {
            cameras: vec![cam(0, 0.0, 0.0)],
            fov_radius: 1.0,
            ..VisionConfig::default()
        };
        let mut m = VisionModel::new(config, Realism::none(), &field());
        let mut rngs = Rngs::from_seed(0);
        let bp = BallParams::default();
        let ball = Ball::at_rest(Vec2::new(0.2, 0.0), &bp);
        let rs = robots(vec![
            robot_at(RobotId::new(Team::Blue, 0), 0.5, 0.0),
            robot_at(RobotId::new(Team::Blue, 1), 3.0, 0.0),
        ]);
        let f = field();
        assert!(m.maybe_capture(SimTime::from_millis(1), &ball, &rs, &f, &bp, &mut rngs));
        let out = m.drain_due(SimTime::from_millis(1000));
        let blue = &out[0].frame.robots_blue;
        assert_eq!(blue.len(), 1, "only the near robot is inside the FOV");
        assert_eq!(blue[0].number, 0);
    }

    #[test]
    fn vision_projection_pushes_a_flying_ball_away_from_the_camera() {
        let camera = Vec3::new(3.0, 0.0, 4.0);
        let radius = 0.0215;
        // On the ground: reported where it is.
        let grounded = project_to_floor(Vec3::new(0.0, 0.0, radius), radius, camera);
        assert!((grounded - Vec2::ZERO).length() < 1e-12);

        // At 0.5 m the ball is reported further from the camera.
        let flying = project_to_floor(Vec3::new(0.0, 0.0, 0.5), radius, camera);
        let cam_xy = Vec2::new(camera.x, camera.y);
        assert!(flying.length() > 0.0);
        assert!(
            (flying - cam_xy).length() > (Vec2::ZERO - cam_xy).length(),
            "projected {flying:?} must be further from the camera"
        );
        // Analytic value: scale = 4 / (4 - 0.4785).
        let scale = 4.0 / (4.0 - (0.5 - radius));
        let expected = cam_xy + (Vec2::ZERO - cam_xy) * scale;
        assert!((flying - expected).length() < 1e-12);

        // The projection is clamped at 90 % of the camera height.
        let very_high = project_to_floor(Vec3::new(0.0, 0.0, 10.0), radius, camera);
        let clamped = cam_xy + (Vec2::ZERO - cam_xy) * (4.0 / (4.0 - 0.9 * 4.0));
        assert!((very_high - clamped).length() < 1e-9);
    }

    #[test]
    fn vision_ball_behind_a_robot_is_invisible() {
        let radius = 0.0215;
        let ball = Vec3::new(0.0, 0.0, radius);
        let camera = Vec3::new(3.0, 0.0, 4.0);
        let specs = RobotSpecs::default();
        // Robot touching the ball, directly between it and the camera.
        let blocker = robot_at(RobotId::new(Team::Blue, 0), specs.radius + radius, 0.0);
        let rs = robots(vec![blocker]);
        let (vis, centroid) = ball_visibility(ball, radius, camera, &rs);
        assert_eq!(vis, 0.0, "fully occluded ball");
        assert_eq!(centroid, ball, "centroid falls back to the true position");

        // With nobody in the way it is fully visible at its own position.
        let empty = BTreeMap::new();
        let (vis, centroid) = ball_visibility(ball, radius, camera, &empty);
        assert!((vis - 1.0).abs() < 1e-12);
        assert!((centroid - ball).length() < 1e-12);
    }

    #[test]
    fn vision_half_covered_ball_centroid_shifts_away_from_the_robot() {
        let radius = 0.0215;
        let ball = Vec3::new(0.0, 0.0, radius);
        let camera = Vec3::new(3.0, 0.0, 4.0);
        let specs = RobotSpecs::default();
        // Robot offset by its own radius in +y: it shadows the +y half of the disc.
        let blocker = robot_at(
            RobotId::new(Team::Blue, 0),
            specs.radius + radius,
            specs.radius,
        );
        let rs = robots(vec![blocker]);
        let (vis, centroid) = ball_visibility(ball, radius, camera, &rs);
        assert!(vis > 0.0 && vis < 1.0, "partially occluded, got {vis}");
        assert!(
            centroid.y < ball.y - 1e-4,
            "centroid {centroid:?} must creep away from the robot at +y"
        );
        assert!((centroid.z - ball.z).abs() < 1e-12, "height is preserved");
    }

    #[test]
    fn vision_capture_cadence_is_the_configured_frame_rate() {
        let mut m = model(auto_cameras(&field(), 2, 6.4, 0.2), Realism::none());
        let mut rngs = Rngs::from_seed(0);
        let outputs = run(&mut m, 1000, &mut rngs);

        // 73.3 Hz on each of two cameras: captures at 0, 13.6, ... 995.9 ms.
        let per_camera = |ci: usize| outputs.iter().filter(|o| o.camera_index == ci).count();
        assert_eq!(per_camera(0), 74, "73.3 Hz over one second");
        assert_eq!(per_camera(1), 74);
        assert_eq!(outputs.len(), 148, "one output per camera capture");

        // Per-camera frame numbers count up independently.
        for ci in 0..2 {
            let numbers: Vec<u32> = outputs
                .iter()
                .filter(|o| o.camera_index == ci)
                .map(|o| o.frame.frame_number)
                .collect();
            assert_eq!(numbers, (0..74).collect::<Vec<_>>());
        }

        // Geometry rides on camera 0 only, every `geometry_every_n_frames`.
        assert!(outputs
            .iter()
            .all(|o| o.geometry.is_none() || o.camera_index == 0));
        let geo_frames: Vec<u32> = outputs
            .iter()
            .filter(|o| o.geometry.is_some())
            .map(|o| o.frame.frame_number)
            .collect();
        // 74 frames of camera 0 in one second and `geometry_every_n_frames`
        // is 73, so the first and the 74th carry it.
        assert_eq!(geo_frames, vec![0, 73]);
    }

    #[test]
    fn vision_geometry_cadence_counts_camera_zero_frames() {
        let config = VisionConfig {
            cameras: auto_cameras(&field(), 2, 6.4, 0.2),
            geometry_every_n_frames: 10,
            ..VisionConfig::default()
        };
        let mut m = VisionModel::new(config, Realism::none(), &field());
        let mut rngs = Rngs::from_seed(0);
        let outputs = run(&mut m, 1000, &mut rngs);
        let geo_frames: Vec<u32> = outputs
            .iter()
            .filter(|o| o.geometry.is_some())
            .map(|o| o.frame.frame_number)
            .collect();
        assert_eq!(geo_frames, vec![0, 10, 20, 30, 40, 50, 60, 70]);
        assert!(
            outputs
                .iter()
                .all(|o| o.geometry.is_none() || o.camera_index == 0),
            "geometry only rides on camera 0"
        );
    }

    #[test]
    fn vision_locked_phase_offsets_stagger_the_cameras() {
        let period = 1.0 / VisionConfig::default().frame_rate;
        let config = VisionConfig {
            cameras: auto_cameras(&field(), 2, 6.4, 0.2),
            camera_phase: CameraPhase::Locked,
            phase_offsets: vec![0.0, period * 0.5],
            ..VisionConfig::default()
        };
        let mut m = VisionModel::new(config, Realism::none(), &field());
        let mut rngs = Rngs::from_seed(0);
        let outputs = run(&mut m, 200, &mut rngs);

        let first = |ci: usize| {
            outputs
                .iter()
                .find(|o| o.camera_index == ci)
                .unwrap()
                .capture_time
                .as_secs_f64()
        };
        let dt = first(1) - first(0);
        assert!(
            (dt - period * 0.5).abs() < 0.002,
            "camera 1 lags half a period, got {dt}"
        );
        // No two cameras share a capture instant here.
        for o in &outputs {
            let same: Vec<usize> = outputs
                .iter()
                .filter(|p| p.capture_time == o.capture_time)
                .map(|p| p.camera_index)
                .collect();
            assert_eq!(same.len(), 1, "capture instants are per camera");
        }

        // Default (no offsets) = every camera on the same schedule.
        let mut m = model(auto_cameras(&field(), 2, 6.4, 0.2), Realism::none());
        let mut rngs = Rngs::from_seed(0);
        let outputs = run(&mut m, 200, &mut rngs);
        assert_eq!(outputs[0].capture_time, outputs[1].capture_time);
        assert_ne!(outputs[0].camera_index, outputs[1].camera_index);
    }

    #[test]
    fn vision_free_running_phase_is_random_but_seeded() {
        let make = |seed: u64| {
            let config = VisionConfig {
                cameras: auto_cameras(&field(), 2, 6.4, 0.2),
                camera_phase: CameraPhase::FreeRunning,
                ..VisionConfig::default()
            };
            let mut m = VisionModel::new(config, Realism::none(), &field());
            let mut rngs = Rngs::from_seed(seed);
            m.ensure_calibration(&mut rngs);
            (m.camera_phase_offset(0), m.camera_phase_offset(1))
        };
        let a = make(7);
        let b = make(7);
        assert_eq!(a, b, "free-running phases are deterministic per seed");
        assert_ne!(make(8), a, "and depend on the seed");
        assert!(a.0 != a.1, "the two cameras run on different phases");
        let period = 1.0 / VisionConfig::default().frame_rate;
        for phase in [a.0, a.1] {
            assert!((0.0..period).contains(&phase), "phase {phase} in [0, T)");
        }

        // The staggered schedule is visible in the capture times.
        let config = VisionConfig {
            cameras: auto_cameras(&field(), 2, 6.4, 0.2),
            camera_phase: CameraPhase::FreeRunning,
            ..VisionConfig::default()
        };
        let mut m = VisionModel::new(config, Realism::none(), &field());
        let mut rngs = Rngs::from_seed(7);
        let outputs = run(&mut m, 500, &mut rngs);
        let first = |ci: usize| {
            outputs
                .iter()
                .find(|o| o.camera_index == ci)
                .unwrap()
                .capture_time
        };
        assert_ne!(first(0), first(1));
    }

    #[test]
    fn vision_outage_suppresses_packets_but_not_frame_numbers() {
        let config = VisionConfig {
            cameras: auto_cameras(&field(), 2, 6.4, 0.2),
            geometry_every_n_frames: 10,
            outages: vec![VisionOutage {
                start: 0.2,
                duration: 0.3,
            }],
            ..VisionConfig::default()
        };
        let mut m = VisionModel::new(config, Realism::none(), &field());
        let mut rngs = Rngs::from_seed(0);
        let outputs = run(&mut m, 1000, &mut rngs);

        for o in &outputs {
            let t = o.capture_time.as_secs_f64();
            assert!(!(0.2..0.5).contains(&t), "packet inside the outage at {t}");
        }
        let cam0: Vec<&VisionOutput> = outputs.iter().filter(|o| o.camera_index == 0).collect();
        // ~22 of the 74 frames fall in the 0.3 s outage.
        assert!(
            (50..=54).contains(&cam0.len()),
            "expected ~52 surviving frames, got {}",
            cam0.len()
        );
        // Frame numbers kept counting through the gap.
        let before = cam0.iter().find(|o| o.capture_time.as_secs_f64() < 0.2);
        let after = cam0.iter().find(|o| o.capture_time.as_secs_f64() >= 0.5);
        let (before, after) = (before.unwrap(), after.unwrap());
        assert!(
            after.frame.frame_number - before.frame.frame_number > 20,
            "frame numbers jump across the outage"
        );
        // And geometry is re-sent on the first frame after the gap.
        assert!(after.geometry.is_some());
    }

    #[test]
    fn vision_delay_queue_releases_at_the_right_time() {
        let realism = Realism::none();
        let delay = realism.vision_delay;
        let mut m = model(auto_cameras(&field(), 2, 6.4, 0.2), realism);
        let mut rngs = Rngs::from_seed(0);
        let ball = Ball::at_rest(Vec2::ZERO, &BallParams::default());
        let rs = BTreeMap::new();
        let f = field();
        let bp = BallParams::default();

        let capture = SimTime::from_millis(1);
        assert!(m.maybe_capture(capture, &ball, &rs, &f, &bp, &mut rngs));
        assert_eq!(m.pending(), 2, "one queued output per camera");

        let due = capture.plus(SimTime::from_secs_f64(delay));
        assert!(m.drain_due(due.minus(SimTime(1))).is_empty(), "not yet due");
        let out = m.drain_due(due);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].capture_time, capture);
        assert_eq!(m.pending(), 0);

        // Timestamps carry the delay and the processing time.
        let frame = &out[0].frame;
        assert!((frame.t_sent - (capture.as_secs_f64() + delay)).abs() < 1e-12);
        assert!(
            (frame.t_capture - (frame.t_sent - Realism::none().vision_processing_time)).abs()
                < 1e-12
        );
    }

    #[test]
    fn vision_reports_exact_positions_with_realism_none() {
        let mut m = model(auto_cameras(&field(), 4, 6.4, 0.2), Realism::none());
        let mut rngs = Rngs::from_seed(42);
        let bp = BallParams::default();
        let ball = Ball::at_rest(Vec2::new(0.25, -0.1), &bp);
        let rs = robots(vec![
            Robot::new(
                RobotId::new(Team::Blue, 3),
                RobotSpecs::default(),
                Vec2::new(-1.0, 0.5),
                0.7,
            ),
            Robot::new(
                RobotId::new(Team::Yellow, 5),
                RobotSpecs::default(),
                Vec2::new(1.0, -0.5),
                -0.2,
            ),
        ]);
        let f = field();
        assert!(m.maybe_capture(SimTime::from_millis(1), &ball, &rs, &f, &bp, &mut rngs));
        let out = m.drain_due(SimTime::from_millis(1000));
        assert_eq!(out.len(), 4, "four cameras, four outputs");

        let mut blue_seen = 0;
        let mut yellow_seen = 0;
        let mut balls_seen = 0;
        for o in &out {
            for r in &o.frame.robots_blue {
                assert_eq!(r.number, 3);
                assert!((r.pos - Vec2::new(-1.0, 0.5)).length() < 1e-12);
                assert!((r.orientation - 0.7).abs() < 1e-12);
                assert!((r.height - RobotSpecs::default().height).abs() < 1e-12);
                blue_seen += 1;
            }
            for r in &o.frame.robots_yellow {
                assert_eq!(r.number, 5);
                assert!((r.pos - Vec2::new(1.0, -0.5)).length() < 1e-12);
                yellow_seen += 1;
            }
            for b in &o.frame.balls {
                // Grounded ball: the projection is the identity.
                assert!((b.pos - Vec2::new(0.25, -0.1)).length() < 1e-9);
                assert!(b.z.is_none(), "z is omitted unless report_ball_z");
                assert!(b.area.unwrap() > 0.0);
                balls_seen += 1;
            }
        }
        assert!(blue_seen > 0 && yellow_seen > 0 && balls_seen > 0);
    }

    #[test]
    fn vision_confidence_is_drawn_and_bounded() {
        let config = VisionConfig {
            cameras: vec![cam(0, 0.0, 0.0)],
            // A wild spread to exercise the clamp.
            confidence_stddev: 5.0,
            ..VisionConfig::default()
        };
        let mut m = VisionModel::new(config, Realism::none(), &field());
        let mut rngs = Rngs::from_seed(3);
        let bp = BallParams::default();
        let ball = Ball::at_rest(Vec2::new(0.5, 0.0), &bp);
        let rs = robots(vec![robot_at(RobotId::new(Team::Blue, 0), 0.0, 0.0)]);
        let f = field();
        let mut seen = 0;
        let mut below_one = 0;
        for t in 1..=500u64 {
            m.maybe_capture(SimTime::from_millis(t), &ball, &rs, &f, &bp, &mut rngs);
        }
        for o in m.drain_due(SimTime::from_millis(10_000)) {
            for c in o
                .frame
                .robots_blue
                .iter()
                .map(|r| r.confidence)
                .chain(o.frame.balls.iter().map(|b| b.confidence))
            {
                assert!(c > 0.0 && c <= 1.0, "confidence {c} outside (0, 1]");
                seen += 1;
                if c < 1.0 {
                    below_one += 1;
                }
            }
        }
        assert!(seen > 50 && below_one > 0, "{seen} draws, {below_one} < 1");

        // The mean sits where it is configured.
        let config = VisionConfig {
            cameras: vec![cam(0, 0.0, 0.0)],
            robot_confidence_mean: 0.9,
            confidence_stddev: 0.05,
            ..VisionConfig::default()
        };
        let mut m = VisionModel::new(config, Realism::none(), &field());
        let mut rngs = Rngs::from_seed(4);
        for t in 1..=2000u64 {
            m.maybe_capture(SimTime::from_millis(t), &ball, &rs, &f, &bp, &mut rngs);
        }
        let confidences: Vec<f64> = m
            .drain_due(SimTime::from_millis(20_000))
            .iter()
            .flat_map(|o| o.frame.robots_blue.iter().map(|r| r.confidence))
            .collect();
        let mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
        assert!((mean - 0.9).abs() < 0.01, "mean confidence {mean}");
    }

    #[test]
    fn vision_duplicate_robot_ids_are_emitted() {
        let config = VisionConfig {
            cameras: vec![cam(0, 0.0, 0.0)],
            duplicate_robot_rate: 1.0,
            ..VisionConfig::default()
        };
        let mut m = VisionModel::new(
            config,
            Realism {
                stddev_robot_p: 0.001,
                ..Realism::none()
            },
            &field(),
        );
        let mut rngs = Rngs::from_seed(0);
        let bp = BallParams::default();
        let ball = Ball::at_rest(Vec2::new(2.0, 2.0), &bp);
        let rs = robots(vec![robot_at(RobotId::new(Team::Blue, 4), 0.0, 0.0)]);
        let f = field();
        assert!(m.maybe_capture(SimTime::from_millis(1), &ball, &rs, &f, &bp, &mut rngs));
        let out = m.drain_due(SimTime::from_millis(1000));
        let blue = &out[0].frame.robots_blue;
        assert_eq!(blue.len(), 2, "the same id twice");
        assert_eq!(blue[0].number, blue[1].number);
        assert_ne!(blue[0].pos, blue[1].pos, "independent noise");

        // Off by default-ish rates: 0 never duplicates.
        let config = VisionConfig {
            cameras: vec![cam(0, 0.0, 0.0)],
            duplicate_robot_rate: 0.0,
            ..VisionConfig::default()
        };
        let mut m = VisionModel::new(config, Realism::none(), &field());
        let mut rngs = Rngs::from_seed(0);
        assert!(m.maybe_capture(SimTime::from_millis(1), &ball, &rs, &f, &bp, &mut rngs));
        let out = m.drain_due(SimTime::from_millis(1000));
        assert_eq!(out[0].frame.robots_blue.len(), 1);
    }

    #[test]
    fn vision_calibration_warp_is_static_and_has_the_configured_spread() {
        let realism = Realism {
            object_position_offset: 0.0,
            calibration_warp_stddev: 0.012,
            calibration_warp_length: 3.0,
            calibration_orientation_offset: 0.02,
            calibration_orientation_warp: 0.02,
            ..Realism::none()
        };
        let mut m = model(auto_cameras(&field(), 2, 6.4, 0.2), realism);
        let mut rngs = Rngs::from_seed(11);
        m.ensure_calibration(&mut rngs);

        // Static: the same position always gets the same offset.
        let p = Vec2::new(1.234, -0.567);
        let a = m.calibration_offset(0, p);
        let b = m.calibration_offset(0, p);
        assert_eq!(a, b);
        assert!(a.length() > 0.0);
        // Per camera: the two cameras disagree about the same point.
        assert_ne!(m.calibration_offset(0, p), m.calibration_offset(1, p));

        // The spatial spread matches `calibration_warp_stddev` per axis.
        let mut xs = Vec::new();
        let mut phis = Vec::new();
        for i in 0..60 {
            for j in 0..45 {
                let q = Vec2::new(-6.0 + i as f64 * 0.2, -4.5 + j as f64 * 0.2);
                xs.push(m.calibration_offset(0, q).x);
                phis.push(m.calibration_orientation(0, q));
            }
        }
        let sd = |v: &[f64]| {
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let sx = sd(&xs);
        assert!(
            (sx / 0.012 - 1.0).abs() < 0.25,
            "warp stddev {sx} should be near 0.012"
        );
        let sphi = sd(&phis);
        assert!(
            (sphi / 0.02 - 1.0).abs() < 0.25,
            "orientation warp stddev {sphi} should be near 0.02"
        );
        // The constant orientation offset shifts the mean off zero.
        let mean_phi = phis.iter().sum::<f64>() / phis.len() as f64;
        assert!((mean_phi.abs() - 0.02).abs() < 0.005, "mean phi {mean_phi}");

        // It is applied to the reported detections, before the white noise.
        let bp = BallParams::default();
        let ball_at = Vec2::new(0.4, 0.9);
        let ball = Ball::at_rest(ball_at, &bp);
        let rs = robots(vec![robot_at(RobotId::new(Team::Blue, 2), p.x, p.y)]);
        let f = field();
        assert!(m.maybe_capture(SimTime::from_millis(1), &ball, &rs, &f, &bp, &mut rngs));
        let out = m.drain_due(SimTime::from_millis(1000));
        let o = out.iter().find(|o| o.camera_index == 0).unwrap();
        let r = o.frame.robots_blue[0];
        assert!((r.pos - (p + a)).length() < 1e-12, "robot carries the warp");
        assert!(
            (r.orientation - m.calibration_orientation(0, p)).abs() < 1e-12,
            "orientation carries the warp"
        );
        let b = o.frame.balls[0];
        assert!((b.pos - (ball_at + m.calibration_offset(0, ball_at))).length() < 1e-9);
    }

    #[test]
    fn vision_position_offset_is_constant_per_camera_in_a_random_direction() {
        let realism = Realism {
            object_position_offset: 0.023,
            ..Realism::none()
        };
        let mut m = model(auto_cameras(&field(), 2, 6.4, 0.2), realism);
        let mut rngs = Rngs::from_seed(5);
        m.ensure_calibration(&mut rngs);
        for ci in 0..2 {
            let o = m.position_offset(ci);
            assert!((o.length() - 0.023).abs() < 1e-12, "magnitude is the knob");
            // With no warp the offset is the whole static error, everywhere.
            assert!((m.calibration_offset(ci, Vec2::new(3.0, -2.0)) - o).length() < 1e-12);
        }
        assert_ne!(m.position_offset(0), m.position_offset(1));
    }

    #[test]
    fn vision_reports_ball_z_when_configured() {
        let config = VisionConfig {
            cameras: auto_cameras(&field(), 1, 6.4, 0.2),
            report_ball_z: true,
            ..VisionConfig::default()
        };
        let mut m = VisionModel::new(config, Realism::none(), &field());
        let mut rngs = Rngs::from_seed(1);
        let bp = BallParams::default();
        let mut ball = Ball::at_rest(Vec2::ZERO, &bp);
        ball.state.pos.z = 0.4;
        let rs = BTreeMap::new();
        let f = field();
        assert!(m.maybe_capture(SimTime::from_millis(1), &ball, &rs, &f, &bp, &mut rngs));
        let out = m.drain_due(SimTime::from_millis(1000));
        let b = out[0].frame.balls[0];
        assert_eq!(b.z, Some(0.4));
        // A camera at the origin sees a ball at the origin straight down.
        assert!(b.pos.length() < 1e-12);
    }

    #[test]
    fn vision_geometry_matches_the_field_and_cameras() {
        let realism = Realism {
            camera_position_error: 0.1,
            ..Realism::none()
        };
        let m = model(auto_cameras(&field(), 4, 6.4, 0.2), realism);
        let mut rngs = Rngs::from_seed(0);
        let f = field();
        let bp = BallParams::default();
        let g = m.geometry(&f, &bp, &mut rngs);
        assert_eq!(g.field, f);
        assert_eq!(g.ball, bp);
        assert_eq!(g.cameras.len(), 4);
        let error_dir = Vec3::new(0.3, 0.7, 0.05).normalize();
        for (i, c) in g.cameras.iter().enumerate() {
            assert_eq!(c.camera_id, i as u32);
            assert_eq!(c.focal_length, VisionConfig::default().focal_length_px);
            assert_eq!(c.principal_point, Vec2::new(300.0, 300.0));
            assert_eq!(c.distortion, 0.0);
            assert_eq!(c.q, [0.0, 1.0, 0.0, 0.0]);
            let truth = m.cameras()[i].position;
            let delta = c.position - truth;
            assert!((delta.length() - 0.1).abs() < 1e-12);
            assert!((delta - error_dir * 0.1).length() < 1e-12);
            // tx/ty/tz are consistent with the reported position.
            assert_eq!(
                c.translation(),
                Vec3::new(-c.position.x, c.position.y, c.position.z)
            );
        }
        // Deterministic: the same call gives the same answer.
        let again = m.geometry(&f, &bp, &mut rngs);
        assert_eq!(g, again);
    }

    #[test]
    fn vision_dropouts_and_noise_are_applied() {
        // Everything always dropped => empty frames.
        let realism = Realism {
            missing_robot_detections: 1.0,
            missing_ball_detections: 1.0,
            ..Realism::none()
        };
        let mut m = model(auto_cameras(&field(), 1, 6.4, 0.2), realism);
        let mut rngs = Rngs::from_seed(0);
        let bp = BallParams::default();
        let ball = Ball::at_rest(Vec2::ZERO, &bp);
        let rs = robots(vec![robot_at(RobotId::new(Team::Blue, 0), 1.0, 1.0)]);
        let f = field();
        assert!(m.maybe_capture(SimTime::from_millis(1), &ball, &rs, &f, &bp, &mut rngs));
        let out = m.drain_due(SimTime::from_millis(1000));
        assert!(out[0].frame.balls.is_empty());
        assert!(out[0].frame.robots_blue.is_empty());

        // Noise moves the reported position but keeps it close.
        let realism = Realism {
            stddev_robot_p: 0.01,
            ..Realism::none()
        };
        let mut m = model(auto_cameras(&field(), 1, 6.4, 0.2), realism);
        let mut rngs = Rngs::from_seed(0);
        assert!(m.maybe_capture(SimTime::from_millis(1), &ball, &rs, &f, &bp, &mut rngs));
        let out = m.drain_due(SimTime::from_millis(1000));
        let r = out[0].frame.robots_blue[0];
        let d = (r.pos - Vec2::new(1.0, 1.0)).length();
        assert!(d > 0.0 && d < 0.1, "noisy but plausible, got {d}");
    }

    #[test]
    fn vision_spurious_balls_sit_on_the_robot_centreline() {
        let config = VisionConfig {
            cameras: auto_cameras(&field(), 1, 6.4, 0.2),
            spurious_ball_forward: 0.13,
            spurious_ball_lateral_stddev: 0.0,
            spurious_ball_area_px: 32.0,
            ..VisionConfig::default()
        };
        let realism = Realism {
            // 1 per frame period: the probability clamps to 1.
            dribbler_ball_detections: 1e6,
            ..Realism::none()
        };
        let mut m = VisionModel::new(config, realism, &field());
        let mut rngs = Rngs::from_seed(0);
        let bp = BallParams::default();
        // The real ball is elsewhere but still in this camera's region.
        let ball = Ball::at_rest(Vec2::new(-4.0, 3.0), &bp);
        let robot = Robot::new(
            RobotId::new(Team::Blue, 0),
            RobotSpecs::default(),
            Vec2::ZERO,
            0.0,
        );
        let rs = robots(vec![robot]);
        let f = field();
        assert!(m.maybe_capture(SimTime::from_millis(1), &ball, &rs, &f, &bp, &mut rngs));
        let out = m.drain_due(SimTime::from_millis(1000));
        let balls = &out[0].frame.balls;
        assert_eq!(balls.len(), 2, "the real ball plus one phantom");
        let phantom = balls
            .iter()
            .find(|b| (b.pos - Vec2::new(0.13, 0.0)).length() < 1e-9)
            .expect("phantom on the centreline 0.13 m ahead of the centre");
        assert_eq!(phantom.area, Some(32.0), "spurious blobs are smaller");

        // With a lateral spread the phantom stays forward but wanders sideways.
        let config = VisionConfig {
            cameras: auto_cameras(&field(), 1, 6.4, 0.2),
            spurious_ball_forward: 0.13,
            spurious_ball_lateral_stddev: 0.07,
            ..VisionConfig::default()
        };
        let mut m = VisionModel::new(
            config,
            Realism {
                dribbler_ball_detections: 1e6,
                ..Realism::none()
            },
            &field(),
        );
        let mut rngs = Rngs::from_seed(1);
        let mut lateral = Vec::new();
        for t in 1..=1000u64 {
            m.maybe_capture(SimTime::from_millis(t), &ball, &rs, &f, &bp, &mut rngs);
        }
        for o in m.drain_due(SimTime::from_millis(20_000)) {
            for b in &o.frame.balls {
                if (b.pos.x - 0.13).abs() < 1e-9 {
                    lateral.push(b.pos.y);
                }
            }
        }
        assert!(lateral.len() > 50, "{} phantoms", lateral.len());
        let mean = lateral.iter().sum::<f64>() / lateral.len() as f64;
        let sd =
            (lateral.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / lateral.len() as f64).sqrt();
        assert!(mean.abs() < 0.02, "centred on the centreline, mean {mean}");
        assert!((sd / 0.07 - 1.0).abs() < 0.25, "lateral spread {sd}");
    }

    #[test]
    fn vision_area_is_flat_in_the_horizontal_and_grows_with_height() {
        let camera = Vec3::new(0.0, 0.0, 6.4);
        let bp = BallParams::default();
        let f = VisionConfig::default().focal_length_px;
        let a0 = VisionConfig::default().area_at_nadir_px;
        let area =
            |x: f64, z: f64| ball_area(Vec3::new(x, 0.0, z), bp.radius, camera, 1.0, f, a0, 0.0);
        // Flat from the nadir to 5 m out: the measured behaviour (vision.md §6).
        let base = area(0.0, bp.radius);
        assert!((base - a0).abs() < 1e-9, "at the nadir it is A0");
        for x in [0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.5] {
            let a = area(x, bp.radius);
            assert!(
                (a - base).abs() < 1e-9,
                "area must not depend on horizontal distance ({x} m gave {a})"
            );
        }
        // Height still grows it, and a little faster than the ideal pinhole.
        let high = area(0.0, 0.575);
        let pinhole = base * (6.4 / (6.4 - (0.575 - bp.radius))).powi(2);
        assert!(high > base * 1.05, "height term applies, got {high}");
        assert!(
            high > pinhole && high < pinhole * 1.15,
            "slightly steeper than a pinhole: {high} vs {pinhole}"
        );
        // Visibility scales it and noise adds to it.
        assert!(
            (area(0.0, bp.radius) * 0.5
                - ball_area(
                    Vec3::new(0.0, 0.0, bp.radius),
                    bp.radius,
                    camera,
                    0.5,
                    f,
                    a0,
                    0.0
                ))
            .abs()
                < 1e-9
        );
    }

    #[test]
    fn vision_area_can_be_omitted_entirely() {
        let config = VisionConfig {
            cameras: auto_cameras(&field(), 1, 6.4, 0.2),
            report_area: false,
            ..VisionConfig::default()
        };
        let mut m = VisionModel::new(config, Realism::none(), &field());
        let mut rngs = Rngs::from_seed(0);
        let bp = BallParams::default();
        let ball = Ball::at_rest(Vec2::ZERO, &bp);
        let rs = BTreeMap::new();
        let f = field();
        assert!(m.maybe_capture(SimTime::from_millis(1), &ball, &rs, &f, &bp, &mut rngs));
        let out = m.drain_due(SimTime::from_millis(1000));
        assert_eq!(out[0].frame.balls[0].area, None, "no area on the wire");
    }
}

//! Vision model: camera regions, floor projection of the ball, occlusion,
//! noise, dropouts, spurious detections, and the delay queue.
//!
//! OWNER: general agent A. Replace the `todo!()` bodies; keep the public API.
//! See `docs/design.md` §5.5.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::ball::Ball;
use crate::field::FieldGeometry;
use crate::params::{BallParams, CameraConfig, Realism, VisionConfig};
use crate::rng::{chance, normal, normal_vec2, Rngs};
use crate::robot::Robot;
use crate::types::{RobotId, SimTime, Team, Vec2, Vec3};

/// ER-Force's arbitrary pixel-area scale factor: the reported `area` is the
/// modelled pixel area multiplied by this ("just make it similar to a real game").
pub const PIXEL_PER_AREA: f64 = 10.0;

/// Fraction of the camera height above which a flying ball is no longer
/// projected (the projection would diverge at the camera plane).
pub const PROJECTION_SCALING_LIMIT: f64 = 0.9;

/// Number of samples per axis of the occlusion grid (ER-Force uses 15x15).
pub const OCCLUSION_SAMPLE_RADIUS: i32 = 7;

/// Fixed direction of the reported-calibration position error (ER-Force uses a
/// deterministic `normalize(0.3, 0.7, 0.05)` offset, not a random one).
const CAMERA_ERROR_DIRECTION: Vec3 = Vec3::new(0.3, 0.7, 0.05);

/// Distance in front of the kicker face at which spurious "dribbler balls" appear.
const DRIBBLER_BALL_FORWARD: f64 = 0.03;

/// A detected ball.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DetectedBall {
    /// Reported horizontal position [m] (floor-projected, noisy).
    pub pos: Vec2,
    /// Reported height [m], only when `report_ball_z`.
    pub z: Option<f64>,
    /// Reported area [px].
    pub area: f64,
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
    /// Reported height [m].
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
    /// Ball model constants.
    pub ball: BallParams,
}

/// The vision generator: owns per-camera counters and the delay queue.
#[derive(Debug, Clone)]
pub struct VisionModel {
    config: VisionConfig,
    realism: Realism,
    cameras: Vec<CameraConfig>,
    frame_numbers: Vec<u32>,
    frames_since_geometry: u32,
    next_frame_time: SimTime,
    queue: VecDeque<(SimTime, VisionOutput)>,
}

/// One camera set's output: all detection frames for one capture instant plus
/// an optional geometry packet to attach to camera 0's wrapper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisionOutput {
    /// Sim time of capture.
    pub capture_time: SimTime,
    /// One frame per camera, in camera order.
    pub frames: Vec<DetectionFrame>,
    /// Geometry to attach, if due this frame.
    pub geometry: Option<GeometryData>,
}

impl VisionModel {
    /// Create the model; auto-places cameras when `config.cameras` is empty.
    pub fn new(config: VisionConfig, realism: Realism, field: &FieldGeometry) -> Self {
        let cameras = if config.cameras.is_empty() {
            auto_cameras(
                field,
                config.default_camera_count,
                config.default_camera_height,
            )
        } else {
            config.cameras.clone()
        };
        let n = cameras.len();
        Self {
            config,
            realism,
            cameras,
            frame_numbers: vec![0; n],
            frames_since_geometry: u32::MAX,
            next_frame_time: SimTime::ZERO,
            queue: VecDeque::new(),
        }
    }

    /// Replace the realism config.
    pub fn set_realism(&mut self, realism: Realism) {
        self.realism = realism;
    }

    /// Replace the camera rig (e.g. from a `SimulatorConfig.geometry.calib`
    /// update). Per-camera frame counters restart and the geometry packet is
    /// re-sent on the next capture; queued outputs are kept. An empty list
    /// re-derives the default rig from the field.
    pub fn set_cameras(&mut self, cameras: Vec<CameraConfig>, field: &FieldGeometry) {
        self.cameras = if cameras.is_empty() {
            auto_cameras(
                field,
                self.config.default_camera_count,
                self.config.default_camera_height,
            )
        } else {
            cameras
        };
        self.frame_numbers = vec![0; self.cameras.len()];
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

    /// Sim time of the next capture.
    pub fn next_capture_time(&self) -> SimTime {
        self.next_frame_time
    }

    /// Interval between captures, derived from `config.frame_rate`.
    pub fn frame_period(&self) -> SimTime {
        let hz = if self.config.frame_rate > 0.0 {
            self.config.frame_rate
        } else {
            60.0
        };
        SimTime::from_secs_f64(1.0 / hz).max(SimTime(1))
    }

    /// Fixed radial position offset this camera applies to every reported
    /// object (ER-Force `object_position_offset`).
    pub fn position_offset(&self, cam: usize) -> Vec2 {
        let strength = self.realism.object_position_offset;
        if strength < 1e-9 {
            return Vec2::ZERO;
        }
        let Some(c) = self.cameras.get(cam) else {
            return Vec2::ZERO;
        };
        let xy = Vec2::new(c.position.x, c.position.y);
        if xy.length() < strength {
            xy
        } else {
            xy.normalize_or_zero() * strength
        }
    }

    /// Called once per substep after physics. If a capture is due at `now`,
    /// generate frames for all cameras and enqueue them with due time
    /// `now + vision_delay`. Returns true if a capture happened.
    pub fn maybe_capture(
        &mut self,
        now: SimTime,
        ball: &Ball,
        robots: &BTreeMap<RobotId, Robot>,
        field: &FieldGeometry,
        ball_params: &BallParams,
        rngs: &mut Rngs,
    ) -> bool {
        if now < self.next_frame_time || self.cameras.is_empty() {
            return false;
        }
        let period = self.frame_period();
        let period_secs = period.as_secs_f64();

        let epoch = self.config.timestamp_epoch_offset;
        let t_sent = now.as_secs_f64() + self.realism.vision_delay + epoch;
        let t_capture = t_sent - self.realism.vision_processing_time;

        let mut frames = Vec::with_capacity(self.cameras.len());
        for ci in 0..self.cameras.len() {
            let camera = self.cameras[ci];
            let frame_number = self.frame_numbers[ci];
            self.frame_numbers[ci] = frame_number.wrapping_add(1);
            let offset = self.position_offset(ci);

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
                let noise = normal_vec2(&mut rngs.vision_noise, self.realism.stddev_robot_p);
                let phi = normal(&mut rngs.vision_noise, self.realism.stddev_robot_phi);
                let det = DetectedRobot {
                    number: id.number,
                    pos: robot.pos + offset + noise,
                    orientation: robot.orientation + phi,
                    height: robot.specs.height,
                    confidence: 1.0,
                };
                match id.team {
                    Team::Blue => frame.robots_blue.push(det),
                    Team::Yellow => frame.robots_yellow.push(det),
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
                    let area = ball_area(
                        centroid,
                        ball_params.radius,
                        camera.position,
                        visibility,
                        self.config.focal_length_px,
                        area_noise,
                    );
                    let noise = normal_vec2(&mut rngs.vision_noise, self.realism.stddev_ball_p);
                    frame.balls.push(DetectedBall {
                        pos: projected + offset + noise,
                        z: if self.config.report_ball_z {
                            Some(ball_pos.z)
                        } else {
                            None
                        },
                        area,
                        confidence: 1.0,
                    });
                }
            }

            // --- spurious "dribbler balls" (the red IR beam of many teams) ---
            let spurious_p = self.realism.dribbler_ball_detections * period_secs;
            if spurious_p > 0.0 {
                for robot in robots.values() {
                    if !chance(&mut rngs.vision_dropout, spurious_p) {
                        continue;
                    }
                    let corner = dribbler_corner(robot);
                    if !self.visible_in_camera(ci, corner) {
                        continue;
                    }
                    let pos3 = Vec3::new(corner.x, corner.y, ball_params.radius);
                    let noise = normal_vec2(&mut rngs.vision_noise, self.realism.stddev_robot_p);
                    let area = ball_area(
                        pos3,
                        ball_params.radius,
                        camera.position,
                        1.0,
                        self.config.focal_length_px,
                        0.0,
                    );
                    frame.balls.push(DetectedBall {
                        pos: corner + offset + noise,
                        z: if self.config.report_ball_z {
                            Some(ball_params.radius)
                        } else {
                            None
                        },
                        area,
                        confidence: 1.0,
                    });
                }
            }

            // Trackers can have systematic errors depending on the ball order.
            if frame.balls.len() > 1 {
                crate::rng::shuffle(&mut rngs.shuffle, &mut frame.balls);
            }

            frames.push(frame);
        }

        let every = self.config.geometry_every_n_frames.max(1);
        let geometry = if self.frames_since_geometry >= every {
            self.frames_since_geometry = 0;
            Some(self.geometry(field, ball_params, rngs))
        } else {
            None
        };
        self.frames_since_geometry = self.frames_since_geometry.saturating_add(1);

        let output = VisionOutput {
            capture_time: now,
            frames,
            geometry,
        };
        let due = now.plus(SimTime::from_secs_f64(self.realism.vision_delay));
        self.queue.push_back((due, output));

        self.next_frame_time = self.next_frame_time.plus(period);
        while self.next_frame_time <= now {
            self.next_frame_time = self.next_frame_time.plus(period);
        }
        true
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
    /// *calibration* error, exactly like ER-Force). `rngs` is unused.
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

    /// ER-Force camera region rule: a point is visible to camera `cam` if its
    /// Manhattan distance to that camera is within `2 * camera_overlap` of the
    /// minimum Manhattan distance over all cameras.
    pub fn visible_in_camera(&self, cam: usize, p: Vec2) -> bool {
        let Some(own) = self.cameras.get(cam) else {
            return false;
        };
        let manhattan = |c: &CameraConfig| (c.position.x - p.x).abs() + (c.position.y - p.y).abs();
        let own_distance = manhattan(own);
        let mut min_distance = f64::INFINITY;
        for c in &self.cameras {
            min_distance = min_distance.min(manhattan(c));
        }
        own_distance <= min_distance + 2.0 * self.realism.camera_overlap
    }
}

/// Place `count` cameras (4: quadrants at (+-L/4, +-W/4); 2: (+-L/4, 0); 1: origin).
///
/// Ids are `0..count` in the order returned: for four cameras that is
/// `(+,+), (-,+), (-,-), (+,-)`; for two `(+, 0), (-, 0)`. Counts other than
/// 1, 2 and 4 are rounded down to the nearest supported layout (0 becomes 1).
pub fn auto_cameras(field: &FieldGeometry, count: u32, height: f64) -> Vec<CameraConfig> {
    let qx = field.length * 0.25;
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

/// Reported ball `area` [px] for a ball whose (possibly occlusion-shifted)
/// centre is at `ball_pos`, seen from `camera`. ER-Force's model: the pixel
/// area of a disc of `radius` at the projected distance, scaled by the visible
/// fraction, with `area_noise` (a pre-drawn Gaussian sample in pixels) added.
pub fn ball_area(
    ball_pos: Vec3,
    radius: f64,
    camera: Vec3,
    visibility: f64,
    focal_length_px: f64,
    area_noise: f64,
) -> f64 {
    let z = (ball_pos.z - radius)
        .max(0.0)
        .min(PROJECTION_SCALING_LIMIT * camera.z.max(0.0));
    let dist = Vec3::new(camera.x - ball_pos.x, camera.y - ball_pos.y, camera.z - z).length();
    let denom = dist * 1000.0 / focal_length_px.max(1e-9) - 1.0;
    if denom <= 1e-9 {
        return 0.0;
    }
    let base = radius * radius * 1e6 * std::f64::consts::PI / (denom * denom);
    let area = visibility * (base + area_noise / PIXEL_PER_AREA).max(0.0);
    area * PIXEL_PER_AREA
}

/// World position of the spurious "dribbler ball" of a robot: on the right-hand
/// corner of the dribbler, 3 cm in front of the kicker face.
pub fn dribbler_corner(robot: &Robot) -> Vec2 {
    let heading = robot.heading();
    let right = Vec2::new(heading.y, -heading.x);
    robot.kicker_face_center()
        + heading * DRIBBLER_BALL_FORWARD
        + right * (robot.specs.dribbler_width * 0.5)
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
    use crate::params::RobotSpecs;
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

    #[test]
    fn vision_auto_camera_placement() {
        let f = field();
        let four = auto_cameras(&f, 4, 4.0);
        assert_eq!(four.len(), 4);
        assert_eq!(
            four.iter().map(|c| c.id).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(four[0].position, Vec3::new(3.0, 2.25, 4.0));
        assert_eq!(four[1].position, Vec3::new(-3.0, 2.25, 4.0));
        assert_eq!(four[2].position, Vec3::new(-3.0, -2.25, 4.0));
        assert_eq!(four[3].position, Vec3::new(3.0, -2.25, 4.0));

        let two = auto_cameras(&f, 2, 4.0);
        assert_eq!(two.len(), 2);
        assert_eq!(two[0].position, Vec3::new(3.0, 0.0, 4.0));
        assert_eq!(two[1].position, Vec3::new(-3.0, 0.0, 4.0));

        let one = auto_cameras(&f, 1, 4.0);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].position, Vec3::new(0.0, 0.0, 4.0));
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
    fn vision_capture_cadence_is_60_hz() {
        let realism = Realism::none();
        let mut m = model(auto_cameras(&field(), 4, 4.0), realism);
        let mut rngs = Rngs::from_seed(0);
        let ball = Ball::at_rest(Vec2::ZERO, &BallParams::default());
        let rs = BTreeMap::new();
        let f = field();
        let bp = BallParams::default();

        let mut captures = 0;
        for ms in 1..=1000u64 {
            if m.maybe_capture(SimTime::from_millis(ms), &ball, &rs, &f, &bp, &mut rngs) {
                captures += 1;
            }
        }
        assert_eq!(captures, 60, "60 Hz over one second");

        let outputs = m.drain_due(SimTime::from_millis(10_000));
        assert_eq!(outputs.len(), 60);
        for o in &outputs {
            assert_eq!(o.frames.len(), 4);
            let ids: Vec<u32> = o.frames.iter().map(|fr| fr.camera_id).collect();
            assert_eq!(ids, vec![0, 1, 2, 3]);
        }
        // Per-camera frame numbers count up independently.
        for (i, o) in outputs.iter().enumerate() {
            for fr in &o.frames {
                assert_eq!(fr.frame_number, i as u32);
            }
        }
        // Geometry on the first frame, then every 30.
        assert!(outputs[0].geometry.is_some());
        for (i, o) in outputs.iter().enumerate() {
            assert_eq!(o.geometry.is_some(), i % 30 == 0, "frame {i}");
        }
    }

    #[test]
    fn vision_delay_queue_releases_at_the_right_time() {
        let realism = Realism::none();
        let delay = realism.vision_delay;
        let mut m = model(auto_cameras(&field(), 2, 4.0), realism);
        let mut rngs = Rngs::from_seed(0);
        let ball = Ball::at_rest(Vec2::ZERO, &BallParams::default());
        let rs = BTreeMap::new();
        let f = field();
        let bp = BallParams::default();

        let capture = SimTime::from_millis(1);
        assert!(m.maybe_capture(capture, &ball, &rs, &f, &bp, &mut rngs));
        assert_eq!(m.pending(), 1);

        let due = capture.plus(SimTime::from_secs_f64(delay));
        assert!(m.drain_due(due.minus(SimTime(1))).is_empty(), "not yet due");
        let out = m.drain_due(due);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].capture_time, capture);
        assert_eq!(m.pending(), 0);

        // Timestamps carry the delay and the processing time.
        let frame = &out[0].frames[0];
        assert!((frame.t_sent - (capture.as_secs_f64() + delay)).abs() < 1e-12);
        assert!(
            (frame.t_capture - (frame.t_sent - Realism::none().vision_processing_time)).abs()
                < 1e-12
        );
    }

    #[test]
    fn vision_reports_exact_positions_with_realism_none() {
        let mut m = model(auto_cameras(&field(), 4, 4.0), Realism::none());
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
        assert_eq!(out.len(), 1);

        let mut blue_seen = 0;
        let mut yellow_seen = 0;
        let mut balls_seen = 0;
        for frame in &out[0].frames {
            for r in &frame.robots_blue {
                assert_eq!(r.number, 3);
                assert!((r.pos - Vec2::new(-1.0, 0.5)).length() < 1e-12);
                assert!((r.orientation - 0.7).abs() < 1e-12);
                assert!((r.height - RobotSpecs::default().height).abs() < 1e-12);
                assert_eq!(r.confidence, 1.0);
                blue_seen += 1;
            }
            for r in &frame.robots_yellow {
                assert_eq!(r.number, 5);
                assert!((r.pos - Vec2::new(1.0, -0.5)).length() < 1e-12);
                yellow_seen += 1;
            }
            for b in &frame.balls {
                // Grounded ball: the projection is the identity.
                assert!((b.pos - Vec2::new(0.25, -0.1)).length() < 1e-9);
                assert!(b.z.is_none(), "z is omitted unless report_ball_z");
                assert!(b.area > 0.0);
                balls_seen += 1;
            }
        }
        assert!(blue_seen > 0 && yellow_seen > 0 && balls_seen > 0);
    }

    #[test]
    fn vision_reports_ball_z_when_configured() {
        let config = VisionConfig {
            cameras: auto_cameras(&field(), 1, 4.0),
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
        let b = out[0].frames[0].balls[0];
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
        let m = model(auto_cameras(&field(), 4, 4.0), realism);
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
            assert_eq!(c.focal_length, 390.0);
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
        let mut m = model(auto_cameras(&field(), 1, 4.0), realism);
        let mut rngs = Rngs::from_seed(0);
        let bp = BallParams::default();
        let ball = Ball::at_rest(Vec2::ZERO, &bp);
        let rs = robots(vec![robot_at(RobotId::new(Team::Blue, 0), 1.0, 1.0)]);
        let f = field();
        assert!(m.maybe_capture(SimTime::from_millis(1), &ball, &rs, &f, &bp, &mut rngs));
        let out = m.drain_due(SimTime::from_millis(1000));
        assert!(out[0].frames[0].balls.is_empty());
        assert!(out[0].frames[0].robots_blue.is_empty());

        // Noise moves the reported position but keeps it close.
        let realism = Realism {
            stddev_robot_p: 0.01,
            ..Realism::none()
        };
        let mut m = model(auto_cameras(&field(), 1, 4.0), realism);
        let mut rngs = Rngs::from_seed(0);
        assert!(m.maybe_capture(SimTime::from_millis(1), &ball, &rs, &f, &bp, &mut rngs));
        let out = m.drain_due(SimTime::from_millis(1000));
        let r = out[0].frames[0].robots_blue[0];
        let d = (r.pos - Vec2::new(1.0, 1.0)).length();
        assert!(d > 0.0 && d < 0.1, "noisy but plausible, got {d}");
    }

    #[test]
    fn vision_spurious_dribbler_balls_appear_at_the_dribbler() {
        let realism = Realism {
            // 1 per frame period: probability clamps to 1.
            dribbler_ball_detections: 1e6,
            ..Realism::none()
        };
        let mut m = model(auto_cameras(&field(), 1, 4.0), realism);
        let mut rngs = Rngs::from_seed(0);
        let bp = BallParams::default();
        // Put the real ball far away so it lands outside... it is still seen by
        // the single camera, so expect two balls.
        let ball = Ball::at_rest(Vec2::new(-4.0, 3.0), &bp);
        let robot = robot_at(RobotId::new(Team::Blue, 0), 0.0, 0.0);
        let expected = dribbler_corner(&robot);
        let rs = robots(vec![robot]);
        let f = field();
        assert!(m.maybe_capture(SimTime::from_millis(1), &ball, &rs, &f, &bp, &mut rngs));
        let out = m.drain_due(SimTime::from_millis(1000));
        let balls = &out[0].frames[0].balls;
        assert_eq!(balls.len(), 2, "the real ball plus one phantom");
        assert!(
            balls.iter().any(|b| (b.pos - expected).length() < 1e-9),
            "phantom ball at the dribbler corner {expected:?}, got {balls:?}"
        );
        // 3 cm in front of the kicker face, on the right-hand corner.
        let specs = RobotSpecs::default();
        assert!((expected.x - (specs.center_to_dribbler + 0.03)).abs() < 1e-12);
        assert!((expected.y + specs.dribbler_width * 0.5).abs() < 1e-12);
    }

    #[test]
    fn vision_out_of_region_objects_are_not_reported() {
        let m = model(
            vec![cam(0, 3.0, 2.25), cam(1, -3.0, 2.25)],
            Realism {
                camera_overlap: 0.1,
                ..Realism::none()
            },
        );
        assert!(m.visible_in_camera(0, Vec2::new(3.0, 2.25)));
        assert!(!m.visible_in_camera(1, Vec2::new(3.0, 2.25)));
        assert!(!m.visible_in_camera(9, Vec2::ZERO), "unknown camera");
    }
}

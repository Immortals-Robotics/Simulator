//! Tunable parameters: ball physics, per-robot specs, vision, realism, and the
//! top-level simulation config. Defaults are the values settled in
//! `docs/design.md` §5. All structs are `serde` so the CLI can load them from TOML.

use serde::{Deserialize, Serialize};

use crate::types::Vec3;

/// Ball physical parameters. These are also what the geometry packet advertises.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BallParams {
    /// Radius [m].
    pub radius: f64,
    /// Mass [kg].
    pub mass: f64,
    /// Sliding deceleration [m/s^2] (negative).
    pub acc_slide: f64,
    /// Rolling deceleration at zero speed [m/s^2] (negative).
    pub acc_roll: f64,
    /// Extra rolling deceleration per m/s of speed [1/s], >= 0. The rolling
    /// deceleration is `acc_roll - roll_speed_coefficient * v` (closed form kept).
    pub roll_speed_coefficient: f64,
    /// Speed [m/s] at which the geometry packet advertises `acc_roll` for clients
    /// that assume a constant.
    pub advertise_roll_speed: f64,
    /// Inertia distribution p = I / (m r^2); 0.4 solid sphere, 0.66 hollow.
    /// The published `k_switch` is `1 / (1 + p)`.
    pub inertia_distribution: f64,
    /// Horizontal velocity factor kept on the first bounce of a chip.
    pub chip_damping_xy_first_hop: f64,
    /// Horizontal velocity factor kept on later bounces.
    pub chip_damping_xy_other_hops: f64,
    /// Vertical velocity factor kept on the first bounce of a chip.
    pub chip_damping_z: f64,
    /// Vertical velocity factor kept on later bounces.
    pub chip_damping_z_other_hops: f64,
    /// Below this apex height [m] the ball is considered grounded.
    pub min_hop_height: f64,
    /// Below this speed [m/s] the ball snaps to rest.
    pub rest_speed: f64,
}

impl BallParams {
    /// Published slide-to-roll velocity ratio.
    pub fn k_switch(&self) -> f64 {
        1.0 / (1.0 + self.inertia_distribution)
    }

    /// Rolling deceleration [m/s^2] at speed `v` (negative).
    pub fn acc_roll_at(&self, v: f64) -> f64 {
        self.acc_roll - self.roll_speed_coefficient.max(0.0) * v.abs()
    }

    /// Rolling deceleration advertised in the geometry packet.
    pub fn advertised_acc_roll(&self) -> f64 {
        self.acc_roll_at(self.advertise_roll_speed)
    }

    /// Named presets: `default` (pooled 2026 fit), `go26` (German Open 2026
    /// carpet), `rc26` (RoboCup 2026 carpet), `erforce`, `tigers` (values those
    /// simulators advertise), `grsim`.
    pub fn preset(name: &str) -> Option<Self> {
        let d = Self::default();
        Some(match name.to_ascii_lowercase().as_str() {
            "default" | "measured" => d,
            "go26" => Self {
                acc_roll: -0.19,
                acc_slide: -3.42,
                chip_damping_xy_first_hop: 0.73,
                ..d
            },
            "rc26" => Self {
                acc_roll: -0.26,
                acc_slide: -3.10,
                chip_damping_xy_first_hop: 0.67,
                ..d
            },
            "erforce" => Self {
                acc_roll: -0.35,
                roll_speed_coefficient: 0.0,
                acc_slide: -3.9,
                inertia_distribution: 1.0 / 0.69 - 1.0,
                chip_damping_xy_first_hop: 0.715,
                chip_damping_xy_other_hops: 1.0,
                chip_damping_z: 0.566,
                chip_damping_z_other_hops: 0.566,
                ..d
            },
            "tigers" => Self {
                acc_roll: -0.26,
                roll_speed_coefficient: 0.0,
                acc_slide: -3.0,
                chip_damping_xy_first_hop: 0.75,
                chip_damping_xy_other_hops: 0.95,
                chip_damping_z: 0.5,
                chip_damping_z_other_hops: 0.5,
                ..d
            },
            "grsim" => Self {
                acc_roll: -0.49,
                roll_speed_coefficient: 0.0,
                acc_slide: -0.49,
                chip_damping_xy_first_hop: 0.6,
                chip_damping_xy_other_hops: 0.96,
                chip_damping_z: 0.42,
                chip_damping_z_other_hops: 0.42,
                ..d
            },
            _ => return None,
        })
    }
}

impl Default for BallParams {
    fn default() -> Self {
        Self {
            radius: 0.0215,
            mass: 0.046,
            acc_slide: -3.3,
            acc_roll: -0.22,
            roll_speed_coefficient: 0.045,
            advertise_roll_speed: 1.5,
            inertia_distribution: 0.5,
            chip_damping_xy_first_hop: 0.70,
            chip_damping_xy_other_hops: 0.87,
            chip_damping_z: 0.46,
            chip_damping_z_other_hops: 0.40,
            min_hop_height: 0.01,
            rest_speed: 0.01,
        }
    }
}

/// Collision restitution/friction parameters for ball contacts. `damp_normal`
/// = 1 is perfectly inelastic, 0 is perfectly elastic (Sumatra convention).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContactParams {
    /// Ball vs robot hull (round part).
    pub ball_robot_normal: f64,
    /// Ball vs robot hull tangential blend toward the surface velocity.
    pub ball_robot_tangent: f64,
    /// Ball vs kicker face normal damping.
    pub ball_kicker_normal: f64,
    /// Ball vs kicker face tangential blend.
    pub ball_kicker_tangent: f64,
    /// Ball vs boundary boards / goal frame normal damping.
    pub ball_wall_normal: f64,
    /// Ball vs boundary boards tangential blend.
    pub ball_wall_tangent: f64,
    /// Factor applied to the ball spin on any contact (1 = keep, 0 = kill).
    pub spin_retention: f64,
    /// Robot vs robot coefficient of restitution.
    pub robot_robot_restitution: f64,
    /// Robot vs robot tangential friction blend.
    pub robot_robot_friction: f64,
    /// Robot vs boundary restitution.
    pub robot_wall_restitution: f64,
    /// Ball landing on a robot's flat top: vertical damping (1 = inelastic).
    pub ball_robot_top_normal: f64,
    /// Ball landing on a robot's flat top: horizontal blend toward the robot's surface velocity.
    pub ball_robot_top_tangent: f64,
    /// Use the flat front chord in robot-robot contacts (false = discs only).
    pub robot_hull_chord_contacts: bool,
}

impl Default for ContactParams {
    fn default() -> Self {
        Self {
            ball_robot_normal: 0.5,
            ball_robot_tangent: 0.3,
            ball_kicker_normal: 0.8,
            ball_kicker_tangent: 0.4,
            ball_wall_normal: 0.5,
            ball_wall_tangent: 0.0,
            spin_retention: 0.6,
            robot_robot_restitution: 0.2,
            robot_robot_friction: 0.1,
            robot_wall_restitution: 0.1,
            ball_robot_top_normal: 0.54,
            ball_robot_top_tangent: 0.3,
            robot_hull_chord_contacts: true,
        }
    }
}

/// Movement limits enforced by the simulated firmware (protocol `RobotLimits`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RobotLimits {
    /// Max absolute speed-up acceleration [m/s^2].
    pub acc_speedup_absolute_max: f64,
    /// Max angular speed-up acceleration [rad/s^2].
    pub acc_speedup_angular_max: f64,
    /// Max absolute brake acceleration [m/s^2].
    pub acc_brake_absolute_max: f64,
    /// Max angular brake acceleration [rad/s^2].
    pub acc_brake_angular_max: f64,
    /// Max absolute velocity [m/s].
    pub vel_absolute_max: f64,
    /// Max angular velocity [rad/s].
    pub vel_angular_max: f64,
}

impl Default for RobotLimits {
    fn default() -> Self {
        Self {
            acc_speedup_absolute_max: 3.5,
            acc_speedup_angular_max: 40.0,
            acc_brake_absolute_max: 5.0,
            acc_brake_angular_max: 40.0,
            vel_absolute_max: 3.0,
            vel_angular_max: 12.0,
        }
    }
}

impl RobotLimits {
    /// Named presets fitted from 2026 game logs: `default`, `tigers`,
    /// `erforce`, `kiks`, `fast` (upper envelope), `grsim` (grSim's limits).
    pub fn preset(name: &str) -> Option<Self> {
        let d = Self::default();
        Some(match name.to_ascii_lowercase().as_str() {
            "default" | "measured" => d,
            "tigers" => Self {
                acc_speedup_absolute_max: 3.2,
                acc_brake_absolute_max: 4.2,
                vel_absolute_max: 3.3,
                acc_speedup_angular_max: 45.0,
                acc_brake_angular_max: 45.0,
                ..d
            },
            "erforce" => Self {
                acc_speedup_absolute_max: 3.7,
                acc_brake_absolute_max: 5.5,
                vel_absolute_max: 3.0,
                acc_speedup_angular_max: 45.0,
                acc_brake_angular_max: 45.0,
                ..d
            },
            "kiks" => Self {
                acc_speedup_absolute_max: 4.2,
                acc_brake_absolute_max: 5.5,
                vel_absolute_max: 3.4,
                acc_speedup_angular_max: 45.0,
                acc_brake_angular_max: 45.0,
                ..d
            },
            "fast" => Self {
                acc_speedup_absolute_max: 4.5,
                acc_brake_absolute_max: 6.0,
                vel_absolute_max: 3.5,
                acc_speedup_angular_max: 50.0,
                acc_brake_angular_max: 50.0,
                vel_angular_max: 15.0,
            },
            "grsim" => Self {
                acc_speedup_absolute_max: 4.0,
                acc_brake_absolute_max: 4.0,
                vel_absolute_max: 5.0,
                acc_speedup_angular_max: 50.0,
                acc_brake_angular_max: 50.0,
                vel_angular_max: 20.0,
            },
            _ => return None,
        })
    }
}

/// Wheel mounting angles [rad], CCW from robot +x (forward) to the wheel's
/// radial direction. Protocol order. The protocol documents them clockwise;
/// `ssl-sim-net` negates on the way in.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WheelAngles {
    /// Front right.
    pub front_right: f64,
    /// Back right.
    pub back_right: f64,
    /// Back left.
    pub back_left: f64,
    /// Front left.
    pub front_left: f64,
}

impl WheelAngles {
    /// Angles in protocol order.
    pub fn as_array(&self) -> [f64; 4] {
        [
            self.front_right,
            self.back_right,
            self.back_left,
            self.front_left,
        ]
    }
}

impl Default for WheelAngles {
    fn default() -> Self {
        Self {
            front_right: (-60.0f64).to_radians(),
            back_right: (-135.0f64).to_radians(),
            back_left: (135.0f64).to_radians(),
            front_left: (60.0f64).to_radians(),
        }
    }
}

/// How the robot body is driven.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriveModel {
    /// Four omni wheels with motor force and traction limits (default).
    Wheels,
    /// Kinematic: the firmware-limited setpoint is integrated directly.
    Ideal,
}

/// Motor and traction parameters for the wheel drive model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DriveParams {
    /// Drive model.
    pub model: DriveModel,
    /// Wheel radius [m].
    pub wheel_radius: f64,
    /// Distance from robot centre to wheel contact point [m].
    pub wheel_mount_radius: f64,
    /// Max wheel surface speed [m/s].
    pub max_wheel_speed: f64,
    /// Max drive force per wheel [N] (motor torque / wheel radius).
    pub max_wheel_force: f64,
    /// Drive force per m/s of wheel-speed error [N/(m/s)].
    pub velocity_gain: f64,
    /// Friction coefficient along the wheel drive direction.
    pub mu_drive: f64,
    /// Friction coefficient along the roller (free) direction.
    pub mu_lateral: f64,
    /// Deceleration when motors are off [m/s^2].
    pub coast_decel: f64,
    /// Angular deceleration when motors are off [rad/s^2].
    pub coast_angular_decel: f64,
}

impl Default for DriveParams {
    fn default() -> Self {
        Self {
            model: DriveModel::Wheels,
            wheel_radius: 0.027,
            wheel_mount_radius: 0.0875,
            max_wheel_speed: 5.0,
            max_wheel_force: 6.0,
            velocity_gain: 60.0,
            mu_drive: 0.8,
            mu_lateral: 0.05,
            coast_decel: 8.0,
            coast_angular_decel: 50.0,
        }
    }
}

/// Kicker parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KickerParams {
    /// Max straight kick speed [m/s].
    pub max_linear_speed: f64,
    /// Max chip kick speed [m/s].
    pub max_chip_speed: f64,
    /// Minimum kick speed [m/s]; commands below this are ignored.
    pub min_speed: f64,
    /// Time to recharge after a kick [s].
    pub charge_time: f64,
    /// Ball centre must be below this height [m] to be kicked.
    pub max_ball_height: f64,
    /// Fraction of the ball's incoming normal velocity cancelled by the kick (0..1).
    pub incoming_damping: f64,
}

impl Default for KickerParams {
    fn default() -> Self {
        Self {
            max_linear_speed: 6.5,
            max_chip_speed: 5.5,
            min_speed: 0.05,
            charge_time: 0.1,
            max_ball_height: 0.05,
            incoming_damping: 1.0,
        }
    }
}

/// Dribbler parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DribblerParams {
    /// Speed [rpm] at which the holding force is at its maximum.
    pub max_speed_rpm: f64,
    /// Mean max acceleration [m/s^2] the dribbler can impart to the ball at full speed.
    pub hold_accel: f64,
    /// Per-robot spread of `hold_accel` [m/s^2]; each robot draws its own value
    /// from N(hold_accel, hold_accel_stddev) with the seeded physics stream (0 = identical robots).
    pub hold_accel_stddev: f64,
    /// Lower clamp for the per-robot draw [m/s^2].
    pub hold_accel_min: f64,
    /// Roller radius [m], used for the back-spin imparted to the ball.
    pub roller_radius: f64,
    /// Depth [m] behind the kicker face where the ball is pulled to (seated point).
    pub seat_depth: f64,
    /// Perfect "glue" mode: ball is snapped to the seated point while dribbling.
    pub glue: bool,
}

impl Default for DribblerParams {
    fn default() -> Self {
        Self {
            max_speed_rpm: 10_000.0,
            hold_accel: 3.0,
            hold_accel_stddev: 1.0,
            hold_accel_min: 1.5,
            roller_radius: 0.007,
            seat_depth: 0.0,
            glue: false,
        }
    }
}

/// Complete per-robot specification. Applied live from `RobotSpecs` messages;
/// missing fields keep their current values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RobotSpecs {
    /// Hull radius [m].
    pub radius: f64,
    /// Height [m].
    pub height: f64,
    /// Mass [kg].
    pub mass: f64,
    /// Yaw moment of inertia [kg m^2]; `None` = 0.5 m r^2.
    pub inertia_z: Option<f64>,
    /// Distance from centre to the kicker face [m]; defines the flat front chord.
    pub center_to_dribbler: f64,
    /// Width of the dribbler bar [m].
    pub dribbler_width: f64,
    /// Ball centre distance from robot centre when seated on the dribbler [m].
    /// Must be >= center_to_dribbler + ball radius - dribbler.seat_depth so the
    /// seated ball sits in front of the kicker face (the hull chord is solid).
    /// Measured 0.097-0.099 m in 2026 games (ball touching the face plane).
    pub shoot_radius: f64,
    /// Firmware limits.
    pub limits: RobotLimits,
    /// Wheel angles.
    pub wheel_angles: WheelAngles,
    /// Drive model parameters.
    pub drive: DriveParams,
    /// Kicker.
    pub kicker: KickerParams,
    /// Dribbler.
    pub dribbler: DribblerParams,
}

impl RobotSpecs {
    /// Default specs with a named `RobotLimits` preset applied.
    pub fn with_limits_preset(name: &str) -> Option<Self> {
        RobotLimits::preset(name).map(|limits| Self {
            limits,
            ..Self::default()
        })
    }

    /// Effective yaw inertia.
    pub fn inertia(&self) -> f64 {
        self.inertia_z
            .unwrap_or(0.5 * self.mass * self.radius * self.radius)
    }

    /// Half of the mouth opening angle [rad]: `acos(center_to_dribbler / radius)`.
    pub fn mouth_half_angle(&self) -> f64 {
        (self.center_to_dribbler / self.radius)
            .clamp(-1.0, 1.0)
            .acos()
    }

    /// Half width of the flat front chord [m].
    pub fn front_half_width(&self) -> f64 {
        (self.radius * self.radius - self.center_to_dribbler * self.center_to_dribbler)
            .max(0.0)
            .sqrt()
    }
}

impl Default for RobotSpecs {
    fn default() -> Self {
        Self {
            radius: 0.09,
            height: 0.15,
            mass: 2.5,
            inertia_z: None,
            center_to_dribbler: 0.075,
            dribbler_width: 0.07,
            shoot_radius: 0.0965,
            limits: RobotLimits::default(),
            wheel_angles: WheelAngles::default(),
            drive: DriveParams::default(),
            kicker: KickerParams::default(),
            dribbler: DribblerParams::default(),
        }
    }
}

/// Relationship between the capture instants of the cameras.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraPhase {
    /// Hardware-triggered: all cameras capture at fixed offsets from a common clock.
    Locked,
    /// Free-running: each camera's phase is a uniformly random constant drawn from the seed.
    FreeRunning,
}

/// A scripted vision outage.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VisionOutage {
    /// Start [s] of sim time.
    pub start: f64,
    /// Duration [s].
    pub duration: f64,
}

/// One simulated camera.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraConfig {
    /// Camera id as reported in detection frames.
    pub id: u32,
    /// True position of the optical centre [m].
    pub position: Vec3,
}

/// Vision generation configuration (geometry of the camera system; noise lives in [`Realism`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    /// Cameras. Empty = derive `default_camera_count` cameras from the field size.
    pub cameras: Vec<CameraConfig>,
    /// Cameras to auto-place when `cameras` is empty (4 or 2, or 1).
    pub default_camera_count: u32,
    /// Camera height used for auto-placement [m].
    pub default_camera_height: f64,
    /// Auto-placed camera x as a fraction of the field length (2-camera rigs
    /// measured at +-0.20 L; the classic assumption is 0.25).
    pub default_camera_x_fraction: f64,
    /// Radius [m] around each camera's nadir beyond which it detects nothing
    /// (hard field-of-view edge); 0 = unlimited.
    pub fov_radius: f64,
    /// How the cameras' capture instants relate.
    pub camera_phase: CameraPhase,
    /// Per-camera capture offsets [s] within the frame period (used by `Locked`); missing = 0.
    pub phase_offsets: Vec<f64>,
    /// Emit `SSL_DetectionBall.area` (a whole tournament ran without it).
    pub report_area: bool,
    /// Blob area [px] of a ball on the floor near the nadir; the area model scales this.
    pub area_at_nadir_px: f64,
    /// Area [px] reported for spurious dribbler-LED balls.
    pub spurious_ball_area_px: f64,
    /// Spurious dribbler balls sit on the robot centreline this far ahead of the centre [m].
    pub spurious_ball_forward: f64,
    /// Lateral spread of spurious dribbler balls [m].
    pub spurious_ball_lateral_stddev: f64,
    /// Mean reported robot confidence.
    pub robot_confidence_mean: f64,
    /// Mean reported ball confidence.
    pub ball_confidence_mean: f64,
    /// Spread of reported confidences (clamped to 0..1).
    pub confidence_stddev: f64,
    /// Probability per robot detection of an extra duplicate entry with the same id.
    pub duplicate_robot_rate: f64,
    /// Scripted vision outages (no packets at all) in sim time.
    pub outages: Vec<VisionOutage>,
    /// Detection frame rate [Hz], all cameras.
    pub frame_rate: f64,
    /// Attach the geometry packet every N frames (1 = every frame).
    pub geometry_every_n_frames: u32,
    /// Report the true ball height in `SSL_DetectionBall.z` (off = omitted, like real vision).
    pub report_ball_z: bool,
    /// Offset added to reported timestamps [s] so they look like Unix time.
    pub timestamp_epoch_offset: f64,
    /// Pinhole focal length [px] used for the ball `area` model.
    pub focal_length_px: f64,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            cameras: Vec::new(),
            default_camera_count: 2,
            default_camera_height: 6.4,
            default_camera_x_fraction: 0.20,
            fov_radius: 6.6,
            camera_phase: CameraPhase::Locked,
            phase_offsets: Vec::new(),
            report_area: true,
            area_at_nadir_px: 63.0,
            spurious_ball_area_px: 32.0,
            spurious_ball_forward: 0.13,
            spurious_ball_lateral_stddev: 0.07,
            robot_confidence_mean: 0.9,
            ball_confidence_mean: 0.9,
            confidence_stddev: 0.05,
            duplicate_robot_rate: 3.0e-5,
            outages: Vec::new(),
            frame_rate: 73.3,
            geometry_every_n_frames: 73,
            report_ball_z: false,
            timestamp_epoch_offset: 0.0,
            focal_length_px: 1420.0,
        }
    }
}

/// Realism knobs. Field-for-field compatible with ER-Force's
/// `RealismConfigErForce` (units converted to SI), plus simulator-specific extras.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Realism {
    /// Gaussian noise on reported ball position [m].
    pub stddev_ball_p: f64,
    /// Gaussian noise on reported robot position [m].
    pub stddev_robot_p: f64,
    /// Gaussian noise on reported robot orientation [rad].
    pub stddev_robot_phi: f64,
    /// Gaussian noise on reported ball area [px].
    pub stddev_ball_area: f64,
    /// Simulate the ball being hidden behind robots.
    pub enable_invisible_ball: bool,
    /// Visible fraction below which the ball is not reported (0..1).
    pub ball_visibility_threshold: f64,
    /// Half of the camera overlap band [m] (regions overlap by `2 * camera_overlap`).
    pub camera_overlap: f64,
    /// Rate of spurious ball detections at robot dribblers [1/s/robot].
    pub dribbler_ball_detections: f64,
    /// Error injected into the reported camera calibration positions [m].
    pub camera_position_error: f64,
    /// Probability a robot command datagram is dropped (0..1).
    pub robot_command_loss: f64,
    /// Probability a robot response datagram is dropped (0..1).
    pub robot_response_loss: f64,
    /// Probability the ball is missing from a camera frame (0..1).
    pub missing_ball_detections: f64,
    /// Probability a robot is missing from a camera frame (0..1).
    pub missing_robot_detections: f64,
    /// Delay between capture and packet send [s].
    pub vision_delay: f64,
    /// Difference between `t_sent` and `t_capture` [s].
    pub vision_processing_time: f64,
    /// Physical dribbler (true) or glue (false).
    pub simulate_dribbling: bool,
    /// Fixed per-camera radial offset applied to all reported objects [m].
    pub object_position_offset: f64,
    /// Fixed delay applied to incoming robot commands [s].
    pub command_delay: f64,
    /// Std of the smooth per-camera position warp [m] (spatially varying calibration error).
    pub calibration_warp_stddev: f64,
    /// Spatial scale of the warp [m] (larger = smoother).
    pub calibration_warp_length: f64,
    /// Constant per-camera orientation offset [rad] (random sign per camera).
    pub calibration_orientation_offset: f64,
    /// Std of the spatially varying orientation warp [rad].
    pub calibration_orientation_warp: f64,
    /// Kick direction error [rad], one draw per kick.
    pub kick_direction_stddev: f64,
    /// Chip launch elevation error [rad], one draw per chip.
    pub chip_angle_stddev: f64,
    /// Multiplicative kick speed error (std of the factor), one draw per kick.
    pub kick_speed_factor_stddev: f64,
}

impl Default for Realism {
    fn default() -> Self {
        Realism::realistic()
    }
}

impl Realism {
    /// No noise, no losses, perfect vision except the standard 35 ms delay.
    pub fn none() -> Self {
        Self {
            stddev_ball_p: 0.0,
            stddev_robot_p: 0.0,
            stddev_robot_phi: 0.0,
            stddev_ball_area: 0.0,
            enable_invisible_ball: true,
            ball_visibility_threshold: 0.4,
            camera_overlap: 0.3,
            dribbler_ball_detections: 0.0,
            camera_position_error: 0.0,
            robot_command_loss: 0.0,
            robot_response_loss: 0.0,
            missing_ball_detections: 0.0,
            missing_robot_detections: 0.0,
            vision_delay: 0.035,
            vision_processing_time: 0.005,
            simulate_dribbling: true,
            object_position_offset: 0.0,
            command_delay: 0.0,
            calibration_warp_stddev: 0.0,
            calibration_warp_length: 3.0,
            calibration_orientation_offset: 0.0,
            calibration_orientation_warp: 0.0,
            kick_direction_stddev: 0.0,
            chip_angle_stddev: 0.0,
            kick_speed_factor_stddev: 0.0,
        }
    }

    /// ER-Force "Friendly" preset.
    pub fn erforce_friendly() -> Self {
        Self {
            stddev_ball_p: 0.0004,
            stddev_robot_p: 0.0003,
            stddev_robot_phi: 0.003,
            stddev_ball_area: 1.0,
            camera_overlap: 1.0,
            dribbler_ball_detections: 0.001,
            camera_position_error: 0.05,
            robot_command_loss: 0.01,
            robot_response_loss: 0.01,
            missing_ball_detections: 0.05,
            missing_robot_detections: 0.02,
            vision_processing_time: 0.010,
            object_position_offset: 0.02,
            ..Self::none()
        }
    }

    /// ER-Force "Realistic" preset.
    pub fn erforce_realistic() -> Self {
        Self {
            stddev_ball_p: 0.0014,
            stddev_robot_p: 0.0013,
            stddev_robot_phi: 0.01,
            stddev_ball_area: 6.5,
            camera_overlap: 1.0,
            dribbler_ball_detections: 0.05,
            camera_position_error: 0.1,
            robot_command_loss: 0.03,
            robot_response_loss: 0.1,
            missing_ball_detections: 0.05,
            missing_robot_detections: 0.02,
            vision_processing_time: 0.010,
            object_position_offset: 0.02,
            ..Self::none()
        }
    }

    /// ER-Force "RC2021" preset (tournament conditions, glued dribbler).
    pub fn erforce_rc2021() -> Self {
        Self {
            stddev_ball_p: 0.0010,
            stddev_robot_p: 0.0013,
            stddev_robot_phi: 0.01,
            stddev_ball_area: 6.5,
            camera_overlap: 1.0,
            dribbler_ball_detections: 0.02,
            camera_position_error: 0.0,
            robot_command_loss: 0.03,
            robot_response_loss: 0.1,
            missing_ball_detections: 0.05,
            missing_robot_detections: 0.0,
            vision_processing_time: 0.010,
            simulate_dribbling: false,
            object_position_offset: 0.0,
            ..Self::none()
        }
    }

    /// Measured from ten 2026 game logs (German Open + RoboCup, pooled). The
    /// default. See `docs/calibration/vision.md` and `dynamics.md`.
    pub fn realistic() -> Self {
        Self {
            stddev_ball_p: 0.0007,
            stddev_robot_p: 0.0005,
            stddev_robot_phi: 0.005,
            stddev_ball_area: 3.3,
            enable_invisible_ball: true,
            ball_visibility_threshold: 0.4,
            camera_overlap: 0.8,
            dribbler_ball_detections: 0.02,
            camera_position_error: 0.02,
            robot_command_loss: 0.01,
            robot_response_loss: 0.01,
            missing_ball_detections: 0.007,
            missing_robot_detections: 0.002,
            vision_delay: 0.022,
            vision_processing_time: 0.0073,
            simulate_dribbling: true,
            object_position_offset: 0.02,
            command_delay: 0.0,
            calibration_warp_stddev: 0.012,
            calibration_warp_length: 3.0,
            calibration_orientation_offset: 0.02,
            calibration_orientation_warp: 0.02,
            kick_direction_stddev: 2.5f64.to_radians(),
            chip_angle_stddev: 6.0f64.to_radians(),
            kick_speed_factor_stddev: 0.10,
        }
    }

    /// German Open 2026 venue: no `area`, many dribbler-LED false balls, no constant camera offset.
    pub fn go26() -> Self {
        Self {
            dribbler_ball_detections: 0.06,
            object_position_offset: 0.005,
            calibration_warp_stddev: 0.014,
            ..Self::realistic()
        }
    }

    /// RoboCup 2026 venue: `area` reported, few false balls, 2-2.6 cm constant x offset between cameras.
    pub fn rc26() -> Self {
        Self {
            dribbler_ball_detections: 0.0015,
            object_position_offset: 0.023,
            calibration_warp_stddev: 0.009,
            ..Self::realistic()
        }
    }

    /// Look up a preset by name (case-insensitive): `none`, `realistic`
    /// (measured, default), `go26`, `rc26`, `erforce_friendly`,
    /// `erforce_realistic`, `erforce_rc2021` (aliases `friendly`, `rc2021`).
    pub fn preset(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "none" => Some(Self::none()),
            "realistic" | "measured" | "default" => Some(Self::realistic()),
            "go26" => Some(Self::go26()),
            "rc26" => Some(Self::rc26()),
            "erforce_friendly" | "friendly" => Some(Self::erforce_friendly()),
            "erforce_realistic" => Some(Self::erforce_realistic()),
            "erforce_rc2021" | "rc2021" => Some(Self::erforce_rc2021()),
            _ => None,
        }
    }
}

/// Top-level simulation configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SimConfig {
    /// Physics substep [s]. Default 1 ms.
    pub substep: f64,
    /// Seed for every random stream.
    pub seed: u64,
    /// Seconds without a command before a robot coasts to a stop.
    pub command_timeout: f64,
    /// Ball parameters.
    pub ball: BallParams,
    /// Contact parameters.
    pub contact: ContactParams,
    /// Default specs for blue robots.
    pub blue_specs: RobotSpecs,
    /// Default specs for yellow robots.
    pub yellow_specs: RobotSpecs,
    /// Vision geometry.
    pub vision: VisionConfig,
    /// Realism.
    pub realism: Realism,
    /// Height of the boundary boards [m]; the ball clears them above this.
    pub wall_height: f64,
    /// Robots placed per team at start (0..=16).
    pub initial_robots_per_team: u8,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            substep: 0.001,
            seed: 0,
            command_timeout: 0.1,
            ball: BallParams::default(),
            contact: ContactParams::default(),
            blue_specs: RobotSpecs::default(),
            yellow_specs: RobotSpecs::default(),
            vision: VisionConfig::default(),
            realism: Realism::default(),
            wall_height: 0.10,
            initial_robots_per_team: 11,
        }
    }
}

impl SimConfig {
    /// Substep as [`crate::SimTime`].
    pub fn substep_time(&self) -> crate::SimTime {
        crate::SimTime::from_secs_f64(self.substep)
    }
}

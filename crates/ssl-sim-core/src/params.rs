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
    /// Rolling deceleration [m/s^2] (negative).
    pub acc_roll: f64,
    /// Inertia distribution p = I / (m r^2); 0.4 solid sphere, 0.66 hollow.
    /// The published `k_switch` is `1 / (1 + p)`.
    pub inertia_distribution: f64,
    /// Horizontal velocity factor kept on the first bounce of a chip.
    pub chip_damping_xy_first_hop: f64,
    /// Horizontal velocity factor kept on later bounces.
    pub chip_damping_xy_other_hops: f64,
    /// Vertical velocity factor kept on every bounce.
    pub chip_damping_z: f64,
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
}

impl Default for BallParams {
    fn default() -> Self {
        Self {
            radius: 0.0215,
            mass: 0.046,
            acc_slide: -3.0,
            acc_roll: -0.30,
            inertia_distribution: 0.5,
            chip_damping_xy_first_hop: 0.75,
            chip_damping_xy_other_hops: 0.95,
            chip_damping_z: 0.50,
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
}

impl Default for ContactParams {
    fn default() -> Self {
        Self {
            ball_robot_normal: 0.5,
            ball_robot_tangent: 0.0,
            ball_kicker_normal: 0.6,
            ball_kicker_tangent: 0.3,
            ball_wall_normal: 0.5,
            ball_wall_tangent: 0.0,
            spin_retention: 0.6,
            robot_robot_restitution: 0.2,
            robot_robot_friction: 0.1,
            robot_wall_restitution: 0.1,
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
            acc_speedup_absolute_max: 4.0,
            acc_speedup_angular_max: 50.0,
            acc_brake_absolute_max: 6.0,
            acc_brake_angular_max: 50.0,
            vel_absolute_max: 3.5,
            vel_angular_max: 20.0,
        }
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
    /// Max acceleration [m/s^2] the dribbler can impart to the ball at full speed.
    pub hold_accel: f64,
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
            hold_accel: 4.0,
            roller_radius: 0.007,
            seat_depth: 0.008,
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
            shoot_radius: 0.0885,
            limits: RobotLimits::default(),
            wheel_angles: WheelAngles::default(),
            drive: DriveParams::default(),
            kicker: KickerParams::default(),
            dribbler: DribblerParams::default(),
        }
    }
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
            default_camera_count: 4,
            default_camera_height: 4.0,
            frame_rate: 60.0,
            geometry_every_n_frames: 30,
            report_ball_z: false,
            timestamp_epoch_offset: 0.0,
            focal_length_px: 390.0,
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
        }
    }

    /// ER-Force "Friendly" preset.
    pub fn friendly() -> Self {
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

    /// ER-Force "Realistic" preset (default).
    pub fn realistic() -> Self {
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
    pub fn rc2021() -> Self {
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

    /// Look up a preset by name (case-insensitive).
    pub fn preset(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "none" => Some(Self::none()),
            "friendly" => Some(Self::friendly()),
            "realistic" => Some(Self::realistic()),
            "rc2021" => Some(Self::rc2021()),
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

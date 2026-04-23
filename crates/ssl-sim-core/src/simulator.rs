use std::collections::HashMap;

use rapier3d::prelude::*;

use crate::types::{
    BallState, MoveCommand, RobotCommand, RobotId, RobotState, Snapshot, Team, TeleportBall,
    TeleportRobot,
};

const DEFAULT_ROBOT_FORMATION: [(f32, f32); 11] = [
    (1.50, 1.12),
    (1.50, 0.00),
    (1.50, -1.12),
    (0.55, 0.00),
    (2.50, 0.00),
    (3.60, 0.00),
    (3.20, 0.75),
    (3.20, -0.75),
    (3.20, 1.50),
    (3.20, -1.50),
    (3.20, 2.25),
];

#[derive(Debug, Clone)]
pub struct SimulatorConfig {
    pub fixed_step_seconds: f32,
    pub initial_robots_per_team: u32,
    pub ball_radius: f32,
    pub ball_mass: f32,
    pub robot_radius: f32,
    pub robot_height: f32,
    pub robot_mass: f32,
    pub dribbler_distance: f32,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            fixed_step_seconds: 0.002,
            initial_robots_per_team: 11,
            ball_radius: 0.0215,
            ball_mass: 0.046,
            robot_radius: 0.09,
            robot_height: 0.15,
            robot_mass: 2.7,
            dribbler_distance: 0.075,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RobotHandles {
    body: RigidBodyHandle,
}

pub struct Simulator {
    config: SimulatorConfig,
    time_seconds: f64,
    frame_number: u64,
    gravity: Vector,
    pipeline: PhysicsPipeline,
    integration: IntegrationParameters,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd: CCDSolver,
    ball_body: RigidBodyHandle,
    robots: HashMap<RobotId, RobotHandles>,
}

impl Simulator {
    pub fn new(config: SimulatorConfig) -> Self {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();

        let ground = RigidBodyBuilder::fixed()
            .translation(Vector::new(0.0, 0.0, -0.01))
            .build();
        let ground_handle = bodies.insert(ground);
        let ground_collider = ColliderBuilder::cuboid(20.0, 20.0, 0.01)
            .friction(0.55)
            .restitution(0.55)
            .build();
        colliders.insert_with_parent(ground_collider, ground_handle, &mut bodies);

        let ball_body = RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 0.0, config.ball_radius))
            .linear_damping(0.18)
            .angular_damping(0.12)
            .ccd_enabled(true)
            .build();
        let ball_body = bodies.insert(ball_body);
        let ball_collider = ColliderBuilder::ball(config.ball_radius)
            .density(config.ball_mass / ball_volume(config.ball_radius))
            .friction(0.35)
            .restitution(0.62)
            .build();
        colliders.insert_with_parent(ball_collider, ball_body, &mut bodies);

        let mut integration = IntegrationParameters::default();
        integration.dt = config.fixed_step_seconds;

        let mut simulator = Self {
            config,
            time_seconds: 0.0,
            frame_number: 0,
            gravity: Vector::new(0.0, 0.0, -9.81),
            pipeline: PhysicsPipeline::new(),
            integration,
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies,
            colliders,
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd: CCDSolver::new(),
            ball_body,
            robots: HashMap::new(),
        };
        simulator.spawn_default_robots();
        simulator
    }

    pub fn config(&self) -> &SimulatorConfig {
        &self.config
    }

    pub fn time_seconds(&self) -> f64 {
        self.time_seconds
    }

    pub fn step(&mut self, dt_seconds: f32) {
        if dt_seconds <= 0.0 {
            return;
        }

        self.integration.dt = dt_seconds;
        self.pipeline.step(
            self.gravity,
            &self.integration,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd,
            &(),
            &(),
        );

        self.time_seconds += f64::from(dt_seconds);
        self.frame_number = self.frame_number.wrapping_add(1);
    }

    pub fn fixed_step(&mut self) {
        self.step(self.config.fixed_step_seconds);
    }

    pub fn spawn_default_robots(&mut self) {
        let robots_per_team = self
            .config
            .initial_robots_per_team
            .min(DEFAULT_ROBOT_FORMATION.len() as u32);

        for id in 0..robots_per_team {
            self.spawn_default_robot(Team::Blue, id);
            self.spawn_default_robot(Team::Yellow, id);
        }
    }

    pub fn apply_robot_command(&mut self, command: RobotCommand) {
        self.ensure_robot(command.id);
        let orientation = self
            .robot_body(command.id)
            .map(robot_orientation)
            .unwrap_or_default();

        if let Some(movement) = command.movement {
            self.apply_robot_movement(command.id, movement, orientation);
        }

        if let Some(kick_speed) = command.kick_speed {
            if kick_speed > 0.0 && self.robot_has_ball(command.id) {
                self.kick_ball(orientation, kick_speed, command.kick_angle_deg);
            }
        }
    }

    pub fn teleport_ball(&mut self, teleport: TeleportBall) {
        let body = self
            .bodies
            .get_mut(self.ball_body)
            .expect("ball rigid body should exist");

        if let Some(position) = teleport.position {
            body.set_translation(position, true);
        }

        if let Some(velocity) = teleport.velocity {
            body.set_linvel(velocity, true);
            body.set_angvel(Vector::ZERO, true);
        }
    }

    pub fn teleport_robot(&mut self, teleport: TeleportRobot) {
        if teleport.present == Some(false) {
            self.remove_robot(teleport.id);
            return;
        }

        self.ensure_robot(teleport.id);
        let z = self.config.robot_height * 0.5;
        let body = self
            .robot_body_mut(teleport.id)
            .expect("robot rigid body should exist after ensure_robot");

        let current = body.translation();
        let x = teleport.x.unwrap_or(current.x);
        let y = teleport.y.unwrap_or(current.y);
        body.set_translation(Vector::new(x, y, z), true);

        if let Some(orientation) = teleport.orientation {
            body.set_rotation(Rotation::from_rotation_z(orientation), true);
        }

        let linvel = body.linvel();
        body.set_linvel(
            Vector::new(
                teleport.vx.unwrap_or(linvel.x),
                teleport.vy.unwrap_or(linvel.y),
                0.0,
            ),
            true,
        );

        let angvel = body.angvel();
        body.set_angvel(
            Vector::new(0.0, 0.0, teleport.angular.unwrap_or(angvel.z)),
            true,
        );
    }

    pub fn snapshot(&self) -> Snapshot {
        let ball = self
            .bodies
            .get(self.ball_body)
            .expect("ball rigid body should exist");

        let mut robots = self
            .robots
            .keys()
            .filter_map(|id| self.robot_state(*id))
            .collect::<Vec<_>>();
        robots.sort_by_key(|robot| match robot.id.team {
            Team::Blue => (0_u8, robot.id.id),
            Team::Yellow => (1_u8, robot.id.id),
        });

        Snapshot {
            time_seconds: self.time_seconds,
            frame_number: self.frame_number,
            ball: BallState {
                position: ball.translation(),
                velocity: ball.linvel(),
            },
            robots,
        }
    }

    fn ensure_robot(&mut self, id: RobotId) {
        if self.robots.contains_key(&id) {
            return;
        }

        let body = RigidBodyBuilder::kinematic_velocity_based()
            .translation(Vector::new(0.0, 0.0, self.config.robot_height * 0.5))
            .linear_damping(0.0)
            .angular_damping(0.0)
            .build();
        let body = self.bodies.insert(body);

        let collider = ColliderBuilder::ball(self.config.robot_radius)
            .density(self.config.robot_mass / ball_volume(self.config.robot_radius))
            .friction(0.8)
            .restitution(0.3)
            .build();
        self.colliders
            .insert_with_parent(collider, body, &mut self.bodies);

        self.robots.insert(id, RobotHandles { body });
    }

    fn spawn_default_robot(&mut self, team: Team, id: u32) {
        let Some((x, y)) = DEFAULT_ROBOT_FORMATION.get(id as usize).copied() else {
            return;
        };
        let (x, orientation) = match team {
            Team::Blue => (-x, 0.0),
            Team::Yellow => (x, std::f32::consts::PI),
        };

        self.teleport_robot(TeleportRobot {
            id: RobotId { team, id },
            present: Some(true),
            x: Some(x),
            y: Some(y),
            orientation: Some(orientation),
            vx: Some(0.0),
            vy: Some(0.0),
            angular: Some(0.0),
        });
    }

    fn remove_robot(&mut self, id: RobotId) {
        if let Some(handles) = self.robots.remove(&id) {
            self.bodies.remove(
                handles.body,
                &mut self.islands,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                true,
            );
        }
    }

    fn apply_robot_movement(&mut self, id: RobotId, movement: MoveCommand, orientation: f32) {
        let (vx, vy, angular) = match movement {
            MoveCommand::GlobalVelocity { x, y, angular } => (x, y, angular),
            MoveCommand::LocalVelocity {
                forward,
                left,
                angular,
            } => {
                let forward_axis = (orientation.cos(), orientation.sin());
                let left_axis = (-orientation.sin(), orientation.cos());
                let velocity_x = forward_axis.0 * forward + left_axis.0 * left;
                let velocity_y = forward_axis.1 * forward + left_axis.1 * left;
                (velocity_x, velocity_y, angular)
            }
            MoveCommand::WheelVelocity {
                front_right,
                back_right,
                back_left,
                front_left,
            } => {
                let forward = (front_right + back_right + back_left + front_left) * 0.25;
                let left = (-front_right + back_right + back_left - front_left) * 0.25;
                let angular = (-front_right - back_right + back_left + front_left)
                    / (4.0 * self.config.robot_radius.max(0.001));
                let forward_axis = (orientation.cos(), orientation.sin());
                let left_axis = (-orientation.sin(), orientation.cos());
                let velocity_x = forward_axis.0 * forward + left_axis.0 * left;
                let velocity_y = forward_axis.1 * forward + left_axis.1 * left;
                (velocity_x, velocity_y, angular)
            }
        };

        if let Some(body) = self.robot_body_mut(id) {
            body.set_linvel(Vector::new(vx, vy, 0.0), true);
            body.set_angvel(Vector::new(0.0, 0.0, angular), true);
        }
    }

    fn kick_ball(&mut self, orientation: f32, speed: f32, angle_deg: f32) {
        let angle = angle_deg.to_radians();
        let horizontal_speed = speed * angle.cos();
        let vertical_speed = speed * angle.sin();
        let velocity = Vector::new(
            horizontal_speed * orientation.cos(),
            horizontal_speed * orientation.sin(),
            vertical_speed,
        );
        self.teleport_ball(TeleportBall {
            position: None,
            velocity: Some(velocity),
        });
    }

    fn robot_has_ball(&self, id: RobotId) -> bool {
        let Some(robot) = self.robot_body(id) else {
            return false;
        };
        let ball = self
            .bodies
            .get(self.ball_body)
            .expect("ball rigid body should exist");
        let robot_pos = robot.translation();
        let ball_pos = ball.translation();
        let orientation = robot_orientation(robot);
        let dribbler = Vector::new(
            robot_pos.x + self.config.dribbler_distance * orientation.cos(),
            robot_pos.y + self.config.dribbler_distance * orientation.sin(),
            self.config.ball_radius,
        );
        let delta = ball_pos - dribbler;
        delta.length() <= self.config.ball_radius + 0.035
    }

    fn robot_state(&self, id: RobotId) -> Option<RobotState> {
        let body = self.robot_body(id)?;
        Some(RobotState {
            id,
            x: body.translation().x,
            y: body.translation().y,
            z: body.translation().z,
            orientation: robot_orientation(body),
            vx: body.linvel().x,
            vy: body.linvel().y,
            angular: body.angvel().z,
            dribbler_ball_contact: self.robot_has_ball(id),
        })
    }

    fn robot_body(&self, id: RobotId) -> Option<&RigidBody> {
        self.robots
            .get(&id)
            .and_then(|handles| self.bodies.get(handles.body))
    }

    fn robot_body_mut(&mut self, id: RobotId) -> Option<&mut RigidBody> {
        self.robots
            .get(&id)
            .and_then(|handles| self.bodies.get_mut(handles.body))
    }
}

impl Default for Simulator {
    fn default() -> Self {
        Self::new(SimulatorConfig::default())
    }
}

fn robot_orientation(body: &RigidBody) -> f32 {
    let forward = body.rotation() * Vector::X;
    forward.y.atan2(forward.x)
}

fn ball_volume(radius: f32) -> f32 {
    4.0 / 3.0 * std::f32::consts::PI * radius.powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chip_kick_gives_ball_vertical_velocity() {
        let mut sim = Simulator::default();
        let id = RobotId {
            team: Team::Blue,
            id: 0,
        };

        sim.teleport_robot(TeleportRobot {
            id,
            present: Some(true),
            x: Some(0.0),
            y: Some(0.0),
            orientation: Some(0.0),
            vx: Some(0.0),
            vy: Some(0.0),
            angular: Some(0.0),
        });
        sim.teleport_ball(TeleportBall {
            position: Some(Vector::new(0.075, 0.0, sim.config.ball_radius)),
            velocity: Some(Vector::ZERO),
        });

        sim.apply_robot_command(RobotCommand {
            id,
            movement: None,
            kick_speed: Some(4.0),
            kick_angle_deg: 35.0,
            dribbler_speed: None,
        });

        assert!(sim.snapshot().ball.velocity.z > 0.0);
    }

    #[test]
    fn default_simulator_starts_with_eleven_robots_per_team() {
        let sim = Simulator::default();
        let snapshot = sim.snapshot();

        assert_eq!(
            snapshot
                .robots
                .iter()
                .filter(|robot| robot.id.team == Team::Blue)
                .count(),
            11
        );
        assert_eq!(
            snapshot
                .robots
                .iter()
                .filter(|robot| robot.id.team == Team::Yellow)
                .count(),
            11
        );
    }

    #[test]
    fn global_velocity_command_moves_robot() {
        let mut sim = Simulator::default();
        let id = RobotId {
            team: Team::Blue,
            id: 0,
        };
        let before = sim
            .snapshot()
            .robots
            .into_iter()
            .find(|robot| robot.id == id)
            .expect("default blue robot 0 should exist");

        sim.apply_robot_command(RobotCommand {
            id,
            movement: Some(MoveCommand::GlobalVelocity {
                x: 1.0,
                y: 0.0,
                angular: 0.0,
            }),
            kick_speed: None,
            kick_angle_deg: 0.0,
            dribbler_speed: None,
        });
        sim.step(0.1);

        let after = sim
            .snapshot()
            .robots
            .into_iter()
            .find(|robot| robot.id == id)
            .expect("default blue robot 0 should exist");

        assert!(after.x > before.x + 0.05);
    }
}

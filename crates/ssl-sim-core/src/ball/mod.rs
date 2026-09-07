//! The ball: state plus the closed-form trajectory model used both to advance
//! the simulation and (exported) for prediction.
//!
//! OWNER: math agent. See `docs/design.md` §5.2.

pub mod trajectory;

use crate::params::BallParams;
use crate::types::{BallState, Team, Vec3};

pub use trajectory::BallTrajectory;

/// Ball with a pending `by_force` mover and the bookkeeping needed for
/// edge-triggered `Goal` / `BallLeftField` events.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Ball {
    /// Kinematic state.
    pub state: BallState,
    /// Active `by_force` target, if any (see `TeleportBall::by_force`).
    pub force_target: Option<Vec3>,
    /// The ball was outside the field lines at the end of the last substep.
    pub was_outside_field: bool,
    /// The goal the ball was inside of at the end of the last substep.
    pub in_goal: Option<Team>,
}

impl Ball {
    /// A ball at rest at `pos` (z = radius).
    pub fn at_rest(pos_xy: crate::types::Vec2, params: &BallParams) -> Self {
        Self {
            state: BallState {
                pos: Vec3::new(pos_xy.x, pos_xy.y, params.radius),
                ..Default::default()
            },
            force_target: None,
            was_outside_field: false,
            in_goal: None,
        }
    }

    /// True if the ball is airborne (above the floor or moving vertically).
    pub fn is_chipped(&self, params: &BallParams) -> bool {
        self.state.pos.z > params.radius + 1e-6 || self.state.vel.z.abs() > 1e-9
    }

    /// Advance the free-flight/rolling model by `dt` (no collisions) on the carpet.
    pub fn advance(&mut self, dt: f64, params: &BallParams) {
        self.advance_on(dt, params, 0.0);
    }

    /// Advance the free-flight/rolling model by `dt` (no collisions) with the
    /// supporting surface at height `floor_z` (a robot top).
    pub fn advance_on(&mut self, dt: f64, params: &BallParams, floor_z: f64) {
        let traj = BallTrajectory::from_state_on(&self.state, params, floor_z);
        self.state = traj.state_at(dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Vec2;

    #[test]
    fn advance_matches_trajectory() {
        let params = BallParams::default();
        let mut ball = Ball::at_rest(Vec2::new(1.0, 0.0), &params);
        assert!(!ball.is_chipped(&params));
        ball.state.vel = Vec3::new(2.0, 0.0, 0.0);
        let expected = BallTrajectory::from_state(&ball.state, &params).state_at(0.001);
        ball.advance(0.001, &params);
        assert_eq!(ball.state, expected);
        ball.state.vel.z = 1.0;
        assert!(ball.is_chipped(&params));
    }
}

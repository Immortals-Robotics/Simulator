//! Rough throughput benchmark: 22 robots, random-ish commands, 1 ms substeps.
//!
//! Run with `cargo run --release -p ssl-sim-core --example bench`.

use std::time::Instant;

use ssl_sim_core::{
    field::Division, MoveCommand, RobotCommand, RobotId, SimConfig, SimTime, Team, World,
};

fn main() {
    let seconds: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20.0);
    let mut world = World::new(SimConfig::default(), Division::A);
    let robots = world.robots().len();
    let substeps = (seconds / world.config().substep).round() as u64;

    let start = Instant::now();
    let mut vision_outputs = 0usize;
    for i in 0..substeps {
        // Re-command every 20 ms with a slowly rotating velocity per robot so
        // the drive model, contacts and dribblers all stay busy.
        if i % 20 == 0 {
            let t = i as f64 * world.config().substep;
            for n in 0..11u8 {
                for team in [Team::Blue, Team::Yellow] {
                    let phase = t * 0.7 + n as f64;
                    let cmd = RobotCommand {
                        movement: Some(MoveCommand::LocalVelocity {
                            forward: 1.5 * phase.cos(),
                            left: 1.0 * phase.sin(),
                            angular: 2.0 * (phase * 0.5).sin(),
                        }),
                        kick_speed: if n == 3 && i % 2000 == 0 {
                            Some(4.0)
                        } else {
                            None
                        },
                        kick_angle_deg: 0.0,
                        dribbler_rpm: Some(5000.0),
                    };
                    let _ = world.set_robot_command(RobotId::new(team, n), cmd);
                }
            }
        }
        world.step();
        vision_outputs += world.drain_vision().len();
    }
    let elapsed = start.elapsed();
    let per_substep_us = elapsed.as_secs_f64() * 1e6 / substeps as f64;
    println!(
        "{robots} robots, {substeps} substeps ({seconds} s sim) in {:.3} s: {per_substep_us:.2} us/substep, {:.1}x realtime, {vision_outputs} vision captures",
        elapsed.as_secs_f64(),
        seconds / elapsed.as_secs_f64()
    );
    let _ = SimTime::ZERO;
}

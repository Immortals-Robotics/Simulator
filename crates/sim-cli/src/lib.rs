use sim_proto::SimulationRequest;
use sim_server::SimulationService;

pub type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

pub fn realtime_entrypoint(steps: u32) -> CliResult<()> {
    let mut service = SimulationService::new();
    for _ in 0..steps {
        let _ = service.handle_step(SimulationRequest {
            delta_time_seconds: 1.0 / 60.0,
        })?;
    }
    Ok(())
}

pub fn fixed_step_entrypoint(steps: u32, dt_seconds: f32) -> CliResult<()> {
    let mut service = SimulationService::new();
    for _ in 0..steps {
        let _ = service.handle_step(SimulationRequest {
            delta_time_seconds: dt_seconds,
        })?;
    }
    Ok(())
}

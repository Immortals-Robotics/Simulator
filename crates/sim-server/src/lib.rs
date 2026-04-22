use sim_core::Simulator;
use sim_proto::{ProtocolError, SimulationRequest, SimulationResponse};

pub struct SimulationService {
    simulator: Simulator,
}

impl Default for SimulationService {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulationService {
    pub fn new() -> Self {
        Self {
            simulator: Simulator::new(),
        }
    }

    pub fn handle_step(
        &mut self,
        request: SimulationRequest,
    ) -> Result<SimulationResponse, ProtocolError> {
        if !request.delta_time_seconds.is_finite() || request.delta_time_seconds < 0.0 {
            return Err(ProtocolError::InvalidDeltaTime);
        }

        let world = self.simulator.step(request.delta_time_seconds);
        Ok(SimulationResponse {
            tick: world.tick,
            simulation_time_seconds: world.simulation_time_seconds,
        })
    }
}

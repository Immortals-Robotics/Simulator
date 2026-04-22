#[derive(Debug, Clone)]
pub struct SimulationRequest {
    pub delta_time_seconds: f32,
}

#[derive(Debug, Clone)]
pub struct SimulationResponse {
    pub tick: u64,
    pub simulation_time_seconds: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolError {
    InvalidDeltaTime,
}

impl core::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProtocolError::InvalidDeltaTime => {
                f.write_str("invalid request: delta_time_seconds must be finite and non-negative")
            }
        }
    }
}

impl std::error::Error for ProtocolError {}

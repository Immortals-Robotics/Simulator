#[derive(Debug, Clone)]
pub struct WorldState {
    pub tick: u64,
    pub simulation_time_seconds: f32,
}

#[derive(Debug, Clone)]
pub struct Simulator {
    world: WorldState,
}

impl Default for Simulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Simulator {
    pub fn new() -> Self {
        Self {
            world: WorldState {
                tick: 0,
                simulation_time_seconds: 0.0,
            },
        }
    }

    pub fn step(&mut self, dt_seconds: f32) -> &WorldState {
        self.world.tick += 1;
        self.world.simulation_time_seconds += dt_seconds.max(0.0);
        &self.world
    }

    pub fn world(&self) -> &WorldState {
        &self.world
    }
}

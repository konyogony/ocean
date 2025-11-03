use rand::Rng;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaveDataUniform {
    wave_vector: [f32; 2],
    amplitude: f32,
    phase_shift: f32,
}

impl Default for WaveDataUniform {
    fn default() -> Self {
        Self::new()
    }
}

impl WaveDataUniform {
    pub fn new() -> Self {
        let mut rng = rand::rng();
        // For now, we generate random values which we can tweak later on...
        WaveDataUniform {
            wave_vector: [rng.random_range(0.0..=0.5), rng.random_range(0.0..=0.5)],
            amplitude: rng.random_range(0.0..=0.4),
            phase_shift: rng.random_range(0.0..=1.0),
        }
    }
}

pub fn gather_wave_data(n_waves: u64) -> Vec<WaveDataUniform> {
    (1..=n_waves).map(|_| WaveDataUniform::default()).collect() // this is SO clean
}

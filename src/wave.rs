use rand::Rng;

use crate::state::WAVE_NUMBER;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaveData {
    wave_vector: [f32; 2],
    amplitude: f32,
    phase_shift: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaveDataUniform {
    pub waves: [WaveData; WAVE_NUMBER],
}

impl Default for WaveData {
    fn default() -> Self {
        Self::new()
    }
}

impl WaveData {
    pub fn new() -> Self {
        let mut rng = rand::rng();
        // For now, we generate random values which we can tweak later on...
        WaveData {
            wave_vector: [rng.random_range(0.0..=0.5), rng.random_range(0.0..=0.5)],
            amplitude: rng.random_range(0.0..=0.15),
            phase_shift: rng.random_range(0.0..=1.0),
        }
    }
}

// Learnt something new about rust generics...
pub fn gather_wave_data(n_waves: usize) -> Vec<WaveData> {
    (0..n_waves).map(|_| WaveData::default()).collect()
}

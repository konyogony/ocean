use rand::Rng;
use std::f32;

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
        // TODO: Find out whats going on here and why we use these values
        let mut rng = rand::rng();

        let theta = rng.random_range(0.0..std::f32::consts::TAU);
        let wavelength = rng.random_range(2.0..60.0);
        let k = std::f32::consts::TAU / wavelength;
        let k1 = theta.cos() * k;
        let k2 = theta.sin() * k;
        let wave_vector = [k1, k2];
        let amplitude = wavelength * rng.random_range(0.004..0.012);

        WaveData {
            wave_vector,
            amplitude,
            phase_shift: rng.random_range(0.0..std::f32::consts::TAU),
        }
    }
}

pub fn gather_wave_data(n_waves: usize) -> Vec<WaveData> {
    (0..n_waves).map(|_| WaveData::default()).collect()
}

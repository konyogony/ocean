use std::f32;

use cgmath::{InnerSpace, Vector2};
use rand::Rng;

const WIND_VECTOR: Vector2<f32> = Vector2::new(-2.1, 2.8); // Magnitude of 3.5
const AMPLITUDE: f32 = 0.0000001;

use crate::state::WAVE_NUMBER;
use rand_distr::{Distribution, Normal};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaveData {
    wave_vector: [f32; 2],
    amplitude: f32,
    phase_shift: f32,
    xi_r: f32,
    xi_i: f32,
    phk: f32,
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

        let normal = Normal::new(0.0, 1.0).unwrap();
        let xi_r = normal.sample(&mut rand::rng()); // Random values from the gaussian distribution
        let xi_i = normal.sample(&mut rand::rng());
        let phk = WaveData::get_phillips_spectrum_value(wave_vector);

        WaveData {
            wave_vector,
            amplitude,
            phase_shift: rng.random_range(0.0..std::f32::consts::TAU),
            xi_r,
            xi_i,
            phk,
        }
    }

    // Get the phillips spectrum value by a complicated equation
}

// Learnt something new about rust generics...
pub fn gather_wave_data(n_waves: usize) -> Vec<WaveData> {
    (0..n_waves).map(|_| WaveData::default()).collect()
}

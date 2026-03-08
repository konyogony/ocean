use cgmath::{InnerSpace, Vector2};
use rand::Rng;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::f32;
use std::f32::consts::PI;
use std::mem;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InitialData {
    pub k_vec: [f32; 2],
    // Supposed to be a complex number
    pub initial_freq_domain: [f32; 2],
    pub initial_freq_domain_conjugate: [f32; 2],
    pub angular_frequency: f32,
    // To get to 32 bytes
    pub _padding: f32,
}

impl InitialData {
    pub fn new(
        n: u32,
        m: u32,
        fft_size: f32,
        subdivisions: u32,
        wind_vector: [f32; 2],
        l_small: f32,
        amplitude: f32,
        max_w: f32,
        rng: &mut StdRng,
    ) -> Self {
        // Now we have to do this, so that we can center everything around the center.
        let n_f = n as f32;
        let m_f = m as f32;
        let size_f = subdivisions as f32;

        let k_x = (2.0 * PI * (n_f - size_f / 2.0)) / fft_size;
        let k_y = (2.0 * PI * (m_f - size_f / 2.0)) / fft_size;
        let k_vec = [k_x, k_y];

        if k_vec == [0.0, 0.0] {
            return Self {
                k_vec,
                initial_freq_domain: [0.0, 0.0],
                initial_freq_domain_conjugate: [0.0, 0.0],
                angular_frequency: 0.0,
                _padding: 0.0,
            };
        }

        let phk = Self::get_phillips_spectrum_value(k_vec, wind_vector, l_small, amplitude, max_w);

        // Random values from the gaussian distribution
        let u1: f32 = rng.random_range(0.0001..1.0);
        let u2: f32 = rng.random_range(0.0001..1.0);

        let mag = (phk / 2.0).sqrt();
        let phase1 = (-2.0 * u1.ln()).sqrt() * mag;
        let angle = 2.0 * PI * u2;

        let xi_r = phase1 * angle.cos();
        let xi_i = phase1 * angle.sin();

        let freq_domain = [xi_r, xi_i];
        let freq_domain_conjugate = [xi_r, -xi_i];

        let k: Vector2<f32> = k_vec.into();
        let k_len = k.magnitude();
        let w = if k_len > 0.0001 {
            (9.81 * k_len).sqrt()
        } else {
            0.0
        };

        Self {
            k_vec,
            initial_freq_domain: freq_domain,
            initial_freq_domain_conjugate: freq_domain_conjugate,
            angular_frequency: w,
            _padding: 0.0,
        }
    }

    pub fn generate_data(
        fft_size: f32,
        subdivisions: u32,
        wind_vector: [f32; 2],
        l_small: f32,
        amplitude: f32,
        max_w: f32,
        seed: u32,
    ) -> Vec<Self> {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut array: Vec<Self> = Vec::new();

        for n in 0..subdivisions {
            for m in 0..subdivisions {
                array.push(Self::new(
                    n,
                    m,
                    fft_size,
                    subdivisions,
                    wind_vector,
                    l_small,
                    amplitude,
                    max_w,
                    &mut rng,
                ));
            }
        }

        array
    }

    pub fn generate_twiddle_factors(fft_subdivisions: u32) -> Vec<[f32; 2]> {
        let max_stages = fft_subdivisions.ilog2();
        let mut twiddles = Vec::<[f32; 2]>::with_capacity((fft_subdivisions - 1) as usize);

        for stage in 0..max_stages {
            let s = 1u32 << stage;
            for offset in 0..s {
                let angle = 2.0 * std::f32::consts::PI * (offset as f32) / (2.0 * s as f32);
                twiddles.push([angle.cos(), angle.sin()])
            }
        }

        twiddles
    }

    pub fn get_phillips_spectrum_value(
        k_vec: [f32; 2],
        wind_vector: [f32; 2],
        l_small: f32,
        amplitude: f32,
        max_w: f32,
    ) -> f32 {
        let k: Vector2<f32> = k_vec.into();
        let k_len = k.magnitude();
        if k_len > max_w || k_len < 0.001 {
            return 0.0;
        }

        let k2 = k_len * k_len;
        let k4 = k2 * k2;

        let k_hat = k.normalize();
        let w: Vector2<f32> = wind_vector.into();
        let w_len = w.magnitude();
        let w_hat = w.normalize();

        let align = cgmath::dot(k_hat, w_hat);
        let align_power = align.abs().powi(3);
        if align < -0.1 {
            return align_power * 0.1;
        }

        let l = w_len * w_len / 9.81;
        let l2 = l * l;

        let exp_term = f32::exp(-1.0 / (k2 * l2));
        let damp = f32::exp(-k2 * l_small * l_small);
        let small_wave_suppression = 1.0 / (1.0 + k_len * 0.5);

        (align_power * amplitude * exp_term * damp * small_wave_suppression) / k4
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    index: u32,
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: (mem::size_of::<[f32; 3]>() + mem::size_of::<[f32; 2]>())
                        as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }

    // UPD: I understood what is happening
    pub fn generate_plane(size: &f32, subdivisions: u32) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let step = size / subdivisions as f32;
        let half_size = size / 2.0;

        for row in 0..subdivisions {
            for col in 0..subdivisions {
                let x = col as f32 * step - half_size;
                let z = row as f32 * step - half_size;

                vertices.push(Vertex {
                    position: [x, 0.0, z],
                    tex_coords: [
                        col as f32 / subdivisions as f32,
                        row as f32 / subdivisions as f32,
                    ],
                    index: (row * subdivisions + col),
                });
            }
        }

        // Adjust index generation too
        for row in 0..(subdivisions - 1) {
            for col in 0..(subdivisions - 1) {
                let top_left = row * subdivisions + col;
                let top_right = top_left + 1;
                let bottom_left = top_left + subdivisions;
                let bottom_right = bottom_left + 1;

                indices.extend_from_slice(&[
                    top_left,
                    bottom_left,
                    top_right,
                    top_right,
                    bottom_left,
                    bottom_right,
                ]);
            }
        }

        (vertices, indices)
    }
}

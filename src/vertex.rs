use cgmath::{InnerSpace, Vector2};
use rand_distr::{Distribution, Normal};
use std::f32;
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
    ) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Now we have to do this, so that we can center everything around the center.
        let n_f = if n < subdivisions / 2 {
            n as f32
        } else {
            (n as f32) - (subdivisions as f32)
        };

        let m_f = if m < subdivisions / 2 {
            m as f32
        } else {
            (m as f32) - (subdivisions as f32)
        };

        let k_vec = [
            (2.0 * f32::consts::PI * n_f) / fft_size,
            (2.0 * f32::consts::PI * m_f) / fft_size,
        ];

        if k_vec == [0.0, 0.0] {
            return Self {
                k_vec,
                initial_freq_domain: [0.0, 0.0],
                initial_freq_domain_conjugate: [0.0, 0.0],
                angular_frequency: 0.0,
                _padding: 0.0,
            };
        }

        let phk = Self::get_phillips_spectrum_value(k_vec, wind_vector, l_small, amplitude);

        // Random values from the gaussian distribution
        let xi_r = normal.sample(&mut rand::rng());
        let xi_i = normal.sample(&mut rand::rng());

        let factor = phk.sqrt() / f32::consts::SQRT_2;

        let k: Vector2<f32> = k_vec.into();
        let gk = 9.81 * k.magnitude();
        let w = gk.sqrt();

        let freq_domain = [xi_r * factor, xi_i * factor];
        let freq_domain_conjugate = [freq_domain[0], -freq_domain[1]];
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
    ) -> (Vec<Self>, f32, f32) {
        let mut array: Vec<Self> = Vec::new();
        let mut max_magnitude = 0.0f32;
        let mut sum_magnitude = 0.0f32;

        for n in 0..subdivisions {
            for m in 0..subdivisions {
                let data = Self::new(
                    n,
                    m,
                    fft_size,
                    subdivisions,
                    wind_vector,
                    l_small,
                    amplitude,
                );
                let mag = (data.initial_freq_domain[0].powi(2)
                    + data.initial_freq_domain[1].powi(2))
                .sqrt();
                max_magnitude = max_magnitude.max(mag);
                sum_magnitude += mag;
                array.push(data);
            }
        }

        let avg_amplitude = sum_magnitude / (subdivisions * subdivisions) as f32;

        (array, max_magnitude, avg_amplitude)
    }

    // This will be used later, dont worry
    pub fn get_phillips_spectrum_value(
        k_vec: [f32; 2],
        wind_vector: [f32; 2],
        l_small: f32,
        amplitude: f32,
    ) -> f32 {
        let k: Vector2<f32> = k_vec.into();
        let k2 = k.magnitude2();
        if k2 <= 0.0000001 {
            return 0.0;
        }

        let k_hat = k.normalize();

        let w: Vector2<f32> = wind_vector.into();
        let w_hat = w.normalize();

        let align = cgmath::dot(k_hat, w_hat);
        let align2 = align.abs().powi(2);

        let l = w.magnitude2() / 9.81;
        let l2 = l * l;
        let k4 = k2 * k2;
        let exp = f32::exp(-1.0 / (k2 * l2));

        // New thing: l_small, dampening for high f
        let damp = f32::exp(-k2 * l_small.powi(2));

        (align2 * amplitude * exp * damp) / k4
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

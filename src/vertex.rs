use cgmath::{InnerSpace, Vector2};
use rand_distr::{Distribution, Normal};
use std::f32;
use std::mem;

const WIND_VECTOR: Vector2<f32> = Vector2::new(-2.1, 2.8); // Magnitude of 3.5
const AMPLITUDE: f32 = 0.0000001;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InitialData {
    pub k_vec: [f32; 2],
    // Supposed to be a complex number
    // The conjugate of a+ib would be a-ib
    pub initial_freq_domain: [f32; 2],
    pub angular_frequency: f32,
}

impl InitialData {
    pub fn new(n: u32, m: u32, size: f32) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Create a new vector k which depends on the position in the grid
        // k_x = 2 * pi * n / N, k_y = 2 * pi * m / N
        let k_vec = [
            (2.0 * f32::consts::PI * n as f32) / size,
            (2.0 * f32::consts::PI * m as f32) / size,
        ];

        // Random values from the gaussian distribution
        let xi_r = normal.sample(&mut rand::rng());
        let xi_i = normal.sample(&mut rand::rng());
        let phk = Self::get_phillips_spectrum_value(k_vec);
        let k: Vector2<f32> = k_vec.into();
        let gk = 9.81 * k.magnitude();
        Self {
            k_vec,
            initial_freq_domain: Self::get_initial_freq_domain(phk, xi_r, xi_i),
            angular_frequency: gk.sqrt(),
        }
    }

    pub fn generate_data(size: f32) -> Vec<Self> {
        let mut array = Vec::new();
        for n in 0..size as u32 {
            for m in 0..size as u32 {
                array.push(Self::new(n, m, size));
            }
        }
        array
    }

    pub fn get_initial_freq_domain(phk: f32, xi_r: f32, xi_i: f32) -> [f32; 2] {
        let sqrt_phk = phk.sqrt();
        let factor = sqrt_phk / f32::consts::SQRT_2;
        [xi_r * factor, xi_i * factor]
    }

    // This will be used later, dont worry
    pub fn get_phillips_spectrum_value(k_vec: [f32; 2]) -> f32 {
        let k: Vector2<f32> = k_vec.into();
        let k2 = k.magnitude2();
        if k2 == 0.0 {
            return 0.0;
        }

        let k_hat = k.normalize();

        let w = WIND_VECTOR;
        let w_hat = WIND_VECTOR.normalize();

        let align = cgmath::dot(k_hat, w_hat).max(0.0);
        let align2 = align * align;

        let l = w.magnitude2() / 9.81;
        let l2 = l * l;
        let k4 = k2 * k2;
        let exp = f32::exp(-1.0 / (k2 * l2));

        (align2 * AMPLITUDE * exp) / k4
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
        // Create empty vectors
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Get the step and length of half the size
        let step = size / subdivisions as f32;
        let half_size = size / 2.0;

        // Create vertices in each direction
        for row in 0..=subdivisions {
            for col in 0..=subdivisions {
                let x = col as f32 * step - half_size;
                let z = row as f32 * step - half_size;

                vertices.push(Vertex {
                    position: [x, 0.0, z],
                    tex_coords: [
                        col as f32 / subdivisions as f32,
                        row as f32 / subdivisions as f32,
                    ],
                    index: (row * (subdivisions + 1) + col),
                });
            }
        }

        // This is just for the indicies
        for row in 0..subdivisions {
            for col in 0..subdivisions {
                let top_left = row * (subdivisions + 1) + col;
                let top_right = top_left + 1;
                let bottom_left = top_left + (subdivisions + 1);
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

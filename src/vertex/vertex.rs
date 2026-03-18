use std::mem;

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

    pub fn generate_plane(size: &f32, subdivisions: u32) -> (Vec<Vertex>, Vec<u32>) {
        let subdivisions = subdivisions.min(1024);

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

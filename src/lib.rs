use std::mem;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

const INVADER_COLOR: [f32; 3] = [0.4, 1.0, 0.4];
pub const VERTICES: &[Vertex] = &[
    // Central Column (3 quads)
    Vertex {
        position: [-0.1, -0.5, 0.0],
        color: INVADER_COLOR,
    }, // 0
    Vertex {
        position: [0.1, -0.5, 0.0],
        color: INVADER_COLOR,
    }, // 1
    Vertex {
        position: [0.1, -0.3, 0.0],
        color: INVADER_COLOR,
    }, // 2
    Vertex {
        position: [-0.1, -0.3, 0.0],
        color: INVADER_COLOR,
    }, // 3
    Vertex {
        position: [-0.1, -0.3, 0.0],
        color: INVADER_COLOR,
    }, // 4
    Vertex {
        position: [0.1, -0.3, 0.0],
        color: INVADER_COLOR,
    }, // 5
    Vertex {
        position: [0.1, 0.1, 0.0],
        color: INVADER_COLOR,
    }, // 6
    Vertex {
        position: [-0.1, 0.1, 0.0],
        color: INVADER_COLOR,
    }, // 7
    Vertex {
        position: [-0.1, 0.1, 0.0],
        color: INVADER_COLOR,
    }, // 8
    Vertex {
        position: [0.1, 0.1, 0.0],
        color: INVADER_COLOR,
    }, // 9
    Vertex {
        position: [0.1, 0.3, 0.0],
        color: INVADER_COLOR,
    }, // 10
    Vertex {
        position: [-0.1, 0.3, 0.0],
        color: INVADER_COLOR,
    }, // 11
    // Left Arm (2 quads)
    Vertex {
        position: [-0.3, -0.3, 0.0],
        color: INVADER_COLOR,
    }, // 12
    Vertex {
        position: [-0.1, -0.3, 0.0],
        color: INVADER_COLOR,
    }, // 13
    Vertex {
        position: [-0.1, -0.1, 0.0],
        color: INVADER_COLOR,
    }, // 14
    Vertex {
        position: [-0.3, -0.1, 0.0],
        color: INVADER_COLOR,
    }, // 15
    Vertex {
        position: [-0.5, -0.1, 0.0],
        color: INVADER_COLOR,
    }, // 16
    Vertex {
        position: [-0.3, -0.1, 0.0],
        color: INVADER_COLOR,
    }, // 17
    Vertex {
        position: [-0.3, 0.1, 0.0],
        color: INVADER_COLOR,
    }, // 18
    Vertex {
        position: [-0.5, 0.1, 0.0],
        color: INVADER_COLOR,
    }, // 19
    // Right Arm (2 quads)
    Vertex {
        position: [0.1, -0.3, 0.0],
        color: INVADER_COLOR,
    }, // 20
    Vertex {
        position: [0.3, -0.3, 0.0],
        color: INVADER_COLOR,
    }, // 21
    Vertex {
        position: [0.3, -0.1, 0.0],
        color: INVADER_COLOR,
    }, // 22
    Vertex {
        position: [0.1, -0.1, 0.0],
        color: INVADER_COLOR,
    }, // 23
    Vertex {
        position: [0.3, -0.1, 0.0],
        color: INVADER_COLOR,
    }, // 24
    Vertex {
        position: [0.5, -0.1, 0.0],
        color: INVADER_COLOR,
    }, // 25
    Vertex {
        position: [0.5, 0.1, 0.0],
        color: INVADER_COLOR,
    }, // 26
    Vertex {
        position: [0.3, 0.1, 0.0],
        color: INVADER_COLOR,
    }, // 27
    // Eyes (2 quads)
    Vertex {
        position: [-0.3, 0.1, 0.0],
        color: INVADER_COLOR,
    }, // 28
    Vertex {
        position: [-0.1, 0.1, 0.0],
        color: INVADER_COLOR,
    }, // 29
    Vertex {
        position: [-0.1, 0.3, 0.0],
        color: INVADER_COLOR,
    }, // 30
    Vertex {
        position: [-0.3, 0.3, 0.0],
        color: INVADER_COLOR,
    }, // 31
    Vertex {
        position: [0.1, 0.1, 0.0],
        color: INVADER_COLOR,
    }, // 32
    Vertex {
        position: [0.3, 0.1, 0.0],
        color: INVADER_COLOR,
    }, // 33
    Vertex {
        position: [0.3, 0.3, 0.0],
        color: INVADER_COLOR,
    }, // 34
    Vertex {
        position: [0.1, 0.3, 0.0],
        color: INVADER_COLOR,
    }, // 35
    // Antennae (2 quads)
    Vertex {
        position: [-0.3, 0.3, 0.0],
        color: INVADER_COLOR,
    }, // 36
    Vertex {
        position: [-0.1, 0.3, 0.0],
        color: INVADER_COLOR,
    }, // 37
    Vertex {
        position: [-0.1, 0.5, 0.0],
        color: INVADER_COLOR,
    }, // 38
    Vertex {
        position: [-0.3, 0.5, 0.0],
        color: INVADER_COLOR,
    }, // 39
    Vertex {
        position: [0.1, 0.3, 0.0],
        color: INVADER_COLOR,
    }, // 40
    Vertex {
        position: [0.3, 0.3, 0.0],
        color: INVADER_COLOR,
    }, // 41
    Vertex {
        position: [0.3, 0.5, 0.0],
        color: INVADER_COLOR,
    }, // 42
    Vertex {
        position: [0.1, 0.5, 0.0],
        color: INVADER_COLOR,
    }, // 43
];

pub const INDICES: &[u16] = &[
    0, 1, 2, 0, 2, 3, // Central column
    4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, // Left arm
    16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23, // Right arm
    24, 25, 26, 24, 26, 27, 28, 29, 30, 28, 30, 31, // Eyes
    32, 33, 34, 32, 34, 35, 36, 37, 38, 36, 38, 39, // Antennae
    40, 41, 42, 40, 42, 43,
];

// Wizardry
impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

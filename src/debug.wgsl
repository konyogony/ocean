// Debug shader to visualize height values

const g: f32 = 9.81;
const pi: f32 = 3.14159;
const MESH_SIZE: f32 = 1024.0;
const MESH_SUBDIVISIONS: u32 = 2048u;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_sky: mat4x4<f32>,
    camera_pos: vec3<f32>,
};

@group(1) @binding(0)
var<storage, read> height_field: array<vec2<f32>>;

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) height: f32,  // Pass height to fragment shader
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Compute correct index
    let row = model.index / (MESH_SUBDIVISIONS + 1u);
    let col = model.index % (MESH_SUBDIVISIONS + 1u);
    let fft_row = min(row, MESH_SUBDIVISIONS - 1u);
    let fft_col = min(col, MESH_SUBDIVISIONS - 1u);
    let fft_index = fft_row * MESH_SUBDIVISIONS + fft_col;
    
    // Read height
    let h_complex = height_field[fft_index];
    let height = h_complex.x * 50.0;  // Massive scale for debugging
    
    // Apply displacement
    let displaced_pos = vec3<f32>(model.position.x, height, model.position.z);
    
    // Simple normal
    let normal = vec3<f32>(0.0, 1.0, 0.0);
    
    out.normal = normal;
    out.tex_coords = model.tex_coords;
    out.world_pos = displaced_pos;
    out.height = height;
    out.clip_position = camera.view_proj * vec4<f32>(displaced_pos, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Color code by height for debugging
    // Blue = negative, White = zero, Red = positive
    var color: vec3<f32>;
    
    if (in.height > 0.1) {
        // Positive = Red
        color = vec3<f32>(1.0, 0.0, 0.0);
    } else if (in.height < -0.1) {
        // Negative = Blue
        color = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        // Near zero = Green
        color = vec3<f32>(0.0, 1.0, 0.0);
    }
    
    return vec4<f32>(color, 1.0);
}

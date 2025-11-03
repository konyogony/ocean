// just some random constants

const g: f32 = 9.81;

// Wave data

struct WaveData {
    wave_vector: vec2<f32>,
    amplitude: f32,
    phase_shift: f32,
};

struct WaveDataUniform {
    waves: array<WaveData, 256>
}

@group(3) @binding(0)
var<uniform> wave_data: WaveDataUniform;

// Camera

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

// Time

struct TimeUniform {
    time_uniform: f32,
}

@group(2) @binding(0)
var<uniform> time: TimeUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    let data = wave_data.waves[model.index];

    let wave_number = length(data.wave_vector);
    let angular_freq = sqrt(g * wave_number);
    let dot_product = data.wave_vector.x * model.position.x + data.wave_vector.y * model.position.y;
    let arg = dot_product - angular_freq * time.time_uniform + data.phase_shift;
    let height = data.amplitude * sin(arg);

    let position = vec3<f32>(model.position.x, height, model.position.z);

    out.tex_coords = model.tex_coords;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    return base;
}

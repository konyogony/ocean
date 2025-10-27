struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct TimeUniform {
    time_uniform: f32,
}

@group(1) @binding(0)
var<uniform> camera: CameraUniform;
@group(2) @binding(0)
var<uniform> time: TimeUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
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
    let pulse_x = sin(model.position.x + time.time_uniform * 0.2);
    let pulse_z = cos(model.position.z + time.time_uniform * 0.1);
    let position = vec3<f32>(model.position.x, model.position.y + pulse_z * pulse_x * 0.5, model.position.z);
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

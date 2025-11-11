// just some random constants

const g: f32 = 9.81;
const WAVE_COUNT: u32 = 64u;

// Wave data

struct WaveData {
    wave_vector: vec2<f32>,
    amplitude: f32,
    phase_shift: f32,
};

struct WaveDataUniform {
    waves: array<WaveData, 64>,
}

@group(2) @binding(0)
var<uniform> wave_data: WaveDataUniform;

// Camera

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_dir: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Time

struct TimeUniform {
    time_uniform: f32,
}

@group(1) @binding(0)
var<uniform> time: TimeUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    // Quite redundant now, but too lazy to remove
    @location(2) index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) normal: vec3<f32>,
};

// Okay, I have to sum all of the waves together, not have 1 vertex = 1 wave
@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    var height = 0.0;
    var dh_dx = 0.0;
    var dh_dz = 0.0;

    for (var i = 0u; i < WAVE_COUNT; i = i + 1u) {
        // Use the wave index to access the right wave data
        let data = wave_data.waves[i];

        // This is for displacing the vertex
        let wave_number = length(data.wave_vector);
        let angular_freq = sqrt(g * wave_number);
        // Fixed dot product (before it mightve been using 0 y-value)
        let dot_product = dot(data.wave_vector, model.position.xz);

        // Convert to dot() function
        let arg = dot_product - angular_freq * time.time_uniform + data.phase_shift;
        height += data.amplitude * sin(arg);

        // This is for calculating the normal
        // Get partial derivative of height with respect to x and y.
        // Check screenshot "partial_dx_dy.png"
        dh_dx += data.wave_vector.x * data.amplitude * cos(arg);
        dh_dz += data.wave_vector.y * data.amplitude * cos(arg); // Y is the Z component
    }

    let position = vec3<f32>(model.position.x, height, model.position.z);

    // Not normalized vector
    let normal = vec3<f32>(-dh_dx, 1.0, -dh_dz);

    out.normal = normalize(normal);
    out.tex_coords = model.tex_coords;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    return out;
}

// I removed textures to play with lighting
// @group(0) @binding(0)
// var t_diffuse: texture_2d<f32>;
// @group(0) @binding(1)
// var s_diffuse: sampler;

// Time to make Blinn-Phong lighting!!!
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_color = vec3<f32>(0.02, 0.15, 0.22); // deep ocean green blue

    // ambient lighting, basically constant light source, provides basic shape details
    let ambient = vec3<f32>(0.15, 0.18, 0.20);

    // diffuse lighting, dependant on the normal & the light source.
    let light_source_color = vec3<f32>(1.0, 0.95, 0.85);
    let light_source_pos = normalize(vec3<f32>(-0.3, 0.8, 0.6));
    let diffuse_strength = max(dot(light_source_pos, in.normal), 0.0);
    let diffuse = diffuse_strength * light_source_color;

    // specular lighting, needs normal, ligth source, view direction
    let view_source = normalize(camera.view_dir);
    let reflect_source = normalize(reflect(-light_source_pos, in.normal));
    var specular_strength = max(dot(view_source, reflect_source), 0.0);
    specular_strength = pow(specular_strength, 128.0); // Higher value, bigger glossynessa (idk how to spell)
    let specular = specular_strength * light_source_color;

    // Phong = Ambient + Diffuse + Specular
    // For taking pictures, make separate screenshots with ambient, diffuse & specular turned down to 0
    let lighting = ambient * 0.5 + diffuse * 0.5 + specular * 0.5;
    let final_color = base_color * lighting;

    return vec4<f32>(final_color, 1.0);
}

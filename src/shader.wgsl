// just some random constants

const g: f32 = 9.81;
const WAVE_COUNT: u32 = 16u;

// Wave data

struct WaveData {
    wave_vector: vec2<f32>,
    amplitude: f32,
    phase_shift: f32,
};

struct WaveDataUniform {
    waves: array<WaveData, WAVE_COUNT>,
}

@group(2) @binding(0)
var<uniform> wave_data: WaveDataUniform;

// Camera

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_sky: mat4x4<f32>,
    camera_pos: vec3<f32>,
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
    @location(2) world_pos: vec3<f32>,
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
    out.world_pos = position;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    return out;
}

// Time to make Blinn-Phong lighting!!!
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_color = vec3<f32>(0.02, 0.15, 0.22); // deep ocean green blue

    let normal = normalize(in.normal);

    // ambient lighting, basically constant light source, provides basic shape details
    let ambient = vec3<f32>(0.75, 0.85, 1.0) * 0.3;

    // diffuse lighting, dependant on the normal & the light source.
    let light_source_color = vec3<f32>(1.0, 0.95, 0.85);
    // direction TO the light source (however, sometimes may be more benefitial to consider the other way)
    // Bearing 324.6deg, pitch +28.13deg, used chatgpt to get better direction vector
    let light_source_dir = normalize(vec3<f32>(-0.511, 0.472, -0.718));
    let diffuse_strength = clamp(dot(light_source_dir, normal), 0.0, 1.0);
    let diffuse = diffuse_strength * light_source_color;

    // specular lighting, needs normal, ligth source, view direction
    let view_dir = normalize(camera.camera_pos - in.world_pos);
    let reflect_dir = normalize(reflect(-light_source_dir, normal));
    var specular_strength = max(dot(view_dir, reflect_dir), 0.0);
    specular_strength = pow(specular_strength, 128.0); // Higher value, bigger glossynessa (idk how to spell)
    let specular = specular_strength * light_source_color;

    // Phong = Ambient + Diffuse + Specular
    // For taking pictures, make separate screenshots with ambient, diffuse & specular turned down to 0
    let final_color = (base_color * (ambient * 0.3 + diffuse * 0.7)) + (specular * 0.5);
    return vec4<f32>(final_color, 1.0);
}

// just some random constants

const g: f32 = 9.81;
const MESH_SIZE: f32 = 1024.0;
const MESH_SUBDIVISIONS: u32 = 2048u;

// Initial frequency domain

struct InitialData {
    initial_frequency_domain: vec2<f32>,
    k_vec: vec2<f32>,
    angular_frequency: f32,
}

@group(2) @binding(0)
var<storage, read_write> initial_data: array<InitialData>;

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

    let position = vec3<f32>(model.position.x, 1.0, model.position.z);

    // Not normalized vector
    let normal = vec3<f32>(0.0, 1.0, 0.0);

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

@compute @workgroup_size(8,8)
fn cmp_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Get the index... somehow?
    let index = id.y * MESH_SUBDIVISIONS + id.x;

    // First we have to evolve the spectrum in time.
    let h_0 = initial_data[index].initial_frequency_domain;
    // Compute the complex conjugate, which is a-ib
    let h_0_star = vec2<f32>(h_0.x, -h_0.y);
    let w_i = initial_data[index].angular_frequency;
    let wt = w_i * time.time_uniform;
    // We have to use eulers formula for complex numbers,
    // e^{iwt} = cos(wt) + i*sin(wt) = vec2(cos(wt),sin(wt))
    let exponent = vec2<f32>(cos(wt), sin(wt));

    let h_tilda = h_0 * exponent + h_0_star * -exponent;

    let k = initial_data[index].k_vec;
    let D_unrotated = normalize(k) * h_tilda;
    // Multiplying by neg i results in rotation -90 deg
    let D_tilda = vec2<f32>(D_unrotated.y, -D_unrotated.x);

    // Then IFFT
}

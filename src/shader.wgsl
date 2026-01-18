// just some random constants

const g: f32 = 9.81;
const pi: f32 = 3.14159;
const MESH_SIZE: f32 = 1024.0;
const MESH_SUBDIVISIONS: u32 = 2048u;
const FFT_SUBDIVISIONS: u32 = 1024u;
const FFT_SIZE: f32 = 2048.0;
const AMPLITUDE_SCALE: f32 = 1.0;
const CHOP: f32 = 1.0;
const WAVE_SCALE: f32 = MESH_SIZE / FFT_SIZE;

// Camera

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_sky: mat4x4<f32>,
    camera_pos: vec3<f32>,
};

@group(1) @binding(0) var<storage, read> height_field: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> height_field_dz: array<vec4<f32>>;

@group(0) @binding(0)
var<uniform> camera: CameraUniform;


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

    @location(3) height: f32,  // Pass height to fragment shader
};

fn get_fft_index(x: u32, y: u32) -> u32 {
    let ix = x % FFT_SUBDIVISIONS;
    let iy = y % FFT_SUBDIVISIONS;
    return iy * FFT_SUBDIVISIONS + ix;
}

// Billinear sampling... 
fn sample_height_field(world_x: f32, world_z: f32) -> vec4<f32> {
    let fft_u = (world_x / FFT_SIZE) + 0.5;
    let fft_v = (world_z / FFT_SIZE) + 0.5;
    let u = fract(fft_u);
    let v = fract(fft_v);
    
    let texel_x = u * f32(FFT_SUBDIVISIONS);
    let texel_y = v * f32(FFT_SUBDIVISIONS);
    
    // Get the four nearest cells
    let x0 = u32(floor(texel_x)) % FFT_SUBDIVISIONS;
    let y0 = u32(floor(texel_y)) % FFT_SUBDIVISIONS;
    let x1 = (x0 + 1u) % FFT_SUBDIVISIONS;
    let y1 = (y0 + 1u) % FFT_SUBDIVISIONS;
    
    let fx = fract(texel_x);
    let fy = fract(texel_y);
    
    // Sample four corners
    let h00 = height_field[y0 * FFT_SUBDIVISIONS + x0];
    let h10 = height_field[y0 * FFT_SUBDIVISIONS + x1];
    let h01 = height_field[y1 * FFT_SUBDIVISIONS + x0];
    let h11 = height_field[y1 * FFT_SUBDIVISIONS + x1];
    
    let h0 = mix(h00, h10, fx);
    let h1 = mix(h01, h11, fx);
    return mix(h0, h1, fy);
}

fn sample_height_field_dz(world_x: f32, world_z: f32) -> vec4<f32> {
    // Same logic for dz buffer
    let fft_u = (world_x / FFT_SIZE) + 0.5;
    let fft_v = (world_z / FFT_SIZE) + 0.5;
    let u = fract(fft_u);
    let v = fract(fft_v);
    
    let texel_x = u * f32(FFT_SUBDIVISIONS);
    let texel_y = v * f32(FFT_SUBDIVISIONS);
    
    let x0 = u32(floor(texel_x)) % FFT_SUBDIVISIONS;
    let y0 = u32(floor(texel_y)) % FFT_SUBDIVISIONS;
    let x1 = (x0 + 1u) % FFT_SUBDIVISIONS;
    let y1 = (y0 + 1u) % FFT_SUBDIVISIONS;
    
    let fx = fract(texel_x);
    let fy = fract(texel_y);
    
    let h00 = height_field_dz[y0 * FFT_SUBDIVISIONS + x0];
    let h10 = height_field_dz[y0 * FFT_SUBDIVISIONS + x1];
    let h01 = height_field_dz[y1 * FFT_SUBDIVISIONS + x0];
    let h11 = height_field_dz[y1 * FFT_SUBDIVISIONS + x1];
    
    let h0 = mix(h00, h10, fx);
    let h1 = mix(h01, h11, fx);
    return mix(h0, h1, fy);
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out:  VertexOutput;
    
    let world_x = model.position.x;
    let world_z = model.position.z;
    
    let h_sample = sample_height_field(world_x, world_z);
    let dz_sample = sample_height_field_dz(world_x, world_z);
    
    let h = h_sample.x;
    let dx = h_sample.z;
    let dz = dz_sample.x;
    
    var displaced_pos = model.position;
    displaced_pos.x += dx * CHOP;
    displaced_pos.y += h;
    displaced_pos.z += dz * CHOP;
    
    let delta = FFT_SIZE / f32(FFT_SUBDIVISIONS);
    let h_left = sample_height_field(world_x - delta, world_z).x;
    let h_right = sample_height_field(world_x + delta, world_z).x;
    let h_down = sample_height_field(world_x, world_z - delta).x;
    let h_up = sample_height_field(world_x, world_z + delta).x;
    
    let normal_strength = 2.0;
    let dx_grad = (h_right - h_left) * normal_strength;
    let dz_grad = (h_up - h_down) * normal_strength;
    
    out.normal = normalize(vec3<f32>(-dx_grad, 1.0, -dz_grad));
    out.tex_coords = model.tex_coords;
    out.world_pos = displaced_pos;
    out.clip_position = camera.view_proj * vec4<f32>(displaced_pos, 1.0);
    
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

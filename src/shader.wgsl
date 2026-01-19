const g: f32 = 9.81;
const pi: f32 = 3.14159;
const LIGHT_DIFFUSE_INTENSITY: f32 = 1.0;
const ROUGHNESS: f32 = 0.10;
const F_0: f32 = 0.02; 

struct OceanSettings {
    mesh_size: f32,         
    mesh_subdivisions: u32, 
    fft_size: f32,          
    fft_subdivisions: u32,  
    pass_num: u32,         
    time_scale: f32,        
    chop_scale: f32,        
    amplitude_scale: f32,   
    wave_scale: f32,      
    wind_vector: vec2<f32>,
    amplitude: f32,         
    l_small: f32,           
    max_w: f32,             
    fovy: f32,              
    zfar: f32,              
    cam_speed: f32,         
    cam_boost: f32,         
    cam_sensitivity: f32,   
}

@group(0) @binding(0)
var<uniform> ocean_settings: OceanSettings;

// Camera

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_sky: mat4x4<f32>,
    camera_pos: vec3<f32>,
};

@group(2) @binding(0) var<storage, read> height_field: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read> height_field_dz: array<vec4<f32>>;

@group(1) @binding(0)
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
    let ix = x % ocean_settings.fft_subdivisions;
    let iy = y % ocean_settings.fft_subdivisions;
    return iy * ocean_settings.fft_subdivisions + ix;
}

// Billinear sampling... 
fn sample_height_field(world_x: f32, world_z: f32) -> vec4<f32> {
    let fft_size = ocean_settings.fft_size;
    let fft_subdivisions = ocean_settings.fft_subdivisions;

    let fft_u = (world_x / fft_size) + 0.5;
    let fft_v = (world_z / fft_size) + 0.5;
    let u = fract(fft_u);
    let v = fract(fft_v);
    
    let texel_x = u * f32(fft_subdivisions);
    let texel_y = v * f32(fft_subdivisions);
    
    // Get the four nearest cells
    let x0 = u32(floor(texel_x)) % fft_subdivisions;
    let y0 = u32(floor(texel_y)) % fft_subdivisions;
    let x1 = (x0 + 1u) % fft_subdivisions;
    let y1 = (y0 + 1u) % fft_subdivisions;
    
    let fx = fract(texel_x);
    let fy = fract(texel_y);
    
    // Sample four corners
    let h00 = height_field[y0 * fft_subdivisions + x0];
    let h10 = height_field[y0 * fft_subdivisions + x1];
    let h01 = height_field[y1 * fft_subdivisions + x0];
    let h11 = height_field[y1 * fft_subdivisions + x1];
    
    let h0 = mix(h00, h10, fx);
    let h1 = mix(h01, h11, fx);
    return mix(h0, h1, fy);
}

fn sample_height_field_dz(world_x: f32, world_z: f32) -> vec4<f32> {
    let fft_size = ocean_settings.fft_size;
    let fft_subdivisions = ocean_settings.fft_subdivisions;

    // Same logic for dz buffer
    let fft_u = (world_x / fft_size) + 0.5;
    let fft_v = (world_z / fft_size) + 0.5;
    let u = fract(fft_u);
    let v = fract(fft_v);
    
    let texel_x = u * f32(fft_subdivisions);
    let texel_y = v * f32(fft_subdivisions);
    
    let x0 = u32(floor(texel_x)) % fft_subdivisions;
    let y0 = u32(floor(texel_y)) % fft_subdivisions;
    let x1 = (x0 + 1u) % fft_subdivisions;
    let y1 = (y0 + 1u) % fft_subdivisions;
    
    let fx = fract(texel_x);
    let fy = fract(texel_y);
    
    let h00 = height_field_dz[y0 * fft_subdivisions + x0];
    let h10 = height_field_dz[y0 * fft_subdivisions + x1];
    let h01 = height_field_dz[y1 * fft_subdivisions + x0];
    let h11 = height_field_dz[y1 * fft_subdivisions + x1];
    
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
    displaced_pos.x += dx * ocean_settings.chop_scale;
    displaced_pos.y += h;
    displaced_pos.z += dz * ocean_settings.chop_scale;
    
    let delta = 0.5;
    
    // Get full samples including chop
    let sample_left = sample_height_field(world_x - delta, world_z);
    let sample_right = sample_height_field(world_x + delta, world_z);
    let sample_down = sample_height_field(world_x, world_z - delta);
    let sample_up = sample_height_field(world_x, world_z + delta);
    
    let dz_left = sample_height_field_dz(world_x - delta, world_z).x;
    let dz_right = sample_height_field_dz(world_x + delta, world_z).x;
    let dz_down = sample_height_field_dz(world_x, world_z - delta).x;
    let dz_up = sample_height_field_dz(world_x, world_z + delta).x;
    
    // Build displaced positions
    let pos_left = vec3<f32>(
        -delta + sample_left.z * ocean_settings.chop_scale,
        sample_left.x,
        0.0 + dz_left * ocean_settings.chop_scale
    );
    let pos_right = vec3<f32>(
        delta + sample_right.z * ocean_settings.chop_scale,
        sample_right.x,
        0.0 + dz_right * ocean_settings.chop_scale
    );
    let pos_down = vec3<f32>(
        0.0 + sample_down.z * ocean_settings.chop_scale,
        sample_down.x,
        -delta + dz_down * ocean_settings.chop_scale
    );
    let pos_up = vec3<f32>(
        0.0 + sample_up.z * ocean_settings.chop_scale,
        sample_up.x,
        delta + dz_up * ocean_settings.chop_scale
    );
    
    // Compute tangent and bitangent
    let tangent = normalize(pos_right - pos_left);
    let bitangent = normalize(pos_up - pos_down);
    
    out.normal = normalize(cross(tangent, bitangent));
    out.height = h;
    out.tex_coords = model.tex_coords;
    out.world_pos = displaced_pos;
    out.clip_position = camera.view_proj * vec4<f32>(displaced_pos, 1.0);
    
    return out;
}

// BPR work in progress
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.normal);
    let light_source_dir = normalize(vec3<f32>(-0.478, 0.469, -0.743));
    let view_dir = normalize(camera.camera_pos - in.world_pos);
    let reflect_dir = reflect(-view_dir, normal);
    let half_dir = normalize(light_source_dir + view_dir);

    let n_dot_light = max(dot(normal, light_source_dir),0.001);
    let n_dot_view = max(dot(normal, view_dir),0.0001);
    let n_dot_half = max(dot(normal, half_dir),0.0);
    let view_dot_half = max(dot(view_dir, half_dir),0.0);

    let k_s = fresnel_func(view_dot_half, F_0);
    let k_d = 1.0 - k_s;

    let deep_ocean = vec3<f32>(0.01, 0.04, 0.12); 
    let shallow_water = vec3<f32>(0.05, 0.3, 0.45);
    
    let material_color = vec3<f32>(0.01, 0.05, 0.1);
    let light_color = vec3<f32>(1.0, 0.98, 0.9);


    let ambient = vec3<f32>(0.03, 0.04, 0.06) * material_color;
    let diffuse = diffuse_func(material_color, k_d);
    let alpha = ROUGHNESS * ROUGHNESS;
    let specular = cook_torrance(n_dot_light, n_dot_view, n_dot_half, alpha, k_s);

    // Sky
    let sky_blue = vec3<f32>(0.1, 0.35, 0.7);
    let horizon_tone = vec3<f32>(0.4, 0.5, 0.6); 
    let fake_sky = mix(horizon_tone, sky_blue, pow(max(reflect_dir.y, 0.0), 0.6));
    
    let reflection = fake_sky * k_s;

    let pbr = (diffuse + vec3<f32>(specular)) * light_color * n_dot_light;
    var color = pbr + reflection + ambient;

    // Tone mapping
    let exposure = 0.9;
    color = vec3<f32>(1.0) - exp(-color * exposure);
    color = pow(color, vec3<f32>(1.0/2.2));

    return vec4<f32>(color, 1.0);
}

fn diffuse_func(f_lambert: vec3<f32>, k_d: f32) -> vec3<f32> {
    return k_d * f_lambert / pi;
}

fn cook_torrance(n_dot_light: f32, n_dot_view: f32, n_dot_half: f32, alpha: f32, fresnel: f32) -> f32 {
    let normal_function = normal_func(n_dot_half, alpha);
    let geometry_function = geometry_func(n_dot_view, n_dot_light, alpha);
    return (normal_function * geometry_function * fresnel) / max(4 * n_dot_light * n_dot_view, 0.1);
}

fn normal_func(n_dot_half: f32, alpha: f32) -> f32 {
    let alpha2 = alpha * alpha;
    let alpha4 = alpha2 * alpha2;
    let denom = (n_dot_half * n_dot_half * (alpha2 - 1.0) + 1.0);
    return alpha2 / (pi * denom * denom);
}

fn smiths(dot_product: f32, alpha: f32) -> f32 {
    let k = ((alpha + 1.0) * (alpha + 1.0)) / 8.0;
    let denom = dot_product * (k - 1.0) + k;
    return dot_product / denom;
}

fn geometry_func(n_dot_view: f32, n_dot_light: f32, alpha: f32) -> f32 {
    return smiths(n_dot_view, alpha) * smiths(n_dot_light, alpha);
}

fn fresnel_func(view_dot_half: f32, f_0: f32) -> f32 {
    let scale = pow(1.0 - view_dot_half, 5.0);
    return f_0 + (1.0 - f_0) * scale;
}

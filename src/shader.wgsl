const g: f32 = 9.81;
const pi: f32 = 3.14159;
const LIGHT_DIFFUSE_INTENSITY: f32 = 1.0;
const ROUGHNESS: f32 = 0.2;
const F_0: f32 = 0.02; 
const SPECULAR_SCALE: f32 = 0.5;
const REFLECTION_SCALE: f32 = 0.6;
const FOAM_SCALE: f32 = 0.8;
// Sub Surface Scattering 
const DEEP_COLOR: vec3<f32> = vec3<f32>(0.0, 0.01, 0.05); 
const SHALLOW_COLOR: vec3<f32> = vec3<f32>(0.0, 0.06, 0.09);
const SSS_COLOR: vec3<f32> = vec3<f32>(0.0, 0.2, 0.15);
const SUN_COLOR: vec3<f32> = vec3<f32>(1.0, 0.9, 0.8);

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
    @location(4) jacobian: f32,
};


@group(3) @binding(0) var t_skybox: texture_cube<f32>;
@group(3) @binding(1) var s_skybox: sampler;

fn get_fft_index(x: u32, y: u32) -> u32 {
    let ix = x % ocean_settings.fft_subdivisions;
    let iy = y % ocean_settings.fft_subdivisions;
    return iy * ocean_settings.fft_subdivisions + ix;
}

// Billinear sampling... 
fn sample_height_field(world_x: f32, world_z: f32) -> vec4<f32> {
    let fft_size = ocean_settings.fft_size;
    let fft_subdivisions = ocean_settings.fft_subdivisions;

    let fft_u = (world_x / fft_size); 
    let fft_v = (world_z / fft_size);
    let u = fract(fft_u); 
    let v = fract(fft_v);
    
    let texel_x = u * f32(fft_subdivisions);
    let texel_y = v * f32(fft_subdivisions);
    
    // Get the four nearest cells
    let x0 = u32(floor(texel_x)) % fft_subdivisions;
    let y0 = u32(floor(texel_y)) % fft_subdivisions;
    let x1 = (x0 + 1u) % fft_subdivisions;
    let y1 = (y0 + 1u) % fft_subdivisions;

    // For smoother normals
    let fx_linear = fract(texel_x);
    let fy_linear = fract(texel_y);
    
    let fx = fx_linear * fx_linear * (3.0 - 2.0 * fx_linear);
    let fy = fy_linear * fy_linear * (3.0 - 2.0 * fy_linear);
    
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
    let fft_u = (world_x / fft_size);
    let fft_v = (world_z / fft_size);
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

    // Flat ocean testing
    // out.normal = vec3<f32>(0.0,1.0,0.0);
    // out.height = 5.0;
    // out.tex_coords = model.tex_coords;
    // out.world_pos = model.position;
    // out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    // return out;
    
    let world_x = model.position.x;
    let world_z = model.position.z;

    let delta = ocean_settings.fft_size / f32(ocean_settings.fft_subdivisions);
    let amp = ocean_settings.amplitude_scale;
    
    let h_sample = sample_height_field(world_x, world_z);
    let dz_sample = sample_height_field_dz(world_x, world_z);
    
    let h = h_sample.x * amp;
    let dx = h_sample.z * amp;
    let dz = dz_sample.x * amp;
    
    var displaced_pos = model.position;
    displaced_pos.x += dx * ocean_settings.chop_scale;
    displaced_pos.y += h;
    displaced_pos.z += dz * ocean_settings.chop_scale;
    
    // Get full samples including chop
    let sample_left = sample_height_field(world_x - delta, world_z);
    let sample_right = sample_height_field(world_x + delta, world_z);
    let sample_down = sample_height_field(world_x, world_z - delta);
    let sample_up = sample_height_field(world_x, world_z + delta);
    
    let dz_left = sample_height_field_dz(world_x - delta, world_z).x * amp;
    let dz_right = sample_height_field_dz(world_x + delta, world_z).x * amp;
    let dz_down = sample_height_field_dz(world_x, world_z - delta).x * amp;
    let dz_up = sample_height_field_dz(world_x, world_z + delta).x * amp;
    
    let h_left = sample_left.x * amp;
    let h_right = sample_right.x * amp;
    let h_down = sample_down.x * amp;
    let h_up = sample_up.x * amp;

    let dx_left = sample_left.z * amp;
    let dx_right = sample_right.z * amp;
    let dx_down = sample_down.z * amp;
    let dx_up = sample_up.z * amp;

    // Compute tangent and bitangent
    let tangent = vec3<f32>(
        (2.0 * delta) + (dx_right - dx_left) * ocean_settings.chop_scale,
        h_right - h_left,
        (dz_right - dz_left) * ocean_settings.chop_scale
    );
    let bitangent = vec3<f32>(
        (dx_up - dx_down) * ocean_settings.chop_scale,
        h_up - h_down,
        (2.0 * delta) + (dz_up - dz_down) * ocean_settings.chop_scale
    );

    // Calculating the jacobian for ocean foam / SSS stuff
    let dDxdx = (dx_right - dx_left) * ocean_settings.chop_scale / (2.0 * delta);
    let dDzdz = (dz_up - dz_down) * ocean_settings.chop_scale / (2.0 * delta);
    let dDxdz = (dx_up - dx_down) * ocean_settings.chop_scale / (2.0 * delta);
    let dDzdx = (dz_right - dz_left) * ocean_settings.chop_scale / (2.0 * delta);
    let jacobian = (1.0 + dDxdx) * (1.0 + dDzdz) - (dDxdz * dDzdx);
    
    // whoopsies
    out.normal = normalize(cross(bitangent, tangent));
    out.height = h;
    out.tex_coords = model.tex_coords;
    out.world_pos = displaced_pos;
    out.clip_position = camera.view_proj * vec4<f32>(displaced_pos, 1.0);
    out.jacobian = jacobian;
    
    return out;
}

// BPR work in progress
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    let normal = normalize(in.normal);
    let light_source_dir = normalize(vec3<f32>(-0.478, 0.469, -0.743));
    let view_dir = normalize(camera.camera_pos - in.world_pos);
    let reflect_dir = reflect(-view_dir, normal);
    let reflect_dir_clamp = clamp(reflect_dir, vec3(-1.0), vec3(1.0));
    let half_dir = normalize(light_source_dir + view_dir);

    let n_dot_light = clamp(dot(normal, light_source_dir), 0.0, 1.0);
    let n_dot_view = clamp(dot(normal, view_dir), 0.001, 1.0);
    let n_dot_half = clamp(dot(normal, half_dir), 0.0, 1.0);
    let view_dot_half = clamp(dot(view_dir, half_dir), 0.0, 1.0);

    let fresnel_sky = fresnel_func(n_dot_view, F_0);
    let fresnel_spec = fresnel_func(view_dot_half, F_0);

    // Foam using jacobian
    let foam_mask = clamp(1.0 - smoothstep(0.1, 0.45, in.jacobian), 0.0, 1.0);
    let water_base = mix(DEEP_COLOR, SHALLOW_COLOR, foam_mask * 0.2);
    let foam_color = vec3<f32>(0.9);
    let foam = foam_color * pow(foam_mask, 1.5);

    let sky_reflection = textureSample(t_skybox, s_skybox, reflect_dir).rgb * REFLECTION_SCALE;

    let alpha = ROUGHNESS * ROUGHNESS;
    let specular_val = cook_torrance(n_dot_light, n_dot_view, n_dot_half, alpha, fresnel_spec);
    let specular = SUN_COLOR * specular_val * SPECULAR_SCALE;

    let ambient = vec3<f32>(0.02, 0.03, 0.04);

    // TODO: Research more
    let sss_factor = max(0.0, dot(view_dir, -light_source_dir));
    let sss_height_mask = pow(clamp(in.height * 0.3 + 0.5, 0.0, 1.0), 3.0); 
    let sss = SSS_COLOR * pow(sss_factor, 4.0) * sss_height_mask * 0.3;

    var color = water_base + sss;
    color = mix(color, sky_reflection, clamp(fresnel_sky, 0.0, 0.9));
    color += specular;
    color = mix(color, foam, foam_mask * FOAM_SCALE);

    // Tone mapping 
    color = aces_tone_map(color);
    color = pow(color, vec3<f32>(1.0/2.2));

    return vec4<f32>(color, 1.0);
}

// Apparently, water does NOT have diffuse
// fn diffuse_func(f_lambert: vec3<f32>, k_d: f32) -> vec3<f32> {
//     return k_d * f_lambert / pi;
// }

fn cook_torrance(n_dot_light: f32, n_dot_view: f32, n_dot_half: f32, alpha: f32, fresnel: f32) -> f32 {
    let normal_function = normal_func(n_dot_half, alpha);
    let geometry_function = geometry_func(n_dot_view, n_dot_light, alpha);
    return (normal_function * geometry_function * fresnel) / (4.0 * n_dot_light * n_dot_view + 1e-4);
}

fn normal_func(n_dot_half: f32, alpha: f32) -> f32 {
    let alpha2 = alpha * alpha;
    let alpha4 = alpha2 * alpha2;
    let denom = (n_dot_half * n_dot_half * (alpha2 - 1.0) + 1.0);
    return alpha2 / (pi * denom * denom);
}

fn smiths(dot_product: f32, alpha: f32) -> f32 {
    let k = ((alpha + 1.0) * (alpha + 1.0)) / 8.0;
    let denom = dot_product * (1.0 - k) + k;
    return dot_product / (denom + 1e-6);
}

fn geometry_func(n_dot_view: f32, n_dot_light: f32, alpha: f32) -> f32 {
    return smiths(n_dot_view, alpha) * smiths(n_dot_light, alpha);
}

fn fresnel_func(view_dot_half: f32, f_0: f32) -> f32 {
    let scale = pow(1.0 - view_dot_half, 5.0);
    return f_0 + (1.0 - f_0) * scale;
}

// HELP
fn aces_tone_map(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

const g: f32 = 9.81;
const pi: f32 = 3.14159;
const CAUSTIC_SCALE: f32 = 4.0;
const CAUSTIC_SPEED: f32 = 0.2;
const CAUSTIC_INTENSITY: f32 = 2.0;
const CAUSTIC_COLOR_TINT: vec3<f32> = vec3(0.95, 1.0, 1.05); 

struct OceanSettings {
    mesh_size: f32,         
    mesh_subdivisions: u32, 
    fft_size: f32,          
    fft_subdivisions: u32,  
    pass_num: u32,         
    time_scale: f32,        
    ocean_seed: u32, 
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
    roughness: f32,
    f_0: f32,
    specular_scale: f32,
    reflection_scale: f32,
    foam_scale: f32,
    sss_distortion_scale: f32,
    deep_color: vec4<f32>,
    shallow_color: vec4<f32>,
    sss_color: vec4<f32>,
    sun_color: vec4<f32>
}

@group(0) @binding(0)
var<uniform> ocean_settings: OceanSettings;

// Camera

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_sky: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
};

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

@group(2) @binding(0) var<storage, read> height_field: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read> height_field_dz: array<vec4<f32>>;
@group(2) @binding(2) var t_normal_map: texture_2d<f32>;
@group(2) @binding(3) var s_normal_map: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    // Quite redundant now, but too lazy to remove
    @location(2) index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) height: f32,  // Pass height to fragment shader
};


@group(3) @binding(0) var t_skybox: texture_cube<f32>;
@group(3) @binding(1) var s_skybox: sampler;

// To get cooler foam
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i + vec2(0.0, 0.0)), 
                   hash21(i + vec2(1.0, 0.0)), u.x),
               mix(hash21(i + vec2(0.0, 1.0)), 
                   hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var current_p = p;
    for (var i = 0u; i < 3u; i++) {
        var n = abs(noise(current_p) * 2.0 - 1.0);
        n = 1.0 - n;
        n = n * n;
        value += n * amplitude;
        current_p *= 2.5; // Higher lacunarity for finer details
        amplitude *= 0.5;
    }
    return value;
}

fn get_noise_normal(pos: vec2<f32>, strength: f32) -> vec3<f32> {
    let eps = 0.1;
    let n = fbm(pos);
    let dx = fbm(pos + vec2(eps, 0.0)) - n;
    let dz = fbm(pos + vec2(0.0, eps)) - n;
    return normalize(vec3<f32>(-dx * strength, 1.0, -dz * strength));
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
    
    let world_xz = model.position.xz;

    let delta = ocean_settings.fft_size / f32(ocean_settings.fft_subdivisions);
    let amp = ocean_settings.amplitude_scale;
    
    let h_dx_sample = sample_height_field(
        world_xz,
        ocean_settings.fft_size,
        ocean_settings.fft_subdivisions
    );
    let dz_sample = sample_dz_field(
        world_xz,
        ocean_settings.fft_size,
        ocean_settings.fft_subdivisions
    );
    
    let h = h_dx_sample.x * amp;
    let dx = h_dx_sample.z * amp;
    let dz = dz_sample.x * amp;
    
    var displaced_pos = model.position;
    displaced_pos.x += dx * ocean_settings.chop_scale;
    displaced_pos.y += h;
    displaced_pos.z += dz * ocean_settings.chop_scale;
    
    
    out.tex_coords = model.tex_coords;
    out.height = h;
    out.world_pos = displaced_pos;
    out.clip_position = camera.view_proj * vec4<f32>(displaced_pos, 1.0);
    
    return out;
}

// BPR work in progress
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    let fft_uv = in.world_pos.xz /  ocean_settings.fft_size;
    let map_data = textureSample(t_normal_map, s_normal_map, fft_uv);
    let normal_geometry = map_data.rgb;
    let jacobian = map_data.a;

    let normal_correct = normalize(normal_geometry);
    let light_source_dir = normalize(vec3(-0.506, 0.471, -0.722));
    let view_dir = normalize(camera.camera_pos - in.world_pos);
    let reflect_dir = reflect(-view_dir, normal_correct);
    let half_dir = normalize(light_source_dir + view_dir);

    let micro_uv = (in.world_pos.xz * 2.0) + ocean_settings.wind_vector * ocean_settings.time_scale * 0.15;
    let micro_normal = get_noise_normal(micro_uv, 0.5);

    let dist = length(camera.camera_pos - in.world_pos);
    let detail_fade = 1.0 - smoothstep(50.0, 500.0, dist);
    let blended_normal = normalize(mix(normal_geometry, micro_normal, detail_fade * 0.3));

    let normal = blended_normal;

    let n_dot_light = clamp(dot(normal, light_source_dir), 0.0, 1.0);
    let n_dot_view = clamp(dot(normal, view_dir), 0.001, 1.0);
    let n_dot_half = clamp(dot(normal, half_dir), 0.0, 1.0);
    let view_dot_half = clamp(dot(view_dir, half_dir), 0.0, 1.0);

    let trans_light_dir = normalize(light_source_dir + normal * ocean_settings.sss_distortion_scale);
    let trans_dot = max(dot(view_dir, -trans_light_dir), 0.0);

    let sss_mask = smoothstep(0.0, 1.0, in.height * 0.5 + 0.5);
    let sss_strength = pow(trans_dot, 4.0) * sss_mask * 2.0; 
    let sss = ocean_settings.sss_color.rgb * sss_strength;

    let fresnel_sky = fresnel_func(n_dot_view, ocean_settings.f_0);
    let fresnel_spec = fresnel_func(view_dot_half, ocean_settings.f_0);

    // Foam using jacobian
    let jacobian_mask = 1.0 - clamp(jacobian, 0.0, 1.0);
    let height_mask = smoothstep(-0.3, 0.5, in.height);
    let foam_uv = (in.world_pos.xz * 0.5) + ocean_settings.wind_vector * ocean_settings.time_scale * 0.05;
    let foam_texture = fbm(foam_uv);
    let foam_threshold = 0.5 - (foam_texture * 0.4); 
    let foam_factor = smoothstep(foam_threshold, foam_threshold + 0.1, jacobian_mask) * height_mask * ocean_settings.foam_scale;

    let water_base = mix(ocean_settings.deep_color.rgb, ocean_settings.shallow_color.rgb, smoothstep(-2.0, 3.0, in.height));

    let reflection_dampener = smoothstep(0.5, 1.0, jacobian);
    let sky_reflection = textureSample(t_skybox, s_skybox, reflect_dir).rgb * ocean_settings.reflection_scale * reflection_dampener * (1.0 - foam_factor);

    let rougness_dynamic = mix(ocean_settings.roughness, 0.6, foam_factor);
    let alpha = rougness_dynamic * rougness_dynamic;
    let specular_val = cook_torrance(n_dot_light, n_dot_view, n_dot_half, alpha, fresnel_spec);
    let specular = ocean_settings.sun_color.rgb * specular_val * ocean_settings.specular_scale * reflection_dampener * (1.0-foam_factor); // cause foam blocks the reflections yk

    let ambient = vec3<f32>(0.02, 0.03, 0.04);

    var color = water_base + sss;

    let depth_fade = smoothstep(-2.0, 2.5, in.height);
    let view_angle_fade = pow(max(view_dir.y, 0.0), 0.40);
    let parallax_offset = -view_dir.xz * 0.25 / max(view_dir.y, 0.1);
    
    let caustic_uv = (in.world_pos.xz * 0.18) + parallax_offset;
    let caustic_pattern = get_caustics_procedural(caustic_uv, camera.time);
    
    let sun_facing = pow(clamp(dot(normal, light_source_dir), 0.0, 1.0), 0.7);
    let foam_mask = 1.0 - foam_factor * 0.7;
    let dist_fade = 1.0 - smoothstep(50.0, 600.0, dist);
    
    let caustic_strength = sun_facing * foam_mask * dist_fade * depth_fade * view_angle_fade * 2.0;
    let caustic_color = caustic_pattern * CAUSTIC_COLOR_TINT * caustic_strength;
    
    color += caustic_color * color * 0.6;

    color = mix(color, sky_reflection, clamp(fresnel_sky, 0.0, 1.0));
    color += specular;
    let foam_color = vec3<f32>(0.95, 0.98, 0.92); 
    color = mix(color, foam_color, clamp(foam_factor, 0.0, 1.0));

    // Tone mapping 
    color = aces_tone_map(color);
    color = pow(color, vec3<f32>(1.0/2.2));

    return vec4<f32>(color, 1.0);
}

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

fn hash13(p3: vec3<f32>) -> f32 {
    var p = fract(p3 * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn hash33(p3: vec3<f32>) -> vec3<f32> {
    var p = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    
    return mix(
        mix(
            mix(hash13(i + vec3(0.0, 0.0, 0.0)), hash13(i + vec3(1.0, 0.0, 0.0)), u.x),
            mix(hash13(i + vec3(0.0, 1.0, 0.0)), hash13(i + vec3(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
        mix(
            mix(hash13(i + vec3(0.0, 0.0, 1.0)), hash13(i + vec3(1.0, 0.0, 1.0)), u.x),
            mix(hash13(i + vec3(0.0, 1.0, 1.0)), hash13(i + vec3(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
        u.z
    );
}

fn fbm3d(p: vec3<f32>, octaves: u32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var current_p = p;
    
    for (var i = 0u; i < octaves; i++) {
        value += amplitude * noise3d(current_p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    
    return value;
}

fn get_caustics_procedural(uv: vec2<f32>, time: f32) -> vec3<f32> {
    let flow_speed = time * CAUSTIC_SPEED;
    let flow1 = vec2(
        fbm3d(vec3(uv * 0.5, flow_speed * 0.3), 3u),
        fbm3d(vec3(uv * 0.5 + vec2(5.2, 1.3), flow_speed * 0.3), 3u)
    ) * 0.3;
    
    let flow2 = vec2(
        fbm3d(vec3(uv * 0.8 + vec2(2.1, 3.4), flow_speed * 0.5), 2u),
        fbm3d(vec3(uv * 0.8 + vec2(7.3, 2.1), flow_speed * 0.5), 2u)
    ) * 0.15;
    
    let distorted_uv = uv + flow1 + flow2;
    
    let scale1 = CAUSTIC_SCALE;
    let scale2 = CAUSTIC_SCALE * 1.7;
    
    let v1 = voronoi_caustic(distorted_uv * scale1, time);
    let v2 = voronoi_caustic(distorted_uv * scale2 + vec2(3.5, 7.2), time * 1.3);
    
    var caustic = v1 * v2;
    caustic = pow(caustic, 3.0);
    caustic = smoothstep(0.1, 0.9, caustic);
    
    let aberration = 0.015;
    let r = pow(voronoi_caustic(distorted_uv * scale1 + vec2(aberration, 0.0), time) * 
                voronoi_caustic(distorted_uv * scale2 + vec2(aberration, 0.0) + vec2(3.5, 7.2), time * 1.3), 3.0);
    let g = caustic;
    let b = pow(voronoi_caustic(distorted_uv * scale1 - vec2(aberration, 0.0), time) * 
                voronoi_caustic(distorted_uv * scale2 - vec2(aberration, 0.0) + vec2(3.5, 7.2), time * 1.3), 3.0);
    
    return vec3(r, g, b) * CAUSTIC_INTENSITY;
}

fn voronoi_caustic(uv: vec2<f32>, time: f32) -> f32 {
    let p = floor(uv);
    let f = fract(uv);
    
    var min_dist = 100.0;
    var second_min = 100.0;
    
    for(var j = -1; j <= 1; j++) {
        for(var i = -1; i <= 1; i++) {
            let neighbor = vec2(f32(i), f32(j));
            let cell_id = p + neighbor;
            
            let h = hash22(cell_id);
            let offset = vec2(
                sin(time * 0.5 + h.x * 6.28) * 0.45 + 0.5,
                cos(time * 0.7 + h.y * 6.28) * 0.45 + 0.5
            );
            
            let point = neighbor + offset - f;
            let d = length(point);
            
            if (d < min_dist) {
                second_min = min_dist;
                min_dist = d;
            } else if (d < second_min) {
                second_min = d;
            }
        }
    }
    
    let edge_dist = second_min - min_dist;
    let caustic = 1.0 - smoothstep(0.0, 0.2, edge_dist);
    return pow(caustic, 1.5);
}

fn sample_height_field(
    world_pos: vec2<f32>,
    fft_size: f32,
    subdivisions: u32
) -> vec4<f32> {
    let uv = world_pos / fft_size;
    let tex_coords = uv * f32(subdivisions);

    let i = floor(tex_coords);
    let f = fract(tex_coords);

    let i00 = vec2<u32>(i);
    let i10 = vec2<u32>(i + vec2(1.0, 0.0));
    let i01 = vec2<u32>(i + vec2(0.0, 1.0));
    let i11 = vec2<u32>(i + vec2(1.0, 1.0));

    let n = subdivisions;
    let idx00 = (i00.y % n) * n + (i00.x % n);
    let idx10 = (i10.y % n) * n + (i10.x % n);
    let idx01 = (i01.y % n) * n + (i01.x % n);
    let idx11 = (i11.y % n) * n + (i11.x % n);

    let s00 = height_field[idx00];
    let s10 = height_field[idx10];
    let s01 = height_field[idx01];
    let s11 = height_field[idx11];

    let x1 = mix(s00, s10, f.x);
    let x2 = mix(s01, s11, f.x);
    return mix(x1, x2, f.y);
}

fn sample_dz_field(
    world_pos: vec2<f32>,
    fft_size: f32,
    subdivisions: u32
) -> vec4<f32> {
    let uv = world_pos / fft_size;
    let tex_coords = uv * f32(subdivisions);

    let i = floor(tex_coords);
    let f = fract(tex_coords);

    let i00 = vec2<u32>(i);
    let i10 = vec2<u32>(i + vec2(1.0, 0.0));
    let i01 = vec2<u32>(i + vec2(0.0, 1.0));
    let i11 = vec2<u32>(i + vec2(1.0, 1.0));

    let n = subdivisions;
    let idx00 = (i00.y % n) * n + (i00.x % n);
    let idx10 = (i10.y % n) * n + (i10.x % n);
    let idx01 = (i01.y % n) * n + (i01.x % n);
    let idx11 = (i11.y % n) * n + (i11.x % n);

    let s00 = height_field_dz[idx00];
    let s10 = height_field_dz[idx10];
    let s01 = height_field_dz[idx01];
    let s11 = height_field_dz[idx11];

    let x1 = mix(s00, s10, f.x);
    let x2 = mix(s01, s11, f.x);
    return mix(x1, x2, f.y);
}

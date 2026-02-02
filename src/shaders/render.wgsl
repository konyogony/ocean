const g: f32 = 9.81;
const pi: f32 = 3.14159;
const SSS_MIN_HEIGHT: f32 = -0.5;
const SSS_MAX_HEIGHT: f32 = 1.5;
const SSS_POWER: f32 = 8.0;
const SSS_INTENSITY: f32 = 2.0;
const DETAIL_FADE: f32 = 800.0;
const AMBIENT_SCALE: f32 = 0.25;

struct OceanSettingsUniform {
    deep_color: vec4<f32>,
    shallow_color: vec4<f32>,
    sss_color: vec4<f32>,
    caustic_color_tint: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color_day_zenith: vec4<f32>,
    sky_color_day_horizon: vec4<f32>,
    sky_color_night_zenith: vec4<f32>,
    sky_color_night_horizon: vec4<f32>,
    sky_color_sunset_orange: vec4<f32>,
    sky_color_sunset_pink: vec4<f32>,
    sky_color_horizon_glow: vec4<f32>,
    moon_color_lit: vec4<f32>,
    moon_color_dark: vec4<f32>,
    cloud_color_night: vec4<f32>,
    cloud_color_day: vec4<f32>,
    wind_vector: vec2<f32>,
    _pad_vec2: vec2<f32>, // Pad
    moon_phase_offset: vec3<f32>, 
    _pad_moon: f32, // Pad
    mesh_size: f32,
    fft_size: f32,
    time_scale: f32,
    chop_scale: f32,
    amplitude_scale: f32,
    wave_scale: f32,
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
    caustic_scale: f32,
    caustic_size: f32,
    caustic_speed: f32,
    caustic_intensity: f32,
    caustic_depth: f32,
    caustic_max_distance: f32,
    micro_normal_strength: f32,
    foam_threshold: f32,
    foam_speed: f32,
    foam_roughness: f32,
    daynight_cycle: f32,
    sun_offset_z: f32,
    sun_size_inner: f32,
    sun_size_outer: f32,
    sun_halo_power: f32,
    moon_radius: f32,
    moon_dist: f32,
    moon_crater_scale: f32,
    moon_halo_power: f32,
    star_count: f32,
    star_threshold: f32,
    star_blink_speed: f32,
    cloud_speed: f32,
    cloud_density_low: f32,
    cloud_density_high: f32,
    mesh_subdivisions: u32,
    fft_subdivisions: u32,
    pass_num: u32,
    ocean_seed: u32,
    caustic_octaves: u32,
    pad_a: vec2<u32>,
    pad_b: vec4<u32>, 
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_sky: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) height: f32,
    @location(3) normal: vec3<f32>,
    @location(4) jacobian: f32,
};

@group(0) @binding(0) var<uniform> ocean_settings: OceanSettingsUniform;

@group(1) @binding(0) var<uniform> camera: CameraUniform;

@group(2) @binding(0) var texture_h_dx: texture_2d<f32>;
@group(2) @binding(1) var texture_dz: texture_2d<f32>;
@group(2) @binding(2) var sampler_ocean: sampler;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_xz = model.position.xz;
    let uv = world_xz / ocean_settings.fft_size; 
    let amp = ocean_settings.amplitude_scale;
    let chop = ocean_settings.chop_scale;
    
    let h_dx_sample = textureSampleLevel(texture_h_dx, sampler_ocean, uv, 0.0);
    let dz_sample = textureSampleLevel(texture_dz, sampler_ocean, uv, 0.0);

    let h = h_dx_sample.x * amp;
    let dx = h_dx_sample.z * amp;
    let dz = dz_sample.x * amp;

    var displaced_pos = vec3(
        world_xz.x + dx * chop,
        model.position.y + h,
        world_xz.y + dz * chop
    );
    
    // Moving everything to FS to get smoother normals
    // Better delta
    // let delta_uv = 1.0 / f32(ocean_settings.fft_subdivisions);
    // let sample_r = textureSampleLevel(texture_h_dx, sampler_ocean, uv + vec2(delta_uv, 0.0), 0.0).x;
    // let sample_l = textureSampleLevel(texture_h_dx, sampler_ocean, uv - vec2(delta_uv, 0.0), 0.0).x;
    // let sample_u = textureSampleLevel(texture_h_dx, sampler_ocean, uv + vec2(0.0, delta_uv), 0.0).x;
    // let sample_d = textureSampleLevel(texture_h_dx, sampler_ocean, uv - vec2(0.0, delta_uv), 0.0).x;
    // let world_step = delta_uv * ocean_settings.fft_size;
    // let ddx_h = (sample_r - sample_l) * amp / (2.0 * world_step);
    // let ddz_h = (sample_u - sample_d) * amp / (2.0 * world_step);
    // 
    // out.normal = normalize(vec3<f32>(-ddx_h, 1.0, -ddz_h));
    // 
    // out.jacobian = (1.0 + (dx/world_step)) * (1.0 + (dz/world_step)); 

    out.normal = vec3(0.0);
    out.jacobian = 1.0;
    out.tex_coords = uv;
    out.height = displaced_pos.y;
    out.world_pos = displaced_pos;
    out.clip_position = camera.view_proj * vec4<f32>(displaced_pos, 1.0);

    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // For cool normals
    let uv = in.tex_coords;
    let amp = ocean_settings.amplitude_scale;
    let chop = ocean_settings.chop_scale;
    let delta_uv = 1.0 / f32(ocean_settings.fft_subdivisions);
    let world_step = delta_uv * ocean_settings.fft_size;

    let sample_r = textureSampleLevel(texture_h_dx, sampler_ocean, uv + vec2(delta_uv, 0.0), 0.0).x;
    let sample_l = textureSampleLevel(texture_h_dx, sampler_ocean, uv - vec2(delta_uv, 0.0), 0.0).x;
    let sample_u = textureSampleLevel(texture_h_dx, sampler_ocean, uv + vec2(0.0, delta_uv), 0.0).x;
    let sample_d = textureSampleLevel(texture_h_dx, sampler_ocean, uv - vec2(0.0, delta_uv), 0.0).x;

    let ddx_h = (sample_r - sample_l) * amp / (2.0 * world_step);
    let ddz_h = (sample_u - sample_d) * amp / (2.0 * world_step);

    let dx_r = textureSample(texture_h_dx, sampler_ocean, uv + vec2(delta_uv, 0.0)).z;
    let dx_l = textureSample(texture_h_dx, sampler_ocean, uv - vec2(delta_uv, 0.0)).z;
    let dz_u = textureSample(texture_dz, sampler_ocean, uv + vec2(0.0, delta_uv)).x;
    let dz_d = textureSample(texture_dz, sampler_ocean, uv - vec2(0.0, delta_uv)).x;
    
    let ddx_dx = (dx_r - dx_l) * amp * chop / (2.0 * world_step);
    let ddz_dz = (dz_u - dz_d) * amp * chop / (2.0 * world_step);
    let jacobian = (1.0 + ddx_dx) * (1.0 + ddz_dz);
    let normal_geometry = normalize(vec3<f32>(-ddx_h, 1.0, -ddz_h));

    // Use new sun position & get stuff from the skybox shader
    let angle = ocean_settings.daynight_cycle * 6.28318;
    let sun_dir = normalize(vec3(sin(angle), cos(angle), ocean_settings.sun_offset_z));
    let sun_up = sun_dir.y;
    let intensity = smoothstep(-0.2, 0.25, sun_up);
    let night_fade = clamp(-sun_up * 4.0, 0.0, 1.0);
    let moon_base = -sun_dir;
    let moon_dir = normalize(moon_base + ocean_settings.moon_phase_offset);
    let sun_light_color = ocean_settings.sun_color.rgb * intensity;
    let moon_light_color = ocean_settings.moon_color_lit.rgb * night_fade * 0.25;

    let light_dir = normalize(sun_dir * intensity + moon_dir * night_fade);
    let light_color = sun_light_color + moon_light_color;
    let zenith = mix(ocean_settings.sky_color_night_zenith.rgb, ocean_settings.sky_color_day_zenith.rgb, intensity);
    let horizon = mix(ocean_settings.sky_color_night_horizon.rgb, ocean_settings.sky_color_day_horizon.rgb, intensity);
    let ambient = mix(horizon, zenith, clamp(normal_geometry.y * 0.5 + 0.5, 0.0, 1.0));

    let view_dir = normalize(camera.camera_pos - in.world_pos);
    let reflect_dir = reflect(-view_dir, normal_geometry);
    let half_dir = normalize(light_dir + view_dir);

    let micro_uv = (in.world_pos.xz * 2.0) + ocean_settings.wind_vector * camera.time * 0.15;
    let micro_normal = get_noise_normal(micro_uv, 0.5);

    let dist = length(camera.camera_pos - in.world_pos);
    let detail_fade = 1.0 - smoothstep(50.0, DETAIL_FADE, dist);
    let T = normalize(cross(vec3(0.0,1.0,0.0), normal_geometry));
    let B = cross(normal_geometry, T);
    
    let micro_ws = normalize(
        micro_normal.x * T +
        micro_normal.y * normal_geometry +
        micro_normal.z * B
    );
    
    let normal = normalize(mix(normal_geometry, micro_ws, detail_fade * ocean_settings.micro_normal_strength));

    let n_dot_light = clamp(dot(normal, light_dir), 0.0, 1.0);
    let n_dot_view = clamp(dot(normal, view_dir), 0.001, 1.0);
    let n_dot_half = clamp(dot(normal, half_dir), 0.0, 1.0);
    let view_dot_half = clamp(dot(view_dir, half_dir), 0.0, 1.0);

    let trans_light_dir = normalize(light_dir + normal * ocean_settings.sss_distortion_scale) * smoothstep(0.5, 1.2, jacobian);
    let trans_dot = max(dot(view_dir, -trans_light_dir), 0.0);

    // basically also from the skybox shader
    // let sunset_timing = exp(-pow(sun_up * 3.5, 2.0)) * smoothstep(-0.3, 0.3, sun_up);
    // let reflection_sun_angle = max(dot(reflect_dir, vec3(sun_dir.x, 0.0, sun_dir.z)), 0.0);
    // let reflection_sunset = pow(reflection_sun_angle, 2.0) * sunset_timing;
    // var sky_reflection = mix(horizon, zenith, pow(max(reflect_dir.y, 0.0), 0.6));
    // sky_reflection += ocean_settings.sky_color_sunset_orange.rgb * reflection_sunset * 2.0;

    let p_back = pow(trans_dot, SSS_POWER);
    let sss_thickness_mask = 1.0 - smoothstep(SSS_MIN_HEIGHT, SSS_MAX_HEIGHT, in.height);
    let sss_strength = p_back * sss_thickness_mask * ocean_settings.sss_distortion_scale * SSS_INTENSITY;
    let sss = mix(ocean_settings.sss_color.rgb, light_color, p_back) * sss_strength;

    let fresnel_sky = fresnel_func(n_dot_view, ocean_settings.f_0);
    let fresnel_spec = fresnel_func(view_dot_half, ocean_settings.f_0);

    // better foam algo
    let jacobian_mask = 1.0 - clamp(jacobian, 0.0, 1.0);
    let height_mask = smoothstep(-0.2, 0.6, in.height);
    let foam_uv = (in.world_pos.xz * ocean_settings.foam_scale) + ocean_settings.wind_vector * camera.time * ocean_settings.foam_speed;
    let foam_val = get_foam(foam_uv, camera.time);
    let wave_crest_mask = smoothstep(ocean_settings.foam_threshold, ocean_settings.foam_threshold + 0.2, jacobian_mask) * height_mask;
    let foam_factor = smoothstep(0.4, 0.7, foam_val) * wave_crest_mask * ocean_settings.foam_scale;

    let turbidity = smoothstep(0.1, 0.5, jacobian); // basically compression
    let base_mix = smoothstep(-2.0, 4.0, in.height);
    var water_base = mix(ocean_settings.deep_color.rgb, ocean_settings.shallow_color.rgb, base_mix);
    water_base = mix(water_base, ocean_settings.sss_color.rgb * 0.5, turbidity * 0.5);
    water_base *= max(intensity, 0.1);

    let reflection_dampener = smoothstep(0.5, 1.0, jacobian);
    let rougness_dynamic = mix(ocean_settings.roughness, ocean_settings.foam_roughness, foam_factor);
    let alpha = rougness_dynamic * rougness_dynamic;
    let specular_val = cook_torrance(n_dot_light, n_dot_view, n_dot_half, alpha, fresnel_spec);
    let specular = light_color * specular_val * ocean_settings.specular_scale * reflection_dampener * (1.0-foam_factor);

    var color = water_base + sss;
    
    let depth_fade = smoothstep(-2.0, 2.5, in.height);
    let view_angle_fade = pow(max(view_dir.y, 0.0), 0.40);
    let parallax_offset = -view_dir.xz * ocean_settings.caustic_depth / max(view_dir.y, 0.1);
    let caustic_uv = (in.world_pos.xz * ocean_settings.caustic_scale) + parallax_offset;
    let caustic_pattern = get_caustics_procedural(caustic_uv, camera.time);
    let sun_facing = pow(clamp(dot(normal, sun_dir), 0.0, 1.0), 0.7);
    let foam_mask = 1.0 - foam_factor * 0.7;
    let dist_fade = 1.0 - smoothstep(50.0, ocean_settings.caustic_max_distance, dist);
    let choppiness_mask = smoothstep(0.7, 1.0, jacobian);

    let caustic_strength = sun_facing * foam_mask * dist_fade * depth_fade * view_angle_fade * choppiness_mask * 2.0;
    let caustic_color = caustic_pattern * ocean_settings.caustic_color_tint.rgb * caustic_strength * max(intensity, 2.0);

    color += caustic_color * ocean_settings.sss_color.rgb * 0.6;
    
    let sky_reflection = get_sky_color(reflect_dir) * ocean_settings.reflection_scale;
    color = mix(color, sky_reflection, clamp(fresnel_sky, 0.0, 1.0));
    color += ambient * AMBIENT_SCALE;
    color += specular;
    
    let foam_color = vec3<f32>(0.95, 0.98, 0.92);
    color = mix(color, foam_color, clamp(foam_factor, 0.0, 1.0));

    color = aces_tone_map(color);
    color = pow(color, vec3<f32>(1.0/2.2));

    return vec4<f32>(color, 1.0);
}

// copied form other shader
fn get_sky_color(view_dir: vec3<f32>) -> vec3<f32> {
    let dir = normalize(view_dir);
    let angle = ocean_settings.daynight_cycle * 6.28318;
    
    let sun_dir = normalize(vec3(sin(angle), cos(angle), ocean_settings.sun_offset_z));
    let sun_up = sun_dir.y;
    
    let moon_base = -sun_dir;
    let moon_dir = normalize(moon_base + ocean_settings.moon_phase_offset); 
    
    let intensity = smoothstep(-0.2, 0.5, sun_up);
    let night_fade = clamp(-sun_up * 3.0, 0.0, 1.0);

    let zenith = mix(ocean_settings.sky_color_night_zenith.rgb, ocean_settings.sky_color_day_zenith.rgb, intensity);
    let horizon = mix(ocean_settings.sky_color_night_horizon.rgb, ocean_settings.sky_color_day_horizon.rgb, intensity);
    var col = mix(horizon, zenith, pow(max(dir.y, 0.0), 0.7));

    let sunset_timing = exp(-pow(sun_up * 4.0, 2.0)) * smoothstep(-0.2, 0.4, sun_up);
    let sunset_angle = max(dot(dir, normalize(vec3(sun_dir.x, 0.0, sun_dir.z))), 0.0);
    let sunset_vertical = smoothstep(-0.1, 0.4, dir.y) * smoothstep(0.7, 0.2, dir.y);
    
    let sunset_mix = mix(ocean_settings.sky_color_sunset_pink.rgb, ocean_settings.sky_color_sunset_orange.rgb, 0.7);
    col += sunset_mix * pow(sunset_angle, 1.5) * sunset_vertical * sunset_timing * 2.5;
    col += ocean_settings.sky_color_horizon_glow.rgb * pow(1.0 - abs(dir.y), 3.0) * sunset_timing * 0.3;

    let sun_dist = dot(dir, sun_dir);
    let sun_disk = smoothstep(ocean_settings.sun_size_inner, ocean_settings.sun_size_outer, sun_dist);
    let sun_halo = pow(max(sun_dist, 0.0), ocean_settings.sun_halo_power) * 0.5;
    col += (sun_disk + sun_halo) * mix(ocean_settings.sun_color.rgb, ocean_settings.sky_color_sunset_orange.rgb, sunset_timing) * intensity;

    var hit_moon = false;
    
    if (dot(dir, moon_dir) > 0.0) {
        let center = moon_dir * ocean_settings.moon_dist;
        let radius = ocean_settings.moon_dist * ocean_settings.moon_radius; 
        
        let oc = -center;
        let b = dot(oc, dir);
        let c = dot(oc, oc) - radius * radius;
        let h = b * b - c;
        
        if (h > 0.0) {
            hit_moon = true;
            let t = -b - sqrt(h);
            let hit_pos = dir * t;
            let normal = normalize(hit_pos - center);
            let uv_moon = normal.xy * 2.0; 
            let craters = 1.0 - moon_fbm(uv_moon * ocean_settings.moon_crater_scale) * 0.6; 
            let diffuse = max(dot(normal, sun_dir), 0.0);
            let terminator = smoothstep(-0.15, 0.15, diffuse);
            let surface_col = mix(ocean_settings.moon_color_dark.rgb, ocean_settings.moon_color_lit.rgb * craters * 0.8, terminator);
            let rim_aa = smoothstep(0.0, 0.05, sqrt(h));
            let moon_visibility = 1.0 - (intensity * 0.7); 
            col = mix(col, surface_col * moon_visibility, rim_aa);
        } 
    }
    
    if (!hit_moon) {
        let moon_dist_ang = dot(dir, moon_dir);
        let moon_glow = pow(max(moon_dist_ang, 0.0), ocean_settings.moon_halo_power) * 0.1 * night_fade;
        col += vec3(0.5, 0.6, 0.8) * moon_glow;
    }

    if (!hit_moon && sun_up < 0.2) {
        let star_grid = floor(dir * ocean_settings.star_count); 
        let star_hash = hash31(star_grid);
        if (star_hash > ocean_settings.star_threshold) {
            let blink = sin(camera.time * (ocean_settings.star_blink_speed + star_hash) + star_hash * 50.0) * 0.4 + 0.6;
            col += vec3(blink * star_hash) * night_fade * smoothstep(0.0, 0.2, dir.y);
        }
    }

    if (dir.y > 0.0) {
        let uv = (dir.xz / (dir.y + 0.25)) * 0.6;
        let time_scale = camera.time * ocean_settings.cloud_speed;
        let d = cloud_density(uv, time_scale);
        let cloud_mask = smoothstep(ocean_settings.cloud_density_low, ocean_settings.cloud_density_high, d) * pow(dir.y, 1.5);
        let d_shadow = cloud_density(uv - sun_dir.xz * 0.05, time_scale);
        let cloud_lit = clamp(d - d_shadow, 0.0, 1.0);
        let cloud_sunset_tint = (ocean_settings.sky_color_sunset_orange.rgb + ocean_settings.sky_color_sunset_pink.rgb * 0.5) * sunset_timing * 0.5;
        let cloud_day_final = mix(ocean_settings.cloud_color_day.rgb * 0.9, vec3(1.0), cloud_lit) + cloud_sunset_tint;
        let cloud_final = mix(ocean_settings.cloud_color_night.rgb, cloud_day_final, intensity);
        col = mix(col, cloud_final, cloud_mask * 0.95);
    }

    return col;
}


fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash_v2(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.13);
    p3 += dot(p3, p3.yzx + 3.333);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise_skybox(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(mix(hash_v2(i + vec2(0.0, 0.0)), hash_v2(i + vec2(1.0, 0.0)), u.x),
               mix(hash_v2(i + vec2(0.0, 1.0)), hash_v2(i + vec2(1.0, 1.0)), u.x), u.y);
}

fn fbm_skybox(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var pos = p;
    for (var i = 0; i < 5; i++) {
        v += a * noise(pos);
        pos *= 2.05;
        a *= 0.5;
    }
    return v;
}

fn moon_fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var pos = p;
    for (var i = 0; i < 6; i++) {
        v += a * abs(noise(pos) * 2.0 - 1.0); 
        pos *= 2.1;
        a *= 0.5;
    }
    return v;
}

fn cloud_density(p: vec2<f32>, t: f32) -> f32 {
    let drift = vec2(t * 0.02, t * 0.01); // Internal drift ratios
    let v1 = fbm(p + drift);
    let v2 = fbm(p - drift * 0.5 + v1 * 0.8);
    return v2;
}


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
        current_p *= 2.5;
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
    let flow_speed = time * ocean_settings.caustic_speed * ocean_settings.time_scale;
    let aberration_strength = 0.01;

    let flow1 = vec2(
        fbm3d(vec3(uv * 0.5, flow_speed * 0.3), 3u),
        fbm3d(vec3(uv * 0.5 + vec2(5.2, 1.3), flow_speed * 0.3), 3u)
    ) * 0.3;

    let flow2 = vec2(
        fbm3d(vec3(uv * 0.8 + vec2(2.1, 3.4), flow_speed * 0.5), 2u),
        fbm3d(vec3(uv * 0.8 + vec2(7.3, 2.1), flow_speed * 0.5), 2u)
    ) * 0.15;

    let distorted_uv = uv + flow1 + flow2;
    var uv_rgb: array<vec2<f32>, 3>;
    uv_rgb[0] = distorted_uv + vec2(aberration_strength, 0.0);
    uv_rgb[1] = distorted_uv;
    uv_rgb[2] = distorted_uv - vec2(aberration_strength, 0.0);

    var value_rgb: vec3<f32> = vec3<f32>(0.0);
    var amplitude_rgb: vec3<f32> = vec3<f32>(1.0);
    var frequency_rgb: vec3<f32> = vec3<f32>(ocean_settings.caustic_size);

    for (var i = 0u; i < ocean_settings.caustic_octaves; i++) {
        let v_r = voronoi_caustic(uv_rgb[0] * frequency_rgb.r + f32(i) * 10.0, time);
        value_rgb.r += pow(v_r, 2.5) * amplitude_rgb.r;

        let v_g = voronoi_caustic(uv_rgb[1] * frequency_rgb.g + f32(i) * 10.0, time);
        value_rgb.g += pow(v_g, 2.5) * amplitude_rgb.g;

        let v_b = voronoi_caustic(uv_rgb[2] * frequency_rgb.b + f32(i) * 10.0, time);
        value_rgb.b += pow(v_b, 2.5) * amplitude_rgb.b;
    }

    let rgb = vec3<f32>(smoothstep(0.6, 1.0, value_rgb.r), smoothstep(0.6, 1.0, value_rgb.g), smoothstep(0.6, 1.0, value_rgb.b));
    return rgb * ocean_settings.caustic_intensity;

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

fn get_foam(uv: vec2<f32>, time: f32) -> f32 {
    let warp_uv = uv * 0.5;
    let warp_strength = 0.5;
    let warp = vec2<f32>(
        noise(warp_uv + vec2(time * 0.1, 0.0)),
        noise(warp_uv + vec2(5.2, 1.3) - vec2(0.0, time * 0.1))
    );

    let p = uv + warp * warp_strength;

    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var current_p = p;
    
    for (var i = 0u; i < 3u; i++) {
        let n = noise(current_p); 
        value += n * amplitude;
        current_p *= 2.0; 
        amplitude *= 0.5;
    }
    return pow(value, 1.5); 
}

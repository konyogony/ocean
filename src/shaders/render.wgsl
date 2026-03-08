const g: f32 = 9.81;
const pi: f32 = 3.14159;

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
    _pad_vec2: vec2<f32>,
    moon_phase_offset: vec3<f32>,
    _pad_moon: f32,
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
    star_size: f32,
    aurora_strength: f32,
    pad_b: vec4<u32>,
    cascade_data: array<vec4<f32>, 6>,
    cascade_count: u32,
    _pad_cascade_0: u32,
    _pad_cascade_1: u32,
    _pad_cascade_2: u32,
    sunset_scatter_color: vec3<f32>,
    sunset_scatter_intensity: f32,
    foam_base_color: vec3<f32>,
    sss_min_height: f32,
    sss_max_height: f32,
    sss_power: f32,
    sss_intensity: f32,
    detail_fade: f32,
    ambient_scale: f32,
    blend_strength: f32,
    bloom_scale: f32,
    reflection_min: f32,
    reflection_max: f32,
    moon_light_dim: f32,
    sky_zenith_gradient_exp: f32,
    horizon_glow_mult: f32,
    sunset_orange_weight: f32,
    sunset_intensity: f32,
    sun_halo_intensity: f32,
    moon_halo_intensity: f32,
    micro_uv_freq: f32,
    micro_time_freq: f32,
    micro_strength_mod: f32,
    foam_crest_width: f32,
    caustic_aberration: f32,
    caustic_smooth_low: f32,
    caustic_smooth_high: f32,
    aurora_brightness: f32,
    aurora_y_threshold: f32,
    water_brightness_mod: f32,
    decay_factor: f32,
    dissipation_factor: f32,
    warp_uv_scale: f32,
    warp_strength: f32,
    warp_time_scale: f32,
    foam_octaves: u32,
    foam_power: f32,
    hash_scale: f32,
    hash_dot: f32,
    steepness_threshold_low: f32,
    steepness_threshold_high: f32,
    y_displacement_weight: f32,
    wave_epsilon: f32,
    _pad_final_0: f32,
    _pad_final_1: vec4<f32>,
    _pad_final_2: vec4<f32>,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_sky: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad1: f32,
    time: f32,
    delta_time: f32,
    _padding: vec2<f32>,
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

@group(3) @binding(0) var foam_texture: texture_2d<f32>;
@group(3) @binding(1) var sampler_foam: sampler;


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
    let h_enhanced = h + pow(abs(h), 1.3) * sign(h) * 0.15;
    let dx = h_dx_sample.z;
    let dz = dz_sample.x;

    var displaced_pos = vec3(
        world_xz.x + dx * chop,
        h_enhanced,
        world_xz.y + dz * chop
    );
    
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
    let angle = (ocean_settings.daynight_cycle - 0.5) * 6.28318;
    let sun_orbit_y = cos(angle);
    let sun_dir = normalize(vec3(sin(angle), sun_orbit_y, ocean_settings.sun_offset_z));
    let sun_up = sun_orbit_y;
    let intensity = smoothstep(-0.3, 0.3, sun_up);
    let night_fade = smoothstep(0.3, -0.3, sun_up);
    let moon_base = -sun_dir;
    let moon_dir = normalize(moon_base + ocean_settings.moon_phase_offset);

    let sun_light_color = ocean_settings.sun_color.rgb * intensity;
    let moon_light_color = ocean_settings.moon_color_lit.rgb * night_fade * ocean_settings.moon_light_dim;
    let light_dir = normalize(sun_dir * intensity + moon_dir * night_fade);
    let light_color = sun_light_color + moon_light_color;

    let zenith = mix(ocean_settings.sky_color_night_zenith.rgb, ocean_settings.sky_color_day_zenith.rgb, intensity);
    let horizon = mix(ocean_settings.sky_color_night_horizon.rgb, ocean_settings.sky_color_day_horizon.rgb, intensity);
    let ambient = mix(horizon, zenith, clamp(normal_geometry.y * 0.5 + 0.5, 0.0, 1.0));

    let dist = length(camera.camera_pos - in.world_pos);
    let cam_height = camera.camera_pos.y;
    let height_fade = 1.0 - smoothstep(20.0, 200.0, cam_height);
    let detail_fade = 1.0 - smoothstep(50.0, ocean_settings.detail_fade, dist) * height_fade;

    let micro_uv = (in.world_pos.xz * ocean_settings.micro_uv_freq) + ocean_settings.wind_vector * camera.time * ocean_settings.micro_time_freq;
    let micro_normal = get_noise_normal(micro_uv, 0.8);

    let T = normalize(cross(vec3(0.0, 1.0, 0.0), normal_geometry));
    let B = cross(normal_geometry, T);
    
    let micro_ws = normalize(
        micro_normal.x * T +
        micro_normal.y * normal_geometry +
        micro_normal.z * B
    );
    
    let normal = normalize(mix(normal_geometry, micro_ws, detail_fade * ocean_settings.micro_normal_strength * ocean_settings.micro_strength_mod));

    let view_dir = normalize(camera.camera_pos - in.world_pos);
    let half_dir = normalize(light_dir + view_dir);
    let reflect_dir = reflect(-view_dir, normal);

    let n_dot_light = clamp(dot(normal, light_dir), 0.0, 1.0);
    let n_dot_view = clamp(dot(normal, view_dir), 0.001, 1.0);
    let n_dot_half = clamp(dot(normal, half_dir), 0.0, 1.0);
    let view_dot_half = clamp(dot(view_dir, half_dir), 0.0, 1.0);

    let trans_light_dir = normalize(light_dir + normal * ocean_settings.sss_distortion_scale) * smoothstep(0.5, 1.2, jacobian);
    let trans_dot = max(dot(view_dir, -trans_light_dir), 0.0);
    let p_back = pow(trans_dot, ocean_settings.sss_power);
    let sss_thickness_mask = 1.0 - smoothstep(ocean_settings.sss_min_height, ocean_settings.sss_max_height, in.height);
    let sss_strength = p_back * sss_thickness_mask * ocean_settings.sss_distortion_scale * ocean_settings.sss_intensity;
    let sss = mix(ocean_settings.sss_color.rgb, light_color, p_back) * sss_strength;

    let fresnel_sky = fresnel_func(n_dot_view, ocean_settings.f_0);
    let fresnel_spec = fresnel_func(view_dot_half, ocean_settings.f_0);

    // better foam algo
    let jacobian_mask = 1.0 - clamp(jacobian, 0.0, 1.0);
    let height_mask = smoothstep(-0.2, 0.6, in.height);

    let foam_uv1 = in.world_pos.xz * 0.08 + camera.time * 0.02;
    let foam_uv2 = in.world_pos.xz * 0.15 - camera.time * 0.015;
    let foam_uv3 = in.world_pos.xz * 0.25 + camera.time * 0.01;

    let foam_val1 = textureSample(foam_texture, sampler_foam, foam_uv1).r;
    let foam_val2 = textureSample(foam_texture, sampler_foam, foam_uv2).r;
    let foam_val3 = textureSample(foam_texture, sampler_foam, foam_uv3).r;
    let foam_val = (foam_val1 + foam_val2 * 0.5 + foam_val3 * 0.3) / 1.8;

    let wave_crest_mask = smoothstep(ocean_settings.foam_threshold, ocean_settings.foam_threshold + ocean_settings.foam_crest_width, jacobian_mask) * height_mask;
    let foam_factor = smoothstep(0.3, 0.8, foam_val) * wave_crest_mask * ocean_settings.foam_scale;

    let turbidity = smoothstep(0.1, 0.5, jacobian); // basically compression
    let base_mix = smoothstep(-2.0, 4.0, in.height);
    var water_base = mix(ocean_settings.deep_color.rgb, ocean_settings.shallow_color.rgb, base_mix);
    water_base = mix(water_base, ocean_settings.sss_color.rgb * 0.5, turbidity * 0.5);
    water_base *= max(intensity, 0.35);

    let reflection_dampener = 1.0 - smoothstep(ocean_settings.reflection_min, ocean_settings.reflection_max, jacobian);
    let rougness_dynamic = mix(ocean_settings.roughness, ocean_settings.foam_roughness, foam_factor);
    let alpha = rougness_dynamic * rougness_dynamic;
    let specular_val = cook_torrance(n_dot_light, n_dot_view, n_dot_half, alpha, fresnel_spec);
    let specular = light_color * specular_val * ocean_settings.specular_scale * reflection_dampener * (1.0-foam_factor);

    var color = water_base + sss;
    
    let view_depth = length(camera.camera_pos - in.world_pos);
    let caustic_depth_mask = smoothstep(2.0, 8.0, view_depth) * smoothstep(ocean_settings.caustic_max_distance, ocean_settings.caustic_max_distance * 0.5, view_depth);
    let depth_fade = smoothstep(-2.0, 2.5, in.height);
    let view_angle = dot(view_dir, vec3(0.0, 1.0, 0.0));
    let caustic_angle_mask = smoothstep(-0.3, -0.7, view_angle);
    let parallax_offset = -view_dir.xz * ocean_settings.caustic_depth / max(view_dir.y, 0.1);
    let caustic_uv = (in.world_pos.xz * ocean_settings.caustic_scale) + parallax_offset;
    let caustic_pattern = get_caustics_procedural(caustic_uv, camera.time);
    let sun_facing = pow(clamp(dot(normal, sun_dir), 0.0, 1.0), 0.7);
    let foam_mask = 1.0 - foam_factor * 0.7;
    let cam_height_fade = 1.0 - smoothstep(30.0, 200.0, cam_height);
    let dist_fade = 1.0 - smoothstep(50.0, ocean_settings.caustic_max_distance, dist) * cam_height_fade;
    let choppiness_mask = smoothstep(0.7, 1.0, jacobian);
    let down_fade = smoothstep(0.8, 0.5, view_dir.y);  // Fades out grain when looking down
    let caustic_strength = sun_facing * foam_mask * dist_fade * depth_fade * caustic_angle_mask * caustic_depth_mask * choppiness_mask * down_fade * 2.0;
    let caustic_color = caustic_pattern * ocean_settings.caustic_color_tint.rgb * caustic_strength * intensity * 2.0;

    color += caustic_color * ocean_settings.sss_color.rgb * 0.6;
    
    let sky_reflection = get_sky_color(reflect_dir) * ocean_settings.reflection_scale;
    let fresnel_clamped = clamp(fresnel_sky * 0.6, 0.0, 0.5);
    color = mix(color, sky_reflection, fresnel_clamped);
    color += ambient * ocean_settings.ambient_scale;
    color += specular;
    
    color = mix(color, ocean_settings.foam_base_color, clamp(foam_factor, 0.0, 1.0));

    color *= ocean_settings.water_brightness_mod;

    color = aces_tone_map(color);
    color = pow(color, vec3<f32>(1.0/2.2));

    return vec4<f32>(color, 1.0);
}


fn get_sky_color(view_dir: vec3<f32>) -> vec3<f32> {
    let dir = normalize(view_dir);

    let angle = (ocean_settings.daynight_cycle - 0.5) * 6.28318;
    let sun_orbit_y = cos(angle);
    let sun_dir = normalize(vec3(sin(angle), sun_orbit_y, ocean_settings.sun_offset_z));
    let sun_up = sun_orbit_y;
    let moon_base = -sun_dir;
    let moon_dir = normalize(moon_base + ocean_settings.moon_phase_offset); 
    
    let intensity = smoothstep(-0.3, 0.3, sun_up);
    let night_fade = smoothstep(0.3, -0.3, sun_up);

    let zenith = mix(ocean_settings.sky_color_night_zenith.rgb, ocean_settings.sky_color_day_zenith.rgb, intensity);
    let horizon = mix(ocean_settings.sky_color_night_horizon.rgb, ocean_settings.sky_color_day_horizon.rgb, intensity);
    var col = mix(horizon, zenith, pow(clamp(dir.y * 0.5 + 0.5, 0.0, 1.0), ocean_settings.sky_zenith_gradient_exp));

    let sunset_timing = exp(-pow(sun_up * 4.0, 2.0));
    let sunset_angle = max(dot(dir, normalize(vec3(sun_dir.x, 0.0, sun_dir.z))), 0.0);
    let sunset_vertical = smoothstep(-0.05, 0.45, dir.y) * smoothstep(0.75, 0.15, dir.y);
    
    let sunset_mix = mix(ocean_settings.sky_color_sunset_pink.rgb, ocean_settings.sky_color_sunset_orange.rgb, ocean_settings.sunset_orange_weight);
    col += sunset_mix * pow(sunset_angle, 1.1) * sunset_vertical * sunset_timing * ocean_settings.sunset_intensity;
    let scatter_tint = mix(vec3(1.0), ocean_settings.sunset_scatter_color, sunset_timing * ocean_settings.sunset_scatter_intensity * intensity);
    col *= scatter_tint;

    let sun_dist = dot(dir, sun_dir);
    let sun_disk = smoothstep(ocean_settings.sun_size_inner, ocean_settings.sun_size_outer, sun_dist);
    let sun_halo = pow(max(sun_dist, 0.0), ocean_settings.sun_halo_power) * ocean_settings.sun_halo_intensity;
    col += (sun_disk + sun_halo) * mix(ocean_settings.sun_color.rgb, ocean_settings.sky_color_sunset_orange.rgb * 1.2, sunset_timing * 0.85) * max(intensity, sunset_timing * 0.5);

    let moon_dot = dot(dir, moon_dir);
    let moon_cos_edge = cos(ocean_settings.moon_radius);
    var hit_moon = false;
    if (moon_dot > moon_cos_edge * 0.98) {
        let moon_up_ref = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(moon_dir.y) > 0.99);
        let moon_right = normalize(cross(moon_dir, moon_up_ref));
        let moon_up = cross(moon_right, moon_dir);

        let perp = dir - moon_dir * moon_dot;
        let local_u = dot(perp, moon_right) / ocean_settings.moon_radius;
        let local_v = dot(perp, moon_up) / ocean_settings.moon_radius;
        let r2 = local_u * local_u + local_v * local_v;

        let rim_aa = smoothstep(1.0, 0.92, r2);

        if (rim_aa > 0.0) {
            hit_moon = true;
            let normal_z = sqrt(max(0.0, 1.0 - r2));
            let normal = normalize(moon_right * local_u + moon_up * local_v + moon_dir * normal_z);

            let phi_m = atan2(normal.z, normal.x);
            let theta_m = asin(clamp(normal.y, -1.0, 1.0));
            let uv_moon = vec2<f32>(phi_m / (2 * pi), theta_m / pi) * ocean_settings.moon_crater_scale;
            let craters = 1.0 - moon_fbm(uv_moon) * 0.6;

            let diffuse = max(dot(normal, sun_dir), 0.0);
            let terminator = smoothstep(-0.15, 0.15, diffuse);
            let surface_col = mix(ocean_settings.moon_color_dark.rgb, ocean_settings.moon_color_lit.rgb * craters * 0.8, terminator);
            let moon_visibility = 1.0 - intensity * 0.7;
            col = mix(col, surface_col * moon_visibility, rim_aa);
        } 
    }
    
    if (!hit_moon) {
        let star_visibility = smoothstep(0.3, -0.1, sun_up); 
        let moon_dist_ang = dot(dir, moon_dir);
        let moon_dist_safe = max(moon_dist_ang, 0.0);
        let moon_glow_mask = smoothstep(0.0, 0.15, moon_dist_ang);
        let moon_glow = moon_glow_mask * pow(moon_dist_safe, ocean_settings.moon_halo_power) * ocean_settings.moon_halo_intensity * night_fade;
        col += vec3(0.5, 0.6, 0.8) * moon_glow;
        
        col += render_stars(dir, camera.time, star_visibility);

        if (ocean_settings.aurora_strength > 0.0) {
            col += get_aurora(dir, camera.time) * night_fade * ocean_settings.aurora_strength;
        }
    }

    if (dir.y > 0.0) {
        let uv = (dir.xz / (dir.y + 0.25)) * 0.6;
        let time_scale = camera.time * ocean_settings.cloud_speed;
        let d = cloud_density(uv, time_scale);
        let cloud_mask = smoothstep(ocean_settings.cloud_density_low, ocean_settings.cloud_density_high, d) * pow(dir.y, 1.5);
        let d_shadow = cloud_density(uv - sun_dir.xz * 0.05, time_scale);
        let cloud_lit = clamp(d - d_shadow, 0.0, 1.0);
        let cloud_sunset_tint = (ocean_settings.sky_color_sunset_orange.rgb + ocean_settings.sky_color_sunset_pink.rgb * 0.5) * exp(-pow(sun_up * 4.0, 2.0)) * 0.55;
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
    let eps = 0.01;
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
    uv_rgb[0] = distorted_uv + vec2(ocean_settings.caustic_aberration, 0.0);
    uv_rgb[1] = distorted_uv;
    uv_rgb[2] = distorted_uv - vec2(ocean_settings.caustic_aberration, 0.0);

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

    let rgb = vec3<f32>(smoothstep(ocean_settings.caustic_smooth_low, ocean_settings.caustic_smooth_high, value_rgb.r), smoothstep(ocean_settings.caustic_smooth_low, ocean_settings.caustic_smooth_high, value_rgb.g), smoothstep(ocean_settings.caustic_smooth_low, ocean_settings.caustic_smooth_high, value_rgb.b));
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

fn render_stars(dir: vec3<f32>, time: f32, star_visibility: f32) -> vec3<f32> {
    if (star_visibility <= 0.0) { return vec3(0.0); }

    let abs_d = abs(dir);

    var face_uv: vec2<f32>;
    var face_id: f32;
    if (abs_d.x >= abs_d.y && abs_d.x >= abs_d.z) {
        face_id = select(0.0, 1.0, dir.x > 0.0);
        face_uv = vec2(dir.z, dir.y) / abs_d.x;
    } else if (abs_d.y >= abs_d.x && abs_d.y >= abs_d.z) {
        face_id = select(2.0, 3.0, dir.y > 0.0);
        face_uv = vec2(dir.x, dir.z) / abs_d.y;
    } else {
        face_id = select(4.0, 5.0, dir.z > 0.0);
        face_uv = vec2(dir.x, dir.y) / abs_d.z;
    }
    let density = max(ocean_settings.star_count * 0.12, 1.0);
    let scaled_uv = face_uv * density;

    let star_grid = floor(scaled_uv);
    let star_frac = fract(scaled_uv) - 0.5;
    let star_hash_in = vec3(star_grid, face_id * 13.7 + 3.1);
    let star_hash = hash31(star_hash_in);

    if (star_hash <= ocean_settings.star_threshold) { return vec3(0.0); }
    let star_radius = clamp(ocean_settings.star_size / 1000.0, 0.001, 0.49);

    let star_dist = length(star_frac);
    let star_dot = 1.0 - smoothstep(0.0, star_radius, star_dist);
    if (star_dot <= 0.0) { return vec3(0.0); }

    let star_phase = hash31(star_hash_in + vec3(123.456, 789.012, 0.0));
    let star_freq = mix(0.5, 3.0, hash31(star_hash_in + vec3(11.1, 22.2, 0.0)));
    let blink = sin(time * ocean_settings.star_blink_speed * star_freq + star_phase * 6.28318) * 0.4 + 0.6;
    let star_brightness = mix(0.7, 1.4, star_phase);
    let star_color = mix(vec3(1.0, 0.9, 0.75), vec3(0.8, 0.9, 1.0), hash31(star_hash_in + vec3(55.5, 66.6, 0.0)));
    let horizon_fade = smoothstep(0.0, 0.15, dir.y);
    return star_color * star_dot * blink * star_brightness * star_visibility * horizon_fade;
}

// yep this is vibe coded..
fn get_aurora(dir: vec3<f32>, time: f32) -> vec3<f32> {
    let normalized_y = clamp(dir.y, -1.0, 1.0);
    if (normalized_y < ocean_settings.aurora_y_threshold) { return vec3(0.0); }
    
    let phi = atan2(dir.z, dir.x);
    let t = time * 0.08;

    let wave = sin(phi * 3.0 + t * 1.20) * 0.100
             + sin(phi * 5.0 - t * 0.75) * 0.050
             + sin(phi * 9.0 + t * 2.30) * 0.025;

    let band_y = normalized_y + wave * (1.0 - abs(normalized_y) * 0.5); // Reduce wave at poles
    let band = exp(-pow((band_y - 0.55) * 4.5, 2.0));

    let uv_a = vec2(sin(phi) * 0.8 * (1.0 - abs(normalized_y) * 0.3), cos(phi) * 0.8 * (1.0 - abs(normalized_y) * 0.3) + normalized_y * 2.5);
    let coarse = fbm(uv_a * vec2(2.8, 2.0) + vec2(t * 0.12, 0.0));
    let fine = fbm(uv_a * vec2(6.5, 3.5) - vec2(t * 0.22, t * 0.08));
    let horizon_fade = smoothstep(0.04, 0.28, normalized_y) * smoothstep(0.95, 0.7, normalized_y);
    let aurora_mask = band * coarse * horizon_fade;

    let aurora_green = vec3(0.05, 0.95, 0.30);
    let aurora_cyan = vec3(0.00, 0.70, 0.95);
    let aurora_purple = vec3(0.55, 0.10, 0.85);
    var aurora_col = mix(aurora_green, aurora_cyan, smoothstep(0.30, 0.60, fine));
    aurora_col = mix(aurora_col, aurora_purple, smoothstep(0.65, 0.85, fine) * 0.5);

    return aurora_col * aurora_mask * ocean_settings.aurora_brightness;
}

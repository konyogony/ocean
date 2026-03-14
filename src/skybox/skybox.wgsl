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
    time_scale: f32,
    chop_scale: f32,
    amplitude_scale: f32,
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
    wave_height_exp: f32,
    wave_height_sharp: f32,
    night_water_floor: f32,
    fresnel_sky_cap: f32,
    caustic_sss_blend: f32,
    _pad_final: array<vec4<f32>, 3>,
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


@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> ocean_settings: OceanSettingsUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec3<f32>,
};


@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj_sky * vec4<f32>(model.position, 1.0);
    out.clip_position.z = out.clip_position.w; 
    out.tex_coords = model.position;
    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.tex_coords);
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

    let horizon_wrap = pow(max(1.0 - abs(dir.y), 0.0), 3.0) * sunset_timing;
    col += ocean_settings.sky_color_horizon_glow.rgb * horizon_wrap * ocean_settings.horizon_glow_mult;

    let sunset_mix = mix(ocean_settings.sky_color_sunset_pink.rgb, ocean_settings.sky_color_sunset_orange.rgb, ocean_settings.sunset_orange_weight);
    col += sunset_mix * pow(sunset_angle, 1.1) * sunset_vertical * sunset_timing * ocean_settings.sunset_intensity;
    
    let scatter_tint = mix(vec3(1.0), ocean_settings.sunset_scatter_color, sunset_timing * ocean_settings.sunset_scatter_intensity * intensity);
    col *= scatter_tint;

    let sun_dist = dot(dir, sun_dir);
    let sun_disk = smoothstep(ocean_settings.sun_size_inner, ocean_settings.sun_size_outer, sun_dist);
    let sun_halo = pow(max(sun_dist, 0.0), ocean_settings.sun_halo_power) * ocean_settings.sun_halo_intensity;
    col += (sun_disk + sun_halo) * mix(ocean_settings.sun_color.rgb, ocean_settings.sky_color_sunset_orange.rgb * 1.25, sunset_timing * 0.85) * max(intensity, sunset_timing * 0.5);

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

    return vec4<f32>(col, 1.0);
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

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(mix(hash_v2(i + vec2(0.0, 0.0)), hash_v2(i + vec2(1.0, 0.0)), u.x),
               mix(hash_v2(i + vec2(0.0, 1.0)), hash_v2(i + vec2(1.0, 1.0)), u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
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
    let drift = vec2(t * 0.02, t * 0.01);
    let v1 = fbm(p + drift);
    let v2 = fbm(p - drift * 0.5 + v1 * 0.8);
    return v2;
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

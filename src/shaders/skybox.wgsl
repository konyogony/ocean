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
    view_dir: vec3<f32>,
    time: f32,
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

const DAY_CYCLE_SPEED: f32 = 0.01;

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
    let drift = vec2(t * 0.02, t * 0.01); // Internal drift ratios
    let v1 = fbm(p + drift);
    let v2 = fbm(p - drift * 0.5 + v1 * 0.8);
    return v2;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.tex_coords);
    // let day_cycle = fract(camera.time * DAY_CYCLE_SPEED);
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

    return vec4<f32>(col, 1.0);
}

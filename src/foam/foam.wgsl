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

@group(0) @binding(0) var<uniform> ocean_settings: OceanSettingsUniform;
@group(1) @binding(0) var<uniform> camera: CameraUniform;

@group(2) @binding(0) var foam_texture_read: texture_storage_2d<rgba16float, read>;
@group(2) @binding(1) var foam_texture_write: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var foam_sampler: sampler;

@group(3) @binding(0) var texture_packed: texture_2d<f32>;
@group(3) @binding(1) var sampler_height_field: sampler;

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
        mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

@compute @workgroup_size(16, 16, 1)
fn compute_foam(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(texture_packed);
    let x = global_id.x;
    let y = global_id.y;
    let chop = ocean_settings.chop_scale;

    // now using a global scale
    let world_step = 1000 / f32(dims.x);

    let r = (x + 1u) % dims.x;
    let l = (x + dims.x - 1u) % dims.x;
    let u = (y + 1u) % dims.y;
    let d = (y + dims.y - 1u) % dims.y;

    // now since packed -> single texture has .r as height, .g as dx and .b as dz
    let data_right = textureLoad(texture_packed, vec2(r, y), 0);
    let data_left = textureLoad(texture_packed, vec2(l, y), 0);
    let data_up = textureLoad(texture_packed, vec2(x, u), 0);
    let data_down = textureLoad(texture_packed, vec2(x, d), 0);

    let dDx_du = (data_right.g - data_left.g) * chop / (2.0 * world_step);
    let dDx_dv = (data_up.g - data_down.g) * chop / (2.0 * world_step);

    let dDz_du = (data_right.b - data_left.b) * chop / (2.0 * world_step);
    let dDz_dv = (data_up.b - data_down.b) * chop / (2.0 * world_step);

    let jacobian = (1.0 + dDx_du) * (1.0 + dDz_dv) - (dDx_dv * dDz_du);

    let breaking = clamp(ocean_settings.foam_threshold - jacobian, 0.0, 1.0);
    let generated = pow(breaking, ocean_settings.foam_power);

    let prev = textureLoad(foam_texture_read, global_id.xy).r;
    let decayed = prev * ocean_settings.decay_factor;

    let result = max(decayed, generated); 
    textureStore(foam_texture_write, global_id.xy, vec4<f32>(clamp(result, 0.0, 1.0), 0.0, 0.0, 1.0));
}

@compute @workgroup_size(16, 16, 1)
fn advect_foam(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(texture_packed);
    let dims_f = vec2<f32>(f32(dims.x), f32(dims.y));
    let id_f = vec2<f32>(f32(global_id.x), f32(global_id.y));

    let master_scale = 1000.0;
    let data = textureLoad(texture_packed, global_id.xy, 0);

    let velocity = vec2<f32>(data.g, data.b) * ocean_settings.chop_scale;
    let advect_dist_meters = velocity * ocean_settings.foam_speed * camera.delta_time;
    let advect_px = (advect_dist_meters / master_scale) * dims_f;

    var sample_pos = id_f - advect_px;
    sample_pos = ((sample_pos % dims_f) + dims_f) % dims_f;

    let f = fract(sample_pos);
    let i = vec2<u32>(vec2<i32>(sample_pos));
    let rx = (i.x + 1u) % dims.x;
    let uy = (i.y + 1u) % dims.y;

    let tl = textureLoad(foam_texture_read, vec2(i.x, i.y)).r;
    let tr = textureLoad(foam_texture_read, vec2(rx, i.y)).r;
    let bl = textureLoad(foam_texture_read, vec2(i.x, uy)).r;
    let br = textureLoad(foam_texture_read, vec2(rx, uy)).r;

    let sampled = mix(mix(tl, tr, f.x), mix(bl, br, f.x), f.y);
    let advected = sampled * ocean_settings.dissipation_factor;
    textureStore(foam_texture_write, global_id.xy, vec4<f32>(clamp(advected, 0.0, 1.0), 0.0, 0.0, 1.0));
}

fn get_foam(uv: vec2<f32>, time: f32) -> f32 {
    let warp_uv = uv * ocean_settings.warp_uv_scale;
    let warp1 = vec2<f32>(
        noise(warp_uv + vec2(time * ocean_settings.warp_time_scale, 0.0)),
        noise(warp_uv + vec2(5.2, 1.3) - vec2(0.0, time * ocean_settings.warp_time_scale))
    );
    let warp2 = vec2<f32>(
        noise(warp_uv * 1.7 + vec2(3.1, 7.4) + time * ocean_settings.warp_time_scale * 0.6),
        noise(warp_uv * 1.7 + vec2(9.8, 2.7) - time * ocean_settings.warp_time_scale * 0.6)
    );
    let p = uv + warp1 * ocean_settings.warp_strength + warp2 * ocean_settings.warp_strength * 0.4;

    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var current_p = p;
    let rot = mat2x2<f32>(0.7962, 0.6050, -0.6050, 0.7962);

    for (var i = 0u; i < ocean_settings.foam_octaves; i++) {
        value += noise(current_p) * amplitude;
        current_p = rot * current_p * 2.0;
        amplitude *= 0.5;
    }
    return 0.3 + value * 0.7;
}

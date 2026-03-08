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

@group(0) @binding(0) var<uniform> ocean_settings: OceanSettingsUniform;
@group(1) @binding(0) var<uniform> camera: CameraUniform;

@group(2) @binding(0) var foam_texture_read: texture_storage_2d<rgba16float, read>;
@group(2) @binding(1) var foam_texture_write: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var foam_sampler: sampler;

@group(3) @binding(0) var texture_h_dx: texture_2d<f32>;
@group(3) @binding(1) var texture_dz: texture_2d<f32>;
@group(3) @binding(2) var sampler_height_field: sampler;

fn get_foam(uv: vec2<f32>, time: f32) -> f32 {
    let warp_uv = uv * ocean_settings.warp_uv_scale;
    let warp = vec2<f32>(
        noise(warp_uv + vec2(time * ocean_settings.warp_time_scale, 0.0)),
        noise(warp_uv + vec2(5.2, 1.3) - vec2(0.0, time * ocean_settings.warp_time_scale))
    );

    let p = uv + warp * ocean_settings.warp_strength;

    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var current_p = p;
    
    for (var i = 0u; i < ocean_settings.foam_octaves; i++) {
        let n = noise(current_p); 
        value += n * amplitude;
        current_p *= 2.0; 
        amplitude *= 0.5;
    }
    return pow(value, ocean_settings.foam_power); 
}

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * ocean_settings.hash_scale);
    p3 += dot(p3, p3.yzx + ocean_settings.hash_dot);
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

@compute @workgroup_size(16, 16, 1)
fn compute_foam(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = vec2<f32>(textureDimensions(foam_texture_write));
    let uv_float = vec2<f32>(global_id.xy) / vec2<f32>(dimensions);

    let h_dx_val = textureLoad(texture_h_dx, global_id.xy, 0);
    let dz_val = textureLoad(texture_dz, global_id.xy, 0);

    let chop = ocean_settings.chop_scale;

    let x_displacement = length(vec2(h_dx_val.z * chop, dz_val.x * chop));
    let y_displacement = abs(h_dx_val.x * chop);
    let steepness_factor = smoothstep(ocean_settings.steepness_threshold_low, ocean_settings.steepness_threshold_high, x_displacement + y_displacement * ocean_settings.y_displacement_weight);

    let noise_foam = get_foam(uv_float * ocean_settings.foam_scale, camera.time);
    let generated_foam_val = steepness_factor * noise_foam;

    let prev_foam_val = textureLoad(foam_texture_read, global_id.xy).r;
    let new_foam_val = max(generated_foam_val, prev_foam_val * ocean_settings.decay_factor);

    textureStore(foam_texture_write, global_id.xy, vec4<f32>(new_foam_val, new_foam_val, new_foam_val, 1.0));
}

@compute @workgroup_size(16, 16, 1)
fn advect_foam(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions_storage = textureDimensions(texture_h_dx);

    let h_dx_val = textureLoad(texture_h_dx, global_id.xy, 0);
    let dz_val = textureLoad(texture_dz, global_id.xy, 0);

    let dx = h_dx_val.z;
    let dz = dz_val.x;
    
    let chop = ocean_settings.chop_scale;

    let advection_dir = vec2(dx * chop, dz * chop);
    let advection_offset_pixels = advection_dir * ocean_settings.foam_speed * camera.delta_time;

    let current_pixel_coords = vec2<f32>(global_id.xy);
    let sampled_pixel_coords_float = current_pixel_coords - advection_offset_pixels;

    let clamped_x = clamp(i32(round(sampled_pixel_coords_float.x)), 0, i32(dimensions_storage.x - 1));
    let clamped_y = clamp(i32(round(sampled_pixel_coords_float.y)), 0, i32(dimensions_storage.y - 1));
    let final_sampled_coords = vec2<u32>(u32(clamped_x), u32(clamped_y));

    let sampled_foam = textureLoad(foam_texture_read, final_sampled_coords).r;
    let advected_val = sampled_foam * ocean_settings.dissipation_factor;

    textureStore(foam_texture_write, global_id.xy, vec4<f32>(advected_val, advected_val, advected_val, 1.0));
}

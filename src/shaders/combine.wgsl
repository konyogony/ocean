
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
    cascade_data: array<vec4<f32>, 6>,
    cascade_count: u32,
    _pad_cascade: vec3<u32>
};

@group(0) @binding(0) var<uniform> ocean_settings: OceanSettingsUniform;

@group(1) @binding(0) var in_h_dx: texture_storage_2d<rgba32float, read>;
@group(1) @binding(1) var in_dz: texture_storage_2d<rgba32float, read>;

@group(1) @binding(2) var combined_read_h_dx: texture_storage_2d<rgba32float, read>;
@group(1) @binding(3) var combined_read_dz: texture_storage_2d<rgba32float, read>;

@group(1) @binding(4) var combined_write_h_dx: texture_storage_2d<rgba32float, write>;
@group(1) @binding(5) var combined_write_dz: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16, 16)
fn combine_cascades(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    let coords = vec2<i32>(x,y);

    let combined_h_dx = textureLoad(combined_read_h_dx, coords);
    let combined_dz = textureLoad(combined_read_dz, coords);

    let cascade_h_dx = textureLoad(in_h_dx, coords);
    let cascade_dz = textureLoad(in_dz, coords);

    // direct sum, no weightings
    let sum_h_dx = combined_h_dx + cascade_h_dx;
    let sum_dz = combined_dz + cascade_dz;

    textureStore(combined_write_h_dx, coords, sum_h_dx);
    textureStore(combined_write_dz, coords, sum_dz);
}

@compute @workgroup_size(16, 16)
fn clear_textures(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    let coords = vec2<i32>(x,y);

    textureStore(combined_write_h_dx, coords, vec4(0));
    textureStore(combined_write_dz, coords, vec4(0));
}

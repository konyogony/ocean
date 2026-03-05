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
    _pad_cascade: vec3<u32>,
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


const DECAY_FACTOR: f32 = 0.98;
const DISSIPATION_FACTOR: f32 = 0.99;

@group(0) @binding(0) var<uniform> ocean_settings: OceanSettingsUniform;
@group(1) @binding(0) var<uniform> camera: CameraUniform;

@group(2) @binding(0) var foam_texture_read: texture_storage_2d<rgba16float, read>;
@group(2) @binding(1) var foam_texture_write: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var foam_sampler: sampler;

@group(3) @binding(0) var texture_h_dx: texture_2d<f32>;
@group(3) @binding(1) var texture_dz: texture_2d<f32>;
@group(3) @binding(2) var sampler_height_field: sampler;

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


// Generating the foam first
@compute @workgroup_size(16, 16, 1)
fn compute_foam(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = vec2<f32>(textureDimensions(foam_texture_write));
    let uv_float = vec2<f32>(global_id.xy) / vec2<f32>(dimensions);

    let h_dx_val = textureLoad(texture_h_dx, global_id.xy, 0);
    let dz_val = textureLoad(texture_dz, global_id.xy, 0);

    let amp = ocean_settings.amplitude_scale;
    let chop = ocean_settings.chop_scale;

    let x_displacement = length(vec2(h_dx_val.z * chop, dz_val.x * chop));
    let y_displacement = abs(h_dx_val.x * chop);
    let steepness_factor = smoothstep(0.1, 0.8, x_displacement + y_displacement * 0.5);

    let noise_foam = get_foam(uv_float * ocean_settings.foam_scale, camera.time);
    let generated_foam_val = mix(noise_foam, 1.0, steepness_factor * ocean_settings.foam_threshold);

    let prev_foam_val = textureLoad(foam_texture_read, global_id.xy).r;
    let new_foam_val = max(generated_foam_val, prev_foam_val * DECAY_FACTOR);

    textureStore(foam_texture_write, global_id.xy, vec4<f32>(new_foam_val, new_foam_val, new_foam_val, 1.0));
}

// then we advect it with time
@compute @workgroup_size(16, 16, 1)
fn advect_foam(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions_storage = textureDimensions(texture_h_dx);
    let uv_float = vec2<f32>(global_id.xy) / vec2<f32>(dimensions_storage);

    let h_dx_val = textureLoad(texture_h_dx, global_id.xy, 0);
    let dz_val = textureLoad(texture_dz, global_id.xy, 0);

    let dx = h_dx_val.z;
    let dz = dz_val.x;
    
    let chop = ocean_settings.chop_scale;

    let advection_dir = vec2(dx * chop, dz * chop);
    let advection_offset_pixels = advection_dir * ocean_settings.foam_speed * camera.delta_time;

    // now we know WHERE it came from
    let current_pixel_coords = vec2<f32>(global_id.xy);
    let sampled_pixel_coords_float = current_pixel_coords - advection_offset_pixels;

    let clamped_x = clamp(i32(round(sampled_pixel_coords_float.x)), 0, i32(dimensions_storage.x - 1));
    let clamped_y = clamp(i32(round(sampled_pixel_coords_float.y)), 0, i32(dimensions_storage.y - 1));
    let final_sampled_coords = vec2<u32>(u32(clamped_x), u32(clamped_y)); // textureLoad takes u32 coords

    let sampled_foam = textureLoad(foam_texture_read, final_sampled_coords).r;
    let advected_val = sampled_foam * DISSIPATION_FACTOR;

    textureStore(foam_texture_write, global_id.xy, vec4<f32>(advected_val, advected_val, advected_val, 1.0));

}


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

@group(0) @binding(0)
var<uniform> ocean_settings: OceanSettingsUniform;

// FFT Data
struct FFTUniform {
    stage: u32,
    is_vertical: u32,
    cascade_index: u32,
    _pad: u32
}

@group(1) @binding(0) var<uniform> config: FFTUniform;
@group(1) @binding(1) var src_h_dx: texture_2d<f32>;
@group(1) @binding(2) var dst_h_dx: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var src_dz: texture_2d<f32>;
@group(1) @binding(4) var dst_dz: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var dst_packed_final: texture_storage_2d<rgba16float, write>;

// Time
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_sky: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad1: f32,
    time: f32,
    delta_time: f32,
    _padding: vec2<f32>,
};

@group(2) @binding(0)
var<uniform> camera: CameraUniform;

// Initial frequency domain
struct InitialData {
    k_vec: vec2<f32>,
    initial_frequency_domain: vec2<f32>,
    initial_frequency_domain_conjugate: vec2<f32>,
    angular_frequency:  f32,
}

@group(3) @binding(0) var<storage, read> initial_data: array<InitialData>;
@group(3) @binding(1) var<storage, read> twiddle_arr: array<vec2<f32>>;

// This shall run only once
@compute @workgroup_size(16,16)
fn update_spectrum(@builtin(global_invocation_id) id: vec3<u32>) {
    // Get the index... somehow?
    let x = id.x;
    let y = id.y;
    let n = ocean_settings.fft_subdivisions;
    let index = y * n + x;
    let cascade_idx = config.cascade_index;
    let current_amplitude = ocean_settings.cascade_data[cascade_idx].y;

    // Mirrored
    let m_x = (n - x) % n;
    let m_y = (n - y) % n;
    let mirrored_index = m_y * n + m_x;

    // Extract data using the index
    let h_0 = initial_data[index].initial_frequency_domain;
    let h_0_mirrored = initial_data[mirrored_index].initial_frequency_domain;
    let h_0_mirrored_conjugate = vec2<f32>(h_0_mirrored.x, -h_0_mirrored.y);

    let w_i = initial_data[index].angular_frequency;
    let wt = w_i * camera.time * ocean_settings.time_scale;
    let cos_wt = cos(wt);
    let sin_wt = sin(wt);

    // e^{iwt} = cos(wt) + i*sin(wt) = vec2(cos(wt),sin(wt))
    let exponent = vec2<f32>(cos_wt, sin_wt);
    let exponent_neg = vec2<f32>(cos_wt, -sin_wt);

    let h_tilda: vec2<f32> = (complex_multiplication(h_0, exponent) + complex_multiplication(h_0_mirrored_conjugate, exponent_neg)) * current_amplitude;
    // No shift needed, since we are dealing w a texture now
    // let shift = select(1.0, -1.0, ((x + y) % 2u) == 1u);
    // let h_tilda_shifted = h_tilda * shift;

    let k = initial_data[index].k_vec;
    let k_len = length(k);
    var h_dx = vec2<f32>(0.0);
    var h_dz = vec2<f32>(0.0);

    if (k_len > ocean_settings.wave_epsilon) {
        let k_norm = k / k_len;
        // i * complex is: (-imag, real)
        h_dx = vec2<f32>(-h_tilda.y * k_norm.x, h_tilda.x * k_norm.x);
        h_dz = vec2<f32>(-h_tilda.y * k_norm.y, h_tilda.x * k_norm.y);
    }

    textureStore(dst_h_dx, vec2<i32>(id.xy), vec4<f32>(h_tilda, h_dx));
    textureStore(dst_dz, vec2<i32>(id.xy), vec4<f32>(h_dz, 0.0, 0.0));
}

// This will run log2(N) * 2 times
@compute @workgroup_size(16,16)
fn fft_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let n = ocean_settings.fft_subdivisions;
    
    // To know which direction we are squishing
    let t = select(x, y, config.is_vertical == 1u);
    let other = select(y, x, config.is_vertical == 1u);

    // Size of the group we r building
    let s = 1u << config.stage;
    let group_idx = t / (2u * s);
    let butterfly_idx = t % (2u * s);
    let pair = butterfly_idx / s;
    let offset = butterfly_idx % s;

    let base_idx = group_idx * (2u * s) + offset;
    let idx0_t = base_idx;
    let idx1_t = base_idx + s;

    var i0: vec2<u32>;
    var i1: vec2<u32>;
    if (config.is_vertical == 1u) {
        i0 = vec2<u32>(other, idx0_t);
        i1 = vec2<u32>(other, idx1_t);
    } else {
        i0 = vec2<u32>(idx0_t, other);
        i1 = vec2<u32>(idx1_t, other);
    }

    let src0_h_dx = textureLoad(src_h_dx, vec2<i32>(i0), 0 );
    let src1_h_dx = textureLoad(src_h_dx, vec2<i32>(i1), 0);

    let src0_dz = textureLoad(src_dz, vec2<i32>(i0), 0);
    let src1_dz = textureLoad(src_dz, vec2<i32>(i1), 0);

    // Positive since Inverse 
    let base = (1u << config.stage) - 1u;
    let twiddle = twiddle_arr[base + offset];

    let rotated_h_dx_xy = complex_multiplication(twiddle, src1_h_dx.xy);
    let rotated_h_dx_zw = complex_multiplication(twiddle, src1_h_dx.zw);
    
    // again, repeat same for dz

    let rotated_dz_xy = complex_multiplication(twiddle, src1_dz.xy);
    var result_dz: vec4<f32>;
    if (pair == 0u) {
        result_dz = vec4<f32>(src0_dz.xy + rotated_dz_xy, 0.0, 0.0);
    } else {
        result_dz = vec4<f32>(src0_dz.xy - rotated_dz_xy, 0.0, 0.0);
    }
    
    // basically checking if last
    if (config.stage == ocean_settings.pass_num - 1u && config.is_vertical == 1u) {
        var res_h: vec2<f32>; var res_dx: vec2<f32>; var res_dz: vec2<f32>;
        
        if (pair == 0u) {
            res_h = (src0_h_dx.xy + rotated_h_dx_xy) / f32(n);
            res_dx = (src0_h_dx.zw + rotated_h_dx_zw) / f32(n);
            res_dz = (src0_dz.xy + rotated_dz_xy) / f32(n);
        } else {
            res_h = (src0_h_dx.xy - rotated_h_dx_xy) / f32(n);
            res_dx = (src0_h_dx.zw - rotated_h_dx_zw) / f32(n);
            res_dz = (src0_dz.xy - rotated_dz_xy) / f32(n);
        }

        // cramming only the REAL parts into the texture
        textureStore(dst_packed_final, vec2<i32>(id.xy), vec4<f32>(res_h.x, res_dx.x, res_dz.x, 1.0));
    } else {
        // normal ping pong
        var res_h_dx: vec4<f32>; var res_dz: vec4<f32>;
        if (pair == 0u) {
            res_h_dx = vec4<f32>(src0_h_dx.xy + rotated_h_dx_xy, src0_h_dx.zw + rotated_h_dx_zw);
            res_dz = vec4<f32>(src0_dz.xy + rotated_dz_xy, 0.0, 0.0);
        } else {
            res_h_dx = vec4<f32>(src0_h_dx.xy - rotated_h_dx_xy, src0_h_dx.zw - rotated_h_dx_zw);
            res_dz = vec4<f32>(src0_dz.xy - rotated_dz_xy, 0.0, 0.0);
        }
        textureStore(dst_h_dx, vec2<i32>(id.xy), res_h_dx);
        textureStore(dst_dz, vec2<i32>(id.xy), res_dz);
    }
}

fn complex_multiplication(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

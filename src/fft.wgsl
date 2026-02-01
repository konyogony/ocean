const g: f32 = 9.81;
const pi: f32 = 3.14159;

struct OceanSettingsUniform {
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
    caustic_scale: f32,
    caustic_size: f32,
    caustic_speed: f32,
    caustic_intensity: f32,
    caustic_octaves: u32,
    caustic_depth: f32,
    caustic_max_distance: f32,
    micro_normal_strength: f32,
    foam_threshold: f32,
    foam_speed: f32,
    foam_roughness: f32,
    caustic_color_tint: vec4<f32>,
    deep_color: vec4<f32>,
    shallow_color: vec4<f32>,
    sss_color: vec4<f32>,
    sun_color: vec4<f32>
}

@group(0) @binding(0)
var<uniform> ocean_settings: OceanSettingsUniform;

// FFT Data
struct FFTUniform {
    stage: u32,
    is_vertical: u32,
}

@group(1) @binding(0) var<uniform> config: FFTUniform;
@group(1) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> dst: array<vec4<f32>>; // THis will hold height and dx
@group(1) @binding(3) var<storage, read> src_dz: array<vec4<f32>>; // While this one only dz
@group(1) @binding(4) var<storage, read_write> dst_dz: array<vec4<f32>>;

// Time
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_sky: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
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

    // Mirrored
    let m_x = (n - x) % n;
    let m_y = (n - y) % n;
    let mirrored_index = m_y * n + m_x;

    // Check if we are at the origin
    // if (id.x == 0u && id.y == 0u) {
    //     dst[index] = vec2<f32>(0.0, 0.0);
    //     return;
    // }

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

    let h_tilda: vec2<f32> = complex_multiplication(h_0, exponent) + complex_multiplication(h_0_mirrored_conjugate, exponent_neg);
    let shift = select(1.0, -1.0, ((x + y) % 2u) == 1u);
    let h_tilda_shifted = h_tilda * shift;

    let k = initial_data[index].k_vec;
    let k_len = length(k);
    var h_dx = vec2<f32>(0.0);
    var h_dz = vec2<f32>(0.0);

    if (k_len > 0.0001) {
        let k_norm = k / k_len;
        // i * complex is: (-imag, real)
        h_dx = vec2<f32>(-h_tilda_shifted.y * k_norm.x, h_tilda_shifted.x * k_norm.x);
        h_dz = vec2<f32>(-h_tilda_shifted.y * k_norm.y, h_tilda_shifted.x * k_norm.y);
    }
    dst[index] = vec4<f32>(h_tilda_shifted, h_dx);
    dst_dz[index] = vec4<f32>(h_dz, 0.0, 0.0); // We have space for something else for the future
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

    let src0 = src[i0.y * n + i0.x];
    let src1 = src[i1.y * n + i1.x];

    // Positive since Inverse 
    let base = (1u << config.stage) - 1u;
    let twiddle = twiddle_arr[base + offset];

    let rotated_src1_xy = complex_multiplication(twiddle, src1.xy);
    let rotated_src1_zw = complex_multiplication(twiddle, src1.zw);
    
    var result: vec4<f32>;
    if (pair == 0u) {
        result = vec4<f32>(src0.xy + rotated_src1_xy, src0.zw + rotated_src1_zw);
    } else {
        result = vec4<f32>(src0.xy - rotated_src1_xy, src0.zw - rotated_src1_zw);
    }
    
    if (config.stage == ocean_settings.pass_num - 1u) {
        result = result / f32(n);
    }

    var write_idx: vec2<u32>;
    if (config.is_vertical == 1u) {
        write_idx = vec2<u32>(other, t);
    } else {
        write_idx = vec2<u32>(t, other);
    }
    dst[write_idx.y * n + write_idx.x] = result;
    
    // again, repeat same for dz
    let src0_dz = src_dz[i0.y * n + i0.x];
    let src1_dz = src_dz[i1.y * n + i1.x];

    let rotated_src1_xy_dz = complex_multiplication(twiddle, src1_dz.xy);
    let rotated_src1_zw_dz = complex_multiplication(twiddle, src1_dz.zw);

    var result_dz: vec4<f32>;
    if (pair == 0u) {
        result_dz = vec4<f32>(src0_dz.xy + rotated_src1_xy_dz, src0_dz.zw + rotated_src1_zw_dz);
    } else {
        result_dz = vec4<f32>(src0_dz.xy - rotated_src1_xy_dz, src0_dz.zw - rotated_src1_zw_dz);
    }
    
    if (config.stage == ocean_settings.pass_num - 1u) {
        result_dz = result_dz / f32(n);
    }

    dst_dz[y * n + x] = result_dz;
}

fn complex_multiplication(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

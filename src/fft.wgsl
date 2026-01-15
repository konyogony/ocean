const g: f32 = 9.81;
const pi: f32 = 3.14159;
const MESH_SIZE: f32 = 1024.0;
const MESH_SUBDIVISIONS: u32 = 2048u;
const PASS_NUM: u32 = 11u;


// FFT Data
struct FFTUniform {
    stage: u32,
    is_vertical: u32,
}

@group(0) @binding(0) var<uniform> config: FFTUniform;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;

// Time
struct TimeUniform {
    time_uniform: f32,
}

@group(1) @binding(0)
var<uniform> time: TimeUniform;


// Initial frequency domain
struct InitialData {
    initial_frequency_domain: vec2<f32>,
    initial_frequency_domain_conjugate: vec2<f32>,
    k_vec: vec2<f32>,
    angular_frequency: f32,
}

@group(2) @binding(0)
var<storage, read> initial_data: array<InitialData>;

// This shall run only once
@compute @workgroup_size(16,16)
fn update_spectrum(@builtin(global_invocation_id) id: vec3<u32>) {
    // Get the index... somehow?
    let index = id.y * MESH_SUBDIVISIONS + id.x;

    // Check if we are at the origin
    if (id.x == 0u && id.y == 0u) {
        dst[index] = vec2<f32>(0.0, 0.0);
        return;
    }

    // Extract data using the index
    let h_0 = initial_data[index].initial_frequency_domain;
    let h_0_star = initial_data[index].initial_frequency_domain_conjugate;
    let w_i = initial_data[index].angular_frequency;
    let k = initial_data[index].k_vec;

    // First we have to evolve the spectrum in time.
    // Compute the complex conjugate, which is a-ib
    // let h_0_star = vec2<f32>(h_0.x, -h_0.y);
    // I made this ^ mistake last time. I thought I could calculate it real-time. But we need -k as param, not k.
    // Now, we are computing that on CPU + using different rand variables and then passing it on.

    // We have to use eulers formula for complex numbers,
    // e^{iwt} = cos(wt) + i*sin(wt) = vec2(cos(wt),sin(wt))
    let wt = w_i * time.time_uniform;
    let exponent = vec2<f32>(cos(wt), sin(wt));
    let exponent_neg = vec2<f32>(exponent.x, -exponent.y);

    let h_tilda: vec2<f32> = complex_multiplication(h_0, exponent) + complex_multiplication(h_0_star, exponent_neg);

    // We cant use normalize(), since it would explode at (0, 0)
    let k_len = length(k);
    let k_norm = k / max(k_len, 1e-9);

    // Multiplying by neg i results in rotation -90 deg
    // Since -i(a+bi) = (b-ai)
    let h_tilda_rotated = vec2<f32>(h_tilda.y, -h_tilda.x);

    // I forgot D_tilda constists of 2 complex numbers 
    let Dx_tilda = k_norm.x * h_tilda_rotated;
    let Dz_tilda = k_norm.y * h_tilda_rotated;

    dst[index] = h_tilda;
}

// This will run log2(N) * 2 times
@compute @workgroup_size(16,16)
fn fft_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let n = MESH_SUBDIVISIONS;
    
    // To know which direction we are squishing
    let t = select(x, y, config.is_vertical == 1u);
    let other = select(y, x, config.is_vertical == 1u);

    // Size of the group we r building
    let s = 1u << config.stage;

    let j = t % s;
    let k = t / (s * 2u);

    let idx0_t = k * s + j;
    let idx1_t = idx0_t + (n / 2u);
    
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
    let angle = pi * f32(j) / f32(s); 
    let twiddle = vec2<f32>(cos(angle), sin(angle));

    let rotated_src1 = complex_multiplication(twiddle, src1);
    
    let is_second_half = (t / s) % 2u;
    var result: vec2<f32>;
    
    if (is_second_half == 0u) {
        result = src0 + rotated_src1;
    } else {
        result = src0 - rotated_src1;
    }

    // Normalization
    if (config.stage == PASS_NUM - 1u) {
        result = result / f32(n);
    }

    dst[y * n + x] = result;
}

fn complex_multiplication(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

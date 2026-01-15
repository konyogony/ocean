// just some random constants

const g: f32 = 9.81;
const pi: f32 = 3.14159;
const MESH_SIZE: f32 = 1024.0;
const MESH_SUBDIVISIONS: u32 = 2048u;

// Camera

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_proj_sky: mat4x4<f32>,
    camera_pos: vec3<f32>,
};

@group(1) @binding(0)
var<storage, read> height_field: array<vec2<f32>>;

@group(0) @binding(0)
var<uniform> camera: CameraUniform;


struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    // Quite redundant now, but too lazy to remove
    @location(2) index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,

    @location(3) height: f32,  // Pass height to fragment shader
};

// Okay, I have to sum all of the waves together, not have 1 vertex = 1 wave
@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    let row = model.index / (MESH_SUBDIVISIONS);
    let col = model.index % (MESH_SUBDIVISIONS);
    
    // Clamp to valid FFT buffer indices
    let fft_row = min(row, MESH_SUBDIVISIONS - 1u);
    let fft_col = min(col, MESH_SUBDIVISIONS - 1u);
    let fft_index = fft_row * MESH_SUBDIVISIONS + fft_col;
    
    // Read height from FFT buffer
    let h_complex = height_field[fft_index];
    let height = h_complex.x * 10; 
    
    // Apply height displacement
    let displaced_pos = vec3<f32>(model.position.x, height, model.position.z);
    
    // Compute normals from height field gradients
    var height_right = height;
    var height_left = height;
    var height_up = height;
    var height_down = height;
    
    if (fft_col < MESH_SUBDIVISIONS - 1u) {
        height_right = height_field[fft_row * MESH_SUBDIVISIONS + (fft_col + 1u)].x;
    }
    if (fft_col > 0u) {
        height_left = height_field[fft_row * MESH_SUBDIVISIONS + (fft_col - 1u)].x;
    }
    if (fft_row < MESH_SUBDIVISIONS - 1u) {
        height_up = height_field[(fft_row + 1u) * MESH_SUBDIVISIONS + fft_col].x;
    }
    if (fft_row > 0u) {
        height_down = height_field[(fft_row - 1u) * MESH_SUBDIVISIONS + fft_col].x;
    }
    
    // Compute gradients (scale by grid spacing)
    let grid_spacing = MESH_SIZE / f32(MESH_SUBDIVISIONS);
    let dx = (height_right - height_left) / (2.0 * grid_spacing);
    let dz = (height_up - height_down) / (2.0 * grid_spacing);
    
    // Normal = (-dh/dx, 1, -dh/dz) normalized
    let normal = normalize(vec3<f32>(-dx, 1.0, -dz));
    
    out.normal = normal;
    out.tex_coords = model.tex_coords;
    out.world_pos = displaced_pos;
    out.clip_position = camera.view_proj * vec4<f32>(displaced_pos, 1.0);
    
    return out;
}

// Time to make Blinn-Phong lighting!!!
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_color = vec3<f32>(0.02, 0.15, 0.22); // deep ocean green blue

    let normal = normalize(in.normal);

    // ambient lighting, basically constant light source, provides basic shape details
    let ambient = vec3<f32>(0.75, 0.85, 1.0) * 0.3;

    // diffuse lighting, dependant on the normal & the light source.
    let light_source_color = vec3<f32>(1.0, 0.95, 0.85);
    // direction TO the light source (however, sometimes may be more benefitial to consider the other way)
    // Bearing 324.6deg, pitch +28.13deg, used chatgpt to get better direction vector
    let light_source_dir = normalize(vec3<f32>(-0.511, 0.472, -0.718));
    let diffuse_strength = clamp(dot(light_source_dir, normal), 0.0, 1.0);
    let diffuse = diffuse_strength * light_source_color;

    // specular lighting, needs normal, ligth source, view direction
    let view_dir = normalize(camera.camera_pos - in.world_pos);
    let reflect_dir = normalize(reflect(-light_source_dir, normal));
    var specular_strength = max(dot(view_dir, reflect_dir), 0.0);
    specular_strength = pow(specular_strength, 128.0); // Higher value, bigger glossynessa (idk how to spell)
    let specular = specular_strength * light_source_color;

    // Phong = Ambient + Diffuse + Specular
    // For taking pictures, make separate screenshots with ambient, diffuse & specular turned down to 0
    let final_color = (base_color * (ambient * 0.3 + diffuse * 0.7)) + (specular * 0.5);
    return vec4<f32>(final_color, 1.0);
}

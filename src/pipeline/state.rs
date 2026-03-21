use anyhow::Result;
use cgmath::{Deg, Zero};
use rand::Rng;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use sysinfo::System;
use wgpu::{util::DeviceExt, Color};
use wgpu_text::glyph_brush::ab_glyph::FontArc;
use wgpu_text::glyph_brush::{BuiltInLineBreaker, HorizontalAlign, VerticalAlign};
use winit::event::ElementState;
use winit::window::Window;

use crate::camera::controller::CameraController;
use crate::camera::instance::CameraInstance;
use crate::camera::uniform::CameraUniform;
use crate::fft::cascade::InitialData;
use crate::settings::builder::OceanSettingsBuilder;
use crate::settings::presets::OceanPreset;
use crate::settings::uniform::OceanSettingsUniform;
use crate::skybox::skybox::Skybox;
use crate::texture::instance::{TextureInstance, DEPTH_FORMAT, FFT_TEXTURE_FORMAT};
use crate::vertex::vertex::Vertex;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FFTUniform {
    pub stage: u32,
    pub is_vertical: u32,
    pub cascade_index: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CombineUniform {
    pub cascade_index: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

pub struct CascadeResources {
    pub size: f32,
    pub texture_ping_h_dx: TextureInstance,
    pub texture_pong_h_dx: TextureInstance,
    pub texture_ping_dz: TextureInstance,
    pub texture_pong_dz: TextureInstance,
    pub texture_packed: TextureInstance,
    pub bind_groups_ping: Vec<wgpu::BindGroup>,
    pub bind_groups_pong: Vec<wgpu::BindGroup>,
    pub height_field_render_bind_group_ping: wgpu::BindGroup,
    pub height_field_render_bind_group_pong: wgpu::BindGroup,
    pub height_field_compute_bind_group_ping: wgpu::BindGroup,
    pub height_field_compute_bind_group_pong: wgpu::BindGroup,
    pub initial_data_group: wgpu::BindGroup,
    pub initial_data_buffer: wgpu::Buffer,
    pub config_buffer: wgpu::Buffer,
    pub output_is_ping: bool,
    pub combine_uniform_buffer: wgpu::Buffer,
    pub cascade_input_bind_group: wgpu::BindGroup,
}

pub struct State {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub is_surface_configured: bool,

    pub render_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub index_buffer: wgpu::Buffer,
    pub camera: CameraInstance,
    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_controller: CameraController,

    pub text_brush: wgpu_text::TextBrush,

    pub depth_texture: TextureInstance,
    pub skybox: Skybox,

    pub last_frame_time_instant: Instant,
    pub fps: f32,
    pub frame_counter: u32,
    pub fps_timer: f32,
    pub gpu_name: String,
    pub cpu_name: String,
    pub kernel_version: String,
    pub os_name: String,
    pub sys: sysinfo::System,
    pub gpu: Box<dyn gfxinfo::Gpu>,
    pub gpu_vram_total: u64,
    pub gpu_vram_used: u64,
    pub gpu_load: f32,
    pub gpu_temp: f32,

    pub fft_compute_pipeline: wgpu::ComputePipeline,
    pub spectrum_pipeline: wgpu::ComputePipeline,

    pub foam_generation_pipeline: wgpu::ComputePipeline,
    pub foam_advection_pipeline: wgpu::ComputePipeline,
    pub foam_texture_ping: TextureInstance,
    pub foam_texture_pong: TextureInstance,
    pub foam_compute_bind_groups: [wgpu::BindGroup; 2],
    pub foam_render_bind_groups: [wgpu::BindGroup; 2],
    pub foam_compute_layout: wgpu::BindGroupLayout,
    pub foam_render_layout: wgpu::BindGroupLayout,
    pub foam_output_is_a: bool,

    pub cascades: Vec<CascadeResources>,
    pub combined_render_bind_group_ping: wgpu::BindGroup,
    pub combined_render_bind_group_pong: wgpu::BindGroup,
    pub combined_cascade_pipeline: wgpu::ComputePipeline,
    pub combined_clear_pipeline: wgpu::ComputePipeline,
    pub combined_bind_group_layout: wgpu::BindGroupLayout,
    pub cascade_input_bind_group_layout: wgpu::BindGroupLayout,
    pub combined_read_write_bind_group_ping: wgpu::BindGroup,
    pub combined_read_write_bind_group_pong: wgpu::BindGroup,

    pub combined_texture_ping: TextureInstance,
    pub combined_texture_pong: TextureInstance,
    pub combined_output_is_ping: bool,

    pub height_field_compute_bind_group_layout: wgpu::BindGroupLayout,
    pub height_field_render_bind_group_layout: wgpu::BindGroupLayout,
    pub initial_data_group_layout: wgpu::BindGroupLayout,
    pub fft_render_group_layout: wgpu::BindGroupLayout,
    pub fft_uniform_buffer: wgpu::Buffer,
    pub fft_uniform_size: u64,
    pub step_size: u64,

    pub current_ocean_preset: OceanPreset,
    pub available_presets: Vec<String>,
    pub preset_modified: bool,
    pub preset_name_to_load: String,
    pub preset_name_to_create: String,
    pub preset_author_to_create: String,
    pub preset_description_to_create: String,
    pub ocean_settings_uniform: OceanSettingsUniform,
    pub ocean_settings_buffer: wgpu::Buffer,
    pub ocean_settings_bind_group: wgpu::BindGroup,
    pub ocean_settings_bind_group_layout: wgpu::BindGroupLayout,

    pub egui_state: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,
    pub show_setting_ui: bool,
    pub show_debug_text: bool,
    pub draft_settings: OceanSettingsUniform,
    pub settings_changed: bool,

    pub window: Arc<Window>,
}

impl State {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: (wgpu::Backends::PRIMARY),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptionsBase {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::wgt::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|format| format.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let ocean_seed = rand::rng().random::<u32>();
        let current_ocean_preset = OceanPreset::load_preset("default", Path::new("presets/"));
        let ocean_settings_uniform = OceanSettingsBuilder::from_preset(&current_ocean_preset)
            .ocean_seed(ocean_seed)
            .build();
        let available_presets = OceanPreset::get_preset_list(Path::new("presets/"))?;

        let ocean_settings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ocean_settings_buffer"),
            contents: bytemuck::cast_slice(&[ocean_settings_uniform]),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let ocean_settings_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ocean_settings_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let ocean_settings_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ocean_settings_bind_group"),
            layout: &ocean_settings_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ocean_settings_buffer.as_entire_binding(),
            }],
        });

        let depth_texture =
            TextureInstance::create_depth_texture(&device, &surface_config, "depth_texture");

        let mut camera = CameraInstance {
            forward: cgmath::Vector3::zero(),
            eye: (0.0, 10.0, 0.0).into(),
            yaw: Deg(-90.0).into(),
            pitch: Deg(-20.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: surface_config.width as f32 / surface_config.height as f32,
            fovy: ocean_settings_uniform.fovy,
            znear: 0.1,
            zfar: ocean_settings_uniform.zfar,
            flip_y: false,
            bearing: Deg(0.0).into(),
        };

        let mut camera_uniform = CameraUniform::default();
        camera_uniform.update_view_proj(&mut camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let camera_controller = CameraController::new(
            ocean_settings_uniform.cam_speed,
            ocean_settings_uniform.cam_sensitivity,
            ocean_settings_uniform.cam_boost,
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./render.wgsl").into()),
        });

        let (verticies, indicies) = Vertex::generate_plane(
            &ocean_settings_uniform.mesh_size,
            ocean_settings_uniform.mesh_subdivisions,
        );

        let twiddle_factor_array =
            InitialData::generate_twiddle_factors(ocean_settings_uniform.fft_subdivisions);

        let twiddle_factor_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fft_twiddle_buffer"),
            contents: bytemuck::cast_slice(&twiddle_factor_array),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let initial_data_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("initial_data_group_layout"),
            });

        let fft_uniform_size = std::mem::size_of::<FFTUniform>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let step_size = (fft_uniform_size + alignment - 1) & !(alignment - 1);
        let total_size = step_size * ocean_settings_uniform.pass_num as u64 * 2;

        let fft_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_uniform_buffer"),
            size: std::mem::size_of::<FFTUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let fft_render_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fft_render_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // read are now as normal textures
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: FFT_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: FFT_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: FFT_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let height_field_render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("height_field_render_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT
                            | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT
                            | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let height_field_compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("height_field_compute_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: FFT_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: FFT_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let combined_texture_ping = TextureInstance::create_texture(
            &device,
            ocean_settings_uniform.fft_subdivisions,
            "combined_ping",
        );
        let combined_texture_pong = TextureInstance::create_storage_texture(
            &device,
            ocean_settings_uniform.fft_subdivisions,
            "combined_pong",
        );

        let combined_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("combined_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: FFT_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let cascade_input_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cascade_input_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let combined_read_write_bind_group_ping =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("combined_read_write_bind_group_ping"),
                layout: &combined_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&combined_texture_ping.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&combined_texture_pong.view),
                    },
                ],
            });

        let combined_read_write_bind_group_pong =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("combined_read_write_bind_group_pong"),
                layout: &combined_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&combined_texture_pong.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&combined_texture_ping.view),
                    },
                ],
            });

        let mut cascades = Vec::new();
        for cascade_index in 0..(ocean_settings_uniform.cascade_count as usize) {
            let initial_data_array = InitialData::generate_data(
                ocean_settings_uniform.cascade_data[cascade_index][0],
                ocean_settings_uniform.fft_subdivisions,
                ocean_settings_uniform.wind_vector,
                ocean_settings_uniform.l_small,
                ocean_settings_uniform.cascade_data[cascade_index][1],
                ocean_settings_uniform.max_w,
                ocean_settings_uniform.ocean_seed,
            );

            let initial_data_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("initial_data_buffer_{cascade_index}")),
                    contents: bytemuck::cast_slice(&initial_data_array),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                });

            let initial_data_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &initial_data_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: initial_data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: twiddle_factor_buffer.as_entire_binding(),
                    },
                ],
                label: Some(&format!("initial_data_group_{cascade_index}")),
            });

            let texture_ping_h_dx = TextureInstance::create_storage_texture(
                &device,
                ocean_settings_uniform.fft_subdivisions,
                &format!("fft_texture_ping_h_dx_{cascade_index}"),
            );
            let texture_pong_h_dx = TextureInstance::create_storage_texture(
                &device,
                ocean_settings_uniform.fft_subdivisions,
                &format!("fft_texture_pong_h_dx_{cascade_index}"),
            );
            let texture_ping_dz = TextureInstance::create_storage_texture(
                &device,
                ocean_settings_uniform.fft_subdivisions,
                &format!("fft_texture_ping_dz_{cascade_index}"),
            );
            let texture_pong_dz = TextureInstance::create_storage_texture(
                &device,
                ocean_settings_uniform.fft_subdivisions,
                &format!("fft_texture_pong_dz_{cascade_index}"),
            );

            let packed_output = TextureInstance::create_storage_texture(
                &device,
                ocean_settings_uniform.fft_subdivisions,
                &format!("fft_packed_output_{cascade_index}"),
            );

            let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fft_config_buffer"),
                size: total_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            for is_vertical in 0..2 {
                for stage in 0..ocean_settings_uniform.pass_num {
                    let i = stage * 2 + is_vertical;
                    let uniform = FFTUniform {
                        stage,
                        is_vertical,
                        cascade_index: cascade_index as u32,
                        _pad: 0,
                    };
                    let offset = i as u64 * step_size;
                    queue.write_buffer(&config_buffer, offset, bytemuck::cast_slice(&[uniform]));
                }
            }

            let (bind_groups_ping, bind_groups_pong) = Self::populate_fft_bind_groups(
                &device,
                &ocean_settings_uniform,
                &fft_render_group_layout,
                &config_buffer,
                &texture_ping_h_dx,
                &texture_pong_h_dx,
                &texture_ping_dz,
                &texture_pong_dz,
                &packed_output,
                fft_uniform_size,
                step_size,
            );

            let height_field_render_bind_group_ping =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("height_field_bind_group_ping_{cascade_index}")),
                    layout: &height_field_render_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&texture_ping_h_dx.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&texture_ping_h_dx.sampler),
                        },
                    ],
                });

            let height_field_render_bind_group_pong =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("height_field_bind_group_pong_{cascade_index}")),
                    layout: &height_field_render_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&texture_pong_h_dx.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&texture_pong_h_dx.sampler),
                        },
                    ],
                });

            let height_field_compute_bind_group_ping =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!(
                        "height_field_compute_bind_group_ping_{cascade_index}"
                    )),
                    layout: &height_field_compute_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&texture_ping_h_dx.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&texture_ping_dz.view),
                        },
                    ],
                });

            let height_field_compute_bind_group_pong =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!(
                        "height_field_compute_bind_group_pong_{cascade_index}"
                    )),
                    layout: &height_field_compute_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&texture_pong_h_dx.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&texture_pong_dz.view),
                        },
                    ],
                });

            let combine_uniform_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("combine_uniform_buffer_{cascade_index}")),
                    contents: bytemuck::cast_slice(&[CombineUniform {
                        cascade_index: cascade_index as u32,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let cascade_input_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("cascade_input_bind_group_{cascade_index}")),
                layout: &cascade_input_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: combine_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&packed_output.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&packed_output.sampler),
                    },
                ],
            });

            let cascade = CascadeResources {
                config_buffer,
                size: ocean_settings_uniform.cascade_data[cascade_index][0],
                output_is_ping: false,
                bind_groups_ping,
                bind_groups_pong,
                height_field_render_bind_group_ping,
                height_field_render_bind_group_pong,
                height_field_compute_bind_group_ping,
                height_field_compute_bind_group_pong,
                texture_ping_h_dx,
                texture_pong_h_dx,
                texture_ping_dz,
                texture_pong_dz,
                initial_data_group,
                texture_packed: packed_output,
                initial_data_buffer,
                combine_uniform_buffer,
                cascade_input_bind_group,
            };

            cascades.push(cascade);
        }

        let (
            foam_texture_ping,
            foam_texture_pong,
            foam_generation_pipeline,
            foam_advection_pipeline,
            foam_compute_layout,
            foam_render_layout,
            foam_compute_bind_groups,
            foam_render_bind_groups,
        ) = Self::init_foam(
            &device,
            &ocean_settings_uniform,
            &ocean_settings_bind_group_layout,
            &height_field_render_bind_group_layout,
            &height_field_compute_bind_group_layout,
            &camera_bind_group_layout,
        );

        let fft_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fft_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../fft/compute.wgsl").into()),
        });

        let fft_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fft_pipeline_layout"),
            bind_group_layouts: &[
                &ocean_settings_bind_group_layout,
                &fft_render_group_layout,
                &camera_bind_group_layout,
                &initial_data_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let fft_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fft_compute_layout"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_shader,
                entry_point: Some("fft_step"),
                compilation_options: Default::default(),
                cache: None,
            });

        let spectrum_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("spectrum_pipeline_layout"),
                bind_group_layouts: &[
                    &ocean_settings_bind_group_layout,
                    &fft_render_group_layout,
                    &camera_bind_group_layout,
                    &initial_data_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let spectrum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("spectrum_update_pipeline"),
            layout: Some(&spectrum_pipeline_layout),
            module: &fft_shader,
            entry_point: Some("update_spectrum"),
            compilation_options: Default::default(),
            cache: None,
        });

        let combined_render_bind_group_ping =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("combined_render_bind_group_ping"),
                layout: &height_field_render_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&combined_texture_ping.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&combined_texture_ping.sampler),
                    },
                ],
            });

        // mirror for pong
        let combined_render_bind_group_pong =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("combined_render_bind_group_pong"),
                layout: &height_field_render_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&combined_texture_pong.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&combined_texture_pong.sampler),
                    },
                ],
            });

        let combine_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("combine_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../fft/cascade.wgsl").into()),
        });

        let combined_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("combined_pipeline_layout"),
                bind_group_layouts: &[
                    &ocean_settings_bind_group_layout,
                    &combined_bind_group_layout,
                    &cascade_input_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let combined_cascade_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("combined_cascade_pipeline"),
                layout: Some(&combined_pipeline_layout),
                module: &combine_shader,
                entry_point: Some("combine_cascades"),
                compilation_options: Default::default(),
                cache: None,
            });

        let combined_clear_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("combined_clear_pipeline"),
                layout: Some(&combined_pipeline_layout),
                module: &combine_shader,
                entry_point: Some("clear_textures"),
                compilation_options: Default::default(),
                cache: None,
            });

        let skybox = Skybox::new(
            &device,
            &surface_config,
            &camera_bind_group_layout,
            &ocean_settings_bind_group_layout,
        )?;

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render_pipeline_layout"),
                bind_group_layouts: &[
                    &ocean_settings_bind_group_layout,
                    &camera_bind_group_layout,
                    &height_field_render_bind_group_layout,
                    &foam_render_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Vertex::desc()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(&verticies),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let num_indices = indicies.len() as u32;

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(&indicies),
            usage: wgpu::BufferUsages::INDEX,
        });

        let depth_stencil = Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let font_bytes = include_bytes!("../../static/JetBrainsMono.ttf");
        let font = FontArc::try_from_slice(font_bytes).unwrap();
        let text_brush = wgpu_text::BrushBuilder::using_font(font)
            .with_depth_stencil(depth_stencil)
            .build(
                &device,
                surface_config.width,
                surface_config.height,
                surface_config.format,
            );

        let egui_context = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_context,
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_config.format,
            egui_wgpu::RendererOptions {
                depth_stencil_format: Some(DEPTH_FORMAT),
                msaa_samples: 1,
                dithering: true,
                predictable_texture_filtering: true,
            },
        );

        let mut sys = System::new_all();
        sys.refresh_all();
        let cpu_name = sys
            .cpus()
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or("Unknown CPU".into());
        let os_name = sysinfo::System::name().unwrap_or(String::from("Unknown OS"));
        let kernel_version =
            sysinfo::System::kernel_version().unwrap_or(String::from("Kernel Version"));
        let adapter_info = adapter.get_info();
        let gpu_name = adapter_info.name.clone();
        let gpu = gfxinfo::active_gpu().expect("Failed to get GPU info");
        let gpu_info = gpu.info();
        let gpu_vram_total = gpu_info.total_vram();
        let gpu_vram_used = gpu_info.used_vram();
        let gpu_load = gpu_info.load_pct() as f32;
        let gpu_temp = gpu_info.temperature() as f32 / 1000.0;

        const _: () = assert!(
            std::mem::size_of::<OceanSettingsUniform>() == 880,
            "OceanSettingsUniform size mismatch! Update _pad_final."
        );

        Ok(Self {
            window,
            device,
            queue,
            surface,
            surface_config,
            render_pipeline,
            vertex_buffer,
            num_indices,
            index_buffer,
            is_surface_configured: false,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            depth_texture,
            skybox,
            fps: 0.0,
            last_frame_time_instant: Instant::now(),
            fps_timer: 0.0,
            frame_counter: 0,
            text_brush,
            fft_uniform_buffer,
            fft_render_group_layout,
            height_field_render_bind_group_layout,
            step_size,
            cascades,
            combined_bind_group_layout,
            cascade_input_bind_group_layout,
            combined_cascade_pipeline,
            combined_output_is_ping: false,
            combined_clear_pipeline,
            combined_texture_ping,
            combined_texture_pong,
            combined_render_bind_group_ping,
            combined_render_bind_group_pong,
            combined_read_write_bind_group_ping,
            combined_read_write_bind_group_pong,
            fft_compute_pipeline,
            foam_render_layout,
            foam_compute_layout,
            foam_advection_pipeline,
            foam_compute_bind_groups,
            foam_output_is_a: true,
            foam_texture_ping,
            foam_texture_pong,
            foam_generation_pipeline,
            foam_render_bind_groups,
            fft_uniform_size,
            spectrum_pipeline,
            height_field_compute_bind_group_layout,
            initial_data_group_layout,
            gpu_name,
            cpu_name,
            kernel_version,
            os_name,
            sys,
            current_ocean_preset,
            available_presets,
            preset_name_to_load: String::from(""),
            preset_name_to_create: String::from(""),
            preset_author_to_create: String::from(""),
            preset_description_to_create: String::from(""),
            ocean_settings_uniform,
            ocean_settings_buffer,
            ocean_settings_bind_group,
            ocean_settings_bind_group_layout,
            camera_bind_group_layout,
            egui_state,
            egui_renderer,
            show_setting_ui: false,
            show_debug_text: true,
            draft_settings: ocean_settings_uniform,
            preset_modified: false,
            settings_changed: false,
            gpu,
            gpu_load,
            gpu_temp,
            gpu_vram_used,
            gpu_vram_total,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.camera.aspect = width as f32 / height as f32;
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);
            self.depth_texture = TextureInstance::create_depth_texture(
                &self.device,
                &self.surface_config,
                "depth_texture",
            );
            self.is_surface_configured = true;
            self.text_brush
                .resize_view(width as f32, height as f32, &self.queue);
        }
    }

    pub fn handle_window_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        if self.show_setting_ui {
            let egui_response = self.egui_state.on_window_event(&self.window, event);
            if egui_response.consumed {
                return true;
            }
        }
        let window_event = event;

        match window_event {
            winit::event::WindowEvent::KeyboardInput { event, .. } => match event.physical_key {
                winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::F4)
                    if event.state == ElementState::Pressed =>
                {
                    self.show_setting_ui = !self.show_setting_ui;
                    self.update_cursor_mode();
                    true
                }
                winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::F3)
                    if event.state == ElementState::Pressed =>
                {
                    self.show_debug_text = !self.show_debug_text;
                    true
                }
                _ => {
                    if !self.show_setting_ui {
                        self.camera_controller.process_window_events(window_event)
                    } else {
                        false
                    }
                }
            },
            _ => false,
        }
    }

    pub fn handle_device_event(&mut self, event: &winit::event::DeviceEvent) -> bool {
        if self.show_setting_ui {
            return false;
        }
        self.camera_controller.process_device_events(event)
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame_time_instant).as_secs_f32();
        self.last_frame_time_instant = now;
        self.frame_counter += 1;
        self.fps_timer += dt;
        if self.fps_timer >= 1.0 {
            self.fps = self.frame_counter as f32 / self.fps_timer;
            self.frame_counter = 0;
            self.fps_timer = 0.0;

            let gpu_info = self.gpu.info();
            self.gpu_vram_total = gpu_info.total_vram();
            self.gpu_vram_used = gpu_info.used_vram();
            self.gpu_load = gpu_info.load_pct() as f32;
            self.gpu_temp = gpu_info.temperature() as f32 / 1000.0;
        }
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform.update_view_proj(&mut self.camera);
        self.camera_uniform.increment_time(dt);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        self.compute_fft();
        self.combine_cascades();
        self.update_foam();
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let depth_view = self.depth_texture.view.clone();

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("render_encoder"),
                });

        let raw_input = self.egui_state.take_egui_input(&self.window);
        let egui_context = self.egui_state.egui_ctx().clone();
        let egui_output = egui_context.run(raw_input, |context| {
            if self.show_setting_ui {
                self.render_settings_ui(context);
            }
        });
        self.egui_state
            .handle_platform_output(&self.window, egui_output.platform_output);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };
        for (id, image_delta) in &egui_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }
        for id in &egui_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        let clipped_primitives = self
            .egui_state
            .egui_ctx()
            .tessellate(egui_output.shapes, screen_descriptor.pixels_per_point);
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        let debug_info_text = self.get_debug_text();
        let debug_info_opacity = if self.show_debug_text { 1.0 } else { 0.0 };

        let debug_info_section = wgpu_text::glyph_brush::Section::default()
            .add_text(
                wgpu_text::glyph_brush::Text::new(debug_info_text.as_str())
                    .with_scale(25.0)
                    .with_color([0.98, 0.98, 0.98, debug_info_opacity]),
            )
            .with_bounds((1280.0, 1440.0))
            .with_layout(
                wgpu_text::glyph_brush::Layout::default_wrap()
                    .h_align(HorizontalAlign::Left)
                    .v_align(VerticalAlign::Top)
                    .line_breaker(BuiltInLineBreaker::UnicodeLineBreaker),
            )
            .with_screen_position((10.0, 10.0));

        let crosshair_section = wgpu_text::glyph_brush::Section::default()
            .add_text(
                wgpu_text::glyph_brush::Text::new("+")
                    .with_scale(20.0)
                    .with_color([0.98, 0.98, 0.98, 1.0]),
            )
            .with_bounds((40.0, 40.0))
            .with_layout(
                wgpu_text::glyph_brush::Layout::default()
                    .h_align(HorizontalAlign::Center)
                    .v_align(VerticalAlign::Center),
            )
            .with_screen_position((
                self.surface_config.width as f32 * 0.5 - 20.0,
                self.surface_config.height as f32 * 0.5 - 20.0,
            ));

        self.text_brush
            .queue(
                &self.device,
                &self.queue,
                [&debug_info_section, &crosshair_section],
            )
            .unwrap();

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("combined_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.skybox.skybox_render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.ocean_settings_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.skybox.skybox_vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                self.skybox.skybox_index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.draw_indexed(0..self.skybox.num_skybox_indices, 0, 0..1);

            let combined_bind_group = if self.combined_output_is_ping {
                &self.combined_render_bind_group_ping
            } else {
                &self.combined_render_bind_group_pong
            };

            let foam_bind_group = if self.foam_output_is_a {
                &self.foam_render_bind_groups[1]
            } else {
                &self.foam_render_bind_groups[0]
            };

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(2, combined_bind_group, &[]);
            render_pass.set_bind_group(3, foam_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);

            self.text_brush.draw(&mut render_pass);

            let mut render_pass = render_pass.forget_lifetime();
            self.egui_renderer
                .render(&mut render_pass, &clipped_primitives, &screen_descriptor);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

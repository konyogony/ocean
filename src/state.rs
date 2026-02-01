use crate::camera::{Camera, CameraController, CameraUniform};
use crate::settings::{OceanPreset, OceanSettingsBuilder, OceanSettingsUniform};
use crate::skybox::Skybox;
use crate::texture::{Texture, DEPTH_FORMAT, FFT_TEXTURE_FORMAT};
use crate::vertex::{InitialData, Vertex};
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

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FFTUniform {
    // This will go from 0 to log_2(N) - 1
    pub stage: u32,
    pub is_vertical: u32,
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
    pub camera: Camera,
    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_controller: CameraController,

    pub text_brush: wgpu_text::TextBrush,

    pub depth_texture: Texture,
    pub skybox: Skybox,

    pub last_frame_time_instant: Instant,
    pub fps: f32,
    pub frame_counter: u32,
    pub fps_timer: f32,
    pub max_magnitude: f32,
    pub avg_magnitude: f32,
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

    pub fft_texture_ping_h_dx: Texture,
    pub fft_texture_pong_h_dx: Texture,
    pub fft_texture_ping_dz: Texture,
    pub fft_texture_pong_dz: Texture,

    pub height_field_bind_group_ping: wgpu::BindGroup,
    pub height_field_bind_group_pong: wgpu::BindGroup,
    pub fft_bind_groups_ping: Vec<wgpu::BindGroup>,
    pub fft_bind_groups_pong: Vec<wgpu::BindGroup>,

    pub fft_group_layout: wgpu::BindGroupLayout,
    pub height_field_bind_group_layout: wgpu::BindGroupLayout,
    pub fft_config_buffer: wgpu::Buffer,
    pub fft_uniform_buffer: wgpu::Buffer,
    pub fft_uniform_size: u64,
    pub step_size: u64,
    pub fft_output_is_a: bool,

    pub fft_min: f32,
    pub fft_max: f32,
    pub fft_avg: f32,
    pub fft_samples_checked: u32,

    pub initial_data_group: wgpu::BindGroup,
    pub initial_data_buffer: wgpu::Buffer,
    pub initial_data_group_layout: wgpu::BindGroupLayout,

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

    pub egui_state: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,
    pub show_setting_ui: bool,
    pub show_debug_text: bool,
    pub draft_settings: OceanSettingsUniform,
    pub settings_changed: bool,

    // For some apparent reason I read that this HAS to be at the bottom (not fact checked)
    pub window: Arc<Window>,
}

impl State {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        // Initial window and surface setup
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
            present_mode: wgpu::PresentMode::Fifo, // Vsync (to like lower gpu stress or whatever). wgpu::PresentMode::Mailbox for uncapped but +idle
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // Ocean settings setup
        let ocean_seed = rand::rng().random::<u32>();
        let current_ocean_preset =
            OceanPreset::load_preset("deep_blue_cinematic", Path::new("presets/"));
        let ocean_settings_uniform = OceanSettingsBuilder::from_preset(&current_ocean_preset)
            .ocean_seed(ocean_seed)
            .build();
        let available_presets = OceanPreset::get_preset_list(Path::new("presets/"))?;

        let ocean_settings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ocean Settings Buffer"),
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
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            OceanSettingsUniform,
                        >() as u64),
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

        // Create a depth texture
        let depth_texture =
            Texture::create_depth_texture(&device, &surface_config, "depth_texture");

        // Camera setup pointing North
        let mut camera = Camera {
            forward: cgmath::Vector3::zero(),
            eye: (0.0, 10.0, 0.0).into(),
            yaw: Deg(-90.0).into(),
            pitch: Deg(-20.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: surface_config.width as f32 / surface_config.height as f32,
            fovy: ocean_settings_uniform.fovy,
            znear: 0.1,
            zfar: ocean_settings_uniform.zfar, // Increase for higher render distance
            flip_y: false,
            bearing: Deg(0.0).into(),
        };

        let mut camera_uniform = CameraUniform::default();
        camera_uniform.update_view_proj(&mut camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
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
                label: Some("camera_bind_group_layour"),
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

        // Or use include_wgsl! next time
        // UPD: include_str!() seems to actually perform better
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
        });

        // Setting up the surface as well as creating initial waves for simple sum of sine waves
        let (verticies, indicies) = Vertex::generate_plane(
            &ocean_settings_uniform.mesh_size,
            ocean_settings_uniform.fft_subdivisions * 2, // TODO: CHANGE
        );

        // Initial Data Initalisation

        let twiddle_factor_array =
            InitialData::generate_twiddle_factors(ocean_settings_uniform.fft_subdivisions);

        let (initial_data_array, max_magnitude, avg_magnitude) = InitialData::generate_data(
            ocean_settings_uniform.fft_size,
            ocean_settings_uniform.fft_subdivisions,
            ocean_settings_uniform.wind_vector,
            ocean_settings_uniform.l_small,
            ocean_settings_uniform.amplitude,
            ocean_settings_uniform.max_w,
            ocean_settings_uniform.ocean_seed,
        );

        let initial_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Initial Data Buffer"),
            contents: bytemuck::cast_slice(&initial_data_array),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let twiddle_factor_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFT Twiddle Buffer"),
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
            label: Some("initial_data_group"),
        });

        // Updated ping-pong texture model

        let fft_texture_ping_h_dx = Texture::create_storage_texture(
            &device,
            ocean_settings_uniform.fft_subdivisions,
            "fft_texture_ping_h_dx",
        );
        let fft_texture_pong_h_dx = Texture::create_storage_texture(
            &device,
            ocean_settings_uniform.fft_subdivisions,
            "fft_texture_pong_h_dx",
        );
        let fft_texture_ping_dz = Texture::create_storage_texture(
            &device,
            ocean_settings_uniform.fft_subdivisions,
            "fft_texture_ping_dz",
        );
        let fft_texture_pong_dz = Texture::create_storage_texture(
            &device,
            ocean_settings_uniform.fft_subdivisions,
            "fft_texture_pong_dz",
        );

        // Ping-Pong buffer model
        let fft_uniform_size = std::mem::size_of::<FFTUniform>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let step_size = (fft_uniform_size + alignment - 1) & !(alignment - 1);

        let total_size = step_size * ocean_settings_uniform.pass_num as u64 * 2;
        let fft_config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT Config Array"),
            size: total_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for stage in 0..ocean_settings_uniform.pass_num {
            for is_vertical in 0..2 {
                let i = stage * 2 + is_vertical;
                let uniform = FFTUniform { stage, is_vertical };
                let offset = i as u64 * step_size;
                queue.write_buffer(&fft_config_buffer, offset, bytemuck::cast_slice(&[uniform]));
            }
        }

        let fft_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT Uniform Buffer"),
            size: std::mem::size_of::<FFTUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let fft_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fft_group_layout"),
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
                // Read for h_dx
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
                // Write for h_dx
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
                // Read for dz
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: FFT_TEXTURE_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Write for dz
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
            ],
        });

        let (fft_bind_groups_ping, fft_bind_groups_pong) = Self::populate_fft_bind_groups(
            &device,
            &ocean_settings_uniform,
            &fft_group_layout,
            &fft_config_buffer,
            &fft_texture_ping_h_dx,
            &fft_texture_pong_h_dx,
            &fft_texture_ping_dz,
            &fft_texture_pong_dz,
            fft_uniform_size,
            step_size,
        );

        let fft_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FFT Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fft.wgsl").into()),
        });

        let fft_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FFT Pipeline Layout"),
            bind_group_layouts: &[
                &ocean_settings_bind_group_layout,
                &fft_group_layout,
                &camera_bind_group_layout,
                &initial_data_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let fft_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("FFT Compute Layout"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_shader,
                entry_point: Some("fft_step"),
                compilation_options: Default::default(),
                cache: None,
            });

        let spectrum_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Spectrum Pipeline Layout"),
                bind_group_layouts: &[
                    &ocean_settings_bind_group_layout,
                    &fft_group_layout,
                    &camera_bind_group_layout,
                    &initial_data_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let spectrum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Spectrum Update Pipeline"),
            layout: Some(&spectrum_pipeline_layout),
            module: &fft_shader,
            entry_point: Some("update_spectrum"),
            compilation_options: Default::default(),
            cache: None,
        });

        let height_field_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("height_field_bind_group_layout"),
                entries: &[
                    // h_dx
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // dz
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let height_field_bind_group_ping = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("height_field_bind_group_ping"),
            layout: &height_field_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&fft_texture_ping_h_dx.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&fft_texture_ping_dz.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&fft_texture_ping_h_dx.sampler),
                },
            ],
        });

        let height_field_bind_group_pong = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("height_field_bind_group_pong"),
            layout: &height_field_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&fft_texture_pong_h_dx.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&fft_texture_pong_dz.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&fft_texture_pong_h_dx.sampler),
                },
            ],
        });

        // Creating the skybox

        let skybox = Skybox::new(&device, &queue, &surface_config, &camera_bind_group_layout)?;

        // Setting up the render pipelines

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &ocean_settings_bind_group_layout,
                    &camera_bind_group_layout,
                    &height_field_bind_group_layout,
                    &skybox.skybox_texture_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Vertex::desc()],
            },
            // Used when converting verticies to triangles
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
                format: crate::texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false, // Antialiasing
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
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&verticies),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let num_indices = indicies.len() as u32;

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indicies),
            usage: wgpu::BufferUsages::INDEX,
        });

        let depth_stencil = Some(wgpu::DepthStencilState {
            format: crate::texture::DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        // Setting up the text

        let font_bytes = include_bytes!("../static/JetBrainsMono.ttf");
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

        // Just collecting system info

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

        let fft_min = -max_magnitude;
        let fft_max = max_magnitude;
        let fft_avg = avg_magnitude;
        let fft_samples_checked = 4096;

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
            initial_data_group,
            fft_output_is_a: true,
            fft_uniform_buffer,
            fft_group_layout,
            height_field_bind_group_layout,
            fft_bind_groups_pong,
            fft_bind_groups_ping,
            fft_texture_ping_h_dx,
            step_size,
            fft_config_buffer,
            fft_texture_pong_h_dx,
            fft_texture_ping_dz,
            fft_texture_pong_dz,
            fft_compute_pipeline,
            fft_uniform_size,
            spectrum_pipeline,
            height_field_bind_group_ping,
            height_field_bind_group_pong,
            initial_data_buffer,
            initial_data_group_layout,
            avg_magnitude,
            max_magnitude,
            gpu_name,
            cpu_name,
            kernel_version,
            os_name,
            sys,
            fft_min,
            fft_max,
            fft_avg,
            fft_samples_checked,
            current_ocean_preset,
            available_presets,
            preset_name_to_load: String::from(""),
            preset_name_to_create: String::from(""),
            preset_author_to_create: String::from(""),
            preset_description_to_create: String::from(""),
            ocean_settings_uniform,
            ocean_settings_buffer,
            ocean_settings_bind_group,
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
            self.depth_texture =
                Texture::create_depth_texture(&self.device, &self.surface_config, "depth_texture");
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
            // Calculate the average fps
            self.fps = self.frame_counter as f32 / self.fps_timer;
            // Reset counter and timer
            self.frame_counter = 0;
            self.fps_timer = 0.0;

            let gpu_info = self.gpu.info();
            self.gpu_vram_total = gpu_info.total_vram();
            self.gpu_vram_used = gpu_info.used_vram();
            self.gpu_load = gpu_info.load_pct() as f32;
            self.gpu_temp = gpu_info.temperature() as f32 / 1000.0;
        }

        // Add dt so the cam speed isnt dependant on the FPS
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform.update_view_proj(&mut self.camera);
        // Okay now we are using real time.
        self.camera_uniform.increment_time(dt);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        self.compute_fft();
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
                    label: Some("Render Encoder"),
                });

        // egui stuff, for settings UI
        let raw_input = self.egui_state.take_egui_input(&self.window);
        // Some hack
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
                label: Some("Combined Render Pass"),
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
                        load: wgpu::LoadOp::Clear(1.0), // Clear depth to the farthest value
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Skybox first
            render_pass.set_pipeline(&self.skybox.skybox_render_pipeline);
            // Camera in the middle?
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.skybox.skybox_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.skybox.skybox_vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                self.skybox.skybox_index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.draw_indexed(0..self.skybox.num_skybox_indices, 0, 0..1);

            // Instead of copying buffers, we can just switch between two bind groups
            let height_field_bind_group = if self.fft_output_is_a {
                &self.height_field_bind_group_ping
            } else {
                &self.height_field_bind_group_pong
            };

            // Ocean second
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(2, height_field_bind_group, &[]);
            render_pass.set_bind_group(3, &self.skybox.skybox_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);

            // Text last
            self.text_brush.draw(&mut render_pass);

            // Some hack
            let mut render_pass = render_pass.forget_lifetime();
            self.egui_renderer
                .render(&mut render_pass, &clipped_primitives, &screen_descriptor);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

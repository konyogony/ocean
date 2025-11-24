use crate::camera::{Camera, CameraController, CameraUniform};
use crate::texture::Texture;
use crate::wave::{gather_wave_data, WaveData, WaveDataUniform};
use anyhow::Result;
use cgmath::{Deg, Zero};
use ocean::{generate_plane, Vertex};
use std::sync::Arc;
use wgpu::{util::DeviceExt, Color};
use wgpu_text::glyph_brush::ab_glyph::FontArc;
use wgpu_text::glyph_brush::{BuiltInLineBreaker, HorizontalAlign, VerticalAlign};
use winit::window::Window;

pub const WAVE_NUMBER: usize = 16;

const SKYBOX_VERTICES: &[f32] = &[
    -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0,
    -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
];

const SKYBOX_INDICES: &[u16] = &[
    0, 1, 2, 2, 3, 0, // Front
    1, 5, 6, 6, 2, 1, // Right
    5, 4, 7, 7, 6, 5, // Back
    4, 0, 3, 3, 7, 4, // Left
    3, 2, 6, 6, 7, 3, // Top
    4, 5, 1, 1, 0, 4, // Bottom
];

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TimeUniform {
    pub time_uniform: f32,
}

impl TimeUniform {
    pub fn increment_time(&mut self, step: f32) {
        self.time_uniform += step
    }
}

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_indices: u32,
    index_buffer: wgpu::Buffer,
    // diffuse_bind_group: wgpu::BindGroup,
    // diffuse_texture: Texture,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_controller: CameraController,

    text_brush: wgpu_text::TextBrush,

    depth_texture: Texture,
    _skybox_texture: Texture,
    skybox_render_pipeline: wgpu::RenderPipeline,
    skybox_bind_group: wgpu::BindGroup,
    skybox_vertex_buffer: wgpu::Buffer,
    skybox_index_buffer: wgpu::Buffer,
    num_skybox_indices: u32,

    time_uniform: TimeUniform,
    time_buffer: wgpu::Buffer,
    time_bind_group: wgpu::BindGroup,
    _wave_data_uniform: WaveDataUniform,
    _wave_data_buffer: wgpu::Buffer,
    wave_data_bind_group: wgpu::BindGroup,

    // For some apparent reason I read that this HAS to be at the bottom
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
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let depth_texture =
            Texture::create_depth_texture(&device, &surface_config, "depth_texture");

        // let diffuse_bytes = include_bytes!("ocean.jpg");
        // let diffuse_texture = Texture::from_bytes(diffuse_bytes, &device, &queue, "tree").unwrap();

        // let texture_bind_group_layout =
        //     device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        //         label: Some("texture_bind_group_layout"),
        //         entries: &[
        //             wgpu::BindGroupLayoutEntry {
        //                 binding: 0,
        //                 count: None,
        //                 visibility: wgpu::ShaderStages::FRAGMENT,
        //                 ty: wgpu::BindingType::Texture {
        //                     sample_type: wgpu::TextureSampleType::Float { filterable: true },
        //                     view_dimension: wgpu::TextureViewDimension::D2,
        //                     multisampled: false,
        //                 },
        //             },
        //             wgpu::BindGroupLayoutEntry {
        //                 binding: 1,
        //                 count: None,
        //                 visibility: wgpu::ShaderStages::FRAGMENT,
        //                 ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        //             },
        //         ],
        //     });

        // let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        //     layout: &texture_bind_group_layout,
        //     label: Some("diffuse_bind_group"),
        //     entries: &[
        //         wgpu::BindGroupEntry {
        //             binding: 0,
        //             resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
        //         },
        //         wgpu::BindGroupEntry {
        //             binding: 1,
        //             resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
        //         },
        //     ],
        // });

        let mut camera = Camera {
            forward: cgmath::Vector3::zero(),
            eye: (0.0, 1.0, 2.0).into(),
            yaw: Deg(-90.0).into(),
            pitch: Deg(-20.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: surface_config.width as f32 / surface_config.height as f32,
            fovy: 60.0,
            znear: 0.1,
            zfar: 1000.0, // Increase for higher render distance
            flip_y: false,
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
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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

        // Speed, Sensitivity, Speed boost (cntrl)
        let camera_controller = CameraController::new(0.02, 0.002, 5.0);

        // Or use include_wgsl! next time
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let time_uniform = TimeUniform { time_uniform: 0.00 };

        let time_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Time Buffer"),
            contents: bytemuck::cast_slice(&[time_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let time_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("time_bind_group_layour"),
            });

        let time_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &time_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: time_buffer.as_entire_binding(),
            }],
            label: Some("time_bind_group"),
        });

        // Actual wave stuff
        let (verticies, indicies) = generate_plane(256.0, 1024);

        let mut waves = [WaveData::default(); WAVE_NUMBER];
        for (i, w) in gather_wave_data(WAVE_NUMBER)
            .into_iter()
            .enumerate()
            .take(WAVE_NUMBER)
        {
            waves[i] = w;
        }

        let wave_data_uniform = WaveDataUniform { waves };

        let wave_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wave Data Buffer"),
            contents: bytemuck::bytes_of(&wave_data_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let wave_data_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("wave_data_bind_group_layour"),
            });

        let wave_data_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &wave_data_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wave_data_buffer.as_entire_binding(),
            }],
            label: Some("wave_data_bind_group"),
        });

        let skybox_texture = Texture::load_skybox_texture(
            &device,
            &queue,
            [
                include_bytes!("../images/px1.png"), // Should be +X
                include_bytes!("../images/nx1.png"), // Should be -X
                include_bytes!("../images/py1.png"), // Should be +Y
                include_bytes!("../images/ny1.png"), // Should be -Y
                include_bytes!("../images/pz1.png"), // Should be +Z
                include_bytes!("../images/nz1.png"), // Should be -Z
            ],
            "skybox_texture",
        )?;

        let skybox_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("skybox.wgsl").into()),
        });

        let skybox_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("skybox_texture_bind_group_layout"),
            });

        let skybox_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &skybox_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&skybox_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&skybox_texture.sampler),
                },
            ],
            label: Some("skybox_bind_group"),
        });

        let skybox_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Skybox Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &skybox_texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let skybox_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Skybox VB"),
            contents: bytemuck::cast_slice(SKYBOX_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let skybox_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Skybox IB"),
            contents: bytemuck::cast_slice(SKYBOX_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_skybox_indices = SKYBOX_INDICES.len() as u32;

        let skybox_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Skybox Render Pipeline"),
                layout: Some(&skybox_render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &skybox_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 3 * 4,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &skybox_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Texture::DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multiview: None,
                multisample: wgpu::MultisampleState::default(),
                cache: None,
            });

        // Boring stuff...

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    //                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &time_bind_group_layout,
                    &wave_data_bind_group_layout,
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
                format: Texture::DEPTH_FORMAT,
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

        // Other library for text
        // let mut font_system = glyphon::FontSystem::new();
        // let swash_cache = glyphon::SwashCache::new();
        // let cache = glyphon::Cache::new(&device);
        // let viewport = glyphon::Viewport::new(&device, &cache);
        // let mut atlas =
        //     glyphon::TextAtlas::new(&device, &queue, &cache, wgpu::TextureFormat::Bgra8UnormSrgb);
        // let text_render =
        //     glyphon::TextRenderer::new(&atlas, &device, wgpu::MultisampleState::default(), None);
        // let mut text_buffer =
        //     glyphon::Buffer::new(&mut font_system, glyphon::Metrics::new(30.0, 42.0));

        // text_buffer.set_size(
        //     &mut font_system,
        //     Some(surface_config.width),
        //     Some(surface_config.height),
        // );
        // text_buffer.set_text(
        //     &mut font_system,
        //     "Hello Worlddddd",
        //     &glyphon::Attrs::new().family(&glyphon::Family::SansSerif),
        //     glyphon::Shaping::Advanced,
        // );
        // text_buffer.shape_until_scroll(&mut font_system, false);

        let depth_stencil = Some(wgpu::DepthStencilState {
            format: Texture::DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let font_bytes = include_bytes!("./JetBrainsMono.ttf");
        let font = FontArc::try_from_slice(font_bytes).unwrap();
        let text_brush = wgpu_text::BrushBuilder::using_font(font)
            .with_depth_stencil(depth_stencil)
            .build(
                &device,
                surface_config.width,
                surface_config.height,
                surface_config.format,
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
            // diffuse_bind_group,
            // diffuse_texture,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            depth_texture,
            skybox_vertex_buffer,
            _skybox_texture: skybox_texture,
            skybox_index_buffer,
            skybox_render_pipeline,
            num_skybox_indices,
            skybox_bind_group,
            time_uniform,
            text_brush,
            time_buffer,
            time_bind_group,
            _wave_data_buffer: wave_data_buffer,
            wave_data_bind_group,
            _wave_data_uniform: wave_data_uniform,
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
        self.camera_controller.process_window_events(event)
    }

    pub fn handle_device_event(&mut self, event: &winit::event::DeviceEvent) -> bool {
        self.camera_controller.process_device_events(event)
    }

    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&mut self.camera);
        self.time_uniform.increment_time(0.016 / 8.0);
        // i think this is 60fps. But then divided
        // by 8 cause it was too fast
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        self.queue.write_buffer(
            &self.time_buffer,
            0,
            bytemuck::cast_slice(&[self.time_uniform]),
        );
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

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        let pos = format!(
            "x: {}, y: {}, z: {}",
            &self.camera.eye.x.floor(),
            &self.camera.eye.y.floor(),
            &self.camera.eye.z.floor()
        );

        let section = wgpu_text::glyph_brush::Section::default()
            .add_text(
                wgpu_text::glyph_brush::Text::new(pos.as_str())
                    .with_scale(30.0)
                    .with_color([1.0, 1.0, 1.0, 1.0]),
            )
            .with_bounds((650.0, 180.0))
            .with_layout(
                wgpu_text::glyph_brush::Layout::default_wrap()
                    .h_align(HorizontalAlign::Left)
                    .v_align(VerticalAlign::Top)
                    .line_breaker(BuiltInLineBreaker::UnicodeLineBreaker),
            )
            .with_screen_position((30.0, 10.0));

        self.text_brush
            .queue(&self.device, &self.queue, [&section])
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
                    view: &self.depth_texture.view,
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
            render_pass.set_pipeline(&self.skybox_render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.skybox_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.skybox_vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                self.skybox_index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.draw_indexed(0..self.num_skybox_indices, 0, 0..1);

            // Ocean second
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.time_bind_group, &[]);
            render_pass.set_bind_group(2, &self.wave_data_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);

            // Text last
            self.text_brush.draw(&mut render_pass);
            // self.text_renderer.prepare_with_custom(
            //     &self.device,
            //     &self.queue,
            //     &mut self.font_system,
            //     &self.text_atlast,
            //     &self.viewport,
            //     [glyphon::TextArea {
            //         buffer: &self.text_buffer,
            //         left: 30.0,
            //         top: 10.0,
            //         scale: 1.0,
            //         bounds: glyphon::TextBounds {
            //             left: 0,
            //             top: 0,
            //             right: 650,
            //             bottom: 180,
            //         },
            //         default_color: glyphon::Color::rgb(255, 255, 255),
            //         custom_glyphs: None,
            //     }],
            //     &self.swash_cache,
            //     None,
            // );
            // self.text_renderer
            //     .render(&self.text_atlast, &self.viewport, &mut render_pass)
            //     .unwrap();
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        // self.text_atlast.trim();

        Ok(())
    }
}

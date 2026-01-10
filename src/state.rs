use crate::camera::{Camera, CameraController, CameraUniform};
use crate::skybox::Skybox;
use crate::texture::Texture;
use crate::vertex::InitialData;
use crate::vertex::Vertex;
use crate::{DESC, VERSION};
use anyhow::Result;
use cgmath::{Deg, Zero};
use chrono::Local;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use wgpu::{util::DeviceExt, Color};
use wgpu_text::glyph_brush::ab_glyph::FontArc;
use wgpu_text::glyph_brush::{BuiltInLineBreaker, HorizontalAlign, VerticalAlign};
use winit::window::Window;

pub const WAVE_NUMBER: usize = 16;
pub const FOVY: f32 = 60.0;
pub const ZFAR: f32 = 1500.0;
pub const DEFAULT_CAM_SPEED: f32 = 0.02;
pub const CAM_SENSITIVITY: f32 = 0.002;
pub const CAM_SPEED_BOOST: f32 = 5.0;
pub const MESH_SIZE: f32 = 1024.0;
pub const MESH_SUBDIVISIONS: u32 = 2048;

// TODO: Figure out if this time unfiorm is correct/needed
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
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_controller: CameraController,

    text_brush: wgpu_text::TextBrush,

    depth_texture: Texture,
    skybox: Skybox,

    time_uniform: TimeUniform,
    time_buffer: wgpu::Buffer,
    last_frame_time_instant: Instant,
    fps: f32,
    frame_counter: u32,
    fps_timer: f32,

    time_bind_group: wgpu::BindGroup,
    initial_data_array: Vec<InitialData>,
    initial_data_buffer: wgpu::Buffer,
    initial_data_group: wgpu::BindGroup,

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
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

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
            fovy: FOVY,
            znear: 0.1,
            zfar: ZFAR, // Increase for higher render distance
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

        let camera_controller =
            CameraController::new(DEFAULT_CAM_SPEED, CAM_SENSITIVITY, CAM_SPEED_BOOST);

        // Or use include_wgsl! next time
        // UPD: include_str!() seems to actually perform better
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // This looks HELLA wonky
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
                label: Some("time_bind_group_layout"),
            });

        let time_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &time_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: time_buffer.as_entire_binding(),
            }],
            label: Some("time_bind_group"),
        });

        // Setting up the surface as well as creating initial waves for simple sum of sine waves
        let (verticies, indicies) = Vertex::generate_plane(&MESH_SIZE, MESH_SUBDIVISIONS);

        let initial_data_array: Vec<InitialData> = InitialData::generate_data(MESH_SIZE);

        let initial_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Initial Data Buffer"),
            contents: bytemuck::cast_slice(&initial_data_array),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let initial_data_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("initial_data_group_layout"),
            });

        let initial_data_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &initial_data_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: initial_data_buffer.as_entire_binding(),
            }],
            label: Some("initial_data_group"),
        });

        // Creating the skybox

        let skybox = Skybox::new(&device, &queue, &surface_config, &camera_bind_group_layout)?;

        // Setting up the render pipelines

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &time_bind_group_layout,
                    &initial_data_group_layout,
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
            time_uniform,
            fps: 0.0,
            last_frame_time_instant: Instant::now(),
            fps_timer: 0.0,
            frame_counter: 0,
            text_brush,
            time_buffer,
            time_bind_group,
            initial_data_array,
            initial_data_buffer,
            initial_data_group,
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
        }

        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&mut self.camera);
        // Okay now we are using real time.
        self.time_uniform.increment_time(dt);
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

        let system_time = SystemTime::now();
        let datetime: chrono::DateTime<Local> = system_time.into();
        let formatted_time = datetime.format("%Y-%m-%d %H:%M:%S.%3f UTC%Z").to_string();

        // [(bearing + 2pi) % 2pi] * [180/pi]
        let bearing_360 = ((self.camera.bearing.0 + std::f32::consts::TAU) % std::f32::consts::TAU)
            * 180.0
            / std::f32::consts::PI;
        // [-pi/2 < pitch < pi/2] * [180/pi]
        let pitch = self
            .camera
            .pitch
            .0
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2)
            * 180.0
            / std::f32::consts::PI;
        let compass_dir = Camera::bearing_to_compass(bearing_360);

        let tri_count = self.num_indices / 3;

        // Data collected by me, formatted by AI
        let debug_info_text = format!(
            "Ocean Simulation v{VERSION}\n\
            Stage: {DESC}\n\
            {formatted_time}\n\
            \n\
            XYZ: {x:.1} / {y:.1} / {z:.1}\n\
            Facing: {dir} ({bearing:.0}°)  Pitch: {pitch:+.1}°\n\
            FOV: {fov:.0}°  ViewDist: {zfar:.0}\n\
            \n\
            FPS: {fps:.0} ({ms:.1} ms)\n\
            Resolution: {w}x{h}\n\
            \n\
            Ocean:\n\
            Size: {size:.0} x {size:.0}\n\
            Subdivisions: {sub}\n\
            Waves: {waves}\n\
            Tris: {tris}",
            VERSION = VERSION,
            DESC = DESC,
            formatted_time = formatted_time,
            x = self.camera.eye.x,
            y = self.camera.eye.y,
            z = self.camera.eye.z,
            dir = compass_dir,
            bearing = bearing_360,
            pitch = pitch,
            fov = FOVY,
            zfar = ZFAR,
            fps = self.fps,
            ms = 1000.0 / self.fps.max(0.001),
            w = self.surface_config.width,
            h = self.surface_config.height,
            size = MESH_SIZE,
            sub = MESH_SUBDIVISIONS,
            waves = WAVE_NUMBER,
            tris = tri_count,
        );

        let debug_info_section = wgpu_text::glyph_brush::Section::default()
            .add_text(
                wgpu_text::glyph_brush::Text::new(debug_info_text.as_str())
                    .with_scale(25.0)
                    .with_color([0.98, 0.98, 0.98, 1.0]),
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

            // Ocean second
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.time_bind_group, &[]);
            render_pass.set_bind_group(2, &self.initial_data_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);

            // Text last
            self.text_brush.draw(&mut render_pass);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

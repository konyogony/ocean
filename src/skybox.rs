use crate::texture::Texture;
use anyhow::Result;
use wgpu::util::DeviceExt;

const SKYBOX_VERTICES: &[f32] = &[
    -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0,
    -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
];

const SKYBOX_INDICES: &[u16] = &[
    0, 1, 2, 2, 3, 0, // +Z
    1, 5, 6, 6, 2, 1, // +X
    5, 4, 7, 7, 6, 5, // -Z
    4, 0, 3, 3, 7, 4, // -X
    3, 2, 6, 6, 7, 3, // +Y
    4, 5, 1, 1, 0, 4, // -Y
];

const SKYBOX_FACES: [&[u8]; 6] = [
    include_bytes!("../static/skybox/px.png"), // Should be +X
    include_bytes!("../static/skybox/nx.png"), // Should be -X
    include_bytes!("../static/skybox/py.png"), // Should be +Y
    include_bytes!("../static/skybox/ny.png"), // Should be -Y
    include_bytes!("../static/skybox/pz.png"), // Should be +Z
    include_bytes!("../static/skybox/nz.png"), // Should be -Z
];

pub struct Skybox {
    pub skybox_texture: Texture,
    pub skybox_render_pipeline: wgpu::RenderPipeline,
    pub skybox_bind_group: wgpu::BindGroup,
    pub skybox_vertex_buffer: wgpu::Buffer,
    pub skybox_index_buffer: wgpu::Buffer,
    pub num_skybox_indices: u32,
}

impl Skybox {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self> {
        let skybox_texture =
            Texture::load_skybox_texture(device, queue, SKYBOX_FACES, "skybox_texture")?;

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
                bind_group_layouts: &[camera_bind_group_layout, &skybox_texture_bind_group_layout],
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
                    format: crate::texture::DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multiview: None,
                multisample: wgpu::MultisampleState::default(),
                cache: None,
            });

        Ok(Self {
            skybox_texture,
            skybox_bind_group,
            skybox_index_buffer,
            skybox_vertex_buffer,
            skybox_render_pipeline,
            num_skybox_indices,
        })
    }
}

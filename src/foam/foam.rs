use crate::pipeline::state::State;
use crate::settings::uniform::OceanSettingsUniform;
use crate::texture::instance::{TextureInstance, FFT_TEXTURE_FORMAT};

impl State {
    pub fn update_foam(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("foam_encoder"),
            });

        let fft_read_bind_group = if self.combined_output_is_ping {
            &self.combined_render_bind_group_ping
        } else {
            &self.combined_render_bind_group_pong
        };
        let (gen_read, gen_write) = if self.foam_output_is_a {
            (0, 1)
        } else {
            (1, 0)
        };

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.foam_generation_pipeline);
            pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
            pass.set_bind_group(1, &self.camera_bind_group, &[]);
            pass.set_bind_group(2, &self.foam_compute_bind_groups[gen_read], &[]);
            pass.set_bind_group(3, fft_read_bind_group, &[]);
            pass.dispatch_workgroups(
                self.ocean_settings_uniform.fft_subdivisions / 16,
                self.ocean_settings_uniform.fft_subdivisions / 16,
                1,
            );
        }

        let adv_read = gen_write;

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.foam_advection_pipeline);
            pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
            pass.set_bind_group(1, &self.camera_bind_group, &[]);
            pass.set_bind_group(2, &self.foam_compute_bind_groups[adv_read], &[]);
            pass.set_bind_group(3, fft_read_bind_group, &[]);
            pass.dispatch_workgroups(
                self.ocean_settings_uniform.fft_subdivisions / 16,
                self.ocean_settings_uniform.fft_subdivisions / 16,
                1,
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.foam_output_is_a = !self.foam_output_is_a;
    }

    pub fn init_foam(
        device: &wgpu::Device,
        ocean_settings_uniform: &OceanSettingsUniform,
        ocean_settings_bind_group_layout: &wgpu::BindGroupLayout,
        height_field_render_bind_group_layout: &wgpu::BindGroupLayout,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> (
        TextureInstance,
        TextureInstance,
        wgpu::ComputePipeline,
        wgpu::ComputePipeline,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        [wgpu::BindGroup; 2],
        [wgpu::BindGroup; 2],
    ) {
        let foam_texture_ping = TextureInstance::create_storage_texture(
            device,
            ocean_settings_uniform.fft_subdivisions,
            "foam_texture_ping",
        );

        let foam_texture_pong = TextureInstance::create_storage_texture(
            device,
            ocean_settings_uniform.fft_subdivisions,
            "foam_texture_pong",
        );

        let foam_compute_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("foam_bind_group_compute_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        count: None,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: FFT_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        count: None,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: FFT_TEXTURE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        count: None,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    },
                ],
            });

        let foam_render_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("foam_bind_group_render_layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let foam_compute_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("foam_bind_group_ping"),
                layout: &foam_compute_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&foam_texture_ping.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&foam_texture_pong.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&foam_texture_ping.sampler),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("foam_bind_group_pong"),
                layout: &foam_compute_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&foam_texture_pong.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&foam_texture_ping.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&foam_texture_ping.sampler),
                    },
                ],
            }),
        ];

        let foam_render_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("foam_render_bind_group_ping"),
                layout: &foam_render_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&foam_texture_ping.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&foam_texture_ping.sampler),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("foam_render_bind_group_pong"),
                layout: &foam_render_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&foam_texture_pong.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&foam_texture_ping.sampler),
                    },
                ],
            }),
        ];

        let foam_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("foam_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./foam.wgsl").into()),
        });

        let foam_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("foam_pipeline_layout"),
            bind_group_layouts: &[
                ocean_settings_bind_group_layout,
                camera_bind_group_layout,
                &foam_compute_layout,
                height_field_render_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let foam_generation_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("foam_generation_pipeline"),
                layout: Some(&foam_pipeline_layout),
                module: &foam_shader,
                entry_point: Some("compute_foam"),
                compilation_options: Default::default(),
                cache: None,
            });

        let foam_advection_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("foam_advection_pipeline"),
                layout: Some(&foam_pipeline_layout),
                module: &foam_shader,
                entry_point: Some("advect_foam"),
                compilation_options: Default::default(),
                cache: None,
            });

        (
            foam_texture_ping,
            foam_texture_pong,
            foam_generation_pipeline,
            foam_advection_pipeline,
            foam_compute_layout,
            foam_render_layout,
            foam_compute_bind_groups,
            foam_render_bind_groups,
        )
    }
}

use crate::pipeline::state::CascadeResources;
use crate::pipeline::state::CombineUniform;
use crate::pipeline::state::FFTUniform;
use crate::pipeline::state::State;
use crate::settings::uniform::OceanSettingsUniform;
use crate::texture::instance::TextureInstance;
use crate::vertex::vertex::Vertex;
use cgmath::{InnerSpace, Vector2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f32::consts::PI;
use wgpu::util::DeviceExt;
use wgpu::TextureView;

impl State {
    pub fn copy_cascades_to_array(&mut self) {
        let n = self.ocean_settings_uniform.fft_subdivisions;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for (i, cascade) in self.cascades.iter().enumerate() {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &cascade.texture_packed.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.cascade_array_texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: i as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: n,
                    height: n,
                    depth_or_array_layers: 1,
                },
            );
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }
    pub fn compute_fft(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fft_multi-pass_encoder"),
            });

        for cascade in &mut self.cascades {
            let passes = (self.ocean_settings_uniform.pass_num * 2) as usize;
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.spectrum_pipeline);
                pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
                pass.set_bind_group(1, &cascade.bind_groups_pong[0], &[]);
                pass.set_bind_group(2, &self.camera_bind_group, &[]);
                pass.set_bind_group(3, &cascade.initial_data_group, &[]);
                pass.dispatch_workgroups(
                    self.ocean_settings_uniform.fft_subdivisions / 16,
                    self.ocean_settings_uniform.fft_subdivisions / 16,
                    1,
                );
            }

            let mut current_reads_from_ping = true;

            for i in 0..passes {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.fft_compute_pipeline);

                let bind_group = if current_reads_from_ping {
                    &cascade.bind_groups_ping[i]
                } else {
                    &cascade.bind_groups_pong[i]
                };

                pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
                pass.set_bind_group(1, bind_group, &[]);
                pass.set_bind_group(2, &self.camera_bind_group, &[]);
                pass.set_bind_group(3, &cascade.initial_data_group, &[]);
                pass.dispatch_workgroups(
                    self.ocean_settings_uniform.fft_subdivisions / 16,
                    self.ocean_settings_uniform.fft_subdivisions / 16,
                    1,
                );

                current_reads_from_ping = !current_reads_from_ping;
            }

            cascade.output_is_ping = !passes.is_multiple_of(2);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn combine_cascades(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("combine_cascades_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.combined_clear_pipeline);
            pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
            pass.set_bind_group(1, &self.combined_read_write_bind_group_ping, &[]);
            pass.set_bind_group(2, &self.cascades[0].cascade_input_bind_group, &[]);
            pass.dispatch_workgroups(
                self.ocean_settings_uniform.fft_subdivisions / 16,
                self.ocean_settings_uniform.fft_subdivisions / 16,
                1,
            );
            pass.set_bind_group(1, &self.combined_read_write_bind_group_pong, &[]);
            pass.dispatch_workgroups(
                self.ocean_settings_uniform.fft_subdivisions / 16,
                self.ocean_settings_uniform.fft_subdivisions / 16,
                1,
            );
        }

        let mut current_output_is_ping = true;

        for cascade_index in 0..self.cascades.len() {
            let combined_bind_group = if current_output_is_ping {
                &self.combined_read_write_bind_group_ping
            } else {
                &self.combined_read_write_bind_group_pong
            };

            let cascade_input_bind_group = &self.cascades[cascade_index].cascade_input_bind_group;

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.combined_cascade_pipeline);
            pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
            pass.set_bind_group(1, combined_bind_group, &[]);
            pass.set_bind_group(2, cascade_input_bind_group, &[]);
            pass.dispatch_workgroups(
                self.ocean_settings_uniform.fft_subdivisions / 16,
                self.ocean_settings_uniform.fft_subdivisions / 16,
                1,
            );

            current_output_is_ping = !current_output_is_ping;
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.combined_output_is_ping = current_output_is_ping;
    }

    pub fn create_cascade_resource(
        &self,
        cascade_index: usize,
        twiddle_factor_buffer: &wgpu::Buffer,
        total_size: u64,
    ) -> CascadeResources {
        let n = self.ocean_settings_uniform.fft_subdivisions;
        let pass_num = self.ocean_settings_uniform.pass_num;

        let initial_data_array = InitialData::generate_data(
            self.ocean_settings_uniform.cascade_data[cascade_index][0],
            n,
            self.ocean_settings_uniform.wind_vector,
            self.ocean_settings_uniform.l_small,
            self.ocean_settings_uniform.cascade_data[cascade_index][1],
            self.ocean_settings_uniform.max_w,
            self.ocean_settings_uniform.ocean_seed,
        );

        let initial_data_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("initial_data_buffer_{cascade_index}")),
                    contents: bytemuck::cast_slice(&initial_data_array),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                });

        let initial_data_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.initial_data_group_layout,
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

        let texture_ping_h_dx = TextureInstance::create_texture(
            &self.device,
            n,
            &format!("fft_texture_ping_h_dx_{cascade_index}"),
        );
        let texture_pong_h_dx = TextureInstance::create_storage_texture(
            &self.device,
            n,
            &format!("fft_texture_pong_h_dx_{cascade_index}"),
        );
        let texture_ping_dz = TextureInstance::create_texture(
            &self.device,
            n,
            &format!("fft_texture_ping_dz_{cascade_index}"),
        );
        let texture_pong_dz = TextureInstance::create_storage_texture(
            &self.device,
            n,
            &format!("fft_texture_pong_dz_{cascade_index}"),
        );

        let packed_output = TextureInstance::create_storage_texture(
            &self.device,
            n,
            &format!("fft_packed_output_{cascade_index}"),
        );

        let config_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_config_buffer"),
            size: total_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for is_vertical in 0..2 {
            for stage in 0..pass_num {
                let i = stage * 2 + is_vertical;
                let uniform = FFTUniform {
                    stage,
                    is_vertical,
                    cascade_index: cascade_index as u32,
                    _pad: 0,
                };
                let offset = i as u64 * self.step_size;
                self.queue
                    .write_buffer(&config_buffer, offset, bytemuck::cast_slice(&[uniform]));
            }
        }

        let (bind_groups_ping, bind_groups_pong) = Self::populate_fft_bind_groups(
            &self.device,
            &self.ocean_settings_uniform,
            &self.fft_render_group_layout,
            &config_buffer,
            &texture_ping_h_dx,
            &texture_pong_h_dx,
            &texture_ping_dz,
            &texture_pong_dz,
            &packed_output,
            self.fft_uniform_size,
            self.step_size,
        );

        let height_field_render_bind_group_ping =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("height_field_bind_group_ping_{cascade_index}")),
                layout: &self.height_field_render_bind_group_layout,
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
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("height_field_bind_group_pong_{cascade_index}")),
                layout: &self.height_field_render_bind_group_layout,
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
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!(
                    "height_field_compute_bind_group_ping_{cascade_index}"
                )),
                layout: &self.height_field_compute_bind_group_layout,
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
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!(
                    "height_field_compute_bind_group_pong_{cascade_index}"
                )),
                layout: &self.height_field_compute_bind_group_layout,
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
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("combine_uniform_buffer_{cascade_index}")),
                    contents: bytemuck::cast_slice(&[CombineUniform {
                        cascade_index: cascade_index as u32,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let cascade_input_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("cascade_input_bind_group_{cascade_index}")),
            layout: &self.cascade_input_bind_group_layout,
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

        CascadeResources {
            config_buffer,
            size: self.ocean_settings_uniform.cascade_data[cascade_index][0],
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
            texture_packed: packed_output,
            initial_data_group,
            initial_data_buffer,
            combine_uniform_buffer,
            cascade_input_bind_group,
        }
    }

    pub fn reinit_fft_resources(&mut self) {
        let n = self.ocean_settings_uniform.fft_subdivisions;
        let pass_num = self.ocean_settings_uniform.pass_num;
        let total_size = self.step_size * pass_num as u64 * 2;

        let new_count = self.ocean_settings_uniform.cascade_count as usize;
        let old_count = self.cascades.len();

        let twiddle_array = InitialData::generate_twiddle_factors(n);
        let twiddle_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("twiddle_buffer"),
                contents: bytemuck::cast_slice(&twiddle_array),
                usage: wgpu::BufferUsages::STORAGE,
            });

        self.combined_texture_ping =
            TextureInstance::create_storage_texture(&self.device, n, "combined_ping");
        self.combined_texture_pong =
            TextureInstance::create_storage_texture(&self.device, n, "combined_pong");

        self.combined_read_write_bind_group_ping =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("combined_read_write_bind_group_ping"),
                layout: &self.combined_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_ping.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_pong.view,
                        ),
                    },
                ],
            });

        self.combined_read_write_bind_group_pong =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("combined_read_write_bind_group_pong"),
                layout: &self.combined_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_pong.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_ping.view,
                        ),
                    },
                ],
            });

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
            &self.device,
            &self.ocean_settings_uniform,
            &self.ocean_settings_bind_group_layout,
            &self.height_field_render_bind_group_layout,
            &self.height_field_compute_bind_group_layout,
            &self.camera_bind_group_layout,
        );
        self.foam_texture_ping = foam_texture_ping;
        self.foam_texture_pong = foam_texture_pong;
        self.foam_advection_pipeline = foam_advection_pipeline;
        self.foam_compute_layout = foam_compute_layout;
        self.foam_compute_bind_groups = foam_compute_bind_groups;
        self.foam_render_bind_groups = foam_render_bind_groups;
        self.foam_render_layout = foam_render_layout;
        self.foam_generation_pipeline = foam_generation_pipeline;

        for (cascade_index, cascade) in self.cascades.iter_mut().enumerate() {
            cascade.output_is_ping = false;
            cascade.texture_ping_h_dx = TextureInstance::create_storage_texture(
                &self.device,
                n,
                &format!("fft_texture_ping_h_dx_{cascade_index}"),
            );
            cascade.texture_pong_h_dx = TextureInstance::create_storage_texture(
                &self.device,
                n,
                &format!("fft_texture_pong_h_dx_{cascade_index}"),
            );
            cascade.texture_ping_dz = TextureInstance::create_storage_texture(
                &self.device,
                n,
                &format!("fft_texture_ping_dz_{cascade_index}"),
            );
            cascade.texture_pong_dz = TextureInstance::create_storage_texture(
                &self.device,
                n,
                &format!("fft_texture_pong_dz_{cascade_index}"),
            );
            cascade.texture_packed = TextureInstance::create_storage_texture(
                &self.device,
                n,
                &format!("fft_packed_output_{cascade_index}"),
            );

            cascade.config_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fft_config_buffer"),
                size: total_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            for is_vertical in 0..2 {
                for stage in 0..pass_num {
                    let i = stage * 2 + is_vertical;
                    let uniform = FFTUniform {
                        stage,
                        is_vertical,
                        cascade_index: cascade_index as u32,
                        _pad: 0,
                    };
                    let offset = i as u64 * self.step_size;
                    self.queue.write_buffer(
                        &cascade.config_buffer,
                        offset,
                        bytemuck::cast_slice(&[uniform]),
                    );
                }
            }

            let initial_data_array = InitialData::generate_data(
                self.ocean_settings_uniform.cascade_data[cascade_index][0],
                n,
                self.ocean_settings_uniform.wind_vector,
                self.ocean_settings_uniform.l_small,
                self.ocean_settings_uniform.cascade_data[cascade_index][1],
                self.ocean_settings_uniform.max_w,
                self.ocean_settings_uniform.ocean_seed,
            );

            cascade.initial_data_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("initial_data_buffer_{cascade_index}")),
                        contents: bytemuck::cast_slice(&initial_data_array),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    });

            cascade.initial_data_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.initial_data_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: cascade.initial_data_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: twiddle_buffer.as_entire_binding(),
                        },
                    ],
                    label: None,
                });

            let (fft_bind_groups_ping, fft_bind_groups_pong) = Self::populate_fft_bind_groups(
                &self.device,
                &self.ocean_settings_uniform,
                &self.fft_render_group_layout,
                &cascade.config_buffer,
                &cascade.texture_ping_h_dx,
                &cascade.texture_pong_h_dx,
                &cascade.texture_ping_dz,
                &cascade.texture_pong_dz,
                &cascade.texture_packed,
                self.fft_uniform_size,
                self.step_size,
            );

            cascade.bind_groups_ping = fft_bind_groups_ping;
            cascade.bind_groups_pong = fft_bind_groups_pong;

            cascade.height_field_render_bind_group_ping =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.height_field_render_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &cascade.texture_ping_h_dx.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                &cascade.texture_ping_h_dx.sampler,
                            ),
                        },
                    ],
                    label: None,
                });

            cascade.height_field_render_bind_group_pong =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.height_field_render_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &cascade.texture_pong_h_dx.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                &cascade.texture_pong_h_dx.sampler,
                            ),
                        },
                    ],
                    label: None,
                });

            cascade.height_field_compute_bind_group_ping =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!(
                        "height_field_compute_bind_group_ping_{cascade_index}"
                    )),
                    layout: &self.height_field_compute_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &cascade.texture_ping_h_dx.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &cascade.texture_ping_dz.view,
                            ),
                        },
                    ],
                });

            cascade.height_field_compute_bind_group_pong =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!(
                        "height_field_compute_bind_group_pong_{cascade_index}"
                    )),
                    layout: &self.height_field_compute_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &cascade.texture_pong_h_dx.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &cascade.texture_pong_dz.view,
                            ),
                        },
                    ],
                });

            cascade.combine_uniform_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("combine_uniform_buffer_{cascade_index}")),
                        contents: bytemuck::cast_slice(&[CombineUniform {
                            cascade_index: cascade_index as u32,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        }]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            cascade.cascade_input_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("cascade_input_bind_group_ping_{cascade_index}")),
                    layout: &self.cascade_input_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: cascade.combine_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &cascade.texture_packed.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                &cascade.texture_packed.sampler,
                            ),
                        },
                    ],
                });
        }

        if new_count < old_count {
            self.cascades.truncate(new_count);
        } else {
            for cascade_index in old_count..new_count {
                let cascade =
                    self.create_cascade_resource(cascade_index, &twiddle_buffer, total_size);
                self.cascades.push(cascade);
            }
        }

        self.combined_render_bind_group_ping =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.height_field_render_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_ping.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            &self.combined_texture_ping.sampler,
                        ),
                    },
                ],
                label: Some("global_combined_render_ping"),
            });

        self.combined_render_bind_group_pong =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.height_field_render_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_pong.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            &self.combined_texture_pong.sampler,
                        ),
                    },
                ],
                label: Some("global_combined_render_pong"),
            });

        self.combined_output_is_ping = new_count.is_multiple_of(2);

        // re-create mesh
        let (verticies, indices) = Vertex::generate_plane(
            &self.ocean_settings_uniform.mesh_size,
            self.ocean_settings_uniform.mesh_subdivisions,
        );

        self.vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex_buffer"),
                contents: bytemuck::cast_slice(&verticies),
                usage: wgpu::BufferUsages::VERTEX,
            });

        self.index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("index_buffer_reinit"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        self.num_indices = indices.len() as u32;
    }

    pub fn create_bind_group(
        device: &wgpu::Device,
        fft_group_layout: &wgpu::BindGroupLayout,
        fft_config_buffer: &wgpu::Buffer,
        fft_uniform_size: u64,
        offset: u64,
        src_h: &TextureView,
        dst_h: &TextureView,
        src_z: &TextureView,
        dst_z: &TextureView,
        packed_output: &TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: fft_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: fft_config_buffer,
                        offset,
                        size: wgpu::BufferSize::new(fft_uniform_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src_h),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(dst_h),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(src_z),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(dst_z),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(packed_output),
                },
            ],
            label: None,
        })
    }

    pub fn populate_fft_bind_groups(
        device: &wgpu::Device,
        ocean_settings_uniform: &OceanSettingsUniform,
        fft_group_layout: &wgpu::BindGroupLayout,
        fft_config_buffer: &wgpu::Buffer,
        fft_texture_ping_h_dx: &TextureInstance,
        fft_texture_pong_h_dx: &TextureInstance,
        fft_texture_ping_dz: &TextureInstance,
        fft_texture_pong_dz: &TextureInstance,
        packed_output: &TextureInstance,
        fft_uniform_size: u64,
        step_size: u64,
    ) -> (Vec<wgpu::BindGroup>, Vec<wgpu::BindGroup>) {
        let mut bind_groups_ping: Vec<wgpu::BindGroup> = Vec::new();
        let mut bind_groups_pong: Vec<wgpu::BindGroup> = Vec::new();

        for i in 0..(ocean_settings_uniform.pass_num * 2) {
            let offset = i as u64 * step_size;

            bind_groups_ping.push(State::create_bind_group(
                device,
                fft_group_layout,
                fft_config_buffer,
                fft_uniform_size,
                offset,
                &fft_texture_ping_h_dx.view,
                &fft_texture_pong_h_dx.view,
                &fft_texture_ping_dz.view,
                &fft_texture_pong_dz.view,
                &packed_output.view,
            ));

            bind_groups_pong.push(State::create_bind_group(
                device,
                fft_group_layout,
                fft_config_buffer,
                fft_uniform_size,
                offset,
                &fft_texture_pong_h_dx.view,
                &fft_texture_ping_h_dx.view,
                &fft_texture_pong_dz.view,
                &fft_texture_ping_dz.view,
                &packed_output.view,
            ));
        }

        (bind_groups_ping, bind_groups_pong)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InitialData {
    pub k_vec: [f32; 2],
    pub initial_freq_domain: [f32; 2],
    pub angular_frequency: f32,
    pub _padding: [f32; 3],
}

impl InitialData {
    pub fn new(
        n: u32,
        m: u32,
        fft_size: f32,
        subdivisions: u32,
        wind_vector: [f32; 2],
        l_small: f32,
        amplitude: f32,
        max_w: f32,
        rng: &mut StdRng,
    ) -> Self {
        let n_f = n as f32;
        let m_f = m as f32;
        let size_f = subdivisions as f32;

        let k_x = (2.0 * PI * (n_f - size_f / 2.0)) / fft_size;
        let k_y = (2.0 * PI * (m_f - size_f / 2.0)) / fft_size;
        let k_vec = [k_x, k_y];

        if k_vec == [0.0, 0.0] {
            return Self {
                k_vec,
                initial_freq_domain: [0.0, 0.0],
                angular_frequency: 0.0,
                _padding: [0.0; 3],
            };
        }

        let phk = Self::get_phillips_spectrum_value(k_vec, wind_vector, l_small, amplitude, max_w);

        // Now using a propper gaussian distribution
        let xi_r = Self::box_muller(rng.random::<f32>().max(1e-6), rng.random::<f32>());
        let xi_i = Self::box_muller(rng.random::<f32>().max(1e-6), rng.random::<f32>());

        let sqrt_ph = (phk / 2.0).sqrt();
        let real = sqrt_ph * xi_r;
        let imag = sqrt_ph * xi_i;

        let freq_domain = [real, imag];

        let k: Vector2<f32> = k_vec.into();
        let k_len = k.magnitude();
        let w = if k_len > 0.0001 {
            (9.81 * k_len).sqrt()
        } else {
            0.0
        };

        Self {
            k_vec,
            initial_freq_domain: freq_domain,
            angular_frequency: w,
            _padding: [0.0; 3],
        }
    }

    pub fn box_muller(u1: f32, u2: f32) -> f32 {
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn generate_data(
        fft_size: f32,
        subdivisions: u32,
        wind_vector: [f32; 2],
        l_small: f32,
        amplitude: f32,
        max_w: f32,
        seed: u32,
    ) -> Vec<Self> {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut array: Vec<Self> = Vec::new();

        for n in 0..subdivisions {
            for m in 0..subdivisions {
                array.push(Self::new(
                    n,
                    m,
                    fft_size,
                    subdivisions,
                    wind_vector,
                    l_small,
                    amplitude,
                    max_w,
                    &mut rng,
                ));
            }
        }

        array
    }

    pub fn generate_twiddle_factors(fft_subdivisions: u32) -> Vec<[f32; 2]> {
        let max_stages = fft_subdivisions.ilog2();
        let mut twiddles = Vec::<[f32; 2]>::with_capacity((fft_subdivisions - 1) as usize);

        for stage in 0..max_stages {
            let s = 1u32 << stage;
            for offset in 0..s {
                let angle = 2.0 * std::f32::consts::PI * (offset as f32) / (2.0 * s as f32);
                twiddles.push([angle.cos(), angle.sin()])
            }
        }
        twiddles
    }

    pub fn get_phillips_spectrum_value(
        k_vec: [f32; 2],
        wind_vector: [f32; 2],
        l_small: f32,
        amplitude: f32,
        max_w: f32,
    ) -> f32 {
        let k: Vector2<f32> = k_vec.into();
        let k_len = k.magnitude();
        if k_len > max_w || k_len < 0.001 {
            return 0.0;
        }

        let k2 = k_len * k_len;
        let k4 = k2 * k2;

        let k_hat = k.normalize();
        let w: Vector2<f32> = wind_vector.into();
        let w_len = w.magnitude();
        let w_hat = w.normalize();

        let align = cgmath::dot(k_hat, w_hat);
        let align2 = if align > 0.0 {
            align.powi(2)
        } else {
            align.powi(2) * 0.07
        };

        let l = w_len * w_len / 9.81;
        let l2 = l * l;

        let exp_term = f32::exp(-1.0 / (k2 * l2));
        let damp = f32::exp(-k2 * l_small * l_small);

        (align2 * amplitude * exp_term * damp) / (k4 + 0.001)
    }
}

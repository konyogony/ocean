use crate::settings::OceanSettingsUniform;
use crate::state::CascadeResources;
use crate::state::FFTUniform;
use crate::state::State;
use crate::texture::Texture;
use crate::vertex::InitialData;
use wgpu::util::DeviceExt;

impl State {
    pub fn compute_fft(&mut self) {
        // First create the encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fft_multi-pass_encoder"),
            });

        for cascade in &mut self.cascades {
            let passes = (self.ocean_settings_uniform.pass_num * 2) as usize;
            // For the spectrum update
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.spectrum_pipeline);
                pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
                pass.set_bind_group(1, &cascade.bind_groups_pong[0], &[]);
                pass.set_bind_group(2, &self.camera_bind_group, &[]);
                pass.set_bind_group(3, &cascade.initial_data_group, &[]);
                pass.dispatch_workgroups(
                    &self.ocean_settings_uniform.fft_subdivisions / 16,
                    &self.ocean_settings_uniform.fft_subdivisions / 16,
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
                    &self.ocean_settings_uniform.fft_subdivisions / 16,
                    &self.ocean_settings_uniform.fft_subdivisions / 16,
                    1,
                );

                current_reads_from_ping = !current_reads_from_ping;
            }

            cascade.output_is_ping = !passes.is_multiple_of(2);
            // if !cascade.output_is_ping {
            //     println!("output in B");
            // }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
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

        let texture_ping_h_dx = Texture::create_storage_texture(
            &self.device,
            n,
            &format!("fft_texture_ping_h_dx_{cascade_index}"),
        );
        let texture_pong_h_dx = Texture::create_storage_texture(
            &self.device,
            n,
            &format!("fft_texture_pong_h_dx_{cascade_index}"),
        );
        let texture_ping_dz = Texture::create_storage_texture(
            &self.device,
            n,
            &format!("fft_texture_ping_dz_{cascade_index}"),
        );
        let texture_pong_dz = Texture::create_storage_texture(
            &self.device,
            n,
            &format!("fft_texture_pong_dz_{cascade_index}"),
        );

        let config_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_config_buffer"),
            size: total_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for stage in 0..pass_num {
            for is_vertical in 0..2 {
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
                        resource: wgpu::BindingResource::TextureView(&texture_ping_dz.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
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
                        resource: wgpu::BindingResource::TextureView(&texture_pong_dz.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
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

        let combined_bind_group_ping = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.combined_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_ping_h_dx.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_ping_dz.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &self.combined_texture_ping_h_dx.view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &self.combined_texture_ping_dz.view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        &self.combined_texture_pong_h_dx.view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(
                        &self.combined_texture_pong_dz.view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(
                        &self.combined_texture_ping_h_dx.sampler,
                    ),
                },
            ],
            label: Some(&format!("combined_bind_group_ping_{cascade_index}")),
        });

        let combined_bind_group_accumulate =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.combined_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_ping_h_dx.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_ping_dz.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_pong_h_dx.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_pong_dz.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_ping_h_dx.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_ping_dz.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(
                            &self.combined_texture_ping_h_dx.sampler,
                        ),
                    },
                ],
                label: Some(&format!("combined_bind_group_accumulate_{cascade_index}")),
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
            initial_data_group,
            initial_data_buffer,
            combined_bind_group_ping,
            combined_bind_group_accumulate,
        }
    }

    pub fn reinit_fft_resources(&mut self) {
        let n = self.ocean_settings_uniform.fft_subdivisions;
        let pass_num = self.ocean_settings_uniform.pass_num;
        let total_size = self.step_size * pass_num as u64 * 2;

        let new_count = self.ocean_settings_uniform.cascade_count as usize;
        let old_count = self.cascades.len();

        // constant across cascades
        let twiddle_array = InitialData::generate_twiddle_factors(n);
        let twiddle_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("twiddle_buffer"),
                contents: bytemuck::cast_slice(&twiddle_array),
                usage: wgpu::BufferUsages::STORAGE,
            });

        for (cascade_index, cascade) in self.cascades.iter_mut().enumerate() {
            cascade.texture_ping_h_dx = Texture::create_storage_texture(
                &self.device,
                n,
                &format!("fft_texture_ping_h_dx_{cascade_index}"),
            );
            cascade.texture_pong_h_dx = Texture::create_storage_texture(
                &self.device,
                n,
                &format!("fft_texture_pong_h_dx_{cascade_index}"),
            );
            cascade.texture_ping_dz = Texture::create_storage_texture(
                &self.device,
                n,
                &format!("fft_texture_ping_dz_{cascade_index}"),
            );
            cascade.texture_pong_dz = Texture::create_storage_texture(
                &self.device,
                n,
                &format!("fft_texture_pong_dz_{cascade_index}"),
            );

            cascade.config_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fft_config_buffer"),
                size: total_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            for stage in 0..pass_num {
                for is_vertical in 0..2 {
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
                            resource: wgpu::BindingResource::TextureView(
                                &cascade.texture_ping_dz.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
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
                            resource: wgpu::BindingResource::TextureView(
                                &cascade.texture_pong_dz.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
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

            cascade.combined_bind_group_ping =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("combined_bind_group_ping_{cascade_index}")),
                    layout: &self.combined_bind_group_layout,
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
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &self.combined_texture_ping_h_dx.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(
                                &self.combined_texture_ping_dz.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(
                                &self.combined_texture_pong_h_dx.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(
                                &self.combined_texture_pong_dz.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::Sampler(
                                &self.combined_texture_ping_h_dx.sampler,
                            ),
                        },
                    ],
                });

            cascade.combined_bind_group_accumulate =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("combined_bind_group_accumulate_{cascade_index}")),
                    layout: &self.combined_bind_group_layout,
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
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &self.combined_texture_pong_h_dx.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(
                                &self.combined_texture_pong_dz.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(
                                &self.combined_texture_ping_h_dx.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(
                                &self.combined_texture_ping_dz.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::Sampler(
                                &self.combined_texture_ping_h_dx.sampler,
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

        self.combined_output_is_ping = new_count % 2 == 1;
    }

    pub fn populate_fft_bind_groups(
        device: &wgpu::Device,
        ocean_settings_uniform: &OceanSettingsUniform,
        fft_group_layout: &wgpu::BindGroupLayout,
        fft_config_buffer: &wgpu::Buffer,
        fft_texture_ping_h_dx: &Texture,
        fft_texture_pong_h_dx: &Texture,
        fft_texture_ping_dz: &Texture,
        fft_texture_pong_dz: &Texture,
        fft_uniform_size: u64,
        step_size: u64,
    ) -> (Vec<wgpu::BindGroup>, Vec<wgpu::BindGroup>) {
        let mut bind_groups_ping: Vec<wgpu::BindGroup> = Vec::new();
        let mut bind_groups_pong: Vec<wgpu::BindGroup> = Vec::new();

        for i in 0..(ocean_settings_uniform.pass_num * 2) {
            let offset = i as u64 * step_size;

            // Read Ping -> Write Pong
            bind_groups_ping.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                        resource: wgpu::BindingResource::TextureView(&fft_texture_ping_h_dx.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&fft_texture_pong_h_dx.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&fft_texture_ping_dz.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&fft_texture_pong_dz.view),
                    },
                ],
                label: None,
            }));

            // Read Pong -> Write Ping
            bind_groups_pong.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                        resource: wgpu::BindingResource::TextureView(&fft_texture_pong_h_dx.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&fft_texture_ping_h_dx.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&fft_texture_pong_dz.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&fft_texture_ping_dz.view),
                    },
                ],
                label: None,
            }));
        }

        (bind_groups_ping, bind_groups_pong)
    }

    pub fn combine_cascades(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("combine_cascades_encoder"),
            });

        // clearing it first
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.combined_clear_pipeline);
            pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
            // temporary bind group for clearing
            let clear_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clear_pong_bind_group"),
                layout: &self.combined_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_ping_h_dx.view, // dummy
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_ping_dz.view, // dummy
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_ping_h_dx.view, // dummy
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_ping_dz.view, // dummy
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_pong_h_dx.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_pong_dz.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(
                            &self.combined_texture_ping_h_dx.sampler,
                        ),
                    },
                ],
            });
            pass.set_bind_group(1, &clear_bind_group, &[]);
            pass.dispatch_workgroups(
                &self.ocean_settings_uniform.fft_subdivisions / 16,
                &self.ocean_settings_uniform.fft_subdivisions / 16,
                1,
            );
        }

        let mut current_output_is_ping = true;

        for cascade in &mut self.cascades {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.combined_cascade_pipeline);
            pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);

            let bind_group = if current_output_is_ping == cascade.output_is_ping {
                &cascade.combined_bind_group_ping
            } else {
                &cascade.combined_bind_group_accumulate
            };

            pass.set_bind_group(1, bind_group, &[]);
            pass.dispatch_workgroups(
                self.ocean_settings_uniform.fft_subdivisions / 16,
                self.ocean_settings_uniform.fft_subdivisions / 16,
                1,
            );
            current_output_is_ping = !current_output_is_ping;
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.combined_output_is_ping =
            (self.ocean_settings_uniform.cascade_count as usize) % 2 == 1;
    }
}

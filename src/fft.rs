use crate::settings::OceanSettingsUniform;
use crate::settings::MAX_CASCADES;
use crate::settings::TOTAL_BINDINGS;
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

        let cascade_count = self.ocean_settings_uniform.cascade_count as usize;

        for c in 0..cascade_count {
            // For the spectrum update
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.spectrum_pipeline);
                pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
                pass.set_bind_group(1, &self.fft_bind_groups_pong[0], &[]);
                pass.set_bind_group(2, &self.camera_bind_group, &[]);
                pass.set_bind_group(3, &self.initial_data_groups[c], &[]);
                pass.dispatch_workgroups(
                    &self.ocean_settings_uniform.fft_subdivisions / 16,
                    &self.ocean_settings_uniform.fft_subdivisions / 16,
                    1,
                );
            }

            let mut current_is_ping = true;

            for i in 0..(self.ocean_settings_uniform.pass_num * 2) as usize {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.fft_compute_pipeline);

                let bind_group = if current_is_ping {
                    &self.fft_bind_groups_ping[i]
                } else {
                    &self.fft_bind_groups_pong[i]
                };

                pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
                pass.set_bind_group(1, bind_group, &[]);
                pass.set_bind_group(2, &self.camera_bind_group, &[]);
                pass.set_bind_group(3, &self.initial_data_groups[c], &[]);
                pass.dispatch_workgroups(
                    &self.ocean_settings_uniform.fft_subdivisions / 16,
                    &self.ocean_settings_uniform.fft_subdivisions / 16,
                    1,
                );

                current_is_ping = !current_is_ping;
            }

            self.fft_output_is_a[c] = current_is_ping;
            if !self.fft_output_is_a[c] {
                println!("output in B");
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn reinit_fft_resources(&mut self) {
        let n = self.ocean_settings_uniform.fft_subdivisions;
        let pass_num = self.ocean_settings_uniform.pass_num;

        for c in 0..MAX_CASCADES {
            self.fft_textures_ping_h_dx[c] = Texture::create_storage_texture(
                &self.device,
                n,
                &format!("fft_texture_ping_h_dx_{c}"),
            );
            self.fft_textures_pong_h_dx[c] = Texture::create_storage_texture(
                &self.device,
                n,
                &format!("fft_texture_pong_h_dx_{c}"),
            );
            self.fft_textures_ping_dz[c] = Texture::create_storage_texture(
                &self.device,
                n,
                &format!("fft_texture_ping_dz_{c}"),
            );
            self.fft_textures_pong_dz[c] = Texture::create_storage_texture(
                &self.device,
                n,
                &format!("fft_texture_pong_dz_{c}"),
            );
        }

        let total_size = self.step_size * pass_num as u64 * 2;
        self.fft_config_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_config_buffer"),
            size: total_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for c in 0..MAX_CASCADES {
            for stage in 0..pass_num {
                for is_vertical in 0..2 {
                    let i = stage * 2 + is_vertical;
                    let uniform = FFTUniform {
                        stage,
                        is_vertical,
                        cascade_index: c as u32,
                        _pad: 0,
                    };
                    let offset = i as u64 * self.step_size;
                    self.queue.write_buffer(
                        &self.fft_config_buffer,
                        offset,
                        bytemuck::cast_slice(&[uniform]),
                    );
                }
            }
        }

        let initial_data_arrays: [_; MAX_CASCADES] = std::array::from_fn(|i| {
            InitialData::generate_data(
                self.ocean_settings_uniform.cascade_sizes[i],
                n,
                self.ocean_settings_uniform.wind_vector,
                self.ocean_settings_uniform.l_small,
                self.ocean_settings_uniform.cascade_amplitudes[i],
                self.ocean_settings_uniform.max_w,
                self.ocean_settings_uniform.ocean_seed,
            )
        });

        for c in 0..MAX_CASCADES {
            self.initial_data_buffers[c] =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("initial_data_buffer_{c}")),
                        contents: bytemuck::cast_slice(&initial_data_arrays[c]),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    });
        }

        let twiddle_arrays: [_; MAX_CASCADES] =
            std::array::from_fn(|_| InitialData::generate_twiddle_factors(n));
        let twiddle_buffers: [_; MAX_CASCADES] = std::array::from_fn(|i| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("twiddle_buffer_{i}")),
                    contents: bytemuck::cast_slice(&twiddle_arrays[i]),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        });

        for c in 0..MAX_CASCADES {
            self.initial_data_groups[c] =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.initial_data_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.initial_data_buffers[c].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: twiddle_buffers[c].as_entire_binding(),
                        },
                    ],
                    label: None,
                });
        }

        let (fft_bind_groups_ping, fft_bind_groups_pong) = Self::populate_fft_bind_groups(
            &self.device,
            &self.ocean_settings_uniform,
            &self.fft_group_layout,
            &self.fft_config_buffer,
            &self.fft_textures_ping_h_dx,
            &self.fft_textures_pong_h_dx,
            &self.fft_textures_ping_dz,
            &self.fft_textures_pong_dz,
            self.fft_uniform_size,
            self.step_size,
        );

        self.fft_bind_groups_ping = fft_bind_groups_ping;
        self.fft_bind_groups_pong = fft_bind_groups_pong;

        for c in 0..MAX_CASCADES {
            self.height_field_render_bind_groups_ping[c] =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.height_field_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.fft_textures_ping_h_dx[c].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.fft_textures_ping_dz[c].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                &self.fft_textures_ping_h_dx[c].sampler,
                            ),
                        },
                    ],
                    label: None,
                });

            self.height_field_render_bind_groups_pong[c] =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &self.height_field_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.fft_textures_pong_h_dx[c].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.fft_textures_pong_dz[c].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                &self.fft_textures_pong_h_dx[c].sampler,
                            ),
                        },
                    ],
                    label: None,
                });

            self.height_field_compute_bind_groups_ping[c] =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("height_field_compute_bind_group_ping_{c}")),
                    layout: &self.height_field_compute_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.fft_textures_ping_h_dx[c].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.fft_textures_ping_dz[c].view,
                            ),
                        },
                    ],
                });

            self.height_field_compute_bind_groups_pong[c] =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("height_field_compute_bind_group_pong_{c}")),
                    layout: &self.height_field_compute_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.fft_textures_pong_h_dx[c].view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.fft_textures_pong_dz[c].view,
                            ),
                        },
                    ],
                });
        }
    }

    pub fn populate_fft_bind_groups(
        device: &wgpu::Device,
        ocean_settings_uniform: &OceanSettingsUniform,
        fft_group_layout: &wgpu::BindGroupLayout,
        fft_config_buffer: &wgpu::Buffer,
        fft_textures_ping_h_dx: &[Texture; MAX_CASCADES],
        fft_textures_pong_h_dx: &[Texture; MAX_CASCADES],
        fft_textures_ping_dz: &[Texture; MAX_CASCADES],
        fft_textures_pong_dz: &[Texture; MAX_CASCADES],
        fft_uniform_size: u64,
        step_size: u64,
    ) -> (Vec<wgpu::BindGroup>, Vec<wgpu::BindGroup>) {
        let mut fft_bind_groups_ping: Vec<wgpu::BindGroup> = Vec::new();
        let mut fft_bind_groups_pong: Vec<wgpu::BindGroup> = Vec::new();

        for i in 0..(ocean_settings_uniform.pass_num * 2) {
            let offset = i as u64 * step_size;

            // Read Ping -> Write Pong
            fft_bind_groups_ping.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: fft_group_layout,
                entries: &std::array::from_fn::<wgpu::BindGroupEntry, TOTAL_BINDINGS, _>(|i| {
                    if i == 0 {
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: fft_config_buffer,
                                offset,
                                size: wgpu::BufferSize::new(fft_uniform_size),
                            }),
                        }
                    } else {
                        let cascade_index = (i - 1) / 4;
                        let local = (i - 1) % 4;
                        let texture = match local {
                            0 => &fft_textures_ping_h_dx[cascade_index],
                            1 => &fft_textures_pong_h_dx[cascade_index],
                            2 => &fft_textures_ping_dz[cascade_index],
                            3 => &fft_textures_pong_dz[cascade_index],
                            _ => unreachable!(),
                        };
                        wgpu::BindGroupEntry {
                            binding: i as u32,
                            resource: wgpu::BindingResource::TextureView(&texture.view),
                        }
                    }
                }),
                label: None,
            }));

            // Read Pong -> Write Ping
            fft_bind_groups_pong.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &fft_group_layout,
                entries: &std::array::from_fn::<wgpu::BindGroupEntry, TOTAL_BINDINGS, _>(|i| {
                    if i == 0 {
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: fft_config_buffer,
                                offset,
                                size: wgpu::BufferSize::new(fft_uniform_size),
                            }),
                        }
                    } else {
                        let cascade_index = (i - 1) / 4;
                        let local = (i - 1) % 4;
                        let texture = match local {
                            0 => &fft_textures_pong_h_dx[cascade_index],
                            1 => &fft_textures_ping_h_dx[cascade_index],
                            2 => &fft_textures_pong_dz[cascade_index],
                            3 => &fft_textures_ping_dz[cascade_index],
                            _ => unreachable!(),
                        };
                        wgpu::BindGroupEntry {
                            binding: i as u32,
                            resource: wgpu::BindingResource::TextureView(&texture.view),
                        }
                    }
                }),
                label: None,
            }));
        }

        (fft_bind_groups_ping, fft_bind_groups_pong)
    }
}

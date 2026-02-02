use crate::settings::OceanSettingsUniform;
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
                label: Some("FFT Multi-Pass Encoder"),
            });

        // For the spectrum update
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.spectrum_pipeline);
            pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
            pass.set_bind_group(1, &self.fft_bind_groups_pong[0], &[]);
            pass.set_bind_group(2, &self.camera_bind_group, &[]);
            pass.set_bind_group(3, &self.initial_data_group, &[]);
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
            pass.set_bind_group(3, &self.initial_data_group, &[]);
            pass.dispatch_workgroups(
                &self.ocean_settings_uniform.fft_subdivisions / 16,
                &self.ocean_settings_uniform.fft_subdivisions / 16,
                1,
            );

            current_is_ping = !current_is_ping;
        }

        self.fft_output_is_a = current_is_ping;
        if !self.fft_output_is_a {
            println!("output in B");
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn reinit_fft_resources(&mut self) {
        let n = self.ocean_settings_uniform.fft_subdivisions;
        let pass_num = self.ocean_settings_uniform.pass_num;

        self.fft_texture_ping_h_dx =
            Texture::create_storage_texture(&self.device, n, "fft_texture_ping_h_dx");
        self.fft_texture_pong_h_dx =
            Texture::create_storage_texture(&self.device, n, "fft_texture_pong_h_dx");
        self.fft_texture_ping_dz =
            Texture::create_storage_texture(&self.device, n, "fft_texture_ping_dz");
        self.fft_texture_pong_dz =
            Texture::create_storage_texture(&self.device, n, "fft_texture_pong_dz");

        let total_size = self.step_size * pass_num as u64 * 2;
        self.fft_config_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_config_buffer"),
            size: total_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for stage in 0..pass_num {
            for is_vertical in 0..2 {
                let i = stage * 2 + is_vertical;
                let uniform = FFTUniform { stage, is_vertical };
                let offset = i as u64 * self.step_size;
                self.queue.write_buffer(
                    &self.fft_config_buffer,
                    offset,
                    bytemuck::cast_slice(&[uniform]),
                );
            }
        }

        let (initial_data_array, _, _) = InitialData::generate_data(
            self.ocean_settings_uniform.fft_size,
            n,
            self.ocean_settings_uniform.wind_vector,
            self.ocean_settings_uniform.l_small,
            self.ocean_settings_uniform.amplitude,
            self.ocean_settings_uniform.max_w,
            self.ocean_settings_uniform.ocean_seed,
        );

        self.initial_data_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Initial Data Buffer"),
                    contents: bytemuck::cast_slice(&initial_data_array),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });

        let twiddle_array = InitialData::generate_twiddle_factors(n);
        let twiddle_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Twiddle Buffer"),
                contents: bytemuck::cast_slice(&twiddle_array),
                usage: wgpu::BufferUsages::STORAGE,
            });

        self.initial_data_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.initial_data_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.initial_data_buffer.as_entire_binding(),
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
            &self.fft_group_layout,
            &self.fft_config_buffer,
            &self.fft_texture_ping_h_dx,
            &self.fft_texture_pong_h_dx,
            &self.fft_texture_ping_dz,
            &self.fft_texture_pong_dz,
            self.fft_uniform_size,
            self.step_size,
        );

        self.fft_bind_groups_ping = fft_bind_groups_ping;
        self.fft_bind_groups_pong = fft_bind_groups_pong;

        self.height_field_bind_group_ping =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.height_field_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.fft_texture_ping_h_dx.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.fft_texture_ping_dz.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(
                            &self.fft_texture_ping_h_dx.sampler,
                        ),
                    },
                ],
                label: None,
            });

        self.height_field_bind_group_pong =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.height_field_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.fft_texture_pong_h_dx.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.fft_texture_pong_dz.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(
                            &self.fft_texture_pong_h_dx.sampler,
                        ),
                    },
                ],
                label: None,
            });
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
        let mut fft_bind_groups_ping: Vec<wgpu::BindGroup> = Vec::new();
        let mut fft_bind_groups_pong: Vec<wgpu::BindGroup> = Vec::new();

        for i in 0..(ocean_settings_uniform.pass_num * 2) {
            let offset = i as u64 * step_size;

            // Read Ping -> Write Pong
            fft_bind_groups_ping.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            fft_bind_groups_pong.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &fft_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &fft_config_buffer,
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

        (fft_bind_groups_ping, fft_bind_groups_pong)
    }
}

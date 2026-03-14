use crate::pipeline::state::CascadeResources;
use crate::pipeline::state::FFTUniform;
use crate::pipeline::state::State;
use crate::texture::instance::TextureInstance;
use cgmath::{InnerSpace, Vector2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f32::consts::PI;
use wgpu::util::DeviceExt;

impl State {
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

        let texture_ping_h_dx = TextureInstance::create_storage_texture(
            &self.device,
            n,
            &format!("fft_texture_ping_h_dx_{cascade_index}"),
        );
        let texture_pong_h_dx = TextureInstance::create_storage_texture(
            &self.device,
            n,
            &format!("fft_texture_pong_h_dx_{cascade_index}"),
        );
        let texture_ping_dz = TextureInstance::create_storage_texture(
            &self.device,
            n,
            &format!("fft_texture_ping_dz_{cascade_index}"),
        );
        let texture_pong_dz = TextureInstance::create_storage_texture(
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

        let combined_bind_group_ping_accumulate =
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
                label: Some(&format!(
                    "combined_bind_group_ping_accumulate_{cascade_index}"
                )),
            });

        let combined_bind_group_pong = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.combined_bind_group_layout,
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
            label: Some(&format!("combined_bind_group_pong_{cascade_index}")),
        });

        let combined_bind_group_pong_accumulate =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.combined_bind_group_layout,
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
                label: Some(&format!(
                    "combined_bind_group_pong_accumulate_{cascade_index}"
                )),
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
            combined_bind_group_ping_accumulate,
            combined_bind_group_pong,
            combined_bind_group_pong_accumulate,
        }
    }

    pub fn combine_cascades(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("combine_cascades_encoder"),
            });

        // clearing both combined textures first
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.combined_clear_pipeline);
            pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
            let clear_pong_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clear_pong_bind_group"),
                layout: &self.combined_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_ping_h_dx.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_ping_dz.view,
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
            pass.set_bind_group(1, &clear_pong_bind_group, &[]);
            pass.dispatch_workgroups(
                &self.ocean_settings_uniform.fft_subdivisions / 16,
                &self.ocean_settings_uniform.fft_subdivisions / 16,
                1,
            );

            let clear_ping_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clear_ping_bind_group"),
                layout: &self.combined_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_pong_h_dx.view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.combined_texture_pong_dz.view,
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
            pass.set_bind_group(1, &clear_ping_bind_group, &[]);
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

            let bind_group = match (cascade.output_is_ping, current_output_is_ping) {
                (true, true) => &cascade.combined_bind_group_ping,
                (true, false) => &cascade.combined_bind_group_ping_accumulate,
                (false, true) => &cascade.combined_bind_group_pong,
                (false, false) => &cascade.combined_bind_group_pong_accumulate,
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

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InitialData {
    pub k_vec: [f32; 2],
    // Supposed to be a complex number
    pub initial_freq_domain: [f32; 2],
    pub initial_freq_domain_conjugate: [f32; 2],
    pub angular_frequency: f32,
    // To get to 32 bytes
    pub _padding: f32,
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
        // Now we have to do this, so that we can center everything around the center.
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
                initial_freq_domain_conjugate: [0.0, 0.0],
                angular_frequency: 0.0,
                _padding: 0.0,
            };
        }

        let phk = Self::get_phillips_spectrum_value(k_vec, wind_vector, l_small, amplitude, max_w);

        let xi_r: f32 = rng.random::<f32>() * 2.0 - 1.0;
        let xi_i: f32 = rng.random::<f32>() * 2.0 - 1.0;

        let sqrt_ph = (phk / 2.0).sqrt();
        let real = sqrt_ph * xi_r;
        let imag = sqrt_ph * xi_i;

        let freq_domain = [real, imag];
        let freq_domain_conjugate = [real, -imag];

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
            initial_freq_domain_conjugate: freq_domain_conjugate,
            angular_frequency: w,
            _padding: 0.0,
        }
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
        // not killing them.
        let align2 = if align > 0.0 {
            align.powi(2)
        } else {
            align.powi(2) * 0.07
        };

        let l = w_len * w_len / 9.81;
        let l2 = l * l;

        let exp_term = f32::exp(-1.0 / (k2 * l2));
        let damp = f32::exp(-k2 * l_small * l_small);
        //    let small_wave_suppression = 1.0 / (1.0 + k_len * 0.5);

        (align2 * amplitude * exp_term * damp) / (k4 + 0.001)
    }
}

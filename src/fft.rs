use crate::state::State;

impl State {
    pub fn compute_fft(&mut self) {
        // First create the encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FFT Multi-Pass Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.spectrum_pipeline);
            pass.set_bind_group(0, &self.ocean_settings_bind_group, &[]);
            pass.set_bind_group(1, &self.fft_bind_groups_b[0], &[]); // Buffer A will have it.
            pass.set_bind_group(2, &self.camera_bind_group, &[]);
            pass.set_bind_group(3, &self.initial_data_group, &[]);
            pass.dispatch_workgroups(
                &self.ocean_settings_uniform.fft_subdivisions / 16,
                &self.ocean_settings_uniform.fft_subdivisions / 16,
                1,
            );
        }

        let mut read_from_a = true;

        for i in 0..(self.ocean_settings_uniform.pass_num * 2) as usize {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.fft_compute_pipeline);

            let bind_group = if read_from_a {
                &self.fft_bind_groups_a[i]
            } else {
                &self.fft_bind_groups_b[i]
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

            read_from_a = !read_from_a;
        }

        self.fft_output_is_a = read_from_a;
        if !self.fft_output_is_a {
            println!("output in B");
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn debug_fft_values_sync(&mut self) {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT Debug Staging"),
            size: 4096 * 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        let subdivisions = self.ocean_settings_uniform.fft_subdivisions as u64;
        let buffer_size = subdivisions * subdivisions * 16;
        encoder.copy_buffer_to_buffer(&self.fft_buffer_a, 0, &staging_buffer, 0, buffer_size);
        self.queue.submit(Some(encoder.finish()));

        self.fft_min = -self.max_magnitude;
        self.fft_max = self.max_magnitude;
        self.fft_avg = self.avg_magnitude;
        self.fft_samples_checked = 4096;
    }
}

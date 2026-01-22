#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OceanSettings {
    pub mesh_size: f32,         // 1024.0
    pub mesh_subdivisions: u32, // 2048
    pub fft_size: f32,          // 1024.0
    pub fft_subdivisions: u32,  // 1024
    pub pass_num: u32,          // log2(fft_subdivisions)
    pub time_scale: f32,        // 10.0
    pub chop_scale: f32,        // 2.5
    pub amplitude_scale: f32,   // 1.0
    pub wave_scale: f32,        // mesh_size / fft_size
    pub wind_vector: [f32; 2],  // (30.0, 20.0)
    pub amplitude: f32,         // 10.0
    pub l_small: f32,           // 0.001
    pub max_w: f32,             // 100.0
    pub fovy: f32,              // 60.0
    pub zfar: f32,              // 1500.0
    pub cam_speed: f32,         // 0.5
    pub cam_boost: f32,         // 5.0
    pub cam_sensitivity: f32,   // 0.002
    pub _padding: f32,
}

pub struct OceanSettingsBuilder {
    mesh_size: f32,
    mesh_subdivisions: u32,
    fft_size: f32,
    fft_subdivisions: u32,
    time_scale: f32,
    chop_scale: f32,
    amplitude_scale: f32,
    wind_vector: [f32; 2],
    amplitude: f32,
    l_small: f32,
    max_w: f32,
    fovy: f32,
    zfar: f32,
    cam_speed: f32,
    cam_boost: f32,
    cam_sensitivity: f32,
}

impl Default for OceanSettingsBuilder {
    fn default() -> Self {
        Self {
            mesh_size: 1000.0,
            mesh_subdivisions: 2048,
            fft_size: 1000.0,
            fft_subdivisions: 128,
            time_scale: 1.5,
            chop_scale: 0.75,
            amplitude_scale: 1.0,
            wind_vector: [61.0, -81.0],
            amplitude: 1.5,
            l_small: 0.001,
            max_w: 100.0,
            fovy: 60.0,
            zfar: 1500.0,
            cam_speed: 0.05,
            cam_boost: 5.0,
            cam_sensitivity: 0.002,
        }
    }
}

impl OceanSettingsBuilder {
    pub fn mesh_size(mut self, v: f32) -> Self {
        self.mesh_size = v;
        self
    }

    pub fn mesh_subdivisions(mut self, v: u32) -> Self {
        self.mesh_subdivisions = v;
        self
    }

    pub fn fft_size(mut self, v: f32) -> Self {
        self.fft_size = v;
        self
    }

    pub fn fft_subdivisions(mut self, v: u32) -> Self {
        self.fft_subdivisions = v;
        self
    }

    pub fn time_scale(mut self, v: f32) -> Self {
        self.time_scale = v;
        self
    }

    pub fn chop_scale(mut self, v: f32) -> Self {
        self.chop_scale = v;
        self
    }

    pub fn amplitude_scale(mut self, v: f32) -> Self {
        self.amplitude_scale = v;
        self
    }

    pub fn wind_vector(mut self, v: [f32; 2]) -> Self {
        self.wind_vector = v;
        self
    }

    pub fn amplitude(mut self, v: f32) -> Self {
        self.amplitude = v;
        self
    }

    pub fn l_small(mut self, v: f32) -> Self {
        self.l_small = v;
        self
    }

    pub fn max_w(mut self, v: f32) -> Self {
        self.max_w = v;
        self
    }

    pub fn fovy(mut self, v: f32) -> Self {
        self.fovy = v;
        self
    }

    pub fn zfar(mut self, v: f32) -> Self {
        self.zfar = v;
        self
    }

    pub fn cam_speed(mut self, v: f32) -> Self {
        self.cam_speed = v;
        self
    }

    pub fn cam_boost(mut self, v: f32) -> Self {
        self.cam_boost = v;
        self
    }

    pub fn cam_sensitivity(mut self, v: f32) -> Self {
        self.cam_sensitivity = v;
        self
    }

    pub fn build(self) -> OceanSettings {
        assert!(self.fft_subdivisions.is_power_of_two());
        assert!(self.fft_size > 0.0);
        assert!(self.mesh_size > 0.0);

        let pass_num = self.fft_subdivisions.ilog2();
        let wave_scale = self.mesh_size / self.fft_size;

        OceanSettings {
            mesh_size: self.mesh_size,
            mesh_subdivisions: self.mesh_subdivisions,
            fft_size: self.fft_size,
            fft_subdivisions: self.fft_subdivisions,
            pass_num,
            time_scale: self.time_scale,
            chop_scale: self.chop_scale,
            amplitude_scale: self.amplitude_scale,
            wave_scale,
            wind_vector: self.wind_vector,
            amplitude: self.amplitude,
            l_small: self.l_small,
            max_w: self.max_w,
            fovy: self.fovy,
            zfar: self.zfar,
            cam_speed: self.cam_speed,
            cam_boost: self.cam_boost,
            cam_sensitivity: self.cam_sensitivity,
            _padding: 0.0,
        }
    }
}

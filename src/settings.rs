#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OceanSettings {
    pub mesh_size: f32,         // 1024.0
    pub mesh_subdivisions: u32, // 2048
    pub fft_size: f32,          // 1024.0
    pub fft_subdivisions: u32,  // 32
    pub pass_num: u32,          // log2(fft_subdivisions)
    pub time_scale: f32,        // 10.0
    pub chop_scale: f32,        // 1.25
    pub amplitude_scale: f32,   // 1.0
    pub wave_scale: f32,        // mesh_size / fft_size
    pub _pad0: f32,
    pub wind_vector: [f32; 2],     // (10.0, 10.0)
    pub amplitude: f32,            // 0.6
    pub l_small: f32,              // 0.001
    pub max_w: f32,                // 1000.0
    pub fovy: f32,                 // 60.0
    pub zfar: f32,                 // 1500.0
    pub cam_speed: f32,            // 0.5
    pub cam_boost: f32,            // 5.0
    pub cam_sensitivity: f32,      // 0.002
    pub roughness: f32,            // 0.05
    pub f_0: f32,                  // 0.02
    pub specular_scale: f32,       // 0.5
    pub reflection_scale: f32,     // 0.6
    pub foam_scale: f32,           // 0.8
    pub sss_distortion_scale: f32, // 0.02
    pub _pad1: [f32; 2],
    pub deep_color: [f32; 4],    // vec3<f32>(0.0, 0.01, 0.05)
    pub shallow_color: [f32; 4], // vec3<f32>(0.0, 0.06, 0.09)
    pub sss_color: [f32; 4],     // vec3<f32>(0.0, 0.2, 0.15)
    pub sun_color: [f32; 4],     // vec3<f32>(1.0, 0.9, 0.8)
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
    roughness: f32,
    f_0: f32,
    specular_scale: f32,
    reflection_scale: f32,
    foam_scale: f32,
    sss_distortion_scale: f32,
    deep_color: [f32; 4],
    shallow_color: [f32; 4],
    sss_color: [f32; 4],
    sun_color: [f32; 4],
}

impl Default for OceanSettingsBuilder {
    fn default() -> Self {
        Self {
            mesh_size: 1000.0,
            mesh_subdivisions: 2048,
            fft_size: 1000.0,
            fft_subdivisions: 64, // Lower number, greater details (like zooming in)
            time_scale: 1.5,
            chop_scale: 1.25,
            amplitude_scale: 1.0,
            wind_vector: [6.0, -8.0],
            amplitude: 0.1,
            l_small: 0.1,
            max_w: 10.0,
            fovy: 60.0,
            zfar: 1500.0,
            cam_speed: 20.0,
            cam_boost: 2.5,
            cam_sensitivity: 0.002,
            roughness: 0.2,
            f_0: 0.02,
            specular_scale: 1.0,
            reflection_scale: 0.2,
            foam_scale: 4.0,
            sss_distortion_scale: 0.2,
            deep_color: [0.0, 0.01, 0.05, 1.0],
            shallow_color: [0.0, 0.06, 0.09, 1.0],
            sss_color: [0.0, 0.4, 0.3, 1.0],
            sun_color: [1.0, 0.9, 0.8, 1.0],
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

    pub fn roughness(mut self, v: f32) -> Self {
        self.roughness = v;
        self
    }

    pub fn f_0(mut self, v: f32) -> Self {
        self.f_0 = v;
        self
    }

    pub fn specular_scale(mut self, v: f32) -> Self {
        self.specular_scale = v;
        self
    }

    pub fn reflection_scale(mut self, v: f32) -> Self {
        self.reflection_scale = v;
        self
    }

    pub fn foam_scale(mut self, v: f32) -> Self {
        self.foam_scale = v;
        self
    }

    pub fn sss_distortion_scale(mut self, v: f32) -> Self {
        self.sss_distortion_scale = v;
        self
    }

    pub fn deep_color(mut self, v: [f32; 4]) -> Self {
        self.deep_color = v;
        self
    }

    pub fn shallow_color(mut self, v: [f32; 4]) -> Self {
        self.shallow_color = v;
        self
    }

    pub fn sss_color(mut self, v: [f32; 4]) -> Self {
        self.sss_color = v;
        self
    }

    pub fn sun_color(mut self, v: [f32; 4]) -> Self {
        self.sun_color = v;
        self
    }

    pub fn rogue() -> Self {
        Self::default()
            .fft_subdivisions(64)
            .time_scale(1.5)
            .chop_scale(1.2)
            .amplitude_scale(1.5)
            .wind_vector([18.0, 14.0])
            .amplitude(0.45)
            .l_small(20.0)
            .max_w(30.0)
            .foam_scale(5.0)
            .roughness(0.25)
            .deep_color([0.0, 0.0, 0.01, 1.0])
            .shallow_color([0.01, 0.03, 0.1, 1.0])
            .sss_color([0.0, 0.5, 0.4, 1.0])
            .sun_color([0.3, 0.3, 0.4, 1.0])
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
            roughness: self.roughness,
            f_0: self.f_0,
            specular_scale: self.specular_scale,
            reflection_scale: self.reflection_scale,
            foam_scale: self.foam_scale,
            sss_distortion_scale: self.sss_distortion_scale,
            deep_color: self.deep_color,
            shallow_color: self.shallow_color,
            sss_color: self.sss_color,
            sun_color: self.sun_color,
            _pad0: 0.0,
            _pad1: [0.0; 2],
        }
    }
}

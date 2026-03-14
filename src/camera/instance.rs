use cgmath::{perspective, Deg, InnerSpace, Matrix4, Point3, Rad, Vector3};

const DIRS: [&str; 8] = [
    "N  (-Z)",
    "NE (+X, -Z)",
    "E  (+X)",
    "SE (+X, +Z)",
    "S  (+Z)",
    "SW (-X, +Z)",
    "W  (-X)",
    "NW (-X, -Z)",
];

pub struct CameraInstance {
    pub forward: cgmath::Vector3<f32>,
    pub eye: cgmath::Point3<f32>,
    pub yaw: Rad<f32>,
    pub pitch: Rad<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
    pub flip_y: bool,
    pub bearing: Rad<f32>,
}

impl CameraInstance {
    // Some more linear algebra magick since opengl and wgpu suck
    // Okay I am SO lost in whats going on here, but now my view works PERFECTLY
    pub fn build_wgpu_projection_matrix_rh(&self) -> Matrix4<f32> {
        let mut proj = perspective(Deg(self.fovy), self.aspect, self.znear, self.zfar);
        if self.flip_y {
            proj.y.y = -proj.y.y;
        }
        proj
    }

    pub fn build_view_projection_matrix(&mut self) -> Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        self.forward =
            Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize();

        let target = self.eye + self.forward;

        let view = Matrix4::look_at_rh(self.eye, target, self.up);
        let proj = self.build_wgpu_projection_matrix_rh();
        proj * view
    }

    pub fn build_skybox_view_projection_matrix(&self) -> Matrix4<f32> {
        let target = Point3::new(0.0, 0.0, 0.0) + self.forward;
        let view = Matrix4::look_at_rh(Point3::new(0.0, 0.0, 0.0), target, self.up);
        let proj = self.build_wgpu_projection_matrix_rh();
        proj * view
    }

    // More slightly confusing maths...
    pub fn update_bearing(&mut self) {
        let f = Vector3::new(self.forward.x, 0.0, -self.forward.z).normalize();
        self.bearing = Rad(f32::atan2(f.x, f.z));
    }

    // Smart.
    pub fn bearing_to_compass(bearing_deg: f32) -> &'static str {
        let idx = ((bearing_deg + 22.5) / 45.0).floor() as usize % 8;
        DIRS[idx]
    }

    pub fn update_camera_settings(&mut self, zfar: f32, fovy: f32) {
        self.fovy = fovy;
        self.zfar = zfar;
    }
}

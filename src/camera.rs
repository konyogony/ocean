use cgmath::{perspective, Deg, InnerSpace, Matrix4, Point3, Rad, SquareMatrix, Vector3};
use winit::event::{DeviceEvent, ElementState, KeyEvent, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

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

pub struct Camera {
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

impl Camera {
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

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view_proj_sky: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _padding: u32,
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
            view_proj_sky: cgmath::Matrix4::identity().into(),
            camera_pos: [0.0; 3],
            _padding: 0,
        }
    }

    pub fn update_view_proj(&mut self, camera: &mut Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
        self.view_proj_sky = camera.build_skybox_view_projection_matrix().into();
        camera.update_bearing();
        self.camera_pos = camera.eye.into();
    }
}

// We can also make scroll correlate to the speed boost
pub struct CameraController {
    pub speed: f32,
    pub sensitivity: f32,
    pub speed_boost: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_cntrl_pressed: bool,
    mouse_dx: f32,
    mouse_dy: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32, speed_boost: f32) -> Self {
        Self {
            speed,
            sensitivity,
            speed_boost,
            is_left_pressed: false,
            is_forward_pressed: false,
            is_right_pressed: false,
            is_backward_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            is_cntrl_pressed: false,
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }
    }

    pub fn process_window_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    KeyCode::KeyW => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyS => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyA => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyD => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    KeyCode::Space => {
                        self.is_up_pressed = is_pressed;
                        true
                    }
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                        self.is_down_pressed = is_pressed;
                        true
                    }
                    KeyCode::ControlLeft | KeyCode::ControlRight => {
                        self.is_cntrl_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn process_device_events(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.mouse_dx += delta.0 as f32;
                self.mouse_dy += delta.1 as f32;
                true
            }
            _ => false,
        }
    }

    pub fn update_all_camera_settings(
        &mut self,
        camera: &mut Camera,
        cam_speed: f32,
        cam_boost: f32,
        cam_sensitivity: f32,
        zfar: f32,
        fovy: f32,
    ) {
        camera.update_camera_settings(zfar, fovy);
        self.sensitivity = cam_sensitivity;
        self.speed = cam_speed;
        self.speed_boost = cam_boost;
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: f32) {
        let speed = match self.is_cntrl_pressed {
            true => self.speed * self.speed_boost,
            false => self.speed,
        };

        camera.yaw += Rad(self.mouse_dx) * self.sensitivity;
        camera.pitch += Rad(-self.mouse_dy) * self.sensitivity;

        // Reset em
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;

        // Lock the rotation
        let max_deg = Deg(89.0).into();
        if camera.pitch > max_deg {
            camera.pitch = max_deg
        } else if camera.pitch < -max_deg {
            camera.pitch = -max_deg
        }

        let (sin_yaw, cos_yaw) = camera.yaw.0.sin_cos();

        let forward = cgmath::Vector3::new(cos_yaw, 0.0, sin_yaw).normalize();
        let right = cgmath::Vector3::new(-sin_yaw, 0.0, cos_yaw).normalize();

        if self.is_forward_pressed {
            camera.eye += forward * speed * dt;
        }
        if self.is_backward_pressed {
            camera.eye -= forward * speed * dt;
        }

        if self.is_right_pressed {
            camera.eye += right * speed * dt;
        }
        if self.is_left_pressed {
            camera.eye -= right * speed * dt;
        }

        if self.is_up_pressed {
            camera.eye += camera.up * speed * dt;
        }
        if self.is_down_pressed {
            camera.eye -= camera.up * speed * dt;
        }
    }
}

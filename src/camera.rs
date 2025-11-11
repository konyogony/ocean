use cgmath::{Deg, InnerSpace, Rad, SquareMatrix};
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

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
}

impl Camera {
    // Some more linear algebra magick since opengl and wgpu suck
    pub fn build_wgpu_projection_matrix_rh(&self) -> cgmath::Matrix4<f32> {
        let f = 1.0 / (self.fovy / 2.0).tan();
        cgmath::Matrix4::from_cols(
            cgmath::Vector4::new(f / self.aspect, 0.0, 0.0, 0.0),
            cgmath::Vector4::new(0.0, f, 0.0, 0.0),
            cgmath::Vector4::new(0.0, 0.0, self.zfar / (self.znear - self.zfar), -1.0),
            cgmath::Vector4::new(
                0.0,
                0.0,
                (self.zfar * self.znear) / (self.znear - self.zfar),
                0.0,
            ),
        )
    }

    pub fn build_view_projection_matrix(&mut self) -> cgmath::Matrix4<f32> {
        // Linear algebra magick
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        // We already calculated the direction we are looking here, just have to add it to rest of
        // the struct. Used for blinn phong model
        self.forward =
            cgmath::Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize();

        let target = self.eye + self.forward;

        let view = cgmath::Matrix4::look_at_rh(self.eye, target, self.up);
        // Didnt know you could do this :/
        let proj = self.build_wgpu_projection_matrix_rh();
        proj * view
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view_dir: [f32; 3],
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
            view_dir: [0.0; 3],
            _padding: 0,
        }
    }

    pub fn update_view_proj(&mut self, camera: &mut Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
        self.view_dir = camera.forward.into();
    }
}

// We can also make scroll correlate to the speed boost
pub struct CameraController {
    speed: f32,
    sensitivity: f32,
    speed_boost: f32,
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
                    // KeyCode::ArrowUp => {
                    //     self.is_arrowup_pressed = is_pressed;
                    //     true
                    // }
                    // KeyCode::ArrowDown => {
                    //     self.is_arrowdown_pressed = is_pressed;
                    //     true
                    // }
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
            // Needs fixing
            DeviceEvent::MouseWheel { delta } => {
                if let MouseScrollDelta::LineDelta(_, y) = delta {
                    self.speed += y.signum() * 10.0;
                }
                true
            }
            _ => false,
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
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
            camera.eye += forward * speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward * speed;
        }

        if self.is_right_pressed {
            camera.eye += right * speed;
        }
        if self.is_left_pressed {
            camera.eye -= right * speed;
        }

        if self.is_up_pressed {
            camera.eye += camera.up * speed;
        }
        if self.is_down_pressed {
            camera.eye -= camera.up * speed;
        }
    }
}

use crate::camera::instance::CameraInstance;
use cgmath::SquareMatrix;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view_proj_sky: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _pad1: f32,
    pub time: f32,
    pub delta_time: f32,
    pub _padding: [f32; 2],
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
            _pad1: 0.0,
            time: 0.0,
            delta_time: 0.0,
            _padding: [0.0, 0.0],
        }
    }

    pub fn update_view_proj(&mut self, camera: &mut CameraInstance) {
        self.view_proj = camera.build_view_projection_matrix().into();
        self.view_proj_sky = camera.build_skybox_view_projection_matrix().into();
        camera.update_bearing();
        self.camera_pos = camera.eye.into();
    }

    pub fn increment_time(&mut self, step: f32) {
        self.time += step;
        self.delta_time = step;
    }
}

use crate::pipeline::state::State;
use crate::{camera::instance::CameraInstance, DESC, VERSION};
use chrono::Local;
use std::time::SystemTime;

impl State {
    pub fn get_debug_text(&mut self) -> String {
        self.sys.refresh_memory();
        self.sys.refresh_cpu_usage();
        let cpu_usage = self.sys.global_cpu_usage();
        let ram_total = self.sys.total_memory() as f32 / 1024.0 / 1024.0;
        let ram_used = self.sys.used_memory() as f32 / 1024.0 / 1024.0;

        let vram_total_gb = self.gpu_vram_total as f32 / 1024.0 / 1024.0 / 1024.0;
        let vram_used_gb = self.gpu_vram_used as f32 / 1024.0 / 1024.0 / 1024.0;

        let system_time = SystemTime::now();
        let datetime: chrono::DateTime<Local> = system_time.into();
        let formatted_time = datetime.format("%Y-%m-%d %H:%M:%S.%3f UTC%Z").to_string();

        let bearing_360 = ((self.camera.bearing.0 + std::f32::consts::TAU) % std::f32::consts::TAU)
            * 180.0
            / std::f32::consts::PI;
        let pitch = self
            .camera
            .pitch
            .0
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2)
            * 180.0
            / std::f32::consts::PI;
        let compass_dir = CameraInstance::bearing_to_compass(bearing_360);

        let wind = format!(
            "({}, {})",
            self.ocean_settings_uniform.wind_vector[0], self.ocean_settings_uniform.wind_vector[1]
        );
        let tri_count = self.num_indices / 3;
        format!(
            "Ocean Simulation v{VERSION}\n\
            Stage: {DESC}\n\
            {formatted_time}\n\
            \n\
            XYZ: {x:.1} / {y:.1} / {z:.1}\n\
            Facing: {dir} ({bearing:.0}°) Pitch: {pitch:+.1}°\n\
            FOV: {fov:.0}° ViewDist: {zfar:.0}\n\
            Cam Speed: {cam_speed} (Boost: {cam_boost})\n\
            \n\
            System:\n\
            FPS: {fps:.0} ({ms:.1} ms)\n\
            Resolution: {w}x{h}\n\
            GPU: {gpu_name} ({gpu_load:.0}% @ {gpu_temp:.0}°C)\n\
            VRAM: {vram_used:.2}/{vram_total:.2} GB\n\
            CPU: {cpu_name} ({cpu_usage:.0}%)\n\
            RAM: {ram_used:.1}/{ram_total:.1} GB\n\
            OS: {os_name} ({kernel})\n\
            Backend: Vulkan (wgpu)\n\
            \n\
            Ocean:\n\
            Mesh Size: {size:.0}m²\n\
            Mesh Subdivisions: {sub}\n\
            FFT Subdivisions: {fft_sub}\n\
            l_small: {l_small}\n\
            Wind Vector: {wind}\n\
            Max Angular Velocity: {max_w}\n\
            Tris: {tris}\n\
            ",
            VERSION = VERSION,
            DESC = DESC,
            formatted_time = formatted_time,
            x = self.camera.eye.x,
            y = self.camera.eye.y,
            z = self.camera.eye.z,
            dir = compass_dir,
            bearing = bearing_360,
            pitch = pitch,
            fov = self.ocean_settings_uniform.fovy,
            zfar = self.ocean_settings_uniform.zfar,
            fps = self.fps,
            ms = 1000.0 / self.fps.max(0.001),
            w = self.surface_config.width,
            h = self.surface_config.height,
            size = self.ocean_settings_uniform.mesh_size,
            sub = self.ocean_settings_uniform.mesh_subdivisions,
            gpu_name = self.gpu_name,
            gpu_load = self.gpu_load,
            gpu_temp = self.gpu_temp,
            vram_used = vram_used_gb,
            vram_total = vram_total_gb,
            cpu_name = self.cpu_name,
            cpu_usage = cpu_usage,
            ram_total = ram_total,
            ram_used = ram_used,
            os_name = self.os_name,
            kernel = self.kernel_version,
            tris = tri_count,
            l_small = self.ocean_settings_uniform.l_small,
            wind = wind,
            fft_sub = self.ocean_settings_uniform.fft_subdivisions,
            max_w = self.ocean_settings_uniform.max_w,
            cam_speed = self.ocean_settings_uniform.cam_speed,
            cam_boost = self.ocean_settings_uniform.cam_boost
        )
    }
}

use crate::camera::Camera;
use crate::settings::OceanSettingsBuilder;
use crate::state::State;
use crate::vertex::InitialData;
use crate::{DESC, VERSION};
use chrono::Local;
use std::time::SystemTime;

macro_rules! settings_slider_ui {
    ($ui:expr, $label:expr, $value:expr, $range:expr, $default:expr, $changed:expr) => {{
        $ui.horizontal(|ui| {
            if ui
                .add(egui::Slider::new($value, $range).text($label))
                .changed()
            {
                $changed = true;
            }
            if ui.small_button("⟲").clicked() {
                *$value = $default;
                $changed = true;
            }
        });
    }};
}

macro_rules! settings_slider_vec4_ui {
    ($ui:expr, $label:expr, $value:expr, $range:expr, $default:expr, $changed:expr) => {{
        $ui.push_id($value as *const _ as usize, |ui| {
            egui::CollapsingHeader::new($label)
                .default_open(false)
                .show(ui, |ui| {
                    let labels = ["R", "G", "B", "A"];

                    for i in 0..4 {
                        ui.horizontal(|ui| {
                            ui.label(labels[i]);
                            if ui.add(egui::Slider::new(&mut $value[i], $range)).changed() {
                                $changed = true;
                            }
                        });
                    }

                    if ui.small_button("⟲ Reset").clicked() {
                        *$value = $default;
                        $changed = true;
                    }
                });
        });
    }};
}

impl State {
    pub fn update_cursor_mode(&mut self) {
        if self.show_setting_ui {
            self.window.set_cursor_visible(true);
            self.window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .ok();
        } else {
            self.window.set_cursor_visible(false);
            if self
                .window
                .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                .is_err()
            {
                self.window
                    .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                    .ok();
            }
        }
    }

    pub fn render_settings_ui(&mut self, context: &egui::Context) {
        let defaults = OceanSettingsBuilder::default().build();

        // Left to implement:
        // mesh_size: f32,
        // mesh_subdivisions: u32,

        egui::Window::new("Ocean Settings")
            .collapsible(true)
            .resizable(true)
            .default_width(400.0)
            .default_height(800.0)
            .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
            .show(context, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("General");

                    settings_slider_ui!(
                        ui,
                        "Time Scale",
                        &mut self.draft_settings.time_scale,
                        0.0..=20.0,
                        defaults.time_scale,
                        self.settings_changed
                    );

                    ui.heading("Wave Parameters");
                    settings_slider_ui!(
                        ui,
                        "Amplitude Scale",
                        &mut self.draft_settings.amplitude_scale,
                        0.0..=20.0,
                        defaults.amplitude_scale,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Chop Scale",
                        &mut self.draft_settings.chop_scale,
                        0.0..=10.0,
                        defaults.chop_scale,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Wind X",
                        &mut self.draft_settings.wind_vector[0],
                        -50.0..=50.0,
                        defaults.wind_vector[0],
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Wind Y",
                        &mut self.draft_settings.wind_vector[1],
                        -50.0..=50.0,
                        defaults.wind_vector[1],
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "FFT Subdivisions (log2)",
                        &mut self.draft_settings.pass_num,
                        0..=14,
                        defaults.pass_num,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "FFT Size",
                        &mut self.draft_settings.fft_size,
                        256.0..=16384.0,
                        defaults.fft_size,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Damping",
                        &mut self.draft_settings.l_small,
                        0.001..=100.5,
                        defaults.l_small,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Max Angular Freq.",
                        &mut self.draft_settings.max_w,
                        0.0..=100.0,
                        defaults.max_w,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Amplitude",
                        &mut self.draft_settings.amplitude,
                        0.0010..=20.0,
                        defaults.amplitude,
                        self.settings_changed
                    );

                    ui.separator();
                    ui.heading("Camera");

                    settings_slider_ui!(
                        ui,
                        "FarZ (Render distance)",
                        &mut self.draft_settings.zfar,
                        100.0..=10_000.0,
                        defaults.zfar,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "FOV",
                        &mut self.draft_settings.fovy,
                        30.0..=120.0,
                        defaults.fovy,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Camera Speed",
                        &mut self.draft_settings.cam_speed,
                        1.0..=100.0,
                        defaults.cam_speed,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Camera Boost",
                        &mut self.draft_settings.cam_boost,
                        0.0..=10.0,
                        defaults.cam_boost,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Mouse Sensitivity",
                        &mut self.draft_settings.cam_sensitivity,
                        0.0001..=0.01,
                        defaults.cam_sensitivity,
                        self.settings_changed
                    );

                    ui.separator();
                    ui.heading("Lighting ");

                    settings_slider_ui!(
                        ui,
                        "Roughness",
                        &mut self.draft_settings.roughness,
                        0.0001..=1.0,
                        defaults.roughness,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Base Reflectance",
                        &mut self.draft_settings.f_0,
                        0.0001..=1.0,
                        defaults.f_0,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Specular Scale",
                        &mut self.draft_settings.specular_scale,
                        0.001..=3.0,
                        defaults.specular_scale,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Reflection Scale",
                        &mut self.draft_settings.reflection_scale,
                        0.001..=3.0,
                        defaults.reflection_scale,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Foam Scale",
                        &mut self.draft_settings.foam_scale,
                        0.001..=10.0,
                        defaults.foam_scale,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "SSS Distortion Scale",
                        &mut self.draft_settings.sss_distortion_scale,
                        0.001..=3.0,
                        defaults.sss_distortion_scale,
                        self.settings_changed
                    );

                    settings_slider_vec4_ui!(
                        ui,
                        "Deep Ocean Color",
                        &mut self.draft_settings.deep_color,
                        0.0..=1.0,
                        defaults.deep_color,
                        self.settings_changed
                    );

                    settings_slider_vec4_ui!(
                        ui,
                        "Shallow Ocean Color",
                        &mut self.draft_settings.shallow_color,
                        0.0..=1.0,
                        defaults.shallow_color,
                        self.settings_changed
                    );

                    settings_slider_vec4_ui!(
                        ui,
                        "SSS Color",
                        &mut self.draft_settings.sss_color,
                        0.0..=1.0,
                        defaults.sss_color,
                        self.settings_changed
                    );

                    settings_slider_vec4_ui!(
                        ui,
                        "Sun Color",
                        &mut self.draft_settings.sun_color,
                        0.0..=1.0,
                        defaults.sun_color,
                        self.settings_changed
                    );

                    ui.separator();

                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(self.settings_changed, egui::Button::new("Save"))
                            .clicked()
                        {
                            self.ocean_settings = self.draft_settings;
                            self.camera_controller.update_all_camera_settings(
                                &mut self.camera,
                                self.draft_settings.cam_speed,
                                self.draft_settings.cam_boost,
                                self.draft_settings.cam_sensitivity,
                                self.draft_settings.zfar,
                                self.draft_settings.fovy,
                            );

                            let fft_subdivisions = 1 << self.draft_settings.pass_num;
                            let (new_initial_data_array, _, _) = InitialData::generate_data(
                                self.draft_settings.fft_size,
                                fft_subdivisions,
                                self.draft_settings.wind_vector,
                                self.draft_settings.l_small,
                                self.draft_settings.amplitude,
                                self.draft_settings.max_w,
                            );

                            self.queue.write_buffer(
                                &self.initial_data_buffer,
                                0,
                                bytemuck::cast_slice(&new_initial_data_array),
                            );

                            self.queue.write_buffer(
                                &self.ocean_settings_buffer,
                                0,
                                bytemuck::cast_slice(&[self.ocean_settings]),
                            );

                            self.settings_changed = false;
                        }

                        if ui.button("Reset All").clicked() {
                            self.draft_settings = defaults;
                            self.settings_changed = true;
                        }

                        if ui.button("Load Rogue Ocean").clicked() {
                            self.draft_settings = OceanSettingsBuilder::rogue().build();
                            self.settings_changed = true;
                        }
                    });
                });
            });
    }

    pub fn get_debug_text(&mut self) -> String {
        self.sys.refresh_memory();
        self.sys.refresh_cpu_usage();
        let cpu_usage = self.sys.global_cpu_usage();
        let ram_total = self.sys.total_memory() as f32 / 1024.0 / 1024.0;
        let ram_used = self.sys.used_memory() as f32 / 1024.0 / 1024.0;

        let system_time = SystemTime::now();
        let datetime: chrono::DateTime<Local> = system_time.into();
        let formatted_time = datetime.format("%Y-%m-%d %H:%M:%S.%3f UTC%Z").to_string();

        // [(bearing + 2pi) % 2pi] * [180/pi]
        let bearing_360 = ((self.camera.bearing.0 + std::f32::consts::TAU) % std::f32::consts::TAU)
            * 180.0
            / std::f32::consts::PI;
        // [-pi/2 < pitch < pi/2] * [180/pi]
        let pitch = self
            .camera
            .pitch
            .0
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2)
            * 180.0
            / std::f32::consts::PI;
        let compass_dir = Camera::bearing_to_compass(bearing_360);

        let wind = format!(
            "({}, {})",
            self.ocean_settings.wind_vector[0], self.ocean_settings.wind_vector[1]
        );
        let tri_count = self.num_indices / 3;
        // Data collected by me, formatted by AI
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
            GPU: {gpu_name}\n\
            CPU: {cpu_name} ({cpu_usage:.0}%)\n\
            RAM: {ram_used:.1}/{ram_total:.1} GB\n\
            OS: {os_name} ({kernel})\n\
            Backend: Vulkan (wgpu)\n\
            \n\
            Ocean:\n\
            Mesh Size: {size:.0}m²\n\
            Mesh Subdivisions: {sub}\n\
            FFT Size: {fft_size}\n\
            FFT Subdivisions: {fft_sub}\n\
            Amplitude: {amp}\n\
            l_small: {l_small}\n\
            Wind Vector: {wind}\n\
            Max Angular Velocity: {max_w}\n\
            Initial spectrum - Max: {i_max:.10}, Avg: {i_avg:.10}\n\
            FFT Output (est) - Min: {fft_min:.6}, Max: {fft_max:.6}, Avg: {fft_avg:.6}\n\
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
            fov = self.ocean_settings.fovy,
            zfar = self.ocean_settings.zfar,
            fps = self.fps,
            ms = 1000.0 / self.fps.max(0.001),
            w = self.surface_config.width,
            h = self.surface_config.height,
            size = self.ocean_settings.mesh_size,
            sub = self.ocean_settings.mesh_subdivisions,
            i_max = self.max_magnitude,
            i_avg = self.avg_magnitude,
            fft_min = self.fft_min,
            fft_max = self.fft_max,
            fft_avg = self.fft_avg,
            gpu_name = self.gpu_name,
            cpu_name = self.cpu_name,
            cpu_usage = cpu_usage,
            ram_total = ram_total,
            ram_used = ram_used,
            os_name = self.os_name,
            kernel = self.kernel_version,
            tris = tri_count,
            amp = self.ocean_settings.amplitude,
            l_small = self.ocean_settings.l_small,
            wind = wind,
            fft_size = self.ocean_settings.fft_size,
            fft_sub = self.ocean_settings.fft_subdivisions,
            max_w = self.ocean_settings.max_w,
            cam_speed = self.ocean_settings.cam_speed,
            cam_boost = self.ocean_settings.cam_boost
        )
    }
}

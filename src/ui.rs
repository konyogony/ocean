use crate::camera::Camera;
use crate::settings::{OceanPreset, OceanSettingsBuilder};
use crate::state::State;
use crate::vertex::InitialData;
use crate::{DESC, VERSION};
use chrono::{Local, TimeZone};
use std::path::Path;
use std::time::SystemTime;

macro_rules! settings_number_ui {
    ($ui:expr, $label:expr, $value:expr, $default:expr, $($changed:expr),+ $(,)?) => {{
        $ui.horizontal(|ui| {
            ui.label($label);

            if ui.add(egui::DragValue::new($value)).changed() {
                $( $changed = true; )+
            }

            if ui.small_button("⟲").clicked() {
                *$value = $default;
                $( $changed = true; )+
            }
        });
    }};
}

macro_rules! settings_slider_ui {
    ($ui:expr, $label:expr, $value:expr, $range:expr, $default:expr, $($changed:expr),+ $(,)?) => {{
        $ui.horizontal(|ui| {
            if ui
                .add(egui::Slider::new($value, $range).text($label))
                .changed()
            {
                $( $changed = true; )+
            }

            if ui.small_button("⟲").clicked() {
                *$value = $default;
                $( $changed = true; )+
            }
        });
    }};
}

macro_rules! settings_slider_vec4_ui {
    ($ui:expr, $label:expr, $value:expr, $range:expr, $default:expr, $($changed:expr),+ $(,)?) => {{
        $ui.push_id($value as *const _ as usize, |ui| {
            egui::CollapsingHeader::new($label)
                .default_open(false)
                .show(ui, |ui| {
                    let labels = ["R", "G", "B", "A"];

                    for i in 0..4 {
                        ui.horizontal(|ui| {
                            ui.label(labels[i]);
                            if ui.add(egui::Slider::new(&mut $value[i], $range)).changed() {
                                $( $changed = true; )+
                            }
                        });
                    }

                    if ui.small_button("⟲ Reset").clicked() {
                        *$value = $default;
                        $( $changed = true; )+
                    }
                });
        });
    }};
}

fn fmt(ts: i64) -> String {
    Local
        .timestamp_opt(ts, 0)
        .unwrap()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string()
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
        let defaults = OceanSettingsBuilder::from_preset(&self.current_ocean_preset).build();

        egui::Window::new("Ocean Settings")
            .collapsible(true)
            .resizable(true)
            .default_width(400.0)
            .default_height(800.0)
            .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
            .show(context, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::CollapsingHeader::new("General")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.label(format!(
                                    "Name: {}{}",
                                    self.current_ocean_preset.preset_name,
                                    if self.preset_modified {
                                        "  • modified"
                                    } else {
                                        ""
                                    }
                                ));
                                ui.label(format!(
                                    "Description: {}",
                                    self.current_ocean_preset.description
                                ));
                                ui.small(format!("Author: {}", self.current_ocean_preset.author));
                                ui.small(format!(
                                    "Created: {}",
                                    fmt(self.current_ocean_preset.created_at)
                                ));
                                ui.small(format!(
                                    "Last modified: {}",
                                    fmt(self.current_ocean_preset.last_modified_at)
                                ));
                            });

                            settings_number_ui!(
                                ui,
                                "Seed",
                                &mut self.draft_settings.ocean_seed,
                                defaults.ocean_seed,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Time Scale",
                                &mut self.draft_settings.time_scale,
                                0.0..=20.0,
                                defaults.time_scale,
                                self.settings_changed,
                                self.preset_modified
                            );
                        });
                    ui.separator();

                    egui::CollapsingHeader::new("Wave Parameters")
                        .default_open(true)
                        .show(ui, |ui| {
                            settings_slider_ui!(
                                ui,
                                "FFT Subdivisions (log2) (buggy)",
                                &mut self.draft_settings.pass_num,
                                0..=defaults.pass_num,
                                defaults.pass_num,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "FFT Size",
                                &mut self.draft_settings.fft_size,
                                256.0..=16384.0,
                                defaults.fft_size,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Amplitude",
                                &mut self.draft_settings.amplitude,
                                0.0010..=20.0,
                                defaults.amplitude,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Amplitude Scale",
                                &mut self.draft_settings.amplitude_scale,
                                0.0..=20.0,
                                defaults.amplitude_scale,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Chop Scale",
                                &mut self.draft_settings.chop_scale,
                                0.0..=10.0,
                                defaults.chop_scale,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Wind X",
                                &mut self.draft_settings.wind_vector[0],
                                -50.0..=50.0,
                                defaults.wind_vector[0],
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Wind Y",
                                &mut self.draft_settings.wind_vector[1],
                                -50.0..=50.0,
                                defaults.wind_vector[1],
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Damping",
                                &mut self.draft_settings.l_small,
                                0.001..=100.5,
                                defaults.l_small,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Max Angular Freq.",
                                &mut self.draft_settings.max_w,
                                0.0..=100.0,
                                defaults.max_w,
                                self.settings_changed,
                                self.preset_modified
                            );
                        });
                    ui.separator();

                    egui::CollapsingHeader::new("Camera")
                        .default_open(false)
                        .show(ui, |ui| {
                            settings_slider_ui!(
                                ui,
                                "FarZ (Render distance)",
                                &mut self.draft_settings.zfar,
                                100.0..=10_000.0,
                                defaults.zfar,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "FOV",
                                &mut self.draft_settings.fovy,
                                30.0..=120.0,
                                defaults.fovy,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Camera Speed",
                                &mut self.draft_settings.cam_speed,
                                1.0..=100.0,
                                defaults.cam_speed,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Camera Boost",
                                &mut self.draft_settings.cam_boost,
                                0.0..=10.0,
                                defaults.cam_boost,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Mouse Sensitivity",
                                &mut self.draft_settings.cam_sensitivity,
                                0.0001..=0.01,
                                defaults.cam_sensitivity,
                                self.settings_changed,
                                self.preset_modified
                            );
                        });
                    ui.separator();

                    egui::CollapsingHeader::new("Lighting & Reflection")
                        .default_open(false)
                        .show(ui, |ui| {
                            settings_slider_ui!(
                                ui,
                                "Roughness",
                                &mut self.draft_settings.roughness,
                                0.0001..=1.0,
                                defaults.roughness,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Base Reflectance",
                                &mut self.draft_settings.f_0,
                                0.0001..=1.0,
                                defaults.f_0,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Specular Scale",
                                &mut self.draft_settings.specular_scale,
                                0.001..=3.0,
                                defaults.specular_scale,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Reflection Scale",
                                &mut self.draft_settings.reflection_scale,
                                0.001..=3.0,
                                defaults.reflection_scale,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "SSS Distortion Scale",
                                &mut self.draft_settings.sss_distortion_scale,
                                0.001..=3.0,
                                defaults.sss_distortion_scale,
                                self.settings_changed,
                                self.preset_modified
                            );
                        });
                    ui.separator();

                    egui::CollapsingHeader::new("Surface Details (Foam & Micro Normals)")
                        .default_open(false)
                        .show(ui, |ui| {
                            settings_slider_ui!(
                                ui,
                                "Micro Normal Strength",
                                &mut self.draft_settings.micro_normal_strength,
                                0.0..=1.0,
                                defaults.micro_normal_strength,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Foam Scale",
                                &mut self.draft_settings.foam_scale,
                                0.001..=10.0,
                                defaults.foam_scale,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Foam Threshold",
                                &mut self.draft_settings.foam_threshold,
                                0.0..=1.0,
                                defaults.foam_threshold,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Foam Speed",
                                &mut self.draft_settings.foam_speed,
                                0.0..=1.0,
                                defaults.foam_speed,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Foam Roughness",
                                &mut self.draft_settings.foam_roughness,
                                0.0..=1.0,
                                defaults.foam_roughness,
                                self.settings_changed,
                                self.preset_modified
                            );
                        });
                    ui.separator();

                    egui::CollapsingHeader::new("Caustics")
                        .default_open(false)
                        .show(ui, |ui| {
                            settings_slider_ui!(
                                ui,
                                "Caustic Scale",
                                &mut self.draft_settings.caustic_scale,
                                0.0..=1.0,
                                defaults.caustic_scale,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Caustic Size",
                                &mut self.draft_settings.caustic_size,
                                0.1..=10.0,
                                defaults.caustic_size,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Caustic Speed",
                                &mut self.draft_settings.caustic_speed,
                                0.0..=2.0,
                                defaults.caustic_speed,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Caustic Intensity",
                                &mut self.draft_settings.caustic_intensity,
                                0.0..=10.0,
                                defaults.caustic_intensity,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_number_ui!(
                                ui,
                                "Caustic Octaves",
                                &mut self.draft_settings.caustic_octaves,
                                defaults.caustic_octaves,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Caustic Depth",
                                &mut self.draft_settings.caustic_depth,
                                0.0..=1.0,
                                defaults.caustic_depth,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_ui!(
                                ui,
                                "Caustic Max Distance",
                                &mut self.draft_settings.caustic_max_distance,
                                100.0..=2000.0,
                                defaults.caustic_max_distance,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_vec4_ui!(
                                ui,
                                "Caustic Color Tint",
                                &mut self.draft_settings.caustic_color_tint,
                                0.0..=2.0,
                                defaults.caustic_color_tint,
                                self.settings_changed,
                                self.preset_modified
                            );
                        });
                    ui.separator();

                    egui::CollapsingHeader::new("Water & Sun Colors")
                        .default_open(false)
                        .show(ui, |ui| {
                            settings_slider_vec4_ui!(
                                ui,
                                "Deep Ocean Color",
                                &mut self.draft_settings.deep_color,
                                0.0..=1.0,
                                defaults.deep_color,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_vec4_ui!(
                                ui,
                                "Shallow Ocean Color",
                                &mut self.draft_settings.shallow_color,
                                0.0..=1.0,
                                defaults.shallow_color,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_vec4_ui!(
                                ui,
                                "SSS Color",
                                &mut self.draft_settings.sss_color,
                                0.0..=1.0,
                                defaults.sss_color,
                                self.settings_changed,
                                self.preset_modified
                            );

                            settings_slider_vec4_ui!(
                                ui,
                                "Sun Color",
                                &mut self.draft_settings.sun_color,
                                0.0..=1.0,
                                defaults.sun_color,
                                self.settings_changed,
                                self.preset_modified
                            );
                        });

                        ui.separator();
                        
                        egui::CollapsingHeader::new("Sky & Atmosphere")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("Simulated Sky");
                                
                                settings_slider_vec4_ui!(
                                    ui, "Day Zenith", &mut self.draft_settings.sky_color_day_zenith, 0.0..=1.0, defaults.sky_color_day_zenith, self.settings_changed, self.preset_modified
                                );
                                settings_slider_vec4_ui!(
                                    ui, "Day Horizon", &mut self.draft_settings.sky_color_day_horizon, 0.0..=1.0, defaults.sky_color_day_horizon, self.settings_changed, self.preset_modified
                                );
                                settings_slider_vec4_ui!(
                                    ui, "Night Zenith", &mut self.draft_settings.sky_color_night_zenith, 0.0..=1.0, defaults.sky_color_night_zenith, self.settings_changed, self.preset_modified
                                );
                                settings_slider_vec4_ui!(
                                    ui, "Night Horizon", &mut self.draft_settings.sky_color_night_horizon, 0.0..=1.0, defaults.sky_color_night_horizon, self.settings_changed, self.preset_modified
                                );
                                
                                ui.separator();
                                ui.label("Sunset");
                                settings_slider_vec4_ui!(
                                    ui, "Sunset Orange", &mut self.draft_settings.sky_color_sunset_orange, 0.0..=1.0, defaults.sky_color_sunset_orange, self.settings_changed, self.preset_modified
                                );
                                settings_slider_vec4_ui!(
                                    ui, "Sunset Pink", &mut self.draft_settings.sky_color_sunset_pink, 0.0..=1.0, defaults.sky_color_sunset_pink, self.settings_changed, self.preset_modified
                                );
                                 settings_slider_vec4_ui!(
                                    ui, "Horizon Glow", &mut self.draft_settings.sky_color_horizon_glow, 0.0..=1.0, defaults.sky_color_horizon_glow, self.settings_changed, self.preset_modified
                                );

                                ui.separator();
                                ui.label("Celestial Bodies");
                                settings_slider_ui!(
                                    ui, "Sun Offset Z", &mut self.draft_settings.sun_offset_z, -1.0..=1.0, defaults.sun_offset_z, self.settings_changed, self.preset_modified
                                );
                                 settings_slider_ui!(
                                    ui, "Sun Halo Power", &mut self.draft_settings.sun_halo_power, 1.0..=1000.0, defaults.sun_halo_power, self.settings_changed, self.preset_modified
                                );
                                
                                settings_slider_vec4_ui!(
                                    ui, "Moon Lit Color", &mut self.draft_settings.moon_color_lit, 0.0..=1.0, defaults.moon_color_lit, self.settings_changed, self.preset_modified
                                );
                                settings_slider_vec4_ui!(
                                    ui, "Moon Dark Color", &mut self.draft_settings.moon_color_dark, 0.0..=1.0, defaults.moon_color_dark, self.settings_changed, self.preset_modified
                                );
                                
                                // Moon Vec3 handling via slider group manually
                                ui.label("Moon Phase Offset");
                                ui.horizontal(|ui| {
                                    if ui.add(egui::Slider::new(&mut self.draft_settings.moon_phase_offset[0], -1.0..=1.0).text("X")).changed() { self.settings_changed = true; self.preset_modified = true; }
                                    if ui.add(egui::Slider::new(&mut self.draft_settings.moon_phase_offset[1], -1.0..=1.0).text("Y")).changed() { self.settings_changed = true; self.preset_modified = true; }
                                    if ui.add(egui::Slider::new(&mut self.draft_settings.moon_phase_offset[2], -1.0..=1.0).text("Z")).changed() { self.settings_changed = true; self.preset_modified = true; }
                                });

                                settings_slider_ui!(
                                    ui, "Moon Radius", &mut self.draft_settings.moon_radius, 0.001..=0.1, defaults.moon_radius, self.settings_changed, self.preset_modified
                                );
                                settings_slider_ui!(
                                    ui, "Moon Distance", &mut self.draft_settings.moon_dist, 10.0..=500.0, defaults.moon_dist, self.settings_changed, self.preset_modified
                                );
                                settings_slider_ui!(
                                    ui, "Moon Craters", &mut self.draft_settings.moon_crater_scale, 0.1..=10.0, defaults.moon_crater_scale, self.settings_changed, self.preset_modified
                                );

                                ui.separator();
                                ui.label("Stars & Clouds");
                                 settings_slider_ui!(
                                    ui, "Star Count", &mut self.draft_settings.star_count, 100.0..=10000.0, defaults.star_count, self.settings_changed, self.preset_modified
                                );
                                settings_slider_ui!(
                                    ui, "Star Threshold", &mut self.draft_settings.star_threshold, 0.9..=0.9999, defaults.star_threshold, self.settings_changed, self.preset_modified
                                );
                                
                                 settings_slider_vec4_ui!(
                                    ui, "Cloud Color Day", &mut self.draft_settings.cloud_color_day, 0.0..=1.0, defaults.cloud_color_day, self.settings_changed, self.preset_modified
                                );
                                 settings_slider_vec4_ui!(
                                    ui, "Cloud Color Night", &mut self.draft_settings.cloud_color_night, 0.0..=1.0, defaults.cloud_color_night, self.settings_changed, self.preset_modified
                                );
                                 settings_slider_ui!(
                                    ui, "Cloud Speed", &mut self.draft_settings.cloud_speed, 0.0..=0.5, defaults.cloud_speed, self.settings_changed, self.preset_modified
                                );
                                 settings_slider_ui!(
                                    ui, "Cloud Density Low", &mut self.draft_settings.cloud_density_low, 0.0..=1.0, defaults.cloud_density_low, self.settings_changed, self.preset_modified
                                );
                                 settings_slider_ui!(
                                    ui, "Cloud Density High", &mut self.draft_settings.cloud_density_high, 0.0..=1.0, defaults.cloud_density_high, self.settings_changed, self.preset_modified
                                );
                    });

                    ui.separator();

                    egui::CollapsingHeader::new("Load Preset")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                egui::ComboBox::from_id_salt("preset_select")
                                    .selected_text(self.preset_name_to_load.clone())
                                    .width(160.0)
                                    .show_ui(ui, |ui| {
                                        for name in &self.available_presets {
                                            if ui
                                                .selectable_label(
                                                    self.preset_name_to_load == *name,
                                                    name,
                                                )
                                                .clicked()
                                            {
                                                self.preset_name_to_load = name.clone();
                                            }
                                        }
                                    });

                                if ui.button("Load").clicked() {
                                    let name = self.preset_name_to_load.trim();
                                    if !name.is_empty() {
                                        let preset =
                                            OceanPreset::load_preset(name, Path::new("presets/"));

                                        let seed = self.ocean_settings_uniform.ocean_seed;

                                        self.current_ocean_preset = preset.clone();
                                        self.draft_settings =
                                            OceanSettingsBuilder::from_preset(&preset).build();
                                        self.draft_settings.ocean_seed = seed;

                                        self.settings_changed = true;
                                        self.preset_modified = false;
                                    }
                                }
                            });
                        });

                    ui.separator();
                    
                    egui::CollapsingHeader::new("Update Current Preset")
                        .default_open(false)
                        .show(ui, |ui| {
                            let save_button_text = format!(
                                "Save Current Draft to Preset: {}",
                                self.current_ocean_preset.preset_name
                            );
                            
                            ui.add_enabled_ui(self.preset_modified, |ui| {
                                if ui.button(save_button_text).clicked() {
                                    OceanPreset::modify_preset(
                                        &self.current_ocean_preset.preset_name,
                                        Path::new("presets/"),
                                        self.current_ocean_preset.clone(),
                                        self.draft_settings,
                                    );
                                    
                                    let name = self.current_ocean_preset.preset_name.trim();
                                    if !name.is_empty() {
                                        let preset = OceanPreset::load_preset(
                                            name,
                                            Path::new("presets/")
                                        );
                                        self.current_ocean_preset = preset;
                                        self.preset_modified = false;
                                    }
                                }
                            });
                            
                            if !self.preset_modified {
                                ui.small("Preset is unmodified. Edit a setting to enable save.");
                            } else {
                                ui.small("Save the modified settings to overwrite the current preset file.");
                            }
                        });

                    ui.separator();

                    egui::CollapsingHeader::new("Create New Preset")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Name");
                                    ui.text_edit_singleline(&mut self.preset_name_to_create);
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Author");
                                    ui.text_edit_singleline(&mut self.preset_author_to_create);
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Description");
                                    ui.text_edit_singleline(&mut self.preset_description_to_create);
                                });

                                if ui.button("Create Preset").clicked() {
                                    let name = self.preset_name_to_create.trim();
                                    let description = self.preset_description_to_create.trim();
                                    let author = self.preset_author_to_create.trim();

                                    if !name.is_empty() {
                                        OceanPreset::create_new_preset(
                                            name,
                                            Path::new("presets/"),
                                            author,
                                            description,
                                            self.draft_settings,
                                        );

                                        self.available_presets =
                                            OceanPreset::get_preset_list(Path::new("presets/"))
                                                .unwrap_or_default();
                                    }
                                }
                            });
                        });
                });

                ui.separator();

                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(self.settings_changed, egui::Button::new("Apply Changes"))
                        .clicked()
                    {
                        let old_sub = self.ocean_settings_uniform.fft_subdivisions;
                        self.ocean_settings_uniform = self.draft_settings;

                        if self.ocean_settings_uniform.fft_subdivisions != old_sub {
                            self.reinit_fft_resources();
                        } else {
                            let (new_initial_data_array, _, _) = InitialData::generate_data(
                                self.draft_settings.fft_size,
                                self.ocean_settings_uniform.fft_subdivisions,
                                self.draft_settings.wind_vector,
                                self.draft_settings.l_small,
                                self.draft_settings.amplitude,
                                self.draft_settings.max_w,
                                self.draft_settings.ocean_seed,
                            );

                            self.queue.write_buffer(
                                &self.initial_data_buffer,
                                0,
                                bytemuck::cast_slice(&new_initial_data_array),
                            );
                        }

                        self.camera_controller.update_all_camera_settings(
                            &mut self.camera,
                            self.draft_settings.cam_speed,
                            self.draft_settings.cam_boost,
                            self.draft_settings.cam_sensitivity,
                            self.draft_settings.zfar,
                            self.draft_settings.fovy,
                        );

                        self.queue.write_buffer(
                            &self.ocean_settings_buffer,
                            0,
                            bytemuck::cast_slice(&[self.ocean_settings_uniform]),
                        );

                        self.settings_changed = false;
                    }

                    if ui.button("Reset to Preset").clicked() {
                        let seed = self.ocean_settings_uniform.ocean_seed;
                        self.draft_settings =
                            OceanSettingsBuilder::from_preset(&self.current_ocean_preset).build();
                        self.draft_settings.ocean_seed = seed;
                        self.settings_changed = true;
                        self.preset_modified = false;
                    }
                });
            });
    }

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
            self.ocean_settings_uniform.wind_vector[0], self.ocean_settings_uniform.wind_vector[1]
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
            fov = self.ocean_settings_uniform.fovy,
            zfar = self.ocean_settings_uniform.zfar,
            fps = self.fps,
            ms = 1000.0 / self.fps.max(0.001),
            w = self.surface_config.width,
            h = self.surface_config.height,
            size = self.ocean_settings_uniform.mesh_size,
            sub = self.ocean_settings_uniform.mesh_subdivisions,
            i_max = self.max_magnitude,
            i_avg = self.avg_magnitude,
            fft_min = self.fft_min,
            fft_max = self.fft_max,
            fft_avg = self.fft_avg,
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
            amp = self.ocean_settings_uniform.amplitude,
            l_small = self.ocean_settings_uniform.l_small,
            wind = wind,
            fft_size = self.ocean_settings_uniform.fft_size,
            fft_sub = self.ocean_settings_uniform.fft_subdivisions,
            max_w = self.ocean_settings_uniform.max_w,
            cam_speed = self.ocean_settings_uniform.cam_speed,
            cam_boost = self.ocean_settings_uniform.cam_boost
        )
    }
}

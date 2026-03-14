// new

use crate::pipeline::state::State;
use crate::settings::builder::OceanSettingsBuilder;
use crate::settings::presets::OceanPreset;
use crate::settings::uniform::MAX_CASCADES;
use crate::ui::widgets::{number, section, slider, slider_vec3, slider_vec4};
use chrono::{Local, TimeZone};
use std::path::Path;

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

        macro_rules! s {
            ($ui:expr, $label:expr, $field:ident, $range:expr) => {
                slider(
                    $ui,
                    $label,
                    &mut self.draft_settings.$field,
                    $range,
                    defaults.$field,
                    &mut self.settings_changed,
                    &mut self.preset_modified,
                )
            };
            ($ui:expr, $label:expr, $field:ident [ $i:expr ], $range:expr) => {
                slider(
                    $ui,
                    $label,
                    &mut self.draft_settings.$field[$i],
                    $range,
                    defaults.$field[$i],
                    &mut self.settings_changed,
                    &mut self.preset_modified,
                )
            };
        }
        macro_rules! n {
            ($ui:expr, $label:expr, $field:ident) => {
                number(
                    $ui,
                    $label,
                    &mut self.draft_settings.$field,
                    defaults.$field,
                    &mut self.settings_changed,
                    &mut self.preset_modified,
                )
            };
        }
        macro_rules! v3 {
            ($ui:expr, $label:expr, $field:ident) => {
                slider_vec3(
                    $ui,
                    $label,
                    &mut self.draft_settings.$field,
                    0.0..=1.0,
                    defaults.$field,
                    &mut self.settings_changed,
                    &mut self.preset_modified,
                )
            };
            ($ui:expr, $label:expr, $field:ident, $range:expr) => {
                slider_vec3(
                    $ui,
                    $label,
                    &mut self.draft_settings.$field,
                    $range,
                    defaults.$field,
                    &mut self.settings_changed,
                    &mut self.preset_modified,
                )
            };
        }
        macro_rules! v4 {
            ($ui:expr, $label:expr, $field:ident) => {
                slider_vec4(
                    $ui,
                    $label,
                    &mut self.draft_settings.$field,
                    0.0..=1.0,
                    defaults.$field,
                    &mut self.settings_changed,
                    &mut self.preset_modified,
                )
            };
            ($ui:expr, $label:expr, $field:ident, $range:expr) => {
                slider_vec4(
                    $ui,
                    $label,
                    &mut self.draft_settings.$field,
                    $range,
                    defaults.$field,
                    &mut self.settings_changed,
                    &mut self.preset_modified,
                )
            };
        }

        egui::Window::new("Ocean Settings")
            .collapsible(true)
            .resizable(true)
            .default_width(400.0)
            .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
            .show(context, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    section(ui, "General", true, |ui| {
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
                        n!(ui, "Seed", ocean_seed);
                        s!(ui, "Time Scale", time_scale, 0.0..=20.0);
                        s!(ui, "Time of Day", daynight_cycle, 0.0..=1.0);
                    });

                    section(ui, "Wave Physics", false, |ui| {
                        s!(ui, "FFT Steps", pass_num, 0..=defaults.pass_num);
                        s!(ui, "Amplitude Scale", amplitude_scale, 0.0..=20.0);
                        s!(ui, "Chop Scale", chop_scale, 0.0..=10.0);
                        s!(ui, "Wind X", wind_vector[0], -50.0..=50.0);
                        s!(ui, "Wind Y", wind_vector[1], -50.0..=50.0);
                        s!(ui, "Damping", l_small, 0.001..=100.5);
                        s!(ui, "Max Angular Freq", max_w, 0.0..=100.0);
                        s!(ui, "Wave Epsilon", wave_epsilon, 0.00001..=0.01);
                        s!(
                            ui,
                            "Y Displacement Weight",
                            y_displacement_weight,
                            0.0..=2.0
                        );
                        s!(ui, "Steepness Low", steepness_threshold_low, 0.0..=1.0);
                        s!(ui, "Steepness High", steepness_threshold_high, 0.0..=1.0);
                    });

                    section(ui, "Cascades", false, |ui| {
                        let count = (self.draft_settings.cascade_count as usize).min(MAX_CASCADES);
                        for i in 0..count {
                            ui.push_id(i, |ui| {
                                egui::CollapsingHeader::new(format!("Cascade {}", i)).show(
                                    ui,
                                    |ui| {
                                        ui.horizontal(|ui| {
                                            if ui
                                                .add(
                                                    egui::Slider::new(
                                                        &mut self.draft_settings.cascade_data[i][0],
                                                        10.0..=16384.0,
                                                    )
                                                    .text("FFT Size"),
                                                )
                                                .changed()
                                            {
                                                self.settings_changed = true;
                                                self.preset_modified = true;
                                            }
                                        });
                                        ui.horizontal(|ui| {
                                            if ui
                                                .add(
                                                    egui::Slider::new(
                                                        &mut self.draft_settings.cascade_data[i][1],
                                                        0.0..=5.0,
                                                    )
                                                    .text("Amplitude"),
                                                )
                                                .changed()
                                            {
                                                self.settings_changed = true;
                                                self.preset_modified = true;
                                            }
                                        });
                                    },
                                );
                            });
                        }
                        ui.horizontal(|ui| {
                            if ui.button("+ Add Cascade").clicked()
                                && (self.draft_settings.cascade_count as usize) < MAX_CASCADES
                            {
                                let idx = self.draft_settings.cascade_count as usize;
                                self.draft_settings.cascade_data[idx] = [250.0, 0.3, 0.0, 0.0];
                                self.draft_settings.cascade_count += 1;
                                self.settings_changed = true;
                                self.preset_modified = true;
                            }
                            if ui.button("− Remove Last").clicked()
                                && self.draft_settings.cascade_count > 1
                            {
                                self.draft_settings.cascade_count -= 1;
                                self.settings_changed = true;
                                self.preset_modified = true;
                            }
                        });
                    });

                    section(ui, "Foam", false, |ui| {
                        ui.label("Simulation");
                        s!(ui, "Decay Factor", decay_factor, 0.9..=1.0);
                        s!(ui, "Dissipation Factor", dissipation_factor, 0.9..=1.0);
                        ui.separator();
                        ui.label("Visuals");
                        s!(ui, "Foam Scale", foam_scale, 0.001..=10.0);
                        s!(ui, "Foam Threshold", foam_threshold, 0.0..=1.0);
                        s!(ui, "Foam Speed", foam_speed, 0.0..=1.0);
                        s!(ui, "Foam Roughness", foam_roughness, 0.0..=1.0);
                        s!(ui, "Foam Crest Width", foam_crest_width, 0.0..=1.0);
                        s!(ui, "Foam Power", foam_power, 0.1..=5.0);
                        n!(ui, "Foam Octaves", foam_octaves);
                        v3!(ui, "Foam Color", foam_base_color);
                    });

                    section(ui, "Micro Normals & UV Warp", false, |ui| {
                        ui.label("Micro Normals");
                        s!(ui, "Normal Strength", micro_normal_strength, 0.0..=1.0);
                        s!(ui, "Normal UV Freq", micro_uv_freq, 0.0001..=0.1);
                        s!(ui, "Normal Time Freq", micro_time_freq, 0.00001..=0.01);
                        s!(ui, "Micro Strength Mod", micro_strength_mod, 0.0..=1.0);
                        ui.separator();
                        ui.label("UV Warp");
                        s!(ui, "Warp Scale", warp_uv_scale, 0.0..=5.0);
                        s!(ui, "Warp Strength", warp_strength, 0.0..=5.0);
                        s!(ui, "Warp Time Scale", warp_time_scale, 0.0..=1.0);
                        ui.separator();
                        ui.label("Hash");
                        s!(ui, "Hash Scale", hash_scale, 0.01..=1.0);
                        s!(ui, "Hash Dot", hash_dot, 1.0..=100.0);
                    });

                    section(ui, "Water Colors & SSS", false, |ui| {
                        v4!(ui, "Deep Ocean", deep_color);
                        v4!(ui, "Shallow Ocean", shallow_color);
                        v4!(ui, "SSS Color", sss_color);
                        v4!(ui, "Sun Color", sun_color);
                        ui.separator();
                        s!(ui, "SSS Min Height", sss_min_height, -5.0..=0.0);
                        s!(ui, "SSS Max Height", sss_max_height, 0.0..=10.0);
                        s!(ui, "SSS Power", sss_power, 0.1..=32.0);
                        s!(ui, "SSS Intensity", sss_intensity, 0.0..=5.0);
                        s!(
                            ui,
                            "SSS Distortion Scale",
                            sss_distortion_scale,
                            0.001..=3.0
                        );
                    });

                    section(ui, "Lighting & Reflections", false, |ui| {
                        s!(ui, "Roughness", roughness, 0.0001..=1.0);
                        s!(ui, "Base Refl (F0)", f_0, 0.0001..=1.0);
                        s!(ui, "Specular Scale", specular_scale, 0.001..=3.0);
                        s!(ui, "Reflection Scale", reflection_scale, 0.001..=3.0);
                        s!(ui, "Reflection Min", reflection_min, 0.0..=1.0);
                        s!(ui, "Reflection Max", reflection_max, 0.0..=1.0);
                        s!(ui, "Blend Strength", blend_strength, 0.0..=1.0);
                        s!(ui, "Ambient Scale", ambient_scale, 0.0..=1.0);
                        s!(ui, "Water Brightness Mod", water_brightness_mod, 0.0..=2.0);
                        s!(ui, "Bloom Scale", bloom_scale, 0.0..=2.0);
                        s!(ui, "Detail Fade", detail_fade, 0.0..=2000.0);
                    });

                    section(ui, "Caustics", false, |ui| {
                        s!(ui, "Intensity", caustic_intensity, 0.0..=10.0);
                        s!(ui, "Scale", caustic_scale, 0.0..=1.0);
                        s!(ui, "Size", caustic_size, 0.1..=10.0);
                        s!(ui, "Speed", caustic_speed, 0.0..=2.0);
                        s!(ui, "Depth", caustic_depth, 0.0..=20.0);
                        s!(ui, "Max Distance", caustic_max_distance, 100.0..=2000.0);
                        s!(ui, "Aberration", caustic_aberration, 0.0..=0.1);
                        s!(ui, "Smooth Low", caustic_smooth_low, 0.0..=1.0);
                        s!(ui, "Smooth High", caustic_smooth_high, 0.0..=2.0);
                        n!(ui, "Caustic Octaves", caustic_octaves);
                        v4!(ui, "Caustic Tint", caustic_color_tint, 0.0..=2.0);
                    });

                    section(ui, "Sky & Sunset", false, |ui| {
                        ui.label("Sky Gradient");
                        v4!(ui, "Day Zenith", sky_color_day_zenith);
                        v4!(ui, "Day Horizon", sky_color_day_horizon);
                        v4!(ui, "Night Zenith", sky_color_night_zenith);
                        v4!(ui, "Night Horizon", sky_color_night_horizon);
                        s!(
                            ui,
                            "Zenith Gradient Exp",
                            sky_zenith_gradient_exp,
                            0.1..=5.0
                        );
                        s!(ui, "Horizon Glow Mult", horizon_glow_mult, 0.0..=5.0);
                        ui.separator();
                        ui.label("Sunset");
                        v4!(ui, "Sunset Orange", sky_color_sunset_orange);
                        v4!(ui, "Sunset Pink", sky_color_sunset_pink);
                        v4!(ui, "Horizon Glow Color", sky_color_horizon_glow);
                        v3!(ui, "Sunset Scatter Color", sunset_scatter_color, 0.0..=2.0);
                        s!(
                            ui,
                            "Sunset Scatter Intensity",
                            sunset_scatter_intensity,
                            0.0..=2.0
                        );
                        s!(ui, "Sunset Orange Weight", sunset_orange_weight, 0.0..=1.0);
                        s!(ui, "Sunset Intensity", sunset_intensity, 0.0..=10.0);
                    });

                    section(ui, "Sun", false, |ui| {
                        s!(ui, "Sun Offset Z", sun_offset_z, -1.0..=1.0);
                        s!(ui, "Sun Size Inner", sun_size_inner, 0.999..=1.0);
                        s!(ui, "Sun Size Outer", sun_size_outer, 0.999..=1.0);
                        s!(ui, "Sun Halo Power", sun_halo_power, 1.0..=1000.0);
                        s!(ui, "Sun Halo Intensity", sun_halo_intensity, 0.0..=1.0);
                    });

                    section(ui, "Moon", false, |ui| {
                        v4!(ui, "Moon Lit Color", moon_color_lit);
                        v4!(ui, "Moon Dark Color", moon_color_dark);
                        ui.label("Moon Phase Offset");
                        ui.horizontal(|ui| {
                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.draft_settings.moon_phase_offset[0],
                                        -1.0..=1.0,
                                    )
                                    .text("X"),
                                )
                                .changed()
                            {
                                self.settings_changed = true;
                                self.preset_modified = true;
                            }
                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.draft_settings.moon_phase_offset[1],
                                        -1.0..=1.0,
                                    )
                                    .text("Y"),
                                )
                                .changed()
                            {
                                self.settings_changed = true;
                                self.preset_modified = true;
                            }
                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.draft_settings.moon_phase_offset[2],
                                        -1.0..=1.0,
                                    )
                                    .text("Z"),
                                )
                                .changed()
                            {
                                self.settings_changed = true;
                                self.preset_modified = true;
                            }
                        });
                        s!(ui, "Moon Radius", moon_radius, 0.001..=0.1);
                        s!(ui, "Moon Distance", moon_dist, 10.0..=500.0);
                        s!(ui, "Moon Craters", moon_crater_scale, 0.1..=10.0);
                        s!(ui, "Moon Halo Power", moon_halo_power, 1.0..=1000.0);
                        s!(ui, "Moon Halo Intensity", moon_halo_intensity, 0.0..=1.0);
                        s!(ui, "Moon Light Dim", moon_light_dim, 0.0..=1.0);
                    });

                    section(ui, "Stars & Aurora", false, |ui| {
                        ui.label("Stars");
                        s!(ui, "Star Count", star_count, 100.0..=10000.0);
                        s!(ui, "Star Threshold", star_threshold, 0.9..=0.9999);
                        s!(ui, "Star Size", star_size, 1.0..=100.0);
                        s!(ui, "Star Blink Speed", star_blink_speed, 0.0..=10.0);
                        ui.separator();
                        ui.label("Aurora");
                        s!(ui, "Aurora Strength", aurora_strength, 0.0..=5.0);
                        s!(ui, "Aurora Brightness", aurora_brightness, 0.0..=10.0);
                        s!(ui, "Aurora Y Threshold", aurora_y_threshold, 0.0..=1.0);
                    });

                    section(ui, "Clouds", false, |ui| {
                        v4!(ui, "Cloud Color Day", cloud_color_day);
                        v4!(ui, "Cloud Color Night", cloud_color_night);
                        s!(ui, "Cloud Speed", cloud_speed, 0.0..=0.5);
                        s!(ui, "Density Low", cloud_density_low, 0.0..=1.0);
                        s!(ui, "Density High", cloud_density_high, 0.0..=1.0);
                    });

                    section(ui, "Camera", false, |ui| {
                        s!(ui, "Render Distance", zfar, 100.0..=10000.0);
                        s!(ui, "FOV", fovy, 30.0..=120.0);
                        s!(ui, "Camera Speed", cam_speed, 1.0..=100.0);
                        s!(ui, "Camera Boost", cam_boost, 0.0..=10.0);
                        s!(ui, "Mouse Sensitivity", cam_sensitivity, 0.0001..=0.01);
                    });

                    section(ui, "Presets Management", false, |ui| {
                        ui.horizontal(|ui| {
                            egui::ComboBox::from_id_salt("ps_load")
                                .selected_text(&self.preset_name_to_load)
                                .show_ui(ui, |ui| {
                                    for name in &self.available_presets {
                                        ui.selectable_value(
                                            &mut self.preset_name_to_load,
                                            name.clone(),
                                            name,
                                        );
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
                        ui.separator();
                        ui.add_enabled_ui(self.preset_modified, |ui| {
                            if ui
                                .button(format!(
                                    "Save to {}",
                                    self.current_ocean_preset.preset_name
                                ))
                                .clicked()
                            {
                                OceanPreset::modify_preset(
                                    &self.current_ocean_preset.preset_name,
                                    Path::new("presets/"),
                                    self.current_ocean_preset.clone(),
                                    self.draft_settings,
                                );
                                let name = self.current_ocean_preset.preset_name.clone();
                                let preset = OceanPreset::load_preset(&name, Path::new("presets/"));
                                self.current_ocean_preset = preset;
                                self.preset_modified = false;
                            }
                        });
                        if !self.preset_modified {
                            ui.small("Preset is unmodified. Edit a setting to enable save.");
                        } else {
                            ui.small(
                                "Save the modified settings to overwrite the current preset file.",
                            );
                        }
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
                                        ui.text_edit_singleline(
                                            &mut self.preset_description_to_create,
                                        );
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
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(self.settings_changed, egui::Button::new("Apply Changes"))
                        .clicked()
                    {
                        let old = self.ocean_settings_uniform;
                        self.ocean_settings_uniform = self.draft_settings;
                        if self.ocean_settings_uniform.fft_subdivisions != old.fft_subdivisions
                            || self.ocean_settings_uniform.cascade_count != old.cascade_count
                            || self.ocean_settings_uniform.wind_vector != old.wind_vector
                            || self.ocean_settings_uniform.ocean_seed != old.ocean_seed
                            || self.ocean_settings_uniform.l_small != old.l_small
                            || self.ocean_settings_uniform.cascade_data != old.cascade_data
                        {
                            self.reinit_fft_resources();
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
}

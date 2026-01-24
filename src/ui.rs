use crate::settings::OceanSettingsBuilder;
use crate::state::State;

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

        egui::Window::new("Ocean Settings")
            .collapsible(true)
            .resizable(true)
            .default_width(400.0)
            .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
            .show(context, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Wave Parameters");

                    settings_slider_ui!(
                        ui,
                        "Time Scale",
                        &mut self.draft_settings.time_scale,
                        0.1..=20.0,
                        defaults.time_scale,
                        self.settings_changed
                    );

                    settings_slider_ui!(
                        ui,
                        "Amplitude",
                        &mut self.draft_settings.amplitude,
                        0.0..=50.0,
                        defaults.amplitude,
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

                    ui.separator();
                    ui.heading("Wind");

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

                    ui.separator();
                    ui.heading("Camera");

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
                        "Mouse Sensitivity",
                        &mut self.draft_settings.cam_sensitivity,
                        0.1..=5.0,
                        defaults.cam_sensitivity,
                        self.settings_changed
                    );

                    ui.separator();
                    ui.heading("Rendering");

                    settings_slider_ui!(
                        ui,
                        "Far Plane",
                        &mut self.draft_settings.zfar,
                        100.0..=10_000.0,
                        defaults.zfar,
                        self.settings_changed
                    );

                    ui.separator();

                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(self.settings_changed, egui::Button::new("Save"))
                            .clicked()
                        {
                            self.ocean_settings = self.draft_settings;
                            self.queue.write_buffer(
                                &self.ocean_settings_buffer,
                                0,
                                bytemuck::cast_slice(&[self.ocean_settings]),
                            );
                        }

                        if ui.button("Reset All").clicked() {
                            self.draft_settings = defaults;
                            self.settings_changed = true;
                        }
                    });
                });
            });
    }
}

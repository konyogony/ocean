use egui::Ui;

pub fn slider<T: egui::emath::Numeric + Copy>(
    ui: &mut Ui,
    label: &str,
    value: &mut T,
    range: std::ops::RangeInclusive<T>,
    default: T,
    changed: &mut bool,
    modified: &mut bool,
) {
    ui.horizontal(|ui| {
        if ui
            .add(egui::Slider::new(value, range).text(label))
            .changed()
        {
            *changed = true;
            *modified = true;
        }
        if ui.small_button("⟲").clicked() {
            *value = default;
            *changed = true;
            *modified = true;
        }
    });
}

pub fn number<T: egui::emath::Numeric + Copy>(
    ui: &mut Ui,
    label: &str,
    value: &mut T,
    default: T,
    changed: &mut bool,
    modified: &mut bool,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        if ui.add(egui::DragValue::new(value)).changed() {
            *changed = true;
            *modified = true;
        }
        if ui.small_button("⟲").clicked() {
            *value = default;
            *changed = true;
            *modified = true;
        }
    });
}

pub fn slider_vec3(
    ui: &mut Ui,
    label: &str,
    value: &mut [f32; 3],
    range: std::ops::RangeInclusive<f32>,
    default: [f32; 3],
    changed: &mut bool,
    modified: &mut bool,
) {
    ui.push_id(label, |ui| {
        egui::CollapsingHeader::new(label)
            .default_open(false)
            .show(ui, |ui| {
                for (i, ch) in ["X/R", "Y/G", "Z/B"].iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(*ch);
                        if ui
                            .add(egui::Slider::new(&mut value[i], range.clone()))
                            .changed()
                        {
                            *changed = true;
                            *modified = true;
                        }
                    });
                }
                if ui.small_button("⟲ Reset").clicked() {
                    *value = default;
                    *changed = true;
                    *modified = true;
                }
            });
    });
}

pub fn slider_vec4(
    ui: &mut Ui,
    label: &str,
    value: &mut [f32; 4],
    range: std::ops::RangeInclusive<f32>,
    default: [f32; 4],
    changed: &mut bool,
    modified: &mut bool,
) {
    ui.push_id(label, |ui| {
        egui::CollapsingHeader::new(label)
            .default_open(false)
            .show(ui, |ui| {
                for (i, ch) in ["R", "G", "B", "A"].iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(*ch);
                        if ui
                            .add(egui::Slider::new(&mut value[i], range.clone()))
                            .changed()
                        {
                            *changed = true;
                            *modified = true;
                        }
                    });
                }
                if ui.small_button("⟲ Reset").clicked() {
                    *value = default;
                    *changed = true;
                    *modified = true;
                }
            });
    });
}

pub fn section(ui: &mut egui::Ui, title: &str, open: bool, body: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(title)
        .default_open(open)
        .show(ui, |ui| body(ui));
    ui.separator();
}

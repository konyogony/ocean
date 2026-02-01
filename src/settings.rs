use anyhow::Result;
use std::{
    fs::{self, read_to_string, write},
    path::Path,
};

#[derive(Debug, serde::Deserialize, serde::Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct OceanPreset {
    pub preset_name: String,
    pub author: String,
    pub description: String,
    pub created_at: i64,
    pub last_modified_at: i64,
    pub fft_size: f32,
    pub fft_subdivisions: u32,
    pub time_scale: f32,
    pub ocean_seed: u32,
    pub chop_scale: f32,
    pub amplitude_scale: f32,
    pub wave_scale: f32,
    pub wind_vector: [f32; 2],
    pub amplitude: f32,
    pub l_small: f32,
    pub max_w: f32,
    pub fovy: f32,
    pub zfar: f32,
    pub cam_speed: f32,
    pub cam_boost: f32,
    pub cam_sensitivity: f32,
    pub roughness: f32,
    pub f_0: f32,
    pub specular_scale: f32,
    pub reflection_scale: f32,
    pub foam_scale: f32,
    pub sss_distortion_scale: f32,
    pub caustic_scale: f32,
    pub caustic_size: f32,
    pub caustic_speed: f32,
    pub caustic_intensity: f32,
    pub caustic_octaves: u32,
    pub caustic_depth: f32,
    pub caustic_max_distance: f32,
    pub micro_normal_strength: f32,
    pub foam_threshold: f32,
    pub foam_speed: f32,
    pub foam_roughness: f32,
    pub caustic_color_tint: [f32; 4],
    pub deep_color: [f32; 4],
    pub shallow_color: [f32; 4],
    pub sss_color: [f32; 4],
    pub sun_color: [f32; 4],
}

impl OceanPreset {
    pub fn get_preset_list(directory: &Path) -> Result<Vec<String>> {
        let mut presets = Vec::new();
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|n| n.to_str()) {
                    presets.push(stem.to_string());
                }
            }
        }
        Ok(presets)
    }

    pub fn load_preset(preset_name: &str, directory: &Path) -> OceanPreset {
        let json_path = directory.join(format!("{preset_name}.json"));
        let data = read_to_string(json_path).expect("Couldnt read file");
        serde_json::from_str(data.as_str()).expect("Couldnt deserialize json")
    }

    pub fn create_new_preset(
        preset_name: &str,
        directory: &Path,
        author: &str,
        description: &str,
        ocean_settings: OceanSettingsUniform,
    ) {
        let new_preset: OceanPreset = OceanPreset {
            preset_name: preset_name.to_string(),
            author: author.to_string(),
            description: description.to_string(),
            created_at: chrono::Local::now().timestamp(),
            last_modified_at: chrono::Local::now().timestamp(),
            fft_size: ocean_settings.fft_size,
            fft_subdivisions: ocean_settings.fft_subdivisions,
            time_scale: ocean_settings.time_scale,
            ocean_seed: ocean_settings.ocean_seed,
            chop_scale: ocean_settings.chop_scale,
            amplitude_scale: ocean_settings.amplitude_scale,
            wave_scale: ocean_settings.wave_scale,
            wind_vector: ocean_settings.wind_vector,
            amplitude: ocean_settings.amplitude,
            l_small: ocean_settings.l_small,
            max_w: ocean_settings.max_w,
            fovy: ocean_settings.fovy,
            zfar: ocean_settings.zfar,
            cam_speed: ocean_settings.cam_speed,
            cam_boost: ocean_settings.cam_boost,
            cam_sensitivity: ocean_settings.cam_sensitivity,
            roughness: ocean_settings.roughness,
            f_0: ocean_settings.f_0,
            specular_scale: ocean_settings.specular_scale,
            reflection_scale: ocean_settings.reflection_scale,
            foam_scale: ocean_settings.foam_scale,
            sss_distortion_scale: ocean_settings.sss_distortion_scale,
            caustic_scale: ocean_settings.caustic_scale,
            caustic_size: ocean_settings.caustic_size,
            caustic_speed: ocean_settings.caustic_speed,
            caustic_intensity: ocean_settings.caustic_intensity,
            caustic_octaves: ocean_settings.caustic_octaves,
            caustic_depth: ocean_settings.caustic_depth,
            caustic_max_distance: ocean_settings.caustic_max_distance,
            micro_normal_strength: ocean_settings.micro_normal_strength,
            foam_threshold: ocean_settings.foam_threshold,
            foam_speed: ocean_settings.foam_speed,
            foam_roughness: ocean_settings.foam_roughness,
            caustic_color_tint: ocean_settings.caustic_color_tint,
            deep_color: ocean_settings.deep_color,
            shallow_color: ocean_settings.shallow_color,
            sss_color: ocean_settings.sss_color,
            sun_color: ocean_settings.sun_color,
        };

        let json_path = directory.join(format!("{preset_name}.json"));
        let json_data =
            serde_json::to_string_pretty(&new_preset).expect("Couldnt format to a json string");
        write(json_path, &json_data).expect("Couldnt write to file");
    }

    pub fn modify_preset(
        preset_name: &str,
        directory: &Path,
        current_ocean_settings: OceanPreset,
        new_ocean_settings: OceanSettingsUniform,
    ) {
        let new_preset: OceanPreset = OceanPreset {
            preset_name: preset_name.to_string(),
            author: current_ocean_settings.author.to_string(),
            description: current_ocean_settings.description.to_string(),
            created_at: current_ocean_settings.created_at,
            last_modified_at: chrono::Local::now().timestamp(),
            fft_size: new_ocean_settings.fft_size,
            fft_subdivisions: new_ocean_settings.fft_subdivisions,
            time_scale: new_ocean_settings.time_scale,
            ocean_seed: new_ocean_settings.ocean_seed,
            chop_scale: new_ocean_settings.chop_scale,
            amplitude_scale: new_ocean_settings.amplitude_scale,
            wave_scale: new_ocean_settings.wave_scale,
            wind_vector: new_ocean_settings.wind_vector,
            amplitude: new_ocean_settings.amplitude,
            l_small: new_ocean_settings.l_small,
            max_w: new_ocean_settings.max_w,
            fovy: new_ocean_settings.fovy,
            zfar: new_ocean_settings.zfar,
            cam_speed: new_ocean_settings.cam_speed,
            cam_boost: new_ocean_settings.cam_boost,
            cam_sensitivity: new_ocean_settings.cam_sensitivity,
            roughness: new_ocean_settings.roughness,
            f_0: new_ocean_settings.f_0,
            specular_scale: new_ocean_settings.specular_scale,
            reflection_scale: new_ocean_settings.reflection_scale,
            foam_scale: new_ocean_settings.foam_scale,
            sss_distortion_scale: new_ocean_settings.sss_distortion_scale,
            caustic_scale: new_ocean_settings.caustic_scale,
            caustic_size: new_ocean_settings.caustic_size,
            caustic_speed: new_ocean_settings.caustic_speed,
            caustic_intensity: new_ocean_settings.caustic_intensity,
            caustic_octaves: new_ocean_settings.caustic_octaves,
            caustic_depth: new_ocean_settings.caustic_depth,
            caustic_max_distance: new_ocean_settings.caustic_max_distance,
            micro_normal_strength: new_ocean_settings.micro_normal_strength,
            foam_threshold: new_ocean_settings.foam_threshold,
            foam_speed: new_ocean_settings.foam_speed,
            foam_roughness: new_ocean_settings.foam_roughness,
            caustic_color_tint: new_ocean_settings.caustic_color_tint,
            deep_color: new_ocean_settings.deep_color,
            shallow_color: new_ocean_settings.shallow_color,
            sss_color: new_ocean_settings.sss_color,
            sun_color: new_ocean_settings.sun_color,
        };

        let json_path = directory.join(format!("{preset_name}.json"));
        let json_data =
            serde_json::to_string_pretty(&new_preset).expect("Couldnt format to a json string");
        write(json_path, &json_data).expect("Couldnt write to file");
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OceanSettingsUniform {
    pub mesh_size: f32,
    pub mesh_subdivisions: u32,
    pub fft_size: f32,
    pub fft_subdivisions: u32,
    pub pass_num: u32,
    pub time_scale: f32,
    pub ocean_seed: u32,
    pub chop_scale: f32,
    pub amplitude_scale: f32,
    pub wave_scale: f32,
    pub wind_vector: [f32; 2],
    pub amplitude: f32,
    pub l_small: f32,
    pub max_w: f32,
    pub fovy: f32,
    pub zfar: f32,
    pub cam_speed: f32,
    pub cam_boost: f32,
    pub cam_sensitivity: f32,
    pub roughness: f32,
    pub f_0: f32,
    pub specular_scale: f32,
    pub reflection_scale: f32,
    pub foam_scale: f32,
    pub sss_distortion_scale: f32,
    pub caustic_scale: f32,
    pub caustic_size: f32,
    pub caustic_speed: f32,
    pub caustic_intensity: f32,
    pub caustic_octaves: u32,
    pub caustic_depth: f32,
    pub caustic_max_distance: f32,
    pub micro_normal_strength: f32,
    pub foam_threshold: f32,
    pub foam_speed: f32,
    pub foam_roughness: f32,
    pub _pad2: [f32; 3],
    pub caustic_color_tint: [f32; 4],
    pub deep_color: [f32; 4],
    pub shallow_color: [f32; 4],
    pub sss_color: [f32; 4],
    pub sun_color: [f32; 4],
}

pub struct OceanSettingsBuilder {
    mesh_size: f32,
    mesh_subdivisions: u32,
    fft_size: f32,
    fft_subdivisions: u32,
    time_scale: f32,
    ocean_seed: u32,
    chop_scale: f32,
    amplitude_scale: f32,
    wind_vector: [f32; 2],
    amplitude: f32,
    l_small: f32,
    max_w: f32,
    fovy: f32,
    zfar: f32,
    cam_speed: f32,
    cam_boost: f32,
    cam_sensitivity: f32,
    roughness: f32,
    f_0: f32,
    specular_scale: f32,
    reflection_scale: f32,
    foam_scale: f32,
    sss_distortion_scale: f32,
    caustic_scale: f32,
    caustic_size: f32,
    caustic_speed: f32,
    caustic_intensity: f32,
    caustic_octaves: u32,
    caustic_depth: f32,
    caustic_max_distance: f32,
    micro_normal_strength: f32,
    foam_threshold: f32,
    foam_speed: f32,
    foam_roughness: f32,
    caustic_color_tint: [f32; 4],
    deep_color: [f32; 4],
    shallow_color: [f32; 4],
    sss_color: [f32; 4],
    sun_color: [f32; 4],
}

impl Default for OceanSettingsBuilder {
    fn default() -> Self {
        Self {
            mesh_size: 1000.0,
            mesh_subdivisions: 2048,
            fft_size: 1000.0,
            fft_subdivisions: 256,
            time_scale: 1.5,
            ocean_seed: 0,
            chop_scale: 1.2,
            amplitude_scale: 1.0,
            wind_vector: [6.0, -8.0],
            amplitude: 0.1,
            l_small: 0.1,
            max_w: 10.0,
            fovy: 60.0,
            zfar: 1500.0,
            cam_speed: 20.0,
            cam_boost: 2.5,
            cam_sensitivity: 0.002,
            roughness: 0.2,
            f_0: 0.02,
            specular_scale: 1.0,
            reflection_scale: 0.2,
            foam_scale: 4.0,
            sss_distortion_scale: 0.2,
            caustic_scale: 0.18,
            caustic_size: 4.0,
            caustic_speed: 0.2,
            caustic_intensity: 2.0,
            caustic_octaves: 1,
            caustic_depth: 0.25,
            caustic_max_distance: 600.0,
            micro_normal_strength: 0.05,
            foam_threshold: 0.5,
            foam_speed: 0.05,
            foam_roughness: 0.6,
            caustic_color_tint: [0.95, 1.0, 1.05, 1.0],
            deep_color: [0.0, 0.01, 0.05, 1.0],
            shallow_color: [0.0, 0.06, 0.09, 1.0],
            sss_color: [0.0, 0.4, 0.3, 1.0],
            sun_color: [1.0, 0.9, 0.8, 1.0],
        }
    }
}

impl OceanSettingsBuilder {
    pub fn mesh_size(mut self, v: f32) -> Self {
        self.mesh_size = v;
        self
    }
    pub fn mesh_subdivisions(mut self, v: u32) -> Self {
        self.mesh_subdivisions = v;
        self
    }

    pub fn fft_size(mut self, v: f32) -> Self {
        self.fft_size = v;
        self
    }

    pub fn fft_subdivisions(mut self, v: u32) -> Self {
        self.fft_subdivisions = v;
        self
    }

    pub fn time_scale(mut self, v: f32) -> Self {
        self.time_scale = v;
        self
    }

    pub fn chop_scale(mut self, v: f32) -> Self {
        self.chop_scale = v;
        self
    }

    pub fn amplitude_scale(mut self, v: f32) -> Self {
        self.amplitude_scale = v;
        self
    }

    pub fn wind_vector(mut self, v: [f32; 2]) -> Self {
        self.wind_vector = v;
        self
    }

    pub fn amplitude(mut self, v: f32) -> Self {
        self.amplitude = v;
        self
    }

    pub fn l_small(mut self, v: f32) -> Self {
        self.l_small = v;
        self
    }

    pub fn max_w(mut self, v: f32) -> Self {
        self.max_w = v;
        self
    }

    pub fn fovy(mut self, v: f32) -> Self {
        self.fovy = v;
        self
    }

    pub fn zfar(mut self, v: f32) -> Self {
        self.zfar = v;
        self
    }

    pub fn cam_speed(mut self, v: f32) -> Self {
        self.cam_speed = v;
        self
    }

    pub fn cam_boost(mut self, v: f32) -> Self {
        self.cam_boost = v;
        self
    }

    pub fn cam_sensitivity(mut self, v: f32) -> Self {
        self.cam_sensitivity = v;
        self
    }

    pub fn roughness(mut self, v: f32) -> Self {
        self.roughness = v;
        self
    }

    pub fn f_0(mut self, v: f32) -> Self {
        self.f_0 = v;
        self
    }

    pub fn specular_scale(mut self, v: f32) -> Self {
        self.specular_scale = v;
        self
    }

    pub fn reflection_scale(mut self, v: f32) -> Self {
        self.reflection_scale = v;
        self
    }

    pub fn foam_scale(mut self, v: f32) -> Self {
        self.foam_scale = v;
        self
    }

    pub fn sss_distortion_scale(mut self, v: f32) -> Self {
        self.sss_distortion_scale = v;
        self
    }

    pub fn deep_color(mut self, v: [f32; 4]) -> Self {
        self.deep_color = v;
        self
    }

    pub fn shallow_color(mut self, v: [f32; 4]) -> Self {
        self.shallow_color = v;
        self
    }

    pub fn sss_color(mut self, v: [f32; 4]) -> Self {
        self.sss_color = v;
        self
    }

    pub fn sun_color(mut self, v: [f32; 4]) -> Self {
        self.sun_color = v;
        self
    }

    pub fn ocean_seed(mut self, v: u32) -> Self {
        self.ocean_seed = v;
        self
    }

    pub fn caustic_scale(mut self, v: f32) -> Self {
        self.caustic_scale = v;
        self
    }

    pub fn caustic_size(mut self, v: f32) -> Self {
        self.caustic_size = v;
        self
    }

    pub fn caustic_speed(mut self, v: f32) -> Self {
        self.caustic_speed = v;
        self
    }

    pub fn caustic_intensity(mut self, v: f32) -> Self {
        self.caustic_intensity = v;
        self
    }

    pub fn caustic_color_tint(mut self, v: [f32; 4]) -> Self {
        self.caustic_color_tint = v;
        self
    }

    pub fn caustic_octaves(mut self, v: u32) -> Self {
        self.caustic_octaves = v;
        self
    }

    pub fn caustic_depth(mut self, v: f32) -> Self {
        self.caustic_depth = v;
        self
    }

    pub fn caustic_max_distance(mut self, v: f32) -> Self {
        self.caustic_max_distance = v;
        self
    }

    pub fn micro_normal_strength(mut self, v: f32) -> Self {
        self.micro_normal_strength = v;
        self
    }

    pub fn foam_threshold(mut self, v: f32) -> Self {
        self.foam_threshold = v;
        self
    }

    pub fn foam_speed(mut self, v: f32) -> Self {
        self.foam_speed = v;
        self
    }

    pub fn foam_roughness(mut self, v: f32) -> Self {
        self.foam_roughness = v;
        self
    }

    pub fn apply_preset(self, preset: &OceanPreset) -> Self {
        self.sun_color(preset.sun_color)
            .sss_color(preset.sss_color)
            .shallow_color(preset.shallow_color)
            .deep_color(preset.deep_color)
            .caustic_color_tint(preset.caustic_color_tint)
            .foam_roughness(preset.foam_roughness)
            .foam_speed(preset.foam_speed)
            .foam_threshold(preset.foam_threshold)
            .foam_scale(preset.foam_scale)
            .micro_normal_strength(preset.micro_normal_strength)
            .caustic_max_distance(preset.caustic_max_distance)
            .caustic_depth(preset.caustic_depth)
            .caustic_octaves(preset.caustic_octaves)
            .caustic_intensity(preset.caustic_intensity)
            .caustic_speed(preset.caustic_speed)
            .caustic_size(preset.caustic_size)
            .caustic_scale(preset.caustic_scale)
            .sss_distortion_scale(preset.sss_distortion_scale)
            .reflection_scale(preset.reflection_scale)
            .specular_scale(preset.specular_scale)
            .f_0(preset.f_0)
            .amplitude_scale(preset.amplitude_scale)
            .amplitude(preset.amplitude)
            .wind_vector(preset.wind_vector)
            .roughness(preset.roughness)
            .max_w(preset.max_w)
            .l_small(preset.l_small)
            .fft_subdivisions(preset.fft_subdivisions)
            .fft_size(preset.fft_size)
    }

    pub fn from_preset(preset: &OceanPreset) -> Self {
        Self::default().apply_preset(preset)
    }

    pub fn build(self) -> OceanSettingsUniform {
        assert!(self.fft_subdivisions.is_power_of_two());
        assert!(self.fft_size > 0.0);
        assert!(self.mesh_size > 0.0);

        let pass_num = self.fft_subdivisions.ilog2();
        let wave_scale = self.mesh_size / self.fft_size;

        OceanSettingsUniform {
            mesh_size: self.mesh_size,
            mesh_subdivisions: self.mesh_subdivisions,
            fft_size: self.fft_size,
            fft_subdivisions: self.fft_subdivisions,
            pass_num,
            time_scale: self.time_scale,
            ocean_seed: self.ocean_seed,
            chop_scale: self.chop_scale,
            amplitude_scale: self.amplitude_scale,
            wave_scale,
            wind_vector: self.wind_vector,
            amplitude: self.amplitude,
            l_small: self.l_small,
            max_w: self.max_w,
            fovy: self.fovy,
            zfar: self.zfar,
            cam_speed: self.cam_speed,
            cam_boost: self.cam_boost,
            cam_sensitivity: self.cam_sensitivity,
            roughness: self.roughness,
            f_0: self.f_0,
            specular_scale: self.specular_scale,
            reflection_scale: self.reflection_scale,
            foam_scale: self.foam_scale,
            sss_distortion_scale: self.sss_distortion_scale,
            caustic_scale: self.caustic_scale,
            caustic_size: self.caustic_size,
            caustic_speed: self.caustic_speed,
            caustic_intensity: self.caustic_intensity,
            caustic_octaves: self.caustic_octaves,
            caustic_depth: self.caustic_depth,
            caustic_max_distance: self.caustic_max_distance,
            micro_normal_strength: self.micro_normal_strength,
            foam_threshold: self.foam_threshold,
            foam_speed: self.foam_speed,
            foam_roughness: self.foam_roughness,
            _pad2: [0.0, 0.0, 0.0],
            caustic_color_tint: self.caustic_color_tint,
            deep_color: self.deep_color,
            shallow_color: self.shallow_color,
            sss_color: self.sss_color,
            sun_color: self.sun_color,
        }
    }
}

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
    pub micro_normal_strength: f32,
    pub foam_threshold: f32,
    pub foam_speed: f32,
    pub foam_roughness: f32,
    pub caustic_scale: f32,
    pub caustic_size: f32,
    pub caustic_speed: f32,
    pub caustic_intensity: f32,
    pub caustic_octaves: u32,
    pub caustic_depth: f32,
    pub caustic_max_distance: f32,
    pub caustic_color_tint: [f32; 4],
    pub deep_color: [f32; 4],
    pub shallow_color: [f32; 4],
    pub sss_color: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_color_day_zenith: [f32; 4],
    pub sky_color_day_horizon: [f32; 4],
    pub sky_color_night_zenith: [f32; 4],
    pub sky_color_night_horizon: [f32; 4],
    pub sky_color_sunset_orange: [f32; 4],
    pub sky_color_sunset_pink: [f32; 4],
    pub sky_color_horizon_glow: [f32; 4],
    pub moon_color_lit: [f32; 4],
    pub moon_color_dark: [f32; 4],
    pub cloud_color_night: [f32; 4],
    pub cloud_color_day: [f32; 4],
    pub sun_offset_z: f32,
    pub sun_size_inner: f32,
    pub sun_size_outer: f32,
    pub sun_halo_power: f32,
    pub moon_radius: f32,
    pub moon_dist: f32,
    pub moon_phase_offset: [f32; 3],
    pub moon_crater_scale: f32,
    pub moon_halo_power: f32,
    pub star_count: f32,
    pub star_threshold: f32,
    pub star_blink_speed: f32,
    pub cloud_speed: f32,
    pub cloud_density_low: f32,
    pub cloud_density_high: f32,
    pub daynight_cycle: f32,
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
        let builder = OceanSettingsBuilder::from_uniform(&ocean_settings);
        let mut preset = OceanPreset::from_builder(builder);

        preset.preset_name = preset_name.to_string();
        preset.author = author.to_string();
        preset.description = description.to_string();
        preset.created_at = chrono::Local::now().timestamp();
        preset.last_modified_at = chrono::Local::now().timestamp();

        let json_path = directory.join(format!("{preset_name}.json"));
        let json_data =
            serde_json::to_string_pretty(&preset).expect("Couldnt format to a json string");
        write(json_path, &json_data).expect("Couldnt write to file");
    }

    pub fn modify_preset(
        preset_name: &str,
        directory: &Path,
        current_ocean_settings: OceanPreset,
        new_ocean_settings: OceanSettingsUniform,
    ) {
        let builder = OceanSettingsBuilder::from_uniform(&new_ocean_settings);
        let mut preset = OceanPreset::from_builder(builder);

        preset.preset_name = preset_name.to_string();
        preset.author = current_ocean_settings.author;
        preset.description = current_ocean_settings.description;
        preset.created_at = current_ocean_settings.created_at;
        preset.last_modified_at = chrono::Local::now().timestamp();

        let json_path = directory.join(format!("{preset_name}.json"));
        let json_data =
            serde_json::to_string_pretty(&preset).expect("Couldnt format to a json string");
        write(json_path, &json_data).expect("Couldnt write to file");
    }

    pub fn from_builder(builder: OceanSettingsBuilder) -> Self {
        Self {
            preset_name: "".into(),
            author: "".into(),
            description: "".into(),
            created_at: 0,
            last_modified_at: 0,
            fft_size: builder.fft_size,
            fft_subdivisions: builder.fft_subdivisions,
            time_scale: builder.time_scale,
            ocean_seed: builder.ocean_seed,
            chop_scale: builder.chop_scale,
            amplitude_scale: builder.amplitude_scale,
            wave_scale: builder.mesh_size / builder.fft_size,
            wind_vector: builder.wind_vector,
            amplitude: builder.amplitude,
            l_small: builder.l_small,
            max_w: builder.max_w,
            fovy: builder.fovy,
            zfar: builder.zfar,
            cam_speed: builder.cam_speed,
            cam_boost: builder.cam_boost,
            cam_sensitivity: builder.cam_sensitivity,
            roughness: builder.roughness,
            f_0: builder.f_0,
            specular_scale: builder.specular_scale,
            reflection_scale: builder.reflection_scale,
            foam_scale: builder.foam_scale,
            sss_distortion_scale: builder.sss_distortion_scale,
            micro_normal_strength: builder.micro_normal_strength,
            foam_threshold: builder.foam_threshold,
            foam_speed: builder.foam_speed,
            foam_roughness: builder.foam_roughness,
            caustic_scale: builder.caustic_scale,
            caustic_size: builder.caustic_size,
            caustic_speed: builder.caustic_speed,
            caustic_intensity: builder.caustic_intensity,
            caustic_octaves: builder.caustic_octaves,
            caustic_depth: builder.caustic_depth,
            caustic_max_distance: builder.caustic_max_distance,
            caustic_color_tint: builder.caustic_color_tint,
            deep_color: builder.deep_color,
            shallow_color: builder.shallow_color,
            sss_color: builder.sss_color,
            sun_color: builder.sun_color,
            sky_color_day_zenith: builder.sky_color_day_zenith,
            sky_color_day_horizon: builder.sky_color_day_horizon,
            sky_color_night_zenith: builder.sky_color_night_zenith,
            sky_color_night_horizon: builder.sky_color_night_horizon,
            sky_color_sunset_orange: builder.sky_color_sunset_orange,
            sky_color_sunset_pink: builder.sky_color_sunset_pink,
            sky_color_horizon_glow: builder.sky_color_horizon_glow,
            moon_color_lit: builder.moon_color_lit,
            moon_color_dark: builder.moon_color_dark,
            cloud_color_night: builder.cloud_color_night,
            cloud_color_day: builder.cloud_color_day,
            sun_offset_z: builder.sun_offset_z,
            sun_size_inner: builder.sun_size_inner,
            sun_size_outer: builder.sun_size_outer,
            sun_halo_power: builder.sun_halo_power,
            moon_radius: builder.moon_radius,
            moon_dist: builder.moon_dist,
            moon_phase_offset: builder.moon_phase_offset,
            moon_crater_scale: builder.moon_crater_scale,
            moon_halo_power: builder.moon_halo_power,
            star_count: builder.star_count,
            star_threshold: builder.star_threshold,
            star_blink_speed: builder.star_blink_speed,
            cloud_speed: builder.cloud_speed,
            cloud_density_low: builder.cloud_density_low,
            cloud_density_high: builder.cloud_density_high,
            daynight_cycle: builder.daynight_cycle,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OceanSettingsUniform {
    pub deep_color: [f32; 4],
    pub shallow_color: [f32; 4],
    pub sss_color: [f32; 4],
    pub caustic_color_tint: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_color_day_zenith: [f32; 4],
    pub sky_color_day_horizon: [f32; 4],
    pub sky_color_night_zenith: [f32; 4],
    pub sky_color_night_horizon: [f32; 4],
    pub sky_color_sunset_orange: [f32; 4],
    pub sky_color_sunset_pink: [f32; 4],
    pub sky_color_horizon_glow: [f32; 4],
    pub moon_color_lit: [f32; 4],
    pub moon_color_dark: [f32; 4],
    pub cloud_color_night: [f32; 4],
    pub cloud_color_day: [f32; 4],
    pub wind_vector: [f32; 2],
    pub _pad_vec2: [f32; 2],
    pub moon_phase_offset: [f32; 3],
    pub _pad_moon: f32,
    pub mesh_size: f32,
    pub fft_size: f32,
    pub time_scale: f32,
    pub chop_scale: f32,
    pub amplitude_scale: f32,
    pub wave_scale: f32,
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
    pub caustic_depth: f32,
    pub caustic_max_distance: f32,
    pub micro_normal_strength: f32,
    pub foam_threshold: f32,
    pub foam_speed: f32,
    pub foam_roughness: f32,
    pub daynight_cycle: f32,
    pub sun_offset_z: f32,
    pub sun_size_inner: f32,
    pub sun_size_outer: f32,
    pub sun_halo_power: f32,
    pub moon_radius: f32,
    pub moon_dist: f32,
    pub moon_crater_scale: f32,
    pub moon_halo_power: f32,
    pub star_count: f32,
    pub star_threshold: f32,
    pub star_blink_speed: f32,
    pub cloud_speed: f32,
    pub cloud_density_low: f32,
    pub cloud_density_high: f32,
    pub mesh_subdivisions: u32,
    pub fft_subdivisions: u32,
    pub pass_num: u32,
    pub ocean_seed: u32,
    pub caustic_octaves: u32,
    pub _pad_final: [u32; 6],
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
    daynight_cycle: f32,
    caustic_color_tint: [f32; 4],
    deep_color: [f32; 4],
    shallow_color: [f32; 4],
    sss_color: [f32; 4],
    sun_color: [f32; 4],
    sky_color_day_zenith: [f32; 4],
    sky_color_day_horizon: [f32; 4],
    sky_color_night_zenith: [f32; 4],
    sky_color_night_horizon: [f32; 4],
    sky_color_sunset_orange: [f32; 4],
    sky_color_sunset_pink: [f32; 4],
    sky_color_horizon_glow: [f32; 4],
    moon_color_lit: [f32; 4],
    moon_color_dark: [f32; 4],
    cloud_color_night: [f32; 4],
    cloud_color_day: [f32; 4],
    sun_offset_z: f32,
    sun_size_inner: f32,
    sun_size_outer: f32,
    sun_halo_power: f32,
    moon_radius: f32,
    moon_dist: f32,
    moon_phase_offset: [f32; 3],
    moon_crater_scale: f32,
    moon_halo_power: f32,
    star_count: f32,
    star_threshold: f32,
    star_blink_speed: f32,
    cloud_speed: f32,
    cloud_density_low: f32,
    cloud_density_high: f32,
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
            daynight_cycle: 0.5,
            caustic_color_tint: [0.95, 1.0, 1.05, 1.0],
            deep_color: [0.0, 0.01, 0.05, 1.0],
            shallow_color: [0.0, 0.06, 0.09, 1.0],
            sss_color: [0.0, 0.4, 0.3, 1.0],
            sun_color: [1.0, 0.9, 0.8, 1.0],
            sky_color_day_zenith: [0.1, 0.35, 0.75, 1.0],
            sky_color_day_horizon: [0.6, 0.75, 0.95, 1.0],
            sky_color_night_zenith: [0.002, 0.004, 0.012, 1.0],
            sky_color_night_horizon: [0.01, 0.012, 0.02, 1.0],
            sky_color_sunset_orange: [1.0, 0.4, 0.1, 1.0],
            sky_color_sunset_pink: [1.0, 0.3, 0.5, 1.0],
            sky_color_horizon_glow: [1.0, 0.5, 0.2, 1.0],
            moon_color_lit: [0.8, 0.8, 0.75, 1.0],
            moon_color_dark: [0.015, 0.015, 0.025, 1.0],
            cloud_color_night: [0.02, 0.02, 0.04, 1.0],
            cloud_color_day: [0.7, 0.7, 0.75, 1.0],
            sun_offset_z: -0.3,
            sun_size_inner: 0.9995,
            sun_size_outer: 0.9998,
            sun_halo_power: 500.0,
            moon_radius: 0.015,
            moon_dist: 100.0,
            moon_phase_offset: [0.2, 0.0, 0.0],
            moon_crater_scale: 4.0,
            moon_halo_power: 400.0,
            star_count: 2000.0,
            star_threshold: 0.9996,
            star_blink_speed: 2.0,
            cloud_speed: 0.05,
            cloud_density_low: 0.4,
            cloud_density_high: 0.8,
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

    pub fn daynight_cycle(mut self, v: f32) -> Self {
        self.daynight_cycle = v;
        self
    }

    pub fn sky_color_day_zenith(mut self, v: [f32; 4]) -> Self {
        self.sky_color_day_zenith = v;
        self
    }

    pub fn sky_color_day_horizon(mut self, v: [f32; 4]) -> Self {
        self.sky_color_day_horizon = v;
        self
    }

    pub fn sky_color_night_zenith(mut self, v: [f32; 4]) -> Self {
        self.sky_color_night_zenith = v;
        self
    }

    pub fn sky_color_night_horizon(mut self, v: [f32; 4]) -> Self {
        self.sky_color_night_horizon = v;
        self
    }

    pub fn sky_color_sunset_orange(mut self, v: [f32; 4]) -> Self {
        self.sky_color_sunset_orange = v;
        self
    }

    pub fn sky_color_sunset_pink(mut self, v: [f32; 4]) -> Self {
        self.sky_color_sunset_pink = v;
        self
    }

    pub fn sky_color_horizon_glow(mut self, v: [f32; 4]) -> Self {
        self.sky_color_horizon_glow = v;
        self
    }

    pub fn moon_color_lit(mut self, v: [f32; 4]) -> Self {
        self.moon_color_lit = v;
        self
    }

    pub fn moon_color_dark(mut self, v: [f32; 4]) -> Self {
        self.moon_color_dark = v;
        self
    }

    pub fn cloud_color_night(mut self, v: [f32; 4]) -> Self {
        self.cloud_color_night = v;
        self
    }

    pub fn cloud_color_day(mut self, v: [f32; 4]) -> Self {
        self.cloud_color_day = v;
        self
    }

    pub fn apply_preset(mut self, preset: &OceanPreset) -> Self {
        self.sun_color = preset.sun_color;
        self.sss_color = preset.sss_color;
        self.shallow_color = preset.shallow_color;
        self.deep_color = preset.deep_color;
        self.caustic_color_tint = preset.caustic_color_tint;
        self.foam_roughness = preset.foam_roughness;
        self.foam_speed = preset.foam_speed;
        self.foam_threshold = preset.foam_threshold;
        self.foam_scale = preset.foam_scale;
        self.micro_normal_strength = preset.micro_normal_strength;
        self.caustic_max_distance = preset.caustic_max_distance;
        self.caustic_depth = preset.caustic_depth;
        self.caustic_octaves = preset.caustic_octaves;
        self.caustic_intensity = preset.caustic_intensity;
        self.caustic_speed = preset.caustic_speed;
        self.caustic_size = preset.caustic_size;
        self.caustic_scale = preset.caustic_scale;
        self.sss_distortion_scale = preset.sss_distortion_scale;
        self.reflection_scale = preset.reflection_scale;
        self.specular_scale = preset.specular_scale;
        self.f_0 = preset.f_0;
        self.amplitude_scale = preset.amplitude_scale;
        self.amplitude = preset.amplitude;
        self.wind_vector = preset.wind_vector;
        self.roughness = preset.roughness;
        self.max_w = preset.max_w;
        self.l_small = preset.l_small;
        self.fft_subdivisions = preset.fft_subdivisions;
        self.fft_size = preset.fft_size;
        self.daynight_cycle = preset.daynight_cycle;
        self.sky_color_day_zenith = preset.sky_color_day_zenith;
        self.sky_color_day_horizon = preset.sky_color_day_horizon;
        self.sky_color_night_zenith = preset.sky_color_night_zenith;
        self.sky_color_night_horizon = preset.sky_color_night_horizon;
        self.sky_color_sunset_orange = preset.sky_color_sunset_orange;
        self.sky_color_sunset_pink = preset.sky_color_sunset_pink;
        self.sky_color_horizon_glow = preset.sky_color_horizon_glow;
        self.moon_color_lit = preset.moon_color_lit;
        self.moon_color_dark = preset.moon_color_dark;
        self.cloud_color_night = preset.cloud_color_night;
        self.cloud_color_day = preset.cloud_color_day;
        self.sun_offset_z = preset.sun_offset_z;
        self.sun_size_inner = preset.sun_size_inner;
        self.sun_size_outer = preset.sun_size_outer;
        self.sun_halo_power = preset.sun_halo_power;
        self.moon_radius = preset.moon_radius;
        self.moon_dist = preset.moon_dist;
        self.moon_phase_offset = preset.moon_phase_offset;
        self.moon_crater_scale = preset.moon_crater_scale;
        self.moon_halo_power = preset.moon_halo_power;
        self.star_count = preset.star_count;
        self.star_threshold = preset.star_threshold;
        self.star_blink_speed = preset.star_blink_speed;
        self.cloud_speed = preset.cloud_speed;
        self.cloud_density_low = preset.cloud_density_low;
        self.cloud_density_high = preset.cloud_density_high;

        self
    }

    pub fn from_preset(preset: &OceanPreset) -> Self {
        Self::default().apply_preset(preset)
    }

    pub fn from_uniform(ocean_uniform: &OceanSettingsUniform) -> Self {
        Self {
            mesh_size: ocean_uniform.mesh_size,
            mesh_subdivisions: ocean_uniform.mesh_subdivisions,
            fft_size: ocean_uniform.fft_size,
            fft_subdivisions: ocean_uniform.fft_subdivisions,
            time_scale: ocean_uniform.time_scale,
            ocean_seed: ocean_uniform.ocean_seed,
            chop_scale: ocean_uniform.chop_scale,
            amplitude_scale: ocean_uniform.amplitude_scale,
            wind_vector: ocean_uniform.wind_vector,
            amplitude: ocean_uniform.amplitude,
            l_small: ocean_uniform.l_small,
            max_w: ocean_uniform.max_w,
            fovy: ocean_uniform.fovy,
            zfar: ocean_uniform.zfar,
            cam_speed: ocean_uniform.cam_speed,
            cam_boost: ocean_uniform.cam_boost,
            cam_sensitivity: ocean_uniform.cam_sensitivity,
            roughness: ocean_uniform.roughness,
            f_0: ocean_uniform.f_0,
            specular_scale: ocean_uniform.specular_scale,
            reflection_scale: ocean_uniform.reflection_scale,
            foam_scale: ocean_uniform.foam_scale,
            sss_distortion_scale: ocean_uniform.sss_distortion_scale,
            caustic_scale: ocean_uniform.caustic_scale,
            caustic_size: ocean_uniform.caustic_size,
            caustic_speed: ocean_uniform.caustic_speed,
            caustic_intensity: ocean_uniform.caustic_intensity,
            caustic_octaves: ocean_uniform.caustic_octaves,
            caustic_depth: ocean_uniform.caustic_depth,
            caustic_max_distance: ocean_uniform.caustic_max_distance,
            micro_normal_strength: ocean_uniform.micro_normal_strength,
            foam_threshold: ocean_uniform.foam_threshold,
            foam_speed: ocean_uniform.foam_speed,
            foam_roughness: ocean_uniform.foam_roughness,
            daynight_cycle: ocean_uniform.daynight_cycle,
            caustic_color_tint: ocean_uniform.caustic_color_tint,
            deep_color: ocean_uniform.deep_color,
            shallow_color: ocean_uniform.shallow_color,
            sss_color: ocean_uniform.sss_color,
            sun_color: ocean_uniform.sun_color,
            sky_color_day_zenith: ocean_uniform.sky_color_day_zenith,
            sky_color_day_horizon: ocean_uniform.sky_color_day_horizon,
            sky_color_night_zenith: ocean_uniform.sky_color_night_zenith,
            sky_color_night_horizon: ocean_uniform.sky_color_night_horizon,
            sky_color_sunset_orange: ocean_uniform.sky_color_sunset_orange,
            sky_color_sunset_pink: ocean_uniform.sky_color_sunset_pink,
            sky_color_horizon_glow: ocean_uniform.sky_color_horizon_glow,
            moon_color_lit: ocean_uniform.moon_color_lit,
            moon_color_dark: ocean_uniform.moon_color_dark,
            cloud_color_night: ocean_uniform.cloud_color_night,
            cloud_color_day: ocean_uniform.cloud_color_day,
            sun_offset_z: ocean_uniform.sun_offset_z,
            sun_size_inner: ocean_uniform.sun_size_inner,
            sun_size_outer: ocean_uniform.sun_size_outer,
            sun_halo_power: ocean_uniform.sun_halo_power,
            moon_radius: ocean_uniform.moon_radius,
            moon_dist: ocean_uniform.moon_dist,
            moon_phase_offset: ocean_uniform.moon_phase_offset,
            moon_crater_scale: ocean_uniform.moon_crater_scale,
            moon_halo_power: ocean_uniform.moon_halo_power,
            star_count: ocean_uniform.star_count,
            star_threshold: ocean_uniform.star_threshold,
            star_blink_speed: ocean_uniform.star_blink_speed,
            cloud_speed: ocean_uniform.cloud_speed,
            cloud_density_low: ocean_uniform.cloud_density_low,
            cloud_density_high: ocean_uniform.cloud_density_high,
        }
    }

    pub fn build(self) -> OceanSettingsUniform {
        assert!(self.fft_subdivisions.is_power_of_two());
        assert!(self.fft_size > 0.0);
        assert!(self.mesh_size > 0.0);

        let pass_num = self.fft_subdivisions.ilog2();
        let wave_scale = self.mesh_size / self.fft_size;

        OceanSettingsUniform {
            deep_color: self.deep_color,
            shallow_color: self.shallow_color,
            sss_color: self.sss_color,
            caustic_color_tint: self.caustic_color_tint,
            sun_color: self.sun_color,
            sky_color_day_zenith: self.sky_color_day_zenith,
            sky_color_day_horizon: self.sky_color_day_horizon,
            sky_color_night_zenith: self.sky_color_night_zenith,
            sky_color_night_horizon: self.sky_color_night_horizon,
            sky_color_sunset_orange: self.sky_color_sunset_orange,
            sky_color_sunset_pink: self.sky_color_sunset_pink,
            sky_color_horizon_glow: self.sky_color_horizon_glow,
            moon_color_lit: self.moon_color_lit,
            moon_color_dark: self.moon_color_dark,
            cloud_color_night: self.cloud_color_night,
            cloud_color_day: self.cloud_color_day,
            wind_vector: self.wind_vector,
            _pad_vec2: [0.0; 2],
            moon_phase_offset: self.moon_phase_offset,
            _pad_moon: 0.0,
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
            daynight_cycle: self.daynight_cycle,
            sun_offset_z: self.sun_offset_z,
            sun_size_inner: self.sun_size_inner,
            sun_size_outer: self.sun_size_outer,
            sun_halo_power: self.sun_halo_power,
            moon_radius: self.moon_radius,
            moon_dist: self.moon_dist,
            moon_crater_scale: self.moon_crater_scale,
            moon_halo_power: self.moon_halo_power,
            star_count: self.star_count,
            star_threshold: self.star_threshold,
            star_blink_speed: self.star_blink_speed,
            cloud_speed: self.cloud_speed,
            cloud_density_low: self.cloud_density_low,
            cloud_density_high: self.cloud_density_high,
            _pad_final: [0, 0, 0, 0, 0, 0],
        }
    }
}

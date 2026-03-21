use crate::settings::builder::OceanSettingsBuilder;
use crate::settings::uniform::OceanSettingsUniform;
use anyhow::Result;
use std::{
    fs::{self, read_to_string, write},
    path::Path,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct OceanPreset {
    pub preset_name: String,
    pub author: String,
    pub description: String,
    pub created_at: i64,
    pub last_modified_at: i64,
    pub mesh_size: f32,
    pub mesh_subdivisions: u32,
    pub fft_subdivisions: u32,
    pub time_scale: f32,
    pub ocean_seed: u32,
    pub chop_scale: f32,
    pub amplitude_scale: f32,
    pub wind_vector: [f32; 2],
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
    pub star_size: f32,
    pub star_threshold: f32,
    pub star_blink_speed: f32,
    pub aurora_strength: f32,
    pub cloud_speed: f32,
    pub cloud_density_low: f32,
    pub cloud_density_high: f32,
    pub daynight_cycle: f32,
    pub cascade_data: Vec<CascadePreset>,
    pub sunset_scatter_color: [f32; 3],
    pub sunset_scatter_intensity: f32,
    pub foam_base_color: [f32; 3],
    pub sss_min_height: f32,
    pub sss_max_height: f32,
    pub sss_power: f32,
    pub sss_intensity: f32,
    pub detail_fade: f32,
    pub ambient_scale: f32,
    pub blend_strength: f32,
    pub bloom_scale: f32,
    pub reflection_min: f32,
    pub reflection_max: f32,
    pub moon_light_dim: f32,
    pub sky_zenith_gradient_exp: f32,
    pub horizon_glow_mult: f32,
    pub sunset_orange_weight: f32,
    pub sunset_intensity: f32,
    pub sun_halo_intensity: f32,
    pub moon_halo_intensity: f32,
    pub micro_uv_freq: f32,
    pub micro_time_freq: f32,
    pub micro_strength_mod: f32,
    pub foam_crest_width: f32,
    pub caustic_aberration: f32,
    pub caustic_smooth_low: f32,
    pub caustic_smooth_high: f32,
    pub aurora_brightness: f32,
    pub aurora_y_threshold: f32,
    pub water_brightness_mod: f32,
    pub decay_factor: f32,
    pub dissipation_factor: f32,
    pub warp_uv_scale: f32,
    pub warp_strength: f32,
    pub warp_time_scale: f32,
    pub foam_octaves: u32,
    pub foam_power: f32,
    pub hash_scale: f32,
    pub hash_dot: f32,
    pub steepness_threshold_low: f32,
    pub steepness_threshold_high: f32,
    pub y_displacement_weight: f32,
    pub wave_epsilon: f32,
    pub wave_height_exp: f32,
    pub wave_height_sharp: f32,
    pub night_water_floor: f32,
    pub fresnel_sky_cap: f32,
    pub caustic_sss_blend: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CascadePreset {
    pub fft_size: f32,
    pub amplitude: f32,
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
            fft_subdivisions: builder.fft_subdivisions,
            mesh_subdivisions: builder.mesh_subdivisions,
            mesh_size: builder.mesh_size,
            time_scale: builder.time_scale,
            ocean_seed: builder.ocean_seed,
            chop_scale: builder.chop_scale,
            amplitude_scale: builder.amplitude_scale,
            wind_vector: builder.wind_vector,
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
            star_size: builder.star_size,
            aurora_strength: builder.aurora_strength,
            cloud_speed: builder.cloud_speed,
            cloud_density_low: builder.cloud_density_low,
            cloud_density_high: builder.cloud_density_high,
            daynight_cycle: builder.daynight_cycle,
            cascade_data: builder.cascade_data,
            sunset_scatter_color: builder.sunset_scatter_color,
            sunset_scatter_intensity: builder.sunset_scatter_intensity,
            foam_base_color: builder.foam_base_color,
            sss_min_height: builder.sss_min_height,
            sss_max_height: builder.sss_max_height,
            sss_power: builder.sss_power,
            sss_intensity: builder.sss_intensity,
            detail_fade: builder.detail_fade,
            ambient_scale: builder.ambient_scale,
            blend_strength: builder.blend_strength,
            bloom_scale: builder.bloom_scale,
            reflection_min: builder.reflection_min,
            reflection_max: builder.reflection_max,
            moon_light_dim: builder.moon_light_dim,
            sky_zenith_gradient_exp: builder.sky_zenith_gradient_exp,
            horizon_glow_mult: builder.horizon_glow_mult,
            sunset_orange_weight: builder.sunset_orange_weight,
            sunset_intensity: builder.sunset_intensity,
            sun_halo_intensity: builder.sun_halo_intensity,
            moon_halo_intensity: builder.moon_halo_intensity,
            micro_uv_freq: builder.micro_uv_freq,
            micro_time_freq: builder.micro_time_freq,
            micro_strength_mod: builder.micro_strength_mod,
            foam_crest_width: builder.foam_crest_width,
            caustic_aberration: builder.caustic_aberration,
            caustic_smooth_low: builder.caustic_smooth_low,
            caustic_smooth_high: builder.caustic_smooth_high,
            aurora_brightness: builder.aurora_brightness,
            aurora_y_threshold: builder.aurora_y_threshold,
            water_brightness_mod: builder.water_brightness_mod,
            decay_factor: builder.decay_factor,
            dissipation_factor: builder.dissipation_factor,
            warp_uv_scale: builder.warp_uv_scale,
            warp_strength: builder.warp_strength,
            warp_time_scale: builder.warp_time_scale,
            foam_octaves: builder.foam_octaves,
            foam_power: builder.foam_power,
            hash_scale: builder.hash_scale,
            hash_dot: builder.hash_dot,
            steepness_threshold_low: builder.steepness_threshold_low,
            steepness_threshold_high: builder.steepness_threshold_high,
            y_displacement_weight: builder.y_displacement_weight,
            wave_epsilon: builder.wave_epsilon,
            wave_height_exp: builder.wave_height_exp,
            wave_height_sharp: builder.wave_height_sharp,
            night_water_floor: builder.night_water_floor,
            fresnel_sky_cap: builder.fresnel_sky_cap,
            caustic_sss_blend: builder.caustic_sss_blend,
        }
    }
}

use anyhow::Result;
use std::{
    fs::{self, read_to_string, write},
    path::Path,
};

pub const MAX_CASCADES: usize = 6;
pub const TOTAL_BINDINGS: usize = 1 + MAX_CASCADES * 4;

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
    pub star_size: f32,
    pub aurora_strength: f32,
    pub pad_b: [u32; 4],
    pub cascade_data: [[f32; 4]; MAX_CASCADES],
    pub cascade_count: u32,
    pub _pad_cascade: [u32; 3],
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
    pub _pad_final: [f32; 9],
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
    star_size: f32,
    aurora_strength: f32,
    cloud_speed: f32,
    cloud_density_low: f32,
    cloud_density_high: f32,
    cascade_data: Vec<CascadePreset>,
    sunset_scatter_color: [f32; 3],
    sunset_scatter_intensity: f32,
    foam_base_color: [f32; 3],
    sss_min_height: f32,
    sss_max_height: f32,
    sss_power: f32,
    sss_intensity: f32,
    detail_fade: f32,
    ambient_scale: f32,
    blend_strength: f32,
    bloom_scale: f32,
    reflection_min: f32,
    reflection_max: f32,
    moon_light_dim: f32,
    sky_zenith_gradient_exp: f32,
    horizon_glow_mult: f32,
    sunset_orange_weight: f32,
    sunset_intensity: f32,
    sun_halo_intensity: f32,
    moon_halo_intensity: f32,
    micro_uv_freq: f32,
    micro_time_freq: f32,
    micro_strength_mod: f32,
    foam_crest_width: f32,
    caustic_aberration: f32,
    caustic_smooth_low: f32,
    caustic_smooth_high: f32,
    aurora_brightness: f32,
    aurora_y_threshold: f32,
    water_brightness_mod: f32,
    decay_factor: f32,
    dissipation_factor: f32,
    warp_uv_scale: f32,
    warp_strength: f32,
    warp_time_scale: f32,
    foam_octaves: u32,
    foam_power: f32,
    hash_scale: f32,
    hash_dot: f32,
    steepness_threshold_low: f32,
    steepness_threshold_high: f32,
    y_displacement_weight: f32,
    wave_epsilon: f32,
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
            star_size: 20.0,
            aurora_strength: 1.0,
            cloud_speed: 0.05,
            cloud_density_low: 0.4,
            cloud_density_high: 0.8,
            cascade_data: vec![
                CascadePreset {
                    fft_size: 1000.0,
                    amplitude: 1.0,
                },
                CascadePreset {
                    fft_size: 250.0,
                    amplitude: 0.3,
                },
                CascadePreset {
                    fft_size: 50.0,
                    amplitude: 0.08,
                },
            ],
            sunset_scatter_color: [1.05, 0.88, 0.72],
            sunset_scatter_intensity: 0.55,
            foam_base_color: [0.95, 0.98, 0.92],
            sss_min_height: -0.5,
            sss_max_height: 1.5,
            sss_power: 8.0,
            sss_intensity: 1.0,
            detail_fade: 800.0,
            ambient_scale: 0.08,
            blend_strength: 0.4,
            bloom_scale: 0.3,
            reflection_min: 0.2,
            reflection_max: 0.9,
            moon_light_dim: 0.25,
            sky_zenith_gradient_exp: 1.5,
            horizon_glow_mult: 1.2,
            sunset_orange_weight: 0.55,
            sunset_intensity: 3.8,
            sun_halo_intensity: 0.02,
            moon_halo_intensity: 0.1,
            micro_uv_freq: 0.01,
            micro_time_freq: 0.001,
            micro_strength_mod: 0.05,
            foam_crest_width: 0.2,
            caustic_aberration: 0.01,
            caustic_smooth_low: 0.6,
            caustic_smooth_high: 1.0,
            aurora_brightness: 2.5,
            aurora_y_threshold: 0.04,
            water_brightness_mod: 0.8,
            decay_factor: 0.98,
            dissipation_factor: 0.99,
            warp_uv_scale: 0.5,
            warp_strength: 0.5,
            warp_time_scale: 0.1,
            foam_octaves: 3,
            foam_power: 1.5,
            hash_scale: 0.1031,
            hash_dot: 33.33,
            steepness_threshold_low: 0.1,
            steepness_threshold_high: 0.8,
            y_displacement_weight: 0.5,
            wave_epsilon: 0.0001,
        }
    }
}

impl OceanSettingsBuilder {
    pub fn mesh_size(mut self, v: f32) -> Self {
        self.mesh_size = v;
        self
    }
    pub fn ocean_seed(mut self, v: u32) -> Self {
        self.ocean_seed = v;
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
    pub fn roughness(mut self, v: f32) -> Self {
        self.roughness = v;
        self
    }
    pub fn daynight_cycle(mut self, v: f32) -> Self {
        self.daynight_cycle = v;
        self
    }
    pub fn sunset_intensity(mut self, v: f32) -> Self {
        self.sunset_intensity = v;
        self
    }
    pub fn water_brightness_mod(mut self, v: f32) -> Self {
        self.water_brightness_mod = v;
        self
    }

    pub fn from_preset(preset: &OceanPreset) -> Self {
        let mut builder = Self::default();
        builder.sun_color = preset.sun_color;
        builder.sss_color = preset.sss_color;
        builder.shallow_color = preset.shallow_color;
        builder.deep_color = preset.deep_color;
        builder.caustic_color_tint = preset.caustic_color_tint;
        builder.foam_roughness = preset.foam_roughness;
        builder.foam_speed = preset.foam_speed;
        builder.foam_threshold = preset.foam_threshold;
        builder.foam_scale = preset.foam_scale;
        builder.micro_normal_strength = preset.micro_normal_strength;
        builder.caustic_max_distance = preset.caustic_max_distance;
        builder.caustic_depth = preset.caustic_depth;
        builder.caustic_octaves = preset.caustic_octaves;
        builder.caustic_intensity = preset.caustic_intensity;
        builder.caustic_speed = preset.caustic_speed;
        builder.caustic_size = preset.caustic_size;
        builder.caustic_scale = preset.caustic_scale;
        builder.sss_distortion_scale = preset.sss_distortion_scale;
        builder.reflection_scale = preset.reflection_scale;
        builder.specular_scale = preset.specular_scale;
        builder.f_0 = preset.f_0;
        builder.amplitude_scale = preset.amplitude_scale;
        builder.amplitude = preset.amplitude;
        builder.wind_vector = preset.wind_vector;
        builder.roughness = preset.roughness;
        builder.max_w = preset.max_w;
        builder.l_small = preset.l_small;
        builder.fft_subdivisions = preset.fft_subdivisions;
        builder.fft_size = preset.fft_size;
        builder.daynight_cycle = preset.daynight_cycle;
        builder.sky_color_day_zenith = preset.sky_color_day_zenith;
        builder.sky_color_day_horizon = preset.sky_color_day_horizon;
        builder.sky_color_night_zenith = preset.sky_color_night_zenith;
        builder.sky_color_night_horizon = preset.sky_color_night_horizon;
        builder.sky_color_sunset_orange = preset.sky_color_sunset_orange;
        builder.sky_color_sunset_pink = preset.sky_color_sunset_pink;
        builder.sky_color_horizon_glow = preset.sky_color_horizon_glow;
        builder.moon_color_lit = preset.moon_color_lit;
        builder.moon_color_dark = preset.moon_color_dark;
        builder.cloud_color_night = preset.cloud_color_night;
        builder.cloud_color_day = preset.cloud_color_day;
        builder.sun_offset_z = preset.sun_offset_z;
        builder.sun_size_inner = preset.sun_size_inner;
        builder.sun_size_outer = preset.sun_size_outer;
        builder.sun_halo_power = preset.sun_halo_power;
        builder.moon_radius = preset.moon_radius;
        builder.moon_dist = preset.moon_dist;
        builder.moon_phase_offset = preset.moon_phase_offset;
        builder.moon_crater_scale = preset.moon_crater_scale;
        builder.moon_halo_power = preset.moon_halo_power;
        builder.star_count = preset.star_count;
        builder.star_threshold = preset.star_threshold;
        builder.star_blink_speed = preset.star_blink_speed;
        builder.cloud_speed = preset.cloud_speed;
        builder.cloud_density_low = preset.cloud_density_low;
        builder.cloud_density_high = preset.cloud_density_high;
        builder.cascade_data = preset.cascade_data.clone();
        builder.sunset_scatter_color = preset.sunset_scatter_color;
        builder.sunset_scatter_intensity = preset.sunset_scatter_intensity;
        builder.foam_base_color = preset.foam_base_color;
        builder.sss_min_height = preset.sss_min_height;
        builder.sss_max_height = preset.sss_max_height;
        builder.sss_power = preset.sss_power;
        builder.sss_intensity = preset.sss_intensity;
        builder.detail_fade = preset.detail_fade;
        builder.ambient_scale = preset.ambient_scale;
        builder.blend_strength = preset.blend_strength;
        builder.bloom_scale = preset.bloom_scale;
        builder.reflection_min = preset.reflection_min;
        builder.reflection_max = preset.reflection_max;
        builder.moon_light_dim = preset.moon_light_dim;
        builder.sky_zenith_gradient_exp = preset.sky_zenith_gradient_exp;
        builder.horizon_glow_mult = preset.horizon_glow_mult;
        builder.sunset_orange_weight = preset.sunset_orange_weight;
        builder.sunset_intensity = preset.sunset_intensity;
        builder.sun_halo_intensity = preset.sun_halo_intensity;
        builder.moon_halo_intensity = preset.moon_halo_intensity;
        builder.micro_uv_freq = preset.micro_uv_freq;
        builder.micro_time_freq = preset.micro_time_freq;
        builder.micro_strength_mod = preset.micro_strength_mod;
        builder.foam_crest_width = preset.foam_crest_width;
        builder.caustic_aberration = preset.caustic_aberration;
        builder.caustic_smooth_low = preset.caustic_smooth_low;
        builder.caustic_smooth_high = preset.caustic_smooth_high;
        builder.aurora_brightness = preset.aurora_brightness;
        builder.aurora_y_threshold = preset.aurora_y_threshold;
        builder.water_brightness_mod = preset.water_brightness_mod;
        builder.decay_factor = preset.decay_factor;
        builder.dissipation_factor = preset.dissipation_factor;
        builder.warp_uv_scale = preset.warp_uv_scale;
        builder.warp_strength = preset.warp_strength;
        builder.warp_time_scale = preset.warp_time_scale;
        builder.foam_octaves = preset.foam_octaves;
        builder.foam_power = preset.foam_power;
        builder.hash_scale = preset.hash_scale;
        builder.hash_dot = preset.hash_dot;
        builder.steepness_threshold_low = preset.steepness_threshold_low;
        builder.steepness_threshold_high = preset.steepness_threshold_high;
        builder.y_displacement_weight = preset.y_displacement_weight;
        builder.wave_epsilon = preset.wave_epsilon;
        builder
    }

    pub fn from_uniform(u: &OceanSettingsUniform) -> Self {
        let mut cascade_vec = Vec::new();
        let count = u.cascade_count.min(MAX_CASCADES as u32) as usize;
        for i in 0..count {
            cascade_vec.push(CascadePreset {
                fft_size: u.cascade_data[i][0],
                amplitude: u.cascade_data[i][1],
            });
        }
        Self {
            mesh_size: u.mesh_size,
            mesh_subdivisions: u.mesh_subdivisions,
            fft_size: u.fft_size,
            fft_subdivisions: u.fft_subdivisions,
            time_scale: u.time_scale,
            ocean_seed: u.ocean_seed,
            chop_scale: u.chop_scale,
            amplitude_scale: u.amplitude_scale,
            wind_vector: u.wind_vector,
            amplitude: u.amplitude,
            l_small: u.l_small,
            max_w: u.max_w,
            fovy: u.fovy,
            zfar: u.zfar,
            cam_speed: u.cam_speed,
            cam_boost: u.cam_boost,
            cam_sensitivity: u.cam_sensitivity,
            roughness: u.roughness,
            f_0: u.f_0,
            specular_scale: u.specular_scale,
            reflection_scale: u.reflection_scale,
            foam_scale: u.foam_scale,
            sss_distortion_scale: u.sss_distortion_scale,
            caustic_scale: u.caustic_scale,
            caustic_size: u.caustic_size,
            caustic_speed: u.caustic_speed,
            caustic_intensity: u.caustic_intensity,
            caustic_octaves: u.caustic_octaves,
            caustic_depth: u.caustic_depth,
            caustic_max_distance: u.caustic_max_distance,
            micro_normal_strength: u.micro_normal_strength,
            foam_threshold: u.foam_threshold,
            foam_speed: u.foam_speed,
            foam_roughness: u.foam_roughness,
            daynight_cycle: u.daynight_cycle,
            caustic_color_tint: u.caustic_color_tint,
            deep_color: u.deep_color,
            shallow_color: u.shallow_color,
            sss_color: u.sss_color,
            sun_color: u.sun_color,
            sky_color_day_zenith: u.sky_color_day_zenith,
            sky_color_day_horizon: u.sky_color_day_horizon,
            sky_color_night_zenith: u.sky_color_night_zenith,
            sky_color_night_horizon: u.sky_color_night_horizon,
            sky_color_sunset_orange: u.sky_color_sunset_orange,
            sky_color_sunset_pink: u.sky_color_sunset_pink,
            sky_color_horizon_glow: u.sky_color_horizon_glow,
            moon_color_lit: u.moon_color_lit,
            moon_color_dark: u.moon_color_dark,
            cloud_color_night: u.cloud_color_night,
            cloud_color_day: u.cloud_color_day,
            sun_offset_z: u.sun_offset_z,
            sun_size_inner: u.sun_size_inner,
            sun_size_outer: u.sun_size_outer,
            sun_halo_power: u.sun_halo_power,
            moon_radius: u.moon_radius,
            moon_dist: u.moon_dist,
            moon_phase_offset: u.moon_phase_offset,
            moon_crater_scale: u.moon_crater_scale,
            moon_halo_power: u.moon_halo_power,
            star_count: u.star_count,
            star_threshold: u.star_threshold,
            star_blink_speed: u.star_blink_speed,
            star_size: u.star_size,
            aurora_strength: u.aurora_strength,
            cloud_speed: u.cloud_speed,
            cloud_density_low: u.cloud_density_low,
            cloud_density_high: u.cloud_density_high,
            cascade_data: cascade_vec,
            sunset_scatter_color: u.sunset_scatter_color,
            sunset_scatter_intensity: u.sunset_scatter_intensity,
            foam_base_color: u.foam_base_color,
            sss_min_height: u.sss_min_height,
            sss_max_height: u.sss_max_height,
            sss_power: u.sss_power,
            sss_intensity: u.sss_intensity,
            detail_fade: u.detail_fade,
            ambient_scale: u.ambient_scale,
            blend_strength: u.blend_strength,
            bloom_scale: u.bloom_scale,
            reflection_min: u.reflection_min,
            reflection_max: u.reflection_max,
            moon_light_dim: u.moon_light_dim,
            sky_zenith_gradient_exp: u.sky_zenith_gradient_exp,
            horizon_glow_mult: u.horizon_glow_mult,
            sunset_orange_weight: u.sunset_orange_weight,
            sunset_intensity: u.sunset_intensity,
            sun_halo_intensity: u.sun_halo_intensity,
            moon_halo_intensity: u.moon_halo_intensity,
            micro_uv_freq: u.micro_uv_freq,
            micro_time_freq: u.micro_time_freq,
            micro_strength_mod: u.micro_strength_mod,
            foam_crest_width: u.foam_crest_width,
            caustic_aberration: u.caustic_aberration,
            caustic_smooth_low: u.caustic_smooth_low,
            caustic_smooth_high: u.caustic_smooth_high,
            aurora_brightness: u.aurora_brightness,
            aurora_y_threshold: u.aurora_y_threshold,
            water_brightness_mod: u.water_brightness_mod,
            decay_factor: u.decay_factor,
            dissipation_factor: u.dissipation_factor,
            warp_uv_scale: u.warp_uv_scale,
            warp_strength: u.warp_strength,
            warp_time_scale: u.warp_time_scale,
            foam_octaves: u.foam_octaves,
            foam_power: u.foam_power,
            hash_scale: u.hash_scale,
            hash_dot: u.hash_dot,
            steepness_threshold_low: u.steepness_threshold_low,
            steepness_threshold_high: u.steepness_threshold_high,
            y_displacement_weight: u.y_displacement_weight,
            wave_epsilon: u.wave_epsilon,
        }
    }

    pub fn build(self) -> OceanSettingsUniform {
        let mut cascade_data = [[0.0f32; 4]; MAX_CASCADES];
        let count = self.cascade_data.len().min(MAX_CASCADES);
        for (i, c) in self.cascade_data.iter().take(count).enumerate() {
            cascade_data[i][0] = c.fft_size;
            cascade_data[i][1] = c.amplitude;
        }

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
            fft_size: self.fft_size,
            time_scale: self.time_scale,
            chop_scale: self.chop_scale,
            amplitude_scale: self.amplitude_scale,
            wave_scale: self.mesh_size / self.fft_size,
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
            mesh_subdivisions: self.mesh_subdivisions,
            fft_subdivisions: self.fft_subdivisions,
            pass_num: self.fft_subdivisions.ilog2(),
            ocean_seed: self.ocean_seed,
            caustic_octaves: self.caustic_octaves,
            star_size: self.star_size,
            aurora_strength: self.aurora_strength,
            pad_b: [0; 4],
            cascade_data,
            cascade_count: count as u32,
            _pad_cascade: [0; 3],
            sunset_scatter_color: self.sunset_scatter_color,
            sunset_scatter_intensity: self.sunset_scatter_intensity,
            foam_base_color: self.foam_base_color,
            sss_min_height: self.sss_min_height,
            sss_max_height: self.sss_max_height,
            sss_power: self.sss_power,
            sss_intensity: self.sss_intensity,
            detail_fade: self.detail_fade,
            ambient_scale: self.ambient_scale,
            blend_strength: self.blend_strength,
            bloom_scale: self.bloom_scale,
            reflection_min: self.reflection_min,
            reflection_max: self.reflection_max,
            moon_light_dim: self.moon_light_dim,
            sky_zenith_gradient_exp: self.sky_zenith_gradient_exp,
            horizon_glow_mult: self.horizon_glow_mult,
            sunset_orange_weight: self.sunset_orange_weight,
            sunset_intensity: self.sunset_intensity,
            sun_halo_intensity: self.sun_halo_intensity,
            moon_halo_intensity: self.moon_halo_intensity,
            micro_uv_freq: self.micro_uv_freq,
            micro_time_freq: self.micro_time_freq,
            micro_strength_mod: self.micro_strength_mod,
            foam_crest_width: self.foam_crest_width,
            caustic_aberration: self.caustic_aberration,
            caustic_smooth_low: self.caustic_smooth_low,
            caustic_smooth_high: self.caustic_smooth_high,
            aurora_brightness: self.aurora_brightness,
            aurora_y_threshold: self.aurora_y_threshold,
            water_brightness_mod: self.water_brightness_mod,
            decay_factor: self.decay_factor,
            dissipation_factor: self.dissipation_factor,
            warp_uv_scale: self.warp_uv_scale,
            warp_strength: self.warp_strength,
            warp_time_scale: self.warp_time_scale,
            foam_octaves: self.foam_octaves,
            foam_power: self.foam_power,
            hash_scale: self.hash_scale,
            hash_dot: self.hash_dot,
            steepness_threshold_low: self.steepness_threshold_low,
            steepness_threshold_high: self.steepness_threshold_high,
            y_displacement_weight: self.y_displacement_weight,
            wave_epsilon: self.wave_epsilon,
            _pad_final: [0.0; 9],
        }
    }
}

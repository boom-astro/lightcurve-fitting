//! Consolidated feature extraction from all fitters.
//!
//! Produces a flat `HashMap<String, Option<f64>>` (~80 features) from
//! nonparametric, parametric, thermal, and 2D GP results.  This is the
//! canonical feature set for downstream ML classifiers.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::common::BandData;
use crate::gp2d::{fit_gp_2d_with_thermal, Gp2dResult, Gp2dThermalResult};
use crate::nonparametric::{fit_nonparametric, NonparametricBandResult};
use crate::parametric::{
    fit_parametric, MultiBazinResult, ParametricBandResult, SviModelName, UncertaintyMethod,
};
use crate::thermal::{fit_thermal, ThermalResult};

/// Alias for the feature map type.
pub type FeatureMap = HashMap<String, Option<f64>>;

// ---------------------------------------------------------------------------
// Model name → integer
// ---------------------------------------------------------------------------

fn model_to_int(m: &SviModelName) -> f64 {
    match m {
        SviModelName::Bazin => 0.0,
        SviModelName::Villar => 1.0,
        SviModelName::MetzgerKN => 2.0,
        SviModelName::Tde => 3.0,
        SviModelName::Arnett => 4.0,
        SviModelName::Magnetar => 5.0,
        SviModelName::ShockCooling => 6.0,
        SviModelName::Afterglow => 7.0,
    }
}

const ALL_MODELS: &[SviModelName] = &[
    SviModelName::Bazin,
    SviModelName::Villar,
    SviModelName::MetzgerKN,
    SviModelName::Tde,
    SviModelName::Arnett,
    SviModelName::Magnetar,
    SviModelName::ShockCooling,
    SviModelName::Afterglow,
];

fn model_name_str(m: &SviModelName) -> &'static str {
    match m {
        SviModelName::Bazin => "Bazin",
        SviModelName::Villar => "Villar",
        SviModelName::MetzgerKN => "MetzgerKN",
        SviModelName::Tde => "Tde",
        SviModelName::Arnett => "Arnett",
        SviModelName::Magnetar => "Magnetar",
        SviModelName::ShockCooling => "ShockCooling",
        SviModelName::Afterglow => "Afterglow",
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn opt(v: Option<f64>) -> Option<f64> {
    v.filter(|x| x.is_finite())
}

fn find_band<'a>(results: &'a [NonparametricBandResult], band: &str) -> Option<&'a NonparametricBandResult> {
    results.iter().find(|r| r.band == band)
}

fn find_param_band<'a>(results: &'a [ParametricBandResult], band: &str) -> Option<&'a ParametricBandResult> {
    results.iter().find(|r| r.band == band)
}

// ---------------------------------------------------------------------------
// Nonparametric features
// ---------------------------------------------------------------------------

fn extract_np_features(np_results: &[NonparametricBandResult], ref_band: &str) -> FeatureMap {
    let mut f = FeatureMap::new();

    // Reference band
    let ref_r = find_band(np_results, ref_band)
        .or_else(|| np_results.first());
    let ref_r = match ref_r {
        Some(r) => r,
        None => return f,
    };

    // Core morphology
    f.insert("np_rise_time".into(), opt(ref_r.rise_time));
    f.insert("np_rise_halfmax".into(), opt(ref_r.rise_halfmax));
    f.insert("np_rise_efold".into(), opt(ref_r.rise_efold));
    f.insert("np_t0".into(), opt(ref_r.t0));
    f.insert("np_peak_mag".into(), opt(ref_r.peak_mag));
    f.insert("np_chi2".into(), opt(ref_r.chi2));
    f.insert("np_baseline_chi2".into(), opt(ref_r.baseline_chi2));
    f.insert("np_fwhm".into(), opt(ref_r.fwhm));
    f.insert("np_rise_rate".into(), opt(ref_r.rise_rate));
    f.insert("np_decay_rate".into(), opt(ref_r.decay_rate));

    // Decay metrics
    f.insert("np_decay_efold".into(), opt(ref_r.decay_efold));
    f.insert("np_decay_halfmax".into(), opt(ref_r.decay_halfmax));
    f.insert("np_dm15".into(), opt(ref_r.dm15));
    f.insert("np_near_peak_rise_rate".into(), opt(ref_r.near_peak_rise_rate));
    f.insert("np_near_peak_decay_rate".into(), opt(ref_r.near_peak_decay_rate));

    // GP derivatives and predictions
    f.insert("np_gp_dfdt_now".into(), opt(ref_r.gp_dfdt_now));
    f.insert("np_gp_dfdt_next".into(), opt(ref_r.gp_dfdt_next));
    f.insert("np_gp_d2fdt2_now".into(), opt(ref_r.gp_d2fdt2_now));
    f.insert("np_gp_predicted_mag_1d".into(), opt(ref_r.gp_predicted_mag_1d));
    f.insert("np_gp_predicted_mag_2d".into(), opt(ref_r.gp_predicted_mag_2d));
    f.insert("np_gp_time_to_peak".into(), opt(ref_r.gp_time_to_peak));
    f.insert("np_gp_extrap_slope".into(), opt(ref_r.gp_extrap_slope));

    // Variability and signal
    f.insert("np_gp_sigma_f".into(), opt(ref_r.gp_sigma_f));
    f.insert("np_gp_peak_to_peak".into(), opt(ref_r.gp_peak_to_peak));
    f.insert("np_gp_snr_max".into(), opt(ref_r.gp_snr_max));
    f.insert("np_gp_dfdt_max".into(), opt(ref_r.gp_dfdt_max));
    f.insert("np_gp_dfdt_min".into(), opt(ref_r.gp_dfdt_min));
    f.insert("np_gp_frac_of_peak".into(), opt(ref_r.gp_frac_of_peak));

    // Uncertainty
    f.insert("np_gp_post_var_mean".into(), opt(ref_r.gp_post_var_mean));
    f.insert("np_gp_post_var_max".into(), opt(ref_r.gp_post_var_max));

    // Statistical shape
    f.insert("np_gp_skewness".into(), opt(ref_r.gp_skewness));
    f.insert("np_gp_kurtosis".into(), opt(ref_r.gp_kurtosis));
    f.insert("np_gp_n_inflections".into(), opt(ref_r.gp_n_inflections));

    // Decay characterization
    f.insert("np_decay_power_law_index".into(), opt(ref_r.decay_power_law_index));
    f.insert("np_decay_power_law_chi2".into(), opt(ref_r.decay_power_law_chi2));

    // TDE vs AGN discrimination
    f.insert("np_von_neumann_ratio".into(), opt(ref_r.von_neumann_ratio));
    f.insert("np_pre_peak_rms".into(), opt(ref_r.pre_peak_rms));
    f.insert("np_rise_amplitude_over_noise".into(), opt(ref_r.rise_amplitude_over_noise));
    f.insert("np_post_peak_monotonicity".into(), opt(ref_r.post_peak_monotonicity));

    // Recurrence
    f.insert("np_n_local_maxima".into(), opt(ref_r.n_local_maxima));

    // GP hyperparameters
    f.insert("np_gp_fit_amp".into(), opt(ref_r.gp_fit_amp));
    f.insert("np_gp_fit_lengthscale".into(), opt(ref_r.gp_fit_lengthscale));

    // Observation counts
    f.insert("np_n_det".into(), Some(ref_r.n_obs as f64));
    f.insert("np_n_upper_limits".into(), Some(ref_r.n_upper_limits as f64));

    // Epoch magnitudes
    f.insert("np_mag_at_30d".into(), opt(ref_r.mag_at_30d));
    f.insert("np_mag_at_60d".into(), opt(ref_r.mag_at_60d));
    f.insert("np_mag_at_90d".into(), opt(ref_r.mag_at_90d));

    // Delta mags from peak
    let peak = ref_r.peak_mag;
    for (key, val) in [("30d", ref_r.mag_at_30d), ("60d", ref_r.mag_at_60d), ("90d", ref_r.mag_at_90d)] {
        f.insert(
            format!("np_delta_mag_{key}"),
            match (peak, val) {
                (Some(p), Some(v)) if p.is_finite() && v.is_finite() => Some(v - p),
                _ => None,
            },
        );
    }

    // Decay deceleration
    let m30 = ref_r.mag_at_30d;
    let m60 = ref_r.mag_at_60d;
    let m90 = ref_r.mag_at_90d;
    f.insert("np_decay_deceleration".into(), match (m30, m60, m90) {
        (Some(a), Some(b), Some(c)) if a.is_finite() && b.is_finite() && c.is_finite() => {
            let early = b - a;
            if early.abs() > 1e-6 { Some((c - b) / early) } else { None }
        }
        _ => None,
    });

    // --- Cross-band features ---
    let g = find_band(np_results, "g");
    let r = find_band(np_results, "r");

    // Peak color g-r
    f.insert("np_peak_color_g_r".into(), match (g, r) {
        (Some(g), Some(r)) => match (g.peak_mag, r.peak_mag) {
            (Some(gp), Some(rp)) if gp.is_finite() && rp.is_finite() => Some(gp - rp),
            _ => None,
        },
        _ => None,
    });

    // Peak time offset g-r
    f.insert("np_peak_time_offset_g_r".into(), match (g, r) {
        (Some(g), Some(r)) => match (g.t0, r.t0) {
            (Some(gt), Some(rt)) if gt.is_finite() && rt.is_finite() => Some(gt - rt),
            _ => None,
        },
        _ => None,
    });

    // Multi-epoch g-r color
    for offset in ["30d", "60d", "90d"] {
        let color = match (g, r) {
            (Some(gb), Some(rb)) => {
                let gm = match offset {
                    "30d" => gb.mag_at_30d, "60d" => gb.mag_at_60d, _ => gb.mag_at_90d,
                };
                let rm = match offset {
                    "30d" => rb.mag_at_30d, "60d" => rb.mag_at_60d, _ => rb.mag_at_90d,
                };
                match (gm, rm) {
                    (Some(a), Some(b)) if a.is_finite() && b.is_finite() => Some(a - b),
                    _ => None,
                }
            }
            _ => None,
        };
        f.insert(format!("np_color_g_r_{offset}"), color);
    }

    // Color evolution rate
    let c30 = f.get("np_color_g_r_30d").copied().flatten();
    let c60 = f.get("np_color_g_r_60d").copied().flatten();
    f.insert("np_color_evolution_rate".into(), match (c30, c60) {
        (Some(a), Some(b)) => Some((b - a) / 30.0),
        _ => None,
    });

    // --- Multi-band aggregates ---
    let vn_vals: Vec<f64> = np_results.iter()
        .filter_map(|r| r.von_neumann_ratio.filter(|v| v.is_finite()))
        .collect();
    f.insert("np_von_neumann_mean".into(),
        if vn_vals.is_empty() { None } else { Some(vn_vals.iter().sum::<f64>() / vn_vals.len() as f64) });

    let rise_vals: Vec<f64> = np_results.iter()
        .filter_map(|r| r.rise_amplitude_over_noise.filter(|v| v.is_finite()))
        .collect();
    f.insert("np_rise_significance_max".into(),
        rise_vals.iter().cloned().fold(None, |acc, v| Some(match acc { Some(a) => f64::max(a, v), None => v })));

    let mono_vals: Vec<f64> = np_results.iter()
        .filter_map(|r| r.post_peak_monotonicity.filter(|v| v.is_finite()))
        .collect();
    f.insert("np_monotonicity_mean".into(),
        if mono_vals.is_empty() { None } else { Some(mono_vals.iter().sum::<f64>() / mono_vals.len() as f64) });

    f.insert("np_n_bands_fit".into(), Some(np_results.len() as f64));

    f
}

// ---------------------------------------------------------------------------
// Parametric features
// ---------------------------------------------------------------------------

fn extract_param_features(param_results: &[ParametricBandResult], ref_band: &str) -> FeatureMap {
    let mut f = FeatureMap::new();

    let ref_r = find_param_band(param_results, ref_band)
        .or_else(|| param_results.first());
    let ref_r = match ref_r {
        Some(r) => r,
        None => return f,
    };

    // Best model
    f.insert("param_best_model".into(), Some(model_to_int(&ref_r.model)));

    // Fit quality
    f.insert("param_pso_chi2".into(), opt(ref_r.pso_chi2));
    f.insert("param_svi_elbo".into(), opt(ref_r.svi_elbo));
    f.insert("param_mag_chi2".into(), opt(ref_r.mag_chi2));
    f.insert("param_n_obs".into(), Some(ref_r.n_obs as f64));

    // SVI posterior means (first 4)
    for i in 0..4 {
        f.insert(
            format!("param_svi_mu_{i}"),
            ref_r.svi_mu.get(i).copied().filter(|v| v.is_finite()),
        );
    }

    // Mean posterior uncertainty
    if !ref_r.svi_log_sigma.is_empty() {
        let mean_sigma: f64 = ref_r.svi_log_sigma.iter()
            .map(|ls| ls.exp())
            .sum::<f64>() / ref_r.svi_log_sigma.len() as f64;
        f.insert("param_svi_sigma_mean".into(), Some(mean_sigma));
    } else {
        f.insert("param_svi_sigma_mean".into(), None);
    }

    // Bazin-specific timescales
    if ref_r.model == SviModelName::Bazin && ref_r.pso_params.len() >= 5 {
        f.insert("param_bazin_rise_tau".into(), Some(ref_r.pso_params[3].exp()));
        f.insert("param_bazin_decay_tau".into(), Some(ref_r.pso_params[4].exp()));
    } else {
        f.insert("param_bazin_rise_tau".into(), None);
        f.insert("param_bazin_decay_tau".into(), None);
    }

    // Delta chi2 across bands
    let mut chi2s: Vec<f64> = param_results.iter()
        .filter_map(|r| r.pso_chi2.filter(|v| v.is_finite()))
        .collect();
    chi2s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    f.insert("param_delta_chi2".into(),
        if chi2s.len() >= 2 { Some(chi2s[1] - chi2s[0]) } else { None });

    // Per-model chi2 and delta-chi2
    let best_chi2 = ref_r.pso_chi2;
    for model in ALL_MODELS {
        let name = model_name_str(model);
        let mc = ref_r.per_model_chi2.get(model).copied().flatten();
        f.insert(format!("param_chi2_{name}"), mc.filter(|v| v.is_finite()));
        f.insert(format!("param_dchi2_{name}"), match (mc, best_chi2) {
            (Some(m), Some(b)) if m.is_finite() && b.is_finite() => Some(m - b),
            _ => None,
        });
    }

    f.insert("param_n_bands_fit".into(), Some(param_results.len() as f64));

    // MultiBazin
    extract_mb_features(ref_r.multi_bazin.as_ref(), &mut f);

    f
}

fn extract_mb_features(mb: Option<&MultiBazinResult>, f: &mut FeatureMap) {
    match mb {
        Some(mb) => {
            f.insert("mb_best_k".into(), Some(mb.best_k as f64));
            f.insert("mb_cost".into(), opt(Some(mb.cost)));
            f.insert("mb_bic".into(), opt(Some(mb.bic)));
            let bic_imp = if mb.per_k_bic.len() >= 2
                && mb.per_k_bic[0].is_finite()
                && mb.per_k_bic[1].is_finite()
            {
                Some(mb.per_k_bic[0] - mb.per_k_bic[1])
            } else {
                None
            };
            f.insert("mb_bic_improvement_k2".into(), bic_imp);
        }
        None => {
            f.insert("mb_best_k".into(), None);
            f.insert("mb_cost".into(), None);
            f.insert("mb_bic".into(), None);
            f.insert("mb_bic_improvement_k2".into(), None);
        }
    }
}

// ---------------------------------------------------------------------------
// Thermal features
// ---------------------------------------------------------------------------

fn extract_thermal_features(thermal: &Option<ThermalResult>) -> FeatureMap {
    let mut f = FeatureMap::new();
    match thermal {
        Some(t) => {
            f.insert("thermal_log_temp_peak".into(), opt(t.log_temp_peak));
            f.insert("thermal_cooling_rate".into(), opt(t.cooling_rate));
            f.insert("thermal_log_temp_peak_err".into(), opt(t.log_temp_peak_err));
            f.insert("thermal_cooling_rate_err".into(), opt(t.cooling_rate_err));
            f.insert("thermal_chi2".into(), opt(t.chi2));
            f.insert("thermal_n_color_obs".into(), Some(t.n_color_obs as f64));
            f.insert("thermal_n_bands_used".into(), Some(t.n_bands_used as f64));
        }
        None => {
            for key in ["thermal_log_temp_peak", "thermal_cooling_rate",
                       "thermal_log_temp_peak_err", "thermal_cooling_rate_err",
                       "thermal_chi2", "thermal_n_color_obs", "thermal_n_bands_used"] {
                f.insert(key.to_string(), None);
            }
        }
    }
    f
}

// ---------------------------------------------------------------------------
// 2D GP features
// ---------------------------------------------------------------------------

fn extract_gp2d_features(
    gp2d_result: &Option<Gp2dResult>,
    gp2d_thermal: &Option<Gp2dThermalResult>,
) -> FeatureMap {
    let mut f = FeatureMap::new();

    match gp2d_result {
        Some(r) => {
            f.insert("gp2d_train_rms".into(), Some(r.train_rms));
            f.insert("gp2d_n_train".into(), Some(r.n_train as f64));
            f.insert("gp2d_n_bands".into(), Some(r.n_bands as f64));
            f.insert("gp2d_amp".into(), Some(r.amp));
            f.insert("gp2d_ls_time".into(), Some(r.ls_time));
            f.insert("gp2d_ls_wave".into(), Some(r.ls_wave));
        }
        None => {
            for key in ["gp2d_train_rms", "gp2d_n_train", "gp2d_n_bands",
                       "gp2d_amp", "gp2d_ls_time", "gp2d_ls_wave"] {
                f.insert(key.to_string(), None);
            }
        }
    }

    match gp2d_thermal {
        Some(t) if !t.log_temps.is_empty() => {
            let n = t.log_temps.len() as f64;

            // Peak temperature (at min chi2)
            let best_idx = t.chi2s.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            f.insert("gp2d_log_temp_peak".into(), Some(t.log_temps[best_idx]));

            // Latest temperature
            f.insert("gp2d_log_temp_latest".into(), Some(*t.log_temps.last().unwrap()));

            // Temperature range
            let tmin = t.log_temps.iter().cloned().fold(f64::INFINITY, f64::min);
            let tmax = t.log_temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            f.insert("gp2d_temp_range".into(), Some(tmax - tmin));

            // Cooling rate via linear regression: slope of log_temp vs time
            let mean_t = t.times.iter().sum::<f64>() / n;
            let mean_lt = t.log_temps.iter().sum::<f64>() / n;
            let mut cov = 0.0;
            let mut var_t = 0.0;
            for i in 0..t.times.len() {
                let dt = t.times[i] - mean_t;
                cov += dt * (t.log_temps[i] - mean_lt);
                var_t += dt * dt;
            }
            let slope = if var_t > 1e-10 { cov / var_t } else { 0.0 };
            f.insert("gp2d_cooling_rate".into(), Some(slope));

            // Chi2 statistics
            let chi2_mean = t.chi2s.iter().sum::<f64>() / n;
            let chi2_min = t.chi2s.iter().cloned().fold(f64::INFINITY, f64::min);
            f.insert("gp2d_thermal_chi2_mean".into(), Some(chi2_mean));
            f.insert("gp2d_thermal_chi2_min".into(), Some(chi2_min));
        }
        _ => {
            for key in ["gp2d_log_temp_peak", "gp2d_log_temp_latest", "gp2d_temp_range",
                       "gp2d_cooling_rate", "gp2d_thermal_chi2_mean", "gp2d_thermal_chi2_min"] {
                f.insert(key.to_string(), None);
            }
        }
    }

    f
}

// ===========================================================================
// Public API
// ===========================================================================

/// Extract features from already-computed fitting results (Layer 1).
///
/// This is useful when results are loaded from JSON or computed elsewhere.
pub fn extract_features_from_results(
    np_results: &[NonparametricBandResult],
    param_results: &[ParametricBandResult],
    thermal: &Option<ThermalResult>,
    gp2d_result: &Option<Gp2dResult>,
    gp2d_thermal: &Option<Gp2dThermalResult>,
    ref_band: &str,
) -> FeatureMap {
    let mut features = FeatureMap::new();
    features.extend(extract_np_features(np_results, ref_band));
    features.extend(extract_param_features(param_results, ref_band));
    features.extend(extract_thermal_features(thermal));
    features.extend(extract_gp2d_features(gp2d_result, gp2d_thermal));
    features
}

/// Run all fitters and extract features for a single source (Layer 2).
///
/// - `mag_bands`: magnitude-space data (for nonparametric, thermal, 2D GP)
/// - `flux_bands`: flux-space data (for parametric)
/// - `ref_band`: reference band for feature extraction (default "r")
pub fn extract_features(
    mag_bands: &HashMap<String, BandData>,
    flux_bands: &HashMap<String, BandData>,
    ref_band: &str,
) -> FeatureMap {
    // Nonparametric + thermal (reuse GP)
    let (np_results, gps) = fit_nonparametric(mag_bands);
    let thermal = fit_thermal(mag_bands, Some(&gps));

    // Parametric (fit_all_models=true for per-model chi2)
    let param_results = fit_parametric(flux_bands, true, UncertaintyMethod::Svi);

    // 2D GP
    let (gp2d_result, gp2d_thermal) = match fit_gp_2d_with_thermal(mag_bands, 50) {
        Some((r, t)) => (Some(r), Some(t)),
        None => (None, None),
    };

    extract_features_from_results(&np_results, &param_results, &thermal, &gp2d_result, &gp2d_thermal, ref_band)
}

/// Run all fitters and extract features for many sources in parallel (Layer 3).
pub fn extract_features_batch(
    sources_mag: &[HashMap<String, BandData>],
    sources_flux: &[HashMap<String, BandData>],
    ref_band: &str,
) -> Vec<FeatureMap> {
    sources_mag.par_iter().zip(sources_flux.par_iter())
        .map(|(mag, flux)| extract_features(mag, flux, ref_band))
        .collect()
}

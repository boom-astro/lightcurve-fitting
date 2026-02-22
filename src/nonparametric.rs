use std::collections::HashMap;

use scirs2_core::ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use sklears_core::traits::{Fit, Predict, Untrained};
use sklears_gaussian_process::{
    kernels::{ConstantKernel, ProductKernel, SumKernel, WhiteKernel, RBF},
    GaussianProcessRegressor, GprTrained, Kernel,
};

use crate::common::{
    compute_decay_rate, compute_fwhm, compute_rise_rate, extract_decay_timescale,
    extract_rise_timescale, finite_or_none, BandData,
};
use crate::gp::{fit_sklears_gp, subsample_data};

/// Result of nonparametric GP fitting for a single band.
/// All f64 fields are `Option<f64>` for JSON safety (NaN → None).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonparametricBandResult {
    pub band: String,
    pub rise_time: Option<f64>,
    pub decay_time: Option<f64>,
    pub t0: Option<f64>,
    pub peak_mag: Option<f64>,
    pub chi2: Option<f64>,
    pub baseline_chi2: Option<f64>,
    pub n_obs: usize,
    pub fwhm: Option<f64>,
    pub rise_rate: Option<f64>,
    pub decay_rate: Option<f64>,
    pub gp_dfdt_now: Option<f64>,
    pub gp_dfdt_next: Option<f64>,
    pub gp_d2fdt2_now: Option<f64>,
    pub gp_predicted_mag_1d: Option<f64>,
    pub gp_predicted_mag_2d: Option<f64>,
    pub gp_time_to_peak: Option<f64>,
    pub gp_extrap_slope: Option<f64>,
    pub gp_sigma_f: Option<f64>,
    pub gp_peak_to_peak: Option<f64>,
    pub gp_snr_max: Option<f64>,
    pub gp_dfdt_max: Option<f64>,
    pub gp_dfdt_min: Option<f64>,
    pub gp_frac_of_peak: Option<f64>,
    pub gp_post_var_mean: Option<f64>,
    pub gp_post_var_max: Option<f64>,
    pub gp_skewness: Option<f64>,
    pub gp_kurtosis: Option<f64>,
    pub gp_n_inflections: Option<f64>,
    /// Power-law decay index: slope of mag vs log10(t - t_peak) for post-peak data.
    /// TDEs: ~4.2 (flux ∝ t^{-5/3}), SNIa: poor fit (exponential decay).
    pub decay_power_law_index: Option<f64>,
    /// Chi2 of the power-law decay fit (lower = better power-law match).
    pub decay_power_law_chi2: Option<f64>,
    /// GP-predicted magnitude at peak + 30 days.
    pub mag_at_30d: Option<f64>,
    /// GP-predicted magnitude at peak + 60 days.
    pub mag_at_60d: Option<f64>,
    /// GP-predicted magnitude at peak + 90 days.
    pub mag_at_90d: Option<f64>,
    /// Von Neumann ratio η of raw magnitudes (time-sorted).
    /// Low (~0.1–0.5) for smooth monotonic evolution (TDE), high (~1.5–2.0) for stochastic (AGN).
    pub von_neumann_ratio: Option<f64>,
    /// Standard deviation of raw magnitudes before GP peak time.
    /// Low for TDEs (quiescent baseline), higher for AGN (ongoing variability).
    pub pre_peak_rms: Option<f64>,
    /// Rise significance: (pre-peak mean mag − peak mag) / median error.
    /// Large for TDEs (clear rise from quiescence), smaller for AGN.
    pub rise_amplitude_over_noise: Option<f64>,
    /// Fraction of consecutive post-peak GP predictions where mag increases (fading).
    /// ~1.0 for TDEs (monotonic decay), ~0.5 for stochastic AGN.
    pub post_peak_monotonicity: Option<f64>,
}

// ---------------------------------------------------------------------------
// FastGP fallback with early-time weighting
// ---------------------------------------------------------------------------

struct FastGP {
    base: GaussianProcessRegressor<Untrained>,
}

impl FastGP {
    fn new(t_max: f64) -> Self {
        let amp = 0.2;
        let cst: Box<dyn Kernel> = Box::new(ConstantKernel::new(amp));
        let lengthscale = (t_max / 16.0).max(0.3).min(12.0);
        let rbf: Box<dyn Kernel> = Box::new(RBF::new(lengthscale));
        let prod = Box::new(ProductKernel::new(vec![cst, rbf]));
        let white = Box::new(WhiteKernel::new(1e-10));
        let kernel = SumKernel::new(vec![prod, white]);

        let base = GaussianProcessRegressor::new()
            .kernel(Box::new(kernel))
            .alpha(1e-10)
            .normalize_y(true);

        Self { base }
    }

    fn fit(
        &self,
        times: &Array1<f64>,
        values: &Array1<f64>,
        errors: &[f64],
    ) -> Option<GaussianProcessRegressor<GprTrained>> {
        let t_min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let t_range = t_max - t_min;
        let early_time_cutoff = t_min + 0.2 * t_range;

        let weighted_errors: Vec<f64> = times
            .iter()
            .zip(errors.iter())
            .map(|(t, e)| {
                if t <= &early_time_cutoff {
                    *e * 0.7
                } else {
                    *e
                }
            })
            .collect();

        let avg_error_var = if !weighted_errors.is_empty() {
            weighted_errors.iter().map(|e| e * e).sum::<f64>() / weighted_errors.len() as f64
        } else {
            1e-4
        };

        let alpha_with_errors = avg_error_var.max(1e-5);
        let gp_with_alpha = self.base.clone().alpha(alpha_with_errors);
        let xt = times.view().insert_axis(Axis(1)).to_owned();
        gp_with_alpha.fit(&xt, values).ok()
    }
}


// ---------------------------------------------------------------------------
// Predictive features
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PredictiveFeatures {
    gp_dfdt_now: f64,
    gp_dfdt_next: f64,
    gp_d2fdt2_now: f64,
    gp_predicted_mag_1d: f64,
    gp_predicted_mag_2d: f64,
    gp_time_to_peak: f64,
    gp_extrap_slope: f64,
    gp_sigma_f: f64,
    gp_peak_to_peak: f64,
    gp_snr_max: f64,
    gp_dfdt_max: f64,
    gp_dfdt_min: f64,
    gp_frac_of_peak: f64,
    gp_post_var_mean: f64,
    gp_post_var_max: f64,
    gp_skewness: f64,
    gp_kurtosis: f64,
    gp_n_inflections: f64,
}

fn nan_predictive_features(t0: f64, t_last: f64) -> PredictiveFeatures {
    PredictiveFeatures {
        gp_dfdt_now: f64::NAN,
        gp_dfdt_next: f64::NAN,
        gp_d2fdt2_now: f64::NAN,
        gp_predicted_mag_1d: f64::NAN,
        gp_predicted_mag_2d: f64::NAN,
        gp_time_to_peak: t0 - t_last,
        gp_extrap_slope: f64::NAN,
        gp_sigma_f: f64::NAN,
        gp_peak_to_peak: f64::NAN,
        gp_snr_max: f64::NAN,
        gp_dfdt_max: f64::NAN,
        gp_dfdt_min: f64::NAN,
        gp_frac_of_peak: f64::NAN,
        gp_post_var_mean: f64::NAN,
        gp_post_var_max: f64::NAN,
        gp_skewness: f64::NAN,
        gp_kurtosis: f64::NAN,
        gp_n_inflections: f64::NAN,
    }
}

fn compute_predictive_features(
    gp: &GaussianProcessRegressor<GprTrained>,
    t_last: f64,
    t0: f64,
    times_pred: &[f64],
    pred: &[f64],
    std: &[f64],
    obs_mags: &[f64],
    obs_errors: &[f64],
) -> PredictiveFeatures {
    let dt = 1.0;
    let tq = vec![
        t_last - dt,
        t_last,
        t_last + dt,
        t_last + 2.0 * dt,
        t_last + 3.0 * dt,
    ];
    let xq = Array2::from_shape_fn((tq.len(), 1), |(i, _)| tq[i]);
    let y = match gp.predict(&xq) {
        Ok(arr) => arr.to_vec(),
        Err(_) => return nan_predictive_features(t0, t_last),
    };

    let f_m1 = y[0];
    let f_0 = y[1];
    let f_p1 = y[2];
    let f_p2 = y[3];
    let f_p3 = y[4];

    // Variability strength
    let gp_sigma_f = if !pred.is_empty() {
        let mean = pred.iter().sum::<f64>() / pred.len() as f64;
        (pred.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / pred.len() as f64).sqrt()
    } else {
        f64::NAN
    };

    let gp_peak_to_peak = if !pred.is_empty() {
        let max_mag = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_mag = pred.iter().cloned().fold(f64::INFINITY, f64::min);
        max_mag - min_mag
    } else {
        f64::NAN
    };

    let gp_snr_max = if !obs_mags.is_empty() && !obs_errors.is_empty() {
        obs_mags
            .iter()
            .zip(obs_errors.iter())
            .map(|(mag, err)| mag.abs() / err)
            .fold(f64::NEG_INFINITY, f64::max)
    } else {
        f64::NAN
    };

    // Derivative features
    let gp_dfdt_max = if pred.len() > 1 && times_pred.len() > 1 {
        let dt_grid = times_pred[1] - times_pred[0];
        (0..pred.len() - 1)
            .map(|i| (pred[i + 1] - pred[i]) / dt_grid)
            .fold(f64::NEG_INFINITY, f64::max)
    } else {
        f64::NAN
    };

    let gp_dfdt_min = if pred.len() > 1 && times_pred.len() > 1 {
        let dt_grid = times_pred[1] - times_pred[0];
        (0..pred.len() - 1)
            .map(|i| (pred[i + 1] - pred[i]) / dt_grid)
            .fold(f64::INFINITY, f64::min)
    } else {
        f64::NAN
    };

    // Phase feature
    let gp_frac_of_peak = if !pred.is_empty() {
        let peak_mag = pred.iter().cloned().fold(f64::INFINITY, f64::min);
        let last_mag = pred.last().copied().unwrap_or(f64::NAN);
        last_mag / peak_mag
    } else {
        f64::NAN
    };

    // Uncertainty quantification
    let gp_post_var_mean = if !std.is_empty() {
        std.iter().map(|s| s * s).sum::<f64>() / std.len() as f64
    } else {
        f64::NAN
    };

    let gp_post_var_max = if !std.is_empty() {
        std.iter().map(|s| s * s).fold(f64::NEG_INFINITY, f64::max)
    } else {
        f64::NAN
    };

    // Statistical shape features
    let (gp_skewness, gp_kurtosis) = if !pred.is_empty() && pred.len() > 3 {
        let mean = pred.iter().sum::<f64>() / pred.len() as f64;
        let variance = pred.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / pred.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev > 1e-10 {
            let skew = pred
                .iter()
                .map(|&x| ((x - mean) / std_dev).powi(3))
                .sum::<f64>()
                / pred.len() as f64;
            let kurt = pred
                .iter()
                .map(|&x| ((x - mean) / std_dev).powi(4))
                .sum::<f64>()
                / pred.len() as f64
                - 3.0;
            (skew, kurt)
        } else {
            (f64::NAN, f64::NAN)
        }
    } else {
        (f64::NAN, f64::NAN)
    };

    // Inflection points
    let gp_n_inflections = if pred.len() > 2 && times_pred.len() > 2 {
        let dt_grid = times_pred[1] - times_pred[0];
        let mut d2: Vec<f64> = Vec::with_capacity(pred.len().saturating_sub(2));
        for i in 1..(pred.len() - 1) {
            let v = (pred[i + 1] - 2.0 * pred[i] + pred[i - 1]) / (dt_grid * dt_grid);
            d2.push(v);
        }
        let eps = 1e-6_f64;
        let mut count = 0usize;
        for i in 0..(d2.len().saturating_sub(1)) {
            let a = d2[i];
            let b = d2[i + 1];
            if a.is_finite() && b.is_finite() && a.abs() > eps && b.abs() > eps && (a * b) < 0.0 {
                count += 1;
            }
        }
        count as f64
    } else {
        f64::NAN
    };

    PredictiveFeatures {
        gp_dfdt_now: (f_0 - f_m1) / dt,
        gp_dfdt_next: (f_p1 - f_0) / dt,
        gp_d2fdt2_now: (f_p1 - 2.0 * f_0 + f_m1) / (dt * dt),
        gp_predicted_mag_1d: f_p1,
        gp_predicted_mag_2d: f_p2,
        gp_time_to_peak: t0 - t_last,
        gp_extrap_slope: (f_p3 - f_p2) / dt,
        gp_sigma_f,
        gp_peak_to_peak,
        gp_snr_max,
        gp_dfdt_max,
        gp_dfdt_min,
        gp_frac_of_peak,
        gp_post_var_mean,
        gp_post_var_max,
        gp_skewness,
        gp_kurtosis,
        gp_n_inflections,
    }
}

// ---------------------------------------------------------------------------
// Decay power-law fitting
// ---------------------------------------------------------------------------

/// Fit mag = A + B * log10(t - t_peak + epsilon) to post-peak GP predictions.
/// Returns (power_law_index B, reduced chi2).
/// TDE flux ∝ t^{-5/3} → B ≈ 2.5 * 5/3 ≈ 4.17 in magnitudes.
fn fit_decay_power_law(
    times_pred: &[f64],
    pred: &[f64],
    std_vec: &[f64],
    peak_idx: usize,
) -> (f64, f64) {
    // Use predictions starting from peak+5 days to avoid the peak itself
    let t_peak = times_pred[peak_idx];
    let epsilon = 1.0; // offset to avoid log(0)

    let mut log_t = Vec::new();
    let mut mag = Vec::new();
    let mut weights = Vec::new();

    for i in (peak_idx + 1)..times_pred.len() {
        let dt = times_pred[i] - t_peak;
        if dt < 5.0 {
            continue;
        }
        if !pred[i].is_finite() {
            continue;
        }
        let w = if i < std_vec.len() && std_vec[i].is_finite() && std_vec[i] > 1e-10 {
            1.0 / (std_vec[i] * std_vec[i])
        } else {
            1.0
        };
        log_t.push((dt + epsilon).log10());
        mag.push(pred[i]);
        weights.push(w);
    }

    if log_t.len() < 3 {
        return (f64::NAN, f64::NAN);
    }

    // Weighted linear regression: mag = A + B * log_t
    let sum_w: f64 = weights.iter().sum();
    let sum_wx: f64 = log_t.iter().zip(weights.iter()).map(|(x, w)| w * x).sum();
    let sum_wy: f64 = mag.iter().zip(weights.iter()).map(|(y, w)| w * y).sum();
    let sum_wxx: f64 = log_t.iter().zip(weights.iter()).map(|(x, w)| w * x * x).sum();
    let sum_wxy: f64 = log_t
        .iter()
        .zip(mag.iter())
        .zip(weights.iter())
        .map(|((x, y), w)| w * x * y)
        .sum();

    let denom = sum_w * sum_wxx - sum_wx * sum_wx;
    if denom.abs() < 1e-20 {
        return (f64::NAN, f64::NAN);
    }

    let b = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
    let a = (sum_wy - b * sum_wx) / sum_w;

    // Compute reduced chi2
    let mut chi2 = 0.0;
    for i in 0..log_t.len() {
        let residual = mag[i] - (a + b * log_t[i]);
        chi2 += weights[i] * residual * residual;
    }
    let chi2_reduced = chi2 / (log_t.len().max(3) - 2) as f64;

    (b, chi2_reduced)
}

/// Predict GP magnitude at a specific time offset from peak.
fn predict_mag_at_offset(
    gp: &GaussianProcessRegressor<GprTrained>,
    t_peak: f64,
    offset_days: f64,
) -> f64 {
    let t = t_peak + offset_days;
    let x = Array2::from_shape_fn((1, 1), |_| t);
    match gp.predict(&x) {
        Ok(arr) => {
            let v = arr[0];
            if v.is_finite() { v } else { f64::NAN }
        }
        Err(_) => f64::NAN,
    }
}

// ---------------------------------------------------------------------------
// Variability / rise features (TDE vs AGN discrimination)
// ---------------------------------------------------------------------------

/// Von Neumann ratio: η = Σ(m_{i+1} - m_i)² / ((n-1) · Var(m)).
/// Must be computed on time-sorted raw magnitudes.
/// Low (~0.1–0.5) for smooth monotonic lightcurves, ~2.0 for white noise.
fn compute_von_neumann_ratio(times: &[f64], mags: &[f64]) -> f64 {
    if mags.len() < 5 {
        return f64::NAN;
    }
    // Sort by time
    let mut idx: Vec<usize> = (0..times.len()).collect();
    idx.sort_by(|&a, &b| times[a].total_cmp(&times[b]));

    let sorted_mags: Vec<f64> = idx.iter().map(|&i| mags[i]).collect();
    let n = sorted_mags.len() as f64;
    let mean = sorted_mags.iter().sum::<f64>() / n;
    let variance = sorted_mags.iter().map(|&m| (m - mean).powi(2)).sum::<f64>() / (n - 1.0);
    if variance < 1e-20 {
        return f64::NAN;
    }
    let delta_sq: f64 = sorted_mags
        .windows(2)
        .map(|w| (w[1] - w[0]).powi(2))
        .sum();
    let eta = delta_sq / ((n - 1.0) * variance);
    eta
}

/// Pre-peak baseline features from raw data.
/// Returns (pre_peak_rms, rise_amplitude_over_noise).
fn compute_pre_peak_features(
    times: &[f64],
    mags: &[f64],
    errors: &[f64],
    t_peak: f64,
    peak_mag: f64,
) -> (f64, f64) {
    // Gather observations before the GP peak time
    let pre: Vec<(f64, f64)> = times
        .iter()
        .zip(mags.iter())
        .zip(errors.iter())
        .filter(|((t, _), _)| **t < t_peak)
        .map(|((_, &m), &e)| (m, e))
        .collect();

    if pre.len() < 3 {
        return (f64::NAN, f64::NAN);
    }

    let pre_mags: Vec<f64> = pre.iter().map(|(m, _)| *m).collect();
    let pre_errs: Vec<f64> = pre.iter().map(|(_, e)| *e).collect();

    let n = pre_mags.len() as f64;
    let mean = pre_mags.iter().sum::<f64>() / n;
    let rms = (pre_mags.iter().map(|&m| (m - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

    // Rise significance: how many error-bars is the rise?
    let mut sorted_errs = pre_errs.clone();
    sorted_errs.sort_by(|a, b| a.total_cmp(b));
    let median_err = sorted_errs[sorted_errs.len() / 2];
    let rise_sig = if median_err > 1e-10 {
        (mean - peak_mag) / median_err // positive = brightened (mag decreased)
    } else {
        f64::NAN
    };

    (rms, rise_sig)
}

/// Fraction of consecutive post-peak GP predictions where mag increases (fading).
fn compute_post_peak_monotonicity(pred: &[f64], peak_idx: usize) -> f64 {
    if peak_idx >= pred.len().saturating_sub(2) {
        return f64::NAN;
    }
    let post = &pred[peak_idx..];
    if post.len() < 3 {
        return f64::NAN;
    }
    let n_pairs = post.len() - 1;
    let n_fading = post
        .windows(2)
        .filter(|w| w[1].is_finite() && w[0].is_finite() && w[1] > w[0]) // fainter = higher mag
        .count();
    n_fading as f64 / n_pairs as f64
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Fit nonparametric GP models to all bands.
///
/// `bands` maps band names to `BandData` containing magnitude values.
///
/// Returns the per-band results and the trained GP for each band (for reuse
/// by downstream fitters such as `fit_thermal`).
pub fn fit_nonparametric(
    bands: &HashMap<String, BandData>,
) -> (Vec<NonparametricBandResult>, HashMap<String, GaussianProcessRegressor<GprTrained>>) {
    if bands.is_empty() {
        return (Vec::new(), HashMap::new());
    }

    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for band_data in bands.values() {
        for &t in &band_data.times {
            t_min = t_min.min(t);
            t_max = t_max.max(t);
        }
    }
    let duration = t_max - t_min;
    if duration <= 0.0 {
        return (Vec::new(), HashMap::new());
    }

    let n_pred = 50;
    let times_pred: Vec<f64> = (0..n_pred)
        .map(|i| t_min + (i as f64) * duration / (n_pred - 1) as f64)
        .collect();
    let times_pred_arr = Array1::from_vec(times_pred.clone());
    let times_pred_2d = times_pred_arr.view().insert_axis(Axis(1)).to_owned();

    let min_points_for_independent_fit = 5;
    let amp_candidates = vec![0.1, 0.3];
    let ls_factors = vec![6.0, 12.0, 24.0];

    let mut results = Vec::new();
    let mut trained_gps: HashMap<String, GaussianProcessRegressor<GprTrained>> = HashMap::new();

    for (band_name, band_data) in bands {
        if band_data.times.len() < min_points_for_independent_fit {
            continue;
        }

        let max_subsample = if band_data.times.len() <= 30 {
            band_data.times.len()
        } else {
            25
        };
        let (times_sub, mags_sub, errors_sub) = subsample_data(
            &band_data.times,
            &band_data.values,
            &band_data.errors,
            max_subsample,
        );

        let times_arr = Array1::from_vec(times_sub);
        let mags_arr = Array1::from_vec(mags_sub);

        // Compute average error variance for alpha candidates
        let avg_error_var = if !errors_sub.is_empty() {
            errors_sub.iter().map(|e| e * e).sum::<f64>() / errors_sub.len() as f64
        } else {
            1e-4
        };
        let alpha_candidates = vec![avg_error_var.max(1e-6)];

        // Compute minimum lengthscale from data sampling
        let mut dt_vec: Vec<f64> = (1..times_arr.len())
            .map(|w| times_arr[w] - times_arr[w - 1])
            .filter(|d| d.is_finite())
            .collect();
        dt_vec.sort_by(|a, b| a.total_cmp(b));
        let median_dt = if !dt_vec.is_empty() {
            dt_vec[dt_vec.len() / 2]
        } else {
            1.0
        };
        let min_lengthscale = (median_dt * 2.0).max(0.1);

        let xt_sub = times_arr.view().insert_axis(Axis(1)).to_owned();
        let times_orig_2d = Array1::from_vec(band_data.times.clone())
            .view()
            .insert_axis(Axis(1))
            .to_owned();

        // Grid search over amplitude, lengthscale, and alpha
        let mut best_gp: Option<GaussianProcessRegressor<GprTrained>> = None;
        let mut best_score = f64::INFINITY;

        for &amp in &amp_candidates {
            for &factor in &ls_factors {
                let lengthscale = (duration / factor).max(0.1);
                if lengthscale < min_lengthscale {
                    continue;
                }

                for &alpha in &alpha_candidates {
                    if let Some(trained) =
                        fit_sklears_gp(&times_arr, &mags_arr, amp, lengthscale, alpha)
                    {
                        if let Ok(pred_at_obs) = trained.predict(&xt_sub) {
                            let mut residuals_sq = 0.0f64;
                            for i in 0..mags_arr.len() {
                                let residual = mags_arr[i] - pred_at_obs[i];
                                residuals_sq += residual * residual;
                            }
                            let rms = (residuals_sq / mags_arr.len() as f64).sqrt();

                            // Compute mean predictive std to penalize overconfident fits
                            let mut mean_pred_std = 0.0f64;
                            if let Ok((pred_std_obs, _)) = trained.predict_with_std(&xt_sub) {
                                let v = pred_std_obs.to_vec();
                                let (ssum, scnt) = v
                                    .iter()
                                    .filter(|s| s.is_finite())
                                    .fold((0.0f64, 0usize), |(s, c), &val| (s + val, c + 1));
                                if scnt > 0 {
                                    mean_pred_std = ssum / scnt as f64;
                                }
                            }

                            // Reject candidates with absurd extrapolated peak magnitudes
                            if let Ok(pred_grid) = trained.predict(&times_pred_2d) {
                                let pred_grid_min =
                                    pred_grid.iter().cloned().fold(f64::INFINITY, f64::min);
                                let obs_min =
                                    mags_arr.iter().cloned().fold(f64::INFINITY, f64::min);
                                if pred_grid_min.is_finite()
                                    && (pred_grid_min - obs_min).abs() > 6.0
                                {
                                    continue;
                                }
                            }

                            // Combined score: fit quality + uncertainty penalty
                            let penalty_coef = 0.6_f64;
                            let score = rms + penalty_coef * mean_pred_std;
                            if score.is_finite() && score < best_score {
                                best_score = score;
                                best_gp = Some(trained);
                            }
                        }
                    }
                }
            }
        }

        // Fallback to FastGP with early-time weighting if grid search failed
        if best_gp.is_none() {
            let fallback = FastGP::new(duration);
            best_gp = fallback.fit(&times_arr, &mags_arr, &errors_sub);
        }

        if let Some(gp_fit) = best_gp {
            trained_gps.insert(band_name.clone(), gp_fit.clone());
            // Get spatially-varying uncertainty from predict_with_std
            let (pred, mut std_vec) =
                if let Ok((pred_arr, pred_std_arr)) = gp_fit.predict_with_std(&times_pred_2d) {
                    (pred_arr.to_vec(), pred_std_arr.to_vec())
                } else if let Ok(pred_arr) = gp_fit.predict(&times_pred_2d) {
                    // Fallback: if predict_with_std fails, use flat uncertainty
                    let p = pred_arr.to_vec();
                    let s = vec![0.1; p.len()];
                    (p, s)
                } else {
                    continue;
                };

            // Compute RMS residual at observed points for uncertainty rescaling
            if let Ok(pred_at_obs_arr) = gp_fit.predict(&times_orig_2d) {
                let pred_at_obs = pred_at_obs_arr.to_vec();

                let mut residuals_sq = 0.0;
                for i in 0..band_data.values.len() {
                    let residual = band_data.values[i] - pred_at_obs[i];
                    residuals_sq += residual * residual;
                }
                let rms_residual = (residuals_sq / band_data.values.len() as f64).sqrt();

                // Scale GP std to match observed RMS with conservative shrinkage
                if let Ok((pred_std_obs, _)) = gp_fit.predict_with_std(&times_orig_2d) {
                    let pred_std_obs_vec = pred_std_obs.to_vec();
                    let (sum, cnt) = pred_std_obs_vec
                        .iter()
                        .filter(|s| s.is_finite())
                        .fold((0.0f64, 0usize), |(s, c), &val| (s + val, c + 1));
                    if cnt > 0 {
                        let mean_pred_std_obs = sum / cnt as f64;
                        if mean_pred_std_obs > 1e-12 && rms_residual.is_finite() {
                            let mut scale = (rms_residual / mean_pred_std_obs).max(0.05).min(5.0);
                            scale *= 0.6; // conservative shrinkage
                            for v in std_vec.iter_mut() {
                                *v *= scale;
                            }
                        }
                    }
                }

                // Compute chi2 and features
                let mut chi2 = 0.0;
                let mut baseline_var = 0.0;
                let mean_mag = band_data.values.iter().sum::<f64>() / band_data.values.len() as f64;
                for i in 0..band_data.values.len() {
                    let residual = band_data.values[i] - pred_at_obs[i];
                    let err_sq = band_data.errors[i] * band_data.errors[i] + 1e-10;
                    chi2 += residual * residual / err_sq;
                    baseline_var += (band_data.values[i] - mean_mag).powi(2) / err_sq;
                }
                let chi2_reduced = chi2 / band_data.values.len().max(1) as f64;
                let baseline_chi2 = baseline_var / band_data.values.len().max(1) as f64;

                let peak_idx = pred
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_finite())
                    .min_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let t0 = times_pred[peak_idx];
                let peak_mag = pred[peak_idx];

                let rise_time = extract_rise_timescale(&times_pred, &pred, peak_idx);
                let decay_time = extract_decay_timescale(&times_pred, &pred, peak_idx);
                let (fwhm_calc, t_before, t_after) = compute_fwhm(&times_pred, &pred, peak_idx);
                let fwhm = if !t_before.is_nan() && !t_after.is_nan() {
                    t_after - t_before
                } else {
                    fwhm_calc
                };
                let rise_rate = compute_rise_rate(&times_pred, &pred);
                let decay_rate = compute_decay_rate(&times_pred, &pred);

                let t_last = band_data
                    .times
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let predictive = compute_predictive_features(
                    &gp_fit,
                    t_last,
                    t0,
                    &times_pred,
                    &pred,
                    &std_vec,
                    &band_data.values,
                    &band_data.errors,
                );

                // Decay power-law fit
                let (pl_index, pl_chi2) =
                    fit_decay_power_law(&times_pred, &pred, &std_vec, peak_idx);

                // GP predictions at fixed offsets from peak
                let mag_30d = predict_mag_at_offset(&gp_fit, t0, 30.0);
                let mag_60d = predict_mag_at_offset(&gp_fit, t0, 60.0);
                let mag_90d = predict_mag_at_offset(&gp_fit, t0, 90.0);

                // TDE vs AGN variability features
                let von_neumann = compute_von_neumann_ratio(
                    &band_data.times, &band_data.values,
                );
                let (pre_peak_rms, rise_over_noise) = compute_pre_peak_features(
                    &band_data.times,
                    &band_data.values,
                    &band_data.errors,
                    t0,
                    peak_mag,
                );
                let monotonicity = compute_post_peak_monotonicity(&pred, peak_idx);

                results.push(NonparametricBandResult {
                    band: band_name.clone(),
                    rise_time: finite_or_none(rise_time),
                    decay_time: finite_or_none(decay_time),
                    t0: finite_or_none(t0),
                    peak_mag: finite_or_none(peak_mag),
                    chi2: finite_or_none(chi2_reduced),
                    baseline_chi2: finite_or_none(baseline_chi2),
                    n_obs: band_data.values.len(),
                    fwhm: finite_or_none(fwhm),
                    rise_rate: finite_or_none(rise_rate),
                    decay_rate: finite_or_none(decay_rate),
                    gp_dfdt_now: finite_or_none(predictive.gp_dfdt_now),
                    gp_dfdt_next: finite_or_none(predictive.gp_dfdt_next),
                    gp_d2fdt2_now: finite_or_none(predictive.gp_d2fdt2_now),
                    gp_predicted_mag_1d: finite_or_none(predictive.gp_predicted_mag_1d),
                    gp_predicted_mag_2d: finite_or_none(predictive.gp_predicted_mag_2d),
                    gp_time_to_peak: finite_or_none(predictive.gp_time_to_peak),
                    gp_extrap_slope: finite_or_none(predictive.gp_extrap_slope),
                    gp_sigma_f: finite_or_none(predictive.gp_sigma_f),
                    gp_peak_to_peak: finite_or_none(predictive.gp_peak_to_peak),
                    gp_snr_max: finite_or_none(predictive.gp_snr_max),
                    gp_dfdt_max: finite_or_none(predictive.gp_dfdt_max),
                    gp_dfdt_min: finite_or_none(predictive.gp_dfdt_min),
                    gp_frac_of_peak: finite_or_none(predictive.gp_frac_of_peak),
                    gp_post_var_mean: finite_or_none(predictive.gp_post_var_mean),
                    gp_post_var_max: finite_or_none(predictive.gp_post_var_max),
                    gp_skewness: finite_or_none(predictive.gp_skewness),
                    gp_kurtosis: finite_or_none(predictive.gp_kurtosis),
                    gp_n_inflections: finite_or_none(predictive.gp_n_inflections),
                    decay_power_law_index: finite_or_none(pl_index),
                    decay_power_law_chi2: finite_or_none(pl_chi2),
                    mag_at_30d: finite_or_none(mag_30d),
                    mag_at_60d: finite_or_none(mag_60d),
                    mag_at_90d: finite_or_none(mag_90d),
                    von_neumann_ratio: finite_or_none(von_neumann),
                    pre_peak_rms: finite_or_none(pre_peak_rms),
                    rise_amplitude_over_noise: finite_or_none(rise_over_noise),
                    post_peak_monotonicity: finite_or_none(monotonicity),
                });
            }
        }
    }

    (results, trained_gps)
}

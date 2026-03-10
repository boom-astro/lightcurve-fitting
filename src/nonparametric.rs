use std::collections::HashMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::common::{
    compute_decay_efold, compute_decay_halfmax, compute_decay_rate, compute_dm15,
    compute_fwhm, compute_near_peak_decay_rate, compute_near_peak_rise_rate,
    compute_rise_efold, compute_rise_halfmax, compute_rise_rate,
    extract_rise_timescale, finite_or_none, mag2flux, BandData,
};
use crate::gp::subsample_data;
use crate::sparse_gp::{DenseGP, SparseGP};

// ---------------------------------------------------------------------------
// GpPredictor trait — abstracts over custom dense and sparse GP
// ---------------------------------------------------------------------------

pub(crate) trait GpPredictor: Send + Sync {
    fn predict_times(&self, times: &[f64]) -> Option<Vec<f64>>;
    fn predict_times_with_std(&self, times: &[f64]) -> Option<(Vec<f64>, Vec<f64>)>;
}

impl GpPredictor for DenseGP {
    fn predict_times(&self, times: &[f64]) -> Option<Vec<f64>> {
        Some(self.predict(times))
    }
    fn predict_times_with_std(&self, times: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        Some(self.predict_with_std(times))
    }
}

impl GpPredictor for SparseGP {
    fn predict_times(&self, times: &[f64]) -> Option<Vec<f64>> {
        Some(self.predict(times))
    }
    fn predict_times_with_std(&self, times: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        Some(self.predict_with_std(times))
    }
}

// ---------------------------------------------------------------------------
// Upper-limit detection helpers
// ---------------------------------------------------------------------------

const SNR_THRESHOLD: f64 = 3.0;

/// Returns boolean mask: true = detection (SNR >= 3), false = upper limit.
fn detection_mask(mags: &[f64], mag_errs: &[f64]) -> Vec<bool> {
    let zp = 23.9;
    mags.iter()
        .zip(mag_errs.iter())
        .map(|(&m, &e)| {
            let (flux, flux_err) = mag2flux(m, e, zp);
            flux > 0.0 && flux_err > 0.0 && (flux / flux_err) >= SNR_THRESHOLD
        })
        .collect()
}

/// Filter BandData to detections only. Returns (filtered_data, n_upper_limits).
fn filter_upper_limits(band_data: &BandData) -> (BandData, usize) {
    let mask = detection_mask(&band_data.values, &band_data.errors);
    let n_upper = mask.iter().filter(|&&d| !d).count();
    let filtered = BandData {
        times: band_data.times.iter().zip(&mask).filter(|(_, &m)| m).map(|(&t, _)| t).collect(),
        values: band_data.values.iter().zip(&mask).filter(|(_, &m)| m).map(|(&v, _)| v).collect(),
        errors: band_data.errors.iter().zip(&mask).filter(|(_, &m)| m).map(|(&e, _)| e).collect(),
    };
    (filtered, n_upper)
}

/// Result of nonparametric GP fitting for a single band.
/// All f64 fields are `Option<f64>` for JSON safety (NaN → None).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonparametricBandResult {
    pub band: String,
    /// Rise e-fold timescale (old method, uses baseline average). Deprecated in favor of rise_halfmax.
    pub rise_time: Option<f64>,
    /// Rise half-max time: days before peak for flux to rise from half-peak to peak.
    pub rise_halfmax: Option<f64>,
    /// Rise e-fold time: days before peak for flux to rise from peak/e to peak.
    pub rise_efold: Option<f64>,
    /// E-folding decay time: days from peak for flux to drop to peak_flux / e.
    pub decay_efold: Option<f64>,
    /// Δm15: magnitude change 15 days after peak (positive for fading).
    pub dm15: Option<f64>,
    /// Half-max decay time: days from peak for flux to drop to half of peak flux.
    pub decay_halfmax: Option<f64>,
    pub t0: Option<f64>,
    pub peak_mag: Option<f64>,
    pub chi2: Option<f64>,
    pub baseline_chi2: Option<f64>,
    pub n_obs: usize,
    pub fwhm: Option<f64>,
    pub rise_rate: Option<f64>,
    pub decay_rate: Option<f64>,
    /// Near-peak rise rate: linear slope (mag/day) in 30 days before peak. Negative = brightening.
    pub near_peak_rise_rate: Option<f64>,
    /// Near-peak decay rate: linear slope (mag/day) in 30 days after peak. Positive = fading.
    pub near_peak_decay_rate: Option<f64>,
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
    /// Number of distinct brightness peaks (GP magnitude local minima).
    /// SNe: 1, CVs: 2+ (multiple outbursts), AGN: variable.
    pub n_local_maxima: Option<f64>,
    /// Number of upper-limit observations (flux SNR < 3) excluded from GP training.
    pub n_upper_limits: usize,
    /// Best-fit GP kernel amplitude (σ² in RBF kernel).
    pub gp_fit_amp: Option<f64>,
    /// Best-fit GP kernel lengthscale (ρ in RBF kernel), in days.
    pub gp_fit_lengthscale: Option<f64>,
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
    gp: &dyn GpPredictor,
    t_last: f64,
    t0: f64,
    times_pred: &[f64],
    pred: &[f64],
    std: &[f64],
    obs_mags: &[f64],
    obs_errors: &[f64],
) -> PredictiveFeatures {
    let dt = 1.0;
    // Query points for forward predictions
    let tq = vec![
        t_last - dt,
        t_last,
        t_last + dt,
        t_last + 2.0 * dt,
        t_last + 3.0 * dt,
    ];
    let y = match gp.predict_times(&tq) {
        Some(v) => v,
        None => return nan_predictive_features(t0, t_last),
    };

    let f_m1 = y[0];
    let f_0 = y[1];
    let f_p1 = y[2];
    let f_p2 = y[3];
    let f_p3 = y[4];

    // Smoothed derivative: average GP slope over a 5-day window centred on t_last.
    // Uses 6 query points at 1-day spacing [t-2, t-1, t, t+1, t+2, t+3] to get
    // 5 finite differences, then averages them. Much more stable than a single-
    // point difference as alerts accumulate.
    let gp_dfdt_now_smooth = {
        let window_pts: Vec<f64> = (-2..=3).map(|i| t_last + i as f64).collect();
        match gp.predict_times(&window_pts) {
            Some(yw) if yw.len() == 6 => {
                let slopes: Vec<f64> = (0..5)
                    .map(|i| yw[i + 1] - yw[i]) // dt=1 so slope = diff
                    .filter(|s| s.is_finite())
                    .collect();
                if slopes.is_empty() {
                    f64::NAN
                } else {
                    slopes.iter().sum::<f64>() / slopes.len() as f64
                }
            }
            _ => (f_0 - f_m1) / dt, // fallback to single-point
        }
    };

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
        gp_dfdt_now: gp_dfdt_now_smooth,
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
    gp: &dyn GpPredictor,
    t_peak: f64,
    offset_days: f64,
) -> f64 {
    let t = t_peak + offset_days;
    match gp.predict_times(&[t]) {
        Some(v) if v[0].is_finite() => v[0],
        _ => f64::NAN,
    }
}

/// Features computed on a fine (1-day) GP prediction grid for accuracy.
struct FineGridMetrics {
    decay_efold: f64,
    dm15: f64,
    decay_halfmax: f64,
    fwhm: f64,
    rise_time: f64,
    rise_halfmax: f64,
    rise_efold: f64,
    rise_rate: f64,
    decay_rate: f64,
    near_peak_rise_rate: f64,
    near_peak_decay_rate: f64,
}

/// Compute features using a fine GP prediction grid around the peak.
fn compute_fine_grid_metrics(
    gp: &dyn GpPredictor,
    _t_peak: f64,
    t_min: f64,
    t_max: f64,
) -> FineGridMetrics {
    // Fine grid: 1-day spacing covering the full observation window
    let start = t_min;
    let end = t_max;
    let n_fine = ((end - start).max(1.0)) as usize + 1;
    // Cap at 2000 points to avoid excessive cost for very long baselines
    let step = if n_fine > 2000 { (end - start) / 1999.0 } else { 1.0 };
    let n_fine = n_fine.min(2000);
    let fine_times: Vec<f64> = (0..n_fine).map(|i| start + i as f64 * step).collect();

    let nan_result = FineGridMetrics {
        decay_efold: f64::NAN, dm15: f64::NAN, decay_halfmax: f64::NAN,
        fwhm: f64::NAN, rise_time: f64::NAN, rise_halfmax: f64::NAN,
        rise_efold: f64::NAN, rise_rate: f64::NAN, decay_rate: f64::NAN,
        near_peak_rise_rate: f64::NAN, near_peak_decay_rate: f64::NAN,
    };

    let fine_pred = match gp.predict_times(&fine_times) {
        Some(p) => p,
        None => return nan_result,
    };

    // Find peak on the fine grid
    let peak_idx = fine_pred.iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let decay_efold = compute_decay_efold(&fine_times, &fine_pred, peak_idx);
    let dm15 = compute_dm15(&fine_times, &fine_pred, peak_idx);
    let decay_halfmax = compute_decay_halfmax(&fine_times, &fine_pred, peak_idx);
    let (fwhm, _, _) = compute_fwhm(&fine_times, &fine_pred, peak_idx);
    let rise_time = extract_rise_timescale(&fine_times, &fine_pred, peak_idx);
    let rise_halfmax = compute_rise_halfmax(&fine_times, &fine_pred, peak_idx);
    let rise_efold = compute_rise_efold(&fine_times, &fine_pred, peak_idx);
    let rise_rate = compute_rise_rate(&fine_times, &fine_pred);
    let decay_rate = compute_decay_rate(&fine_times, &fine_pred);
    let near_peak_rise_rate = compute_near_peak_rise_rate(&fine_times, &fine_pred, peak_idx, 30.0);
    let near_peak_decay_rate = compute_near_peak_decay_rate(&fine_times, &fine_pred, peak_idx, 30.0);

    FineGridMetrics {
        decay_efold, dm15, decay_halfmax, fwhm, rise_time, rise_halfmax, rise_efold,
        rise_rate, decay_rate, near_peak_rise_rate, near_peak_decay_rate,
    }
}

/// Compute features from a coarse grid (fallback when no GP predictor available).
#[allow(dead_code)]
fn compute_coarse_grid_metrics(
    times_pred: &[f64],
    pred: &[f64],
    peak_idx: usize,
) -> FineGridMetrics {
    let (fwhm, _, _) = compute_fwhm(times_pred, pred, peak_idx);
    FineGridMetrics {
        decay_efold: compute_decay_efold(times_pred, pred, peak_idx),
        dm15: compute_dm15(times_pred, pred, peak_idx),
        decay_halfmax: compute_decay_halfmax(times_pred, pred, peak_idx),
        fwhm,
        rise_time: extract_rise_timescale(times_pred, pred, peak_idx),
        rise_halfmax: compute_rise_halfmax(times_pred, pred, peak_idx),
        rise_efold: compute_rise_efold(times_pred, pred, peak_idx),
        rise_rate: compute_rise_rate(times_pred, pred),
        decay_rate: compute_decay_rate(times_pred, pred),
        near_peak_rise_rate: compute_near_peak_rise_rate(times_pred, pred, peak_idx, 30.0),
        near_peak_decay_rate: compute_near_peak_decay_rate(times_pred, pred, peak_idx, 30.0),
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
    delta_sq / ((n - 1.0) * variance)
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

/// Count distinct local brightness maxima (magnitude minima) in the GP prediction.
///
/// A local max is a point brighter than both neighbours by at least `threshold` mag.
/// CVs typically show multiple outbursts (n >= 2), while SNe have exactly 1.
fn count_local_maxima(pred: &[f64], threshold: f64) -> usize {
    if pred.len() < 3 {
        return if pred.is_empty() { 0 } else { 1 };
    }
    let mut count = 0;
    for i in 1..pred.len() - 1 {
        if !pred[i].is_finite() || !pred[i - 1].is_finite() || !pred[i + 1].is_finite() {
            continue;
        }
        // In magnitude space, brighter = lower value, so a local brightness max
        // is a local minimum in mag.
        if pred[i] < pred[i - 1] - threshold && pred[i] < pred[i + 1] - threshold {
            count += 1;
        }
    }
    // Check endpoints
    if pred.len() >= 2 {
        if pred[0].is_finite() && pred[1].is_finite() && pred[0] < pred[1] - threshold {
            count += 1;
        }
        let n = pred.len();
        if pred[n - 1].is_finite() && pred[n - 2].is_finite() && pred[n - 1] < pred[n - 2] - threshold {
            count += 1;
        }
    }
    count.max(1) // at least 1 if we have data
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
/// Returns the per-band results and the trained GP for each band (for reuse
/// by downstream fitters such as `fit_thermal`).
pub fn fit_nonparametric(
    bands: &HashMap<String, BandData>,
) -> (Vec<NonparametricBandResult>, HashMap<String, DenseGP>) {
    fit_nonparametric_with_opts(bands, 25)
}

/// Fit nonparametric GP models for many sources simultaneously on the GPU.
///
/// Each source's bands are independently fit. Returns results in the same
/// order as the input `sources` slice.
#[cfg(feature = "cuda")]
pub fn fit_nonparametric_batch_gpu(
    gpu: &crate::gpu::GpuContext,
    sources: &[HashMap<String, BandData>],
) -> Vec<(Vec<NonparametricBandResult>, HashMap<String, DenseGP>)> {
    fit_nonparametric_batch_gpu_with_opts(gpu, sources, 25)
}

/// Like [`fit_nonparametric_batch_gpu`] but with configurable subsample size.
#[cfg(feature = "cuda")]
pub fn fit_nonparametric_batch_gpu_with_opts(
    gpu: &crate::gpu::GpuContext,
    sources: &[HashMap<String, BandData>],
    max_dense_subsample: usize,
) -> Vec<(Vec<NonparametricBandResult>, HashMap<String, DenseGP>)> {
    use crate::gpu::GpBandInput;

    if sources.is_empty() {
        return Vec::new();
    }

    // Compute global time range across all sources
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for source_bands in sources {
        for band_data in source_bands.values() {
            for &t in &band_data.times {
                t_min = t_min.min(t);
                t_max = t_max.max(t);
            }
        }
    }
    let duration = t_max - t_min;
    if duration <= 0.0 {
        return sources.iter().map(|_| (Vec::new(), HashMap::new())).collect();
    }

    let n_pred = 50;
    let times_pred: Vec<f64> = (0..n_pred)
        .map(|i| t_min + (i as f64) * duration / (n_pred - 1) as f64)
        .collect();
    let amp_candidates: Vec<f64> = vec![0.3, 1.0, 3.0, 10.0];
    let ls_factors: &[f64] = &[3.0, 6.0, 12.0, 24.0, 48.0];
    let ls_candidates: Vec<f64> = ls_factors.iter().map(|f| (duration / f).max(0.1)).collect();

    // Flatten all bands from all sources into a single batch
    struct BandMeta {
        source_idx: usize,
        band_name: String,
        band_data_idx: usize,  // index into flat band_data_refs
        n_upper_limits: usize,
    }

    let mut gpu_inputs: Vec<GpBandInput> = Vec::new();
    let mut band_metas: Vec<BandMeta> = Vec::new();
    let mut band_data_refs: Vec<&BandData> = Vec::new();
    let mut det_data_vec: Vec<BandData> = Vec::new();

    for (src_idx, source_bands) in sources.iter().enumerate() {
        for (band_name, band_data) in source_bands {
            if band_data.times.len() < 5 {
                continue;
            }
            let (det_data, n_upper) = filter_upper_limits(band_data);
            if det_data.times.len() < 5 {
                continue;
            }
            let noise_var: Vec<f64> = det_data.errors.iter().map(|e| (e * e).max(1e-6)).collect();
            gpu_inputs.push(GpBandInput {
                times: det_data.times.clone(),
                mags: det_data.values.clone(),
                noise_var,
            });
            band_metas.push(BandMeta {
                source_idx: src_idx,
                band_name: band_name.clone(),
                band_data_idx: band_data_refs.len(),
                n_upper_limits: n_upper,
            });
            band_data_refs.push(band_data);
            det_data_vec.push(det_data);
        }
    }

    if gpu_inputs.is_empty() {
        return sources.iter().map(|_| (Vec::new(), HashMap::new())).collect();
    }

    // Launch GPU batch GP fit
    let gpu_outputs = match gpu.batch_gp_fit(
        &gpu_inputs, &times_pred, &amp_candidates, &ls_candidates, max_dense_subsample,
    ) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("GPU batch GP fit failed: {e}, falling back to CPU");
            return sources.iter().map(|bands| fit_nonparametric_with_opts(bands, max_dense_subsample)).collect();
        }
    };

    // Process GPU outputs: extract features on CPU using GPU predictions
    let mut results: Vec<(Vec<NonparametricBandResult>, HashMap<String, DenseGP>)> =
        sources.iter().map(|_| (Vec::new(), HashMap::new())).collect();

    for (idx, (meta, gpu_out)) in band_metas.iter().zip(gpu_outputs.into_iter()).enumerate() {
        let band_data = band_data_refs[meta.band_data_idx];
        let det_data = &det_data_vec[idx];
        let band_name = &meta.band_name;

        // Check if GPU produced valid predictions
        let pred = &gpu_out.pred_grid;
        let has_valid = pred.iter().any(|v| v.is_finite());
        if !has_valid {
            continue;
        }

        let mut std_vec = gpu_out.std_grid.clone();
        let pred_at_obs = &gpu_out.pred_at_obs;

        // Scale GP std (same logic as CPU path) — using detections only
        let rms_residual = {
            let mut rss = 0.0;
            for i in 0..det_data.values.len() {
                if i < pred_at_obs.len() {
                    let r = det_data.values[i] - pred_at_obs[i];
                    rss += r * r;
                }
            }
            (rss / det_data.values.len().max(1) as f64).sqrt()
        };

        let (sum_std, cnt_std) = std_vec.iter()
            .filter(|s| s.is_finite())
            .fold((0.0f64, 0usize), |(s, c), &val| (s + val, c + 1));
        if cnt_std > 0 {
            let mean_pred_std = sum_std / cnt_std as f64;
            if mean_pred_std > 1e-12 && rms_residual.is_finite() {
                let mut scale = (rms_residual / mean_pred_std).max(0.05).min(5.0);
                scale *= 0.6;
                for v in std_vec.iter_mut() {
                    *v *= scale;
                }
            }
        }

        // Chi2 and baseline chi2 (detections only)
        let mean_mag = det_data.values.iter().sum::<f64>() / det_data.values.len() as f64;
        let mut chi2 = 0.0;
        let mut baseline_var = 0.0;
        for i in 0..det_data.values.len() {
            if i < pred_at_obs.len() {
                let residual = det_data.values[i] - pred_at_obs[i];
                let err_sq = det_data.errors[i] * det_data.errors[i] + 1e-10;
                chi2 += residual * residual / err_sq;
                baseline_var += (det_data.values[i] - mean_mag).powi(2) / err_sq;
            }
        }
        let chi2_reduced = chi2 / det_data.values.len().max(1) as f64;
        let baseline_chi2 = baseline_var / det_data.values.len().max(1) as f64;

        // Peak detection
        let peak_idx = pred.iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let t0 = times_pred[peak_idx];
        let peak_mag = pred[peak_idx];

        let t_last = band_data.times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let t_min = *times_pred.first().unwrap_or(&t0);
        let t_max = *times_pred.last().unwrap_or(&t0);
        let fine = if let Some(ref gp) = gpu_out.dense_gp {
                compute_fine_grid_metrics(gp, t0, t_min, t_max)
            } else {
                compute_coarse_grid_metrics(&times_pred, pred, peak_idx)
            };
        let rise_time = fine.rise_time;
        let rise_halfmax = fine.rise_halfmax;
        let rise_efold = fine.rise_efold;
        let decay_efold = fine.decay_efold;
        let dm15 = fine.dm15;
        let decay_halfmax = fine.decay_halfmax;
        let fwhm = fine.fwhm;
        let rise_rate = fine.rise_rate;
        let decay_rate = fine.decay_rate;
        let near_peak_rise_rate = fine.near_peak_rise_rate;
        let near_peak_decay_rate = fine.near_peak_decay_rate;
        let predictive = if let Some(ref gp) = gpu_out.dense_gp {
            compute_predictive_features(
                gp, t_last, t0, &times_pred, pred, &std_vec,
                &det_data.values, &det_data.errors,
            )
        } else {
            nan_predictive_features(t0, t_last)
        };

        // Decay power-law fit
        let (pl_index, pl_chi2) = fit_decay_power_law(&times_pred, pred, &std_vec, peak_idx);

        // GP predictions at fixed offsets
        let (mag_30d, mag_60d, mag_90d) = if let Some(ref gp) = gpu_out.dense_gp {
            (
                predict_mag_at_offset(gp, t0, 30.0),
                predict_mag_at_offset(gp, t0, 60.0),
                predict_mag_at_offset(gp, t0, 90.0),
            )
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };

        // TDE vs AGN features
        let von_neumann = compute_von_neumann_ratio(&band_data.times, &band_data.values);
        let (pre_peak_rms, rise_over_noise) = compute_pre_peak_features(
            &band_data.times, &band_data.values, &band_data.errors, t0, peak_mag,
        );
        let monotonicity = compute_post_peak_monotonicity(pred, peak_idx);
        let n_local_maxima = count_local_maxima(pred, 0.1);

        let result = NonparametricBandResult {
            band: band_name.to_string(),
            rise_time: finite_or_none(rise_time),
            rise_halfmax: finite_or_none(rise_halfmax),
            rise_efold: finite_or_none(rise_efold),
            decay_efold: finite_or_none(decay_efold),
            dm15: finite_or_none(dm15),
            decay_halfmax: finite_or_none(decay_halfmax),
            t0: finite_or_none(t0),
            peak_mag: finite_or_none(peak_mag),
            chi2: finite_or_none(chi2_reduced),
            baseline_chi2: finite_or_none(baseline_chi2),
            n_obs: band_data.values.len(),
            fwhm: finite_or_none(fwhm),
            rise_rate: finite_or_none(rise_rate),
            decay_rate: finite_or_none(decay_rate),
            near_peak_rise_rate: finite_or_none(near_peak_rise_rate),
            near_peak_decay_rate: finite_or_none(near_peak_decay_rate),
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
            n_local_maxima: Some(n_local_maxima as f64),
            n_upper_limits: meta.n_upper_limits,
            gp_fit_amp: if let Some(ref gp) = gpu_out.dense_gp {
                finite_or_none(gp.kernel_amp())
            } else { None },
            gp_fit_lengthscale: if let Some(ref gp) = gpu_out.dense_gp {
                finite_or_none(gp.kernel_lengthscale())
            } else { None },
        };

        let (ref mut src_results, ref mut src_gps) = results[meta.source_idx];
        src_results.push(result);
        if let Some(gp) = gpu_out.dense_gp {
            src_gps.insert(band_name.clone(), gp);
        }
    }

    results
}

/// Like [`fit_nonparametric`] but with configurable subsample size for the
/// dense GP used for thermal reuse when the band has many observations.
pub fn fit_nonparametric_with_opts(
    bands: &HashMap<String, BandData>,
    max_dense_subsample: usize,
) -> (Vec<NonparametricBandResult>, HashMap<String, DenseGP>) {
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
    let amp_candidates: &[f64] = &[0.3, 1.0, 3.0, 10.0];
    let ls_factors: &[f64] = &[3.0, 6.0, 12.0, 24.0, 48.0];

    // Process bands in parallel
    let band_outputs: Vec<_> = bands.par_iter()
        .filter(|(_, bd)| bd.times.len() >= 5)
        .filter_map(|(band_name, band_data)| {
            process_band(
                band_name, band_data, duration, &times_pred,
                amp_candidates, ls_factors, max_dense_subsample,
            )
        })
        .collect();

    let mut results = Vec::with_capacity(band_outputs.len());
    let mut trained_gps: HashMap<String, DenseGP> = HashMap::new();
    for (result, gp_opt) in band_outputs {
        results.push(result);
        if let Some((name, gp)) = gp_opt {
            trained_gps.insert(name, gp);
        }
    }

    // If no per-band results were produced (all bands < 5 points), try a
    // cross-band GP that pools observations from all bands after subtracting
    // per-band mean magnitudes. This enables feature extraction from as few
    // as 5 total alerts spread across multiple bands.
    if results.is_empty() {
        let total_obs: usize = bands.values().map(|b| b.times.len()).sum();
        if total_obs >= 5 {
            if let Some(combined) = fit_combined_band(
                bands, duration, &times_pred, amp_candidates, ls_factors,
            ) {
                results.push(combined);
            }
        }
    }

    (results, trained_gps)
}

/// Pool observations from all bands into a single "combined" GP fit.
///
/// Subtracts per-band mean magnitudes so that the GP fits the shared light
/// curve *shape* rather than the absolute brightness in each filter. The
/// result is returned with `band = "combined"` and `peak_mag` expressed
/// relative to the mean-subtracted curve (i.e. the magnitude offset from
/// the per-band average).
fn fit_combined_band(
    bands: &HashMap<String, BandData>,
    duration: f64,
    times_pred: &[f64],
    amp_candidates: &[f64],
    ls_factors: &[f64],
) -> Option<NonparametricBandResult> {
    // Pool all bands, subtracting per-band means so that the GP fits the
    // shared light-curve *shape*. Track the grand mean to restore absolute
    // magnitudes afterwards.
    let mut all_times = Vec::new();
    let mut all_vals = Vec::new();
    let mut all_errs = Vec::new();
    let mut grand_sum = 0.0_f64;
    let mut grand_n = 0usize;

    for bd in bands.values() {
        if bd.times.is_empty() {
            continue;
        }
        let mean_mag = bd.values.iter().sum::<f64>() / bd.values.len() as f64;
        for i in 0..bd.times.len() {
            if bd.values[i].is_finite() && bd.errors[i].is_finite() {
                all_times.push(bd.times[i]);
                all_vals.push(bd.values[i] - mean_mag);
                all_errs.push(bd.errors[i]);
                grand_sum += bd.values[i];
                grand_n += 1;
            }
        }
    }

    if all_times.len() < 5 {
        return None;
    }
    let grand_mean = grand_sum / grand_n as f64;

    // Sort by time
    let mut order: Vec<usize> = (0..all_times.len()).collect();
    order.sort_by(|&a, &b| all_times[a].total_cmp(&all_times[b]));
    let times: Vec<f64> = order.iter().map(|&i| all_times[i]).collect();
    let values: Vec<f64> = order.iter().map(|&i| all_vals[i]).collect();
    let errors: Vec<f64> = order.iter().map(|&i| all_errs[i]).collect();

    let combined = BandData {
        times: times.clone(),
        values: values.clone(),
        errors: errors.clone(),
    };

    // Reuse process_band with max_dense_subsample=25
    process_band(
        "combined", &combined, duration, times_pred,
        amp_candidates, ls_factors, 25,
    )
    .map(|(mut result, _)| {
        // Restore absolute magnitudes by adding back the grand mean.
        // peak_mag, mag_at_30d/60d/90d, gp_predicted_mag_1d/2d are all in
        // the mean-subtracted system and need the offset.
        if let Some(ref mut v) = result.peak_mag { *v += grand_mean; }
        if let Some(ref mut v) = result.gp_predicted_mag_1d { *v += grand_mean; }
        if let Some(ref mut v) = result.gp_predicted_mag_2d { *v += grand_mean; }
        if let Some(ref mut v) = result.mag_at_30d { *v += grand_mean; }
        if let Some(ref mut v) = result.mag_at_60d { *v += grand_mean; }
        if let Some(ref mut v) = result.mag_at_90d { *v += grand_mean; }
        result
    })
}

/// Process a single band: fit GP, extract features, and build a DenseGP for thermal reuse.
fn process_band(
    band_name: &str,
    band_data: &BandData,
    duration: f64,
    times_pred: &[f64],
    amp_candidates: &[f64],
    ls_factors: &[f64],
    max_dense_subsample: usize,
) -> Option<(NonparametricBandResult, Option<(String, DenseGP)>)> {
    // Filter out upper limits (flux SNR < 3) from GP training data
    let (det_data, n_upper_limits) = filter_upper_limits(band_data);
    if det_data.times.len() < 5 {
        return None;
    }

    // Compute minimum lengthscale from data sampling
    let mut sorted_times = det_data.times.clone();
    sorted_times.sort_by(|a, b| a.total_cmp(b));
    let mut dt_vec: Vec<f64> = (1..sorted_times.len())
        .map(|w| sorted_times[w] - sorted_times[w - 1])
        .filter(|d| d.is_finite() && *d > 0.0)
        .collect();
    dt_vec.sort_by(|a, b| a.total_cmp(b));
    let median_dt = if !dt_vec.is_empty() {
        dt_vec[dt_vec.len() / 2]
    } else {
        1.0
    };
    let min_lengthscale = (median_dt * 2.0).max(0.1);

    // ---------------------------------------------------------------
    // Fit GP: sparse for large n, dense for small n
    // ---------------------------------------------------------------
    let use_sparse = det_data.times.len() > 100;
    let gp_predictor: Box<dyn GpPredictor>;
    let mut thermal_gp: Option<(String, DenseGP)> = None;
    let noise_var: Vec<f64> = det_data.errors.iter().map(|e| (e * e).max(1e-6)).collect();
    let mut fit_amp = f64::NAN;
    let mut fit_lengthscale = f64::NAN;

    if use_sparse {
        // Sparse GP (FITC) — uses detections only
        let n_det = det_data.times.len();
        let m_inducing = 30.min(n_det);

        let mut best_sparse: Option<SparseGP> = None;
        let mut best_score = f64::INFINITY;

        for &amp in amp_candidates {
            for &factor in ls_factors {
                let lengthscale = (duration / factor).max(0.1);
                if lengthscale < min_lengthscale { continue; }
                if let Some(sgp) = SparseGP::fit(
                    &det_data.times, &det_data.values, &noise_var,
                    amp, lengthscale, m_inducing,
                ) {
                    let score = sgp.approx_nlml(
                        &det_data.times, &det_data.values, &noise_var,
                    );
                    if score.is_finite() && score < best_score {
                        best_score = score;
                        best_sparse = Some(sgp);
                        fit_amp = amp;
                        fit_lengthscale = lengthscale;
                    }
                }
            }
        }

        gp_predictor = Box::new(best_sparse?);

        // Fit a DenseGP on subsampled data for thermal reuse
        let (times_sub, mags_sub, errs_sub) = subsample_data(
            &det_data.times, &det_data.values, &det_data.errors, max_dense_subsample,
        );
        let nv_sub: Vec<f64> = errs_sub.iter().map(|e| (e * e).max(1e-6)).collect();
        'outer_sparse: for &amp in amp_candidates {
            for &factor in ls_factors {
                let ls = (duration / factor).max(0.1);
                if let Some(dgp) = DenseGP::fit(&times_sub, &mags_sub, &nv_sub, amp, ls) {
                    thermal_gp = Some((band_name.to_string(), dgp));
                    break 'outer_sparse;
                }
            }
        }
    } else {
        // Custom dense GP — fits on detections only (n ≤ 100)
        let mut best_dense: Option<DenseGP> = None;
        let mut best_score = f64::INFINITY;

        for &amp in amp_candidates {
            for &factor in ls_factors {
                let lengthscale = (duration / factor).max(0.1);
                if lengthscale < min_lengthscale { continue; }
                if let Some(dgp) = DenseGP::fit(
                    &det_data.times, &det_data.values, &noise_var,
                    amp, lengthscale,
                ) {
                    let rms = dgp.train_rms(&det_data.values);
                    if rms.is_finite() && rms < best_score {
                        best_score = rms;
                        best_dense = Some(dgp);
                        fit_amp = amp;
                        fit_lengthscale = lengthscale;
                    }
                }
            }
        }

        let dgp = best_dense?;
        thermal_gp = Some((band_name.to_string(), dgp.clone()));
        gp_predictor = Box::new(dgp);
    };

    // ---------------------------------------------------------------
    // Feature extraction (unified for both GP types via GpPredictor)
    // ---------------------------------------------------------------
    let (pred, mut std_vec) =
        if let Some((p, s)) = gp_predictor.predict_times_with_std(times_pred) {
            (p, s)
        } else if let Some(p) = gp_predictor.predict_times(times_pred) {
            let s = vec![0.1; p.len()];
            (p, s)
        } else {
            return None;
        };

    let pred_at_obs = gp_predictor.predict_times(&det_data.times)?;

    let mut residuals_sq = 0.0;
    for i in 0..det_data.values.len() {
        let residual = det_data.values[i] - pred_at_obs[i];
        residuals_sq += residual * residual;
    }
    let rms_residual = (residuals_sq / det_data.values.len() as f64).sqrt();

    // Scale GP std: use a subsample for uncertainty estimation
    let max_for_std = 200;
    let std_sample_times = if det_data.times.len() > max_for_std {
        let step = det_data.times.len() as f64 / max_for_std as f64;
        (0..max_for_std)
            .map(|i| det_data.times[((i as f64 + 0.5) * step) as usize])
            .collect::<Vec<_>>()
    } else {
        det_data.times.clone()
    };

    if let Some((_, std_at_sample)) = gp_predictor.predict_times_with_std(&std_sample_times) {
        let (sum, cnt) = std_at_sample
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

    // Compute chi2 and features (detections only)
    let mut chi2 = 0.0;
    let mut baseline_var = 0.0;
    let mean_mag = det_data.values.iter().sum::<f64>() / det_data.values.len() as f64;
    for i in 0..det_data.values.len() {
        let residual = det_data.values[i] - pred_at_obs[i];
        let err_sq = det_data.errors[i] * det_data.errors[i] + 1e-10;
        chi2 += residual * residual / err_sq;
        baseline_var += (det_data.values[i] - mean_mag).powi(2) / err_sq;
    }
    let chi2_reduced = chi2 / det_data.values.len().max(1) as f64;
    let baseline_chi2 = baseline_var / det_data.values.len().max(1) as f64;

    let peak_idx = pred
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let t0 = times_pred[peak_idx];
    let peak_mag = pred[peak_idx];

    let t_last = band_data
        .times
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let t_min = *times_pred.first().unwrap_or(&t0);
    let t_max = *times_pred.last().unwrap_or(&t0);
    let fine = compute_fine_grid_metrics(gp_predictor.as_ref(), t0, t_min, t_max);
    let rise_time = fine.rise_time;
    let rise_halfmax = fine.rise_halfmax;
    let rise_efold = fine.rise_efold;
    let decay_efold = fine.decay_efold;
    let dm15 = fine.dm15;
    let decay_halfmax = fine.decay_halfmax;
    let fwhm = fine.fwhm;
    let rise_rate = fine.rise_rate;
    let decay_rate = fine.decay_rate;
    let near_peak_rise_rate = fine.near_peak_rise_rate;
    let near_peak_decay_rate = fine.near_peak_decay_rate;
    let predictive = compute_predictive_features(
        gp_predictor.as_ref(),
        t_last,
        t0,
        times_pred,
        &pred,
        &std_vec,
        &det_data.values,
        &det_data.errors,
    );

    // Decay power-law fit
    let (pl_index, pl_chi2) =
        fit_decay_power_law(times_pred, &pred, &std_vec, peak_idx);

    // GP predictions at fixed offsets from peak
    let mag_30d = predict_mag_at_offset(gp_predictor.as_ref(), t0, 30.0);
    let mag_60d = predict_mag_at_offset(gp_predictor.as_ref(), t0, 60.0);
    let mag_90d = predict_mag_at_offset(gp_predictor.as_ref(), t0, 90.0);

    // TDE vs AGN variability features
    let von_neumann = compute_von_neumann_ratio(&band_data.times, &band_data.values);
    let (pre_peak_rms, rise_over_noise) = compute_pre_peak_features(
        &band_data.times,
        &band_data.values,
        &band_data.errors,
        t0,
        peak_mag,
    );
    let monotonicity = compute_post_peak_monotonicity(&pred, peak_idx);
    let n_local_maxima = count_local_maxima(&pred, 0.1);

    let result = NonparametricBandResult {
        band: band_name.to_string(),
        rise_time: finite_or_none(rise_time),
        rise_halfmax: finite_or_none(rise_halfmax),
        rise_efold: finite_or_none(rise_efold),
        decay_efold: finite_or_none(decay_efold),
        dm15: finite_or_none(dm15),
        decay_halfmax: finite_or_none(decay_halfmax),
        t0: finite_or_none(t0),
        peak_mag: finite_or_none(peak_mag),
        chi2: finite_or_none(chi2_reduced),
        baseline_chi2: finite_or_none(baseline_chi2),
        n_obs: band_data.values.len(),
        fwhm: finite_or_none(fwhm),
        rise_rate: finite_or_none(rise_rate),
        decay_rate: finite_or_none(decay_rate),
        near_peak_rise_rate: finite_or_none(near_peak_rise_rate),
        near_peak_decay_rate: finite_or_none(near_peak_decay_rate),
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
        n_local_maxima: Some(n_local_maxima as f64),
        n_upper_limits,
        gp_fit_amp: finite_or_none(fit_amp),
        gp_fit_lengthscale: finite_or_none(fit_lengthscale),
    };

    Some((result, thermal_gp))
}

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::nonparametric::NonparametricBandResult;
use crate::parametric::ParametricBandResult;
use crate::thermal::ThermalResult;

/// Per-band time/value/error triplet for fitting.
#[derive(Debug, Clone)]
pub struct BandData {
    pub times: Vec<f64>,
    pub values: Vec<f64>,
    pub errors: Vec<f64>,
}

/// Combined result of nonparametric + parametric + thermal lightcurve fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightcurveFittingResult {
    pub nonparametric: Vec<NonparametricBandResult>,
    pub parametric: Vec<ParametricBandResult>,
    pub thermal: Option<ThermalResult>,
}

// ---------------------------------------------------------------------------
// Photometry conversion helpers (standalone, no PhotometryMag dependency)
// ---------------------------------------------------------------------------

const FACTOR: f64 = 1.0857362047581294; // 2.5 / ln(10)

/// Convert magnitude + error to flux + flux_error (ZP = 23.9).
pub fn mag2flux(mag: f64, mag_err: f64, zp: f64) -> (f64, f64) {
    let flux = 10.0_f64.powf(-0.4 * (mag - zp));
    let fluxerr = mag_err / FACTOR * flux;
    (flux, fluxerr)
}

/// Build per-band magnitude data from raw photometry arrays.
///
/// Groups by band name, converts JD to relative time (days from minimum JD).
pub fn build_mag_bands(
    times: &[f64],
    mags: &[f64],
    mag_errs: &[f64],
    bands: &[String],
) -> HashMap<String, BandData> {
    if times.is_empty() {
        return HashMap::new();
    }

    let jd_min = times.iter().copied().fold(f64::INFINITY, f64::min);

    let mut result: HashMap<String, BandData> = HashMap::new();
    for i in 0..times.len() {
        let mag = mags[i];
        let mag_err = mag_errs[i];
        if !mag.is_finite() || !mag_err.is_finite() {
            continue;
        }
        let entry = result
            .entry(bands[i].clone())
            .or_insert_with(|| BandData {
                times: Vec::new(),
                values: Vec::new(),
                errors: Vec::new(),
            });
        entry.times.push(times[i] - jd_min);
        entry.values.push(mag);
        entry.errors.push(mag_err);
    }

    result
}

/// Build per-band flux data from raw photometry arrays.
///
/// Groups by band name, converts JD to relative time, and converts mag to flux
/// using `mag2flux()` with ZP = 23.9.
pub fn build_flux_bands(
    times: &[f64],
    mags: &[f64],
    mag_errs: &[f64],
    bands: &[String],
) -> HashMap<String, BandData> {
    if times.is_empty() {
        return HashMap::new();
    }

    let jd_min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let zp = 23.9;

    let mut result: HashMap<String, BandData> = HashMap::new();
    for i in 0..times.len() {
        let (flux, flux_err) = mag2flux(mags[i], mag_errs[i], zp);
        if !flux.is_finite() || !flux_err.is_finite() || flux <= 0.0 || flux_err <= 0.0 {
            continue;
        }
        let entry = result
            .entry(bands[i].clone())
            .or_insert_with(|| BandData {
                times: Vec::new(),
                values: Vec::new(),
                errors: Vec::new(),
            });
        entry.times.push(times[i] - jd_min);
        entry.values.push(flux);
        entry.errors.push(flux_err);
    }

    result
}

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

pub fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

pub fn extract_rise_timescale(times: &[f64], mags: &[f64], peak_idx: usize) -> f64 {
    if peak_idx == 0 || peak_idx >= mags.len() {
        return f64::NAN;
    }
    let peak_mag = mags[peak_idx];
    let n = peak_idx.min(3);
    let baseline = if n > 0 {
        mags[..n].iter().sum::<f64>() / n as f64
    } else {
        peak_mag + 0.5
    };
    let target_amp = baseline + (peak_mag - baseline) * (1.0 - (-1.0_f64).exp());
    let mut closest_t_before = f64::NAN;
    let mut closest_diff = f64::INFINITY;
    for i in 0..peak_idx {
        let diff = (mags[i] - target_amp).abs();
        if diff < closest_diff {
            closest_diff = diff;
            closest_t_before = times[i];
        }
    }
    if closest_t_before.is_nan() {
        return f64::NAN;
    }
    times[peak_idx] - closest_t_before
}

pub fn extract_decay_timescale(times: &[f64], mags: &[f64], peak_idx: usize) -> f64 {
    if peak_idx >= mags.len() - 1 {
        return f64::NAN;
    }
    let peak_mag = mags[peak_idx];
    let baseline = if peak_idx < mags.len() - 1 {
        let end_idx = ((peak_idx + mags.len()) / 2).min(mags.len() - 1);
        let count = (end_idx - peak_idx).max(1);
        mags[peak_idx + 1..=end_idx].iter().sum::<f64>() / count as f64
    } else {
        peak_mag + 0.5
    };
    let target_amp = baseline + (peak_mag - baseline) * (-1.0_f64).exp();
    let mut closest_t_after = f64::NAN;
    let mut closest_diff = f64::INFINITY;
    for i in (peak_idx + 1)..mags.len() {
        let diff = (mags[i] - target_amp).abs();
        if diff < closest_diff {
            closest_diff = diff;
            closest_t_after = times[i];
        }
    }
    if closest_t_after.is_nan() {
        return f64::NAN;
    }
    closest_t_after - times[peak_idx]
}

pub fn compute_fwhm(times: &[f64], mags: &[f64], peak_idx: usize) -> (f64, f64, f64) {
    if peak_idx >= mags.len() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let peak_mag = mags[peak_idx];
    let half_max_mag = peak_mag + 0.75;
    let mut t_before = f64::NAN;
    for i in (0..peak_idx).rev() {
        if mags[i] >= half_max_mag {
            t_before = times[i];
            break;
        }
    }
    let mut t_after = f64::NAN;
    for i in (peak_idx + 1)..mags.len() {
        if mags[i] >= half_max_mag {
            t_after = times[i];
            break;
        }
    }
    if t_before.is_nan() || t_after.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    (t_after - t_before, t_before, t_after)
}

pub fn compute_rise_rate(times: &[f64], mags: &[f64]) -> f64 {
    if times.len() < 2 {
        return f64::NAN;
    }
    let n_early = (times.len() as f64 * 0.25).ceil() as usize;
    let n_early = n_early.max(2).min(times.len() - 1);
    let early_times = &times[0..n_early];
    let early_mags = &mags[0..n_early];
    let n = early_times.len() as f64;
    let sum_t: f64 = early_times.iter().sum();
    let sum_m: f64 = early_mags.iter().sum();
    let sum_tt: f64 = early_times.iter().map(|t| t * t).sum();
    let sum_tm: f64 = early_times
        .iter()
        .zip(early_mags.iter())
        .map(|(t, m)| t * m)
        .sum();
    let denominator = n * sum_tt - sum_t * sum_t;
    if denominator.abs() < 1e-10 {
        return f64::NAN;
    }
    (n * sum_tm - sum_t * sum_m) / denominator
}

pub fn compute_decay_rate(times: &[f64], mags: &[f64]) -> f64 {
    if times.len() < 2 {
        return f64::NAN;
    }
    let n_late = (times.len() as f64 * 0.25).ceil() as usize;
    let n_late = n_late.max(2).min(times.len());
    let start_idx = times.len() - n_late;
    let late_times = &times[start_idx..];
    let late_mags = &mags[start_idx..];
    let n = late_times.len() as f64;
    let sum_t: f64 = late_times.iter().sum();
    let sum_m: f64 = late_mags.iter().sum();
    let sum_tt: f64 = late_times.iter().map(|t| t * t).sum();
    let sum_tm: f64 = late_times
        .iter()
        .zip(late_mags.iter())
        .map(|(t, m)| t * m)
        .sum();
    let denominator = n * sum_tt - sum_t * sum_t;
    if denominator.abs() < 1e-10 {
        return f64::NAN;
    }
    (n * sum_tm - sum_t * sum_m) / denominator
}

/// Convert NaN/Inf to None for JSON safety.
pub fn finite_or_none(v: f64) -> Option<f64> {
    if v.is_finite() {
        Some(v)
    } else {
        None
    }
}

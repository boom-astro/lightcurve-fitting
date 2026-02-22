use std::collections::HashMap;

use argmin::core::{CostFunction, Error as ArgminError, Executor, State};
use argmin::solver::particleswarm::ParticleSwarm;
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use sklears_core::traits::Predict;
use sklears_gaussian_process::{GaussianProcessRegressor, GprTrained};

use crate::common::BandData;
use crate::gp::{fit_sklears_gp, subsample_data};

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// hc/k in Å·K units for blackbody exponent: hc / k_B = 1.4388e8 Å·K
const HC_OVER_K: f64 = 1.4388e8;

/// ZTF effective wavelengths in Å
const LAMBDA_G: f64 = 4770.0;
const LAMBDA_R: f64 = 6231.0;
const LAMBDA_I: f64 = 7625.0;

/// Map band name to effective wavelength in Å.
fn band_wavelength(band: &str) -> Option<f64> {
    match band {
        "g" => Some(LAMBDA_G),
        "r" => Some(LAMBDA_R),
        "i" => Some(LAMBDA_I),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Blackbody color model
// ---------------------------------------------------------------------------

/// Compute the blackbody color difference in magnitudes between a reference
/// band and another band at temperature `temp` (in Kelvin).
///
/// Δm = -2.5 * log10( (λ/λ_ref)^3 * (exp(hc/λkT) - 1) / (exp(hc/λ_ref·kT) - 1) )
///
/// Uses Wien-regime approximation when exponents are very large (T very low)
/// to avoid overflow.
fn bb_color_mag(lambda_ref: f64, lambda: f64, temp: f64) -> f64 {
    let x_ref = HC_OVER_K / (lambda_ref * temp);
    let x = HC_OVER_K / (lambda * temp);

    // Wien regime guard: if exponents are large, use exp approximation
    // (exp(x) - 1 ≈ exp(x) when x >> 1)
    if x > 500.0 || x_ref > 500.0 {
        // Δm = -2.5 * (3*log10(λ/λ_ref) + (x - x_ref) * log10(e))
        let log10_e = std::f64::consts::LOG10_E;
        return -2.5 * (3.0 * (lambda / lambda_ref).log10() + (x - x_ref) * log10_e);
    }

    let ratio = (lambda / lambda_ref).powi(3) * (x.exp() - 1.0) / (x_ref.exp() - 1.0);
    if ratio <= 0.0 || !ratio.is_finite() {
        return f64::NAN;
    }
    -2.5 * ratio.log10()
}

/// Temperature model: T(t) = 10^(log_T0 + cooling_rate * t)
fn temperature_at(log_t0: f64, cooling_rate: f64, t: f64) -> f64 {
    10.0_f64.powf(log_t0 + cooling_rate * t)
}

// ---------------------------------------------------------------------------
// PSO cost function
// ---------------------------------------------------------------------------

/// A single color observation: time, observed color (mag difference), and error.
#[derive(Clone)]
struct ColorObs {
    time: f64,
    observed_color: f64,
    error: f64,
    lambda: f64,
}

#[derive(Clone)]
struct ThermalCost {
    observations: Vec<ColorObs>,
    lambda_ref: f64,
}

impl CostFunction for ThermalCost {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        let log_t0 = p[0];
        let cooling_rate = p[1];

        let mut chi2 = 0.0;
        for obs in &self.observations {
            let temp = temperature_at(log_t0, cooling_rate, obs.time);
            let model_color = bb_color_mag(self.lambda_ref, obs.lambda, temp);
            if !model_color.is_finite() {
                return Ok(1e10);
            }
            let residual = obs.observed_color - model_color;
            let err_sq = obs.error * obs.error + 1e-10;
            chi2 += residual * residual / err_sq;
        }

        let n = self.observations.len().max(1) as f64;
        Ok(chi2 / n)
    }
}

// ---------------------------------------------------------------------------
// Result struct
// ---------------------------------------------------------------------------

/// Result of blackbody temperature fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalResult {
    pub log_temp_peak: Option<f64>,
    pub cooling_rate: Option<f64>,
    pub log_temp_peak_err: Option<f64>,
    pub cooling_rate_err: Option<f64>,
    pub chi2: Option<f64>,
    pub n_color_obs: usize,
    pub n_bands_used: usize,
    pub ref_band: String,
}

// ---------------------------------------------------------------------------
// GP fitting for reference band (lighter grid search than nonparametric)
// ---------------------------------------------------------------------------

/// Fit a GP to the reference band with a smaller grid search (3×3×2 = 18 combos).
fn fit_ref_band_gp(
    band_data: &BandData,
    duration: f64,
) -> Option<sklears_gaussian_process::GaussianProcessRegressor<GprTrained>> {
    let max_subsample = if band_data.times.len() <= 30 {
        band_data.times.len()
    } else {
        25
    };
    let (times_sub, mags_sub, errors_sub) =
        subsample_data(&band_data.times, &band_data.values, &band_data.errors, max_subsample);

    let times_arr = Array1::from_vec(times_sub);
    let mags_arr = Array1::from_vec(mags_sub.clone());

    let avg_error_var = if !errors_sub.is_empty() {
        errors_sub.iter().map(|e| e * e).sum::<f64>() / errors_sub.len() as f64
    } else {
        1e-4
    };

    let amp_candidates = [0.1, 0.3];
    let ls_factors = [6.0, 12.0, 24.0];
    let alpha_candidates = [avg_error_var.max(1e-6)];

    let xt_sub = scirs2_core::ndarray::Array1::from_vec(band_data.times.clone())
        .view()
        .insert_axis(scirs2_core::ndarray::Axis(1))
        .to_owned();

    let mut best_gp = None;
    let mut best_score = f64::INFINITY;

    for &amp in &amp_candidates {
        for &factor in &ls_factors {
            let lengthscale = (duration / factor).max(0.1);

            for &alpha in &alpha_candidates {
                if let Some(trained) =
                    fit_sklears_gp(&times_arr, &mags_arr, amp, lengthscale, alpha)
                {
                    let xt_fit = times_arr
                        .view()
                        .insert_axis(scirs2_core::ndarray::Axis(1))
                        .to_owned();
                    if let Ok(pred_at_obs) = trained.predict(&xt_fit) {
                        let mut residuals_sq = 0.0f64;
                        for i in 0..mags_arr.len() {
                            let residual = mags_arr[i] - pred_at_obs[i];
                            residuals_sq += residual * residual;
                        }
                        let rms = (residuals_sq / mags_arr.len() as f64).sqrt();

                        // Reject wildly extrapolating fits
                        if let Ok(pred_full) = trained.predict(&xt_sub) {
                            let pred_min =
                                pred_full.iter().cloned().fold(f64::INFINITY, f64::min);
                            let obs_min =
                                mags_arr.iter().cloned().fold(f64::INFINITY, f64::min);
                            if pred_min.is_finite() && (pred_min - obs_min).abs() > 6.0 {
                                continue;
                            }
                        }

                        if rms.is_finite() && rms < best_score {
                            best_score = rms;
                            best_gp = Some(trained);
                        }
                    }
                }
            }
        }
    }

    best_gp
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Fit a blackbody temperature model to cross-band color differences.
///
/// If `prefit_gps` is provided (e.g. from `fit_nonparametric`), the reference
/// band GP is reused instead of fitting a new one.
pub fn fit_thermal(
    bands: &HashMap<String, BandData>,
    prefit_gps: Option<&HashMap<String, GaussianProcessRegressor<GprTrained>>>,
) -> Option<ThermalResult> {
    // Choose reference band: prefer g (bluest = most temperature-sensitive)
    let ref_band_name = if bands.get("g").map_or(false, |b| b.times.len() >= 5) {
        "g"
    } else if bands.get("r").map_or(false, |b| b.times.len() >= 5) {
        "r"
    } else {
        return None;
    };

    let ref_data = bands.get(ref_band_name)?;
    let lambda_ref = band_wavelength(ref_band_name)?;

    // Compute overall time range
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
        return None;
    }

    // Reuse pre-fitted GP if available, otherwise fit one
    let gp = if let Some(gps) = prefit_gps {
        if let Some(existing) = gps.get(ref_band_name) {
            existing.clone()
        } else {
            fit_ref_band_gp(ref_data, duration)?
        }
    } else {
        fit_ref_band_gp(ref_data, duration)?
    };

    // Build color observations from non-reference bands
    let mut color_obs = Vec::new();
    let mut bands_used = std::collections::HashSet::new();

    for (band_name, band_data) in bands {
        if band_name == ref_band_name {
            continue;
        }
        let lambda = match band_wavelength(band_name) {
            Some(l) => l,
            None => continue,
        };

        for i in 0..band_data.times.len() {
            let t = band_data.times[i];
            let obs_mag = band_data.values[i];
            let obs_err = band_data.errors[i];

            // Predict reference-band magnitude at this time via GP
            let t_2d = scirs2_core::ndarray::Array2::from_shape_fn((1, 1), |_| t);
            let ref_mag = match gp.predict(&t_2d) {
                Ok(pred) => pred[0],
                Err(_) => continue,
            };

            if !ref_mag.is_finite() {
                continue;
            }

            // Color = ref_mag - obs_mag (ref band minus this band)
            let observed_color = ref_mag - obs_mag;
            // Propagate error: GP uncertainty + photometric error
            // Use photometric error as lower bound (GP uncertainty is harder to get per-point)
            let color_err = (obs_err * obs_err + 0.02 * 0.02).sqrt(); // 0.02 mag GP floor

            color_obs.push(ColorObs {
                time: t,
                observed_color,
                error: color_err,
                lambda,
            });
            bands_used.insert(band_name.clone());
        }
    }

    let n_color_obs = color_obs.len();
    let n_bands_used = bands_used.len();

    // Need at least 3 color observations
    if n_color_obs < 3 {
        return Some(ThermalResult {
            log_temp_peak: None,
            cooling_rate: None,
            log_temp_peak_err: None,
            cooling_rate_err: None,
            chi2: None,
            n_color_obs,
            n_bands_used,
            ref_band: ref_band_name.to_string(),
        });
    }

    // PSO bounds: log_T0 ∈ [3.0, 6.0], cooling_rate ∈ [-0.05, 0.01]
    let lower = vec![3.0, -0.05];
    let upper = vec![6.0, 0.01];

    let problem = ThermalCost {
        observations: color_obs.clone(),
        lambda_ref,
    };

    // Run 2 PSO restarts to estimate uncertainties (2D problem converges fast)
    let n_restarts = 2;
    let mut best_params_all: Vec<[f64; 2]> = Vec::new();
    let mut best_cost_overall = f64::INFINITY;
    let mut best_params_overall = [4.5, -0.005]; // sensible default

    for _ in 0..n_restarts {
        let solver = ParticleSwarm::new((lower.clone(), upper.clone()), 10);
        let res = Executor::new(problem.clone(), solver)
            .configure(|state| state.max_iters(30))
            .run();

        if let Ok(result) = res {
            let cost = result.state().get_cost();
            if let Some(particle) = result.state().get_best_param() {
                let pos = &particle.position;
                best_params_all.push([pos[0], pos[1]]);
                if cost < best_cost_overall {
                    best_cost_overall = cost;
                    best_params_overall = [pos[0], pos[1]];
                }
            }
        }
    }

    if best_params_all.is_empty() {
        return Some(ThermalResult {
            log_temp_peak: None,
            cooling_rate: None,
            log_temp_peak_err: None,
            cooling_rate_err: None,
            chi2: None,
            n_color_obs,
            n_bands_used,
            ref_band: ref_band_name.to_string(),
        });
    }

    // Estimate uncertainties from std dev of best particles across restarts
    let (log_t0_err, cooling_rate_err) = if best_params_all.len() >= 2 {
        let mean_log_t0 =
            best_params_all.iter().map(|p| p[0]).sum::<f64>() / best_params_all.len() as f64;
        let mean_cr =
            best_params_all.iter().map(|p| p[1]).sum::<f64>() / best_params_all.len() as f64;
        let var_log_t0 = best_params_all
            .iter()
            .map(|p| (p[0] - mean_log_t0).powi(2))
            .sum::<f64>()
            / best_params_all.len() as f64;
        let var_cr = best_params_all
            .iter()
            .map(|p| (p[1] - mean_cr).powi(2))
            .sum::<f64>()
            / best_params_all.len() as f64;
        (Some(var_log_t0.sqrt()), Some(var_cr.sqrt()))
    } else {
        (None, None)
    };

    let chi2 = if best_cost_overall.is_finite() {
        Some(best_cost_overall)
    } else {
        None
    };

    Some(ThermalResult {
        log_temp_peak: Some(best_params_overall[0]),
        cooling_rate: Some(best_params_overall[1]),
        log_temp_peak_err: log_t0_err,
        cooling_rate_err,
        chi2,
        n_color_obs,
        n_bands_used,
        ref_band: ref_band_name.to_string(),
    })
}

use std::collections::HashMap;

use argmin::core::{CostFunction, Error as ArgminError, Executor, State};
use argmin::solver::particleswarm::ParticleSwarm;
use serde::{Deserialize, Serialize};

use crate::common::BandData;
use crate::gp::subsample_data;
use crate::sparse_gp::DenseGP;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// hc/k in Å·K units for blackbody exponent: hc / k_B = 1.4388e8 Å·K
const HC_OVER_K: f64 = 1.4388e8;

/// Effective wavelengths in Å for supported bands.
/// ZTF: g=4770, r=6231, i=7625
/// LSST: u=3671, g=4827, r=6223, i=7546, z=8691, y=9712
const LAMBDA_U: f64 = 3671.0; // LSST u
const LAMBDA_G: f64 = 4770.0; // ZTF g / LSST g (~4827, close enough)
const LAMBDA_R: f64 = 6231.0; // ZTF r / LSST r (~6223, close enough)
const LAMBDA_I: f64 = 7625.0; // ZTF i / LSST i (~7546, close enough)
const LAMBDA_Z: f64 = 8691.0; // LSST z
const LAMBDA_Y: f64 = 9712.0; // LSST y

/// Find the first band name from a list of aliases that exists in the data with ≥5 points.
fn find_band_name<'a>(bands: &'a HashMap<String, BandData>, aliases: &[&str]) -> Option<&'a str> {
    for alias in aliases {
        if let Some(bd) = bands.get(*alias) {
            if bd.times.len() >= 5 {
                return bands.keys().find(|k| k.as_str() == *alias).map(|k| k.as_str());
            }
        }
    }
    None
}

/// Map band name to effective wavelength in Å.
/// Recognizes ZTF ("ztfg", "ztfr", "ztfi"), LSST ("lsstu".."lssty"),
/// and short names ("u", "g", "r", "i", "z", "y").
fn band_wavelength(band: &str) -> Option<f64> {
    match band {
        "u" | "lsstu" | "U" => Some(LAMBDA_U),
        "B" => Some(4361.0),
        "g" | "ztfg" | "lsstg" => Some(LAMBDA_G),
        "c" => Some(5330.0),   // ATLAS cyan
        "V" => Some(5448.0),
        "r" | "ztfr" | "lsstr" => Some(LAMBDA_R),
        "R" => Some(6555.0),
        "o" => Some(6790.0),   // ATLAS orange
        "i" | "ztfi" | "lssti" => Some(LAMBDA_I),
        "I" => Some(8060.0),
        "z" | "lsstz" => Some(LAMBDA_Z),
        "y" | "lssty" => Some(LAMBDA_Y),
        "J" => Some(12350.0),
        "H" => Some(16620.0),
        "K" | "Ks" => Some(21590.0),
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
    /// log10(temperature) at the latest observation time.
    pub log_temp_latest: Option<f64>,
    pub cooling_rate: Option<f64>,
    pub log_temp_peak_err: Option<f64>,
    pub cooling_rate_err: Option<f64>,
    pub chi2: Option<f64>,
    pub n_color_obs: usize,
    pub n_bands_used: usize,
    pub ref_band: String,
}

// ---------------------------------------------------------------------------
// GP fitting for reference band
// ---------------------------------------------------------------------------

/// Fit a DenseGP to the reference band with a grid search.
fn fit_ref_band_gp(band_data: &BandData, duration: f64) -> Option<DenseGP> {
    let max_subsample = if band_data.times.len() <= 30 {
        band_data.times.len()
    } else {
        25
    };
    let (times_sub, mags_sub, errors_sub) =
        subsample_data(&band_data.times, &band_data.values, &band_data.errors, max_subsample);
    let noise_var: Vec<f64> = errors_sub.iter().map(|e| (e * e).max(1e-6)).collect();

    let amp_candidates = [0.1, 0.3];
    let ls_factors = [6.0, 12.0, 24.0];

    let mut best_gp: Option<DenseGP> = None;
    let mut best_score = f64::INFINITY;

    for &amp in &amp_candidates {
        for &factor in &ls_factors {
            let lengthscale = (duration / factor).max(0.1);
            if let Some(gp) = DenseGP::fit(&times_sub, &mags_sub, &noise_var, amp, lengthscale) {
                let pred_at_obs = gp.predict(&times_sub);
                let mut rss = 0.0f64;
                for i in 0..mags_sub.len() {
                    let r = mags_sub[i] - pred_at_obs[i];
                    rss += r * r;
                }
                let rms = (rss / mags_sub.len() as f64).sqrt();

                // Reject wildly extrapolating fits
                let pred_full = gp.predict(&band_data.times);
                let pred_min = pred_full.iter().cloned().fold(f64::INFINITY, f64::min);
                let obs_min = mags_sub.iter().cloned().fold(f64::INFINITY, f64::min);
                if pred_min.is_finite() && (pred_min - obs_min).abs() > 6.0 {
                    continue;
                }

                if rms.is_finite() && rms < best_score {
                    best_score = rms;
                    best_gp = Some(gp);
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
    prefit_gps: Option<&HashMap<String, DenseGP>>,
) -> Option<ThermalResult> {
    // Choose reference band: prefer bluest available (most temperature-sensitive).
    // Try u (LSST) → g (ZTF/LSST) → r (ZTF/LSST).
    let ref_band_name = find_band_name(bands, &["u", "lsstu"])
        .or_else(|| find_band_name(bands, &["g", "ztfg", "lsstg"]))
        .or_else(|| find_band_name(bands, &["r", "ztfr", "lsstr"]))?;

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
            let ref_pred = gp.predict(&[t]);
            let ref_mag = ref_pred[0];

            if !ref_mag.is_finite() {
                continue;
            }

            // Color = ref_mag - obs_mag (ref band minus this band)
            let observed_color = ref_mag - obs_mag;
            // Propagate error: GP uncertainty + photometric error
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
            log_temp_latest: None,
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
            log_temp_latest: None,
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

    // Compute temperature at the latest observation time, clamped to physical range
    let t_last = color_obs
        .iter()
        .map(|o| o.time)
        .fold(f64::NEG_INFINITY, f64::max);
    let log_temp_latest = if t_last.is_finite() {
        let lt = best_params_overall[0] + best_params_overall[1] * t_last;
        if lt.is_finite() { Some(lt.clamp(2.0, 7.0)) } else { None }
    } else {
        None
    };

    Some(ThermalResult {
        log_temp_peak: Some(best_params_overall[0]),
        log_temp_latest,
        cooling_rate: Some(best_params_overall[1]),
        log_temp_peak_err: log_t0_err,
        cooling_rate_err,
        chi2,
        n_color_obs,
        n_bands_used,
        ref_band: ref_band_name.to_string(),
    })
}

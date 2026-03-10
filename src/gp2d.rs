//! 2D Gaussian Process for joint time × wavelength fitting.
//!
//! Fits a single GP surface to multi-band photometry, where each observation
//! is a point in (time, log10_wavelength) space. This naturally captures
//! cross-band color evolution and makes SED extraction trivial.
//!
//! Key advantages over per-band 1D GPs:
//! - Bands constrain each other: a gap in g-band is informed by r-band data
//! - Color evolution falls out of the GP posterior without explicit color matching
//! - Thermal (blackbody) fitting reduces to evaluating the GP at fixed time,
//!   sweeping wavelength, and fitting a Planck curve

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::common::BandData;

// ---------------------------------------------------------------------------
// 2D anisotropic RBF kernel
// ---------------------------------------------------------------------------

/// k((t1,w1), (t2,w2)) = amp * exp(-0.5 * ((t1-t2)/l_t)^2 - 0.5 * ((w1-w2)/l_w)^2)
#[inline(always)]
fn rbf_2d(
    t1: f64, w1: f64,
    t2: f64, w2: f64,
    amp: f64, inv_2lt2: f64, inv_2lw2: f64,
) -> f64 {
    let dt = t1 - t2;
    let dw = w1 - w2;
    amp * (-dt * dt * inv_2lt2 - dw * dw * inv_2lw2).exp()
}

// ---------------------------------------------------------------------------
// Cholesky & solves (reused from sparse_gp.rs, inlined here for independence)
// ---------------------------------------------------------------------------

fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut s = a[j * n + j];
        for k in 0..j {
            s -= a[j * n + k] * a[j * n + k];
        }
        if s <= 0.0 { return false; }
        a[j * n + j] = s.sqrt();
        let ljj = a[j * n + j];
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / ljj;
        }
        for i in 0..j { a[i * n + j] = 0.0; }
    }
    true
}

fn solve_l(l: &[f64], b: &[f64], x: &mut [f64], n: usize) {
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i { s -= l[i * n + j] * x[j]; }
        x[i] = s / l[i * n + i];
    }
}

fn solve_lt(l: &[f64], b: &[f64], x: &mut [f64], n: usize) {
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n { s -= l[j * n + i] * x[j]; }
        x[i] = s / l[i * n + i];
    }
}

// ---------------------------------------------------------------------------
// Effective wavelengths (duplicated from thermal.rs for independence)
// ---------------------------------------------------------------------------

fn band_wavelength(band: &str) -> Option<f64> {
    match band {
        "u" | "lsstu" | "U" => Some(3671.0),
        "B" => Some(4361.0),
        "g" | "ztfg" | "lsstg" => Some(4770.0),
        "c" => Some(5330.0),   // ATLAS cyan
        "V" => Some(5448.0),
        "r" | "ztfr" | "lsstr" => Some(6231.0),
        "R" => Some(6555.0),
        "o" => Some(6790.0),   // ATLAS orange
        "i" | "ztfi" | "lssti" => Some(7625.0),
        "I" => Some(8060.0),
        "z" | "lsstz" => Some(8691.0),
        "y" | "lssty" => Some(9712.0),
        "J" => Some(12350.0),
        "H" => Some(16620.0),
        "K" | "Ks" => Some(21590.0),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// DenseGP2D
// ---------------------------------------------------------------------------

/// 2D GP for joint time × wavelength fitting.
#[derive(Clone)]
pub struct DenseGP2D {
    /// Training times (relative days).
    t: Vec<f64>,
    /// Training log10(wavelength_angstrom).
    w: Vec<f64>,
    /// Cholesky factor (n×n, row-major).
    l: Vec<f64>,
    /// Alpha = (K + σ²I)^{-1} (y - y_mean).
    alpha: Vec<f64>,
    n: usize,
    amp: f64,
    inv_2lt2: f64,
    inv_2lw2: f64,
    y_mean: f64,
}

impl DenseGP2D {
    /// Fit a 2D GP to observations at (time, log10_wavelength) with values y.
    pub fn fit(
        times: &[f64],
        log_wavelengths: &[f64],
        values: &[f64],
        noise_var: &[f64],
        amp: f64,
        ls_time: f64,
        ls_wave: f64,
    ) -> Option<Self> {
        let n = times.len();
        if n == 0 { return None; }

        let inv_2lt2 = 0.5 / (ls_time * ls_time);
        let inv_2lw2 = 0.5 / (ls_wave * ls_wave);
        let y_mean = values.iter().sum::<f64>() / n as f64;

        // Build K + noise*I
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = rbf_2d(
                    times[i], log_wavelengths[i],
                    times[j], log_wavelengths[j],
                    amp, inv_2lt2, inv_2lw2,
                );
                k[i * n + j] = v;
                k[j * n + i] = v;
            }
            k[i * n + i] += noise_var[i].max(1e-10);
        }

        if !cholesky(&mut k, n) { return None; }
        let l = k;

        let y_centered: Vec<f64> = values.iter().map(|v| v - y_mean).collect();
        let mut tmp = vec![0.0; n];
        let mut alpha = vec![0.0; n];
        solve_l(&l, &y_centered, &mut tmp, n);
        solve_lt(&l, &tmp, &mut alpha, n);

        Some(DenseGP2D {
            t: times.to_vec(),
            w: log_wavelengths.to_vec(),
            l, alpha, n, amp, inv_2lt2, inv_2lw2, y_mean,
        })
    }

    /// Predict at query points (times, log_wavelengths).
    pub fn predict(&self, qt: &[f64], qw: &[f64]) -> Vec<f64> {
        let nq = qt.len();
        let mut out = Vec::with_capacity(nq);
        for i in 0..nq {
            let mut dot = 0.0;
            for j in 0..self.n {
                dot += rbf_2d(qt[i], qw[i], self.t[j], self.w[j],
                    self.amp, self.inv_2lt2, self.inv_2lw2) * self.alpha[j];
            }
            out.push(dot + self.y_mean);
        }
        out
    }

    /// Predict with posterior standard deviation.
    pub fn predict_with_std(&self, qt: &[f64], qw: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let nq = qt.len();
        let mut means = Vec::with_capacity(nq);
        let mut stds = Vec::with_capacity(nq);
        let mut k_star = vec![0.0; self.n];
        let mut v = vec![0.0; self.n];

        for i in 0..nq {
            let mut dot = 0.0;
            for j in 0..self.n {
                k_star[j] = rbf_2d(qt[i], qw[i], self.t[j], self.w[j],
                    self.amp, self.inv_2lt2, self.inv_2lw2);
                dot += k_star[j] * self.alpha[j];
            }
            means.push(dot + self.y_mean);

            solve_l(&self.l, &k_star, &mut v, self.n);
            let vtv: f64 = v[..self.n].iter().map(|x| x * x).sum();
            let var = (self.amp - vtv).max(1e-10);
            stds.push(var.sqrt());
        }

        (means, stds)
    }

    /// Train RMS (predictions at training points vs targets).
    pub fn train_rms(&self, values: &[f64]) -> f64 {
        let pred = self.predict(&self.t, &self.w);
        let n = pred.len().max(1) as f64;
        let rss: f64 = pred.iter().zip(values.iter())
            .map(|(p, v)| (p - v) * (p - v)).sum();
        (rss / n).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a 2D GP fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gp2dResult {
    /// Best-fit hyperparameters: (amp, ls_time, ls_wave).
    pub amp: f64,
    pub ls_time: f64,
    pub ls_wave: f64,
    /// Train RMS.
    pub train_rms: f64,
    /// Number of training points.
    pub n_train: usize,
    /// Number of bands used.
    pub n_bands: usize,
    /// Band names in order.
    pub bands: Vec<String>,
    /// Log10 wavelengths for each band (same order as `bands`).
    pub band_log_wavelengths: Vec<f64>,
}

/// Thermal result extracted from the 2D GP surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gp2dThermalResult {
    /// Time grid (relative days) at which temperature was estimated.
    pub times: Vec<f64>,
    /// log10(T_bb) at each time.
    pub log_temps: Vec<f64>,
    /// Reduced chi2 of blackbody fit at each time.
    pub chi2s: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Blackbody fitting from GP surface
// ---------------------------------------------------------------------------

/// Planck function in magnitude space (up to a constant offset).
/// Returns relative magnitude: -2.5 * log10(B_lambda) where
/// B_lambda ∝ lambda^{-5} / (exp(hc / lambda k T) - 1)
fn planck_relative_mag(lambda_angstrom: f64, temp: f64) -> f64 {
    // hc/k in Å·K
    const HC_OVER_K: f64 = 1.4387769e8; // h*c/k_B in Å·K
    let x = HC_OVER_K / (lambda_angstrom * temp);
    // B_lambda ∝ lambda^-5 / (exp(x) - 1)
    let log_b = -5.0 * lambda_angstrom.ln() - (x.exp() - 1.0).ln();
    -2.5 * log_b / std::f64::consts::LN_10
}

/// Fit a blackbody temperature to magnitudes at several wavelengths.
/// Returns (log10_temp, chi2, offset) where offset is the magnitude zero-point.
fn fit_bb_temperature(
    wavelengths_angstrom: &[f64],
    mags: &[f64],
    mag_errs: &[f64],
) -> Option<(f64, f64, f64)> {
    if wavelengths_angstrom.len() < 2 { return None; }

    // Grid search over log10(T) from 3.0 to 5.5 (1,000 K to 316,000 K)
    let n_grid = 200;
    let mut best_chi2 = f64::INFINITY;
    let mut best_log_t = 4.0;
    let mut best_offset = 0.0;
    let n = wavelengths_angstrom.len() as f64;

    for i in 0..n_grid {
        let log_t = 3.0 + (i as f64) * 2.5 / (n_grid as f64 - 1.0);
        let temp = 10.0_f64.powf(log_t);

        // Compute model magnitudes and best-fit offset
        let model: Vec<f64> = wavelengths_angstrom.iter()
            .map(|&lam| planck_relative_mag(lam, temp))
            .collect();

        // Weighted least-squares offset: sum(w * (data - model)) / sum(w)
        let mut sum_w = 0.0;
        let mut sum_w_dm = 0.0;
        for j in 0..mags.len() {
            let w = 1.0 / (mag_errs[j] * mag_errs[j]).max(1e-10);
            sum_w += w;
            sum_w_dm += w * (mags[j] - model[j]);
        }
        let offset = sum_w_dm / sum_w;

        // Chi2
        let chi2: f64 = mags.iter().zip(model.iter()).zip(mag_errs.iter())
            .map(|((&m, &mod_m), &e)| {
                let r = m - mod_m - offset;
                r * r / (e * e).max(1e-10)
            })
            .sum::<f64>() / (n - 2.0).max(1.0);

        if chi2 < best_chi2 {
            best_chi2 = chi2;
            best_log_t = log_t;
            best_offset = offset;
        }
    }

    Some((best_log_t, best_chi2, best_offset))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fit a 2D GP to multi-band magnitude data.
///
/// Concatenates all bands into a single training set with (time, log10_wavelength)
/// inputs. Performs grid search over hyperparameters.
///
/// Returns the fitted GP and a result struct, or None if insufficient data.
pub fn fit_gp_2d(
    mag_bands: &HashMap<String, BandData>,
) -> Option<(DenseGP2D, Gp2dResult)> {
    // Collect bands with known wavelengths
    let mut all_times = Vec::new();
    let mut all_log_wav = Vec::new();
    let mut all_mags = Vec::new();
    let mut all_noise_var = Vec::new();
    let mut band_names = Vec::new();
    let mut band_log_wavs = Vec::new();

    for (name, bd) in mag_bands {
        let wav = match band_wavelength(name) {
            Some(w) => w,
            None => continue,
        };
        if bd.times.len() < 3 { continue; }

        let log_w = wav.log10();
        band_names.push(name.clone());
        band_log_wavs.push(log_w);

        for i in 0..bd.times.len() {
            all_times.push(bd.times[i]);
            all_log_wav.push(log_w);
            all_mags.push(bd.values[i]);
            let e = bd.errors[i];
            all_noise_var.push(e * e);
        }
    }

    let n = all_times.len();
    if n < 5 || band_names.len() < 2 { return None; }

    // Subsample if too large for dense GP (O(n^3))
    let max_n = 200;
    let (sub_t, sub_w, sub_m, sub_nv) = if n <= max_n {
        (all_times.clone(), all_log_wav.clone(), all_mags.clone(), all_noise_var.clone())
    } else {
        let step = n as f64 / max_n as f64;
        let mut st = Vec::with_capacity(max_n);
        let mut sw = Vec::with_capacity(max_n);
        let mut sm = Vec::with_capacity(max_n);
        let mut sn = Vec::with_capacity(max_n);
        for i in 0..max_n {
            let idx = (i as f64 * step) as usize;
            st.push(all_times[idx]);
            sw.push(all_log_wav[idx]);
            sm.push(all_mags[idx]);
            sn.push(all_noise_var[idx]);
        }
        (st, sw, sm, sn)
    };

    // Time extent
    let t_min = sub_t.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = sub_t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let duration = (t_max - t_min).max(1.0);

    // Wavelength extent (in log10 space)
    let w_min = sub_w.iter().cloned().fold(f64::INFINITY, f64::min);
    let w_max = sub_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w_range = (w_max - w_min).max(0.01);

    // Grid search over hyperparameters
    let amp_candidates = [0.3, 1.0, 3.0, 10.0];
    let ls_time_candidates = [duration / 4.0, duration / 8.0, duration / 16.0];
    // Wavelength lengthscale: the range in log10(Å) between bands is ~0.2
    // (g=3.678, i=3.882). Try a range of scales.
    let ls_wave_candidates = [w_range * 2.0, w_range, w_range / 2.0];

    let mut best_rms = f64::INFINITY;
    let mut best_gp: Option<DenseGP2D> = None;
    let mut best_hp = (0.3, duration / 8.0, w_range);

    for &amp in &amp_candidates {
        for &ls_t in &ls_time_candidates {
            for &ls_w in &ls_wave_candidates {
                if let Some(gp) = DenseGP2D::fit(&sub_t, &sub_w, &sub_m, &sub_nv, amp, ls_t, ls_w) {
                    let rms = gp.train_rms(&sub_m);
                    if rms < best_rms {
                        best_rms = rms;
                        best_gp = Some(gp);
                        best_hp = (amp, ls_t, ls_w);
                    }
                }
            }
        }
    }

    let gp = best_gp?;
    let result = Gp2dResult {
        amp: best_hp.0,
        ls_time: best_hp.1,
        ls_wave: best_hp.2,
        train_rms: best_rms,
        n_train: sub_t.len(),
        n_bands: band_names.len(),
        bands: band_names,
        band_log_wavelengths: band_log_wavs,
    };

    Some((gp, result))
}

/// Extract thermal (blackbody temperature) evolution from a fitted 2D GP.
///
/// At each time in `eval_times`, evaluates the GP at all known band wavelengths
/// and fits a Planck curve to the resulting SED.
pub fn extract_thermal_from_gp2d(
    gp: &DenseGP2D,
    band_wavelengths: &[(String, f64)], // (name, wavelength_angstrom)
    eval_times: &[f64],
) -> Gp2dThermalResult {
    let n_bands = band_wavelengths.len();
    let log_wavs: Vec<f64> = band_wavelengths.iter().map(|(_, w)| w.log10()).collect();
    let wavs: Vec<f64> = band_wavelengths.iter().map(|(_, w)| *w).collect();

    let mut times_out = Vec::new();
    let mut log_temps = Vec::new();
    let mut chi2s = Vec::new();

    for &t in eval_times {
        // Evaluate GP at this time across all wavelengths
        let qt: Vec<f64> = vec![t; n_bands];
        let (mags, stds) = gp.predict_with_std(&qt, &log_wavs);

        // Fit blackbody
        if let Some((log_t, chi2, _offset)) = fit_bb_temperature(&wavs, &mags, &stds) {
            times_out.push(t);
            log_temps.push(log_t);
            chi2s.push(chi2);
        }
    }

    Gp2dThermalResult {
        times: times_out,
        log_temps,
        chi2s,
    }
}

/// Convenience: fit 2D GP and extract thermal evolution in one call.
pub fn fit_gp_2d_with_thermal(
    mag_bands: &HashMap<String, BandData>,
    n_time_steps: usize,
) -> Option<(Gp2dResult, Gp2dThermalResult)> {
    let (gp, result) = fit_gp_2d(mag_bands)?;

    // Build time grid spanning the data
    let t_min = gp.t.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = gp.t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eval_times: Vec<f64> = (0..n_time_steps)
        .map(|i| t_min + (t_max - t_min) * i as f64 / (n_time_steps - 1).max(1) as f64)
        .collect();

    // Band wavelengths
    let band_wavs: Vec<(String, f64)> = result.bands.iter()
        .filter_map(|name| band_wavelength(name).map(|w| (name.clone(), w)))
        .collect();

    let thermal = extract_thermal_from_gp2d(&gp, &band_wavs, &eval_times);

    Some((result, thermal))
}

/// Make [`band_wavelength`] available to other modules.
pub fn get_band_wavelength(band: &str) -> Option<f64> {
    band_wavelength(band)
}

// ---------------------------------------------------------------------------
// GPU batch 2D GP
// ---------------------------------------------------------------------------

/// Fit 2D GP + thermal for many sources on GPU.
///
/// Returns (Gp2dResult, Gp2dThermalResult) per source, or None for sources
/// with insufficient data.
#[cfg(feature = "cuda")]
pub fn fit_gp_2d_batch_gpu(
    gpu: &crate::gpu::GpuContext,
    sources: &[HashMap<String, BandData>],
    n_time_steps: usize,
) -> Vec<Option<(Gp2dResult, Gp2dThermalResult)>> {
    use crate::gpu::Gp2dInput;

    if sources.is_empty() {
        return Vec::new();
    }

    // Prepare per-source input data
    struct SourcePrep {
        input: Gp2dInput,
        band_names: Vec<String>,
        band_log_wavs: Vec<f64>,
        band_wavs: Vec<(String, f64)>,
        t_min: f64,
        t_max: f64,
    }

    let mut valid_sources: Vec<(usize, SourcePrep)> = Vec::new();

    for (idx, mag_bands) in sources.iter().enumerate() {
        let mut times = Vec::new();
        let mut waves = Vec::new();
        let mut mags = Vec::new();
        let mut noise_var = Vec::new();
        let mut band_names = Vec::new();
        let mut band_log_wavs = Vec::new();
        let mut band_wavs = Vec::new();

        for (name, bd) in mag_bands {
            let wav = match band_wavelength(name) {
                Some(w) => w,
                None => continue,
            };
            if bd.times.len() < 3 { continue; }

            let log_w = wav.log10();
            if !band_names.contains(name) {
                band_names.push(name.clone());
                band_log_wavs.push(log_w);
                band_wavs.push((name.clone(), wav));
            }

            for i in 0..bd.times.len() {
                times.push(bd.times[i]);
                waves.push(log_w);
                mags.push(bd.values[i]);
                let e = bd.errors[i];
                noise_var.push(e * e);
            }
        }

        if times.len() < 5 || band_names.len() < 2 {
            continue;
        }

        let t_min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        valid_sources.push((idx, SourcePrep {
            input: Gp2dInput { times, waves, mags, noise_var },
            band_names,
            band_log_wavs,
            band_wavs,
            t_min,
            t_max,
        }));
    }

    if valid_sources.is_empty() {
        return vec![None; sources.len()];
    }

    // Build query grid: n_time_steps × n_unique_wavelengths
    // Use global union of all band wavelengths
    let mut all_wavs: Vec<f64> = Vec::new();
    for (_, prep) in &valid_sources {
        for &lw in &prep.band_log_wavs {
            if !all_wavs.iter().any(|&w| (w - lw).abs() < 1e-6) {
                all_wavs.push(lw);
            }
        }
    }
    all_wavs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_waves = all_wavs.len();

    // Time grid: use global time range
    let global_t_min = valid_sources.iter().map(|(_, p)| p.t_min).fold(f64::INFINITY, f64::min);
    let global_t_max = valid_sources.iter().map(|(_, p)| p.t_max).fold(f64::NEG_INFINITY, f64::max);
    let global_duration = (global_t_max - global_t_min).max(1.0);

    let eval_times: Vec<f64> = (0..n_time_steps)
        .map(|i| global_t_min + global_duration * i as f64 / (n_time_steps - 1).max(1) as f64)
        .collect();

    // Build flat query grid: time × wave
    let n_pred = n_time_steps * n_waves;
    let mut query_times = Vec::with_capacity(n_pred);
    let mut query_waves = Vec::with_capacity(n_pred);
    for &t in &eval_times {
        for &w in &all_wavs {
            query_times.push(t);
            query_waves.push(w);
        }
    }

    // Hyperparameter candidates
    let amp_candidates = [0.3, 1.0, 3.0, 10.0];
    let w_range = all_wavs.last().unwrap() - all_wavs.first().unwrap();
    let w_range = w_range.max(0.01);
    let lst_candidates = [global_duration / 4.0, global_duration / 8.0, global_duration / 16.0, global_duration / 32.0];
    let lsw_candidates = [w_range * 2.0, w_range, w_range / 2.0, w_range / 4.0];
    // 4 × 4 × 4 = 64 combos, fits in GP2D_MAX_HP

    let inputs: Vec<Gp2dInput> = valid_sources.iter().map(|(_, p)| {
        Gp2dInput {
            times: p.input.times.clone(),
            waves: p.input.waves.clone(),
            mags: p.input.mags.clone(),
            noise_var: p.input.noise_var.clone(),
        }
    }).collect();

    let gpu_results = match gpu.batch_gp_2d(
        &inputs, &query_times, &query_waves,
        &amp_candidates, &lst_candidates, &lsw_candidates,
        40, // max_subsample
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("GPU 2D GP batch failed: {}, falling back to CPU", e);
            return sources.iter()
                .map(|bands| fit_gp_2d_with_thermal(bands, n_time_steps))
                .collect();
        }
    };

    // Build results with thermal extraction
    let mut results: Vec<Option<(Gp2dResult, Gp2dThermalResult)>> = vec![None; sources.len()];

    for (gi, (orig_idx, prep)) in valid_sources.iter().enumerate() {
        let gout = &gpu_results[gi];
        if !gout.success {
            continue;
        }

        let gp2d_result = Gp2dResult {
            amp: gout.amp,
            ls_time: gout.ls_time,
            ls_wave: gout.ls_wave,
            train_rms: gout.train_rms,
            n_train: gout.n_train,
            n_bands: prep.band_names.len(),
            bands: prep.band_names.clone(),
            band_log_wavelengths: prep.band_log_wavs.clone(),
        };

        // Extract thermal from GPU predictions
        // pred_grid is [n_time_steps × n_waves] (time-major)
        let mut th_times = Vec::new();
        let mut th_log_temps = Vec::new();
        let mut th_chi2s = Vec::new();

        let band_wavs_ang: Vec<f64> = prep.band_wavs.iter().map(|(_, w)| *w).collect();
        let band_log_wavs: Vec<f64> = prep.band_wavs.iter().map(|(_, w)| w.log10()).collect();

        for ti in 0..n_time_steps {
            // Find the pred values at this time for this source's bands
            let mut mags_at_t = Vec::new();
            let mut stds_at_t = Vec::new();
            let mut wavs_at_t = Vec::new();

            for (bi, &blw) in band_log_wavs.iter().enumerate() {
                // Find index in all_wavs
                if let Some(wi) = all_wavs.iter().position(|&w| (w - blw).abs() < 1e-6) {
                    let pred_idx = ti * n_waves + wi;
                    let pred = gout.pred_grid[pred_idx];
                    let std = gout.std_grid[pred_idx];
                    if pred.is_finite() && std.is_finite() {
                        mags_at_t.push(pred);
                        stds_at_t.push(std);
                        wavs_at_t.push(band_wavs_ang[bi]);
                    }
                }
            }

            if wavs_at_t.len() >= 2 {
                if let Some((log_t, chi2, _)) = fit_bb_temperature(&wavs_at_t, &mags_at_t, &stds_at_t) {
                    th_times.push(eval_times[ti]);
                    th_log_temps.push(log_t);
                    th_chi2s.push(chi2);
                }
            }
        }

        let thermal = Gp2dThermalResult {
            times: th_times,
            log_temps: th_log_temps,
            chi2s: th_chi2s,
        };

        results[*orig_idx] = Some((gp2d_result, thermal));
    }

    results
}

//! Chromatic blackbody kilonova model with a prior trained on POSSIS models.
//!
//! The photosphere is a single blackbody whose luminosity and temperature evolve
//! as low-order polynomials in x = log10(t − t_exp), with t in days:
//!
//!   log10 L = log_l1 + a1·x + a2·x²        log10 T = log_t1 − β·x
//!
//! so `log_l1` and `log_t1` are the luminosity (erg/s) and temperature (K) one
//! day after the explosion. Each band's AB magnitude is the blackbody evaluated
//! at the band's effective wavelength plus a fixed offset c0 + c1·x, which
//! absorbs the departure of kilonova spectra from a blackbody (mostly an r-band
//! deficit). The offsets and a multivariate-normal prior on the five parameters
//! come from fits to 1,420 POSSIS Bu2019 models (Bulla 2019) in g, r and i over
//! 0.3 to 6 days.
//!
//! In magnitude space `log_l1` adds the same constant to every band, so it is
//! solved exactly at each trial point. The four shape parameters are fit by
//! Levenberg–Marquardt; a free explosion time is searched on a grid and then
//! refined jointly.
//!
//! `fit_kilonova` fits each light curve twice, once unconstrained and once held
//! to the prior, and reports the χ² increase from the prior (`prior_cost`). The
//! cost is small for kilonovae and approximates a log Bayes factor for a
//! kilonova against an arbitrary blackbody.

use std::collections::HashMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::common::BandData;

pub const KILONOVA_PARAM_NAMES: [&str; 5] = ["log_l1", "a1", "a2", "log_t1", "beta"];

/// Prior mean of (log_l1, a1, a2, log_t1, beta) over the POSSIS Bu2019 fits.
pub const KILONOVA_PRIOR_MEAN: [f64; 5] = [
    41.629614091444985,
    0.2808372156230851,
    -1.8793073325847935,
    3.770662925560441,
    0.38540599767451444,
];

/// Prior covariance matching `KILONOVA_PRIOR_MEAN`.
pub const KILONOVA_PRIOR_COV: [[f64; 5]; 5] = [
    [0.09341350435990478, -0.04272435872584099, -0.048376214009666026, 0.012745890395985476, 0.007388732897827619],
    [-0.04272435872584099, 0.2840884226143154, -0.12962772396549263, -0.014552316187450474, -0.03176777343702196],
    [-0.048376214009666026, -0.12962772396549263, 0.3803867498941516, 0.015340811932543465, 0.023590717404701207],
    [0.012745890395985476, -0.014552316187450474, 0.015340811932543465, 0.015672997350381596, 0.01482290743854647],
    [0.007388732897827619, -0.03176777343702196, 0.023590717404701207, 0.01482290743854647, 0.018418533349995304],
];

const SHAPE_LO: [f64; 4] = [-5.0, -8.0, 3.0, -1.5];
const SHAPE_HI: [f64; 4] = [5.0, 3.0, 4.7, 2.5];
const SHAPE_STARTS: [[f64; 4]; 2] = [
    [0.2808372156230851, -1.8793073325847935, 3.770662925560441, 0.38540599767451444],
    [0.0, -1.5, 3.75, 0.4],
];
const N_TEXP_GRID: usize = 10;
/// Latest allowed explosion time, in days before the first observation.
const TEXP_MIN_LEAD: f64 = 0.05;

const H_PLANCK: f64 = 6.62607015e-27;
const C_LIGHT: f64 = 2.99792458e10;
const SIGMA_SB: f64 = 5.670374e-5;
const PC10_CM: f64 = 3.0857e19;
const HC_OVER_K_ANG: f64 = 1.4387769e8;

// ---------------------------------------------------------------------------
// Bands
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct BandInfo {
    lambda: f64,
    c0: f64,
    c1: f64,
}

/// Per-band offsets from a blackbody, c0 + c1·log10(t), for the POSSIS grid bands.
fn sed_offset(canonical: char) -> (f64, f64) {
    match canonical {
        'g' => (0.054718108384186476, -0.20783429761199412),
        'r' => (0.0564331058022488, 0.21325317910577474),
        'i' => (-0.0777315084946671, 0.15201701017236538),
        'z' => (-0.1314659912859724, 0.10817219745392252),
        'y' => (-0.07760097418505162, -0.13455795294380155),
        'J' => (0.023550516501264218, 0.00023857388670880928),
        'H' => (0.03547043573416041, 0.03658645520399672),
        'K' => (0.11881926136352475, -0.1965302990556245),
        _ => (0.0, 0.0),
    }
}

/// Effective wavelength (Å) and offsets for a band name, or None if unsupported.
/// u band is excluded: its offset scatters by 1.1 mag across the grid.
fn band_info(band: &str) -> Option<BandInfo> {
    let (lambda, canonical) = match band {
        "ztfg" => (4746.0, 'g'),
        "ztfr" => (6366.0, 'r'),
        "ztfi" => (7829.0, 'i'),
        "lsstg" => (4827.0, 'g'),
        "lsstr" => (6223.0, 'r'),
        "lssti" => (7546.0, 'i'),
        "lsstz" => (8691.0, 'z'),
        "lssty" => (9712.0, 'y'),
        "ps1::g" => (4810.0, 'g'),
        "ps1::r" => (6170.0, 'r'),
        "ps1::i" => (7520.0, 'i'),
        "ps1::z" => (8660.0, 'z'),
        "ps1::y" => (9620.0, 'y'),
        "g" | "sdssg" => (4686.0, 'g'),
        "r" | "sdssr" => (6166.0, 'r'),
        "i" | "sdssi" => (7480.0, 'i'),
        "z" | "sdssz" => (8932.0, 'z'),
        "y" => (9620.0, 'y'),
        "J" | "2massj" => (12350.0, 'J'),
        "H" | "2massh" => (16620.0, 'H'),
        "K" | "Ks" | "2massks" | "2massk" => (21590.0, 'K'),
        _ => return None,
    };
    let (c0, c1) = sed_offset(canonical);
    Some(BandInfo { lambda, c0, c1 })
}

/// True if `fit_kilonova` uses this band name.
pub fn kilonova_band_supported(band: &str) -> bool {
    band_info(band).is_some()
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// log10 of the frequency-dependent constants in F_ν = L·B_ν(T) / (4σT⁴D²) at D = 10 pc.
#[inline]
fn log_fnu_const(lambda_ang: f64) -> f64 {
    let nu = C_LIGHT / (lambda_ang * 1e-8);
    (2.0 * H_PLANCK * nu.powi(3) / (C_LIGHT * C_LIGHT)).log10()
        - (4.0 * SIGMA_SB).log10()
        - 2.0 * PC10_CM.log10()
}

/// Absolute AB magnitude of the blackbody part with log_l = 0; add −2.5·log_l for luminosity.
#[inline]
fn bb_mag_unit_lum(lambda_ang: f64, k_lambda: f64, log_t: f64) -> f64 {
    let x = (HC_OVER_K_ANG / (lambda_ang * 10f64.powf(log_t))).clamp(1e-8, 700.0);
    // log10 B_ν minus its constant part; ln(1 − e^−x) stays finite for large x.
    let log_planck = -(x + (-(-x).exp()).ln_1p()) / std::f64::consts::LN_10;
    -2.5 * (k_lambda - 4.0 * log_t + log_planck) - 48.6
}

#[inline]
fn mag_unit_lum(info: &BandInfo, k_lambda: f64, shape: &[f64], t_since: f64) -> f64 {
    let x = t_since.max(1e-3).log10();
    let log_l_rel = shape[0] * x + shape[1] * x * x;
    let log_t = shape[2] - shape[3] * x;
    bb_mag_unit_lum(info.lambda, k_lambda, log_t) - 2.5 * log_l_rel + info.c0 + info.c1 * x
}

/// Absolute AB magnitudes of the model at `times_since_explosion` (days) in `band`.
///
/// `params` is (log_l1, a1, a2, log_t1, beta). Returns None for an unsupported band.
pub fn kilonova_mags(params: &[f64], times_since_explosion: &[f64], band: &str) -> Option<Vec<f64>> {
    let info = band_info(band)?;
    let k = log_fnu_const(info.lambda);
    Some(
        times_since_explosion
            .iter()
            .map(|&t| mag_unit_lum(&info, k, &params[1..5], t) - 2.5 * params[0])
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Options and results
// ---------------------------------------------------------------------------

/// Settings for `fit_kilonova`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KilonovaOptions {
    /// Fixed explosion (merger) time in the data's time frame, e.g. from a GW trigger.
    /// None leaves it free between `t_exp_window` days and 0.05 days before the first point.
    pub t_exp: Option<f64>,
    pub t_exp_window: f64,
    /// Distance modulus. When set, magnitudes become absolute and log_l1 joins the prior;
    /// otherwise log_l1 absorbs the distance and stays free.
    pub distance_modulus: Option<f64>,
    /// Systematic error (mag) added in quadrature to every point.
    pub sys_err: f64,
    /// Factor applied to the prior standard deviations.
    pub prior_inflate: f64,
}

impl Default for KilonovaOptions {
    fn default() -> Self {
        Self { t_exp: None, t_exp_window: 3.0, distance_modulus: None, sys_err: 0.1, prior_inflate: 1.5 }
    }
}

/// One fit of the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KilonovaFit {
    /// (log_l1, a1, a2, log_t1, beta). Without a distance modulus, log_l1 is
    /// log10 L − 0.4·DM, the luminosity a source at 10 pc would need.
    pub params: Vec<f64>,
    pub t_exp: f64,
    /// χ² of the data alone.
    pub chi2: f64,
    /// χ² plus the prior's Mahalanobis distance squared (equal to chi2 for the free fit).
    pub objective: f64,
    /// RMS magnitude residual.
    pub rms: f64,
}

/// Result of `fit_kilonova`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KilonovaResult {
    pub param_names: Vec<String>,
    /// Fit held to the kilonova prior.
    pub kn_prior: KilonovaFit,
    /// Fit with the same bounds and no prior.
    pub free: KilonovaFit,
    /// kn_prior.objective − free.chi2; small for kilonova-like light curves.
    pub prior_cost: f64,
    /// Temperature (K) one day after t_exp in the prior-held fit.
    pub temperature_1d: f64,
    /// Time of peak bolometric luminosity after t_exp (days) in the prior-held fit, if it peaks.
    pub t_peak_bol: Option<f64>,
    pub n_obs: usize,
    pub bands_used: Vec<String>,
    pub bands_skipped: Vec<String>,
    pub t_exp_fixed: bool,
}

// ---------------------------------------------------------------------------
// Fitting
// ---------------------------------------------------------------------------

struct Obs {
    t: f64,
    mag: f64,
    w: f64,
    info: BandInfo,
    k: f64,
}

struct Prior {
    idx: Vec<usize>,
    mean: Vec<f64>,
    chol: Vec<Vec<f64>>,
}

impl Prior {
    fn new(idx: Vec<usize>, inflate: f64) -> Option<Self> {
        let n = idx.len();
        let mut cov = vec![vec![0.0; n]; n];
        for (a, &i) in idx.iter().enumerate() {
            for (b, &j) in idx.iter().enumerate() {
                cov[a][b] = KILONOVA_PRIOR_COV[i][j] * inflate * inflate;
            }
        }
        let chol = cholesky(&cov)?;
        let mean = idx.iter().map(|&i| KILONOVA_PRIOR_MEAN[i]).collect();
        Some(Self { idx, mean, chol })
    }

    /// Whitened residuals L⁻¹(θ_idx − μ).
    fn whiten(&self, theta: &[f64]) -> Vec<f64> {
        let d: Vec<f64> = self.idx.iter().zip(&self.mean).map(|(&i, m)| theta[i] - m).collect();
        forward_sub(&self.chol, &d)
    }
}

/// Residual vector with log_l1 solved exactly, plus that log_l1.
///
/// Every residual is affine in log_l1 (data: −2.5·w; prior: a fixed whitened
/// column), so the optimum is a one-dimensional linear least-squares solution.
fn profiled_residuals(obs: &[Obs], prior: Option<&Prior>, shape: &[f64], t_exp: f64) -> (Vec<f64>, f64) {
    let mut r: Vec<f64> = Vec::with_capacity(obs.len() + 5);
    let mut g: Vec<f64> = Vec::with_capacity(obs.len() + 5);
    for o in obs {
        r.push((mag_unit_lum(&o.info, o.k, shape, o.t - t_exp) - o.mag) * o.w);
        g.push(-2.5 * o.w);
    }
    if let Some(p) = prior {
        let theta0 = [0.0, shape[0], shape[1], shape[2], shape[3]];
        let z0 = p.whiten(&theta0);
        let unit = [1.0, 0.0, 0.0, 0.0, 0.0];
        let dz: Vec<f64> = if p.idx.contains(&0) {
            let base = p.whiten(&[0.0; 5]);
            p.whiten(&unit).iter().zip(&base).map(|(a, b)| a - b).collect()
        } else {
            vec![0.0; z0.len()]
        };
        r.extend(z0);
        g.extend(dz);
    }
    let gg: f64 = g.iter().map(|v| v * v).sum();
    let gr: f64 = g.iter().zip(&r).map(|(a, b)| a * b).sum();
    let log_l1 = if gg > 0.0 { -gr / gg } else { 0.0 };
    for (ri, gi) in r.iter_mut().zip(&g) {
        *ri += log_l1 * gi;
    }
    (r, log_l1)
}

/// Bounded Levenberg–Marquardt with a forward-difference Jacobian. Returns (x, Σr²).
fn levenberg_marquardt<F: Fn(&[f64]) -> Vec<f64>>(f: F, x0: &[f64], lo: &[f64], hi: &[f64], max_iter: usize) -> (Vec<f64>, f64) {
    let n = x0.len();
    let clamp = |x: &mut Vec<f64>| {
        for j in 0..n {
            x[j] = x[j].clamp(lo[j], hi[j]);
        }
    };
    let mut x = x0.to_vec();
    clamp(&mut x);
    let mut r = f(&x);
    let mut cost: f64 = r.iter().map(|v| v * v).sum();
    let mut lambda = 1e-3;
    for _ in 0..max_iter {
        let m = r.len();
        let mut jac = vec![vec![0.0; n]; m];
        for j in 0..n {
            let mut h = 1e-6 * x[j].abs().max(1.0);
            if x[j] + h > hi[j] {
                h = -h;
            }
            let mut xp = x.clone();
            xp[j] += h;
            let rp = f(&xp);
            for i in 0..m {
                jac[i][j] = (rp[i] - r[i]) / h;
            }
        }
        let mut a = vec![vec![0.0; n]; n];
        let mut g = vec![0.0; n];
        for i in 0..m {
            for p in 0..n {
                g[p] += jac[i][p] * r[i];
                for q in 0..=p {
                    a[p][q] += jac[i][p] * jac[i][q];
                }
            }
        }
        for p in 0..n {
            for q in 0..p {
                a[q][p] = a[p][q];
            }
        }
        let mut improved = false;
        while lambda < 1e10 {
            let mut damped = a.clone();
            for p in 0..n {
                damped[p][p] += lambda * a[p][p].max(1e-12);
            }
            let step = match solve_spd(&damped, &g.iter().map(|v| -v).collect::<Vec<_>>()) {
                Some(s) => s,
                None => {
                    lambda *= 4.0;
                    continue;
                }
            };
            let mut xn: Vec<f64> = x.iter().zip(&step).map(|(a, b)| a + b).collect();
            clamp(&mut xn);
            let rn = f(&xn);
            let cn: f64 = rn.iter().map(|v| v * v).sum();
            if cn.is_finite() && cn < cost {
                let dx: f64 = xn.iter().zip(&x).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                let rel = (cost - cn) / (1.0 + cost);
                x = xn;
                r = rn;
                cost = cn;
                lambda = (lambda / 3.0).max(1e-12);
                improved = true;
                if rel < 1e-10 && dx < 1e-8 {
                    return (x, cost);
                }
                break;
            }
            lambda *= 4.0;
        }
        if !improved {
            break;
        }
    }
    (x, cost)
}

fn fit_once(obs: &[Obs], prior: Option<&Prior>, t_exp_fixed: Option<f64>, t_exp_range: (f64, f64)) -> KilonovaFit {
    let mut best: Option<(Vec<f64>, f64, f64)> = None; // (shape, t_exp, cost)
    let texp_grid: Vec<f64> = match t_exp_fixed {
        Some(t) => vec![t],
        None => (0..N_TEXP_GRID)
            .map(|k| t_exp_range.0 + (t_exp_range.1 - t_exp_range.0) * k as f64 / (N_TEXP_GRID - 1) as f64)
            .collect(),
    };
    for &te in &texp_grid {
        for start in SHAPE_STARTS.iter() {
            let (shape, cost) = levenberg_marquardt(
                |s| profiled_residuals(obs, prior, s, te).0,
                start,
                &SHAPE_LO,
                &SHAPE_HI,
                60,
            );
            if best.as_ref().map_or(true, |b| cost < b.2) {
                best = Some((shape, te, cost));
            }
        }
    }
    let (mut shape, mut t_exp, mut cost) = best.expect("at least one start");
    if t_exp_fixed.is_none() {
        let x0 = [shape[0], shape[1], shape[2], shape[3], t_exp];
        let lo = [SHAPE_LO[0], SHAPE_LO[1], SHAPE_LO[2], SHAPE_LO[3], t_exp_range.0];
        let hi = [SHAPE_HI[0], SHAPE_HI[1], SHAPE_HI[2], SHAPE_HI[3], t_exp_range.1];
        let (x, c) = levenberg_marquardt(|p| profiled_residuals(obs, prior, &p[..4], p[4]).0, &x0, &lo, &hi, 100);
        if c < cost {
            shape = x[..4].to_vec();
            t_exp = x[4];
            cost = c;
        }
    }
    let (r, log_l1) = profiled_residuals(obs, prior, &shape, t_exp);
    let chi2: f64 = r[..obs.len()].iter().map(|v| v * v).sum();
    let rms = (obs
        .iter()
        .zip(&r)
        .map(|(o, ri)| (ri / o.w).powi(2))
        .sum::<f64>()
        / obs.len() as f64)
        .sqrt();
    KilonovaFit {
        params: vec![log_l1, shape[0], shape[1], shape[2], shape[3]],
        t_exp,
        chi2,
        objective: cost,
        rms,
    }
}

/// Fit the chromatic blackbody kilonova model to magnitude photometry.
///
/// `bands` holds AB magnitudes per band (as from `build_mag_bands`); unsupported
/// band names are skipped and listed in `bands_skipped`. Returns None with fewer
/// than three usable points or a non-positive explosion-time window.
pub fn fit_kilonova(bands: &HashMap<String, BandData>, opts: &KilonovaOptions) -> Option<KilonovaResult> {
    let mut names: Vec<&String> = bands.keys().collect();
    names.sort();
    let dm = opts.distance_modulus.unwrap_or(0.0);
    let mut obs = Vec::new();
    let (mut used, mut skipped) = (Vec::new(), Vec::new());
    for name in names {
        let bd = &bands[name];
        let Some(info) = band_info(name) else {
            skipped.push(name.clone());
            continue;
        };
        let k = log_fnu_const(info.lambda);
        let mut any = false;
        for i in 0..bd.times.len() {
            let (t, m, e) = (bd.times[i], bd.values[i], bd.errors[i]);
            if !(t.is_finite() && m.is_finite() && e.is_finite() && e >= 0.0) {
                continue;
            }
            let err = (e * e + opts.sys_err * opts.sys_err).sqrt();
            if err <= 0.0 {
                continue;
            }
            obs.push(Obs { t, mag: m - dm, w: 1.0 / err, info, k });
            any = true;
        }
        if any {
            used.push(name.clone());
        }
    }
    if obs.len() < 3 {
        return None;
    }
    let t_first = obs.iter().map(|o| o.t).fold(f64::INFINITY, f64::min);
    let t_exp_range = (t_first - opts.t_exp_window, t_first - TEXP_MIN_LEAD);
    if opts.t_exp.is_none() && !(t_exp_range.1 > t_exp_range.0) {
        return None;
    }
    if let Some(te) = opts.t_exp {
        if !(te < t_first) {
            return None;
        }
    }
    let idx = if opts.distance_modulus.is_some() { vec![0, 1, 2, 3, 4] } else { vec![1, 2, 3, 4] };
    let prior = Prior::new(idx, opts.prior_inflate)?;

    let free = fit_once(&obs, None, opts.t_exp, t_exp_range);
    let kn_prior = fit_once(&obs, Some(&prior), opts.t_exp, t_exp_range);
    let p = &kn_prior.params;
    let t_peak_bol = (p[2] < 0.0).then(|| 10f64.powf(-p[1] / (2.0 * p[2])));
    Some(KilonovaResult {
        param_names: KILONOVA_PARAM_NAMES.iter().map(|s| s.to_string()).collect(),
        prior_cost: kn_prior.objective - free.chi2,
        temperature_1d: 10f64.powf(p[3]),
        t_peak_bol,
        n_obs: obs.len(),
        bands_used: used,
        bands_skipped: skipped,
        t_exp_fixed: opts.t_exp.is_some(),
        kn_prior,
        free,
    })
}

/// `fit_kilonova` over many sources in parallel.
pub fn fit_kilonova_batch(sources: &[HashMap<String, BandData>], opts: &KilonovaOptions) -> Vec<Option<KilonovaResult>> {
    sources.par_iter().map(|s| fit_kilonova(s, opts)).collect()
}

// ---------------------------------------------------------------------------
// Small dense linear algebra
// ---------------------------------------------------------------------------

fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if !(s > 0.0) {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

fn forward_sub(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = (b[i] - (0..i).map(|k| l[i][k] * y[k]).sum::<f64>()) / l[i][i];
    }
    y
}

fn solve_spd(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let l = cholesky(a)?;
    let y = forward_sub(&l, b);
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (y[i] - (i + 1..n).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
    }
    Some(x)
}

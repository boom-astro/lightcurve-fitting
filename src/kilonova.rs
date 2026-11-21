//! Chromatic blackbody kilonova model with a prior trained on kilonova model grids.
//!
//! The photosphere is a single blackbody whose luminosity and temperature evolve
//! as low-order polynomials in x = log10(t − t_exp), with t in days:
//!
//!   log10 L = log_l1 + a1·x + a2·x²        log10 T = log_t1 − β·x
//!
//! so `log_l1` and `log_t1` are the luminosity (erg/s) and temperature (K) one
//! day after the explosion. Each band's AB magnitude is the blackbody integrated
//! through the band's transmission curve plus a fixed offset c0 + c1·x, which
//! absorbs the departure of kilonova spectra from a blackbody. The bandpass
//! tables, the offsets and a Gaussian prior on the five parameters are generated
//! by `scripts/train_kilonova_prior.py`; the built-in ones come from POSSIS Bu2019
//! models (Bulla 2019). A prior trained on other grids, with its own offsets and
//! any number of mixture components, can be passed in `KilonovaOptions::prior`.
//!
//! In magnitude space `log_l1` adds the same constant to every band, so it is
//! solved exactly at each trial point. The four shape parameters are fit by
//! Levenberg–Marquardt; a free explosion time is searched on a grid and then
//! refined jointly.
//!
//! `fit_kilonova` fits each light curve once unconstrained and once held to each
//! prior component, and reports the χ² increase from the prior (`prior_cost`).
//! The cost is small for kilonovae and approximates a log Bayes factor for a
//! kilonova against an arbitrary blackbody.

use std::collections::HashMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::common::BandData;
use crate::kilonova_tables::{
    BANDPASS_TABLES, LOGT_MIN, LOGT_STEP, N_LOGT, PRIOR_COVS, PRIOR_MEANS, PRIOR_WEIGHTS, SED_OFFSETS,
};

pub const KILONOVA_PARAM_NAMES: [&str; 5] = ["log_l1", "a1", "a2", "log_t1", "beta"];

const SHAPE_LO: [f64; 4] = [-5.0, -8.0, 3.0, -1.5];
const SHAPE_HI: [f64; 4] = [5.0, 3.0, 4.7, 2.5];
const N_TEXP_GRID: usize = 10;
/// Latest allowed explosion time, in days before the first observation.
const TEXP_MIN_LEAD: f64 = 0.05;

// ---------------------------------------------------------------------------
// Bands
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct BandInfo {
    table: &'static [f64; N_LOGT],
    c0: f64,
    c1: f64,
}

/// Bandpass (sncosmo name) and grid band for a band name, or None if unsupported.
///
/// Bare names (`g`, `r`, `i`, `z`, `y`, `J`, `H`, `K`) mean the filters the model
/// grids were synthesised in: SDSS griz, Swope Y, J and H, and CSP K. u band is
/// excluded because kilonova spectra depart from a blackbody there by more than
/// a magnitude, differently in every model.
fn band_lookup(band: &str) -> Option<(&'static str, char)> {
    Some(match band {
        "ztfg" => ("ztfg", 'g'),
        "ztfr" => ("ztfr", 'r'),
        "ztfi" => ("ztfi", 'i'),
        "lsstg" => ("lsstg", 'g'),
        "lsstr" => ("lsstr", 'r'),
        "lssti" => ("lssti", 'i'),
        "lsstz" => ("lsstz", 'z'),
        "lssty" => ("lssty", 'y'),
        "ps1::g" => ("ps1::g", 'g'),
        "ps1::r" => ("ps1::r", 'r'),
        "ps1::i" => ("ps1::i", 'i'),
        "ps1::z" => ("ps1::z", 'z'),
        "ps1::y" => ("ps1::y", 'y'),
        "g" | "sdssg" => ("sdssg", 'g'),
        "r" | "sdssr" => ("sdssr", 'r'),
        "i" | "sdssi" => ("sdssi", 'i'),
        "z" | "sdssz" => ("sdssz", 'z'),
        "y" | "swope2::y" => ("swope2::y", 'y'),
        "J" | "swope2::j" => ("swope2::j", 'J'),
        "H" | "swope2::h" => ("swope2::h", 'H'),
        "K" | "Ks" | "cspk" => ("cspk", 'K'),
        "2massj" => ("2massj", 'J'),
        "2massh" => ("2massh", 'H'),
        "2massks" | "2massk" => ("2massks", 'K'),
        _ => return None,
    })
}

fn band_info(band: &str, offsets: Option<&HashMap<String, (f64, f64)>>) -> Option<BandInfo> {
    let (bandpass, canonical) = band_lookup(band)?;
    let table = &BANDPASS_TABLES.iter().find(|(n, _)| *n == bandpass)?.1;
    let (c0, c1) = match offsets {
        Some(map) => *map.get(&canonical.to_string())?,
        None => {
            let (_, c0, c1) = *SED_OFFSETS.iter().find(|(c, _, _)| *c == canonical)?;
            (c0, c1)
        }
    };
    Some(BandInfo { table, c0, c1 })
}

/// True if `fit_kilonova` uses this band name.
pub fn kilonova_band_supported(band: &str) -> bool {
    band_info(band, None).is_some()
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Bandpass-integrated AB magnitude of a blackbody with L = 1 erg/s at 10 pc.
/// Linear interpolation in log10 T, held constant beyond the table.
#[inline]
fn unit_lum_mag(table: &[f64; N_LOGT], log_t: f64) -> f64 {
    let u = ((log_t - LOGT_MIN) / LOGT_STEP).clamp(0.0, (N_LOGT - 1) as f64);
    let k = (u as usize).min(N_LOGT - 2);
    let f = u - k as f64;
    table[k] * (1.0 - f) + table[k + 1] * f
}

/// Model magnitude with log_l1 = 0; the full magnitude subtracts 2.5·log_l1.
#[inline]
fn mag_unit_lum(info: &BandInfo, shape: &[f64], t_since: f64) -> f64 {
    let x = t_since.max(1e-3).log10();
    let log_l_rel = shape[0] * x + shape[1] * x * x;
    let log_t = shape[2] - shape[3] * x;
    unit_lum_mag(info.table, log_t) - 2.5 * log_l_rel + info.c0 + info.c1 * x
}

/// Absolute AB magnitudes of the model at `times_since_explosion` (days) in `band`.
///
/// `params` is (log_l1, a1, a2, log_t1, beta). Uses the built-in offsets; returns None
/// for an unsupported band.
pub fn kilonova_mags(params: &[f64], times_since_explosion: &[f64], band: &str) -> Option<Vec<f64>> {
    let info = band_info(band, None)?;
    Some(
        times_since_explosion
            .iter()
            .map(|&t| mag_unit_lum(&info, &params[1..5], t) - 2.5 * params[0])
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Prior
// ---------------------------------------------------------------------------

/// Gaussian-mixture prior on (log_l1, a1, a2, log_t1, beta), with the per-band
/// offsets it was trained with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KilonovaPrior {
    pub weights: Vec<f64>,
    pub means: Vec<Vec<f64>>,
    pub covs: Vec<Vec<Vec<f64>>>,
    /// (c0, c1) per grid band ("g", "r", "i", "z", "y", "J", "H", "K"); None uses the
    /// built-in offsets. A band missing from the map is skipped.
    #[serde(default)]
    pub offsets: Option<HashMap<String, (f64, f64)>>,
}

impl Default for KilonovaPrior {
    /// The prior trained by `scripts/train_kilonova_prior.py`.
    fn default() -> Self {
        Self {
            weights: PRIOR_WEIGHTS.to_vec(),
            means: PRIOR_MEANS.iter().map(|m| m.to_vec()).collect(),
            covs: PRIOR_COVS.iter().map(|c| c.iter().map(|r| r.to_vec()).collect()).collect(),
            offsets: None,
        }
    }
}

impl KilonovaPrior {
    /// Mean of the highest-weight component.
    pub fn dominant_mean(&self) -> Vec<f64> {
        let k = (0..self.weights.len())
            .max_by(|&a, &b| self.weights[a].partial_cmp(&self.weights[b]).unwrap())
            .unwrap_or(0);
        self.means[k].clone()
    }
}

/// One mixture component restricted to `idx`, ready to whiten parameter vectors.
struct PriorComponent {
    idx: Vec<usize>,
    mean: Vec<f64>,
    chol: Vec<Vec<f64>>,
    /// −2 ln w + ln det Σ, so that comparing components compares −2 ln of their densities.
    log_norm: f64,
}

impl PriorComponent {
    fn new(prior: &KilonovaPrior, k: usize, idx: &[usize], inflate: f64) -> Option<Self> {
        let (w, mean, cov) = (prior.weights.get(k)?, prior.means.get(k)?, prior.covs.get(k)?);
        if mean.len() != 5 || cov.len() != 5 || cov.iter().any(|r| r.len() != 5) || !(*w > 0.0) {
            return None;
        }
        let n = idx.len();
        let mut sub = vec![vec![0.0; n]; n];
        for (a, &i) in idx.iter().enumerate() {
            for (b, &j) in idx.iter().enumerate() {
                sub[a][b] = cov[i][j] * inflate * inflate;
            }
        }
        let chol = cholesky(&sub)?;
        let log_det: f64 = 2.0 * (0..n).map(|i| chol[i][i].ln()).sum::<f64>();
        Some(Self {
            idx: idx.to_vec(),
            mean: idx.iter().map(|&i| mean[i]).collect(),
            chol,
            log_norm: -2.0 * w.ln() + log_det,
        })
    }

    /// Whitened residuals L⁻¹(θ_idx − μ).
    fn whiten(&self, theta: &[f64]) -> Vec<f64> {
        let d: Vec<f64> = self.idx.iter().zip(&self.mean).map(|(&i, m)| theta[i] - m).collect();
        forward_sub(&self.chol, &d)
    }
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
    /// Time of the last non-detection before the first point. A free explosion time is
    /// kept after it (and after `t_exp_window` days before the first point).
    pub t_last_nondet: Option<f64>,
    /// Distance modulus. When set, magnitudes become absolute and log_l1 joins the prior;
    /// otherwise log_l1 absorbs the distance and stays free.
    pub distance_modulus: Option<f64>,
    /// Systematic error (mag) added in quadrature to every point.
    pub sys_err: f64,
    /// Factor applied to the prior standard deviations.
    pub prior_inflate: f64,
    /// Prior to use in place of the built-in one, e.g. retrained on other model grids.
    pub prior: Option<KilonovaPrior>,
}

impl Default for KilonovaOptions {
    fn default() -> Self {
        Self {
            t_exp: None,
            t_exp_window: 3.0,
            t_last_nondet: None,
            distance_modulus: None,
            sys_err: 0.1,
            prior_inflate: 1.5,
            prior: None,
        }
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
    /// χ² plus the prior penalty (equal to chi2 for the free fit).
    pub objective: f64,
    /// RMS magnitude residual.
    pub rms: f64,
}

/// Result of `fit_kilonova`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KilonovaResult {
    pub param_names: Vec<String>,
    /// Fit held to the prior component that gives the lowest objective.
    pub kn_prior: KilonovaFit,
    /// Index of that prior component.
    pub prior_component: usize,
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
}

/// Residual vector with log_l1 solved exactly, plus that log_l1.
///
/// Every residual is affine in log_l1 (data: −2.5·w; prior: a fixed whitened
/// column), so the optimum is a one-dimensional linear least-squares solution.
fn profiled_residuals(obs: &[Obs], prior: Option<&PriorComponent>, shape: &[f64], t_exp: f64) -> (Vec<f64>, f64) {
    let mut r: Vec<f64> = Vec::with_capacity(obs.len() + 5);
    let mut g: Vec<f64> = Vec::with_capacity(obs.len() + 5);
    for o in obs {
        r.push((mag_unit_lum(&o.info, shape, o.t - t_exp) - o.mag) * o.w);
        g.push(-2.5 * o.w);
    }
    if let Some(p) = prior {
        let z0 = p.whiten(&[0.0, shape[0], shape[1], shape[2], shape[3]]);
        let dz: Vec<f64> = if p.idx.contains(&0) {
            let base = p.whiten(&[0.0; 5]);
            p.whiten(&[1.0, 0.0, 0.0, 0.0, 0.0]).iter().zip(&base).map(|(a, b)| a - b).collect()
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
            let Some(step) = solve_spd(&damped, &g.iter().map(|v| -v).collect::<Vec<_>>()) else {
                lambda *= 4.0;
                continue;
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

fn fit_once(
    obs: &[Obs],
    prior: Option<&PriorComponent>,
    starts: &[[f64; 4]],
    t_exp_fixed: Option<f64>,
    t_exp_range: (f64, f64),
) -> KilonovaFit {
    let mut best: Option<(Vec<f64>, f64, f64)> = None; // (shape, t_exp, cost)
    let texp_grid: Vec<f64> = match t_exp_fixed {
        Some(t) => vec![t],
        None => (0..N_TEXP_GRID)
            .map(|k| t_exp_range.0 + (t_exp_range.1 - t_exp_range.0) * k as f64 / (N_TEXP_GRID - 1) as f64)
            .collect(),
    };
    for &te in &texp_grid {
        for start in starts {
            let (shape, cost) =
                levenberg_marquardt(|s| profiled_residuals(obs, prior, s, te).0, start, &SHAPE_LO, &SHAPE_HI, 60);
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
    let rms = (obs.iter().zip(&r).map(|(o, ri)| (ri / o.w).powi(2)).sum::<f64>() / obs.len() as f64).sqrt();
    KilonovaFit { params: vec![log_l1, shape[0], shape[1], shape[2], shape[3]], t_exp, chi2, objective: cost, rms }
}

/// Fit the chromatic blackbody kilonova model to magnitude photometry.
///
/// `bands` holds AB magnitudes per band (as from `build_mag_bands`); unsupported
/// band names are skipped and listed in `bands_skipped`. Returns None with fewer
/// than three usable points, a non-positive explosion-time window, or an invalid
/// prior (mismatched lengths or a covariance that is not positive definite).
pub fn fit_kilonova(bands: &HashMap<String, BandData>, opts: &KilonovaOptions) -> Option<KilonovaResult> {
    let default_prior;
    let prior = match &opts.prior {
        Some(p) => p,
        None => {
            default_prior = KilonovaPrior::default();
            &default_prior
        }
    };
    let mut names: Vec<&String> = bands.keys().collect();
    names.sort();
    let dm = opts.distance_modulus.unwrap_or(0.0);
    let mut obs = Vec::new();
    let (mut used, mut skipped) = (Vec::new(), Vec::new());
    for name in names {
        let bd = &bands[name];
        let Some(info) = band_info(name, prior.offsets.as_ref()) else {
            skipped.push(name.clone());
            continue;
        };
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
            obs.push(Obs { t, mag: m - dm, w: 1.0 / err, info });
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
    let mut t_lo = t_first - opts.t_exp_window;
    if let Some(tn) = opts.t_last_nondet.filter(|tn| tn.is_finite() && *tn < t_first) {
        t_lo = t_lo.max(tn);
    }
    // Keep at least half the window when a non-detection sits within TEXP_MIN_LEAD of the first point.
    let t_exp_range = (t_lo, t_first - TEXP_MIN_LEAD.min(0.5 * (t_first - t_lo)));
    if opts.t_exp.is_none() && !(t_exp_range.1 > t_exp_range.0) {
        return None;
    }
    if let Some(te) = opts.t_exp {
        if !(te < t_first) {
            return None;
        }
    }

    if prior.weights.is_empty() || prior.means.len() != prior.weights.len() || prior.covs.len() != prior.weights.len() {
        return None;
    }
    let idx: &[usize] = if opts.distance_modulus.is_some() { &[0, 1, 2, 3, 4] } else { &[1, 2, 3, 4] };
    let components: Vec<PriorComponent> = (0..prior.weights.len())
        .map(|k| PriorComponent::new(prior, k, idx, opts.prior_inflate))
        .collect::<Option<_>>()?;
    let norm_min = components.iter().map(|c| c.log_norm).fold(f64::INFINITY, f64::min);

    let mut starts: Vec<[f64; 4]> = vec![[0.0, -1.5, 3.75, 0.4]];
    let dom = prior.dominant_mean();
    starts.push([dom[1], dom[2], dom[3], dom[4]]);
    let free = fit_once(&obs, None, &starts, opts.t_exp, t_exp_range);

    let mut best: Option<(KilonovaFit, usize)> = None;
    for (k, comp) in components.iter().enumerate() {
        let m = &prior.means[k];
        let comp_starts = [[m[1], m[2], m[3], m[4]], starts[0]];
        let mut fit = fit_once(&obs, Some(comp), &comp_starts, opts.t_exp, t_exp_range);
        fit.objective += comp.log_norm - norm_min;
        if best.as_ref().map_or(true, |(b, _)| fit.objective < b.objective) {
            best = Some((fit, k));
        }
    }
    let (kn_prior, prior_component) = best?;
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
        prior_component,
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

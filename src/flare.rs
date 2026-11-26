//! Stellar flare fitting with the Davenport et al. (2014) empirical template.
//!
//! A single flare is
//!
//!   F(t) = A · T((t − t_peak) / t_half)
//!
//! where T is the GJ 1243 median flare shape: a quartic rise on −1 ≤ x ≤ 0 and
//! a double-exponential decay for x > 0,
//!
//!   T(x) = 1 + 1.941x − 0.175x² − 2.246x³ − 1.125x⁴      (−1 ≤ x ≤ 0)
//!   T(x) = 0.6890·e^(−1.600x) + 0.3030·e^(−0.2783x)        (x > 0)
//!
//! All template coefficients are fixed, so each flare has three free
//! parameters: peak time, FWHM (t_half) and peak amplitude. Complex events are
//! modelled as a sum of templates, with the number of components chosen by BIC.
//!
//! Amplitudes and the baseline enter linearly, so they are solved by weighted
//! least squares at every trial (t_peak, t_half) (variable projection). That
//! leaves a 2K-dimensional nonlinear search for K flares, done by a grid seed
//! followed by Nelder-Mead.
//!
//! Finite exposures are handled exactly: the template has a closed-form
//! antiderivative, so the model for each point is the template averaged over
//! its exposure window rather than sampled at its midpoint.

use argmin::core::{CostFunction, Error as ArgminError, Executor};
use argmin::solver::neldermead::NelderMead;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::common::BandData;

// ---------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------

const RISE: [f64; 5] = [1.0, 1.94053, -0.175084, -2.24588, -1.12498];
const DECAY_AMP: [f64; 2] = [0.689008, 0.302963];
const DECAY_RATE: [f64; 2] = [-1.60053, -0.278318];

/// Davenport et al. (2014) flare template at x = (t − t_peak) / t_half.
#[inline]
pub fn davenport_template(x: f64) -> f64 {
    if x < -1.0 {
        0.0
    } else if x <= 0.0 {
        RISE[0] + x * (RISE[1] + x * (RISE[2] + x * (RISE[3] + x * RISE[4])))
    } else {
        DECAY_AMP[0] * (DECAY_RATE[0] * x).exp() + DECAY_AMP[1] * (DECAY_RATE[1] * x).exp()
    }
}

#[inline]
fn rise_antiderivative(x: f64) -> f64 {
    x * (RISE[0]
        + x * (RISE[1] / 2.0
            + x * (RISE[2] / 3.0 + x * (RISE[3] / 4.0 + x * RISE[4] / 5.0))))
}

/// ∫ T(x) dx over [a, b], piecewise in closed form. `b` may be +∞.
pub fn davenport_template_integral(a: f64, b: f64) -> f64 {
    if !(b > a) {
        return 0.0;
    }
    let mut total = 0.0;
    let (lo, hi) = (a.max(-1.0), b.min(0.0));
    if hi > lo {
        total += rise_antiderivative(hi) - rise_antiderivative(lo);
    }
    let lo = a.max(0.0);
    if b > lo {
        // e^{r·lo}·(e^{r·(b−lo)} − 1) avoids cancelling two nearly equal tails.
        for j in 0..2 {
            let r = DECAY_RATE[j];
            total += DECAY_AMP[j] / r * (r * lo).exp() * (r * (b - lo)).exp_m1();
        }
    }
    total
}

/// Total template area ∫ T dx ≈ 1.827; a flare's equivalent duration is
/// `amplitude · t_half · DAVENPORT_AREA`.
pub fn davenport_area() -> f64 {
    davenport_template_integral(-1.0, f64::INFINITY)
}

/// Template averaged over [x − hw, x + hw]; `hw` is the half exposure in units of t_half.
#[inline]
fn template_binned(x: f64, hw: f64) -> f64 {
    if hw > 1e-9 {
        if x + hw <= -1.0 {
            return 0.0;
        }
        davenport_template_integral(x - hw, x + hw) / (2.0 * hw)
    } else {
        davenport_template(x)
    }
}

/// Flux of one flare at `t`, averaged over an exposure of length `exposure_time`
/// centred on `t` (0 for instantaneous sampling).
#[inline]
pub fn flare_flux(t: f64, t_peak: f64, t_half: f64, amplitude: f64, exposure_time: f64) -> f64 {
    amplitude * template_binned((t - t_peak) / t_half, 0.5 * exposure_time / t_half)
}

/// Sum of flare templates (no baseline) evaluated at `times`.
pub fn flare_model(times: &[f64], flares: &[FlareComponent], exposure_time: f64) -> Vec<f64> {
    times
        .iter()
        .map(|&t| {
            flares
                .iter()
                .map(|f| flare_flux(t, f.t_peak, f.t_half, f.amplitude, exposure_time))
                .sum()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Options and results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlareFitOptions {
    /// Largest number of template components tried.
    pub max_flares: usize,
    /// Exposure length in the same unit as the times; `None` samples the
    /// template at each timestamp.
    pub exposure_time: Option<f64>,
    /// Polynomial order of the additive baseline (0 = constant, 1 = linear).
    pub baseline_order: usize,
    /// BIC improvement required to accept each additional component,
    /// including the first one over a baseline-only model.
    pub min_delta_bic: f64,
    /// Bounds on t_half; `None` uses 5% of the median cadence and a quarter
    /// of the window length.
    pub t_half_min: Option<f64>,
    pub t_half_max: Option<f64>,
}

impl Default for FlareFitOptions {
    fn default() -> Self {
        FlareFitOptions {
            max_flares: 4,
            exposure_time: None,
            baseline_order: 0,
            min_delta_bic: 10.0,
            t_half_min: None,
            t_half_max: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlareComponent {
    pub t_peak: f64,
    /// Full width at half maximum of the template.
    pub t_half: f64,
    /// Peak flux above the baseline.
    pub amplitude: f64,
    /// ∫ flare dt in the input time unit; the equivalent duration when the
    /// flux is normalised to the quiescent level.
    pub equivalent_duration: f64,
    pub t_peak_err: Option<f64>,
    pub t_half_err: Option<f64>,
    pub amplitude_err: Option<f64>,
    pub equivalent_duration_err: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlareFitResult {
    /// Selected components, sorted by peak time. Empty when no flare beats a
    /// baseline-only model by `min_delta_bic`.
    pub flares: Vec<FlareComponent>,
    /// Baseline polynomial coefficients in powers of (t − t_ref).
    pub baseline: Vec<f64>,
    pub t_ref: f64,
    pub chi2: f64,
    pub reduced_chi2: f64,
    pub bic: f64,
    /// BIC for K = 0, 1, … components, up to the first K that was rejected.
    pub per_k_bic: Vec<f64>,
    pub n_obs: usize,
}

impl FlareFitResult {
    /// Baseline + all flare components at `times`.
    pub fn model(&self, times: &[f64], exposure_time: f64) -> Vec<f64> {
        let mut out = flare_model(times, &self.flares, exposure_time);
        for (o, &t) in out.iter_mut().zip(times) {
            let dt = t - self.t_ref;
            let mut p = 1.0;
            for &c in &self.baseline {
                *o += c * p;
                p *= dt;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Variable-projection problem
// ---------------------------------------------------------------------------

struct FlareProblem {
    /// Times relative to t_ref.
    t: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    t_ref: f64,
    /// Baseline columns, each of length n.
    base: Vec<Vec<f64>>,
    half_exp: f64,
    /// Bounds on [t_peak, ln t_half], relative to t_ref.
    tp_lo: f64,
    tp_hi: f64,
    lnth_lo: f64,
    lnth_hi: f64,
    /// Typical spacing used for grid seeding.
    cadence: f64,
    /// Time each point represents (midpoint spacing, capped across gaps),
    /// used to measure how much of a component's area the data cover.
    support: Vec<f64>,
}

/// Minimum fraction of a component's area that must fall on observed points.
/// Below it the amplitude is mostly extrapolation (e.g. a spike placed in a
/// gap), so the component is held at zero; at 0.5 at most half of any ED is
/// unobserved.
const MIN_COVERAGE: f64 = 0.5;

struct LinearSolution {
    /// Flare amplitudes followed by baseline coefficients.
    coef: Vec<f64>,
    chi2: f64,
}

impl FlareProblem {
    fn n(&self) -> usize {
        self.t.len()
    }

    fn flare_column(&self, tp: f64, th: f64, out: &mut [f64]) {
        let hw = self.half_exp / th;
        for (o, &t) in out.iter_mut().zip(&self.t) {
            *o = template_binned((t - tp) / th, hw);
        }
    }

    /// Solve for non-negative flare amplitudes and a free baseline, given
    /// the template columns for K flares (row-major, K × n). Flares with
    /// `usable[j] == false` are held at zero amplitude.
    fn solve_linear(&self, cols: &[f64], k: usize, usable: &[bool]) -> LinearSolution {
        let n = self.n();
        let nb = self.base.len();
        let p = k + nb;
        let col = |j: usize| -> &[f64] {
            if j < k { &cols[j * n..(j + 1) * n] } else { &self.base[j - k] }
        };

        let mut gram = vec![0.0; p * p];
        let mut rhs = vec![0.0; p];
        for a in 0..p {
            let ca = col(a);
            rhs[a] = (0..n).map(|i| self.w[i] * ca[i] * self.y[i]).sum();
            for b in a..p {
                let cb = col(b);
                let v: f64 = (0..n).map(|i| self.w[i] * ca[i] * cb[i]).sum();
                gram[a * p + b] = v;
                gram[b * p + a] = v;
            }
        }

        // Active-set loop: drop the most negative (or degenerate) flare
        // column and re-solve until all active amplitudes are non-negative.
        let mut active: Vec<bool> = (0..p).map(|j| j >= k || usable[j]).collect();
        let mut coef = vec![0.0; p];
        for _ in 0..=k {
            let idx: Vec<usize> = (0..p).filter(|&j| active[j]).collect();
            let m = idx.len();
            let mut g = vec![0.0; m * m];
            let mut r = vec![0.0; m];
            for (a, &ia) in idx.iter().enumerate() {
                r[a] = rhs[ia];
                for (b, &ib) in idx.iter().enumerate() {
                    g[a * m + b] = gram[ia * p + ib];
                }
            }
            coef.iter_mut().for_each(|c| *c = 0.0);
            if !cholesky_solve(&mut g, &mut r, m) {
                // Singular system: a flare column is (near) zero or collinear.
                match (0..k).filter(|&j| active[j]).min_by(|&a, &b| {
                    gram[a * p + a].total_cmp(&gram[b * p + b])
                }) {
                    Some(j) => {
                        active[j] = false;
                        continue;
                    }
                    None => break,
                }
            }
            for (a, &ia) in idx.iter().enumerate() {
                coef[ia] = r[a];
            }
            match (0..k)
                .filter(|&j| active[j] && coef[j] < 0.0)
                .min_by(|&a, &b| coef[a].total_cmp(&coef[b]))
            {
                Some(j) => active[j] = false,
                None => break,
            }
        }

        let mut chi2 = 0.0;
        for i in 0..n {
            let mut m = 0.0;
            for j in 0..p {
                m += coef[j] * col(j)[i];
            }
            let r = self.y[i] - m;
            chi2 += self.w[i] * r * r;
        }
        LinearSolution { coef, chi2 }
    }

    /// Profiled χ² for nonlinear parameters θ = [t_peak_1, ln t_half_1, …].
    fn profile(&self, theta: &[f64], buf: &mut Vec<f64>) -> LinearSolution {
        let k = theta.len() / 2;
        let n = self.n();
        buf.resize(k * n, 0.0);
        for j in 0..k {
            self.flare_column(theta[2 * j], theta[2 * j + 1].exp(), &mut buf[j * n..(j + 1) * n]);
        }
        let area = davenport_area();
        let usable: Vec<bool> = (0..k)
            .map(|j| {
                let col = &buf[j * n..(j + 1) * n];
                let covered: f64 = col.iter().zip(&self.support).map(|(c, s)| c * s).sum();
                covered >= MIN_COVERAGE * theta[2 * j + 1].exp() * area
            })
            .collect();
        self.solve_linear(buf, k, &usable)
    }

    /// Squared bound violation in units of each parameter's allowed range.
    fn bound_excess(&self, theta: &[f64]) -> f64 {
        let mut e = 0.0;
        for j in 0..theta.len() / 2 {
            let (tp, lt) = (theta[2 * j], theta[2 * j + 1]);
            let wtp = self.tp_hi - self.tp_lo;
            let wlt = self.lnth_hi - self.lnth_lo;
            e += ((self.tp_lo - tp).max(0.0) / wtp).powi(2) + ((tp - self.tp_hi).max(0.0) / wtp).powi(2);
            e += ((self.lnth_lo - lt).max(0.0) / wlt).powi(2) + ((lt - self.lnth_hi).max(0.0) / wlt).powi(2);
        }
        e
    }

    fn clamp(&self, theta: &mut [f64]) {
        for j in 0..theta.len() / 2 {
            theta[2 * j] = theta[2 * j].clamp(self.tp_lo, self.tp_hi);
            theta[2 * j + 1] = theta[2 * j + 1].clamp(self.lnth_lo, self.lnth_hi);
        }
    }
}

struct ProfileCost<'a> {
    problem: &'a FlareProblem,
}

impl CostFunction for ProfileCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<f64, ArgminError> {
        let excess = self.problem.bound_excess(theta);
        let mut clamped = theta.clone();
        self.problem.clamp(&mut clamped);
        let mut buf = Vec::new();
        let chi2 = self.problem.profile(&clamped, &mut buf).chi2;
        Ok(chi2 + 1e6 * excess)
    }
}

/// Cholesky solve of a small SPD system in place; returns false if not PD.
fn cholesky_solve(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if !(d > 1e-12 * a[j * n + j].abs().max(1e-300)) {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    true
}

/// Inverse of a small SPD matrix via Cholesky, column by column.
fn spd_inverse(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut inv = vec![0.0; n * n];
    for c in 0..n {
        let mut m = a.to_vec();
        let mut e = vec![0.0; n];
        e[c] = 1.0;
        if !cholesky_solve(&mut m, &mut e, n) {
            return None;
        }
        for r in 0..n {
            inv[r * n + c] = e[r];
        }
    }
    Some(inv)
}

fn nelder_mead(problem: &FlareProblem, start: Vec<f64>, max_iters: u64) -> (Vec<f64>, f64) {
    let cost = ProfileCost { problem };
    let start_cost = cost.cost(&start).unwrap_or(f64::INFINITY);
    let step_tp = 0.5 * problem.cadence;
    let step_lt = 0.3;
    let mut simplex = vec![start.clone()];
    for i in 0..start.len() {
        let mut v = start.clone();
        v[i] += if i % 2 == 0 { step_tp } else { step_lt };
        simplex.push(v);
    }
    NelderMead::<Vec<f64>, f64>::new(simplex)
        .with_sd_tolerance(1e-5)
        .ok()
        .and_then(|solver| {
            Executor::new(cost, solver)
                .configure(|s| s.max_iters(max_iters))
                .run()
                .ok()
        })
        .and_then(|res| res.state().best_param.clone())
        .map(|mut p| {
            problem.clamp(&mut p);
            let c = ProfileCost { problem }.cost(&p).unwrap_or(f64::INFINITY);
            (p, c)
        })
        .filter(|(_, c)| *c <= start_cost)
        .unwrap_or((start, start_cost))
}

// ---------------------------------------------------------------------------
// Fitting
// ---------------------------------------------------------------------------

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    let n = xs.len();
    if n % 2 == 1 { xs[n / 2] } else { 0.5 * (xs[n / 2 - 1] + xs[n / 2]) }
}

fn build_problem(times: &[f64], flux: &[f64], flux_err: &[f64], opts: &FlareFitOptions) -> Option<FlareProblem> {
    let mut rows: Vec<(f64, f64, f64)> = times
        .iter()
        .zip(flux)
        .zip(flux_err)
        .map(|((&t, &f), &e)| (t, f, e))
        .filter(|(t, f, e)| t.is_finite() && f.is_finite() && e.is_finite() && *e > 0.0)
        .collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = rows.len();
    let nb = opts.baseline_order + 1;
    if n < nb + 4 {
        return None;
    }
    let t_ref = rows[0].0;
    let t: Vec<f64> = rows.iter().map(|r| r.0 - t_ref).collect();
    let span = t[n - 1];
    if !(span > 0.0) {
        return None;
    }
    let mut dts: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 0.0).collect();
    let cadence = if dts.is_empty() { span / n as f64 } else { median(&mut dts) };
    let exposure = opts.exposure_time.unwrap_or(0.0).max(0.0);

    let th_min = opts.t_half_min.unwrap_or(0.05 * cadence).max(1e-12);
    let th_max = opts.t_half_max.unwrap_or(0.25 * span).max(2.0 * th_min);
    let cap = cadence.max(exposure);
    let support = (0..n)
        .map(|i| {
            let left = if i > 0 { t[i] - t[i - 1] } else { t[1] - t[0] };
            let right = if i + 1 < n { t[i + 1] - t[i] } else { left };
            0.5 * (left.min(cap) + right.min(cap))
        })
        .collect();

    let base = (0..nb).map(|o| t.iter().map(|&x| x.powi(o as i32)).collect()).collect();
    Some(FlareProblem {
        y: rows.iter().map(|r| r.1).collect(),
        w: rows.iter().map(|r| 1.0 / (r.2 * r.2)).collect(),
        t,
        t_ref,
        base,
        half_exp: 0.5 * exposure,
        tp_lo: -0.1 * span,
        tp_hi: span,
        lnth_lo: th_min.ln(),
        lnth_hi: th_max.ln(),
        cadence: cadence.max(exposure),
        support,
    })
}

/// Grid-search a new component on top of `prev` (held fixed), centred on the
/// largest positive residual.
fn seed_new_flare(problem: &FlareProblem, prev: &[f64], buf: &mut Vec<f64>) -> Vec<f64> {
    let n = problem.n();
    let sol = problem.profile(prev, buf);
    let k = prev.len() / 2;
    let mut best_i = 0;
    let mut best_s = f64::NEG_INFINITY;
    for i in 0..n {
        let mut m = 0.0;
        for j in 0..k {
            m += sol.coef[j] * buf[j * n + i];
        }
        for (b, c) in problem.base.iter().zip(&sol.coef[k..]) {
            m += c * b[i];
        }
        let s = (problem.y[i] - m) * problem.w[i].sqrt();
        if s > best_s {
            best_s = s;
            best_i = i;
        }
    }

    const N_TP: usize = 11;
    const N_TH: usize = 20;
    let t0 = problem.t[best_i];
    let dt = problem.cadence;
    let mut theta = prev.to_vec();
    theta.extend_from_slice(&[t0, 0.0]);
    let mut best = (f64::INFINITY, theta.clone());
    for a in 0..N_TP {
        let tp = (t0 - 1.5 * dt + 2.5 * dt * a as f64 / (N_TP - 1) as f64)
            .clamp(problem.tp_lo, problem.tp_hi);
        for b in 0..N_TH {
            let lt = problem.lnth_lo
                + (problem.lnth_hi - problem.lnth_lo) * b as f64 / (N_TH - 1) as f64;
            theta[2 * k] = tp;
            theta[2 * k + 1] = lt;
            let chi2 = problem.profile(&theta, buf).chi2;
            if chi2 < best.0 {
                best = (chi2, theta.clone());
            }
        }
    }
    best.1
}

/// Fit K template components plus a baseline, choosing K by BIC.
///
/// Times, flux and errors are parallel arrays for a single band; points with
/// non-finite values or non-positive errors are dropped. Returns `None` when
/// there are too few points to fit.
pub fn fit_flares(
    times: &[f64],
    flux: &[f64],
    flux_err: &[f64],
    opts: &FlareFitOptions,
) -> Option<FlareFitResult> {
    let problem = build_problem(times, flux, flux_err, opts)?;
    let n = problem.n();
    let nb = problem.base.len();
    let ln_n = (n as f64).ln();
    let mut buf = Vec::new();

    let bic = |chi2: f64, k: usize| chi2 + (3 * k + nb) as f64 * ln_n;
    let mut best_theta: Vec<f64> = Vec::new();
    let mut best_bic = bic(problem.profile(&best_theta, &mut buf).chi2, 0);
    let mut per_k_bic = vec![best_bic];

    for k in 1..=opts.max_flares {
        if n < 3 * k + nb + 1 {
            break;
        }
        let seed = seed_new_flare(&problem, &best_theta, &mut buf);
        let iters = 150 * seed.len() as u64;
        let (theta, _) = nelder_mead(&problem, seed, iters);
        let (theta, chi2) = nelder_mead(&problem, theta, iters);
        let k_bic = bic(chi2, k);
        per_k_bic.push(k_bic);
        if k_bic < best_bic - opts.min_delta_bic {
            best_bic = k_bic;
            best_theta = theta;
        } else {
            break;
        }
    }
    Some(finalize(&problem, &best_theta, per_k_bic))
}

/// Amplitudes, uncertainties and fit statistics for the selected nonlinear
/// parameters θ = [t_peak_1, ln t_half_1, …] (relative to t_ref).
fn finalize(problem: &FlareProblem, best_theta: &[f64], per_k_bic: Vec<f64>) -> FlareFitResult {
    let n = problem.n();
    let nb = problem.base.len();
    let ln_n = (n as f64).ln();
    let bic = |chi2: f64, k: usize| chi2 + (3 * k + nb) as f64 * ln_n;
    let mut buf = Vec::new();

    // Drop components whose amplitude was clamped to zero.
    let sol = problem.profile(best_theta, &mut buf);
    let k_all = best_theta.len() / 2;
    let theta: Vec<f64> = (0..k_all)
        .filter(|&j| sol.coef[j] > 0.0)
        .flat_map(|j| [best_theta[2 * j], best_theta[2 * j + 1]])
        .collect();
    let sol = problem.profile(&theta, &mut buf);
    let k = theta.len() / 2;

    let errs = parameter_errors(problem, &theta, &sol);
    let dof = (n as f64 - (3 * k + nb) as f64).max(1.0);
    let area = davenport_area();
    let mut flares: Vec<FlareComponent> = (0..k)
        .map(|j| {
            let th = theta[2 * j + 1].exp();
            let amp = sol.coef[j];
            let e = errs.as_ref().map(|e| e[j]);
            FlareComponent {
                t_peak: theta[2 * j] + problem.t_ref,
                t_half: th,
                amplitude: amp,
                equivalent_duration: amp * th * area,
                t_peak_err: e.map(|e| e.0),
                t_half_err: e.map(|e| e.1),
                amplitude_err: e.map(|e| e.2),
                equivalent_duration_err: e.map(|e| e.3),
            }
        })
        .collect();
    flares.sort_by(|a, b| a.t_peak.total_cmp(&b.t_peak));

    FlareFitResult {
        flares,
        baseline: sol.coef[k..].to_vec(),
        t_ref: problem.t_ref,
        chi2: sol.chi2,
        reduced_chi2: sol.chi2 / dof,
        bic: bic(sol.chi2, k),
        per_k_bic,
        n_obs: n,
    }
}

/// Fit many independent windows in parallel.
pub fn fit_flares_batch(windows: &[BandData], opts: &FlareFitOptions) -> Vec<Option<FlareFitResult>> {
    windows
        .par_iter()
        .map(|w| fit_flares(&w.times, &w.values, &w.errors, opts))
        .collect()
}

// ---------------------------------------------------------------------------
// GPU batch
// ---------------------------------------------------------------------------

/// Pack windows for the GPU, rescaled per window. Returns each window's time
/// scale s, with t' = t / s and ln t_half' = ln t_half − ln s. Flux is shifted
/// and scaled too, but the GPU returns only nonlinear parameters, so that
/// needs no inverse.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn pack_for_gpu(
    problems: &[(usize, FlareProblem)],
    opts: &FlareFitOptions,
) -> (crate::gpu_types::FlareBatchInput, Vec<f64>) {
    use crate::gpu_types::{FlareBatchInput, FLARE_WINDOW_PARAMS};
    let total: usize = problems.iter().map(|(_, p)| p.n()).sum();
    let mut input = FlareBatchInput {
        times: Vec::with_capacity(total),
        flux: Vec::with_capacity(total),
        weight: Vec::with_capacity(total),
        support: Vec::with_capacity(total),
        offsets: Vec::with_capacity(problems.len() + 1),
        window_params: Vec::with_capacity(problems.len() * FLARE_WINDOW_PARAMS),
        max_flares: opts.max_flares,
        n_baseline: opts.baseline_order + 1,
        min_delta_bic: opts.min_delta_bic,
        template_area: davenport_area(),
        // The search parks components on the coverage boundary; a small margin
        // keeps float32 kernels from accepting one the double-precision
        // finalisation then rejects.
        min_coverage: MIN_COVERAGE * (1.0 + 1e-3),
    };
    let mut scales = Vec::with_capacity(problems.len());
    input.offsets.push(0);
    for (_, p) in problems {
        let ts = p.cadence;
        let shift = median(&mut p.y.clone());
        let sigma = median(&mut p.w.iter().map(|w| 1.0 / w.sqrt()).collect::<Vec<_>>());
        input.times.extend(p.t.iter().map(|t| t / ts));
        input.flux.extend(p.y.iter().map(|y| (y - shift) / sigma));
        input.weight.extend(p.w.iter().map(|w| w * sigma * sigma));
        input.support.extend(p.support.iter().map(|s| s / ts));
        input.offsets.push(input.times.len() as i32);
        input.window_params.extend_from_slice(&[
            p.half_exp / ts,
            p.tp_lo / ts,
            p.tp_hi / ts,
            p.lnth_lo - ts.ln(),
            p.lnth_hi - ts.ln(),
            p.cadence / ts,
        ]);
        scales.push(ts);
    }
    (input, scales)
}

/// [`fit_flares_batch`] with the component search on the GPU.
///
/// The GPU runs the full K = 0, 1, … search (seed grid, Nelder-Mead, BIC) in
/// float32 (Metal) or float64 (CUDA). Amplitudes, baseline, uncertainties and
/// fit statistics are then computed on the CPU in double precision from the
/// selected peak times and widths, exactly as in [`fit_flares`].
///
/// Supports up to `FLARE_GPU_MAX_K` components and a constant or linear baseline.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn fit_flares_batch_gpu(
    gpu: &crate::gpu::GpuContext,
    windows: &[BandData],
    opts: &FlareFitOptions,
) -> Result<Vec<Option<FlareFitResult>>, String> {
    use crate::gpu_types::{FLARE_GPU_MAX_BASELINE_ORDER, FLARE_GPU_MAX_K};
    if opts.max_flares > FLARE_GPU_MAX_K {
        return Err(format!("GPU flare fitting supports max_flares <= {FLARE_GPU_MAX_K}"));
    }
    if opts.baseline_order > FLARE_GPU_MAX_BASELINE_ORDER {
        return Err(format!(
            "GPU flare fitting supports baseline_order <= {FLARE_GPU_MAX_BASELINE_ORDER}"
        ));
    }
    let problems: Vec<(usize, FlareProblem)> = windows
        .par_iter()
        .enumerate()
        .filter_map(|(i, w)| build_problem(&w.times, &w.values, &w.errors, opts).map(|p| (i, p)))
        .collect();
    let mut results: Vec<Option<FlareFitResult>> = vec![None; windows.len()];
    if problems.is_empty() {
        return Ok(results);
    }
    let (input, scales) = pack_for_gpu(&problems, opts);
    let outputs = gpu.batch_flare_fit(&input)?;
    let fitted: Vec<(usize, FlareFitResult)> = problems
        .par_iter()
        .zip(outputs.par_iter().zip(scales.par_iter()))
        .map(|((i, p), (out, &ts))| {
            let theta: Vec<f64> = out.theta.chunks(2).flat_map(|c| [c[0] * ts, c[1] + ts.ln()]).collect();
            (*i, finalize(p, &theta, out.per_k_bic.clone()))
        })
        .collect();
    for (i, r) in fitted {
        results[i] = Some(r);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Uncertainties
// ---------------------------------------------------------------------------

/// (σ_tpeak, σ_thalf, σ_A, σ_ED) for one component.
type FlareErrors = (f64, f64, f64, f64);

/// Gauss-Newton covariance (Jᵀ W J)⁻¹ over all parameters at the optimum,
/// inflated by the reduced χ² when it exceeds 1.
fn parameter_errors(problem: &FlareProblem, theta: &[f64], sol: &LinearSolution) -> Option<Vec<FlareErrors>> {
    let n = problem.n();
    let k = theta.len() / 2;
    let nb = problem.base.len();
    let p = 3 * k + nb;
    if k == 0 || n <= p {
        return None;
    }
    let mut jac = vec![0.0; p * n];
    let mut plus = vec![0.0; n];
    let mut minus = vec![0.0; n];
    for j in 0..k {
        let tp = theta[2 * j];
        let th = theta[2 * j + 1].exp();
        let amp = sol.coef[j];
        let h = 1e-4 * th;
        problem.flare_column(tp + h, th, &mut plus);
        problem.flare_column(tp - h, th, &mut minus);
        for i in 0..n {
            jac[(3 * j) * n + i] = amp * (plus[i] - minus[i]) / (2.0 * h);
        }
        problem.flare_column(tp, th + h, &mut plus);
        problem.flare_column(tp, th - h, &mut minus);
        for i in 0..n {
            jac[(3 * j + 1) * n + i] = amp * (plus[i] - minus[i]) / (2.0 * h);
        }
        problem.flare_column(tp, th, &mut jac[(3 * j + 2) * n..(3 * j + 3) * n]);
    }
    for (b, col) in problem.base.iter().enumerate() {
        jac[(3 * k + b) * n..(3 * k + b + 1) * n].copy_from_slice(col);
    }

    let mut fisher = vec![0.0; p * p];
    for a in 0..p {
        for b in a..p {
            let v: f64 = (0..n).map(|i| problem.w[i] * jac[a * n + i] * jac[b * n + i]).sum();
            fisher[a * p + b] = v;
            fisher[b * p + a] = v;
        }
    }
    let mut cov = spd_inverse(&fisher, p)?;
    let scale = (sol.chi2 / (n - p) as f64).max(1.0);
    cov.iter_mut().for_each(|c| *c *= scale);
    let area = davenport_area();
    Some(
        (0..k)
            .map(|j| {
                let th = theta[2 * j + 1].exp();
                let amp = sol.coef[j];
                let c = |a: usize, b: usize| cov[(3 * j + a) * p + 3 * j + b];
                let var_ed =
                    area * area * (th * th * c(2, 2) + amp * amp * c(1, 1) + 2.0 * amp * th * c(1, 2));
                (c(0, 0).sqrt(), c(1, 1).sqrt(), c(2, 2).sqrt(), var_ed.max(0.0).sqrt())
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_in_gap_is_held_at_zero() {
        // 1-min cadence with a 5-min gap and an excess on the first point after it.
        let minute = 1.0 / 1440.0;
        let times: Vec<f64> = (0..200).filter(|i| !(100..105).contains(i)).map(|i| i as f64 * minute).collect();
        let mut flux = vec![0.0; times.len()];
        flux[100] = 5e-3;
        let err = vec![1e-3; times.len()];
        let opts = FlareFitOptions { exposure_time: Some(minute), ..Default::default() };
        let p = build_problem(&times, &flux, &err, &opts).unwrap();
        let mut buf = Vec::new();

        // A narrow component peaking inside the gap only reaches the data
        // through a sliver of its tail, so its amplitude is unconstrained.
        let in_gap = [102.0 * minute, (0.1 * minute).ln()];
        assert_eq!(p.profile(&in_gap, &mut buf).coef[0], 0.0);
        let on_data = [105.0 * minute, (0.1 * minute).ln()];
        assert!(p.profile(&on_data, &mut buf).coef[0] > 0.0);
    }
}

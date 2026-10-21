use std::cell::RefCell;
use std::collections::HashMap;

use argmin::core::{CostFunction, Executor, Gradient, Error as ArgminError};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::common::{finite_or_none, BandData};

/// Selects the uncertainty estimation method used after PSO model selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UncertaintyMethod {
    /// Stochastic Variational Inference (default).
    #[default]
    Svi,
    /// Laplace approximation using Hessian at MAP estimate.
    Laplace,
}

// Physical constants (CGS)
const MSUN_CGS: f64 = 1.989e33;
const C_CGS: f64 = 2.998e10;
const SECS_PER_DAY: f64 = 86400.0;
const H_PLANCK: f64 = 6.626e-27; // erg·s
const K_BOLTZMANN: f64 = 1.381e-16; // erg/K
const SIGMA_SB: f64 = 5.6704e-5; // erg/cm²/s/K⁴
// radiation constant (unused for now, but available if needed for multi-layer)
// const A_RAD: f64 = 4.0 * SIGMA_SB / C_CGS;

/// Sigma inflation factor for mean-field VI posteriors.
const SIGMA_INFLATION_FACTOR: f64 = 4.0;

const MAX_CACHE: usize = 2000;

// ---------------------------------------------------------------------------
// Manual Adam optimizer
// ---------------------------------------------------------------------------

struct ManualAdam {
    m: Vec<f64>,
    v: Vec<f64>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: usize,
}

impl ManualAdam {
    fn new(n_params: usize, lr: f64) -> Self {
        Self {
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
        }
    }

    fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            let g = grads[i];
            if !g.is_finite() {
                continue;
            }
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ---------------------------------------------------------------------------
// Model definitions
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SviModelName {
    Bazin,
    Villar,
    MetzgerKN,
    Tde,
    Arnett,
    Magnetar,
    ShockCooling,
    Afterglow,
}

impl std::fmt::Display for SviModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SviModelName::Bazin => write!(f, "Bazin"),
            SviModelName::Villar => write!(f, "Villar"),
            SviModelName::MetzgerKN => write!(f, "MetzgerKN"),
            SviModelName::Tde => write!(f, "Tde"),
            SviModelName::Arnett => write!(f, "Arnett"),
            SviModelName::Magnetar => write!(f, "Magnetar"),
            SviModelName::ShockCooling => write!(f, "ShockCooling"),
            SviModelName::Afterglow => write!(f, "Afterglow"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SviModel {
    Bazin,
    Villar,
    MetzgerKN,
    Tde,
    Arnett,
    Magnetar,
    ShockCooling,
    Afterglow,
}

impl SviModel {
    fn n_params(self) -> usize {
        match self {
            SviModel::Bazin => 6,
            SviModel::Villar => 7,
            SviModel::MetzgerKN => 5,
            SviModel::Tde => 7,
            SviModel::Arnett => 5,
            SviModel::Magnetar => 5,
            SviModel::ShockCooling => 5,
            SviModel::Afterglow => 6,
        }
    }

    fn sigma_extra_idx(self) -> usize {
        match self {
            SviModel::Bazin => 5,
            SviModel::Villar => 6,
            SviModel::MetzgerKN => 4,
            SviModel::Tde => 6,
            SviModel::Arnett => 4,
            SviModel::Magnetar => 4,
            SviModel::ShockCooling => 4,
            SviModel::Afterglow => 5,
        }
    }

    fn t0_idx(self) -> usize {
        match self {
            SviModel::Bazin => 2,
            SviModel::Villar => 3,
            SviModel::MetzgerKN => 3,
            SviModel::Tde => 2,
            SviModel::Arnett => 1,
            SviModel::Magnetar => 1,
            SviModel::ShockCooling => 1,
            SviModel::Afterglow => 1,
        }
    }

    fn is_sequential(self) -> bool {
        matches!(self, SviModel::MetzgerKN)
    }

    fn to_name(self) -> SviModelName {
        match self {
            SviModel::Bazin => SviModelName::Bazin,
            SviModel::Villar => SviModelName::Villar,
            SviModel::MetzgerKN => SviModelName::MetzgerKN,
            SviModel::Tde => SviModelName::Tde,
            SviModel::Arnett => SviModelName::Arnett,
            SviModel::Magnetar => SviModelName::Magnetar,
            SviModel::ShockCooling => SviModelName::ShockCooling,
            SviModel::Afterglow => SviModelName::Afterglow,
        }
    }

    fn from_name(name: &SviModelName) -> Self {
        match name {
            SviModelName::Bazin => SviModel::Bazin,
            SviModelName::Villar => SviModel::Villar,
            SviModelName::MetzgerKN => SviModel::MetzgerKN,
            SviModelName::Tde => SviModel::Tde,
            SviModelName::Arnett => SviModel::Arnett,
            SviModelName::Magnetar => SviModel::Magnetar,
            SviModelName::ShockCooling => SviModel::ShockCooling,
            SviModelName::Afterglow => SviModel::Afterglow,
        }
    }
}

// ---------------------------------------------------------------------------
// Model evaluation functions
// ---------------------------------------------------------------------------

#[inline]
fn bazin_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let b = params[1];
    let t0 = params[2];
    let tau_rise = params[3].exp();
    let tau_fall = params[4].exp();
    let dt = t - t0;
    let e_fall = (-dt / tau_fall).exp();
    let sig = 1.0 / (1.0 + (-dt / tau_rise).exp());
    a * e_fall * sig + b
}

#[inline]
fn bazin_flux_grad(params: &[f64], t: f64, out: &mut [f64]) {
    let a = params[0].exp();
    let t0 = params[2];
    let tau_rise = params[3].exp();
    let tau_fall = params[4].exp();
    let dt = t - t0;
    let e_fall = (-dt / tau_fall).exp();
    let sig = 1.0 / (1.0 + (-dt / tau_rise).exp());
    let base = a * e_fall * sig;
    out[0] = base;
    out[1] = 1.0;
    out[2] = base * (1.0 / tau_fall - (1.0 - sig) / tau_rise);
    out[3] = -base * (1.0 - sig) * dt / tau_rise;
    out[4] = base * dt / tau_fall;
    out[5] = 0.0;
}

#[inline]
fn bazin_flux_eval_and_grad(params: &[f64], t: f64, out: &mut [f64]) -> f64 {
    bazin_eval_and_grad_with(params[0].exp(), params[1], params[2], params[3].exp(), params[4].exp(), t, out)
}

/// `bazin_flux_eval_and_grad` with exp(log_A), exp(log_tau_*) precomputed.
#[inline]
fn bazin_eval_and_grad_with(a: f64, b: f64, t0: f64, tau_rise: f64, tau_fall: f64, t: f64, out: &mut [f64]) -> f64 {
    let dt = t - t0;
    let e_fall = (-dt / tau_fall).exp();
    let sig = 1.0 / (1.0 + (-dt / tau_rise).exp());
    let base = a * e_fall * sig;
    out[0] = base;
    out[1] = 1.0;
    out[2] = base * (1.0 / tau_fall - (1.0 - sig) / tau_rise);
    out[3] = -base * (1.0 - sig) * dt / tau_rise;
    out[4] = base * dt / tau_fall;
    out[5] = 0.0;
    base + b
}

#[inline]
fn villar_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let beta = params[1];
    let gamma = params[2].exp();
    let t0 = params[3];
    let tau_rise = params[4].exp();
    let tau_fall = params[5].exp();
    let phase = t - t0;
    let sig_rise = 1.0 / (1.0 + (-phase / tau_rise).exp());
    let k = 10.0;
    let w = 1.0 / (1.0 + (-k * (phase - gamma)).exp());
    let piece_left = 1.0 - beta * phase;
    let piece_right = (1.0 - beta * gamma) * ((gamma - phase) / tau_fall).exp();
    let piece = (1.0 - w) * piece_left + w * piece_right;
    a * sig_rise * piece
}

#[inline]
fn villar_flux_grad(params: &[f64], t: f64, out: &mut [f64]) {
    let a = params[0].exp();
    let beta = params[1];
    let gamma = params[2].exp();
    let t0 = params[3];
    let tau_rise = params[4].exp();
    let tau_fall = params[5].exp();
    let phase = t - t0;
    let k = 10.0;
    let sig_rise = 1.0 / (1.0 + (-phase / tau_rise).exp());
    let w = 1.0 / (1.0 + (-k * (phase - gamma)).exp());
    let piece_left = 1.0 - beta * phase;
    let e_decay = ((gamma - phase) / tau_fall).exp();
    let piece_right = (1.0 - beta * gamma) * e_decay;
    let piece = (1.0 - w) * piece_left + w * piece_right;
    let flux = a * sig_rise * piece;

    out[0] = flux;

    let d_pl_dbeta = -phase;
    let d_pr_dbeta = -gamma * e_decay;
    out[1] = a * sig_rise * ((1.0 - w) * d_pl_dbeta + w * d_pr_dbeta);

    let dw_dgamma = -k * w * (1.0 - w);
    let dw_dloggamma = dw_dgamma * gamma;
    let dpr_dgamma = e_decay * (-beta + (1.0 - beta * gamma) / tau_fall);
    let dpr_dloggamma = dpr_dgamma * gamma;
    let d_piece_dloggamma = dw_dloggamma * (piece_right - piece_left) + w * dpr_dloggamma;
    out[2] = a * sig_rise * d_piece_dloggamma;

    let dsig_dphase = sig_rise * (1.0 - sig_rise) / tau_rise;
    let dsig_dt0 = -dsig_dphase;
    let dw_dphase = k * w * (1.0 - w);
    let dw_dt0 = -dw_dphase;
    let dpl_dt0 = beta;
    let dpr_dt0 = (1.0 - beta * gamma) * e_decay / tau_fall;
    let d_piece_dt0 = dw_dt0 * (piece_right - piece_left) + (1.0 - w) * dpl_dt0 + w * dpr_dt0;
    out[3] = a * (dsig_dt0 * piece + sig_rise * d_piece_dt0);

    out[4] = a * piece * sig_rise * (1.0 - sig_rise) * (-phase / tau_rise);

    let d_pr_dlogtf = piece_right * (phase - gamma) / tau_fall;
    out[5] = a * sig_rise * w * d_pr_dlogtf;

    out[6] = 0.0;
}

#[inline]
fn villar_flux_eval_and_grad(params: &[f64], t: f64, out: &mut [f64]) -> f64 {
    let a = params[0].exp();
    let beta = params[1];
    let gamma = params[2].exp();
    let t0 = params[3];
    let tau_rise = params[4].exp();
    let tau_fall = params[5].exp();
    let phase = t - t0;
    let k = 10.0;
    let sig_rise = 1.0 / (1.0 + (-phase / tau_rise).exp());
    let w = 1.0 / (1.0 + (-k * (phase - gamma)).exp());
    let piece_left = 1.0 - beta * phase;
    let e_decay = ((gamma - phase) / tau_fall).exp();
    let piece_right = (1.0 - beta * gamma) * e_decay;
    let piece = (1.0 - w) * piece_left + w * piece_right;
    let flux = a * sig_rise * piece;

    out[0] = flux;

    let d_pl_dbeta = -phase;
    let d_pr_dbeta = -gamma * e_decay;
    out[1] = a * sig_rise * ((1.0 - w) * d_pl_dbeta + w * d_pr_dbeta);

    let dw_dgamma = -k * w * (1.0 - w);
    let dw_dloggamma = dw_dgamma * gamma;
    let dpr_dgamma = e_decay * (-beta + (1.0 - beta * gamma) / tau_fall);
    let dpr_dloggamma = dpr_dgamma * gamma;
    let d_piece_dloggamma = dw_dloggamma * (piece_right - piece_left) + w * dpr_dloggamma;
    out[2] = a * sig_rise * d_piece_dloggamma;

    let dsig_dphase = sig_rise * (1.0 - sig_rise) / tau_rise;
    let dsig_dt0 = -dsig_dphase;
    let dw_dphase = k * w * (1.0 - w);
    let dw_dt0 = -dw_dphase;
    let dpl_dt0 = beta;
    let dpr_dt0 = (1.0 - beta * gamma) * e_decay / tau_fall;
    let d_piece_dt0 = dw_dt0 * (piece_right - piece_left) + (1.0 - w) * dpl_dt0 + w * dpr_dt0;
    out[3] = a * (dsig_dt0 * piece + sig_rise * d_piece_dt0);

    out[4] = a * piece * sig_rise * (1.0 - sig_rise) * (-phase / tau_rise);

    let d_pr_dlogtf = piece_right * (phase - gamma) / tau_fall;
    out[5] = a * sig_rise * w * d_pr_dlogtf;

    out[6] = 0.0;
    flux
}

#[inline]
fn tde_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let b = params[1];
    let t0 = params[2];
    let tau_rise = params[3].exp();
    let tau_fall = params[4].exp();
    let alpha = params[5];
    let phase = t - t0;
    let sig = 1.0 / (1.0 + (-phase / tau_rise).exp());
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let w = 1.0 + phase_soft / tau_fall;
    let decay = w.powf(-alpha);
    a * sig * decay + b
}

#[inline]
fn tde_flux_grad(params: &[f64], t: f64, out: &mut [f64]) {
    let a = params[0].exp();
    let t0 = params[2];
    let tau_rise = params[3].exp();
    let tau_fall = params[4].exp();
    let alpha = params[5];
    let phase = t - t0;
    let sig = 1.0 / (1.0 + (-phase / tau_rise).exp());
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_phase = phase.exp() / (1.0 + phase.exp());
    let w = 1.0 + phase_soft / tau_fall;
    let decay = w.powf(-alpha);
    let base = a * sig * decay;

    out[0] = base;
    out[1] = 1.0;
    let dsig_dt0 = -sig * (1.0 - sig) / tau_rise;
    let ddecay_dt0 = alpha * w.powf(-alpha - 1.0) * sig_phase / tau_fall;
    out[2] = a * (dsig_dt0 * decay + sig * ddecay_dt0);
    out[3] = a * decay * (-sig * (1.0 - sig) * phase / tau_rise);
    out[4] = a * sig * alpha * w.powf(-alpha - 1.0) * phase_soft / tau_fall;
    out[5] = a * sig * (-w.ln()) * decay;
    out[6] = 0.0;
}

#[inline]
fn tde_flux_eval_and_grad(params: &[f64], t: f64, out: &mut [f64]) -> f64 {
    let a = params[0].exp();
    let t0 = params[2];
    let tau_rise = params[3].exp();
    let tau_fall = params[4].exp();
    let alpha = params[5];
    let phase = t - t0;
    let sig = 1.0 / (1.0 + (-phase / tau_rise).exp());
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_phase = phase.exp() / (1.0 + phase.exp());
    let w = 1.0 + phase_soft / tau_fall;
    let decay = w.powf(-alpha);
    let base = a * sig * decay;

    out[0] = base;
    out[1] = 1.0;
    let dsig_dt0 = -sig * (1.0 - sig) / tau_rise;
    let ddecay_dt0 = alpha * w.powf(-alpha - 1.0) * sig_phase / tau_fall;
    out[2] = a * (dsig_dt0 * decay + sig * ddecay_dt0);
    out[3] = a * decay * (-sig * (1.0 - sig) * phase / tau_rise);
    out[4] = a * sig * alpha * w.powf(-alpha - 1.0) * phase_soft / tau_fall;
    out[5] = a * sig * (-w.ln()) * decay;
    out[6] = 0.0;
    base + params[1]
}

#[inline]
fn arnett_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let t0 = params[1];
    let tau_m = params[2].exp();
    let logit_f = params[3];
    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let f = 1.0 / (1.0 + (-logit_f).exp());
    const TAU_NI: f64 = 8.8;
    const TAU_CO: f64 = 111.3;
    let e_ni = (-phase_soft / TAU_NI).exp();
    let e_co = (-phase_soft / TAU_CO).exp();
    let heat = f * e_ni + (1.0 - f) * e_co;
    let x = phase_soft / tau_m;
    let trap = 1.0 - (-x * x).exp();
    a * heat * trap
}

#[inline]
fn arnett_flux_grad(params: &[f64], t: f64, out: &mut [f64]) {
    let a = params[0].exp();
    let t0 = params[1];
    let tau_m = params[2].exp();
    let logit_f = params[3];
    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_p = phase.exp() / (1.0 + phase.exp());
    let f = 1.0 / (1.0 + (-logit_f).exp());
    const TAU_NI: f64 = 8.8;
    const TAU_CO: f64 = 111.3;
    let e_ni = (-phase_soft / TAU_NI).exp();
    let e_co = (-phase_soft / TAU_CO).exp();
    let heat = f * e_ni + (1.0 - f) * e_co;
    let x = phase_soft / tau_m;
    let exp_x2 = (-x * x).exp();
    let trap = 1.0 - exp_x2;
    let flux = a * heat * trap;

    out[0] = flux;
    let dheat_dps = -f * e_ni / TAU_NI - (1.0 - f) * e_co / TAU_CO;
    let dtrap_dps = 2.0 * phase_soft * exp_x2 / (tau_m * tau_m);
    out[1] = a * (-sig_p) * (dheat_dps * trap + heat * dtrap_dps);
    out[2] = -2.0 * a * heat * exp_x2 * x * x;
    out[3] = a * trap * (e_ni - e_co) * f * (1.0 - f);
    out[4] = 0.0;
}

#[inline]
fn arnett_flux_eval_and_grad(params: &[f64], t: f64, out: &mut [f64]) -> f64 {
    let a = params[0].exp();
    let t0 = params[1];
    let tau_m = params[2].exp();
    let logit_f = params[3];
    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_p = phase.exp() / (1.0 + phase.exp());
    let f = 1.0 / (1.0 + (-logit_f).exp());
    const TAU_NI: f64 = 8.8;
    const TAU_CO: f64 = 111.3;
    let e_ni = (-phase_soft / TAU_NI).exp();
    let e_co = (-phase_soft / TAU_CO).exp();
    let heat = f * e_ni + (1.0 - f) * e_co;
    let x = phase_soft / tau_m;
    let exp_x2 = (-x * x).exp();
    let trap = 1.0 - exp_x2;
    let flux = a * heat * trap;

    out[0] = flux;
    let dheat_dps = -f * e_ni / TAU_NI - (1.0 - f) * e_co / TAU_CO;
    let dtrap_dps = 2.0 * phase_soft * exp_x2 / (tau_m * tau_m);
    out[1] = a * (-sig_p) * (dheat_dps * trap + heat * dtrap_dps);
    out[2] = -2.0 * a * heat * exp_x2 * x * x;
    out[3] = a * trap * (e_ni - e_co) * f * (1.0 - f);
    out[4] = 0.0;
    flux
}

#[inline]
fn magnetar_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let t0 = params[1];
    let tau_sd = params[2].exp();
    let tau_diff = params[3].exp();
    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let w = 1.0 + phase_soft / tau_sd;
    let spindown = w.powi(-2);
    let x = phase_soft / tau_diff;
    let trap = 1.0 - (-x * x).exp();
    a * spindown * trap
}

#[inline]
fn magnetar_flux_grad(params: &[f64], t: f64, out: &mut [f64]) {
    let a = params[0].exp();
    let t0 = params[1];
    let tau_sd = params[2].exp();
    let tau_diff = params[3].exp();
    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_p = phase.exp() / (1.0 + phase.exp());
    let w = 1.0 + phase_soft / tau_sd;
    let spindown = w.powi(-2);
    let x = phase_soft / tau_diff;
    let exp_x2 = (-x * x).exp();
    let trap = 1.0 - exp_x2;
    let flux = a * spindown * trap;

    out[0] = flux;
    let dspindown_dps = -2.0 * w.powi(-3) / tau_sd;
    let dtrap_dps = 2.0 * phase_soft * exp_x2 / (tau_diff * tau_diff);
    out[1] = a * (-sig_p) * (dspindown_dps * trap + spindown * dtrap_dps);
    out[2] = a * trap * 2.0 * phase_soft * w.powi(-3) / tau_sd;
    out[3] = -2.0 * a * spindown * exp_x2 * x * x;
    out[4] = 0.0;
}

#[inline]
fn magnetar_flux_eval_and_grad(params: &[f64], t: f64, out: &mut [f64]) -> f64 {
    let a = params[0].exp();
    let t0 = params[1];
    let tau_sd = params[2].exp();
    let tau_diff = params[3].exp();
    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_p = phase.exp() / (1.0 + phase.exp());
    let w = 1.0 + phase_soft / tau_sd;
    let spindown = w.powi(-2);
    let x = phase_soft / tau_diff;
    let exp_x2 = (-x * x).exp();
    let trap = 1.0 - exp_x2;
    let flux = a * spindown * trap;

    out[0] = flux;
    let dspindown_dps = -2.0 * w.powi(-3) / tau_sd;
    let dtrap_dps = 2.0 * phase_soft * exp_x2 / (tau_diff * tau_diff);
    out[1] = a * (-sig_p) * (dspindown_dps * trap + spindown * dtrap_dps);
    out[2] = a * trap * 2.0 * phase_soft * w.powi(-3) / tau_sd;
    out[3] = -2.0 * a * spindown * exp_x2 * x * x;
    out[4] = 0.0;
    flux
}

#[inline]
fn shockcooling_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let t0 = params[1];
    let n = params[2];
    let tau_tr = params[3].exp();
    let phase = t - t0;
    let sig5 = 1.0 / (1.0 + (-phase * 5.0).exp());
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let cooling = phase_soft.powf(-n);
    let ratio = phase_soft / tau_tr;
    let cutoff = (-ratio * ratio).exp();
    a * sig5 * cooling * cutoff
}

#[inline]
fn shockcooling_flux_grad(params: &[f64], t: f64, out: &mut [f64]) {
    let a = params[0].exp();
    let t0 = params[1];
    let n = params[2];
    let tau_tr = params[3].exp();
    let phase = t - t0;
    let sig5 = 1.0 / (1.0 + (-phase * 5.0).exp());
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_p = phase.exp() / (1.0 + phase.exp());
    let cooling = phase_soft.powf(-n);
    let ratio = phase_soft / tau_tr;
    let cutoff = (-ratio * ratio).exp();
    let base = cooling * cutoff;
    let flux = a * sig5 * base;

    out[0] = flux;
    out[1] = a
        * base
        * (-5.0 * sig5 * (1.0 - sig5)
            + sig5 * sig_p * (n / phase_soft + 2.0 * phase_soft / (tau_tr * tau_tr)));
    out[2] = -flux * phase_soft.ln();
    out[3] = flux * 2.0 * ratio * ratio;
    out[4] = 0.0;
}

#[inline]
fn shockcooling_flux_eval_and_grad(params: &[f64], t: f64, out: &mut [f64]) -> f64 {
    let a = params[0].exp();
    let t0 = params[1];
    let n = params[2];
    let tau_tr = params[3].exp();
    let phase = t - t0;
    let sig5 = 1.0 / (1.0 + (-phase * 5.0).exp());
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_p = phase.exp() / (1.0 + phase.exp());
    let cooling = phase_soft.powf(-n);
    let ratio = phase_soft / tau_tr;
    let cutoff = (-ratio * ratio).exp();
    let base = cooling * cutoff;
    let flux = a * sig5 * base;

    out[0] = flux;
    out[1] = a
        * base
        * (-5.0 * sig5 * (1.0 - sig5)
            + sig5 * sig_p * (n / phase_soft + 2.0 * phase_soft / (tau_tr * tau_tr)));
    out[2] = -flux * phase_soft.ln();
    out[3] = flux * 2.0 * ratio * ratio;
    out[4] = 0.0;
    flux
}

#[inline]
fn afterglow_flux_eval(params: &[f64], t: f64) -> f64 {
    let a = params[0].exp();
    let t0 = params[1];
    let t_b = params[2].exp();
    let alpha1 = params[3];
    let alpha2 = params[4];
    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let r = phase_soft / t_b;
    let ln_r = r.ln();
    let u1 = (2.0 * alpha1 * ln_r).exp();
    let u2 = (2.0 * alpha2 * ln_r).exp();
    let u = u1 + u2;
    a * u.powf(-0.5)
}

#[inline]
fn afterglow_flux_grad(params: &[f64], t: f64, out: &mut [f64]) {
    let a = params[0].exp();
    let t0 = params[1];
    let t_b = params[2].exp();
    let alpha1 = params[3];
    let alpha2 = params[4];
    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_p = phase.exp() / (1.0 + phase.exp());
    let r = phase_soft / t_b;
    let ln_r = r.ln();
    let u1 = (2.0 * alpha1 * ln_r).exp();
    let u2 = (2.0 * alpha2 * ln_r).exp();
    let u = u1 + u2;
    let flux = a * u.powf(-0.5);

    out[0] = flux;
    let du_dps = (2.0 * alpha1 * u1 + 2.0 * alpha2 * u2) / phase_soft;
    let dflux_dps = a * (-0.5) * u.powf(-1.5) * du_dps;
    out[1] = dflux_dps * (-sig_p);
    let du_dlog_tb = -(2.0 * alpha1 * u1 + 2.0 * alpha2 * u2);
    out[2] = a * (-0.5) * u.powf(-1.5) * du_dlog_tb;
    out[3] = a * (-0.5) * u.powf(-1.5) * 2.0 * ln_r * u1;
    out[4] = a * (-0.5) * u.powf(-1.5) * 2.0 * ln_r * u2;
    out[5] = 0.0;
}

#[inline]
fn afterglow_flux_eval_and_grad(params: &[f64], t: f64, out: &mut [f64]) -> f64 {
    let a = params[0].exp();
    let t0 = params[1];
    let t_b = params[2].exp();
    let alpha1 = params[3];
    let alpha2 = params[4];
    let phase = t - t0;
    let phase_soft = (1.0 + phase.exp()).ln() + 1e-6;
    let sig_p = phase.exp() / (1.0 + phase.exp());
    let r = phase_soft / t_b;
    let ln_r = r.ln();
    let u1 = (2.0 * alpha1 * ln_r).exp();
    let u2 = (2.0 * alpha2 * ln_r).exp();
    let u = u1 + u2;
    let flux = a * u.powf(-0.5);

    out[0] = flux;
    let du_dps = (2.0 * alpha1 * u1 + 2.0 * alpha2 * u2) / phase_soft;
    let dflux_dps = a * (-0.5) * u.powf(-1.5) * du_dps;
    out[1] = dflux_dps * (-sig_p);
    let du_dlog_tb = -(2.0 * alpha1 * u1 + 2.0 * alpha2 * u2);
    out[2] = a * (-0.5) * u.powf(-1.5) * du_dlog_tb;
    out[3] = a * (-0.5) * u.powf(-1.5) * 2.0 * ln_r * u1;
    out[4] = a * (-0.5) * u.powf(-1.5) * 2.0 * ln_r * u2;
    out[5] = 0.0;
    flux
}

#[inline]
fn eval_model(model: SviModel, params: &[f64], t: f64) -> f64 {
    match model {
        SviModel::Bazin => bazin_flux_eval(params, t),
        SviModel::Villar => villar_flux_eval(params, t),
        SviModel::Tde => tde_flux_eval(params, t),
        SviModel::Arnett => arnett_flux_eval(params, t),
        SviModel::Magnetar => magnetar_flux_eval(params, t),
        SviModel::ShockCooling => shockcooling_flux_eval(params, t),
        SviModel::Afterglow => afterglow_flux_eval(params, t),
        SviModel::MetzgerKN => panic!("MetzgerKN requires batch evaluation"),
    }
}

#[inline]
#[allow(dead_code)]
fn eval_model_grad(model: SviModel, params: &[f64], t: f64, out: &mut [f64]) {
    match model {
        SviModel::Bazin => bazin_flux_grad(params, t, out),
        SviModel::Villar => villar_flux_grad(params, t, out),
        SviModel::Tde => tde_flux_grad(params, t, out),
        SviModel::Arnett => arnett_flux_grad(params, t, out),
        SviModel::Magnetar => magnetar_flux_grad(params, t, out),
        SviModel::ShockCooling => shockcooling_flux_grad(params, t, out),
        SviModel::Afterglow => afterglow_flux_grad(params, t, out),
        SviModel::MetzgerKN => panic!("MetzgerKN requires batch evaluation"),
    }
}

#[inline]
fn eval_model_and_grad(model: SviModel, params: &[f64], t: f64, out: &mut [f64]) -> f64 {
    match model {
        SviModel::Bazin => bazin_flux_eval_and_grad(params, t, out),
        SviModel::Villar => villar_flux_eval_and_grad(params, t, out),
        SviModel::Tde => tde_flux_eval_and_grad(params, t, out),
        SviModel::Arnett => arnett_flux_eval_and_grad(params, t, out),
        SviModel::Magnetar => magnetar_flux_eval_and_grad(params, t, out),
        SviModel::ShockCooling => shockcooling_flux_eval_and_grad(params, t, out),
        SviModel::Afterglow => afterglow_flux_eval_and_grad(params, t, out),
        SviModel::MetzgerKN => panic!("MetzgerKN requires batch evaluation"),
    }
}

// ---------------------------------------------------------------------------
// Metzger 1-zone kilonova
// ---------------------------------------------------------------------------

// Cache the ODE solve for repeated evaluations at the same parameters. The key
// is exact (bit patterns of the physical params and the grid extent), so a hit
// returns the same grid a fresh solve would: finite-difference gradients and
// Hessians see every parameter change.
type KnCacheKey = (u64, u64, u64, u64);
type KnCacheVal = (Vec<f64>, Vec<f64>); // (grid_t_day, grid_norm)

/// LRU cache 
struct KnRingCache {
    keys: Vec<KnCacheKey>,
    vals: Vec<KnCacheVal>,
    map: HashMap<KnCacheKey, usize>,
    next: usize,
    capacity: usize,
}

impl KnRingCache {
    fn new(capacity: usize) -> Self {
        Self {
            keys: Vec::with_capacity(capacity),
            vals: Vec::with_capacity(capacity),
            map: HashMap::with_capacity(capacity),
            next: 0,
            capacity,
        }
    }

    fn get(&self, key: &KnCacheKey) -> Option<&KnCacheVal> {
        self.map.get(key).map(|&idx| &self.vals[idx])
    }

    fn insert(&mut self, key: KnCacheKey, val: KnCacheVal) {
        if self.keys.len() < self.capacity {
            let idx = self.keys.len();
            self.keys.push(key);
            self.vals.push(val);
            self.map.insert(key, idx);
        } else {
            // Evict oldest entry
            let idx = self.next;
            self.map.remove(&self.keys[idx]);
            self.keys[idx] = key;
            self.vals[idx] = val;
            self.map.insert(key, idx);
            self.next = (self.next + 1) % self.capacity;
        }
    }
}

thread_local! {
    static KN_CACHE: RefCell<KnRingCache> = RefCell::new(KnRingCache::new(MAX_CACHE));
}

// the actual ODE solve, factored out so we can cache it
fn metzger_kn_solve_ode(m_ej: f64, v_ej: f64, kappa_r: f64, phase_max: f64) -> Option<(Vec<f64>, Vec<f64>)> {
    let n_grid: usize = 200;
    let log_t_min = 0.01f64.ln();
    let log_t_max = (phase_max * 1.05).ln();

    let grid_t_day: Vec<f64> = (0..n_grid)
        .map(|i| (log_t_min + (log_t_max - log_t_min) * i as f64 / (n_grid - 1) as f64).exp())
        .collect();

    let ye: f64 = 0.1;
    let xn0: f64 = 1.0 - 2.0 * ye;

    let scale: f64 = 1e40;
    let e0 = 0.5 * m_ej * v_ej * v_ej;
    let mut e_th = e0 / scale;
    let mut e_kin = e0 / scale;
    let mut v = v_ej;
    let mut r = grid_t_day[0] * SECS_PER_DAY * v;

    let mut grid_lrad: Vec<f64> = Vec::with_capacity(n_grid);

    for i in 0..n_grid {
        let t_day = grid_t_day[i];
        let t_sec = t_day * SECS_PER_DAY;

        let eth_factor = 0.34 * t_day.powf(0.74);
        let eth = 0.36
            * ((-0.56 * t_day).exp()
                + if eth_factor > 1e-10 {
                    (1.0 + eth_factor).ln() / eth_factor
                } else {
                    1.0
                });

        let xn = xn0 * (-t_sec / 900.0).exp();
        let eps_neutron = 3.2e14 * xn;
        let time_term = (0.5 - ((t_sec - 1.3) / 0.11).atan() / std::f64::consts::PI).max(1e-30);
        let eps_rp = 2e18 * eth * time_term.powf(1.3);
        let l_heat = m_ej * (eps_neutron + eps_rp) / scale;

        let xr = 1.0 - xn0;
        let xn_decayed = xn0 - xn;
        let kappa_eff = 0.4 * xn_decayed + kappa_r * xr;

        let t_diff =
            3.0 * kappa_eff * m_ej / (4.0 * std::f64::consts::PI * C_CGS * v * t_sec) + r / C_CGS;

        let l_rad = if e_th > 0.0 && t_diff > 0.0 {
            e_th / t_diff
        } else {
            0.0
        };
        grid_lrad.push(l_rad);

        let l_pdv = if r > 0.0 { e_th * v / r } else { 0.0 };

        if i < n_grid - 1 {
            let dt_sec = (grid_t_day[i + 1] - grid_t_day[i]) * SECS_PER_DAY;
            e_th += (l_heat - l_pdv - l_rad) * dt_sec;
            if e_th < 0.0 {
                e_th = 0.0;
            }
            e_kin += l_pdv * dt_sec;
            v = (2.0 * e_kin * scale / m_ej).sqrt().min(C_CGS);
            r += v * dt_sec;
        }
    }

    let l_peak = grid_lrad.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if l_peak <= 0.0 || !l_peak.is_finite() {
        return None;
    }
    let grid_norm: Vec<f64> = grid_lrad.iter().map(|l| l / l_peak).collect();
    Some((grid_t_day, grid_norm))
}

fn metzger_kn_eval_batch(params: &[f64], obs_times: &[f64]) -> Vec<f64> {
    let m_ej = 10f64.powf(params[0]) * MSUN_CGS;
    let v_ej = 10f64.powf(params[1]) * C_CGS;
    let kappa_r = 10f64.powf(params[2]);
    let t0 = params[3];

    let phases: Vec<f64> = obs_times.iter().map(|&t| t - t0).collect();
    let phase_max = phases.iter().cloned().fold(0.01f64, f64::max);
    if phase_max <= 0.01 {
        return vec![0.0; obs_times.len()];
    }

    let key = (params[0].to_bits(), params[1].to_bits(), params[2].to_bits(), phase_max.to_bits());

    let cached = KN_CACHE.with(|c| c.borrow().get(&key).cloned());
    let (grid_t_day, grid_norm) = if let Some(val) = cached {
        val
    } else {
        match metzger_kn_solve_ode(m_ej, v_ej, kappa_r, phase_max) {
            Some(val) => {
                KN_CACHE.with(|c| {
                    c.borrow_mut().insert(key, val.clone());
                });
                val
            }
            None => return vec![0.0; obs_times.len()],
        }
    };

    let n_grid = grid_t_day.len();
    phases
        .iter()
        .map(|&phase| {
            if phase <= 0.0 {
                return 0.0;
            }
            if phase <= grid_t_day[0] {
                return grid_norm[0];
            }
            if phase >= grid_t_day[n_grid - 1] {
                return *grid_norm.last().unwrap();
            }
            let idx = grid_t_day
                .partition_point(|&gt| gt < phase)
                .min(n_grid - 1)
                .max(1);
            let frac = (phase - grid_t_day[idx - 1]) / (grid_t_day[idx] - grid_t_day[idx - 1]);
            grid_norm[idx - 1] + frac * (grid_norm[idx] - grid_norm[idx - 1])
        })
        .collect()
}

/// Evaluate the Metzger 1-zone kilonova model returning per-band apparent AB
/// magnitudes via blackbody emission, matching the NMMA `metzger_lc` approach.
///
/// # Parameters
/// - `params`: `[log10(mej/Msun), log10(vej/c), log10(kappa), t0_offset]`
/// - `obs_times`: rest-frame days since explosion
/// - `band_frequencies_hz`: slice of `(band_name, frequency_Hz)` for each filter
/// - `d_l_cm`: luminosity distance in cm
///
/// # Returns
/// `HashMap<band_name, Vec<app_mag>>` — apparent AB magnitudes per band per time.
pub fn metzger_kn_mags(
    params: &[f64],
    obs_times: &[f64],
    band_frequencies_hz: &[(&str, f64)],
    d_l_cm: f64,
) -> HashMap<String, Vec<f64>> {
    let m_ej = 10f64.powf(params[0]) * MSUN_CGS;
    let v_ej = 10f64.powf(params[1]) * C_CGS;
    let kappa_r = 10f64.powf(params[2]);
    let t0 = params[3];

    let phases: Vec<f64> = obs_times.iter().map(|&t| t - t0).collect();
    let phase_max = phases.iter().cloned().fold(0.01f64, f64::max);

    let faint = 99.0;
    if phase_max <= 0.01 {
        let mut out = HashMap::new();
        for &(name, _) in band_frequencies_hz {
            out.insert(name.to_string(), vec![faint; obs_times.len()]);
        }
        return out;
    }

    // Build log-spaced time grid
    let n_grid: usize = 200;
    let log_t_min = 0.01f64.ln();
    let log_t_max = (phase_max * 1.05).ln();
    let grid_t_day: Vec<f64> = (0..n_grid)
        .map(|i| (log_t_min + (log_t_max - log_t_min) * i as f64 / (n_grid - 1) as f64).exp())
        .collect();

    let ye: f64 = 0.1;
    let xn0: f64 = 1.0 - 2.0 * ye;

    let scale: f64 = 1e40;
    let e0 = 0.5 * m_ej * v_ej * v_ej;
    let mut e_th = e0 / scale;
    let mut e_kin = e0 / scale;
    let mut v = v_ej;
    let mut r = grid_t_day[0] * SECS_PER_DAY * v;

    // Store L_rad (scaled) and R at each grid point
    let mut grid_lrad = Vec::with_capacity(n_grid);
    let mut grid_r = Vec::with_capacity(n_grid);

    for i in 0..n_grid {
        let t_day = grid_t_day[i];
        let t_sec = t_day * SECS_PER_DAY;

        let eth_factor = 0.34 * t_day.powf(0.74);
        let eth = 0.36
            * ((-0.56 * t_day).exp()
                + if eth_factor > 1e-10 {
                    (1.0 + eth_factor).ln() / eth_factor
                } else {
                    1.0
                });

        let xn = xn0 * (-t_sec / 900.0).exp();
        let eps_neutron = 3.2e14 * xn;
        let time_term = (0.5 - ((t_sec - 1.3) / 0.11).atan() / std::f64::consts::PI).max(1e-30);
        let eps_rp = 2e18 * eth * time_term.powf(1.3);
        let l_heat = m_ej * (eps_neutron + eps_rp) / scale;

        let xr = 1.0 - xn0;
        let xn_decayed = xn0 - xn;
        let kappa_eff = 0.4 * xn_decayed + kappa_r * xr;

        let t_diff =
            3.0 * kappa_eff * m_ej / (4.0 * std::f64::consts::PI * C_CGS * v * t_sec) + r / C_CGS;

        let l_rad = if e_th > 0.0 && t_diff > 0.0 {
            e_th / t_diff
        } else {
            0.0
        };
        grid_lrad.push(l_rad);
        grid_r.push(r);

        let l_pdv = if r > 0.0 { e_th * v / r } else { 0.0 };

        if i < n_grid - 1 {
            let dt_sec = (grid_t_day[i + 1] - grid_t_day[i]) * SECS_PER_DAY;
            e_th += (l_heat - l_pdv - l_rad) * dt_sec;
            if e_th < 0.0 {
                e_th = 0.0;
            }
            e_kin += l_pdv * dt_sec;
            v = (2.0 * e_kin * scale / m_ej).sqrt().min(C_CGS);
            r += v * dt_sec;
        }
    }

    // Compute effective temperature on grid:
    // T_eff = (E_th * scale / (a_rad * (4/3)π R³))^0.25  (from NMMA)
    // Then L_bol = 4π R² σ T⁴, but we use E_th-derived T directly.
    // NMMA uses: temp = 1e10 * (3 * E / (arad * 4π R³))^0.25
    // where E is in 1e40 units. So L_real = grid_lrad * scale (erg/s).
    // T = (L_real / (4π R² σ))^0.25
    let mut grid_temp = Vec::with_capacity(n_grid);
    let dist_sq = d_l_cm * d_l_cm;

    for i in 0..n_grid {
        let l_real = grid_lrad[i] * scale; // erg/s
        let r_i = grid_r[i]; // cm
        if l_real <= 0.0 || r_i <= 0.0 {
            grid_temp.push(0.0);
            continue;
        }
        let t_eff = (l_real / (4.0 * std::f64::consts::PI * r_i * r_i * SIGMA_SB)).powf(0.25);
        grid_temp.push(if t_eff.is_finite() { t_eff } else { 0.0 });
    }

    // For each band, interpolate T_eff and R to obs times, compute BB flux → AB mag
    let mut result = HashMap::new();
    for &(band_name, nu) in band_frequencies_hz {
        let mags: Vec<f64> = phases
            .iter()
            .map(|&phase| {
                if phase <= 0.0 {
                    return faint;
                }
                // Interpolate T_eff and R from grid
                let (temp, r_photo) = if phase <= grid_t_day[0] {
                    (grid_temp[0], grid_r[0])
                } else if phase >= grid_t_day[n_grid - 1] {
                    (grid_temp[n_grid - 1], grid_r[n_grid - 1])
                } else {
                    let idx = grid_t_day
                        .partition_point(|&gt| gt < phase)
                        .min(n_grid - 1)
                        .max(1);
                    let frac =
                        (phase - grid_t_day[idx - 1]) / (grid_t_day[idx] - grid_t_day[idx - 1]);
                    let t_interp = grid_temp[idx - 1] + frac * (grid_temp[idx] - grid_temp[idx - 1]);
                    let r_interp = grid_r[idx - 1] + frac * (grid_r[idx] - grid_r[idx - 1]);
                    (t_interp, r_interp)
                };

                if temp <= 0.0 || r_photo <= 0.0 {
                    return faint;
                }

                // Planck spectral flux density at frequency nu, temperature temp:
                // F_ν = (2h ν³/c²) / (exp(hν/kT) - 1) × (R/d)²
                let x = H_PLANCK * nu / (K_BOLTZMANN * temp);
                let x_clamped = x.min(700.0); // avoid overflow
                let bb_factor = 2.0 * std::f64::consts::PI * H_PLANCK * nu * nu * nu / (C_CGS * C_CGS);
                let f_nu = bb_factor / x_clamped.exp_m1() * (r_photo * r_photo / dist_sq);

                if f_nu <= 0.0 || !f_nu.is_finite() {
                    return faint;
                }

                // AB magnitude: m = -2.5 log10(F_ν) - 48.6
                -2.5 * f_nu.log10() - 48.6
            })
            .collect();
        result.insert(band_name.to_string(), mags);
    }
    result
}

#[inline]
fn eval_model_batch(model: SviModel, params: &[f64], times: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; times.len()];
    eval_model_batch_into(model, params, times, &mut out);
    out
}

/// MetzgerKN has no amplitude parameter, so its flux is rescaled to peak at 1
/// over the non-upper-limit points. Scales `preds` in place and returns the
/// factor (1.0 for every other model, or when the peak is degenerate).
fn apply_metzger_kn_scale(model: SviModel, preds: &mut [f64], is_upper: &[bool]) -> f64 {
    if model != SviModel::MetzgerKN {
        return 1.0;
    }
    let max_pred = preds
        .iter()
        .zip(is_upper.iter())
        .filter(|(_, is_up)| !**is_up)
        .map(|(p, _)| *p)
        .fold(f64::NEG_INFINITY, f64::max);
    if !(max_pred > 1e-10 && max_pred.is_finite()) {
        return 1.0;
    }
    let scale = (1.0 / max_pred).clamp(0.1, 10.0);
    for pred in preds.iter_mut() {
        *pred *= scale;
    }
    scale
}

/// Model predictions at the band's times, with the MetzgerKN rescale applied.
fn scaled_preds(model: SviModel, params: &[f64], data: &BandFitData) -> (Vec<f64>, f64) {
    let mut preds = eval_model_batch(model, params, &data.times);
    let scale = apply_metzger_kn_scale(model, &mut preds, &data.is_upper);
    (preds, scale)
}

/// Write model predictions into a pre-allocated buffer (avoids allocation in hot loops).
fn eval_model_batch_into(model: SviModel, params: &[f64], times: &[f64], out: &mut [f64]) {
    if model.is_sequential() {
        let batch = metzger_kn_eval_batch(params, times);
        out[..times.len()].copy_from_slice(&batch);
    } else if model == SviModel::Bazin {
        let (a, b, t0) = (params[0].exp(), params[1], params[2]);
        let (tau_rise, tau_fall) = (params[3].exp(), params[4].exp());
        for (i, &t) in times.iter().enumerate() {
            let dt = t - t0;
            let e_fall = (-dt / tau_fall).exp();
            let sig = 1.0 / (1.0 + (-dt / tau_rise).exp());
            out[i] = a * e_fall * sig + b;
        }
    } else {
        for (i, &t) in times.iter().enumerate() {
            out[i] = eval_model(model, params, t);
        }
    }
}

fn metzger_kn_grad_batch_into(params: &[f64], times: &[f64], out: &mut [f64]) {
    let n_times = times.len();
    let n_params = 5;
    let n_phys = 4;
    let base = metzger_kn_eval_batch(params, times);
    let eps = 1e-5;
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for j in 0..n_phys {
        let mut p_plus = params.to_vec();
        p_plus[j] += eps;
        let f_plus = metzger_kn_eval_batch(&p_plus, times);
        for i in 0..n_times {
            out[i * n_params + j] = (f_plus[i] - base[i]) / eps;
        }
    }
}

// ---------------------------------------------------------------------------
// PSO cost function
// ---------------------------------------------------------------------------

#[inline]
fn erf_approx(x: f64) -> f64 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let val = 1.0 - poly * (-a * a).exp();
    if x >= 0.0 {
        val
    } else {
        -val
    }
}

#[inline]
fn log_normal_cdf(x: f64) -> f64 {
    if x > 8.0 {
        return 0.0;
    }
    if x < -30.0 {
        return -0.5 * x * x - 0.5 * (2.0 * std::f64::consts::PI).ln() - (-x).ln();
    }
    let z = -x * std::f64::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * z.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erfc_z = poly * (-z * z).exp();
    let phi = if z >= 0.0 {
        0.5 * erfc_z
    } else {
        1.0 - 0.5 * erfc_z
    };
    (phi.max(1e-300)).ln()
}

/// phi(x)/Phi(x) = d/dx log Phi(x). Computed
/// from `log_normal_cdf` so it stays finite deep in the lower tail, where
/// `0.5 * (1 + erf(x/√2))` cancels to zero.
#[inline]
fn normal_pdf_over_cdf(x: f64) -> f64 {
    (-0.5 * x * x - 0.5 * (2.0 * std::f64::consts::PI).ln() - log_normal_cdf(x)).exp()
}

// ---------------------------------------------------------------------------
// MultiBazin: variable-component sum of Bazin pulses
// ---------------------------------------------------------------------------
//
// Parameterisation for K components:
//   [log_A_1, t0_1, log_tau_rise_1, log_tau_fall_1,   // component 1
//    log_A_2, t0_2, log_tau_rise_2, log_tau_fall_2,   // component 2
//    ...
//    B, log_sigma_extra]                               // shared (last 2)
//
// Total params = 4*K + 2.

const MULTI_BAZIN_MAX_K: usize = 4;
/// Params per Bazin component (log_A, t0, log_tau_rise, log_tau_fall).
const MULTI_BAZIN_COMP_PARAMS: usize = 4;

fn multi_bazin_eval_batch(params: &[f64], k: usize, times: &[f64], out: &mut [f64]) {
    // exp(log_A), exp(log_tau_*) per component, hoisted out of the time loop
    let mut comps = [(0.0, 0.0, 0.0, 0.0); MULTI_BAZIN_MAX_K];
    for c in 0..k {
        let off = c * MULTI_BAZIN_COMP_PARAMS;
        comps[c] = (params[off].exp(), params[off + 1], params[off + 2].exp(), params[off + 3].exp());
    }
    let b = params[k * MULTI_BAZIN_COMP_PARAMS];
    for (i, &t) in times.iter().enumerate() {
        let mut sum = 0.0;
        for &(a, t0, tau_rise, tau_fall) in &comps[..k] {
            let dt = t - t0;
            sum += a * (-dt / tau_fall).exp() * (1.0 / (1.0 + (-dt / tau_rise).exp()));
        }
        out[i] = sum + b;
    }
}

/// Number of parameters for MultiBazin with K components.
fn multi_bazin_n_params(k: usize) -> usize {
    MULTI_BAZIN_COMP_PARAMS * k + 2 // +B +log_sigma_extra
}

fn multi_bazin_sigma_extra_idx(k: usize) -> usize {
    MULTI_BAZIN_COMP_PARAMS * k + 1
}

/// PSO bounds for MultiBazin with K components.
fn multi_bazin_bounds(k: usize, t_range: (f64, f64)) -> (Vec<f64>, Vec<f64>) {
    let n = multi_bazin_n_params(k);
    let mut lo = Vec::with_capacity(n);
    let mut hi = Vec::with_capacity(n);
    let (t_min, t_max) = t_range;
    for _ in 0..k {
        lo.extend_from_slice(&[-3.0, t_min - 30.0, -2.0, -2.0]); // log_A, t0, log_tau_rise, log_tau_fall
        hi.extend_from_slice(&[3.0, t_max + 30.0, 5.0, 6.0]);
    }
    lo.push(-0.3); hi.push(0.3);   // B
    lo.push(-5.0); hi.push(0.0);   // log_sigma_extra
    (lo, hi)
}

/// Cost function for MultiBazin PSO.
fn multi_bazin_cost(
    params: &[f64],
    k: usize,
    times: &[f64],
    flux: &[f64],
    obs_var: &[f64],
    is_upper: &[bool],
    upper_flux: &[f64],
    pred_buf: &mut [f64],
) -> f64 {
    let se_idx = multi_bazin_sigma_extra_idx(k);
    let sigma_extra = params[se_idx].exp();
    let sigma_extra_sq = sigma_extra * sigma_extra;
    multi_bazin_eval_batch(params, k, times, pred_buf);
    let n = times.len().max(1) as f64;
    let mut neg_ll = 0.0;
    for i in 0..times.len() {
        let pred = pred_buf[i];
        if !pred.is_finite() {
            return 1e99;
        }
        let total_var = obs_var[i] + sigma_extra_sq;
        if is_upper[i] {
            let z = (upper_flux[i] - pred) / total_var.sqrt();
            neg_ll -= log_normal_cdf(z);
        } else {
            let diff = pred - flux[i];
            neg_ll += diff * diff / total_var + total_var.ln();
        }
    }
    neg_ll / n
}

/// MultiBazin cost (same as `multi_bazin_cost`) with an analytic gradient,
/// for the L-BFGS search paths.
struct MultiBazinCost<'a> {
    k: usize,
    times: &'a [f64],
    flux: &'a [f64],
    obs_var: &'a [f64],
    is_upper: &'a [bool],
    upper_flux: &'a [f64],
}

impl LocalObjective for MultiBazinCost<'_> {
    fn n_obs(&self) -> usize {
        self.times.len()
    }

    fn value(&self, p: &[f64], buf: &mut [f64]) -> f64 {
        multi_bazin_cost(p, self.k, self.times, self.flux, self.obs_var, self.is_upper, self.upper_flux, buf)
    }

    fn value_and_grad(&self, p: &[f64], grad: &mut [f64], dpred: &mut [f64]) -> f64 {
        let k = self.k;
        let b_idx = k * MULTI_BAZIN_COMP_PARAMS;
        let se_idx = b_idx + 1;
        let sigma_extra_sq = (2.0 * p[se_idx]).exp();
        let n = self.times.len().max(1) as f64;
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        let mut comps = [(0.0, 0.0, 0.0, 0.0); MULTI_BAZIN_MAX_K];
        for c in 0..k {
            let off = c * MULTI_BAZIN_COMP_PARAMS;
            comps[c] = (p[off].exp(), p[off + 1], p[off + 2].exp(), p[off + 3].exp());
        }
        let mut neg_ll = 0.0;
        for i in 0..self.times.len() {
            let t = self.times[i];
            // Prediction and d(pred)/d(component params)
            let mut pred = p[b_idx];
            for (c, &(a, t0, tau_rise, tau_fall)) in comps[..k].iter().enumerate() {
                let off = c * MULTI_BAZIN_COMP_PARAMS;
                let dt = t - t0;
                let sig = 1.0 / (1.0 + (-dt / tau_rise).exp());
                let f = a * (-dt / tau_fall).exp() * sig;
                pred += f;
                dpred[off] = f;
                dpred[off + 1] = f * (1.0 / tau_fall - (1.0 - sig) / tau_rise);
                dpred[off + 2] = -f * (1.0 - sig) * dt / tau_rise;
                dpred[off + 3] = f * dt / tau_fall;
            }
            dpred[b_idx] = 1.0;
            if !pred.is_finite() || dpred[..b_idx].iter().any(|d| !d.is_finite()) {
                for g in grad.iter_mut() {
                    *g = 0.0;
                }
                return 1e99;
            }
            let total_var = self.obs_var[i] + sigma_extra_sq;
            let (scale, d_var) = if self.is_upper[i] {
                let sqrt_var = total_var.sqrt();
                let z = (self.upper_flux[i] - pred) / sqrt_var;
                neg_ll -= log_normal_cdf(z);
                let ratio = normal_pdf_over_cdf(z);
                (ratio / sqrt_var, ratio * z / (2.0 * total_var))
            } else {
                let diff = pred - self.flux[i];
                neg_ll += diff * diff / total_var + total_var.ln();
                (2.0 * diff / total_var, -diff * diff / (total_var * total_var) + 1.0 / total_var)
            };
            for j in 0..=b_idx {
                grad[j] += scale * dpred[j];
            }
            grad[se_idx] += d_var * 2.0 * sigma_extra_sq;
        }
        for g in grad.iter_mut() {
            *g /= n;
        }
        neg_ll / n
    }
}

/// Starting points for the L-BFGS MultiBazin search. K=1: the Bazin fit and
/// pulses ending at the brightest detection; K>1: the greedy residual seed with a few
/// rise/fall times for the new component.
fn multi_bazin_seeds(data: &BandFitData, k: usize, init: Option<&[f64]>, bazin_fit: Option<&[f64]>, lo: &[f64], hi: &[f64]) -> Vec<Vec<f64>> {
    let clamp = |mut v: Vec<f64>| {
        for (i, x) in v.iter_mut().enumerate() {
            *x = x.clamp(lo[i], hi[i]);
        }
        v
    };
    let mut seeds = Vec::new();
    if k == 1 {
        if let Some(b) = bazin_fit {
            // Bazin params are [log_A, B, t0, log_tau_rise, log_tau_fall, log_se]
            seeds.push(clamp(vec![b[0], b[2], b[3], b[4], b[1], b[5]]));
        }
        let peak = (0..data.times.len())
            .filter(|&i| !data.is_upper[i])
            .max_by(|&a, &b| data.flux[a].partial_cmp(&data.flux[b]).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(pi) = peak {
            let (t_peak, log_peak) = (data.times[pi], data.flux[pi].max(1e-3).ln());
            for (dt, log_rise, log_fall) in [(-5.0, 0.5, 2.5), (-15.0, 1.5, 3.5), (0.0, 0.0, 1.5)] {
                // Peak of exp(-dt/tf)·sigmoid(dt/tr) is ≈ the pulse amplitude; lift log_A by ~ln 2
                seeds.push(clamp(vec![log_peak + 0.7, t_peak + dt, log_rise, log_fall, 0.0, -3.0]));
            }
        }
    } else if let Some(init) = init {
        let off = (k - 1) * MULTI_BAZIN_COMP_PARAMS;
        seeds.push(init.to_vec());
        for (log_rise, log_fall) in [(0.0, 2.0), (1.5, 3.0)] {
            let mut s = init.to_vec();
            s[off + 2] = log_rise;
            s[off + 3] = log_fall;
            // Bazin peaks ~one rise time after t0
            s[off + 1] -= log_rise.exp();
            seeds.push(clamp(s));
        }
    }
    seeds
}

/// L-BFGS search for one K from every seed; PSO if there are no seeds
/// (e.g. a band with no detections and no Bazin fit).
fn multi_bazin_local_search(
    problem: &MultiBazinCost<'_>,
    lo: &[f64],
    hi: &[f64],
    seeds: &[Vec<f64>],
) -> (Vec<f64>, f64) {
    if seeds.is_empty() {
        let mut buf = vec![0.0; problem.n_obs()];
        return pso_minimize(|p: &[f64]| problem.value(p, &mut buf), lo, hi, 30, 60, 12, 42, None);
    }
    let mut best = (Vec::new(), f64::INFINITY);
    for seed in seeds {
        // Greedy seeds start near a fit; K>1 polishes are capped tighter
        let max_iters = if problem.k == 1 { 100 } else { 60 };
        let (x, c) = projected_lbfgs(problem, seed, lo, hi, max_iters);
        if c < best.1 {
            best = (x, c);
        }
    }
    best
}

/// Greedy MultiBazin fit: fit K=1..max_k, each time initializing the new
/// component from the residual of the previous best fit.
/// `bazin_fit` (Bazin-model params, if the band was already fit with Bazin)
/// seeds K=1.
fn fit_multi_bazin(data: &BandFitData, bazin_fit: Option<&[f64]>) -> MultiBazinResult {
    let n_obs = data.times.len();
    if n_obs < 5 {
        return MultiBazinResult {
            best_k: 1,
            params: Vec::new(),
            cost: f64::NAN,
            bic: f64::NAN,
            per_k_cost: Vec::new(),
            per_k_bic: Vec::new(),
        };
    }

    let t_min = data.times.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = data.times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let t_range = (t_min, t_max);
    let n_obs_f = n_obs as f64;
    let ln_n = n_obs_f.ln();

    let mut pred_buf = vec![0.0; n_obs];
    let mut residual_buf = vec![0.0; n_obs];

    let mut best_k = 1;
    let mut best_params: Vec<f64> = Vec::new();
    let mut best_cost = f64::INFINITY;
    let mut best_bic = f64::INFINITY;
    let mut per_k_cost = Vec::with_capacity(MULTI_BAZIN_MAX_K);
    let mut per_k_bic = Vec::with_capacity(MULTI_BAZIN_MAX_K);

    // Previous fit params used to seed next K
    let mut prev_params: Vec<f64> = Vec::new();

    for k in 1..=MULTI_BAZIN_MAX_K {
        let np = multi_bazin_n_params(k);
        let (lo, hi) = multi_bazin_bounds(k, t_range);

        // Build initial guess by extending the previous fit
        let init: Option<Vec<f64>> = if k == 1 {
            None // pure PSO
        } else if prev_params.len() == multi_bazin_n_params(k - 1) {
            // Copy previous components, add a new component seeded from residual
            let mut init = Vec::with_capacity(np);
            let prev_n_comp_params = (k - 1) * MULTI_BAZIN_COMP_PARAMS;
            init.extend_from_slice(&prev_params[..prev_n_comp_params]);

            // Compute residuals from previous fit
            multi_bazin_eval_batch(&prev_params, k - 1, &data.times, &mut pred_buf);
            for i in 0..n_obs {
                residual_buf[i] = data.flux[i] - pred_buf[i];
            }

            // Find peak residual to seed new component
            let mut peak_idx = 0;
            let mut peak_val = f64::NEG_INFINITY;
            for i in 0..n_obs {
                if !data.is_upper[i] && residual_buf[i] > peak_val {
                    peak_val = residual_buf[i];
                    peak_idx = i;
                }
            }
            let seed_t0 = data.times[peak_idx];
            let seed_log_a = peak_val.max(1e-10).ln();
            init.extend_from_slice(&[seed_log_a, seed_t0, 1.0, 1.0]); // new component

            // Copy B and sigma_extra from previous
            let prev_b = prev_params[prev_n_comp_params];
            let prev_se = prev_params[prev_n_comp_params + 1];
            init.push(prev_b);
            init.push(prev_se);

            // Clamp to bounds
            for i in 0..np {
                init[i] = init[i].clamp(lo[i], hi[i]);
            }
            Some(init)
        } else {
            None
        };

        let problem = MultiBazinCost {
            k,
            times: &data.times,
            flux: &data.flux,
            obs_var: &data.obs_var,
            is_upper: &data.is_upper,
            upper_flux: &data.upper_flux,
        };
        let seeds = multi_bazin_seeds(data, k, init.as_deref(), bazin_fit, &lo, &hi);
        let (params1, cost1) = multi_bazin_local_search(&problem, &lo, &hi, &seeds);

        let k_bic = 2.0 * cost1 * n_obs_f + (np as f64) * ln_n;
        per_k_cost.push(cost1);
        per_k_bic.push(k_bic);

        if k_bic < best_bic {
            best_bic = k_bic;
            best_cost = cost1;
            best_k = k;
            best_params = params1.clone();
        }

        prev_params = params1;

        // Early stop: if adding a component didn't help BIC, unlikely that more will
        if k > 1 && k_bic > per_k_bic[k - 2] + 2.0 {
            // Fill remaining slots
            for _ in (k + 1)..=MULTI_BAZIN_MAX_K {
                per_k_cost.push(f64::NAN);
                per_k_bic.push(f64::NAN);
            }
            break;
        }
    }

    MultiBazinResult {
        best_k,
        params: best_params,
        cost: best_cost,
        bic: best_bic,
        per_k_cost,
        per_k_bic,
    }
}

/// Result of MultiBazin greedy fit for a single band.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiBazinResult {
    /// Number of components selected by BIC.
    pub best_k: usize,
    /// Best-fit parameters for the selected K.
    pub params: Vec<f64>,
    /// Reduced negative log-likelihood at best K.
    pub cost: f64,
    /// BIC at best K.
    pub bic: f64,
    /// Cost for each K tried (index 0 = K=1).
    pub per_k_cost: Vec<f64>,
    /// BIC for each K tried (index 0 = K=1).
    pub per_k_bic: Vec<f64>,
}

#[derive(Clone)]
struct PsoCost<'a> {
    times: &'a [f64],
    flux: &'a [f64],
    #[allow(dead_code)]
    flux_err: &'a [f64],
    obs_var: &'a [f64],
    is_upper: &'a [bool],
    upper_flux: &'a [f64],
    model: SviModel,
    /// Population priors: (center, width) per parameter. Empty = no prior.
    pop_priors: Vec<(f64, f64)>,
}

impl PsoCost<'_> {
    /// Villar physical constraint (JAX-calibrated, Sushant/villar-pso): the
    /// penalty cost for parameters that violate it, `None` otherwise.
    #[inline]
    fn villar_penalty(&self, p: &[f64]) -> Option<f64> {
        if self.model != SviModel::Villar {
            return None;
        }
        // Params: [log_A, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
        let constraint = villar_constraint(p[2], p[1], p[4], p[5]);
        (constraint > 0.0).then(|| 1e12 + 10_000.0 * constraint)
    }

    /// Hot-path cost evaluation that takes a slice (no Vec allocation) and
    /// reuses a prediction buffer.
    #[inline]
    fn cost_from_slice(&self, p: &[f64], pred_buf: &mut [f64]) -> f64 {
        if let Some(penalty) = self.villar_penalty(p) {
            return penalty;
        }
        eval_model_batch_into(self.model, p, self.times, pred_buf);

        apply_metzger_kn_scale(self.model, &mut pred_buf[..self.times.len()], self.is_upper);

        self.cost_from_preds(p, pred_buf)
    }

    /// Cost given precomputed model predictions for `p`.
    #[inline]
    fn cost_from_preds(&self, p: &[f64], preds: &[f64]) -> f64 {
        if let Some(penalty) = self.villar_penalty(p) {
            return penalty;
        }
        let se_idx = self.model.sigma_extra_idx();
        let sigma_extra = p[se_idx].exp();
        let sigma_extra_sq = sigma_extra * sigma_extra;
        let n = self.times.len().max(1) as f64;
        let mut neg_ll = 0.0;
        for i in 0..self.times.len() {
            let pred = preds[i];
            if !pred.is_finite() {
                return 1e99;
            }
            let total_var = self.obs_var[i] + sigma_extra_sq;
            if self.is_upper[i] {
                let z = (self.upper_flux[i] - pred) / total_var.sqrt();
                neg_ll -= log_normal_cdf(z);
            } else {
                let diff = pred - self.flux[i];
                neg_ll += diff * diff / total_var + total_var.ln();
            }
        }

        // Population prior penalty (Gaussian, added to NLL/n so it scales
        // with data size — prior influence decreases with more observations).
        if !self.pop_priors.is_empty() {
            let mut neg_lp = 0.0;
            for (j, &(center, width)) in self.pop_priors.iter().enumerate() {
                if j < p.len() && width > 0.0 {
                    let z = (p[j] - center) / width;
                    neg_lp += 0.5 * z * z;
                }
            }
            neg_ll += neg_lp / n;
        }

        neg_ll / n
    }
}

impl CostFunction for PsoCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        let mut preds = vec![0.0; self.times.len()];
        Ok(self.cost_from_slice(p, &mut preds))
    }
}

impl PsoCost<'_> {
    /// Cost and gradient in one pass over the data (analytic models only;
    /// MetzgerKN's output rescaling is not differentiated here).
    fn cost_and_grad(&self, p: &[f64], grad: &mut [f64], model_grad: &mut [f64]) -> f64 {
        let n_obs = self.times.len();
        let n_params = p.len();
        let n = n_obs.max(1) as f64;

        let se_idx = self.model.sigma_extra_idx();
        let sigma_extra = p[se_idx].exp();
        let sigma_extra_sq = sigma_extra * sigma_extra;

        for g in grad.iter_mut() {
            *g = 0.0;
        }
        // Per-call constants of the Bazin model, hoisted out of the obs loop
        let bazin_consts = (self.model == SviModel::Bazin).then(|| (p[0].exp(), p[3].exp(), p[4].exp()));
        let mut neg_ll = 0.0;

        for i in 0..n_obs {
            let pred = match bazin_consts {
                Some((a, tau_rise, tau_fall)) => bazin_eval_and_grad_with(a, p[1], p[2], tau_rise, tau_fall, self.times[i], model_grad),
                None => eval_model_and_grad(self.model, p, self.times[i], model_grad),
            };
            if !pred.is_finite() {
                // Zero gradient for non-finite predictions
                for g in grad.iter_mut() {
                    *g = 0.0;
                }
                return 1e99;
            }
            let total_var = self.obs_var[i] + sigma_extra_sq;

            if self.is_upper[i] {
                // Gradient of -log(Phi(z)) where z = (upper - pred) / sqrt(total_var)
                let sqrt_var = total_var.sqrt();
                let z = (self.upper_flux[i] - pred) / sqrt_var;
                neg_ll -= log_normal_cdf(z);
                // d/dz [-log(Phi(z))] = -phi(z)/Phi(z)
                let ratio = normal_pdf_over_cdf(z);
                // dz/dpred = -1/sqrt(total_var), dz/dse = z * sigma_extra_sq / total_var
                for j in 0..n_params {
                    grad[j] += ratio * model_grad[j] / sqrt_var; // -(-1/sqrt_var) = +
                }
                // sigma_extra gradient
                grad[se_idx] += ratio * z * sigma_extra_sq / total_var;
            } else {
                let diff = pred - self.flux[i];
                neg_ll += diff * diff / total_var + total_var.ln();
                // d/dp [diff²/total_var + ln(total_var)] = 2*diff/total_var * dpred/dp
                let scale = 2.0 * diff / total_var;
                for j in 0..n_params {
                    grad[j] += scale * model_grad[j];
                }
                // sigma_extra gradient: d/d(log_se) = d/dse * se
                // d/dse [diff²/(obs+se²) + ln(obs+se²)] = -2*se*diff²/(total_var)² + 2*se/total_var
                // multiply by se for d/d(log_se)
                let dvar_dlse = 2.0 * sigma_extra_sq; // d(sigma_extra²)/d(log_sigma_extra)
                grad[se_idx] += (-diff * diff / (total_var * total_var) + 1.0 / total_var) * dvar_dlse;
            }
        }

        // Population prior penalty (matches cost_from_preds: weighted by 1/n
        // before the final /n)
        for (j, &(center, width)) in self.pop_priors.iter().enumerate() {
            if j < n_params && width > 0.0 {
                let z = (p[j] - center) / width;
                neg_ll += 0.5 * z * z / n;
                grad[j] += z / (width * n);
            }
        }

        for g in grad.iter_mut() {
            *g /= n;
        }
        neg_ll / n
    }
}

impl Gradient for PsoCost<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let mut grad = vec![0.0; p.len()];
        let mut model_grad = vec![0.0; p.len()];
        self.cost_and_grad(p, &mut grad, &mut model_grad);
        Ok(grad)
    }
}

/// Objective for the box-bounded local and multi-start optimizers.
trait LocalObjective {
    /// Length of the scratch buffer `value` needs (number of observations).
    fn n_obs(&self) -> usize;
    /// Cost exactly as the global search sees it.
    fn value(&self, p: &[f64], buf: &mut [f64]) -> f64;
    /// Cost and gradient; `scratch` has the length of `p`.
    fn value_and_grad(&self, p: &[f64], grad: &mut [f64], scratch: &mut [f64]) -> f64;
}

impl LocalObjective for PsoCost<'_> {
    fn n_obs(&self) -> usize {
        self.times.len()
    }
    fn value(&self, p: &[f64], buf: &mut [f64]) -> f64 {
        self.cost_from_slice(p, buf)
    }
    fn value_and_grad(&self, p: &[f64], grad: &mut [f64], scratch: &mut [f64]) -> f64 {
        self.cost_and_grad(p, grad, scratch)
    }
}

/// Bounded L-BFGS with projection onto the box and Armijo backtracking.
///
/// Variables at a bound whose gradient points outward are frozen for the
/// step; the search direction is the two-loop L-BFGS recursion on the rest.
fn projected_lbfgs<P: LocalObjective>(
    problem: &P,
    x0: &[f64],
    lower: &[f64],
    upper: &[f64],
    max_iters: usize,
) -> (Vec<f64>, f64) {
    const M: usize = 6;
    let dim = x0.len();
    let mut x: Vec<f64> = (0..dim).map(|d| x0[d].clamp(lower[d], upper[d])).collect();
    let mut g = vec![0.0; dim];
    let mut model_grad = vec![0.0; dim];
    let mut f = problem.value_and_grad(&x, &mut g, &mut model_grad);
    if !f.is_finite() || f >= 1e99 {
        return (x, f);
    }

    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(M);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(M);
    let mut free = vec![true; dim];
    let mut dir = vec![0.0; dim];
    let mut x_new = vec![0.0; dim];
    let mut g_new = vec![0.0; dim];
    let mut alpha_buf = [0.0; M];

    for _ in 0..max_iters {
        let mut pg_norm = 0.0f64;
        for d in 0..dim {
            free[d] = upper[d] > lower[d]
                && !(x[d] <= lower[d] && g[d] > 0.0)
                && !(x[d] >= upper[d] && g[d] < 0.0);
            if free[d] {
                pg_norm = pg_norm.max(g[d].abs());
            }
        }
        if pg_norm < 1e-8 {
            break;
        }

        // Two-loop recursion on free variables
        for d in 0..dim {
            dir[d] = if free[d] { -g[d] } else { 0.0 };
        }
        let k = s_hist.len();
        for h in (0..k).rev() {
            let (sh, yh) = (&s_hist[h], &y_hist[h]);
            let sy: f64 = (0..dim).filter(|&d| free[d]).map(|d| sh[d] * yh[d]).sum();
            if sy <= 1e-12 {
                alpha_buf[h] = 0.0;
                continue;
            }
            let a = (0..dim).filter(|&d| free[d]).map(|d| sh[d] * dir[d]).sum::<f64>() / sy;
            alpha_buf[h] = a;
            for d in 0..dim {
                if free[d] {
                    dir[d] -= a * yh[d];
                }
            }
        }
        let gamma = if k > 0 {
            let (sh, yh) = (&s_hist[k - 1], &y_hist[k - 1]);
            let sy: f64 = (0..dim).filter(|&d| free[d]).map(|d| sh[d] * yh[d]).sum();
            let yy: f64 = (0..dim).filter(|&d| free[d]).map(|d| yh[d] * yh[d]).sum();
            if sy > 1e-12 && yy > 0.0 { sy / yy } else { 1.0 }
        } else {
            // First step: move at most ~10% of the box along the steepest coordinate
            let w = (0..dim).filter(|&d| free[d]).map(|d| upper[d] - lower[d]).fold(f64::INFINITY, f64::min);
            (0.1 * w / pg_norm).min(1.0)
        };
        for d in 0..dim {
            dir[d] *= gamma;
        }
        for h in 0..k {
            let (sh, yh) = (&s_hist[h], &y_hist[h]);
            let sy: f64 = (0..dim).filter(|&d| free[d]).map(|d| sh[d] * yh[d]).sum();
            if sy <= 1e-12 {
                continue;
            }
            let b = (0..dim).filter(|&d| free[d]).map(|d| yh[d] * dir[d]).sum::<f64>() / sy;
            for d in 0..dim {
                if free[d] {
                    dir[d] += sh[d] * (alpha_buf[h] - b);
                }
            }
        }

        let gd: f64 = (0..dim).map(|d| g[d] * dir[d]).sum();
        if gd >= 0.0 {
            // Not a descent direction: drop curvature history, use steepest descent
            s_hist.clear();
            y_hist.clear();
            for d in 0..dim {
                dir[d] = if free[d] { -g[d] } else { 0.0 };
            }
        }

        // Projected Armijo backtracking
        let mut step = 1.0;
        let mut accepted = false;
        let mut f_new = f;
        for _ in 0..30 {
            for d in 0..dim {
                x_new[d] = (x[d] + step * dir[d]).clamp(lower[d], upper[d]);
            }
            f_new = problem.value_and_grad(&x_new, &mut g_new, &mut model_grad);
            let decrease: f64 = (0..dim).map(|d| g[d] * (x_new[d] - x[d])).sum();
            if f_new.is_finite() && f_new <= f + 1e-4 * decrease {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            if s_hist.is_empty() {
                break;
            }
            s_hist.clear();
            y_hist.clear();
            continue;
        }

        let s_vec: Vec<f64> = (0..dim).map(|d| x_new[d] - x[d]).collect();
        let y_vec: Vec<f64> = (0..dim).map(|d| g_new[d] - g[d]).collect();
        if s_vec.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum::<f64>() > 1e-12 {
            if s_hist.len() == M {
                s_hist.remove(0);
                y_hist.remove(0);
            }
            s_hist.push(s_vec);
            y_hist.push(y_vec);
        }

        let rel_change = (f - f_new).abs() / f.abs().max(1.0);
        x.copy_from_slice(&x_new);
        g.copy_from_slice(&g_new);
        f = f_new;
        if rel_change < 1e-10 {
            break;
        }
    }

    // Report the cost exactly as the global search sees it
    let mut pred_buf = vec![0.0; problem.n_obs()];
    let f_check = problem.value(&x, &mut pred_buf);
    (x, f_check)
}

// ---------------------------------------------------------------------------
// L-BFGS local refinement after PSO
// ---------------------------------------------------------------------------

/// Polish PSO solution with bounded L-BFGS. Falls back to PSO result on failure.
fn lbfgs_refine(
    problem: &PsoCost<'_>,
    pso_params: Vec<f64>,
    pso_cost: f64,
    lower: &[f64],
    upper: &[f64],
) -> (Vec<f64>, f64) {
    // MetzgerKN uses numerical gradients (ODE) — skip L-BFGS
    if problem.model == SviModel::MetzgerKN {
        return (pso_params, pso_cost);
    }

    // Wrap into a bounded problem that projects back onto [lower, upper]
    let bounded = BoundedPsoCost {
        inner: problem,
        lower: lower.to_vec(),
        upper: upper.to_vec(),
    };

    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 7).with_tolerance_grad(1e-6).unwrap();

    let result = Executor::new(bounded, solver)
        .configure(|state| state.param(pso_params.clone()).max_iters(50))
        .run();

    match result {
        Ok(res) => {
            let best = res.state().best_param.clone().unwrap_or(pso_params.clone());
            // Project back to bounds
            let clamped: Vec<f64> = best.iter().enumerate()
                .map(|(d, v)| v.clamp(lower[d], upper[d]))
                .collect();
            // Verify the cost actually improved
            let mut preds = vec![0.0; problem.times.len()];
            let final_cost = problem.cost_from_slice(&clamped, &mut preds);
            if final_cost < pso_cost && final_cost.is_finite() {
                (clamped, final_cost)
            } else {
                (pso_params, pso_cost)
            }
        }
        Err(_) => (pso_params, pso_cost),
    }
}

/// Wrapper that clamps parameters to bounds during L-BFGS.
struct BoundedPsoCost<'a> {
    inner: &'a PsoCost<'a>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for BoundedPsoCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        let clamped: Vec<f64> = p.iter().enumerate()
            .map(|(d, v)| v.clamp(self.lower[d], self.upper[d]))
            .collect();
        self.inner.cost(&clamped)
    }
}

impl Gradient for BoundedPsoCost<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let clamped: Vec<f64> = p.iter().enumerate()
            .map(|(d, v)| v.clamp(self.lower[d], self.upper[d]))
            .collect();
        let mut grad = self.inner.gradient(&clamped)?;
        // Zero out gradient for dimensions at bounds (projected gradient)
        for d in 0..p.len() {
            if (p[d] <= self.lower[d] && grad[d] > 0.0) || (p[d] >= self.upper[d] && grad[d] < 0.0) {
                grad[d] = 0.0;
            }
        }
        Ok(grad)
    }
}

// ---------------------------------------------------------------------------
// Profile likelihood for t0
// ---------------------------------------------------------------------------

fn pso_bounds(model: SviModel) -> (Vec<f64>, Vec<f64>) {
    match model {
        SviModel::Bazin => (
            vec![-3.0, -0.3, -100.0, -2.0, -2.0, -5.0],
            vec![3.0, 0.3, 100.0, 5.0, 6.0, 0.0],
        ),
        SviModel::Villar => (
            // JAX-calibrated bounds (Sushant/villar-pso), converted from log10→ln
            // and kept slightly wider for single-band generality.
            // [log_A, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
            vec![-1.5, -0.02, 1.0, -60.0, -1.0, 1.0, -6.0],
            vec![1.5,  0.05,  5.0,  40.0,  3.0, 5.5,  0.0],
        ),
        SviModel::MetzgerKN => (
            vec![-3.0, -2.0, -1.0, -2.0, -5.0],
            vec![-0.5, -0.5, 2.0, 1.0, 0.0],
        ),
        SviModel::Tde => (
            vec![-3.0, -1.0, -100.0, -2.0, -1.0, 0.5, -5.0],
            vec![3.0, 1.0, 100.0, 5.0, 6.0, 4.0, 0.0],
        ),
        SviModel::Arnett => (
            vec![-3.0, -100.0, 0.5, -3.0, -5.0],
            vec![3.0, 100.0, 4.5, 3.0, 0.0],
        ),
        SviModel::Magnetar => (
            vec![-3.0, -100.0, 0.0, 0.5, -5.0],
            vec![3.0, 100.0, 6.0, 4.5, 0.0],
        ),
        SviModel::ShockCooling => (
            vec![-3.0, -100.0, 0.1, -1.0, -5.0],
            vec![3.0, 100.0, 3.0, 4.0, 0.0],
        ),
        SviModel::Afterglow => (
            vec![-3.0, -100.0, -2.0, -2.0, 0.5, -5.0],
            vec![3.0, 100.0, 6.0, 3.0, 5.0, 0.0],
        ),
    }
}

/// Population-informed priors for PSO cost regularization.
///
/// Returns (center, width) for each parameter. Based on the Villar model
/// population posteriors from Kenworthy+ 2024 (superphot+), Table 2.
/// For non-Villar models, returns broad uninformative priors.
///
/// All values are in the internal parameterization (natural log for log-params).
fn population_priors(model: SviModel) -> Vec<(f64, f64)> {
    let ln10 = std::f64::consts::LN_10;
    match model {
        SviModel::Villar => {
            // JAX-calibrated priors (Sushant/villar-pso): log10 means/stds → ln
            // [log_A, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
            vec![
                (-0.009 * ln10, 0.10 * ln10),   // log_A: JAX μ=-0.009, σ=0.10 in log10
                (0.009, 0.010),                   // beta: linear
                (1.327 * ln10, 0.40 * ln10),     // log_gamma: JAX μ=1.327, σ=0.40 in log10
                (-12.0, 15.0),                    // t0: JAX μ=-12, σ=15
                (0.360 * ln10, 0.35 * ln10),     // log_tau_rise: JAX μ=0.36, σ=0.35 in log10
                (1.454 * ln10, 0.40 * ln10),     // log_tau_fall: JAX μ=1.454, σ=0.40 in log10
                (-1.033 * ln10, 0.50 * ln10),    // log_sigma_extra: JAX μ=-1.033, σ=0.50 in log10
            ]
        }
        SviModel::Bazin => {
            // Bazin: [log_A, B, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
            vec![
                (0.0, 2.0),    // log_A
                (0.0, 0.2),    // B (baseline)
                (0.0, 30.0),   // t0
                (1.5, 2.5),    // log_tau_rise
                (3.0, 2.5),    // log_tau_fall
                (-3.0, 2.0),   // log_sigma_extra
            ]
        }
        // Other models: return broad priors (no population info)
        _ => {
            let np = model.n_params();
            vec![(0.0, 5.0); np]
        }
    }
}

/// Population priors as used in the fitting cost.
///
/// The t0 entry is disabled (width 0): t0 is in band time (days since the
/// source's first observation), so a fixed-center prior would pull it toward
/// t = 0. t0 is constrained by `data_t0_window` instead.
fn population_priors_for_fit(model: SviModel) -> Vec<(f64, f64)> {
    let mut priors = population_priors(model);
    priors[model.t0_idx()].1 = 0.0;
    priors
}

/// Public accessor for GPU PSO to use population priors.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn population_priors_for_gpu(model: crate::gpu_types::GpuModelName) -> Vec<(f64, f64)> {
    population_priors(SviModel::from_name(&model.to_svi_name()))
}

/// Data-informed t0 bounds for one band.
///
/// t0 may be up to 30 days before the first observation (rise not observed)
/// but no more than 100 days before the peak, and at most 10 days after the
/// peak (peak may be noisy). Band times are offset from the source's first
/// observation in any band, so they can start well past 100 days; this window
/// replaces the static t0 range in `pso_bounds` rather than intersecting it.
fn data_t0_window(times: &[f64], flux: &[f64], is_upper: &[bool]) -> Option<(f64, f64)> {
    if times.is_empty() {
        return None;
    }
    let mut peak_time = times[0];
    let mut peak_flux = f64::NEG_INFINITY;
    for i in 0..times.len() {
        if !is_upper[i] && flux[i] > peak_flux {
            peak_flux = flux[i];
            peak_time = times[i];
        }
    }
    let t_first = times.iter().cloned().fold(f64::INFINITY, f64::min);
    Some(((t_first - 30.0).max(peak_time - 100.0), peak_time + 10.0))
}

fn pso_bounds_no_t0(model: SviModel) -> (Vec<f64>, Vec<f64>) {
    let (mut lower, mut upper) = pso_bounds(model);
    let idx = model.t0_idx();
    lower.remove(idx);
    upper.remove(idx);
    (lower, upper)
}

// ---------------------------------------------------------------------------
// PSO with linearly-decaying inertia, velocity clamping, and wall absorption
// ---------------------------------------------------------------------------

/// Run particle-swarm optimisation with:
/// - Linear inertia decay (w_max → w_min) for exploration→exploitation;
/// - Velocity clamping to V_max = fraction of domain width per dimension;
/// - Wall absorption: velocity zeroed on dimensions that hit bounds;
/// - Stall-based early stopping (relative improvement threshold);
/// - Optional seed particle placed at index 0.
fn pso_minimize(
    mut cost_fn: impl FnMut(&[f64]) -> f64,
    lower: &[f64],
    upper: &[f64],
    n_particles: usize,
    max_iters: usize,
    stall_iters: usize,
    seed: u64,
    seed_particle: Option<&[f64]>,
) -> (Vec<f64>, f64) {
    let dim = lower.len();
    let mut rng = SmallRng::seed_from_u64(seed);

    // PSO hyper-parameters
    let w_max = 0.9; // initial inertia (exploratory)
    let w_min = 0.4; // final inertia (exploitative)
    let c1 = 1.5; // cognitive
    let c2 = 1.5; // social

    // Velocity clamp: half the domain width per dimension
    let v_max: Vec<f64> = (0..dim).map(|d| 0.5 * (upper[d] - lower[d])).collect();

    // Flat storage: [n_particles * dim]
    let total = n_particles * dim;
    let mut positions = vec![0.0; total];
    let mut velocities = vec![0.0; total];
    let mut pbest_pos = vec![0.0; total];
    let mut pbest_cost = vec![f64::INFINITY; n_particles];

    let mut gbest_pos = vec![0.0; dim];
    let mut gbest_cost = f64::INFINITY;

    // Initialise particles uniformly in bounds. The two arms draw from the RNG
    // in different orders; each is kept so fits stay reproducible.
    for p in 0..n_particles {
        let base = p * dim;
        match seed_particle {
            Some(sp) => {
                if p == 0 {
                    positions[base..base + dim].copy_from_slice(sp);
                } else {
                    for d in 0..dim {
                        positions[base + d] = lower[d] + rng.random::<f64>() * (upper[d] - lower[d]);
                    }
                }
                for d in 0..dim {
                    velocities[base + d] =
                        v_max[d] * 0.2 * (2.0 * rng.random::<f64>() - 1.0);
                }
            }
            None => {
                for d in 0..dim {
                    positions[base + d] = lower[d] + rng.random::<f64>() * (upper[d] - lower[d]);
                    velocities[base + d] =
                        v_max[d] * 0.2 * (2.0 * rng.random::<f64>() - 1.0);
                }
            }
        }
        let cost = cost_fn(&positions[base..base + dim]);
        pbest_cost[p] = cost;
        pbest_pos[base..base + dim].copy_from_slice(&positions[base..base + dim]);
        if cost < gbest_cost {
            gbest_cost = cost;
            gbest_pos.copy_from_slice(&positions[base..base + dim]);
        }
    }

    let mut iters_without_improvement = 0usize;
    let mut prev_gbest = gbest_cost;
    let inv_max_iters = 1.0 / max_iters as f64;

    for iter in 0..max_iters {
        // Linearly decay inertia weight
        let w = w_max - (w_max - w_min) * (iter as f64) * inv_max_iters;

        for p in 0..n_particles {
            let base = p * dim;

            // Update velocity with clamping and wall absorption
            for d in 0..dim {
                let r1: f64 = rng.random();
                let r2: f64 = rng.random();
                let mut v = w * velocities[base + d]
                    + c1 * r1 * (pbest_pos[base + d] - positions[base + d])
                    + c2 * r2 * (gbest_pos[d] - positions[base + d]);

                // Clamp velocity magnitude
                v = v.clamp(-v_max[d], v_max[d]);

                let new_pos = positions[base + d] + v;

                // Wall absorption: if hitting a bound, clamp position and zero velocity
                if new_pos <= lower[d] {
                    positions[base + d] = lower[d];
                    velocities[base + d] = 0.0;
                } else if new_pos >= upper[d] {
                    positions[base + d] = upper[d];
                    velocities[base + d] = 0.0;
                } else {
                    positions[base + d] = new_pos;
                    velocities[base + d] = v;
                }
            }

            let cost = cost_fn(&positions[base..base + dim]);
            if cost < pbest_cost[p] {
                pbest_cost[p] = cost;
                pbest_pos[base..base + dim]
                    .copy_from_slice(&positions[base..base + dim]);
                if cost < gbest_cost {
                    gbest_cost = cost;
                    gbest_pos.copy_from_slice(&positions[base..base + dim]);
                }
            }
        }

        // Stall detection: check relative improvement
        let improved = prev_gbest - gbest_cost > 0.01 * prev_gbest.abs().max(1e-10);
        if improved {
            iters_without_improvement = 0;
            prev_gbest = gbest_cost;
        } else {
            iters_without_improvement += 1;
            if iters_without_improvement >= stall_iters {
                break;
            }
        }
    }

    (gbest_pos, gbest_cost)
}

// ---------------------------------------------------------------------------
// Multi-start L-BFGS (alternative to PSO)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Profiled-amplitude t0 grid
// ---------------------------------------------------------------------------

// Search sizes for `profile_grid_fit`, tuned on real ZTF bands against a
// heavy-search reference optimum.
/// Number of t0 grid points across the t0 bounds.
const PG_N_T0: usize = 8;
/// Shape-parameter screen points per t0, per shape dimension.
const PG_N_SHAPE: usize = 8;
/// Number of t0 bins refined with L-BFGS.
const PG_N_STARTS: usize = 4;
/// Best shapes so far, re-evaluated at every later t0.
const PG_N_CARRY: usize = 2;
/// Best bins whose t0 is re-scanned at ±1/3 grid spacing before refinement.
const PG_N_ZOOM: usize = 2;
/// Distinct shape candidates kept per t0 (separate shape basins).
const PG_PER_BIN: usize = 2;

/// Index of the linear baseline parameter, for models that have one.
fn baseline_idx(model: SviModel) -> Option<usize> {
    match model {
        SviModel::Bazin | SviModel::Tde => Some(1),
        _ => None,
    }
}

/// Set log_A (index 0), the baseline and log_sigma_extra of `p` to their
/// best values given unit-amplitude shape predictions, and return the cost.
///
/// Every analytic model is `exp(log_A) * shape(t) (+ B)`, so for fixed shape
/// the amplitude and baseline are a weighted least-squares solve over the
/// detections. sigma_extra is set from the excess residual variance, and the
/// weights are updated once with it. Upper limits and priors enter only via
/// the final exact cost.
fn profile_linear_params(
    problem: &PsoCost<'_>,
    p: &mut [f64],
    shape: &[f64],
    preds: &mut [f64],
    lower: &[f64],
    upper: &[f64],
) -> f64 {
    let model = problem.model;
    let se_idx = model.sigma_extra_idx();
    let b_idx = baseline_idx(model);
    let n_obs = problem.times.len();
    if shape.iter().any(|v| !v.is_finite()) {
        return 1e99;
    }

    let mut se_sq = 0.0;
    let (mut amp, mut base) = (1.0, 0.0);
    for _ in 0..2 {
        // Weighted normal equations for flux ≈ amp * shape + base
        let (mut sw, mut ss, mut s1, mut sy, mut s1y) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..n_obs {
            if problem.is_upper[i] {
                continue;
            }
            let w = 1.0 / (problem.obs_var[i] + se_sq);
            let (x, y) = (shape[i], problem.flux[i]);
            sw += w;
            ss += w * x * x;
            s1 += w * x;
            sy += w * x * y;
            s1y += w * y;
        }
        if ss <= 0.0 {
            return 1e99;
        }
        amp = sy / ss;
        base = 0.0;
        if let Some(bi) = b_idx {
            let det = ss * sw - s1 * s1;
            if det > 1e-12 * ss * sw {
                base = ((ss * s1y - s1 * sy) / det).clamp(lower[bi], upper[bi]);
            }
            amp = (sy - base * s1) / ss;
        }
        let log_a = if amp > 0.0 { amp.ln() } else { lower[0] };
        amp = log_a.clamp(lower[0], upper[0]).exp();

        // Excess scatter beyond the quoted errors
        let (mut excess, mut n_det) = (0.0, 0usize);
        for i in 0..n_obs {
            if !problem.is_upper[i] {
                let r = problem.flux[i] - (amp * shape[i] + base);
                excess += r * r - problem.obs_var[i];
                n_det += 1;
            }
        }
        se_sq = (excess / n_det.max(1) as f64).max(1e-12);
        se_sq = (0.5 * se_sq.ln()).clamp(lower[se_idx], upper[se_idx]).exp().powi(2);
    }

    p[0] = amp.ln();
    if let Some(bi) = b_idx {
        p[bi] = base;
    }
    p[se_idx] = 0.5 * se_sq.ln();
    for i in 0..n_obs {
        preds[i] = amp * shape[i] + base;
    }
    problem.cost_from_preds(p, preds)
}

/// Global search exploiting model structure (analytic models only):
/// - Amplitude, baseline and sigma_extra are never sampled: each candidate
///   gets its best values from `profile_linear_params`;
/// - t0 (the main source of multimodality) is scanned on a regular grid, and
///   the remaining shape parameters are Latin-hypercube screened at each t0;
/// - Bounded L-BFGS on all parameters from the best `PG_N_STARTS` t0 bins.
fn profile_grid_fit(
    problem: &PsoCost<'_>,
    lower: &[f64],
    upper: &[f64],
    seed_points: &[&[f64]],
    seed: u64,
) -> (Vec<f64>, f64) {
    let model = problem.model;
    let dim = lower.len();
    let t0_idx = model.t0_idx();
    let se_idx = model.sigma_extra_idx();
    let b_idx = baseline_idx(model);
    let shape_dims: Vec<usize> = (1..dim)
        .filter(|&d| d != t0_idx && d != se_idx && Some(d) != b_idx)
        .collect();
    let n_obs = problem.times.len();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut shape = vec![0.0; n_obs];
    let mut preds = vec![0.0; n_obs];

    // Evaluate unit-amplitude shape, then profile the linear parameters
    let mut score = |p: &mut Vec<f64>| -> f64 {
        p[0] = 0.0;
        if let Some(bi) = b_idx {
            p[bi] = 0.0;
        }
        eval_model_batch_into(model, p, problem.times, &mut shape);
        profile_linear_params(problem, p, &shape, &mut preds, lower, upper)
    };

    // Best candidate per t0 bin (runner-up shapes go to `extra`, which is
    // not zoomed); seed points form extra bins
    let (t0_lo, t0_hi) = (lower[t0_idx], upper[t0_idx]);
    let n_t0 = if t0_hi - t0_lo > 1e-9 { PG_N_T0 } else { 1 };
    let mut bins: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n_t0 + seed_points.len());
    for sp in seed_points {
        let mut x: Vec<f64> = (0..dim).map(|d| sp[d].clamp(lower[d], upper[d])).collect();
        let c = score(&mut x);
        bins.push((x, c));
    }
    let n_shape = (PG_N_SHAPE * shape_dims.len()).max(1);
    let spacing = (t0_hi - t0_lo) / n_t0 as f64;
    // Best shape-parameter vectors so far (cost, values over shape_dims)
    let mut elite: Vec<(f64, Vec<f64>)> = Vec::with_capacity(PG_N_CARRY + 1);
    let mut extra: Vec<(Vec<f64>, f64)> = Vec::new();
    for k in 0..n_t0 {
        let t0 = t0_lo + spacing * (k as f64 + 0.5);
        // Latin hypercube over the shape dimensions at this t0
        let strata: Vec<Vec<usize>> = shape_dims
            .iter()
            .map(|_| {
                let mut perm: Vec<usize> = (0..n_shape).collect();
                for i in (1..n_shape).rev() {
                    let j = rng.random_range(0..=i);
                    perm.swap(i, j);
                }
                perm
            })
            .collect();
        let mut screened: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n_shape + elite.len());
        let mut x = vec![0.0; dim];
        x[t0_idx] = t0;
        for i in 0..n_shape {
            for (si, &d) in shape_dims.iter().enumerate() {
                let u = (strata[si][i] as f64 + rng.random::<f64>()) / n_shape as f64;
                x[d] = lower[d] + u * (upper[d] - lower[d]);
            }
            let c = score(&mut x);
            screened.push((x.clone(), c));
        }
        for (_, sh) in &elite {
            for (si, &d) in shape_dims.iter().enumerate() {
                x[d] = sh[si];
            }
            let c = score(&mut x);
            screened.push((x.clone(), c));
        }
        screened.retain(|c| c.1.is_finite());
        screened.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Up to `per_bin` candidates whose shapes differ by >25% of the box
        let mut kept: Vec<(Vec<f64>, f64)> = Vec::with_capacity(PG_PER_BIN);
        for cand in screened {
            if kept.len() >= PG_PER_BIN {
                break;
            }
            let distinct = kept.iter().all(|k| {
                shape_dims.iter().any(|&d| (cand.0[d] - k.0[d]).abs() / (upper[d] - lower[d]).max(1e-12) > 0.25)
            });
            if distinct {
                kept.push(cand);
            }
        }
        if kept.is_empty() {
            continue;
        }
        let sh: Vec<f64> = shape_dims.iter().map(|&d| kept[0].0[d]).collect();
        elite.push((kept[0].1, sh));
        elite.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        elite.truncate(PG_N_CARRY);
        let mut kept = kept.into_iter();
        bins.extend(kept.next());
        extra.extend(kept);
    }

    // Zoom: re-scan t0 at ±1/3 spacing around the best bins, keeping their shape
    if n_t0 > 1 {
        bins.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for bin in bins.iter_mut().take(PG_N_ZOOM) {
            let center = bin.0.clone();
            for off in [-spacing / 3.0, spacing / 3.0] {
                let mut x = center.clone();
                x[t0_idx] = (center[t0_idx] + off).clamp(t0_lo, t0_hi);
                let cost = score(&mut x);
                if cost < bin.1 {
                    *bin = (x, cost);
                }
            }
        }
    }
    bins.extend(extra);
    if bins.is_empty() {
        return (lower.iter().zip(upper).map(|(l, u)| 0.5 * (l + u)).collect(), 1e99);
    }
    bins.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut best = bins[0].clone();
    for (x0, _) in bins.iter().take(PG_N_STARTS) {
        let (x, c) = projected_lbfgs(problem, x0, lower, upper, 100);
        if c < best.1 {
            best = (x, c);
        }
    }
    best
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + i as f64 * step).collect()
}

fn profile_t0_search(
    data: &BandFitData,
    model: SviModel,
    pilot_params: &[f64],
    pilot_cost: f64,
) -> (Vec<f64>, f64) {
    let t0_idx = model.t0_idx();
    let pilot_t0 = pilot_params[t0_idx];
    let t_first = data.times.iter().cloned().fold(f64::INFINITY, f64::min);
    let t0_min = (pilot_t0 - 5.0).max(t_first - 10.0);
    let t0_max = pilot_t0 + 5.0;

    let run_at_t0 = |t0: f64| -> (Vec<f64>, f64) {
        if model != SviModel::MetzgerKN {
            // Pin t0 by collapsing its bounds; polish from the pilot fit
            let (mut lower, mut upper) = pso_bounds(model);
            lower[t0_idx] = t0;
            upper[t0_idx] = t0;
            let problem = PsoCost {
                times: &data.times,
                flux: &data.flux,
                flux_err: &data.flux_err,
                obs_var: &data.obs_var,
                is_upper: &data.is_upper,
                upper_flux: &data.upper_flux,
                model,
                pop_priors: Vec::new(),
            };
            let mut warm = pilot_params.to_vec();
            warm[t0_idx] = t0;
            return projected_lbfgs(&problem, &warm, &lower, &upper, 100);
        }
        let (lower, upper) = pso_bounds_no_t0(model);
        let problem = PsoCost {
            times: &data.times,
            flux: &data.flux,
            flux_err: &data.flux_err,
            obs_var: &data.obs_var,
            is_upper: &data.is_upper,
            upper_flux: &data.upper_flux,
            model,
            pop_priors: Vec::new(), // profile t0 search: no prior (t0 is being profiled)
        };
        let mut pred_buf = vec![0.0; data.times.len()];
        let mut full = vec![0.0; lower.len() + 1];
        let cost_fn = |params: &[f64]| -> f64 {
            // Reconstruct full params with t0 inserted
            let mut fi = 0;
            for (i, &val) in params.iter().enumerate() {
                if i == t0_idx {
                    full[fi] = t0;
                    fi += 1;
                }
                full[fi] = val;
                fi += 1;
            }
            if t0_idx >= params.len() {
                full[fi] = t0;
            }
            problem.cost_from_slice(&full, &mut pred_buf)
        };
        let (best_reduced, cost) = pso_minimize(
            cost_fn, &lower, &upper, 30, 60, 12, 42, None,
        );
        let mut full = Vec::with_capacity(best_reduced.len() + 1);
        for (i, &val) in best_reduced.iter().enumerate() {
            if i == t0_idx {
                full.push(t0);
            }
            full.push(val);
        }
        if t0_idx >= best_reduced.len() {
            full.push(t0);
        }
        (full, cost)
    };

    let mut best_params = pilot_params.to_vec();
    let mut best_cost = pilot_cost;
    let mut best_t0 = pilot_t0;

    for &t0 in &linspace(t0_min, t0_max, 3) {
        let (params, cost) = run_at_t0(t0);
        if cost < best_cost && !params.is_empty() {
            best_cost = cost;
            best_params = params;
            best_t0 = t0;
        }
    }

    let fine_min = (best_t0 - 0.5).max(t0_min);
    let fine_max = (best_t0 + 0.5).min(t0_max);
    for &t0 in &linspace(fine_min, fine_max, 2) {
        let (params, cost) = run_at_t0(t0);
        if cost < best_cost && !params.is_empty() {
            best_cost = cost;
            best_params = params;
        }
    }

    (best_params, best_cost)
}

// ---------------------------------------------------------------------------
// PSO model selection
// ---------------------------------------------------------------------------

/// Bazin-first early stopping threshold for PSO model selection.
///
/// If Bazin's per-observation negative log-likelihood is below this value,
/// the remaining 7 models are skipped.  For well-fit data with typical
/// photometric errors the cost lands in the range [−4, 0]; a threshold of
/// 2.0 means "Bazin is an acceptable fit" and avoids spending ~170 ms on
/// exotic models that won't win.
const BAZIN_GOOD_ENOUGH: f64 = 2.0;

fn pso_fit_single_model_t0(data: &BandFitData, model: SviModel, ref_t0: Option<f64>) -> (SviModel, Vec<f64>, f64) {
    let (mut lower, mut upper) = pso_bounds(model);
    let problem = PsoCost {
        times: &data.times,
        flux: &data.flux,
        flux_err: &data.flux_err,
        obs_var: &data.obs_var,
        is_upper: &data.is_upper,
        upper_flux: &data.upper_flux,
        model,
        pop_priors: population_priors_for_fit(model),
    };

    // --- Data-informed t0 bounds ---
    let t0_idx = model.t0_idx();
    let n_obs = data.times.len();
    if let Some((t0_lo, t0_hi)) = data_t0_window(&data.times, &data.flux, &data.is_upper) {
        lower[t0_idx] = t0_lo;
        upper[t0_idx] = t0_hi;
    }

    // clamp t0 to ±1 day around reference band if provided
    if let Some(t0_ref) = ref_t0 {
        let clamped_lo = (t0_ref - 1.0).max(lower[t0_idx]);
        let clamped_hi = (t0_ref + 1.0).min(upper[t0_idx]);
        if clamped_lo < clamped_hi {
            lower[t0_idx] = clamped_lo;
            upper[t0_idx] = clamped_hi;
        }
    }

    // Build seed particles with informed initial guesses
    let pop_prior = population_priors(model);
    let dim = lower.len();

    // Seed A: t0 just before peak flux, other params at population prior means
    let seed_at_peak: Option<Vec<f64>> = if n_obs > 0 {
        let mut peak_time = data.times[0];
        let mut peak_flux = f64::NEG_INFINITY;
        for i in 0..n_obs {
            if !data.is_upper[i] && data.flux[i] > peak_flux {
                peak_flux = data.flux[i];
                peak_time = data.times[i];
            }
        }
        let mut seed = vec![0.0; dim];
        for d in 0..dim {
            // Use population prior mean, clamped to bounds
            seed[d] = if d < pop_prior.len() {
                pop_prior[d].0.clamp(lower[d], upper[d])
            } else {
                0.5 * (lower[d] + upper[d])
            };
        }
        seed[t0_idx] = (peak_time - 5.0).clamp(lower[t0_idx], upper[t0_idx]);
        Some(seed)
    } else {
        None
    };

    let seed_points: Vec<&[f64]> = seed_at_peak.iter().map(|v| v.as_slice()).collect();

    // Analytic models: profiled-amplitude t0 grid + L-BFGS. MetzgerKN (ODE,
    // no analytic gradient) uses PSO.
    if model != SviModel::MetzgerKN {
        let (params, cost) = profile_grid_fit(&problem, &lower, &upper, &seed_points, 42);
        return (model, params, cost);
    }

    // --- Multi-restart PSO with data-informed seeding ---
    // Restart 0: standard random init
    // Restart 1: seed one particle at estimated t0 = peak_time - small_offset
    let seeds: &[u64] = &[42, 137];
    let mut best_params = Vec::new();
    let mut best_chi2 = f64::INFINITY;

    // MetzgerKN uses numerical ODE so keep more particles; analytic models use fewer
    let n_particles = if matches!(model, SviModel::MetzgerKN) { 30 } else { 20 };
    let stall = if matches!(model, SviModel::MetzgerKN) { 12 } else { 8 };

    for (i, &seed) in seeds.iter().enumerate() {
        let mut pred_buf = vec![0.0; data.times.len()];
        let cost_fn = |p: &[f64]| problem.cost_from_slice(p, &mut pred_buf);

        let (params, chi2) = match i {
            1 => {
                // Second restart: seed at peak
                if let Some(ref sp) = seed_at_peak {
                    pso_minimize(cost_fn, &lower, &upper, n_particles, 60, stall, seed, Some(sp))
                } else {
                    pso_minimize(cost_fn, &lower, &upper, n_particles, 60, stall, seed, None)
                }
            }
            _ => pso_minimize(cost_fn, &lower, &upper, n_particles, 60, stall, seed, None),
        };

        if chi2 < best_chi2 {
            best_chi2 = chi2;
            best_params = params;
        }
    }
    // L-BFGS polish: gradient-based refinement from PSO solution
    let (refined_params, refined_chi2) = lbfgs_refine(&problem, best_params, best_chi2, &lower, &upper);
    (model, refined_params, refined_chi2)
}

fn pso_model_select_t0(data: &BandFitData, fit_all_models: bool, ref_t0: Option<f64>) -> (SviModel, Vec<f64>, f64, HashMap<SviModelName, Option<f64>>, HashMap<SviModelName, Vec<f64>>) {
    let all_models: &[SviModel] = &[
        SviModel::Bazin,
        SviModel::Arnett,
        SviModel::Tde,
        SviModel::Afterglow,
        SviModel::Villar,
        SviModel::Magnetar,
        SviModel::ShockCooling,
        SviModel::MetzgerKN,
    ];

    let mut all_chi2: HashMap<SviModelName, Option<f64>> = HashMap::new();
    let mut all_params: HashMap<SviModelName, Vec<f64>> = HashMap::new();

    let n_obs = data.times.len().max(1) as f64;
    let ln_n = n_obs.ln();

    // AICc when n < 40 (handles sparse lightcurves better), BIC otherwise.
    let ic = |cost: f64, model: SviModel| -> f64 {
        let k = model.n_params() as f64;
        let total_nll = 2.0 * cost * n_obs;
        if n_obs < 40.0 && n_obs > k + 1.0 {
            total_nll + 2.0 * k + 2.0 * k * (k + 1.0) / (n_obs - k - 1.0)
        } else {
            total_nll + k * ln_n
        }
    };

    // Run Bazin first as an early-stop gate
    let (_, bazin_params, bazin_chi2) = pso_fit_single_model_t0(data, SviModel::Bazin, ref_t0);
    all_chi2.insert(SviModelName::Bazin, finite_or_none(bazin_chi2));
    if fit_all_models {
        all_params.insert(SviModelName::Bazin, bazin_params.clone());
    }

    let mut best_model = SviModel::Bazin;
    let mut best_params = bazin_params;
    let mut best_chi2 = bazin_chi2;
    let mut best_ic = ic(bazin_chi2, SviModel::Bazin);

    if !fit_all_models && bazin_chi2 < BAZIN_GOOD_ENOUGH {
        // Bazin is good enough — fill None for skipped models
        for &model in all_models {
            all_chi2.entry(model.to_name()).or_insert(None);
        }
    } else {
        // Run remaining 7 models in parallel
        let rest = &all_models[1..];
        let results: Vec<(SviModel, Vec<f64>, f64)> = rest
            .par_iter()
            .map(|&model| pso_fit_single_model_t0(data, model, ref_t0))
            .collect();

        for (model, params, chi2) in results {
            all_chi2.insert(model.to_name(), finite_or_none(chi2));
            if fit_all_models {
                all_params.insert(model.to_name(), params.clone());
            }
            let model_ic = ic(chi2, model);
            if model_ic < best_ic {
                best_ic = model_ic;
                best_chi2 = chi2;
                best_model = model;
                best_params = params;
            }
        }
    }

    (best_model, best_params, best_chi2, all_chi2, all_params)
}

// ---------------------------------------------------------------------------
// Priors and initialization
// ---------------------------------------------------------------------------

fn prior_params(model: SviModel) -> Vec<(f64, f64)> {
    match model {
        SviModel::Bazin => vec![(0.0, 2.0); 6],
        SviModel::Villar => vec![(0.0, 1.0); 7],
        SviModel::MetzgerKN => {
            vec![
                (-1.75, 1.25),
                (-1.25, 0.75),
                (0.5, 1.5),
                (-0.5, 1.5),
                (0.0, 2.0),
            ]
        }
        SviModel::Tde => {
            let mut p = vec![(0.0, 2.0); 7];
            p[5] = (1.67, 1.0);
            p
        }
        SviModel::Arnett => {
            let mut p = vec![(0.0, 2.0); 5];
            p[2] = (2.3, 1.0);
            p
        }
        SviModel::Magnetar => {
            let mut p = vec![(0.0, 2.0); 5];
            p[2] = (3.0, 1.5);
            p[3] = (2.3, 1.0);
            p
        }
        SviModel::ShockCooling => {
            let mut p = vec![(0.0, 2.0); 5];
            p[2] = (0.5, 1.0);
            p[3] = (1.0, 2.0);
            p
        }
        SviModel::Afterglow => {
            let mut p = vec![(0.0, 2.0); 6];
            p[2] = (1.0, 2.0);
            p[3] = (0.8, 1.0);
            p[4] = (2.0, 1.0);
            p
        }
    }
}

fn init_variational_means(model: SviModel, data: &BandFitData) -> Vec<f64> {
    let mut peak_val = f64::NEG_INFINITY;
    let mut peak_idx = 0;
    for (i, &f) in data.flux.iter().enumerate() {
        if f > peak_val {
            peak_val = f;
            peak_idx = i;
        }
    }
    let t_peak = data.times[peak_idx];

    match model {
        SviModel::Bazin => {
            vec![
                peak_val.max(0.01).ln(),
                0.0,
                t_peak,
                2.0_f64.ln(),
                20.0_f64.ln(),
                -3.0,
            ]
        }
        SviModel::Villar => {
            vec![
                peak_val.max(0.01).ln(),
                0.01,
                10.0_f64.ln(),
                t_peak,
                2.0_f64.ln(),
                20.0_f64.ln(),
                (-3.0_f64).ln().max(-5.0),
            ]
        }
        SviModel::MetzgerKN => {
            vec![-2.0, -0.7, 0.5, t_peak - 2.0, -3.0]
        }
        SviModel::Tde => {
            vec![
                peak_val.max(0.01).ln(),
                0.0,
                t_peak - 10.0,
                2.0_f64.ln(),
                20.0_f64.ln(),
                1.67,
                -3.0,
            ]
        }
        SviModel::Arnett => {
            vec![
                peak_val.max(0.01).ln(),
                t_peak - 15.0,
                10.0_f64.ln(),
                0.0,
                -3.0,
            ]
        }
        SviModel::Magnetar => {
            vec![
                peak_val.max(0.01).ln(),
                t_peak - 10.0,
                20.0_f64.ln(),
                10.0_f64.ln(),
                -3.0,
            ]
        }
        SviModel::ShockCooling => {
            vec![
                peak_val.max(0.01).ln(),
                t_peak - 2.0,
                0.5,
                5.0_f64.ln(),
                -3.0,
            ]
        }
        SviModel::Afterglow => {
            vec![
                peak_val.max(0.01).ln(),
                t_peak - 5.0,
                10.0_f64.ln(),
                0.8,
                2.2,
                -3.0,
            ]
        }
    }
}

// ---------------------------------------------------------------------------
// SVI fitting
// ---------------------------------------------------------------------------

struct SviFitResult {
    model: SviModel,
    mu: Vec<f64>,
    log_sigma: Vec<f64>,
    elbo: f64,
}

fn svi_fit(
    model: SviModel,
    data: &BandFitData,
    n_steps: usize,
    n_samples: usize,
    lr: f64,
    pso_init: Option<&[f64]>,
) -> SviFitResult {
    let n_params = model.n_params();
    let n_variational = 2 * n_params;
    let n_obs = data.times.len();

    let mut var_params = vec![0.0; n_variational];
    let init_mu = if let Some(pso_params) = pso_init {
        pso_params.to_vec()
    } else {
        init_variational_means(model, data)
    };
    for i in 0..n_params {
        var_params[i] = init_mu[i];
    }
    // estimate initial sigma from local curvature at PSO solution
    if pso_init.is_some() {
        let f_center = neg_log_posterior(model, data, &init_mu);
        for i in 0..n_params {
            let h = 1e-3 * init_mu[i].abs().max(0.1);
            let mut p = init_mu.clone();
            p[i] = init_mu[i] + h;
            let f_plus = neg_log_posterior(model, data, &p);
            p[i] = init_mu[i] - h;
            let f_minus = neg_log_posterior(model, data, &p);
            let curv = ((f_plus + f_minus - 2.0 * f_center) / (h * h)).max(1e-6);
            let log_sig = (1.0 / curv.sqrt()).ln();
            var_params[n_params + i] = if log_sig.is_finite() { log_sig.clamp(-6.0, 2.0) } else { -1.0 };
        }
    } else {
        for i in 0..n_params {
            var_params[n_params + i] = -1.0;
        }
    }

    let mut adam = ManualAdam::new(n_variational, lr);
    let mut rng = SmallRng::seed_from_u64(42);
    let obs_var = &data.obs_var;
    let se_idx = model.sigma_extra_idx();
    let priors = prior_params(model);

    let mut eps = vec![0.0; n_params];
    let mut theta = vec![0.0; n_params];
    let mut sigma = vec![0.0; n_params];
    let mut preds = vec![0.0; n_obs];
    let mut grad_buf = vec![0.0; n_params];
    let mut kn_grads_flat = if model.is_sequential() {
        vec![0.0; n_obs * n_params]
    } else {
        Vec::new()
    };
    let mut grad_mu = vec![0.0; n_params];
    let mut grad_log_sigma = vec![0.0; n_params];
    let mut dll_dtheta = vec![0.0; n_params];
    let mut neg_elbo_grad = vec![0.0; n_variational];

    let mut final_elbo = f64::NEG_INFINITY;

    // Early stopping: track smoothed ELBO and stop when it stalls.
    let svi_stall_iters = 30;
    let svi_min_iters = 60; // always run at least this many
    let ema_alpha = 0.2;
    let mut ema_elbo = f64::NEG_INFINITY;
    let mut best_ema_elbo = f64::NEG_INFINITY;
    let mut iters_without_improvement = 0usize;

    let base_lr = lr;
    for _step in 0..n_steps {
        // Cosine annealing LR
        adam.lr = if _step < 50 {
            base_lr * 0.1 + base_lr * 0.9 * _step as f64 / 50.0
        } else {
            let progress = (_step - 50) as f64 / (n_steps - 50).max(1) as f64;
            base_lr * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()))
        };
        let mu = &var_params[..n_params];
        let log_sigma = &var_params[n_params..];
        for j in 0..n_params {
            sigma[j] = log_sigma[j].exp();
        }

        for j in 0..n_params {
            grad_mu[j] = 0.0;
            grad_log_sigma[j] = 0.0;
        }
        let mut elbo_sum = 0.0;

        for _ in 0..n_samples {
            for j in 0..n_params {
                let u1: f64 = rng.random::<f64>().max(1e-10);
                let u2: f64 = rng.random::<f64>();
                eps[j] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                theta[j] = mu[j] + sigma[j] * eps[j];
            }

            let sigma_extra = theta[se_idx].exp();
            let sigma_extra_sq = sigma_extra * sigma_extra;

            if model.is_sequential() {
                let batch = metzger_kn_eval_batch(&theta, &data.times);
                preds.copy_from_slice(&batch);
                metzger_kn_grad_batch_into(&theta, &data.times, &mut kn_grads_flat);
            }

            if model == SviModel::MetzgerKN {
                let max_pred = preds
                    .iter()
                    .zip(data.is_upper.iter())
                    .filter(|(_, is_up)| !**is_up)
                    .map(|(p, _)| *p)
                    .fold(f64::NEG_INFINITY, f64::max);
                if max_pred > 1e-10 && max_pred.is_finite() {
                    let raw_scale = 1.0 / max_pred;
                    let scale = raw_scale.clamp(0.1, 10.0);
                    if (raw_scale - scale).abs() / raw_scale > 0.5 {
                        continue;
                    }
                    for pred in preds.iter_mut() {
                        *pred *= scale;
                    }
                    for g in kn_grads_flat.iter_mut() {
                        *g *= scale;
                    }
                }
            }

            let mut log_lik = 0.0;
            for j in 0..n_params {
                dll_dtheta[j] = 0.0;
            }
            // Per-sample constants of the Bazin model, hoisted out of the obs loop
            let bazin_consts = (model == SviModel::Bazin).then(|| (theta[0].exp(), theta[3].exp(), theta[4].exp()));

            for i in 0..n_obs {
                let pred;
                if model.is_sequential() {
                    pred = preds[i];
                    let base_off = i * n_params;
                    grad_buf.copy_from_slice(&kn_grads_flat[base_off..base_off + n_params]);
                } else if let Some((a, tau_rise, tau_fall)) = bazin_consts {
                    pred = bazin_eval_and_grad_with(a, theta[1], theta[2], tau_rise, tau_fall, data.times[i], &mut grad_buf);
                } else {
                    pred = eval_model_and_grad(model, &theta, data.times[i], &mut grad_buf);
                }
                if !pred.is_finite() {
                    continue;
                }
                let total_var = obs_var[i] + sigma_extra_sq;
                let sigma_total = total_var.sqrt();

                if data.is_upper[i] {
                    let z = (data.upper_flux[i] - pred) / sigma_total;
                    log_lik += log_normal_cdf(z);

                    let phi_z = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
                    let cdf_z =
                        (0.5 * (1.0 + erf_approx(z * std::f64::consts::FRAC_1_SQRT_2))).max(1e-300);
                    let dll_dpred = -phi_z / (cdf_z * sigma_total);

                    for j in 0..n_params {
                        if j != se_idx && grad_buf[j].is_finite() {
                            dll_dtheta[j] += dll_dpred * grad_buf[j];
                        }
                    }

                    let dz_dlse =
                        -(data.upper_flux[i] - pred) * sigma_extra_sq / (sigma_total * total_var);
                    dll_dtheta[se_idx] += (phi_z / cdf_z) * dz_dlse;
                } else {
                    let residual = data.flux[i] - pred;
                    let inv_total = 1.0 / total_var;
                    let r2 = residual * residual;
                    log_lik +=
                        -0.5 * (r2 * inv_total + (2.0 * std::f64::consts::PI * total_var).ln());

                    for j in 0..n_params {
                        if j != se_idx && grad_buf[j].is_finite() {
                            dll_dtheta[j] += residual * inv_total * grad_buf[j];
                        }
                    }

                    dll_dtheta[se_idx] += (r2 * inv_total * inv_total - inv_total) * sigma_extra_sq;
                }
            }

            // Prior centered on PSO init (not the original prior center)
            // to prevent mu from drifting away from the optimum.
            // sigma_extra keeps original prior center since PSO pins it at -5.
            let mut log_prior = 0.0;
            for j in 0..n_params {
                let (_orig_center, width) = priors[j];
                let center = init_mu[j];
                let var = width * width;
                log_prior += -0.5 * (theta[j] - center).powi(2) / var;
                let dlp = -(theta[j] - center) / var;
                let df_dtheta = dll_dtheta[j] + dlp;
                grad_mu[j] += df_dtheta;
                grad_log_sigma[j] += df_dtheta * sigma[j] * eps[j];
            }

            elbo_sum += log_lik + log_prior;
        }

        let ns = n_samples as f64;
        for j in 0..n_params {
            grad_mu[j] /= ns;
            grad_log_sigma[j] /= ns;
        }
        elbo_sum /= ns;

        let log_sigma = &var_params[n_params..];
        let entropy: f64 = log_sigma.iter().sum::<f64>()
            + 0.5 * n_params as f64 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln();
        final_elbo = elbo_sum + entropy;

        for j in 0..n_params {
            neg_elbo_grad[j] = -grad_mu[j];
            neg_elbo_grad[n_params + j] = -(grad_log_sigma[j] + 1.0);
        }

        // Prevent SVI blow up.
        let grad_norm_sq: f64 = neg_elbo_grad.iter().map(|g| g * g).sum();
        if grad_norm_sq > 100.0 {
            let scale = 10.0 / grad_norm_sq.sqrt();
            for g in neg_elbo_grad.iter_mut() {
                *g *= scale;
            }
        }

        adam.step(&mut var_params, &neg_elbo_grad);

        for i in 0..n_params {
            var_params[n_params + i] = var_params[n_params + i].clamp(-6.0, 2.0);
        }

        // Early stopping based on smoothed ELBO
        if _step >= svi_min_iters {
            if ema_elbo == f64::NEG_INFINITY {
                ema_elbo = final_elbo;
            } else {
                ema_elbo = ema_alpha * final_elbo + (1.0 - ema_alpha) * ema_elbo;
            }
            let improved = ema_elbo - best_ema_elbo > 0.01 * best_ema_elbo.abs().max(1e-10);
            if improved {
                best_ema_elbo = ema_elbo;
                iters_without_improvement = 0;
            } else {
                iters_without_improvement += 1;
                if iters_without_improvement >= svi_stall_iters {
                    break;
                }
            }
        }
    }

    let mu = var_params[..n_params].to_vec();
    let log_inflation = SIGMA_INFLATION_FACTOR.ln();
    let log_sigma: Vec<f64> = var_params[n_params..]
        .iter()
        .map(|ls| ls + log_inflation)
        .collect();

    SviFitResult {
        model,
        mu,
        log_sigma,
        elbo: final_elbo,
    }
}

// ---------------------------------------------------------------------------
// Laplace approximation
// ---------------------------------------------------------------------------

/// Unnormalized negative log-posterior: -log p(D|θ) - log p(θ).
///
/// Unlike `PsoCost::cost` this is NOT divided by n, because Laplace needs
/// the true curvature of the posterior.
fn neg_log_posterior(model: SviModel, data: &BandFitData, params: &[f64]) -> f64 {
    let se_idx = model.sigma_extra_idx();
    let sigma_extra = params[se_idx].exp();
    let sigma_extra_sq = sigma_extra * sigma_extra;
    let (preds, _) = scaled_preds(model, params, data);

    let mut neg_ll = 0.0;
    for i in 0..data.times.len() {
        let pred = preds[i];
        if !pred.is_finite() {
            return 1e99;
        }
        let total_var = data.obs_var[i] + sigma_extra_sq;
        if data.is_upper[i] {
            let z = (data.upper_flux[i] - pred) / total_var.sqrt();
            neg_ll -= log_normal_cdf(z);
        } else {
            let diff = pred - data.flux[i];
            neg_ll += 0.5 * (diff * diff / total_var + (2.0 * std::f64::consts::PI * total_var).ln());
        }
    }

    // Prior: Gaussian with per-parameter (center, width)
    let priors = prior_params(model);
    let mut neg_lp = 0.0;
    for j in 0..params.len() {
        let (center, width) = priors[j];
        let var = width * width;
        neg_lp += 0.5 * (params[j] - center).powi(2) / var;
    }

    neg_ll + neg_lp
}

/// Jacobi eigendecomposition for a symmetric matrix stored row-major.
///
/// Returns `(eigenvalues, eigenvectors_row_major)`.
/// Suitable for matrices up to ~7×7 (our max param count).
fn symmetric_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    // Initialize eigenvectors to identity
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-15 {
            break;
        }

        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let theta = if (app - aqq).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply Givens rotation
        for i in 0..n {
            let aip = a[i * n + p];
            let aiq = a[i * n + q];
            a[i * n + p] = c * aip + s * aiq;
            a[i * n + q] = -s * aip + c * aiq;
        }
        for j in 0..n {
            let apj = a[p * n + j];
            let aqj = a[q * n + j];
            a[p * n + j] = c * apj + s * aqj;
            a[q * n + j] = -s * apj + c * aqj;
        }
        // Fix diagonal and zero the off-diagonal
        a[p * n + p] = c * c * app + 2.0 * c * s * apq + s * s * aqq;
        a[q * n + q] = s * s * app - 2.0 * c * s * apq + c * c * aqq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        // Update eigenvectors
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip + s * viq;
            v[i * n + q] = -s * vip + c * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

/// Compute `log(sqrt(diag(H⁻¹)))` from the Hessian, clamping non-positive eigenvalues.
fn log_sigma_from_hessian(hessian: &[f64], n: usize) -> Vec<f64> {
    let (eigenvalues, eigenvectors) = symmetric_eigen(hessian, n);

    // Clamp eigenvalues: non-positive → large variance (conservative)
    let min_eigenvalue = 1e-6;
    let inv_eigenvalues: Vec<f64> = eigenvalues
        .iter()
        .map(|&ev| 1.0 / ev.max(min_eigenvalue))
        .collect();

    // Reconstruct diagonal of V * diag(1/λ) * V^T
    let mut diag_inv = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..n {
            let vik = eigenvectors[i * n + k];
            sum += vik * vik * inv_eigenvalues[k];
        }
        diag_inv[i] = sum;
    }

    // log(sqrt(diag)) = 0.5 * log(diag)
    diag_inv.iter().map(|&d| 0.5 * d.max(1e-30).ln()).collect()
}

/// Zero the Hessian row and column of each flagged parameter. A parameter whose
/// own curvature is non-finite then gets the wide fallback sigma from
/// `log_sigma_from_hessian`, without inflating the sigmas of the parameters it
/// was coupled to.
fn decouple_params(hessian: &mut [f64], n: usize, flagged: &[bool]) {
    for (i, &f) in flagged.iter().enumerate() {
        if f {
            for k in 0..n {
                hessian[i * n + k] = 0.0;
                hessian[k * n + i] = 0.0;
            }
        }
    }
}

/// Laplace approximation: compute Hessian of neg-log-posterior at MAP
/// estimate via central finite differences, then invert for uncertainties.
///
/// The `sigma_extra` parameter is held fixed at its MAP value when computing
/// the Hessian for the remaining ("shape") parameters.  This prevents
/// `sigma_extra` from absorbing all curvature information, which previously
/// left all shape-parameter uncertainties at the uninformative prior width.
/// The `sigma_extra` slot in the returned `log_sigma` is set from the prior.
fn laplace_fit(model: SviModel, data: &BandFitData, map_params: &[f64]) -> SviFitResult {
    let n_full = map_params.len();
    let se_idx = model.sigma_extra_idx();
    let f0 = neg_log_posterior(model, data, map_params);

    // Indices of shape parameters (everything except sigma_extra)
    let shape_idxs: Vec<usize> = (0..n_full).filter(|&i| i != se_idx).collect();
    let n_shape = shape_idxs.len();

    // Hessian via central finite differences (single step size)
    let mut hessian = vec![0.0; n_shape * n_shape];
    let mut no_curvature = vec![false; n_shape];
    let mut buf = map_params.to_vec();

    for si in 0..n_shape {
        let i = shape_idxs[si];
        let hi = 1e-4 * map_params[i].abs().max(1.0);
        for sj in si..n_shape {
            let j = shape_idxs[sj];
            let hj = 1e-4 * map_params[j].abs().max(1.0);

            // Central difference: (f(+h,+h) - f(+h,-h) - f(-h,+h) + f(-h,-h)) / (4*hi*hj)
            buf.copy_from_slice(map_params); buf[i] += hi; buf[j] += hj;
            let fpp = neg_log_posterior(model, data, &buf);
            buf.copy_from_slice(map_params); buf[i] += hi; buf[j] -= hj;
            let fpm = neg_log_posterior(model, data, &buf);
            buf.copy_from_slice(map_params); buf[i] -= hi; buf[j] += hj;
            let fmp = neg_log_posterior(model, data, &buf);
            buf.copy_from_slice(map_params); buf[i] -= hi; buf[j] -= hj;
            let fmm = neg_log_posterior(model, data, &buf);

            let h_ij = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
            if si == sj && !h_ij.is_finite() {
                no_curvature[si] = true;
            }
            let h_ij = if h_ij.is_finite() { h_ij } else { 0.0 };
            hessian[si * n_shape + sj] = h_ij;
            hessian[sj * n_shape + si] = h_ij;
        }
    }
    decouple_params(&mut hessian, n_shape, &no_curvature);

    let shape_log_sigma = log_sigma_from_hessian(&hessian, n_shape);

    // Reassemble full log_sigma: shape params from Hessian, sigma_extra from prior
    let priors = prior_params(model);
    let mut log_sigma = vec![0.0; n_full];
    let mut si = 0;
    for i in 0..n_full {
        if i == se_idx {
            // sigma_extra: use prior width as uncertainty
            log_sigma[i] = priors[i].1.ln();
        } else {
            log_sigma[i] = shape_log_sigma[si];
            si += 1;
        }
    }

    SviFitResult {
        model,
        mu: map_params.to_vec(),
        log_sigma,
        elbo: -f0,
    }
}

// ---------------------------------------------------------------------------
// Post-SVI t0 profile refinement
// ---------------------------------------------------------------------------

/// Profile-likelihood refinement for ALL parameters.
///
/// For each parameter, scans a grid holding all other params fixed at their
/// MAP values, finds the ΔLL=0.5 contour, and sets log_sigma to
/// max(SVI_sigma, profile_sigma). This prevents SVI from underestimating
/// uncertainties due to its mean-field (diagonal) approximation.
fn profile_refine_all(result: &mut SviFitResult, data: &BandFitData) {
    let model = result.model;
    let n_params = result.mu.len();
    let se_idx = model.sigma_extra_idx();
    let n_obs = data.times.len();
    let obs_var = &data.obs_var;

    // Compute log-likelihood at MAP
    let (map_preds, _) = scaled_preds(model, &result.mu, data);
    let se = result.mu[se_idx].exp();
    let se_sq = se * se;
    let mut map_ll = 0.0;
    for i in 0..n_obs {
        if !map_preds[i].is_finite() { continue; }
        let tv = obs_var[i] + se_sq;
        if data.is_upper[i] {
            map_ll += log_normal_cdf((data.upper_flux[i] - map_preds[i]) / tv.sqrt());
        } else {
            let r = data.flux[i] - map_preds[i];
            map_ll += -0.5 * (r * r / tv + (2.0 * std::f64::consts::PI * tv).ln());
        }
    }

    let threshold = map_ll - 0.5; // 1-sigma contour
    let (mut lower, mut upper) = pso_bounds(model);
    if let Some((t0_lo, t0_hi)) = data_t0_window(&data.times, &data.flux, &data.is_upper) {
        let t0_idx = model.t0_idx();
        lower[t0_idx] = t0_lo;
        upper[t0_idx] = t0_hi;
    }

    for j in 0..n_params {
        // Skip sigma_extra — already handled by prior width
        if j == se_idx { continue; }

        let mu_j = result.mu[j];
        let svi_sigma = result.log_sigma[j].exp();

        // Scan the full PSO bounds for this parameter — SVI sigma may
        // drastically underestimate the true width.
        let scan_lo = lower[j];
        let scan_hi = upper[j];
        if scan_hi - scan_lo < 1e-10 { continue; }

        let n_grid = 9usize;
        let mut lo_1sig = mu_j;
        let mut hi_1sig = mu_j;

        for gi in 0..n_grid {
            let val = scan_lo + (scan_hi - scan_lo) * gi as f64 / (n_grid - 1).max(1) as f64;
            let mut p = result.mu.clone();
            p[j] = val;

            let (preds, _) = scaled_preds(model, &p, data);
            let se_p = p[se_idx].exp();
            let se_sq_p = se_p * se_p;
            let mut ll = 0.0;
            for i in 0..n_obs {
                if !preds[i].is_finite() { continue; }
                let tv = obs_var[i] + se_sq_p;
                if data.is_upper[i] {
                    ll += log_normal_cdf((data.upper_flux[i] - preds[i]) / tv.sqrt());
                } else {
                    let r = data.flux[i] - preds[i];
                    ll += -0.5 * (r * r / tv + (2.0 * std::f64::consts::PI * tv).ln());
                }
            }

            if ll >= threshold {
                lo_1sig = lo_1sig.min(val);
                hi_1sig = hi_1sig.max(val);
            }
        }

        let profile_sigma = ((hi_1sig - lo_1sig) / 2.0).max(1e-6);

        // Use the largest of: SVI sigma, profile sigma, and population prior width.
        // The prior floor ensures that when data can't constrain a parameter
        // (e.g., beta when only the decline is observed), we report the
        // population scatter rather than an artificially narrow posterior.
        let pop_priors = population_priors(model);
        let prior_sigma = if j < pop_priors.len() { pop_priors[j].1 } else { 0.0 };

        let best_sigma = svi_sigma.max(profile_sigma).max(prior_sigma);
        result.log_sigma[j] = best_sigma.ln();
    }
}

fn profile_t0_refine(result: &mut SviFitResult, data: &BandFitData) {
    let model = result.model;
    let t0_idx = model.t0_idx();
    let se_idx = model.sigma_extra_idx();

    let mu_t0 = result.mu[t0_idx];

    let t_first = data.times.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut peak_idx = 0;
    let mut peak_val = f64::NEG_INFINITY;
    for (i, &f) in data.flux.iter().enumerate() {
        if f > peak_val {
            peak_val = f;
            peak_idx = i;
        }
    }
    let t_peak = data.times[peak_idx];
    let t0_lo = t_first - 30.0;
    let t0_hi = t_peak + 5.0;
    if t0_lo >= t0_hi {
        return;
    }

    let n_grid: usize = 15;
    let obs_var = &data.obs_var;
    let sigma_extra = result.mu[se_idx].exp();
    let sigma_extra_sq = sigma_extra * sigma_extra;

    let n_obs = data.times.len();
    let mut params = result.mu.clone();
    let mut best_t0 = mu_t0;
    let mut best_params = result.mu.clone();
    let mut best_ll = f64::NEG_INFINITY;

    let log_a_idx = 0;
    let has_baseline = matches!(model, SviModel::Bazin | SviModel::Tde);

    // Score the current t0 first, so a grid point only replaces it if it fits better
    for gi in 0..=n_grid {
        let t0 = if gi == 0 {
            mu_t0
        } else {
            t0_lo + (t0_hi - t0_lo) * (gi - 1) as f64 / (n_grid - 1).max(1) as f64
        };

        for j in 0..params.len() {
            params[j] = result.mu[j];
        }
        params[t0_idx] = t0;

        // Analytically re-fit amplitude (MetzgerKN has none: its params[0] is log10 M_ej)
        let mut preds: Vec<f64> = if model != SviModel::MetzgerKN {
            let saved_log_a = params[log_a_idx];
            params[log_a_idx] = 0.0;
            let saved_b = if has_baseline {
                let v = params[1];
                params[1] = 0.0;
                v
            } else {
                0.0
            };

            let shapes = eval_model_batch(model, &params, &data.times);

            if has_baseline {
                let mut sw = 0.0;
                let mut sy = 0.0;
                let mut sf = 0.0;
                let mut syf = 0.0;
                let mut sff = 0.0;
                for i in 0..n_obs {
                    if data.is_upper[i] || !shapes[i].is_finite() {
                        continue;
                    }
                    let w = 1.0 / (obs_var[i] + sigma_extra_sq);
                    sw += w;
                    sy += w * data.flux[i];
                    sf += w * shapes[i];
                    syf += w * data.flux[i] * shapes[i];
                    sff += w * shapes[i] * shapes[i];
                }
                let det = sw * sff - sf * sf;
                if det.abs() > 1e-20 {
                    let a_opt = (sw * syf - sf * sy) / det;
                    let b_opt = (sff * sy - sf * syf) / det;
                    if a_opt > 1e-10 {
                        params[log_a_idx] = a_opt.ln();
                        params[1] = b_opt;
                    } else {
                        params[log_a_idx] = saved_log_a;
                        params[1] = saved_b;
                    }
                } else {
                    params[log_a_idx] = saved_log_a;
                    params[1] = saved_b;
                }
            } else {
                let mut num = 0.0;
                let mut den = 0.0;
                for i in 0..n_obs {
                    if data.is_upper[i] || !shapes[i].is_finite() {
                        continue;
                    }
                    let w = 1.0 / (obs_var[i] + sigma_extra_sq);
                    num += w * data.flux[i] * shapes[i];
                    den += w * shapes[i] * shapes[i];
                }
                if den > 1e-20 && num / den > 1e-10 {
                    params[log_a_idx] = (num / den).ln();
                } else {
                    params[log_a_idx] = saved_log_a;
                }
            }

            // Reuse the shapes from the amplitude fit instead of re-evaluating
            // the model: preds = A * shapes + B (or just A * shapes).
            let fitted_a = params[log_a_idx].exp();
            let fitted_b = if has_baseline { params[1] } else { 0.0 };
            shapes.iter().map(|&s| fitted_a * s + fitted_b).collect()
        } else {
            eval_model_batch(model, &params, &data.times)
        };
        apply_metzger_kn_scale(model, &mut preds, &data.is_upper);

        let se = params[se_idx].exp();
        let se_sq = se * se;
        let mut ll = 0.0;
        for i in 0..n_obs {
            let pred = preds[i];
            if !pred.is_finite() {
                continue;
            }
            let total_var = obs_var[i] + se_sq;
            if data.is_upper[i] {
                let z = (data.upper_flux[i] - pred) / total_var.sqrt();
                ll += log_normal_cdf(z);
            } else {
                let residual = data.flux[i] - pred;
                ll += -0.5
                    * (residual * residual / total_var
                        + (2.0 * std::f64::consts::PI * total_var).ln());
            }
        }

        if ll > best_ll {
            best_ll = ll;
            best_t0 = t0;
            best_params = params.clone();
        }
    }

    // Profile width: find the range where LL > best_ll - 0.5 (1-sigma for 1 parameter)
    let mut t0_lo_1sig = best_t0;
    let mut t0_hi_1sig = best_t0;
    let threshold = best_ll - 0.5;

    // Scan from best_t0 downward
    for gi in (0..n_grid).rev() {
        let t0 = t0_lo + (t0_hi - t0_lo) * gi as f64 / (n_grid - 1).max(1) as f64;
        if t0 >= best_t0 {
            continue;
        }
        let mut p = result.mu.clone();
        p[t0_idx] = t0;
        // Quick re-eval at this t0 (without re-fitting amplitude for speed)
        let (preds_at, _) = scaled_preds(model, &p, data);
        let se = p[se_idx].exp();
        let se_sq = se * se;
        let mut ll_at = 0.0;
        for i in 0..n_obs {
            if !preds_at[i].is_finite() { continue; }
            let tv = obs_var[i] + se_sq;
            if data.is_upper[i] {
                ll_at += log_normal_cdf((data.upper_flux[i] - preds_at[i]) / tv.sqrt());
            } else {
                let r = data.flux[i] - preds_at[i];
                ll_at += -0.5 * (r * r / tv + (2.0 * std::f64::consts::PI * tv).ln());
            }
        }
        if ll_at >= threshold {
            t0_lo_1sig = t0;
        } else {
            break;
        }
    }

    // Scan from best_t0 upward
    for gi in 0..n_grid {
        let t0 = t0_lo + (t0_hi - t0_lo) * gi as f64 / (n_grid - 1).max(1) as f64;
        if t0 <= best_t0 {
            continue;
        }
        let mut p = result.mu.clone();
        p[t0_idx] = t0;
        let (preds_at, _) = scaled_preds(model, &p, data);
        let se = p[se_idx].exp();
        let se_sq = se * se;
        let mut ll_at = 0.0;
        for i in 0..n_obs {
            if !preds_at[i].is_finite() { continue; }
            let tv = obs_var[i] + se_sq;
            if data.is_upper[i] {
                ll_at += log_normal_cdf((data.upper_flux[i] - preds_at[i]) / tv.sqrt());
            } else {
                let r = data.flux[i] - preds_at[i];
                ll_at += -0.5 * (r * r / tv + (2.0 * std::f64::consts::PI * tv).ln());
            }
        }
        if ll_at >= threshold {
            t0_hi_1sig = t0;
        } else {
            break;
        }
    }

    let profile_sigma = ((t0_hi_1sig - t0_lo_1sig) / 2.0).max(0.1);

    // Keep the amplitude (and baseline) re-fitted at best_t0, not just t0
    result.mu = best_params;
    result.mu[t0_idx] = best_t0;
    result.log_sigma[t0_idx] = profile_sigma.ln();
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct BandFitData {
    times: Vec<f64>,
    flux: Vec<f64>,
    flux_err: Vec<f64>,
    /// Precomputed observation variance: `flux_err[i]² + 1e-10`.
    obs_var: Vec<f64>,
    is_upper: Vec<bool>,
    upper_flux: Vec<f64>,
    #[allow(dead_code)]
    noise_frac_median: f64,
    peak_flux_obs: f64,
}

fn median_f64(xs: &mut [f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(|a, b| a.total_cmp(b));
    let mid = xs.len() / 2;
    if xs.len() % 2 == 0 {
        Some((xs[mid - 1] + xs[mid]) / 2.0)
    } else {
        Some(xs[mid])
    }
}

fn flux_to_mag(flux: f64, zp: f64) -> f64 {
    -2.5 * flux.log10() + zp
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

fn unit_scale() -> f64 {
    1.0
}

/// Result of parametric SVI fitting for a single band.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricBandResult {
    pub band: String,
    pub model: SviModelName,
    pub pso_params: Vec<f64>,
    pub pso_chi2: Option<f64>,
    pub svi_mu: Vec<f64>,
    pub svi_log_sigma: Vec<f64>,
    pub svi_elbo: Option<f64>,
    pub n_obs: usize,
    pub mag_chi2: Option<f64>,
    /// Factor to multiply `eval_model_flux(model, pso_params, ..)` by to get
    /// the flux the fit used. 1.0 except for MetzgerKN.
    #[serde(default = "unit_scale")]
    pub pso_flux_scale: f64,
    /// As `pso_flux_scale`, for `svi_mu`.
    #[serde(default = "unit_scale")]
    pub svi_flux_scale: f64,
    pub per_model_chi2: HashMap<SviModelName, Option<f64>>,
    /// PSO best-fit parameters for every model attempted (populated when
    /// fit_all_models is true).
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub per_model_params: HashMap<SviModelName, Vec<f64>>,
    /// Which uncertainty method was used for this result.
    #[serde(default)]
    pub uncertainty_method: UncertaintyMethod,
    /// MultiBazin greedy fit: variable number of Bazin components.
    /// BIC selects K; best_k > 1 suggests recurrent or multi-peaked transients.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multi_bazin: Option<MultiBazinResult>,
}

/// Evaluate a parametric model at the given times.
///
/// `model` selects which lightcurve model to use.
/// `params` are the internal (transformed) parameters — pass `pso_params` or
/// `svi_mu` from a `ParametricBandResult` directly.
/// `times` are the time values (relative days) at which to evaluate.
///
/// Returns a Vec of predicted flux values, one per time point. For MetzgerKN,
/// multiply by the result's `pso_flux_scale` or `svi_flux_scale` (matching the
/// params passed) to reproduce the flux the fit used.
pub fn eval_model_flux(model: SviModelName, params: &[f64], times: &[f64]) -> Vec<f64> {
    let m = match model {
        SviModelName::Bazin => SviModel::Bazin,
        SviModelName::Villar => SviModel::Villar,
        SviModelName::MetzgerKN => SviModel::MetzgerKN,
        SviModelName::Tde => SviModel::Tde,
        SviModelName::Arnett => SviModel::Arnett,
        SviModelName::Magnetar => SviModel::Magnetar,
        SviModelName::ShockCooling => SviModel::ShockCooling,
        SviModelName::Afterglow => SviModel::Afterglow,
    };
    if m == SviModel::MetzgerKN {
        return metzger_kn_eval_batch(params, times);
    }
    times.iter().map(|&t| eval_model(m, params, t)).collect()
}

/// Pre-computed PSO result for one band, from GPU batch fitting.
#[derive(Clone)]
pub struct GpuPsoBandResult {
    /// Best model name.
    pub model: SviModelName,
    /// Best-fit PSO parameters (normalized, as used by the model).
    pub pso_params: Vec<f64>,
    /// PSO cost (negative log-likelihood, normalized by n_obs).
    pub pso_cost: f64,
    /// Per-model chi2 (cost × 2, may have None for models not fitted).
    pub per_model_chi2: HashMap<SviModelName, Option<f64>>,
    /// Per-model best params (populated when fit_all_models=true).
    pub per_model_params: HashMap<SviModelName, Vec<f64>>,
    /// MultiBazin result (if fitted on GPU).
    pub multi_bazin: Option<MultiBazinResult>,
}

/// Finalize parametric results using pre-computed GPU PSO params.
///
/// Runs SVI/Laplace uncertainty estimation on CPU (seeded from GPU PSO),
/// computes mag-space chi2, and assembles the final ParametricBandResult.
///
/// `flux_bands`: maps band_name → BandData in flux space.
/// `gpu_results`: per-band GPU PSO results.
/// `band_names`: the band each entry of `gpu_results` belongs to. The first
///   entry's `multi_bazin` is kept.
/// `method`: uncertainty estimation method.
pub fn finalize_parametric_from_gpu(
    flux_bands: &HashMap<String, BandData>,
    gpu_results: &[GpuPsoBandResult],
    band_names: &[String],
    method: UncertaintyMethod,
) -> Vec<ParametricBandResult> {
    assert_eq!(band_names.len(), gpu_results.len(), "band_names must be parallel to gpu_results");
    if flux_bands.is_empty() || gpu_results.is_empty() {
        return Vec::new();
    }

    let snr_threshold = 3.0;
    let svi_lr = 0.005;
    let svi_n_steps = 500;
    let svi_n_samples = 16;
    let zp = 23.9;

    // Build BandFitData for each unique band
    let mut band_data_map: HashMap<String, (BandFitData, Vec<f64>)> = HashMap::new();

    for (band_name, band_data) in flux_bands {
        let fluxes = &band_data.values;
        let flux_errs = &band_data.errors;
        let times = &band_data.times;

        if fluxes.is_empty() { continue; }
        let peak_flux = fluxes.iter().cloned().fold(f64::MIN, f64::max);
        if peak_flux <= 0.0 { continue; }

        let normalized_flux: Vec<f64> = fluxes.iter().map(|f| f / peak_flux).collect();
        let normalized_err: Vec<f64> = flux_errs.iter().map(|e| e / peak_flux).collect();

        let is_upper: Vec<bool> = fluxes.iter().zip(flux_errs.iter())
            .map(|(f, e)| *e > 0.0 && (*f / *e) < snr_threshold)
            .collect();
        let upper_flux: Vec<f64> = flux_errs.iter()
            .map(|e| snr_threshold * e / peak_flux)
            .collect();

        let mut frac_noises: Vec<f64> = normalized_flux.iter().zip(normalized_err.iter())
            .filter_map(|(f, e)| if *f > 0.0 { Some(e / f) } else { None })
            .collect();
        let noise_frac_median = median_f64(&mut frac_noises).unwrap_or(0.0);

        let mags_obs: Vec<f64> = fluxes.iter().map(|f| flux_to_mag(*f, zp)).collect();
        let obs_var: Vec<f64> = normalized_err.iter().map(|e| e * e + 1e-10).collect();

        let fit_data = BandFitData {
            times: times.clone(),
            flux: normalized_flux,
            flux_err: normalized_err,
            obs_var,
            is_upper,
            upper_flux,
            noise_frac_median,
            peak_flux_obs: peak_flux,
        };

        band_data_map.insert(band_name.clone(), (fit_data, mags_obs));
    }

    let mut results: Vec<ParametricBandResult> = Vec::new();

    for (band_idx, (gpu, band_name)) in gpu_results.iter().zip(band_names).enumerate() {

        let (data, mags_obs) = match band_data_map.get(band_name) {
            Some(d) => d,
            None => continue,
        };

        if gpu.pso_params.is_empty() { continue; }

        let pso_model = SviModel::from_name(&gpu.model);

        // SVI/Laplace on CPU, seeded from GPU PSO
        let mut svi_result = match method {
            UncertaintyMethod::Svi => svi_fit(
                pso_model, data, svi_n_steps, svi_n_samples, svi_lr,
                Some(&gpu.pso_params),
            ),
            UncertaintyMethod::Laplace => laplace_fit(pso_model, data, &gpu.pso_params),
        };

        profile_t0_refine(&mut svi_result, data);
        profile_refine_all(&mut svi_result, data);

        // Mag-space chi2
        let (svi_preds, svi_flux_scale) = scaled_preds(pso_model, &svi_result.mu, data);
        let (pso_preds, pso_flux_scale) = scaled_preds(pso_model, &gpu.pso_params, data);
        let n_pos_svi = svi_preds.iter().filter(|&&p| p * data.peak_flux_obs > 0.0).count();
        let use_pso_for_mag = n_pos_svi < data.times.len() / 2;
        let mag_preds = if use_pso_for_mag { pso_preds } else { svi_preds };
        let mut mag_chi2_sum = 0.0;
        let mut mag_chi2_n = 0usize;
        for i in 0..data.times.len() {
            let pred_flux = mag_preds[i] * data.peak_flux_obs;
            if pred_flux > 0.0 && data.flux[i] * data.peak_flux_obs > 0.0 {
                let mag_pred = flux_to_mag(pred_flux, zp);
                let mag_obs = mags_obs[i];
                let mag_err = 1.0857 * data.flux_err[i] / data.flux[i];
                if mag_err > 0.0 {
                    let residual = mag_obs - mag_pred;
                    mag_chi2_sum += residual * residual / (mag_err * mag_err);
                    mag_chi2_n += 1;
                }
            }
        }
        let mag_chi2 = if mag_chi2_n > 0 { mag_chi2_sum / mag_chi2_n as f64 } else { f64::NAN };

        results.push(ParametricBandResult {
            band: band_name.clone(),
            model: gpu.model.clone(),
            pso_params: gpu.pso_params.clone(),
            pso_chi2: finite_or_none(gpu.pso_cost),
            svi_mu: svi_result.mu,
            svi_log_sigma: svi_result.log_sigma,
            svi_elbo: finite_or_none(svi_result.elbo),
            n_obs: data.times.len(),
            mag_chi2: finite_or_none(mag_chi2),
            pso_flux_scale,
            svi_flux_scale,
            per_model_chi2: gpu.per_model_chi2.clone(),
            per_model_params: gpu.per_model_params.clone(),
            uncertainty_method: method,
            multi_bazin: if band_idx == 0 { gpu.multi_bazin.clone() } else { None },
        });
    }

    results
}

/// Return the SVI prior (centers, widths) for a model, with centers recentered
/// on PSO init params (except sigma_extra which keeps original center).
///
/// Used by the GPU SVI path to prepare `SviBatchInput`.
pub fn svi_prior_for_model(model_name: &SviModelName, pso_params: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let model = SviModel::from_name(model_name);
    let priors = prior_params(model);
    let se_idx = model.sigma_extra_idx();
    let np = model.n_params();
    let mut centers = vec![0.0; np];
    let mut widths = vec![0.0; np];
    for j in 0..np {
        let (orig_center, width) = priors[j];
        // Center on PSO params for all except sigma_extra
        centers[j] = if j == se_idx { orig_center } else { pso_params[j.min(pso_params.len() - 1)] };
        widths[j] = width;
    }
    (centers, widths)
}

/// Return the model_id (matching CUDA kernel), n_params, and sigma_extra index
/// for a given model name.
pub fn svi_model_meta(model_name: &SviModelName) -> (usize, usize, usize) {
    let model = SviModel::from_name(model_name);
    let model_id = match model_name {
        SviModelName::Bazin => 0,
        SviModelName::Villar => 1,
        SviModelName::Tde => 2,
        SviModelName::Arnett => 3,
        SviModelName::Magnetar => 4,
        SviModelName::ShockCooling => 5,
        SviModelName::Afterglow => 6,
        SviModelName::MetzgerKN => 7,
    };
    (model_id, model.n_params(), model.sigma_extra_idx())
}

/// Finalize parametric results using pre-computed GPU SVI outputs.
///
/// Like `finalize_parametric_from_gpu` but skips CPU SVI — uses pre-computed
/// mu/log_sigma/elbo from GPU SVI kernel.
///
/// `svi_outputs` and `band_names` are parallel to `gpu_results`.
pub fn finalize_parametric_with_gpu_svi(
    flux_bands: &HashMap<String, BandData>,
    gpu_results: &[GpuPsoBandResult],
    svi_outputs: &[(Vec<f64>, Vec<f64>, f64)], // (mu, log_sigma, elbo) per band
    band_names: &[String],
) -> Vec<ParametricBandResult> {
    assert_eq!(band_names.len(), gpu_results.len(), "band_names must be parallel to gpu_results");
    if flux_bands.is_empty() || gpu_results.is_empty() || svi_outputs.is_empty() {
        return Vec::new();
    }

    let zp = 23.9;
    let snr_threshold = 3.0;

    // Build BandFitData for each unique band
    let mut band_data_map: HashMap<String, (BandFitData, Vec<f64>)> = HashMap::new();

    for (band_name, band_data) in flux_bands {
        let fluxes = &band_data.values;
        let flux_errs = &band_data.errors;
        let times = &band_data.times;

        if fluxes.is_empty() { continue; }
        let peak_flux = fluxes.iter().cloned().fold(f64::MIN, f64::max);
        if peak_flux <= 0.0 { continue; }

        let normalized_flux: Vec<f64> = fluxes.iter().map(|f| f / peak_flux).collect();
        let normalized_err: Vec<f64> = flux_errs.iter().map(|e| e / peak_flux).collect();

        let is_upper: Vec<bool> = fluxes.iter().zip(flux_errs.iter())
            .map(|(f, e)| *e > 0.0 && (*f / *e) < snr_threshold)
            .collect();
        let upper_flux: Vec<f64> = flux_errs.iter()
            .map(|e| snr_threshold * e / peak_flux)
            .collect();

        let mut frac_noises: Vec<f64> = normalized_flux.iter().zip(normalized_err.iter())
            .filter_map(|(f, e)| if *f > 0.0 { Some(e / f) } else { None })
            .collect();
        let noise_frac_median = median_f64(&mut frac_noises).unwrap_or(0.0);

        let mags_obs: Vec<f64> = fluxes.iter().map(|f| flux_to_mag(*f, zp)).collect();
        let obs_var: Vec<f64> = normalized_err.iter().map(|e| e * e + 1e-10).collect();

        let fit_data = BandFitData {
            times: times.clone(),
            flux: normalized_flux,
            flux_err: normalized_err,
            obs_var,
            is_upper,
            upper_flux,
            noise_frac_median,
            peak_flux_obs: peak_flux,
        };

        band_data_map.insert(band_name.clone(), (fit_data, mags_obs));
    }

    let mut results: Vec<ParametricBandResult> = Vec::new();

    for (band_idx, (gpu, band_name)) in gpu_results.iter().zip(band_names).enumerate() {
        if band_idx >= svi_outputs.len() { break; }
        let (ref svi_mu, ref svi_log_sigma, svi_elbo) = svi_outputs[band_idx];

        let (data, mags_obs) = match band_data_map.get(band_name) {
            Some(d) => d,
            None => continue,
        };

        if gpu.pso_params.is_empty() { continue; }

        let pso_model = SviModel::from_name(&gpu.model);

        // Apply sigma inflation to match CPU
        let log_inflation = SIGMA_INFLATION_FACTOR.ln();
        let inflated_log_sigma: Vec<f64> = svi_log_sigma.iter().map(|ls| ls + log_inflation).collect();

        // profile_t0_refine on the GPU SVI result
        let mut svi_result = SviFitResult {
            model: pso_model,
            mu: svi_mu.clone(),
            log_sigma: inflated_log_sigma,
            elbo: svi_elbo,
        };
        profile_t0_refine(&mut svi_result, data);
        profile_refine_all(&mut svi_result, data);

        // Mag-space chi2
        let (svi_preds, svi_flux_scale) = scaled_preds(pso_model, &svi_result.mu, data);
        let (pso_preds, pso_flux_scale) = scaled_preds(pso_model, &gpu.pso_params, data);
        let n_pos_svi = svi_preds.iter().filter(|&&p| p * data.peak_flux_obs > 0.0).count();
        let use_pso_for_mag = n_pos_svi < data.times.len() / 2;
        let mag_preds = if use_pso_for_mag { pso_preds } else { svi_preds };
        let mut mag_chi2_sum = 0.0;
        let mut mag_chi2_n = 0usize;
        for i in 0..data.times.len() {
            let pred_flux = mag_preds[i] * data.peak_flux_obs;
            if pred_flux > 0.0 && data.flux[i] * data.peak_flux_obs > 0.0 {
                let mag_pred = flux_to_mag(pred_flux, zp);
                let mag_obs = mags_obs[i];
                let mag_err = 1.0857 * data.flux_err[i] / data.flux[i];
                if mag_err > 0.0 {
                    let residual = mag_obs - mag_pred;
                    mag_chi2_sum += residual * residual / (mag_err * mag_err);
                    mag_chi2_n += 1;
                }
            }
        }
        let mag_chi2 = if mag_chi2_n > 0 { mag_chi2_sum / mag_chi2_n as f64 } else { f64::NAN };

        results.push(ParametricBandResult {
            band: band_name.clone(),
            model: gpu.model.clone(),
            pso_params: gpu.pso_params.clone(),
            pso_chi2: finite_or_none(gpu.pso_cost),
            svi_mu: svi_result.mu,
            svi_log_sigma: svi_result.log_sigma,
            svi_elbo: finite_or_none(svi_result.elbo),
            n_obs: data.times.len(),
            mag_chi2: finite_or_none(mag_chi2),
            pso_flux_scale,
            svi_flux_scale,
            per_model_chi2: gpu.per_model_chi2.clone(),
            per_model_params: gpu.per_model_params.clone(),
            uncertainty_method: UncertaintyMethod::Svi,
            multi_bazin: if band_idx == 0 { gpu.multi_bazin.clone() } else { None },
        });
    }

    results
}

/// Finalize parametric results for ALL models (not just the best).
///
/// `gpu_results` and `svi_outputs` are parallel arrays, one entry per (band, model)
/// pair. `band_names` gives the band name for each entry (matching the ordering).
///
/// Returns one `ParametricBandResult` per (band, model) pair.
pub fn finalize_all_models_with_gpu_svi(
    flux_bands: &HashMap<String, BandData>,
    gpu_results: &[GpuPsoBandResult],
    svi_outputs: &[(Vec<f64>, Vec<f64>, f64)],
    band_names: &[String],
) -> Vec<ParametricBandResult> {
    if flux_bands.is_empty() || gpu_results.is_empty() || svi_outputs.is_empty() {
        return Vec::new();
    }

    let zp = 23.9;
    let snr_threshold = 3.0;

    // Build BandFitData for each unique band
    let mut band_data_map: HashMap<String, (BandFitData, Vec<f64>)> = HashMap::new();

    for (band_name, band_data) in flux_bands {
        let fluxes = &band_data.values;
        let flux_errs = &band_data.errors;
        let times = &band_data.times;

        if fluxes.is_empty() { continue; }
        let peak_flux = fluxes.iter().cloned().fold(f64::MIN, f64::max);
        if peak_flux <= 0.0 { continue; }

        let normalized_flux: Vec<f64> = fluxes.iter().map(|f| f / peak_flux).collect();
        let normalized_err: Vec<f64> = flux_errs.iter().map(|e| e / peak_flux).collect();

        let is_upper: Vec<bool> = fluxes.iter().zip(flux_errs.iter())
            .map(|(f, e)| *e > 0.0 && (*f / *e) < snr_threshold).collect();
        let upper_flux: Vec<f64> = flux_errs.iter()
            .map(|e| snr_threshold * e / peak_flux).collect();

        let mut frac_noises: Vec<f64> = normalized_flux.iter().zip(normalized_err.iter())
            .filter_map(|(f, e)| if *f > 0.0 { Some(e / f) } else { None })
            .collect();
        let noise_frac_median = median_f64(&mut frac_noises).unwrap_or(0.0);

        let mags_obs: Vec<f64> = fluxes.iter().map(|f| flux_to_mag(*f, zp)).collect();
        let obs_var: Vec<f64> = normalized_err.iter().map(|e| e * e + 1e-10).collect();

        let fit_data = BandFitData {
            times: times.clone(),
            flux: normalized_flux,
            flux_err: normalized_err,
            obs_var,
            is_upper,
            upper_flux,
            noise_frac_median,
            peak_flux_obs: peak_flux,
        };

        band_data_map.insert(band_name.clone(), (fit_data, mags_obs));
    }

    let mut results: Vec<ParametricBandResult> = Vec::new();

    for (idx, gpu) in gpu_results.iter().enumerate() {
        if idx >= svi_outputs.len() || idx >= band_names.len() { break; }
        let band_name = &band_names[idx];
        let (ref svi_mu, ref svi_log_sigma, svi_elbo) = svi_outputs[idx];

        let (data, mags_obs) = match band_data_map.get(band_name) {
            Some(d) => d,
            None => continue,
        };

        if gpu.pso_params.is_empty() || svi_mu.is_empty() { continue; }

        let pso_model = SviModel::from_name(&gpu.model);

        let mut svi_result = SviFitResult {
            model: pso_model,
            mu: svi_mu.clone(),
            log_sigma: svi_log_sigma.clone(),
            elbo: svi_elbo,
        };
        profile_t0_refine(&mut svi_result, data);
        profile_refine_all(&mut svi_result, data);

        // Mag-space chi2
        let (svi_preds, svi_flux_scale) = scaled_preds(pso_model, &svi_result.mu, data);
        let (pso_preds, pso_flux_scale) = scaled_preds(pso_model, &gpu.pso_params, data);
        let n_pos_svi = svi_preds.iter().filter(|&&p| p * data.peak_flux_obs > 0.0).count();
        let use_pso_for_mag = n_pos_svi < data.times.len() / 2;
        let mag_preds = if use_pso_for_mag { pso_preds } else { svi_preds };
        let mut mag_chi2_sum = 0.0;
        let mut mag_chi2_n = 0usize;
        for i in 0..data.times.len() {
            let pred_flux = mag_preds[i] * data.peak_flux_obs;
            if pred_flux > 0.0 && data.flux[i] * data.peak_flux_obs > 0.0 {
                let mag_pred = flux_to_mag(pred_flux, zp);
                let mag_obs = mags_obs[i];
                let mag_err = 1.0857 * data.flux_err[i] / data.flux[i];
                if mag_err > 0.0 {
                    let residual = mag_obs - mag_pred;
                    mag_chi2_sum += residual * residual / (mag_err * mag_err);
                    mag_chi2_n += 1;
                }
            }
        }
        let mag_chi2 = if mag_chi2_n > 0 { mag_chi2_sum / mag_chi2_n as f64 } else { f64::NAN };

        results.push(ParametricBandResult {
            band: band_name.clone(),
            model: gpu.model.clone(),
            pso_params: gpu.pso_params.clone(),
            pso_chi2: finite_or_none(gpu.pso_cost),
            svi_mu: svi_result.mu,
            svi_log_sigma: svi_result.log_sigma,
            svi_elbo: finite_or_none(svi_result.elbo),
            n_obs: data.times.len(),
            mag_chi2: finite_or_none(mag_chi2),
            pso_flux_scale,
            svi_flux_scale,
            per_model_chi2: gpu.per_model_chi2.clone(),
            per_model_params: HashMap::new(), // Not needed — each model has its own entry
            uncertainty_method: UncertaintyMethod::Svi,
            multi_bazin: gpu.multi_bazin.clone(),
        });
    }

    results
}

/// Normalise one band to peak flux and flag upper limits for fitting.
/// Fit parametric lightcurve models to all bands.
///
/// `bands` maps band names to `BandData` containing flux values.
/// Model selection fits each analytic model with a profiled-amplitude t0 grid
/// plus L-BFGS (MetzgerKN with PSO), then the chosen `method` estimates uncertainty.
/// Fit parametric models to per-band flux data.
///
/// If `forced_model` is `Some`, only that model is fit (no model selection).
/// Otherwise, model selection chooses the best model by AICc/BIC.
pub fn fit_parametric_model(
    bands: &HashMap<String, BandData>,
    fit_all_models: bool,
    method: UncertaintyMethod,
    forced_model: Option<SviModelName>,
) -> Vec<ParametricBandResult> {
    if bands.is_empty() {
        return Vec::new();
    }

    let snr_threshold = 3.0;
    let svi_lr = 0.005;
    let svi_n_steps = 500;
    let svi_n_samples = 16;
    let zp = 23.9;

    let mut band_entries: Vec<(String, BandFitData, Vec<f64>)> = Vec::new();

    for (band_name, band_data) in bands {
        let fluxes = &band_data.values;
        let flux_errs = &band_data.errors;
        let times = &band_data.times;

        if fluxes.is_empty() {
            continue;
        }
        let peak_flux = fluxes.iter().cloned().fold(f64::MIN, f64::max);
        if peak_flux <= 0.0 {
            continue;
        }

        let normalized_flux: Vec<f64> = fluxes.iter().map(|f| f / peak_flux).collect();
        let normalized_err: Vec<f64> = flux_errs.iter().map(|e| e / peak_flux).collect();

        let is_upper: Vec<bool> = fluxes
            .iter()
            .zip(flux_errs.iter())
            .map(|(f, e)| *e > 0.0 && (*f / *e) < snr_threshold)
            .collect();
        let upper_flux: Vec<f64> = flux_errs
            .iter()
            .map(|e| snr_threshold * e / peak_flux)
            .collect();

        let mut frac_noises: Vec<f64> = normalized_flux
            .iter()
            .zip(normalized_err.iter())
            .filter_map(|(f, e)| if *f > 0.0 { Some(e / f) } else { None })
            .collect();
        let noise_frac_median = median_f64(&mut frac_noises).unwrap_or(0.0);

        let mags_obs: Vec<f64> = fluxes.iter().map(|f| flux_to_mag(*f, zp)).collect();

        let obs_var: Vec<f64> = normalized_err.iter().map(|e| e * e + 1e-10).collect();

        let fit_data = BandFitData {
            times: times.clone(),
            flux: normalized_flux,
            flux_err: normalized_err,
            obs_var,
            is_upper,
            upper_flux,
            noise_frac_median,
            peak_flux_obs: peak_flux,
        };

        band_entries.push((band_name.clone(), fit_data, mags_obs));
    }

    band_entries.sort_by(|a, b| b.1.times.len().cmp(&a.1.times.len()));

    let mut results: Vec<ParametricBandResult> = Vec::new();
    let mut ref_t0: Option<f64> = None; // set after reference band fit

    for (band_idx, (band_name, data, mags_obs)) in band_entries.iter().enumerate() {
        // Step 1: PSO model selection (or forced single model)
        // non-reference bands get t0 clamped to ±1 day of reference
        let t0_hint = if band_idx > 0 { ref_t0 } else { None };
        let (pso_model, pso_params, pso_chi2, per_model_chi2, per_model_params) = if let Some(model) = forced_model {
            let (_, params, chi2) = pso_fit_single_model_t0(data, SviModel::from_name(&model), t0_hint);
            let mut chi2_map = HashMap::new();
            chi2_map.insert(model.clone(), finite_or_none(chi2));
            let mut params_map = HashMap::new();
            if fit_all_models {
                params_map.insert(model.clone(), params.clone());
            }
            (SviModel::from_name(&model), params, chi2, chi2_map, params_map)
        } else {
            pso_model_select_t0(data, fit_all_models, t0_hint)
        };

        if pso_params.is_empty() {
            continue;
        }

        // Step 1b: Profile t0 (reference band only, SVI only — Laplace
        // relies on the cheaper post-fit profile_t0_refine instead)
        let (pso_params, pso_chi2) = if band_idx == 0 && method == UncertaintyMethod::Svi {
            profile_t0_search(data, pso_model, &pso_params, pso_chi2)
        } else {
            (pso_params, pso_chi2)
        };

        // Step 2: Uncertainty estimation (SVI or Laplace)
        let mut svi_result = match method {
            UncertaintyMethod::Svi => svi_fit(
                pso_model,
                data,
                svi_n_steps,
                svi_n_samples,
                svi_lr,
                Some(&pso_params),
            ),
            UncertaintyMethod::Laplace => laplace_fit(pso_model, data, &pso_params),
        };

        // Step 3: Profile-likelihood refinement of t0 and all params
        profile_t0_refine(&mut svi_result, data);
        profile_refine_all(&mut svi_result, data);

        // lock in reference band t0 for remaining bands
        if band_idx == 0 {
            ref_t0 = Some(svi_result.mu[pso_model.t0_idx()]);
        }

        // Compute reduced chi² in magnitude space using SVI mu.
        // Fall back to PSO params if SVI gives too few positive predictions.
        let (svi_preds, svi_flux_scale) = scaled_preds(pso_model, &svi_result.mu, data);
        let (pso_preds, pso_flux_scale) = scaled_preds(pso_model, &pso_params, data);
        let n_pos_svi = svi_preds.iter().filter(|&&p| p * data.peak_flux_obs > 0.0).count();
        let use_pso_for_mag = n_pos_svi < data.times.len() / 2;
        let mag_preds = if use_pso_for_mag { pso_preds } else { svi_preds };
        let mut mag_chi2_sum = 0.0;
        let mut mag_chi2_n = 0usize;
        for i in 0..data.times.len() {
            let pred_flux = mag_preds[i] * data.peak_flux_obs;
            if pred_flux > 0.0 && data.flux[i] > 0.0 {
                let mag_pred = flux_to_mag(pred_flux, zp);
                let mag_obs = mags_obs[i];
                // Both data.flux_err and data.flux are normalized by peak_flux_obs,
                // so flux_err/flux gives the correct fractional error.
                let mag_err = 1.0857 * data.flux_err[i] / data.flux[i];
                if mag_err > 0.0 {
                    let residual = mag_obs - mag_pred;
                    mag_chi2_sum += residual * residual / (mag_err * mag_err);
                    mag_chi2_n += 1;
                }
            }
        }
        let mag_chi2 = if mag_chi2_n > 0 {
            mag_chi2_sum / mag_chi2_n as f64
        } else {
            f64::NAN
        };

        // MultiBazin: run on the first (most-populated) band
        let multi_bazin = if band_idx == 0 {
            let bazin_fit = (pso_model == SviModel::Bazin).then_some(pso_params.as_slice());
            let mb = fit_multi_bazin(data, bazin_fit);
            if mb.params.is_empty() { None } else { Some(mb) }
        } else {
            None
        };

        results.push(ParametricBandResult {
            band: band_name.clone(),
            model: pso_model.to_name(),
            pso_params: pso_params.clone(),
            pso_chi2: finite_or_none(pso_chi2),
            svi_mu: svi_result.mu.clone(),
            svi_log_sigma: svi_result.log_sigma.clone(),
            svi_elbo: finite_or_none(svi_result.elbo),
            n_obs: data.times.len(),
            mag_chi2: finite_or_none(mag_chi2),
            pso_flux_scale,
            svi_flux_scale,
            per_model_chi2,
            per_model_params,
            uncertainty_method: method,
            multi_bazin,
        });
    }

    results
}

/// Backwards-compatible wrapper: fit with automatic model selection.
pub fn fit_parametric(
    bands: &HashMap<String, BandData>,
    fit_all_models: bool,
    method: UncertaintyMethod,
) -> Vec<ParametricBandResult> {
    // Use joint multi-band Villar fitting when ≥2 bands with data are available.
    // This produces better fits via band-balanced likelihood and shared structure
    // (JAX-calibrated, Sushant/villar-pso).
    let bands_with_data: usize = bands.values()
        .filter(|b| !b.values.is_empty())
        .count();
    if bands_with_data >= 2 && !fit_all_models {
        let multiband_results = fit_parametric_multiband(bands, method);
        if !multiband_results.is_empty() {
            return multiband_results;
        }
    }
    // Fall back to per-band fitting
    fit_parametric_model(bands, fit_all_models, method, None)
}

// ---------------------------------------------------------------------------
// Multi-band joint Villar fitting (à la superphot+, de Soto et al. 2024)
// ---------------------------------------------------------------------------

/// Relative prior for a non-reference band parameter (de Soto+ 2024, Table 2).
/// Returns (mean_offset, sigma_offset) for each Villar param.
/// Offsets are additive in log-space (multiplicative in linear) except for
/// beta (additive in linear) and t0 (additive in days).
fn relative_band_priors() -> Vec<(f64, f64, bool)> {
    // (mean, sigma, is_additive)
    // JAX-calibrated offset priors (Sushant/villar-pso), converted from log10→ln.
    // Villar: [log_A, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
    let ln10 = std::f64::consts::LN_10;
    vec![
        (-0.015 * ln10, 0.08 * ln10, false),  // log_A: JAX μ=-0.015, σ=0.08 in log10
        (-0.001, 0.008, true),                  // beta: linear offset
        (-0.095 * ln10, 0.25 * ln10, false),  // log_gamma: JAX μ=-0.095, σ=0.25 in log10
        (-1.181, 1.5, true),                    // t0: JAX μ=-1.181 days, σ=1.5
        (-0.195 * ln10, 0.35 * ln10, false),  // log_tau_rise: JAX μ=-0.195, σ=0.35 in log10
        (-0.323 * ln10, 0.25 * ln10, false),  // log_tau_fall: JAX μ=-0.323, σ=0.25 in log10
        (-0.025 * ln10, 0.30 * ln10, false),  // log_sigma_extra: JAX μ=-0.025, σ=0.30 in log10
    ]
}

/// Villar physical validity constraint (matches JAX `_villar_constraint`).
/// Evaluates for one band's physical parameters.
/// Returns 0 if valid, positive penalty otherwise.
fn villar_constraint(log_gamma: f64, beta: f64, log_tau_rise: f64, log_tau_fall: f64) -> f64 {
    let gamma = log_gamma.exp();
    let tau_rise = log_tau_rise.exp();
    let tau_fall = log_tau_fall.exp();
    let c1 = (gamma * beta - 1.0).max(0.0);
    let c2 = ((-gamma / tau_rise).exp() * (tau_fall / tau_rise - 1.0) - 1.0).max(0.0);
    let c3 = (beta * tau_fall - 1.0 + beta * gamma).max(0.0);
    c1 + c2 + c3
}

/// Joint multi-band Villar fit with relative priors between bands.
///
/// Fits all bands simultaneously by parameterizing the reference band with
/// the full 7 Villar params and each additional band as offsets from reference.
/// The offsets are regularized by population-level relative priors from
/// de Soto et al. 2024 (superphot+).
///
/// The joint parameter vector is:
///   [ref_log_A, ref_beta, ref_log_gamma, ref_t0, ref_log_tau_rise, ref_log_tau_fall, ref_log_se,
///    off1_log_A, off1_beta, off1_log_gamma, off1_t0, off1_log_tau_rise, off1_log_tau_fall, off1_log_se,
///    off2_..., ...]
///
/// Total: 7 + 7*(n_bands-1) parameters.
pub fn fit_parametric_multiband(
    bands: &HashMap<String, BandData>,
    method: UncertaintyMethod,
) -> Vec<ParametricBandResult> {
    if bands.is_empty() {
        return Vec::new();
    }

    let model = SviModel::Villar;
    let n_model_params = model.n_params(); // 7
    let snr_threshold = 3.0;
    let zp = 23.9;

    // Build per-band data, sorted by n_obs descending (reference = first)
    let mut band_entries: Vec<(String, BandFitData, Vec<f64>)> = Vec::new();
    for (band_name, band_data) in bands {
        let fluxes = &band_data.values;
        let flux_errs = &band_data.errors;
        let times = &band_data.times;
        if fluxes.is_empty() { continue; }
        let peak_flux = fluxes.iter().cloned().fold(f64::MIN, f64::max);
        if peak_flux <= 0.0 { continue; }

        let normalized_flux: Vec<f64> = fluxes.iter().map(|f| f / peak_flux).collect();
        let normalized_err: Vec<f64> = flux_errs.iter().map(|e| e / peak_flux).collect();
        let is_upper: Vec<bool> = fluxes.iter().zip(flux_errs.iter())
            .map(|(f, e)| *e > 0.0 && (*f / *e) < snr_threshold).collect();
        let upper_flux: Vec<f64> = flux_errs.iter().map(|e| snr_threshold * e / peak_flux).collect();
        let mut frac_noises: Vec<f64> = normalized_flux.iter().zip(normalized_err.iter())
            .filter_map(|(f, e)| if *f > 0.0 { Some(e / f) } else { None }).collect();
        let noise_frac_median = median_f64(&mut frac_noises).unwrap_or(0.0);
        let mags_obs: Vec<f64> = fluxes.iter().map(|f| flux_to_mag(*f, zp)).collect();
        let obs_var: Vec<f64> = normalized_err.iter().map(|e| e * e + 1e-10).collect();

        band_entries.push((band_name.clone(), BandFitData {
            times: times.clone(), flux: normalized_flux, flux_err: normalized_err,
            obs_var, is_upper, upper_flux, noise_frac_median, peak_flux_obs: peak_flux,
        }, mags_obs));
    }
    band_entries.sort_by(|a, b| b.1.times.len().cmp(&a.1.times.len()));

    if band_entries.is_empty() {
        return Vec::new();
    }

    let n_bands = band_entries.len();
    let n_joint = n_model_params + n_model_params * (n_bands - 1);

    // Build joint bounds: reference params + offset params
    let (ref_lower, ref_upper) = pso_bounds(model);
    let rel_priors = relative_band_priors();

    let mut joint_lower = ref_lower.clone();
    let mut joint_upper = ref_upper.clone();

    for _ in 1..n_bands {
        for &(_mean, sigma, _is_add) in rel_priors.iter() {
            // Offset bounds: ±3σ from prior
            joint_lower.push(-3.0 * sigma);
            joint_upper.push(3.0 * sigma);
        }
    }

    // Data-informed t0 bounds for reference band
    let ref_data = &band_entries[0].1;
    let t0_idx = model.t0_idx();
    {
        let mut peak_time = ref_data.times[0];
        let mut peak_flux = f64::NEG_INFINITY;
        for i in 0..ref_data.times.len() {
            if !ref_data.is_upper[i] && ref_data.flux[i] > peak_flux {
                peak_flux = ref_data.flux[i];
                peak_time = ref_data.times[i];
            }
        }
        let t_first = ref_data.times.iter().cloned().fold(f64::INFINITY, f64::min);
        joint_lower[t0_idx] = (t_first - 30.0).max(joint_lower[t0_idx]);
        joint_upper[t0_idx] = (peak_time + 10.0).min(joint_upper[t0_idx]);
    }

    // Joint cost function: band-balanced NLL + physical constraints + priors
    // (JAX-calibrated, Sushant/villar-pso)
    let se_idx = model.sigma_extra_idx();
    let pop_prior_ref = population_priors(model);

    // Pre-compute band sizes for band-balanced weighting
    let band_n_obs: Vec<usize> = band_entries.iter().map(|e| e.1.times.len()).collect();
    let n_total_obs: usize = band_n_obs.iter().sum();

    let joint_cost = |p: &[f64]| -> f64 {
        let ref_params = &p[..n_model_params];

        // Physical constraint on reference band
        // Villar params: [log_A, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
        let ref_constraint = villar_constraint(ref_params[2], ref_params[1], ref_params[4], ref_params[5]);
        if ref_constraint > 0.0 {
            return 1e12 + 10_000.0 * ref_constraint;
        }

        // Band-balanced likelihood: each band contributes equally
        // NLL_band / n_band * (n_total / n_bands)
        let mut balanced_nll = 0.0;

        // Reference band
        {
            let data = &band_entries[0].1;
            let n = data.times.len();
            let sigma_extra = ref_params[se_idx].exp();
            let sigma_extra_sq = sigma_extra * sigma_extra;
            let mut preds = vec![0.0; n];
            eval_model_batch_into(model, ref_params, &data.times, &mut preds);
            let mut band_nll = 0.0;
            for i in 0..n {
                if !preds[i].is_finite() { return 1e99; }
                let tv = data.obs_var[i] + sigma_extra_sq;
                if data.is_upper[i] {
                    band_nll -= log_normal_cdf((data.upper_flux[i] - preds[i]) / tv.sqrt());
                } else {
                    let diff = preds[i] - data.flux[i];
                    band_nll += diff * diff / tv + tv.ln();
                }
            }
            // Band-balanced: normalize by band size, weight equally
            if n > 0 {
                balanced_nll += (band_nll / n as f64) * (n_total_obs as f64 / n_bands as f64);
            }
        }

        // Non-reference bands
        for b in 1..n_bands {
            let data = &band_entries[b].1;
            let n = data.times.len();
            let off_start = n_model_params + (b - 1) * n_model_params;
            let offsets = &p[off_start..off_start + n_model_params];

            // Reconstruct band params from reference + offsets
            let mut band_params = vec![0.0; n_model_params];
            for j in 0..n_model_params {
                band_params[j] = ref_params[j] + offsets[j];
            }

            // Physical constraint on this band
            let band_constraint = villar_constraint(band_params[2], band_params[1], band_params[4], band_params[5]);
            if band_constraint > 0.0 {
                return 1e12 + 10_000.0 * band_constraint;
            }

            let sigma_extra = band_params[se_idx].exp();
            let sigma_extra_sq = sigma_extra * sigma_extra;
            let mut preds = vec![0.0; n];
            eval_model_batch_into(model, &band_params, &data.times, &mut preds);
            let mut band_nll = 0.0;
            for i in 0..n {
                if !preds[i].is_finite() { return 1e99; }
                let tv = data.obs_var[i] + sigma_extra_sq;
                if data.is_upper[i] {
                    band_nll -= log_normal_cdf((data.upper_flux[i] - preds[i]) / tv.sqrt());
                } else {
                    let diff = preds[i] - data.flux[i];
                    band_nll += diff * diff / tv + tv.ln();
                }
            }
            if n > 0 {
                balanced_nll += (band_nll / n as f64) * (n_total_obs as f64 / n_bands as f64);
            }
        }

        let n_total = n_total_obs.max(1) as f64;

        // Reference band population prior
        let mut neg_lp = 0.0;
        for (j, &(center, width)) in pop_prior_ref.iter().enumerate() {
            if j < n_model_params && width > 0.0 {
                let z = (ref_params[j] - center) / width;
                neg_lp += 0.5 * z * z;
            }
        }

        // Relative priors on band offsets
        for b in 1..n_bands {
            let off_start = n_model_params + (b - 1) * n_model_params;
            for (j, &(mean, sigma, _is_add)) in rel_priors.iter().enumerate() {
                if sigma > 0.0 {
                    let z = (p[off_start + j] - mean) / sigma;
                    neg_lp += 0.5 * z * z;
                }
            }
        }

        (balanced_nll + neg_lp) / n_total
    };

    // Run PSO on the joint space
    let (best_params, best_cost) = {
        let seeds: &[u64] = &[42, 137, 271];
        let mut best = Vec::new();
        let mut best_c = f64::INFINITY;

        // Build a seeded particle with prior means
        let mut seed_particle = vec![0.0; n_joint];
        for j in 0..n_model_params {
            seed_particle[j] = pop_prior_ref[j].0.clamp(joint_lower[j], joint_upper[j]);
        }
        // Set t0 near peak of reference band
        {
            let ref_data = &band_entries[0].1;
            let mut peak_time = ref_data.times[0];
            let mut peak_flux = f64::NEG_INFINITY;
            for i in 0..ref_data.times.len() {
                if !ref_data.is_upper[i] && ref_data.flux[i] > peak_flux {
                    peak_flux = ref_data.flux[i];
                    peak_time = ref_data.times[i];
                }
            }
            seed_particle[t0_idx] = (peak_time - 5.0).clamp(joint_lower[t0_idx], joint_upper[t0_idx]);
        }
        // Offsets at prior means
        for b in 1..n_bands {
            let off = n_model_params + (b - 1) * n_model_params;
            for (j, &(mean, _sigma, _is_add)) in rel_priors.iter().enumerate() {
                seed_particle[off + j] = mean.clamp(joint_lower[off + j], joint_upper[off + j]);
            }
        }

        for (i, &seed) in seeds.iter().enumerate() {
            // JAX-calibrated PSO config (Sushant/villar-pso):
            // 200 particles, 1500 max iters, 60 stall iters
            let (params, cost) = pso_minimize(|p| joint_cost(p), &joint_lower, &joint_upper,
                200, 1500, 60, seed, (i == 1).then_some(seed_particle.as_slice()));
            if cost < best_c {
                best_c = cost;
                best = params;
            }
        }
        (best, best_c)
    };

    if best_params.is_empty() {
        return Vec::new();
    }

    // Extract per-band results
    let ref_params = &best_params[..n_model_params];
    let svi_lr = 0.005;
    let svi_n_steps = 500;
    let svi_n_samples = 16;

    let mut results = Vec::new();
    for (b, (band_name, data, mags_obs)) in band_entries.iter().enumerate() {
        let band_params: Vec<f64> = if b == 0 {
            ref_params.to_vec()
        } else {
            let off_start = n_model_params + (b - 1) * n_model_params;
            let offsets = &best_params[off_start..off_start + n_model_params];
            (0..n_model_params).map(|j| ref_params[j] + offsets[j]).collect()
        };

        // Run per-band SVI/Laplace from the joint solution
        let mut svi_result = match method {
            UncertaintyMethod::Svi => svi_fit(model, data, svi_n_steps, svi_n_samples, svi_lr, Some(&band_params)),
            UncertaintyMethod::Laplace => laplace_fit(model, data, &band_params),
        };
        profile_t0_refine(&mut svi_result, data);
        profile_refine_all(&mut svi_result, data);

        // Mag chi2
        let (preds, svi_flux_scale) = scaled_preds(model, &svi_result.mu, data);
        let (_, pso_flux_scale) = scaled_preds(model, &band_params, data);
        let mut mag_chi2_sum = 0.0;
        let mut mag_chi2_n = 0usize;
        for i in 0..data.times.len() {
            let pred_flux = preds[i] * data.peak_flux_obs;
            if pred_flux > 0.0 && data.flux[i] * data.peak_flux_obs > 0.0 {
                let mag_pred = flux_to_mag(pred_flux, zp);
                let mag_obs = mags_obs[i];
                let mag_err = 1.0857 * data.flux_err[i] / data.flux[i];
                if mag_err > 0.0 {
                    let r = mag_obs - mag_pred;
                    mag_chi2_sum += r * r / (mag_err * mag_err);
                    mag_chi2_n += 1;
                }
            }
        }
        let mag_chi2 = if mag_chi2_n > 0 { mag_chi2_sum / mag_chi2_n as f64 } else { f64::NAN };

        let mut per_model_chi2 = HashMap::new();
        per_model_chi2.insert(SviModelName::Villar, finite_or_none(best_cost));

        // MultiBazin: run on the first (most-populated) band
        let multi_bazin = if b == 0 {
            let mb = fit_multi_bazin(data, None);
            if mb.params.is_empty() { None } else { Some(mb) }
        } else {
            None
        };

        results.push(ParametricBandResult {
            band: band_name.clone(),
            model: SviModelName::Villar,
            pso_params: band_params,
            pso_chi2: finite_or_none(best_cost),
            svi_mu: svi_result.mu,
            svi_log_sigma: svi_result.log_sigma,
            svi_elbo: finite_or_none(svi_result.elbo),
            n_obs: data.times.len(),
            mag_chi2: finite_or_none(mag_chi2),
            pso_flux_scale,
            svi_flux_scale,
            per_model_chi2,
            per_model_params: HashMap::new(),
            uncertainty_method: method,
            multi_bazin,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Worst relative error of the MultiBazin analytic gradient vs central differences.
    fn multi_bazin_worst_grad_error(with_upper_limits: bool) -> f64 {
        // Two pulses on a baseline, noise-free
        let times: Vec<f64> = (0..30).map(|i| i as f64 * 3.0).collect();
        let truth = [0.0, 10.0, 0.5, 2.5, -0.7, 45.0, 1.0, 3.0, 0.02, -3.0];
        let mut flux = vec![0.0; times.len()];
        multi_bazin_eval_batch(&truth, 2, &times, &mut flux);
        let n = times.len();
        let mut is_upper = vec![false; n];
        if with_upper_limits {
            is_upper[0] = true;
            is_upper[n - 1] = true;
        }
        let obs_var = vec![0.01f64.powi(2) + 1e-10; n];
        let upper_flux = vec![0.05; n];

        let mut rng = SmallRng::seed_from_u64(1);
        let (mut buf, mut worst) = (vec![0.0; n], 0.0f64);
        for k in 1..=3 {
            let problem = MultiBazinCost { k, times: &times, flux: &flux, obs_var: &obs_var, is_upper: &is_upper, upper_flux: &upper_flux };
            let (lo, hi) = multi_bazin_bounds(k, (0.0, 87.0));
            for _ in 0..20 {
                let p: Vec<f64> = (0..lo.len()).map(|j| lo[j] + (hi[j] - lo[j]) * (0.3 + 0.4 * rng.random::<f64>())).collect();
                let (mut g, mut scratch) = (vec![0.0; p.len()], vec![0.0; p.len()]);
                let f = problem.value_and_grad(&p, &mut g, &mut scratch);
                if f > 1e3 {
                    continue; // finite differences lose precision at absurd costs
                }
                assert!((f - problem.value(&p, &mut buf)).abs() < 1e-9 * f.abs().max(1.0));
                for j in 0..p.len() {
                    let h = 1e-6 * (hi[j] - lo[j]);
                    let (mut a, mut b) = (p.clone(), p.clone());
                    a[j] += h;
                    b[j] -= h;
                    let fd = (problem.value(&a, &mut buf) - problem.value(&b, &mut buf)) / (2.0 * h);
                    worst = worst.max((fd - g[j]).abs() / fd.abs().max(g[j].abs()).max(1e-4));
                }
            }
        }
        worst
    }

    #[test]
    fn multi_bazin_gradient_matches_finite_differences() {
        let worst = multi_bazin_worst_grad_error(false);
        assert!(worst < 1e-4, "worst relative gradient error {worst:e}");
        // Deep-tail upper limits: phi/Phi is not the exact derivative of the
        // erf approximation there, but must stay close (and finite)
        let worst = multi_bazin_worst_grad_error(true);
        assert!(worst < 5e-2, "worst relative gradient error with upper limits {worst:e}");
    }

    #[test]
    fn decoupled_param_keeps_others_sigma() {
        // Parameter 0's curvature was non-finite (zeroed); parameter 1 is well
        // measured (H = 9 -> sigma = 1/3) and coupled to it.
        let mut hessian = vec![0.0, 1.0, 1.0, 9.0];
        decouple_params(&mut hessian, 2, &[true, false]);
        let sigma: Vec<f64> = log_sigma_from_hessian(&hessian, 2).iter().map(|l| l.exp()).collect();
        assert!(sigma[0] > 100.0, "unmeasured parameter should get the wide fallback, got {}", sigma[0]);
        assert!((sigma[1] - 1.0 / 3.0).abs() < 1e-9, "coupled parameter's sigma changed: {}", sigma[1]);
    }
}

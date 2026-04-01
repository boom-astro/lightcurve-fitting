//! Shared types for GPU backends (CUDA, Metal).
//!
//! This module is always compiled and contains backend-agnostic type definitions
//! used by both the CUDA and Metal GPU backends.

use std::ffi::c_int;

/// Which model to evaluate on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuModelName {
    Bazin,
    Villar,
    Tde,
    Arnett,
    Magnetar,
    ShockCooling,
    Afterglow,
    MetzgerKN,
}

impl GpuModelName {
    /// Number of parameters (including sigma_extra) for each model.
    pub fn n_params(self) -> usize {
        match self {
            GpuModelName::Bazin => 6,
            GpuModelName::Villar => 7,
            GpuModelName::Tde => 7,
            GpuModelName::Arnett => 5,
            GpuModelName::Magnetar => 5,
            GpuModelName::ShockCooling => 5,
            GpuModelName::Afterglow => 6,
            GpuModelName::MetzgerKN => 5,
        }
    }

    /// Convert to the parametric module's model name type.
    pub fn to_svi_name(self) -> crate::parametric::SviModelName {
        match self {
            GpuModelName::Bazin => crate::parametric::SviModelName::Bazin,
            GpuModelName::Villar => crate::parametric::SviModelName::Villar,
            GpuModelName::Tde => crate::parametric::SviModelName::Tde,
            GpuModelName::Arnett => crate::parametric::SviModelName::Arnett,
            GpuModelName::Magnetar => crate::parametric::SviModelName::Magnetar,
            GpuModelName::ShockCooling => crate::parametric::SviModelName::ShockCooling,
            GpuModelName::Afterglow => crate::parametric::SviModelName::Afterglow,
            GpuModelName::MetzgerKN => crate::parametric::SviModelName::MetzgerKN,
        }
    }

    /// Integer model ID matching the GPU kernel's switch statement.
    pub fn model_id(self) -> c_int {
        match self {
            GpuModelName::Bazin => 0,
            GpuModelName::Villar => 1,
            GpuModelName::Tde => 2,
            GpuModelName::Arnett => 3,
            GpuModelName::Magnetar => 4,
            GpuModelName::ShockCooling => 5,
            GpuModelName::Afterglow => 6,
            GpuModelName::MetzgerKN => 7,
        }
    }

    /// PSO parameter bounds (lower, upper) matching the CPU implementation.
    pub fn pso_bounds(self) -> (Vec<f64>, Vec<f64>) {
        match self {
            GpuModelName::Bazin => (
                vec![-3.0, -1.0, -100.0, -2.0, -2.0, -5.0],
                vec![3.0, 1.0, 100.0, 5.0, 6.0, 0.0],
            ),
            GpuModelName::Villar => (
                // JAX-calibrated bounds (Sushant/villar-pso), converted from log10→ln
                // and kept slightly wider for single-band generality.
                // [log_A, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
                vec![-1.5, -0.02, 1.0, -60.0, -1.0, 1.0, -6.0],
                vec![1.5,  0.05,  5.0,  40.0,  3.0, 5.5,  0.0],
            ),
            GpuModelName::Tde => (
                vec![-3.0, -1.0, -100.0, -2.0, -1.0, 0.5, -5.0],
                vec![3.0, 1.0, 100.0, 5.0, 6.0, 4.0, 0.0],
            ),
            GpuModelName::Arnett => (
                vec![-3.0, -100.0, 0.5, -3.0, -5.0],
                vec![3.0, 100.0, 4.5, 3.0, 0.0],
            ),
            GpuModelName::Magnetar => (
                vec![-3.0, -100.0, 0.0, 0.5, -5.0],
                vec![3.0, 100.0, 6.0, 4.5, 0.0],
            ),
            GpuModelName::ShockCooling => (
                vec![-3.0, -100.0, 0.1, -1.0, -5.0],
                vec![3.0, 100.0, 3.0, 4.0, 0.0],
            ),
            GpuModelName::Afterglow => (
                vec![-3.0, -100.0, -2.0, -2.0, 0.5, -5.0],
                vec![3.0, 100.0, 6.0, 3.0, 5.0, 0.0],
            ),
            GpuModelName::MetzgerKN => (
                vec![-3.0, -2.0, -1.0, -2.0, -5.0],
                vec![-0.5, -0.5, 2.0, 1.0, 0.0],
            ),
        }
    }
}

/// All GPU-supported models.
pub const ALL_GPU_MODELS: &[GpuModelName] = &[
    GpuModelName::Bazin,
    GpuModelName::Arnett,
    GpuModelName::Tde,
    GpuModelName::Afterglow,
    GpuModelName::Villar,
    GpuModelName::Magnetar,
    GpuModelName::ShockCooling,
    GpuModelName::MetzgerKN,
];

// ---------------------------------------------------------------------------
// Source data for batch operations
// ---------------------------------------------------------------------------

/// Pre-normalized observation data for one source.
#[derive(Clone)]
pub struct BatchSource {
    pub times: Vec<f64>,
    pub flux: Vec<f64>,
    pub obs_var: Vec<f64>,
    pub is_upper: Vec<bool>,
    pub upper_flux: Vec<f64>,
}

/// Result of GPU batch PSO for one source.
#[derive(Debug, Clone)]
pub struct BatchPsoResult {
    pub params: Vec<f64>,
    pub cost: f64,
}

// ---------------------------------------------------------------------------
// GP types
// ---------------------------------------------------------------------------

/// Input data for one band in batch GP fitting.
pub struct GpBandInput {
    pub times: Vec<f64>,
    pub mags: Vec<f64>,
    pub noise_var: Vec<f64>,
}

/// Output of GPU GP fitting for one band.
pub struct GpBandOutput {
    /// Reconstructed DenseGP for additional predictions (thermal reuse, etc.)
    pub dense_gp: Option<crate::sparse_gp::DenseGP>,
    /// Mean predictions at the shared grid points (50 points).
    pub pred_grid: Vec<f64>,
    /// Std predictions at the shared grid points.
    pub std_grid: Vec<f64>,
    /// Mean predictions at the band's observation points (for chi2).
    pub pred_at_obs: Vec<f64>,
}

// ---------------------------------------------------------------------------
// 2D GP types
// ---------------------------------------------------------------------------

/// Input data for one source's 2D GP fit.
pub struct Gp2dInput {
    pub times: Vec<f64>,
    pub waves: Vec<f64>,  // log10(wavelength_angstrom) for each obs
    pub mags: Vec<f64>,
    pub noise_var: Vec<f64>,
}

/// Output from batch 2D GP fitting for one source.
pub struct Gp2dOutput {
    pub pred_grid: Vec<f64>,
    pub std_grid: Vec<f64>,
    pub amp: f64,
    pub ls_time: f64,
    pub ls_wave: f64,
    pub train_rms: f64,
    pub n_train: usize,
    pub success: bool,
}

// ---------------------------------------------------------------------------
// SVI types
// ---------------------------------------------------------------------------

/// Input for GPU SVI fit for one source.
pub struct SviBatchInput {
    pub model_id: usize,
    pub pso_params: Vec<f64>,
    pub se_idx: usize,
    pub prior_centers: Vec<f64>,
    pub prior_widths: Vec<f64>,
}

/// Output from GPU SVI fit for one source.
#[derive(Clone, Debug)]
pub struct SviBatchOutput {
    pub mu: Vec<f64>,
    pub log_sigma: Vec<f64>,
    pub elbo: f64,
}

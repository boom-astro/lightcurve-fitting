pub mod batch;
pub mod common;
pub mod gp;
pub mod nonparametric;
pub mod parametric;
pub mod thermal;

pub use batch::{FastFitResult, fit_batch_fast, fit_batch_parametric};
pub use common::{BandData, LightcurveFittingResult, build_mag_bands, build_flux_bands};
pub use nonparametric::{fit_nonparametric, NonparametricBandResult};
pub use parametric::{eval_model_flux, fit_parametric, ParametricBandResult, SviModelName};
pub use thermal::{fit_thermal, ThermalResult};

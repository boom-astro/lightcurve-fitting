pub mod common;
pub mod gp;
pub mod nonparametric;
pub mod parametric;
pub mod thermal;

pub use common::{BandData, LightcurveFittingResult, build_mag_bands, build_flux_bands};
pub use nonparametric::{fit_nonparametric, NonparametricBandResult};
pub use parametric::{fit_parametric, ParametricBandResult, SviModelName};
pub use thermal::{fit_thermal, ThermalResult};

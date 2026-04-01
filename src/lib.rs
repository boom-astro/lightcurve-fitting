pub mod batch;
pub mod common;
pub mod gp;
pub mod gpu_types;
#[cfg(feature = "cuda")]
pub mod gpu_cuda;
#[cfg(feature = "metal")]
pub mod gpu_metal;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod gpu;
pub mod nonparametric;
pub mod parametric;
pub mod gp2d;
pub mod sparse_gp;
pub mod thermal;
pub mod features;

pub use batch::{FastFitResult, fit_batch_fast, fit_batch_parametric};
pub use common::{BandData, LightcurveFittingResult, build_mag_bands, build_flux_bands, build_raw_flux_bands};
pub use nonparametric::{fit_nonparametric, fit_nonparametric_with_opts, NonparametricBandResult};
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use nonparametric::{fit_nonparametric_batch_gpu, fit_nonparametric_batch_gpu_with_opts};
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use gpu::{GpBandInput, GpBandOutput};
pub use parametric::{eval_model_flux, metzger_kn_mags, fit_parametric, fit_parametric_model, fit_parametric_multiband, finalize_parametric_from_gpu, finalize_parametric_with_gpu_svi, finalize_all_models_with_gpu_svi, svi_prior_for_model, svi_model_meta, GpuPsoBandResult, MultiBazinResult, ParametricBandResult, SviModelName, UncertaintyMethod};
pub use thermal::{fit_thermal, ThermalResult};
pub use gp::fit_gp_predict;
pub use gp2d::{fit_gp_2d, fit_gp_2d_with_thermal, get_band_wavelength, DenseGP2D, Gp2dResult, Gp2dThermalResult};
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use gp2d::fit_gp_2d_batch_gpu;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use gpu::{Gp2dInput, Gp2dOutput};
#[cfg(any(feature = "cuda", feature = "metal"))]
pub use gpu::{SviBatchInput, SviBatchOutput};
pub use features::{FeatureMap, extract_features, extract_features_from_results, extract_features_batch};

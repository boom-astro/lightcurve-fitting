use std::collections::HashMap;

use rayon::prelude::*;

use crate::common::BandData;
use crate::nonparametric::{fit_nonparametric, NonparametricBandResult};
use crate::parametric::{fit_parametric, ParametricBandResult};
use crate::thermal::{fit_thermal, ThermalResult};

/// Result of the fast (nonparametric + thermal) fitting path for a single source.
#[derive(Debug, Clone)]
pub struct FastFitResult {
    pub nonparametric: Vec<NonparametricBandResult>,
    pub thermal: Option<ThermalResult>,
}

/// Run nonparametric + thermal fitting on a batch of sources in parallel.
///
/// Each source is represented as pre-built per-band magnitude data.
/// Sources are processed independently via Rayon.
pub fn fit_batch_fast(sources: &[HashMap<String, BandData>]) -> Vec<FastFitResult> {
    sources
        .par_iter()
        .map(|mag_bands| {
            let (nonparametric, gps) = fit_nonparametric(mag_bands);
            let thermal = fit_thermal(mag_bands, Some(&gps));
            FastFitResult {
                nonparametric,
                thermal,
            }
        })
        .collect()
}

/// Run parametric fitting on a batch of sources in parallel.
///
/// Each source is represented as pre-built per-band flux data.
/// Sources are processed independently via Rayon.
pub fn fit_batch_parametric(sources: &[HashMap<String, BandData>]) -> Vec<Vec<ParametricBandResult>> {
    sources
        .par_iter()
        .map(|flux_bands| fit_parametric(flux_bands))
        .collect()
}

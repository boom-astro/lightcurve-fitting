use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};
use pythonize::pythonize;

use ::lightcurve_fitting::{
    build_flux_bands as rs_build_flux_bands, build_raw_flux_bands as rs_build_raw_flux_bands,
    build_mag_bands as rs_build_mag_bands,
    eval_model_flux as rs_eval_model_flux, fit_batch_fast as rs_fit_batch_fast,
    fit_batch_parametric as rs_fit_batch_parametric,
    fit_nonparametric as rs_fit_nonparametric,
    fit_nonparametric_with_opts as rs_fit_nonparametric_with_opts,
    fit_parametric as rs_fit_parametric,
    fit_thermal as rs_fit_thermal, fit_gp_predict as rs_fit_gp_predict,
    fit_gp_2d_with_thermal as rs_fit_gp_2d_with_thermal,
    metzger_kn_mags as rs_metzger_kn_mags,
    extract_features as rs_extract_features,
    extract_features_batch as rs_extract_features_batch,
    BandData, FastFitResult, SviModelName, UncertaintyMethod,
};

// ---------------------------------------------------------------------------
// Opaque band-data wrapper
// ---------------------------------------------------------------------------

/// Opaque wrapper around per-band photometry data.
///
/// Construct via `build_mag_bands()`, `build_flux_bands()`, or
/// `BandDataMap.from_dict()`.
#[pyclass(name = "BandDataMap")]
#[derive(Clone)]
pub struct PyBandDataMap {
    inner: HashMap<String, BandData>,
}

#[pymethods]
impl PyBandDataMap {
    /// Create from a Python dict of ``{band_name: (times, values, errors)}``.
    ///
    /// Each value can be a tuple or list of three equal-length sequences of floats.
    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut map = HashMap::new();
        for (key, value) in dict.iter() {
            let band_name: String = key.extract()?;
            let (times, values, errors): (Vec<f64>, Vec<f64>, Vec<f64>) = value.extract()?;
            map.insert(
                band_name,
                BandData {
                    times,
                    values,
                    errors,
                },
            );
        }
        Ok(Self { inner: map })
    }

    /// Convert to a Python dict of ``{band_name: (times, values, errors)}``.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, data) in &self.inner {
            let t = PyList::new(py, &data.times)?;
            let v = PyList::new(py, &data.values)?;
            let e = PyList::new(py, &data.errors)?;
            let tuple = PyTuple::new(py, [t.into_any(), v.into_any(), e.into_any()])?;
            dict.set_item(name, tuple)?;
        }
        Ok(dict)
    }

    /// Number of bands.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let mut bands: Vec<&str> = self.inner.keys().map(|s| s.as_str()).collect();
        bands.sort();
        format!("BandDataMap(bands={:?})", bands)
    }
}

// ---------------------------------------------------------------------------
// Band construction helpers
// ---------------------------------------------------------------------------

/// Build per-band magnitude data from flat photometry arrays.
///
/// Groups observations by band name and converts times to relative days
/// (from minimum JD). Non-finite values are dropped.
#[pyfunction]
fn build_mag_bands(
    times: Vec<f64>,
    mags: Vec<f64>,
    mag_errs: Vec<f64>,
    bands: Vec<String>,
) -> PyResult<PyBandDataMap> {
    let inner = rs_build_mag_bands(&times, &mags, &mag_errs, &bands);
    Ok(PyBandDataMap { inner })
}

/// Build per-band flux data from flat magnitude arrays.
///
/// Like `build_mag_bands` but converts magnitudes to flux (ZP = 23.9).
#[pyfunction]
fn build_flux_bands(
    times: Vec<f64>,
    mags: Vec<f64>,
    mag_errs: Vec<f64>,
    bands: Vec<String>,
) -> PyResult<PyBandDataMap> {
    let inner = rs_build_flux_bands(&times, &mags, &mag_errs, &bands);
    Ok(PyBandDataMap { inner })
}

/// Build per-band flux data from raw (already linear) flux arrays.
///
/// Use when flux values are already in linear units (not magnitudes).
#[pyfunction]
fn build_raw_flux_bands(
    times: Vec<f64>,
    fluxes: Vec<f64>,
    flux_errs: Vec<f64>,
    bands: Vec<String>,
) -> PyResult<PyBandDataMap> {
    let inner = rs_build_raw_flux_bands(&times, &fluxes, &flux_errs, &bands);
    Ok(PyBandDataMap { inner })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_uncertainty_method(method: &str) -> PyResult<UncertaintyMethod> {
    match method.to_lowercase().as_str() {
        "svi" => Ok(UncertaintyMethod::Svi),
        "laplace" => Ok(UncertaintyMethod::Laplace),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown uncertainty method '{}'. Expected 'svi' or 'laplace'.",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Individual fitters
// ---------------------------------------------------------------------------

/// Fit nonparametric GP models to all bands.
///
/// Returns a list of dicts, one per band, containing GP-derived features
/// (peak_mag, rise_time, decay_time, FWHM, derivatives, etc.).
///
/// Args:
///     bands: Magnitude-space band data.
///     max_subsample: Maximum number of points for the dense GP subsample
///         (default 25). Only affects bands with >100 observations that use
///         the sparse FITC approximation — controls the secondary DenseGP
///         used for thermal reuse.
#[pyfunction]
#[pyo3(signature = (bands, max_subsample=25))]
fn fit_nonparametric(py: Python<'_>, bands: &PyBandDataMap, max_subsample: usize) -> PyResult<PyObject> {
    let inner = bands.inner.clone();
    let results = py.allow_threads(|| {
        let (results, _gps) = rs_fit_nonparametric_with_opts(&inner, max_subsample);
        results
    });
    Ok(pythonize(py, &results)?.unbind())
}

/// Fit parametric lightcurve models (Bazin, Villar, TDE, etc.) to all bands.
///
/// **Input must be flux data** (from `build_flux_bands`).
///
/// Args:
///     bands: Flux-space band data.
///     fit_all_models: If True, return per-model chi2 and parameters for
///         every model, not just the best.
///
/// Returns a list of dicts, one per band.
#[pyfunction]
#[pyo3(signature = (bands, fit_all_models=false, method="svi"))]
fn fit_parametric(
    py: Python<'_>,
    bands: &PyBandDataMap,
    fit_all_models: bool,
    method: &str,
) -> PyResult<PyObject> {
    let method = parse_uncertainty_method(method)?;
    let inner = bands.inner.clone();
    let results = py.allow_threads(|| rs_fit_parametric(&inner, fit_all_models, method));
    Ok(pythonize(py, &results)?.unbind())
}

/// Fit a blackbody temperature model to cross-band color differences.
///
/// Returns a dict with temperature and cooling-rate estimates, or None if
/// there are fewer than two photometric bands.
#[pyfunction]
fn fit_thermal(py: Python<'_>, bands: &PyBandDataMap) -> PyResult<PyObject> {
    let inner = bands.inner.clone();
    let result = py.allow_threads(|| rs_fit_thermal(&inner, None));
    Ok(pythonize(py, &result)?.unbind())
}

/// Combined nonparametric + thermal fitting that reuses GP fits internally.
///
/// Equivalent to calling `fit_nonparametric` followed by `fit_thermal`, but
/// avoids refitting the reference-band GP for the thermal step.
///
/// Returns a dict with ``"nonparametric"`` (list of dicts) and ``"thermal"``
/// (dict or None) keys.
#[pyfunction]
#[pyo3(signature = (bands, max_subsample=25))]
fn fit_fast(py: Python<'_>, bands: &PyBandDataMap, max_subsample: usize) -> PyResult<PyObject> {
    let inner = bands.inner.clone();
    let result = py.allow_threads(|| {
        let (nonpar, gps) = rs_fit_nonparametric_with_opts(&inner, max_subsample);
        let thermal = rs_fit_thermal(&inner, Some(&gps));
        FastFitResult {
            nonparametric: nonpar,
            thermal,
        }
    });
    Ok(pythonize(py, &result)?.unbind())
}

// ---------------------------------------------------------------------------
// Model evaluation
// ---------------------------------------------------------------------------

/// Evaluate a parametric model at the given times.
///
/// Args:
///     model: Model name string (e.g. ``"Bazin"``, ``"Villar"``).
///         Pass ``result['model']`` from a ``fit_parametric`` result.
///     params: Internal (transformed) parameter vector.
///         Pass ``result['pso_params']`` or ``result['svi_mu']`` directly.
///     times: Time values (relative days) at which to evaluate.
///
/// Returns a list of predicted flux values, one per time point.
#[pyfunction]
fn eval_model(model: &str, params: Vec<f64>, times: Vec<f64>) -> PyResult<Vec<f64>> {
    let m: SviModelName = serde_json::from_value(serde_json::Value::String(model.to_string()))
        .map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown model '{}'. Expected one of: Bazin, Villar, MetzgerKN, Tde, Arnett, Magnetar, ShockCooling, Afterglow",
                model
            ))
        })?;
    Ok(rs_eval_model_flux(m, &params, &times))
}

// ---------------------------------------------------------------------------
// Batch fitters
// ---------------------------------------------------------------------------

/// Batch nonparametric + thermal fitting for multiple sources (parallel).
///
/// Args:
///     sources: List of `BandDataMap` objects.
///
/// Returns a list of dicts (one per source), each containing
/// ``"nonparametric"`` and ``"thermal"`` keys.
#[pyfunction]
fn fit_batch_fast(py: Python<'_>, sources: Vec<PyBandDataMap>) -> PyResult<PyObject> {
    let inner: Vec<HashMap<String, BandData>> = sources.into_iter().map(|s| s.inner).collect();
    let results = py.allow_threads(|| rs_fit_batch_fast(&inner));
    Ok(pythonize(py, &results)?.unbind())
}

/// Batch parametric fitting for multiple sources (parallel).
///
/// Args:
///     sources: List of `BandDataMap` objects (flux data).
///     fit_all_models: If True, return all model fits per band.
///
/// Returns a list of (list of dicts), one inner list per source.
#[pyfunction]
#[pyo3(signature = (sources, fit_all_models=false, method="svi"))]
fn fit_batch_parametric(
    py: Python<'_>,
    sources: Vec<PyBandDataMap>,
    fit_all_models: bool,
    method: &str,
) -> PyResult<PyObject> {
    let method = parse_uncertainty_method(method)?;
    let inner: Vec<HashMap<String, BandData>> = sources.into_iter().map(|s| s.inner).collect();
    let results = py.allow_threads(|| rs_fit_batch_parametric(&inner, fit_all_models, method));
    Ok(pythonize(py, &results)?.unbind())
}

// ---------------------------------------------------------------------------
// GP prediction
// ---------------------------------------------------------------------------

/// Fit a Gaussian Process to training data and predict at query points.
///
/// Performs a grid search over amplitude and lengthscale candidates, using
/// the mean squared measurement error as the GP alpha parameter.
///
/// Args:
///     train_times: Training time values.
///     train_values: Training observed values (magnitudes).
///     train_errors: Training measurement errors (mag errors).
///     query_times: Times at which to predict.
///     amp_candidates: Amplitude values to try in grid search.
///     ls_candidates: Lengthscale values to try in grid search.
///     snr_threshold: If set, exclude points with flux SNR below this
///         value before fitting (default: None = keep all points).
///         A typical value is 3.0.
///
/// Returns a tuple of (predictions, std_devs) or None if fitting fails.
#[pyfunction]
#[pyo3(signature = (train_times, train_values, train_errors, query_times, amp_candidates, ls_candidates, snr_threshold=None))]
fn fit_gp_predict(
    py: Python<'_>,
    train_times: Vec<f64>,
    train_values: Vec<f64>,
    train_errors: Vec<f64>,
    query_times: Vec<f64>,
    amp_candidates: Vec<f64>,
    ls_candidates: Vec<f64>,
    snr_threshold: Option<f64>,
) -> PyResult<Option<(Vec<f64>, Vec<f64>)>> {
    let result = py.allow_threads(|| {
        rs_fit_gp_predict(
            &train_times,
            &train_values,
            &train_errors,
            &query_times,
            &amp_candidates,
            &ls_candidates,
            snr_threshold,
        )
    });
    Ok(result)
}

// ---------------------------------------------------------------------------
// Kilonova model
// ---------------------------------------------------------------------------

/// Evaluate the Metzger kilonova model in physical magnitude space.
///
/// Unlike `eval_model("MetzgerKN", ...)` which returns normalized flux,
/// this evaluates the full physical model and returns AB magnitudes per band.
///
/// Args:
///     params: Physical parameters [log10(M_ej/Msun), log10(v_ej/c),
///         log10(kappa), t0_offset].
///     times: Observation times (days).
///     band_frequencies: List of (band_name, frequency_hz) tuples, e.g.
///         [("g", 6.3e14), ("r", 4.7e14), ("i", 3.9e14)].
///     d_l_cm: Luminosity distance in cm.
///
/// Returns a dict of ``{band_name: [mag_values]}``.
#[pyfunction]
fn metzger_kn_mags(
    py: Python<'_>,
    params: Vec<f64>,
    times: Vec<f64>,
    band_frequencies: Vec<(String, f64)>,
    d_l_cm: f64,
) -> PyResult<PyObject> {
    let bands: Vec<(&str, f64)> = band_frequencies.iter().map(|(n, f)| (n.as_str(), *f)).collect();
    let result = py.allow_threads(|| rs_metzger_kn_mags(&params, &times, &bands, d_l_cm));
    Ok(pythonize(py, &result)?.unbind())
}

// ---------------------------------------------------------------------------
// Batch nonparametric
// ---------------------------------------------------------------------------

/// Batch nonparametric GP fitting for multiple sources (parallel).
///
/// Like `fit_batch_fast` but without the thermal step.
///
/// Args:
///     sources: List of `BandDataMap` objects (magnitude data).
///
/// Returns a list of (list of dicts), one inner list per source.
#[pyfunction]
fn fit_batch_nonparametric(py: Python<'_>, sources: Vec<PyBandDataMap>) -> PyResult<PyObject> {
    let inner: Vec<HashMap<String, BandData>> = sources.into_iter().map(|s| s.inner).collect();
    let results: Vec<Vec<::lightcurve_fitting::NonparametricBandResult>> = py.allow_threads(|| {
        use rayon::prelude::*;
        inner.par_iter()
            .map(|bands| {
                let (results, _gps) = rs_fit_nonparametric(bands);
                results
            })
            .collect()
    });
    Ok(pythonize(py, &results)?.unbind())
}

// ---------------------------------------------------------------------------
// 2D GP
// ---------------------------------------------------------------------------

/// Fit a 2D GP (time x wavelength) to multi-band magnitude data and extract
/// blackbody temperature evolution from the resulting SED surface.
///
/// Returns a dict with "gp" (GP hyperparameters and diagnostics) and "thermal"
/// (temperature evolution) sub-dicts, or None if insufficient data.
#[pyfunction]
#[pyo3(signature = (bands, n_time_steps=50))]
fn fit_gp_2d(py: Python<'_>, bands: &PyBandDataMap, n_time_steps: usize) -> PyResult<PyObject> {
    let result = py.allow_threads(|| rs_fit_gp_2d_with_thermal(&bands.inner, n_time_steps));
    match result {
        Some((gp_result, thermal_result)) => {
            let dict = PyDict::new(py);
            dict.set_item("gp", pythonize(py, &gp_result)?)?;
            dict.set_item("thermal", pythonize(py, &thermal_result)?)?;
            Ok(dict.into_any().unbind())
        }
        None => Ok(py.None()),
    }
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

/// Extract ~80 features from a single source by running all fitters.
///
/// Args:
///     mag_bands: Magnitude-space band data (for nonparametric, thermal, 2D GP).
///     flux_bands: Flux-space band data (for parametric).
///     ref_band: Reference band for feature extraction (default "r").
///
/// Returns a dict of {feature_name: value_or_None}.
#[pyfunction]
#[pyo3(signature = (mag_bands, flux_bands, ref_band="r"))]
fn extract_features(py: Python<'_>, mag_bands: &PyBandDataMap, flux_bands: &PyBandDataMap, ref_band: &str) -> PyResult<PyObject> {
    let mag = mag_bands.inner.clone();
    let flux = flux_bands.inner.clone();
    let rb = ref_band.to_string();
    let features = py.allow_threads(move || rs_extract_features(&mag, &flux, &rb));
    let dict = PyDict::new(py);
    for (k, v) in &features {
        dict.set_item(k, *v)?;
    }
    Ok(dict.into_any().unbind())
}

/// Extract features for multiple sources in parallel.
///
/// Args:
///     mag_sources: List of magnitude-space BandDataMap objects.
///     flux_sources: List of flux-space BandDataMap objects.
///     ref_band: Reference band (default "r").
///
/// Returns a list of feature dicts.
#[pyfunction]
#[pyo3(signature = (mag_sources, flux_sources, ref_band="r"))]
fn extract_features_batch(py: Python<'_>, mag_sources: Vec<PyBandDataMap>, flux_sources: Vec<PyBandDataMap>, ref_band: &str) -> PyResult<PyObject> {
    let mags: Vec<HashMap<String, BandData>> = mag_sources.into_iter().map(|s| s.inner).collect();
    let fluxes: Vec<HashMap<String, BandData>> = flux_sources.into_iter().map(|s| s.inner).collect();
    let rb = ref_band.to_string();
    let all_features = py.allow_threads(move || rs_extract_features_batch(&mags, &fluxes, &rb));
    let dicts: Vec<_> = all_features.iter().map(|features| {
        let dict = PyDict::new(py);
        for (k, v) in features {
            dict.set_item(k, *v).unwrap();
        }
        dict.into_any()
    }).collect();
    let list = PyList::new(py, &dicts)?;
    Ok(list.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn lightcurve_fitting(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBandDataMap>()?;
    m.add_function(wrap_pyfunction!(build_mag_bands, m)?)?;
    m.add_function(wrap_pyfunction!(build_flux_bands, m)?)?;
    m.add_function(wrap_pyfunction!(build_raw_flux_bands, m)?)?;
    m.add_function(wrap_pyfunction!(fit_nonparametric, m)?)?;
    m.add_function(wrap_pyfunction!(fit_parametric, m)?)?;
    m.add_function(wrap_pyfunction!(fit_thermal, m)?)?;
    m.add_function(wrap_pyfunction!(fit_fast, m)?)?;
    m.add_function(wrap_pyfunction!(eval_model, m)?)?;
    m.add_function(wrap_pyfunction!(fit_batch_fast, m)?)?;
    m.add_function(wrap_pyfunction!(fit_batch_nonparametric, m)?)?;
    m.add_function(wrap_pyfunction!(fit_batch_parametric, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gp_predict, m)?)?;
    m.add_function(wrap_pyfunction!(metzger_kn_mags, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gp_2d, m)?)?;
    m.add_function(wrap_pyfunction!(extract_features, m)?)?;
    m.add_function(wrap_pyfunction!(extract_features_batch, m)?)?;
    Ok(())
}

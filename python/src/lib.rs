use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};
use pythonize::pythonize;

use ::lightcurve_fitting::{
    build_flux_bands as rs_build_flux_bands, build_mag_bands as rs_build_mag_bands,
    eval_model_flux as rs_eval_model_flux, fit_batch_fast as rs_fit_batch_fast,
    fit_batch_parametric as rs_fit_batch_parametric,
    fit_nonparametric as rs_fit_nonparametric, fit_parametric as rs_fit_parametric,
    fit_thermal as rs_fit_thermal, BandData, FastFitResult, SviModelName, UncertaintyMethod,
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
#[pyfunction]
fn fit_nonparametric(py: Python<'_>, bands: &PyBandDataMap) -> PyResult<PyObject> {
    let inner = bands.inner.clone();
    let results = py.allow_threads(|| {
        let (results, _gps) = rs_fit_nonparametric(&inner);
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
fn fit_fast(py: Python<'_>, bands: &PyBandDataMap) -> PyResult<PyObject> {
    let inner = bands.inner.clone();
    let result = py.allow_threads(|| {
        let (nonpar, gps) = rs_fit_nonparametric(&inner);
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
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn lightcurve_fitting(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBandDataMap>()?;
    m.add_function(wrap_pyfunction!(build_mag_bands, m)?)?;
    m.add_function(wrap_pyfunction!(build_flux_bands, m)?)?;
    m.add_function(wrap_pyfunction!(fit_nonparametric, m)?)?;
    m.add_function(wrap_pyfunction!(fit_parametric, m)?)?;
    m.add_function(wrap_pyfunction!(fit_thermal, m)?)?;
    m.add_function(wrap_pyfunction!(fit_fast, m)?)?;
    m.add_function(wrap_pyfunction!(eval_model, m)?)?;
    m.add_function(wrap_pyfunction!(fit_batch_fast, m)?)?;
    m.add_function(wrap_pyfunction!(fit_batch_parametric, m)?)?;
    Ok(())
}

# API Reference

Complete Python API reference for the `lightcurve_fitting` module.

---

## Classes

### `BandDataMap`

Opaque wrapper around per-band photometry data. Construct via
`build_mag_bands()`, `build_flux_bands()`, or `BandDataMap.from_dict()`.

#### `BandDataMap.from_dict(dict) -> BandDataMap` (classmethod)

Create from a Python dict of `{band_name: (times, values, errors)}`.

Each value must be a tuple or list of three equal-length sequences of floats.

**Args:**
- `dict` (dict[str, tuple[list[float], list[float], list[float]]]): Band data
  keyed by band name.

**Returns:** `BandDataMap`

```python
bands = BandDataMap.from_dict({
    "g": ([0.0, 1.0, 2.0], [18.5, 18.2, 18.8], [0.05, 0.04, 0.06]),
    "r": ([0.0, 1.0, 2.0], [18.8, 18.5, 19.0], [0.06, 0.05, 0.07]),
})
```

#### `BandDataMap.to_dict() -> dict`

Convert to a Python dict of `{band_name: (times, values, errors)}`.

**Returns:** `dict[str, tuple[list[float], list[float], list[float]]]`

#### `len(bands) -> int`

Returns the number of bands.

---

## Band Construction

### `build_mag_bands(times, mags, mag_errs, bands) -> BandDataMap`

Build per-band magnitude data from flat photometry arrays.

Groups observations by band name and converts times to relative days (from
minimum JD). Non-finite values are dropped.

**Args:**
- `times` (list[float]): Observation times (e.g. JD or MJD).
- `mags` (list[float]): Magnitudes.
- `mag_errs` (list[float]): Magnitude errors.
- `bands` (list[str]): Band name for each observation (e.g. `"ztfg"`, `"ztfr"`).

**Returns:** `BandDataMap` in magnitude space.

### `build_flux_bands(times, mags, mag_errs, bands) -> BandDataMap`

Build per-band flux data from flat magnitude arrays.

Like `build_mag_bands` but converts magnitudes to flux (zero-point = 23.9).

**Args:**
- `times` (list[float]): Observation times.
- `mags` (list[float]): Magnitudes.
- `mag_errs` (list[float]): Magnitude errors.
- `bands` (list[str]): Band name for each observation.

**Returns:** `BandDataMap` in flux space.

---

## Individual Fitters

### `fit_nonparametric(bands, max_subsample=25) -> list[dict]`

Fit nonparametric GP models to all bands.

**Args:**
- `bands` (`BandDataMap`): Magnitude-space band data.
- `max_subsample` (int, default 25): Maximum number of points for the dense GP
  subsample. Only affects bands with >100 observations that use the sparse FITC
  approximation.

**Returns:** `list[dict]` -- one dict per band, with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `band` | str | Band name |
| `n_obs` | int | Number of observations in this band |
| `rise_time` | float or null | Rise time from half-max to peak (days) |
| `decay_time` | float or null | Decay time from peak to half-max (days) |
| `t0` | float or null | Time of GP peak (relative days) |
| `peak_mag` | float or null | Peak magnitude from GP |
| `chi2` | float or null | GP fit chi-squared |
| `baseline_chi2` | float or null | Chi-squared of a flat (baseline) model |
| `fwhm` | float or null | Full width at half maximum (days) |
| `rise_rate` | float or null | Rise rate (mag/day) |
| `decay_rate` | float or null | Decay rate (mag/day) |
| `gp_dfdt_now` | float or null | GP first derivative at current epoch |
| `gp_dfdt_next` | float or null | GP first derivative one step ahead |
| `gp_d2fdt2_now` | float or null | GP second derivative at current epoch |
| `gp_predicted_mag_1d` | float or null | GP-predicted magnitude 1 day ahead |
| `gp_predicted_mag_2d` | float or null | GP-predicted magnitude 2 days ahead |
| `gp_time_to_peak` | float or null | Time from current epoch to GP peak |
| `gp_extrap_slope` | float or null | Extrapolation slope beyond data |
| `gp_sigma_f` | float or null | GP signal amplitude hyperparameter |
| `gp_peak_to_peak` | float or null | Peak-to-peak magnitude range in GP |
| `gp_snr_max` | float or null | Maximum signal-to-noise ratio |
| `gp_dfdt_max` | float or null | Maximum first derivative of GP |
| `gp_dfdt_min` | float or null | Minimum first derivative of GP |
| `gp_frac_of_peak` | float or null | Current magnitude as fraction of peak |
| `gp_post_var_mean` | float or null | Mean GP posterior variance |
| `gp_post_var_max` | float or null | Maximum GP posterior variance |
| `gp_skewness` | float or null | Skewness of GP magnitude distribution |
| `gp_kurtosis` | float or null | Kurtosis of GP magnitude distribution |
| `gp_n_inflections` | float or null | Number of inflection points in GP |
| `decay_power_law_index` | float or null | Slope of mag vs log10(t - t_peak) post-peak. TDEs: ~4.2, SN Ia: steep (exponential) |
| `decay_power_law_chi2` | float or null | Chi-squared of the power-law decay fit |
| `mag_at_30d` | float or null | GP-predicted magnitude at peak + 30 days |
| `mag_at_60d` | float or null | GP-predicted magnitude at peak + 60 days |
| `mag_at_90d` | float or null | GP-predicted magnitude at peak + 90 days |
| `von_neumann_ratio` | float or null | Von Neumann ratio of raw magnitudes. Low (~0.1-0.5) for smooth (TDE), high (~1.5-2.0) for stochastic (AGN) |
| `pre_peak_rms` | float or null | Std dev of raw magnitudes before GP peak |
| `rise_amplitude_over_noise` | float or null | (pre-peak mean - peak mag) / median error |
| `post_peak_monotonicity` | float or null | Fraction of consecutive post-peak GP steps where mag increases. ~1.0 for TDE, ~0.5 for AGN |

### `fit_parametric(bands, fit_all_models=False, method="svi") -> list[dict]`

Fit parametric lightcurve models (Bazin, Villar, TDE, etc.) to all bands.

**Input must be flux data** (from `build_flux_bands`).

**Args:**
- `bands` (`BandDataMap`): Flux-space band data.
- `fit_all_models` (bool, default False): If True, return per-model chi2 and
  parameters for every model, not just the best.
- `method` (str, default `"svi"`): Uncertainty method. One of `"svi"` or
  `"laplace"`.

**Returns:** `list[dict]` -- one dict per band, with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `band` | str | Band name |
| `model` | str | Best-fit model name (e.g. `"Bazin"`, `"Villar"`, `"Tde"`, `"MetzgerKN"`, `"Arnett"`, `"Magnetar"`, `"ShockCooling"`, `"Afterglow"`) |
| `pso_params` | list[float] | PSO best-fit parameter vector (internal/transformed space) |
| `pso_chi2` | float or null | Chi-squared of the PSO best fit |
| `svi_mu` | list[float] | SVI/Laplace posterior mean parameter vector |
| `svi_log_sigma` | list[float] | SVI/Laplace log-posterior-std parameter vector |
| `svi_elbo` | float or null | SVI evidence lower bound (or Laplace log-evidence) |
| `n_obs` | int | Number of observations in this band |
| `mag_chi2` | float or null | Chi-squared in magnitude space |
| `per_model_chi2` | dict[str, float or null] | Chi-squared for each model attempted |
| `per_model_params` | dict[str, list[float]] | PSO params for each model (only when `fit_all_models=True`) |
| `uncertainty_method` | str | Which uncertainty method was used (`"Svi"` or `"Laplace"`) |
| `multi_bazin` | dict or null | MultiBazin greedy fit result (first band only). Keys: `best_k` (int), `params` (list), `cost` (float), `bic` (float), `per_k_cost` (list), `per_k_bic` (list). `best_k > 1` indicates multi-peaked or recurrent behavior. |

### `fit_thermal(bands) -> dict or None`

Fit a blackbody temperature model to cross-band color differences.

**Args:**
- `bands` (`BandDataMap`): Magnitude-space band data. Requires at least two
  bands with >= 5 observations each.

**Returns:** `dict` or `None`. Dict keys:

| Key | Type | Description |
|-----|------|-------------|
| `log_temp_peak` | float or null | log10 of peak blackbody temperature (K) |
| `cooling_rate` | float or null | Temperature cooling rate |
| `log_temp_peak_err` | float or null | Uncertainty on log_temp_peak |
| `cooling_rate_err` | float or null | Uncertainty on cooling_rate |
| `chi2` | float or null | Thermal model chi-squared |
| `n_color_obs` | int | Number of color (cross-band) observations used |
| `n_bands_used` | int | Number of photometric bands used |
| `ref_band` | str | Reference band name |

### `fit_fast(bands, max_subsample=25) -> dict`

Combined nonparametric + thermal fitting that reuses GP fits internally.

Equivalent to calling `fit_nonparametric` followed by `fit_thermal`, but avoids
refitting the reference-band GP for the thermal step.

**Args:**
- `bands` (`BandDataMap`): Magnitude-space band data.
- `max_subsample` (int, default 25): See `fit_nonparametric`.

**Returns:** `dict` with two keys:
- `"nonparametric"` (list[dict]): Same format as `fit_nonparametric` output.
- `"thermal"` (dict or None): Same format as `fit_thermal` output.

---

## Model Evaluation

### `eval_model(model, params, times) -> list[float]`

Evaluate a parametric model at the given times.

**Args:**
- `model` (str): Model name string. One of: `"Bazin"`, `"Villar"`,
  `"MetzgerKN"`, `"Tde"`, `"Arnett"`, `"Magnetar"`, `"ShockCooling"`,
  `"Afterglow"`.
- `params` (list[float]): Internal (transformed) parameter vector. Pass
  `result["pso_params"]` or `result["svi_mu"]` from a `fit_parametric` result.
- `times` (list[float]): Time values (relative days) at which to evaluate.

**Returns:** `list[float]` -- predicted flux values, one per time point.

---

## Gaussian Process Prediction

### `fit_gp_predict(train_times, train_values, train_errors, query_times, amp_candidates, ls_candidates) -> tuple[list[float], list[float]] or None`

Fit a Gaussian Process to training data and predict at query points.

Performs a grid search over amplitude and lengthscale candidates, using the
mean squared measurement error as the GP alpha parameter.

**Args:**
- `train_times` (list[float]): Training time values.
- `train_values` (list[float]): Training observed values.
- `train_errors` (list[float]): Training measurement errors.
- `query_times` (list[float]): Times at which to predict.
- `amp_candidates` (list[float]): Amplitude values to try in grid search.
- `ls_candidates` (list[float]): Lengthscale values to try in grid search.

**Returns:** `tuple[list[float], list[float]]` of `(predictions, std_devs)`, or
`None` if fitting fails.

---

## Kilonova Model

### `metzger_kn_mags(params, times, band_frequencies, d_l_cm) -> dict`

Evaluate the Metzger kilonova model in physical magnitude space.

Unlike `eval_model("MetzgerKN", ...)` which returns normalized flux, this
evaluates the full physical model and returns AB magnitudes per band.

**Args:**
- `params` (list[float]): Physical parameters
  `[log10(M_ej/Msun), log10(v_ej/c), log10(kappa), t0_offset]`.
- `times` (list[float]): Observation times (days).
- `band_frequencies` (list[tuple[str, float]]): List of `(band_name, frequency_hz)`
  tuples, e.g. `[("g", 6.3e14), ("r", 4.7e14), ("i", 3.9e14)]`.
- `d_l_cm` (float): Luminosity distance in cm.

**Returns:** `dict[str, list[float]]` -- `{band_name: [mag_values]}`.

```python
import lightcurve_fitting as lcf

params = [-2.0, -0.7, 0.5, 0.0]   # log Mej, log vej, log kappa, t0
times  = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
band_freqs = [("g", 6.3e14), ("r", 4.7e14), ("i", 3.9e14)]
d_l = 1.26e27   # 40 Mpc in cm

mags = lcf.metzger_kn_mags(params, times, band_freqs, d_l)
# mags = {"g": [...], "r": [...], "i": [...]}
```

---

## Batch Fitters

All batch functions process sources in parallel using Rayon.

### `fit_batch_fast(sources) -> list[dict]`

Batch nonparametric + thermal fitting for multiple sources.

**Args:**
- `sources` (list[BandDataMap]): List of magnitude-space band data objects.

**Returns:** `list[dict]` -- one dict per source, each containing
`"nonparametric"` and `"thermal"` keys (same format as `fit_fast`).

### `fit_batch_nonparametric(sources) -> list[list[dict]]`

Batch nonparametric GP fitting for multiple sources.

Like `fit_batch_fast` but without the thermal step.

**Args:**
- `sources` (list[BandDataMap]): List of magnitude-space band data objects.

**Returns:** `list[list[dict]]` -- one inner list per source (same format as
`fit_nonparametric`).

### `fit_batch_parametric(sources, fit_all_models=False, method="svi") -> list[list[dict]]`

Batch parametric fitting for multiple sources.

**Args:**
- `sources` (list[BandDataMap]): List of flux-space band data objects.
- `fit_all_models` (bool, default False): If True, return all model fits per band.
- `method` (str, default `"svi"`): Uncertainty method (`"svi"` or `"laplace"`).

**Returns:** `list[list[dict]]` -- one inner list per source (same format as
`fit_parametric`).

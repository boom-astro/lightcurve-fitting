# Kilonova Blackbody Model

## Overview

`fit_kilonova` fits a fast, closed-form kilonova model jointly to all bands of a
magnitude light curve and scores how kilonova-like it is. The photosphere is a
single blackbody whose luminosity and temperature evolve as low-order
polynomials in log time, integrated through each filter's transmission curve,
with fixed per-band offsets and a parameter prior learned from kilonova model
grids. The built-in calibration comes from the POSSIS Bu2019 grid of binary
neutron star and neutron star–black hole models; other calibrations can be
trained with `scripts/train_kilonova_prior.py` and passed in at runtime.

## Model

With $x = \log_{10}(t - t_\mathrm{exp})$ and $t$ in days,

$$\log_{10} L = \ell_1 + a_1 x + a_2 x^2, \qquad \log_{10} T = \tau_1 - \beta x,$$

so $\ell_1$ (`log_l1`) and $\tau_1$ (`log_t1`) are the luminosity in erg/s and
the temperature in K one day after the explosion, and the bolometric light
curve peaks at $t_\mathrm{peak} = 10^{-a_1 / 2a_2}$. The AB magnitude in band
$b$ is the blackbody $F_\nu = L\,B_\nu(T) / (4\sigma T^4 D^2)$ integrated
through the band's transmission (photon-weighted), plus a fixed offset,

$$m_b(t) = M_\mathrm{AB}\big[L(t), T(t); b\big] + c_{0,b} + c_{1,b}\,x .$$

The bandpass integrals are tabulated on a grid of $\log_{10} T$ from 2.9 to 4.8
in steps of 0.01 (`src/kilonova_tables.rs`). Integration matters for cool
photospheres: at 2,000 K, ZTF g is 0.46 mag fainter than a blackbody evaluated
at the band's effective wavelength.

The offsets absorb the departure of kilonova spectra from a blackbody. They are
the median residuals of the model grid from a blackbody fit across grizyJHK at
each epoch, fit as linear functions of $x$ over 0.5 to 10 days. The built-in
(POSSIS) values:

| Band | $c_0$ | $c_1$ | Scatter across the grid |
|---|---|---|---|
| g | +0.056 | −0.170 | 0.17 |
| r | +0.072 | +0.132 | 0.15 |
| i | −0.069 | +0.082 | 0.14 |
| z | −0.124 | +0.033 | 0.13 |
| y | −0.131 | +0.131 | 0.15 |
| J | +0.010 | +0.002 | 0.09 |
| H | +0.069 | −0.020 | 0.11 |
| K | +0.119 | −0.192 | 0.16 |

Supported band names are ZTF (`ztfg`, `ztfr`, `ztfi`), LSST (`lsstg` to
`lssty`), Pan-STARRS (`ps1::g` to `ps1::y`), SDSS (`sdssg` to `sdssz`) and
2MASS (`2massj`, `2massh`, `2massks`). Bare names (`g`, `r`, `i`, `z`, `y`,
`J`, `H`, `K`) mean the filters the model grids were synthesised in: SDSS griz,
Swope Y, J and H, and CSP K, following the filter set in Bulla's POSSIS
light-curve script. u band is skipped, because kilonova spectra depart from a
blackbody there by more than a magnitude, differently in every model; other
unsupported bands are skipped and listed in `bands_skipped`.

## Prior

Fitting the model to 1,523 POSSIS models in g, r and i from 0.3 to 6 days, with
the merger time known, gives the built-in Gaussian prior:

| Parameter | Mean | Standard deviation |
|---|---|---|
| `log_l1` | 41.64 | 0.31 |
| `a1` | 0.26 | 0.54 |
| `a2` | −1.88 | 0.62 |
| `log_t1` | 3.77 (5,900 K) | 0.13 |
| `beta` | 0.389 | 0.140 |

`log_t1` and `beta` are strongly correlated (+0.85): models that are hotter at
one day cool faster. By default the standard deviations are inflated by 1.5
(`prior_inflate`). Without a distance, `log_l1` absorbs the distance modulus
and the prior applies only to the four shape parameters; with
`distance_modulus` set, the magnitudes become absolute and `log_l1` joins the
prior.

`KilonovaPrior` holds a Gaussian mixture (weights, means, covariances) and,
optionally, the per-band offsets it was trained with. With more than one
component, the light curve is fit under each component and the lowest
$\chi^2 + D_M^2 - 2\ln w_k + \ln\det\Sigma_k$ is kept, relative to the
smallest $-2\ln w_k + \ln\det\Sigma_k$.

## Training other calibrations

`scripts/train_kilonova_prior.py` learns the offsets and mixture priors from
any set of light-curve grids in the POSSIS text format, weighting families
equally, and `scripts/generate_kilonova_tables.py` turns its output into
`src/kilonova_tables.rs`. The built-in tables were made with

```
python scripts/train_kilonova_prior.py --possis POSSIS/BNS POSSIS/NSBH --out possis.json
python scripts/generate_kilonova_tables.py possis.json --components 1
```

`scripts/kilonova_priors/` holds two calibrations as JSON, usable as runtime
priors: `possis.json` (the built-in one) and `possis_kasen.json`, trained on
POSSIS, the Kasen et al. (2017) grid and 600 blue+red combinations of Kasen
models summed in flux.

On simulated ZTF windows (see Validation) the broader prior recognises
Kasen-like kilonovae better and POSSIS-like kilonovae worse, and AT2017gfo
ranks best against contaminants under the POSSIS prior:

| Prior (single Gaussian) | AUC, POSSIS KNe, 3 d / 5 d | AUC, Kasen KNe | AUC, blue+red Kasen KNe | Contaminant windows scoring below AT2017gfo |
|---|---|---|---|---|
| POSSIS (built-in) | 0.855 / 0.915 | 0.63 / 0.74 | 0.66 / 0.75 | 35% |
| POSSIS + blue+red Kasen | 0.77 / 0.85 | 0.70 / 0.80 | 0.72 / 0.81 | 46% |
| POSSIS + Kasen + blue+red Kasen | 0.76 / 0.84 | 0.74 / 0.81 | 0.78 / 0.85 | 46% |

Gaussian mixtures with two or three components fit the clustered grid
parameters better by BIC but separate kilonovae from contaminants worse, and
rank AT2017gfo below 62 to 91% of contaminant windows.

## Fitting

In magnitude space `log_l1` adds the same constant to every band, and every
residual (data and whitened prior) is affine in it, so it is solved exactly at
each trial point. The four shape parameters are fit by bounded
Levenberg–Marquardt. A free explosion time is searched on a 10-point grid
between `t_exp_window` (default 3) and 0.05 days before the first point, and no
earlier than `t_last_nondet` when a last non-detection is given; the best grid
point is refined jointly with the shape parameters. `max_days` restricts the
fit to points within that many days of the first one. A systematic error
(`sys_err`, default 0.1 mag) is added in quadrature to every point.

Each light curve is fit with no prior (`free`) and held to the prior
(`kn_prior`). The score is

$$\mathrm{prior\_cost} = \big(\chi^2 + D_M^2\big)_\mathrm{kn\_prior} - \chi^2_\mathrm{free},$$

where $D_M^2$ is the Mahalanobis distance of the parameters from the prior. It
approximates a log Bayes factor for a kilonova against an arbitrary blackbody:
kilonovae fit both ways equally well, while sources that stay hot, do not cool,
or have non-thermal spectra pay a large cost under the prior.

## Features

`extract_features` (and the `extract-features` binary) fit the first 10 days of
each light curve (`kilonova_feature_options`) and add `kn_prior_cost`,
`kn_chi2_per_obs`, `kn_free_chi2_per_obs`, `kn_a1`, `kn_a2`, `kn_log_t1`,
`kn_beta`, `kn_t_exp`, `kn_free_log_t1`, `kn_free_beta` and `kn_n_obs`. The fit
costs about 2 ms per source, 0.1% of `extract_features`, so it runs on the CPU
in every path.

## Validation

- **Grid fidelity.** On held-out POSSIS models the model reproduces g, r and i
  from 0.5 to 5 days with a median RMS of 0.16 mag, of which 0.07 mag is the
  grid's own Monte Carlo noise.
- **AT2017gfo.** With the merger time and distance known, the free fit
  describes grizyJHK from 0.4 to 10 days with an RMS of 0.32 mag. On a ZTF-like
  subset (g and r, one point per night over the first 3.5 days, merger time
  free) the prior cost is 4.9. Both cases are tests in
  `tests/test_kilonova.rs`, checked against an independent SciPy fit.
- **Separation from contaminants.** On simulated ZTF windows (3,000 POSSIS
  kilonovae against 2,711 three-day and 3,190 five-day windows of real ZTF
  supernovae, cataclysmic variables, AGN and TDEs plus synthetic afterglows,
  all with the same cadence) the prior cost alone gives AUCs of 0.855 and 0.915
  and recovers 14% and 21% of kilonovae at a 1% false-positive rate. Combined
  with fade and reddening rates in a classifier, recovery at 1% rises to 44 to
  51%.
- **Speed.** About 2 ms per short light curve on one core, and 0.5 ms per light
  curve with `fit_kilonova_batch` on 10 cores (Apple M1 Max).

## Limitations

- The built-in offsets and prior come from one model family; see the table
  above for the effect of broader training sets.
- The grid band assignment (SDSS griz, Swope YJH, CSP K) follows Bulla's
  light-curve script; the Kasen grid is assumed to use the same filters.
- A single photosphere misses the near-infrared excess of some models. A second
  blackbody at constant temperature with the same luminosity history improved
  grizyJHK fits to the grids by 0.03 to 0.06 mag and did not improve AT2017gfo,
  so it is not included.
- The likelihood uses magnitudes and detections only.

## Usage

### Rust

```rust
use lightcurve_fitting::{build_mag_bands, fit_kilonova, KilonovaOptions};

let bands = build_mag_bands(&times, &mags, &mag_errs, &band_names);
let result = fit_kilonova(&bands, &KilonovaOptions::default()).unwrap();
println!("prior cost {:.2}, T(1 d) = {:.0} K", result.prior_cost, result.temperature_1d);

// GW-triggered: merger time known (in the bands' time frame) and distance known.
let opts = KilonovaOptions { t_exp: Some(-0.4), distance_modulus: Some(36.2), ..Default::default() };
let result = fit_kilonova(&bands, &opts);
```

### Python

```python
import json
import lightcurve_fitting as lcf

bands = lcf.build_mag_bands(times, mags, mag_errs, band_names)
res = lcf.fit_kilonova(bands)                         # merger time free, no distance
res["prior_cost"], res["kn_prior"]["params"], res["free"]["t_exp"]

res = lcf.fit_kilonova(bands, t_exp=-0.4, distance_modulus=36.2)
res = lcf.fit_kilonova(bands, t_last_nondet=-1.1, max_days=10.0)
results = lcf.fit_kilonova_batch([bands, ...])

# a broader prior, with the offsets it was trained with
prior = json.load(open("scripts/kilonova_priors/possis_kasen.json"))
res = lcf.fit_kilonova(bands, prior=prior)

# model magnitudes (absolute AB) at days since explosion
lcf.kilonova_mags([41.64, 0.26, -1.88, 3.77, 0.389], [0.5, 1.0, 2.0], "ztfr")
```

`build_mag_bands` sets t = 0 at the earliest observation, so a known merger
time or last non-detection is passed relative to that point.

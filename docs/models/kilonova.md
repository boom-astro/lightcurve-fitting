# Kilonova Blackbody Model

## Overview

`fit_kilonova` fits a fast, closed-form kilonova model jointly to all bands of a
magnitude light curve and scores how kilonova-like it is. The photosphere is a
single blackbody whose luminosity and temperature evolve as low-order
polynomials in log time, with fixed per-band offsets and a parameter prior
learned from the POSSIS Bu2019 grid of binary neutron star and neutron
star–black hole models.

## Model

With $x = \log_{10}(t - t_\mathrm{exp})$ and $t$ in days,

$$\log_{10} L = \ell_1 + a_1 x + a_2 x^2, \qquad \log_{10} T = \tau_1 - \beta x,$$

so $\ell_1$ (`log_l1`) and $\tau_1$ (`log_t1`) are the luminosity in erg/s and
the temperature in K one day after the explosion, and the bolometric light
curve peaks at $t_\mathrm{peak} = 10^{-a_1 / 2a_2}$. The AB magnitude in band
$b$ is the blackbody $F_\nu = L\,B_\nu(T) / (4\sigma T^4 D^2)$ evaluated at the
band's effective wavelength plus a fixed offset,

$$m_b(t) = M_\mathrm{AB}\big[L(t), T(t), \lambda_b\big] + c_{0,b} + c_{1,b}\,x .$$

The offsets absorb the departure of kilonova spectra from a blackbody. They are
the median residuals of POSSIS spectra from a blackbody fit across grizyJHK at
each epoch, fit as linear functions of $x$ over 0.5 to 10 days:

| Band | $c_0$ | $c_1$ | Scatter across the grid |
|---|---|---|---|
| g | +0.055 | −0.208 | 0.17 |
| r | +0.056 | +0.213 | 0.15 |
| i | −0.078 | +0.152 | 0.14 |
| z | −0.131 | +0.108 | 0.14 |
| y | −0.078 | −0.135 | 0.14 |
| J | +0.024 | +0.000 | 0.09 |
| H | +0.035 | +0.037 | 0.11 |
| K | +0.119 | −0.197 | 0.17 |

Supported band names are ZTF (`ztfg`, `ztfr`, `ztfi`), LSST (`lsstg` to
`lssty`), Pan-STARRS (`ps1::g` to `ps1::y`), SDSS or generic (`g`, `r`, `i`,
`z`, `y`) and 2MASS (`J`, `H`, `K`, `2massj`, ...). u band is skipped, because
its offset scatters by 1.1 mag across the grid; other unsupported bands are
skipped and listed in `bands_skipped`.

## Prior

Fitting the model to 1,420 POSSIS models in g, r and i from 0.3 to 6 days, with
the merger time known, gives a multivariate-normal prior:

| Parameter | Mean | Standard deviation |
|---|---|---|
| `log_l1` | 41.63 | 0.31 |
| `a1` | 0.28 | 0.53 |
| `a2` | −1.88 | 0.62 |
| `log_t1` | 3.77 (5,900 K) | 0.13 |
| `beta` | 0.385 | 0.136 |

`log_t1` and `beta` are strongly correlated (+0.87): models that are hotter at
one day cool faster. The full covariance is `KILONOVA_PRIOR_COV`. By default
the standard deviations are inflated by 1.5 (`prior_inflate`). Without a
distance, `log_l1` absorbs the distance modulus and the prior applies only to
the four shape parameters; with `distance_modulus` set, the magnitudes become
absolute and `log_l1` joins the prior.

## Fitting

In magnitude space `log_l1` adds the same constant to every band, and every
residual (data and whitened prior) is affine in it, so it is solved exactly at
each trial point. The four shape parameters are fit by bounded
Levenberg–Marquardt from two starts. A free explosion time is searched on a
10-point grid between `t_exp_window` (default 3) and 0.05 days before the first
point, and the best grid point is refined jointly with the shape parameters.
A systematic error (`sys_err`, default 0.1 mag) is added in quadrature to every
point.

Each light curve is fit twice: once with no prior (`free`) and once held to the
prior (`kn_prior`). The score is

$$\mathrm{prior\_cost} = \big(\chi^2 + D_M^2\big)_\mathrm{kn\_prior} - \chi^2_\mathrm{free},$$

where $D_M^2$ is the Mahalanobis distance of the parameters from the prior. It
approximates a log Bayes factor for a kilonova against an arbitrary blackbody:
kilonovae fit both ways equally well, while sources that stay hot, do not cool,
or have non-thermal spectra pay a large cost under the prior.

## Validation

- **Grid fidelity.** On 300 held-out POSSIS models the model reproduces g, r
  and i from 0.5 to 5 days with a median RMS of 0.16 mag, of which 0.07 mag is
  the grid's own Monte Carlo noise.
- **AT2017gfo.** With the merger time and distance known, the free fit
  describes grizyJHK from 0.4 to 10 days with an RMS of 0.33 mag; K is underfit
  by about 0.3 mag at 2 to 4 days. On a ZTF-like subset (g and r, one point per
  night over the first 3.5 days, merger time free) the prior cost is 5.0. Both
  cases are tests in `tests/test_kilonova.rs`.
- **Separation from contaminants.** On simulated 3-day ZTF windows (3,000
  POSSIS kilonovae against 2,711 windows of real ZTF supernovae, cataclysmic
  variables, AGN and TDEs plus synthetic afterglows, all with the same cadence)
  the prior cost alone gives an AUC of 0.86 and recovers 14% of kilonovae at a
  1% false-positive rate; on 5-day windows, 0.92 and 21%. Combined with fade
  and reddening rates in a classifier, recovery at 1% rises to 44 to 51%.
- **Speed.** About 2 ms per light curve for both fits on one core, and 0.5 ms
  per light curve with `fit_kilonova_batch` on 10 cores (Apple M1 Max).

## Limitations

- The offsets and prior come from one model family (Bu2019lm); AT2017gfo sits
  at the 97th percentile of the prior in g, r and i.
- Blackbody magnitudes are evaluated at effective wavelengths, without
  bandpass integration.
- A single photosphere misses the near-infrared excess of some models.
- The likelihood uses magnitudes and detections only; upper limits are not
  used.

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
import lightcurve_fitting as lcf

bands = lcf.build_mag_bands(times, mags, mag_errs, band_names)
res = lcf.fit_kilonova(bands)                  # merger time free, no distance
res["prior_cost"], res["kn_prior"]["params"], res["free"]["t_exp"]

res = lcf.fit_kilonova(bands, t_exp=-0.4, distance_modulus=36.2)
results = lcf.fit_kilonova_batch([bands, ...])

# model magnitudes (absolute AB) at days since explosion
lcf.kilonova_mags([41.63, 0.28, -1.88, 3.77, 0.385], [0.5, 1.0, 2.0], "ztfr")
```

`build_mag_bands` sets t = 0 at the earliest observation, so a known merger
time is passed relative to that point.

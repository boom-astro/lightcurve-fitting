# lightcurve-fitting

Lightcurve fitting library for astronomical transients, written in Rust.

Provides three complementary fitters that operate on multi-band photometry (ZTF g/r/i):

| Fitter | Input | Method | Key outputs |
|--------|-------|--------|-------------|
| **Nonparametric** | magnitudes | Gaussian-process interpolation | peak mag, t0, rise/decay timescales, FWHM, GP derivatives, von Neumann ratio, power-law decay index |
| **Parametric** | fluxes | PSO model selection → SVI or Laplace uncertainty | best-fit model (Bazin, Villar, TDE, Arnett, Magnetar, …), posterior means & uncertainties |
| **Thermal** | magnitudes | Blackbody color fitting via PSO | log temperature at peak, cooling rate |

## Quick start

```rust
use lightcurve_fitting::{
    build_mag_bands, build_flux_bands,
    fit_nonparametric, fit_parametric, fit_thermal,
    UncertaintyMethod,
};

// times, mags, mag_errs, bands: parallel Vecs from your photometry table
let mag_bands  = build_mag_bands(&times, &mags, &mag_errs, &bands);
let flux_bands = build_flux_bands(&times, &mags, &mag_errs, &bands);

let (np_results, trained_gps) = fit_nonparametric(&mag_bands);
let p_results  = fit_parametric(&flux_bands, false, UncertaintyMethod::Laplace);
let t_result   = fit_thermal(&mag_bands, Some(&trained_gps));
```

## Building

```sh
cargo build --release
```

## Testing

Run the unit and integration tests:

```sh
cargo test
```

Run the throughput benchmark (10 synthetic sources × 3 fitters):

```sh
cargo test --release throughput -- --ignored
```

This writes `wall_time.txt` with the elapsed seconds.

## CI

Two GitHub Actions workflows are included:

- **test.yml** — runs `cargo test` on every push to `main` and on PRs.
- **throughput.yml** — builds in release mode, runs the throughput benchmark, uploads `wall_time.txt` as an artifact, and on PRs compares against the `main` baseline. Fails if wall time regresses by more than 10%.

## License

GPL-3.0

# lightcurve-fitting

Lightcurve fitting library for astronomical transients, written in Rust.

Provides three complementary fitters that operate on multi-band photometry (ZTF g/r/i), plus a single-band stellar flare fitter:

| Fitter | Input | Method | Key outputs |
|--------|-------|--------|-------------|
| **Nonparametric** | magnitudes | Gaussian-process interpolation | peak mag, t0, rise/decay timescales, FWHM, GP derivatives, von Neumann ratio, power-law decay index |
| **Parametric** | fluxes | PSO model selection → SVI or Laplace uncertainty | best-fit model (Bazin, Villar, TDE, Arnett, Magnetar, …), posterior means & uncertainties |
| **Thermal** | magnitudes | Blackbody color fitting via PSO | log temperature at peak, cooling rate |
| **Kilonova** | magnitudes | Joint multi-band, bandpass-integrated blackbody with a grid-trained prior, Levenberg–Marquardt | luminosity and temperature evolution, merger time, kilonova-prior cost |
| **Flares** | relative flux | Sum of Davenport (2014) templates, variable projection, BIC for the number of components | peak time, FWHM, amplitude, equivalent duration (with uncertainties) per component |

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

## GPU acceleration (optional)

When built with `--features cuda`, the parametric fitter uses CUDA batch PSO
to fit all sources simultaneously on a GPU. This provides significant speedup
at scale (see [Benchmarks](#benchmarks)).

```sh
cargo build --release --features cuda
```

Requires CUDA toolkit (tested with 12.x). The build system auto-detects
`nvcc` via the `CUDA_PATH` environment variable or standard locations.

The stellar flare fitter also has a GPU batch path, `fit_flares_batch_gpu`,
for both `--features cuda` and `--features metal`
(see [docs/models/flares.md](docs/models/flares.md)).

## Building

```sh
cargo build --release
```

## Testing

Run the unit and integration tests:

```sh
cargo test
```

Run GPU-specific tests (requires CUDA):

```sh
cargo test --features cuda --test test_gpu -- --nocapture
```

## Benchmarks

Throughput benchmarks measure scaling across two axes:

- **Source-count scaling**: fixed 30 pts/band, 10–1,000 sources
- **Point-count scaling**: fixed 100 sources, 10–500 pts/band

Run the full benchmark suite:

```sh
# With GPU
cargo test --release --features cuda --test bench_throughput -- --ignored --nocapture

# CPU only
cargo test --release --test bench_throughput -- --ignored --nocapture

# Generate plots
python3 benchmarks/plot_throughput.py

# Submit as SLURM job
sbatch benchmarks/run_bench.sh
```

Results are written to `benchmarks/throughput_results.csv`.
See [docs/benchmarks.md](docs/benchmarks.md) for full methodology and results.

## CI

Two GitHub Actions workflows are included:

- **test.yml** — runs `cargo test` on every push to `main` and on PRs.
- **throughput.yml** — builds in release mode, runs the throughput benchmark, uploads `wall_time.txt` as an artifact, and on PRs compares against the `main` baseline. Fails if wall time regresses by more than 10%.

## License

MIT

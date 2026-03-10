//! Throughput benchmarks for lightcurve-fitting.
//!
//! Separate tests for nonparametric and parametric fitters, each sweeping
//! over point counts [10, 100, 1000, 10000] for documentation tables.
//!
//! Run individual benchmarks:
//!   cargo test --release --test bench_throughput nonparametric_point_scaling -- --ignored --nocapture
//!   cargo test --release --test bench_throughput parametric_point_scaling -- --ignored --nocapture
//!   cargo test --release --test bench_throughput nonparametric_source_scaling -- --ignored --nocapture
//!   cargo test --release --test bench_throughput parametric_source_scaling -- --ignored --nocapture
//!
//! Run all:
//!   cargo test --release --test bench_throughput -- --ignored --nocapture
//!
//! With GPU:
//!   cargo test --release --features cuda --test bench_throughput -- --ignored --nocapture

mod synthetic;

use lightcurve_fitting::{build_flux_bands, build_mag_bands, fit_nonparametric, fit_parametric, fit_batch_parametric, UncertaintyMethod};
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "cuda")]
use lightcurve_fitting::gpu::{BatchSource, GpuBatchData, GpuContext};
#[cfg(feature = "cuda")]
use lightcurve_fitting::fit_nonparametric_batch_gpu;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const N_WARMUP: usize = 1;
const N_REPEAT: usize = 3;

// Point-count scaling
const POINT_COUNTS: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096, 8192];
const FIXED_SOURCES: usize = 100;

// Source-count scaling
const SOURCE_COUNTS: &[usize] = &[10, 100, 500, 1000];
const FIXED_POINTS: usize = 30;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct CsvWriter {
    rows: Vec<String>,
}

impl CsvWriter {
    fn new() -> Self {
        Self { rows: Vec::new() }
    }

    fn add_row(
        &mut self,
        fitter: &str,
        backend: &str,
        n_sources: usize,
        n_points_per_band: usize,
        wall_sec: f64,
    ) {
        let total_obs = n_sources * n_points_per_band * 3;
        let throughput_src = n_sources as f64 / wall_sec;
        let throughput_pts = total_obs as f64 / wall_sec;
        self.rows.push(format!(
            "{},{},{},{},{},{:.4},{:.0},{:.0}",
            fitter, backend, n_sources, n_points_per_band,
            total_obs, wall_sec, throughput_src, throughput_pts,
        ));
    }

    fn write_csv(&self, path: &str) {
        let mut f = std::fs::File::create(path).expect("failed to create CSV");
        writeln!(f, "fitter,backend,n_sources,n_points_per_band,total_obs,wall_sec,throughput_src_per_sec,throughput_pts_per_sec").unwrap();
        for row in &self.rows {
            writeln!(f, "{}", row).unwrap();
        }
        eprintln!("\nResults written to {path}");
    }
}

fn print_header(title: &str) {
    eprintln!("\n{:=<80}", "");
    eprintln!("{}", title);
    eprintln!("{:=<80}", "");
    eprintln!("{:<12} {:>8} {:>10} {:>12} {:>14} {:>14}",
        "backend", "n_src", "pts/band", "wall_sec", "src/sec", "obs/sec");
    eprintln!("{:-<80}", "");
}

fn print_row(backend: &str, n_sources: usize, n_pts: usize, wall: f64) {
    let total_obs = n_sources * n_pts * 3;
    eprintln!("{:<12} {:>8} {:>10} {:>12.4} {:>14.0} {:>14.0}",
        backend, n_sources, n_pts, wall,
        n_sources as f64 / wall, total_obs as f64 / wall);
}

fn bench_best_of<F: Fn()>(warmup: usize, repeat: usize, f: F) -> f64 {
    for _ in 0..warmup { f(); }
    let mut best = f64::MAX;
    for _ in 0..repeat {
        let t0 = Instant::now();
        f();
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed < best { best = elapsed; }
    }
    best
}

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn flux_bands_to_batch_source(
    times: &[f64], mags: &[f64], errs: &[f64], bands: &[String],
) -> Option<BatchSource> {
    let flux_bands = build_flux_bands(times, mags, errs, bands);
    let (_, band_data) = flux_bands.iter().max_by_key(|(_, b)| b.values.len())?;

    let fluxes = &band_data.values;
    let flux_errs = &band_data.errors;
    let times = &band_data.times;

    let peak_flux = fluxes.iter().cloned().fold(f64::MIN, f64::max);
    if peak_flux <= 0.0 { return None; }

    let snr_threshold = 3.0;
    Some(BatchSource {
        times: times.clone(),
        flux: fluxes.iter().map(|f| f / peak_flux).collect(),
        obs_var: flux_errs.iter().map(|e| { let n = e / peak_flux; n * n + 1e-10 }).collect(),
        is_upper: fluxes.iter().zip(flux_errs.iter())
            .map(|(f, e)| *e > 0.0 && (*f / *e) < snr_threshold).collect(),
        upper_flux: flux_errs.iter().map(|e| snr_threshold * e / peak_flux).collect(),
    })
}

// ===========================================================================
// Nonparametric benchmarks
// ===========================================================================

#[test]
#[ignore]
fn nonparametric_point_scaling() {
    print_header(&format!(
        "Nonparametric GP — point-count scaling (n_sources={})", FIXED_SOURCES));

    let mut csv = CsvWriter::new();

    for &n_pts in POINT_COUNTS {
        let sources = synthetic::generate_n_sources(FIXED_SOURCES, n_pts);

        let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
            for (t, m, e, b) in &sources {
                let bands = build_mag_bands(t, m, e, b);
                let _ = fit_nonparametric(&bands);
            }
        });
        print_row("CPU", FIXED_SOURCES, n_pts, wall);
        csv.add_row("nonparametric", "CPU", FIXED_SOURCES, n_pts, wall);

        #[cfg(feature = "cuda")]
        {
            let ctx = GpuContext::new(0).expect("CUDA init");
            let mag_band_sets: Vec<_> = sources.iter()
                .map(|(t, m, e, b)| build_mag_bands(t, m, e, b))
                .collect();
            let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
                let _ = fit_nonparametric_batch_gpu(&ctx, &mag_band_sets);
            });
            print_row("GPU", FIXED_SOURCES, n_pts, wall);
            csv.add_row("nonparametric", "GPU", FIXED_SOURCES, n_pts, wall);
        }
    }

    csv.write_csv("benchmarks/nonparametric_points.csv");
}

#[test]
#[ignore]
fn nonparametric_source_scaling() {
    print_header(&format!(
        "Nonparametric GP — source-count scaling (pts/band={})", FIXED_POINTS));

    let mut csv = CsvWriter::new();

    for &n_src in SOURCE_COUNTS {
        let sources = synthetic::generate_n_sources(n_src, FIXED_POINTS);

        let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
            for (t, m, e, b) in &sources {
                let bands = build_mag_bands(t, m, e, b);
                let _ = fit_nonparametric(&bands);
            }
        });
        print_row("CPU", n_src, FIXED_POINTS, wall);
        csv.add_row("nonparametric", "CPU", n_src, FIXED_POINTS, wall);

        #[cfg(feature = "cuda")]
        {
            let ctx = GpuContext::new(0).expect("CUDA init");
            let mag_band_sets: Vec<_> = sources.iter()
                .map(|(t, m, e, b)| build_mag_bands(t, m, e, b))
                .collect();
            let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
                let _ = fit_nonparametric_batch_gpu(&ctx, &mag_band_sets);
            });
            print_row("GPU", n_src, FIXED_POINTS, wall);
            csv.add_row("nonparametric", "GPU", n_src, FIXED_POINTS, wall);
        }
    }

    csv.write_csv("benchmarks/nonparametric_sources.csv");
}

// ===========================================================================
// Parametric benchmarks
// ===========================================================================

#[test]
#[ignore]
fn parametric_point_scaling() {
    print_header(&format!(
        "Parametric PSO+Laplace — point-count scaling (n_sources={})", FIXED_SOURCES));

    let mut csv = CsvWriter::new();

    for &n_pts in POINT_COUNTS {
        let sources = synthetic::generate_n_sources(FIXED_SOURCES, n_pts);

        let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
            for (t, m, e, b) in &sources {
                let bands = build_flux_bands(t, m, e, b);
                let _ = fit_parametric(&bands, false, UncertaintyMethod::Laplace);
            }
        });
        print_row("CPU", FIXED_SOURCES, n_pts, wall);
        csv.add_row("parametric", "CPU", FIXED_SOURCES, n_pts, wall);

        // CPU-parallel: rayon par_iter over sources
        let all_bands: Vec<_> = sources.iter()
            .map(|(t, m, e, b)| build_flux_bands(t, m, e, b))
            .collect();
        let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
            let _ = fit_batch_parametric(&all_bands, false, UncertaintyMethod::Laplace);
        });
        print_row("CPU-par", FIXED_SOURCES, n_pts, wall);
        csv.add_row("parametric", "CPU-par", FIXED_SOURCES, n_pts, wall);

        #[cfg(feature = "cuda")]
        {
            let ctx = GpuContext::new(0).expect("CUDA init");
            let batch_sources: Vec<BatchSource> = sources.iter()
                .filter_map(|(t, m, e, b)| flux_bands_to_batch_source(t, m, e, b))
                .collect();
            if !batch_sources.is_empty() {
                let data = GpuBatchData::new(&batch_sources).expect("GPU upload");
                let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
                    let _ = ctx.batch_model_select(&data, 30, 50, 10, 2.0).unwrap();
                });
                print_row("GPU", FIXED_SOURCES, n_pts, wall);
                csv.add_row("parametric", "GPU", FIXED_SOURCES, n_pts, wall);

                // MultiBazin GPU (greedy K=1..4)
                let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
                    let _ = ctx.batch_pso_multi_bazin(&data, &batch_sources, 30, 60, 12, 42).unwrap();
                });
                print_row("GPU-MB", FIXED_SOURCES, n_pts, wall);
                csv.add_row("multi_bazin", "GPU", FIXED_SOURCES, n_pts, wall);
            }
        }
    }

    csv.write_csv("benchmarks/parametric_points.csv");
}

#[test]
#[ignore]
fn parametric_source_scaling() {
    print_header(&format!(
        "Parametric PSO+Laplace — source-count scaling (pts/band={})", FIXED_POINTS));

    let mut csv = CsvWriter::new();

    for &n_src in SOURCE_COUNTS {
        let sources = synthetic::generate_n_sources(n_src, FIXED_POINTS);

        let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
            for (t, m, e, b) in &sources {
                let bands = build_flux_bands(t, m, e, b);
                let _ = fit_parametric(&bands, false, UncertaintyMethod::Laplace);
            }
        });
        print_row("CPU", n_src, FIXED_POINTS, wall);
        csv.add_row("parametric", "CPU", n_src, FIXED_POINTS, wall);

        // CPU-parallel: rayon par_iter over sources
        let all_bands: Vec<_> = sources.iter()
            .map(|(t, m, e, b)| build_flux_bands(t, m, e, b))
            .collect();
        let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
            let _ = fit_batch_parametric(&all_bands, false, UncertaintyMethod::Laplace);
        });
        print_row("CPU-par", n_src, FIXED_POINTS, wall);
        csv.add_row("parametric", "CPU-par", n_src, FIXED_POINTS, wall);

        #[cfg(feature = "cuda")]
        {
            let ctx = GpuContext::new(0).expect("CUDA init");
            let batch_sources: Vec<BatchSource> = sources.iter()
                .filter_map(|(t, m, e, b)| flux_bands_to_batch_source(t, m, e, b))
                .collect();
            if !batch_sources.is_empty() {
                let data = GpuBatchData::new(&batch_sources).expect("GPU upload");
                let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
                    let _ = ctx.batch_model_select(&data, 30, 50, 10, 2.0).unwrap();
                });
                print_row("GPU", n_src, FIXED_POINTS, wall);
                csv.add_row("parametric", "GPU", n_src, FIXED_POINTS, wall);

                // MultiBazin GPU (greedy K=1..4)
                let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
                    let _ = ctx.batch_pso_multi_bazin(&data, &batch_sources, 30, 60, 12, 42).unwrap();
                });
                print_row("GPU-MB", n_src, FIXED_POINTS, wall);
                csv.add_row("multi_bazin", "GPU", n_src, FIXED_POINTS, wall);
            }
        }
    }

    csv.write_csv("benchmarks/parametric_sources.csv");
}

// ===========================================================================
// Combined (legacy — produces the full CSV for docs)
// ===========================================================================

#[test]
#[ignore]
fn throughput_all() {
    let mut csv = CsvWriter::new();

    // Nonparametric point scaling
    print_header(&format!(
        "Nonparametric GP — point-count scaling (n_sources={})", FIXED_SOURCES));
    for &n_pts in POINT_COUNTS {
        let sources = synthetic::generate_n_sources(FIXED_SOURCES, n_pts);
        let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
            for (t, m, e, b) in &sources {
                let _ = fit_nonparametric(&build_mag_bands(t, m, e, b));
            }
        });
        print_row("CPU", FIXED_SOURCES, n_pts, wall);
        csv.add_row("nonparametric", "CPU", FIXED_SOURCES, n_pts, wall);
    }

    // Parametric point scaling
    print_header(&format!(
        "Parametric PSO+Laplace — point-count scaling (n_sources={})", FIXED_SOURCES));
    for &n_pts in POINT_COUNTS {
        let sources = synthetic::generate_n_sources(FIXED_SOURCES, n_pts);
        let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
            for (t, m, e, b) in &sources {
                let _ = fit_parametric(&build_flux_bands(t, m, e, b), false, UncertaintyMethod::Laplace);
            }
        });
        print_row("CPU", FIXED_SOURCES, n_pts, wall);
        csv.add_row("parametric", "CPU", FIXED_SOURCES, n_pts, wall);
    }

    // Nonparametric source scaling
    print_header(&format!(
        "Nonparametric GP — source-count scaling (pts/band={})", FIXED_POINTS));
    for &n_src in SOURCE_COUNTS {
        let sources = synthetic::generate_n_sources(n_src, FIXED_POINTS);
        let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
            for (t, m, e, b) in &sources {
                let _ = fit_nonparametric(&build_mag_bands(t, m, e, b));
            }
        });
        print_row("CPU", n_src, FIXED_POINTS, wall);
        csv.add_row("nonparametric", "CPU", n_src, FIXED_POINTS, wall);
    }

    // Parametric source scaling
    print_header(&format!(
        "Parametric PSO+Laplace — source-count scaling (pts/band={})", FIXED_POINTS));
    for &n_src in SOURCE_COUNTS {
        let sources = synthetic::generate_n_sources(n_src, FIXED_POINTS);
        let wall = bench_best_of(N_WARMUP, N_REPEAT, || {
            for (t, m, e, b) in &sources {
                let _ = fit_parametric(&build_flux_bands(t, m, e, b), false, UncertaintyMethod::Laplace);
            }
        });
        print_row("CPU", n_src, FIXED_POINTS, wall);
        csv.add_row("parametric", "CPU", n_src, FIXED_POINTS, wall);
    }

    csv.write_csv("benchmarks/throughput_results.csv");
}

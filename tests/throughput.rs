mod synthetic;

use lightcurve_fitting::{build_flux_bands, build_mag_bands, fit_nonparametric, fit_parametric, fit_thermal};
use std::time::Instant;

#[test]
#[ignore] // Only run explicitly for benchmarking: cargo test throughput -- --ignored
fn throughput_benchmark() {
    let n_sources = 10;
    let n_per_band = 30;
    let sources = synthetic::generate_n_sources(n_sources, n_per_band);

    let start = Instant::now();

    for (i, (times, mags, errs, bands)) in sources.iter().enumerate() {
        let mag_bands = build_mag_bands(times, mags, errs, bands);
        let flux_bands = build_flux_bands(times, mags, errs, bands);

        let (np_results, trained_gps) = fit_nonparametric(&mag_bands);
        assert!(
            !np_results.is_empty(),
            "source {i}: nonparametric should return results"
        );

        let p_results = fit_parametric(&flux_bands);
        assert!(
            !p_results.is_empty(),
            "source {i}: parametric should return results"
        );

        let t_result = fit_thermal(&mag_bands, Some(&trained_gps));
        assert!(
            t_result.is_some(),
            "source {i}: thermal should return Some"
        );

        eprintln!("source {}/{} done", i + 1, n_sources);
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!("Total wall time: {elapsed:.2}s for {n_sources} sources");

    std::fs::write("wall_time.txt", format!("{elapsed:.2}"))
        .expect("failed to write wall_time.txt");
}

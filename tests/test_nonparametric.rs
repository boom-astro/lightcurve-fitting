mod synthetic;

use lightcurve_fitting::{build_mag_bands, fit_nonparametric};

#[test]
fn nonparametric_returns_results_for_each_band() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 123);
    let mag_bands = build_mag_bands(&times, &mags, &errs, &bands);
    let (results, _gps) = fit_nonparametric(&mag_bands);

    assert!(
        !results.is_empty(),
        "should return at least one band result"
    );

    let result_bands: Vec<&str> = results.iter().map(|r| r.band.as_str()).collect();
    for band_name in mag_bands.keys() {
        assert!(
            result_bands.contains(&band_name.as_str()),
            "missing result for band {band_name}"
        );
    }
}

#[test]
fn nonparametric_peak_mag_is_brightest() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 456);
    let mag_bands = build_mag_bands(&times, &mags, &errs, &bands);
    let (results, _gps) = fit_nonparametric(&mag_bands);

    for result in &results {
        if let Some(peak) = result.peak_mag {
            // Peak mag (from GP) should be brighter than or close to the faintest observation
            let band_data = &mag_bands[&result.band];
            let faintest_obs = band_data
                .values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                peak < faintest_obs + 1.0,
                "peak_mag ({peak}) should be brighter than faintest obs ({faintest_obs})"
            );
        }
    }
}

#[test]
fn nonparametric_t0_in_range() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 789);
    let mag_bands = build_mag_bands(&times, &mags, &errs, &bands);
    let (results, _gps) = fit_nonparametric(&mag_bands);

    // t0 should be within the time range of the data (relative time)
    let t_max: f64 = mag_bands
        .values()
        .flat_map(|b| b.times.iter().copied())
        .fold(f64::NEG_INFINITY, f64::max);

    for result in &results {
        if let Some(t0) = result.t0 {
            assert!(
                t0 >= -10.0 && t0 <= t_max + 10.0,
                "t0 ({t0}) should be near the data range [0, {t_max}]"
            );
        }
    }
}

#[test]
fn nonparametric_chi2_finite() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 321);
    let mag_bands = build_mag_bands(&times, &mags, &errs, &bands);
    let (results, _gps) = fit_nonparametric(&mag_bands);

    for result in &results {
        if let Some(chi2) = result.chi2 {
            assert!(chi2.is_finite(), "chi2 should be finite");
            assert!(chi2 >= 0.0, "chi2 should be non-negative, got {chi2}");
        }
    }
}

#[test]
fn nonparametric_new_features_present() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 654);
    let mag_bands = build_mag_bands(&times, &mags, &errs, &bands);
    let (results, _gps) = fit_nonparametric(&mag_bands);

    for result in &results {
        // These fields exist (may be None if data insufficient, but struct has them)
        let _ = result.von_neumann_ratio;
        let _ = result.decay_power_law_index;
        let _ = result.mag_at_30d;
        let _ = result.mag_at_60d;
        let _ = result.mag_at_90d;
        let _ = result.pre_peak_rms;
        let _ = result.rise_amplitude_over_noise;
        let _ = result.post_peak_monotonicity;
    }

    // With 30 points per band, von_neumann_ratio should be computable for at least one band
    let any_von_neumann = results.iter().any(|r| r.von_neumann_ratio.is_some());
    assert!(
        any_von_neumann,
        "at least one band should have von_neumann_ratio"
    );
}

#[test]
fn nonparametric_n_obs_correct() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 111);
    let mag_bands = build_mag_bands(&times, &mags, &errs, &bands);
    let (results, _gps) = fit_nonparametric(&mag_bands);

    for result in &results {
        let expected = mag_bands[&result.band].values.len();
        assert_eq!(
            result.n_obs, expected,
            "n_obs should match band data length"
        );
    }
}

#[test]
fn nonparametric_empty_bands() {
    let (results, gps) = fit_nonparametric(&std::collections::HashMap::new());
    assert!(results.is_empty());
    assert!(gps.is_empty());
}

mod synthetic;

use lightcurve_fitting::{build_mag_bands, fit_thermal};

#[test]
fn thermal_returns_result() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 700);
    let mag_bands = build_mag_bands(&times, &mags, &errs, &bands);
    let result = fit_thermal(&mag_bands);
    assert!(result.is_some(), "fit_thermal should return Some");
}

#[test]
fn thermal_log_temp_in_range() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 800);
    let mag_bands = build_mag_bands(&times, &mags, &errs, &bands);
    let result = fit_thermal(&mag_bands).unwrap();

    if let Some(log_temp) = result.log_temp_peak {
        assert!(
            log_temp >= 3.0 && log_temp <= 6.0,
            "log_temp_peak should be in [3.0, 6.0], got {log_temp}"
        );
    }
}

#[test]
fn thermal_chi2_finite() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 900);
    let mag_bands = build_mag_bands(&times, &mags, &errs, &bands);
    let result = fit_thermal(&mag_bands).unwrap();

    if let Some(chi2) = result.chi2 {
        assert!(chi2.is_finite(), "chi2 should be finite, got {chi2}");
        assert!(chi2 >= 0.0, "chi2 should be non-negative, got {chi2}");
    }
}

#[test]
fn thermal_ref_band_is_g_or_r() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 1000);
    let mag_bands = build_mag_bands(&times, &mags, &errs, &bands);
    let result = fit_thermal(&mag_bands).unwrap();

    assert!(
        result.ref_band == "g" || result.ref_band == "r",
        "ref_band should be g or r, got {}",
        result.ref_band
    );
}

#[test]
fn thermal_n_bands_used() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 1100);
    let mag_bands = build_mag_bands(&times, &mags, &errs, &bands);
    let result = fit_thermal(&mag_bands).unwrap();

    // With g, r, i we should use at least 1 non-reference band
    assert!(
        result.n_bands_used >= 1,
        "should use at least 1 non-reference band, got {}",
        result.n_bands_used
    );
}

#[test]
fn thermal_n_color_obs() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 1200);
    let mag_bands = build_mag_bands(&times, &mags, &errs, &bands);
    let result = fit_thermal(&mag_bands).unwrap();

    // With 30 points in r and i bands, should have many color observations
    assert!(
        result.n_color_obs >= 10,
        "should have at least 10 color observations, got {}",
        result.n_color_obs
    );
}

#[test]
fn thermal_empty_bands() {
    let result = fit_thermal(&std::collections::HashMap::new());
    assert!(result.is_none(), "empty bands should return None");
}

#[test]
fn thermal_single_band_returns_none() {
    // With only one band, thermal fitting needs a reference band + at least one other
    let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let mags = vec![20.0, 19.5, 19.0, 19.5, 20.0];
    let errs = vec![0.1; 5];
    let bands: Vec<String> = vec!["r".to_string(); 5];
    let mag_bands = lightcurve_fitting::build_mag_bands(&times, &mags, &errs, &bands);
    let result = fit_thermal(&mag_bands);
    // Should return Some but with None fields (no color info) or None
    if let Some(r) = result {
        // Only one band, n_color_obs should be 0
        assert_eq!(r.n_color_obs, 0);
    }
}

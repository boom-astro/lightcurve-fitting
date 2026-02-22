mod synthetic;

use lightcurve_fitting::common::{
    build_flux_bands, build_mag_bands, compute_decay_rate, compute_fwhm, compute_rise_rate,
    extract_decay_timescale, extract_rise_timescale, finite_or_none, mag2flux, median,
};

// ---------------------------------------------------------------------------
// mag2flux
// ---------------------------------------------------------------------------

#[test]
fn mag2flux_zeropoint() {
    // mag = 23.9 (ZP) → flux = 1.0
    let (flux, _) = mag2flux(23.9, 0.1, 23.9);
    assert!((flux - 1.0).abs() < 1e-6, "flux at ZP should be 1.0, got {flux}");
}

#[test]
fn mag2flux_brighter() {
    // mag = 21.4 → flux = 10^(0.4 * 2.5) = 10.0
    let (flux, _) = mag2flux(21.4, 0.1, 23.9);
    assert!(
        (flux - 10.0).abs() < 0.01,
        "mag=21.4 should give flux~10.0, got {flux}"
    );
}

#[test]
fn mag2flux_error_propagation() {
    let (flux, flux_err) = mag2flux(23.9, 0.1, 23.9);
    // fluxerr = mag_err / FACTOR * flux, FACTOR ≈ 1.0857
    let expected_err = 0.1 / 1.0857362047581294 * flux;
    assert!(
        (flux_err - expected_err).abs() < 1e-8,
        "flux_err mismatch"
    );
}

// ---------------------------------------------------------------------------
// build_mag_bands
// ---------------------------------------------------------------------------

#[test]
fn build_mag_bands_groups_by_band() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(10, 42);
    let result = build_mag_bands(&times, &mags, &errs, &bands);
    assert!(result.contains_key("g"));
    assert!(result.contains_key("r"));
    assert!(result.contains_key("i"));
    assert_eq!(result.len(), 3);
}

#[test]
fn build_mag_bands_relative_time() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(10, 42);
    let result = build_mag_bands(&times, &mags, &errs, &bands);
    // All times should be relative (minimum across all bands should be 0)
    let global_min: f64 = result
        .values()
        .flat_map(|b| b.times.iter().copied())
        .fold(f64::INFINITY, f64::min);
    assert!(
        global_min.abs() < 1e-10,
        "minimum relative time should be ~0, got {global_min}"
    );
}

#[test]
fn build_mag_bands_correct_count() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(10, 42);
    let result = build_mag_bands(&times, &mags, &errs, &bands);
    for band_data in result.values() {
        assert_eq!(band_data.times.len(), 10);
        assert_eq!(band_data.values.len(), 10);
        assert_eq!(band_data.errors.len(), 10);
    }
}

#[test]
fn build_mag_bands_empty() {
    let result = build_mag_bands(&[], &[], &[], &[]);
    assert!(result.is_empty());
}

// ---------------------------------------------------------------------------
// build_flux_bands
// ---------------------------------------------------------------------------

#[test]
fn build_flux_bands_positive_values() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(10, 42);
    let result = build_flux_bands(&times, &mags, &errs, &bands);
    for band_data in result.values() {
        for &v in &band_data.values {
            assert!(v > 0.0, "flux should be positive, got {v}");
        }
        for &e in &band_data.errors {
            assert!(e > 0.0, "flux error should be positive, got {e}");
        }
    }
}

#[test]
fn build_flux_bands_groups_by_band() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(10, 42);
    let result = build_flux_bands(&times, &mags, &errs, &bands);
    assert_eq!(result.len(), 3);
    assert!(result.contains_key("g"));
    assert!(result.contains_key("r"));
    assert!(result.contains_key("i"));
}

#[test]
fn build_flux_bands_empty() {
    let result = build_flux_bands(&[], &[], &[], &[]);
    assert!(result.is_empty());
}

// ---------------------------------------------------------------------------
// median
// ---------------------------------------------------------------------------

#[test]
fn median_odd() {
    let mut v = vec![3.0, 1.0, 2.0];
    assert_eq!(median(&mut v), Some(2.0));
}

#[test]
fn median_even() {
    let mut v = vec![4.0, 1.0, 3.0, 2.0];
    assert_eq!(median(&mut v), Some(2.5));
}

#[test]
fn median_single() {
    let mut v = vec![5.0];
    assert_eq!(median(&mut v), Some(5.0));
}

#[test]
fn median_empty() {
    let mut v: Vec<f64> = vec![];
    assert_eq!(median(&mut v), None);
}

// ---------------------------------------------------------------------------
// finite_or_none
// ---------------------------------------------------------------------------

#[test]
fn finite_or_none_nan() {
    assert_eq!(finite_or_none(f64::NAN), None);
}

#[test]
fn finite_or_none_inf() {
    assert_eq!(finite_or_none(f64::INFINITY), None);
    assert_eq!(finite_or_none(f64::NEG_INFINITY), None);
}

#[test]
fn finite_or_none_finite() {
    assert_eq!(finite_or_none(3.14), Some(3.14));
    assert_eq!(finite_or_none(0.0), Some(0.0));
    assert_eq!(finite_or_none(-1.0), Some(-1.0));
}

// ---------------------------------------------------------------------------
// Timescale extraction on synthetic peaked data
// ---------------------------------------------------------------------------

fn make_peaked_data() -> (Vec<f64>, Vec<f64>) {
    // A simple peaked magnitude curve (lower mag = brighter)
    // Rise from mag=22 to mag=20, then decay to mag=23
    let times: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let mags: Vec<f64> = times
        .iter()
        .map(|&t| {
            let dt = t - 15.0;
            20.0 + 2.0 * (1.0 - (-dt.abs() / 10.0).exp()) + if dt < 0.0 { 0.5 } else { 0.0 }
        })
        .collect();
    (times, mags)
}

#[test]
fn extract_rise_timescale_reasonable() {
    let (times, mags) = make_peaked_data();
    // Find peak (minimum mag)
    let peak_idx = mags
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0;
    let rise = extract_rise_timescale(&times, &mags, peak_idx);
    assert!(rise.is_finite(), "rise timescale should be finite");
    assert!(rise > 0.0, "rise timescale should be positive, got {rise}");
}

#[test]
fn extract_decay_timescale_reasonable() {
    let (times, mags) = make_peaked_data();
    let peak_idx = mags
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0;
    let decay = extract_decay_timescale(&times, &mags, peak_idx);
    assert!(decay.is_finite(), "decay timescale should be finite");
    assert!(
        decay > 0.0,
        "decay timescale should be positive, got {decay}"
    );
}

#[test]
fn extract_rise_timescale_edge_peak_at_zero() {
    let times = vec![0.0, 1.0, 2.0];
    let mags = vec![19.0, 20.0, 21.0];
    let rise = extract_rise_timescale(&times, &mags, 0);
    assert!(rise.is_nan(), "rise at peak_idx=0 should be NaN");
}

// ---------------------------------------------------------------------------
// FWHM, rise_rate, decay_rate
// ---------------------------------------------------------------------------

#[test]
fn compute_fwhm_peaked_data() {
    let (times, mags) = make_peaked_data();
    let peak_idx = mags
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0;
    let (fwhm, t_before, t_after) = compute_fwhm(&times, &mags, peak_idx);
    // fwhm may or may not be finite depending on whether 0.75 mag threshold is crossed
    if fwhm.is_finite() {
        assert!(fwhm > 0.0, "FWHM should be positive, got {fwhm}");
        assert!(
            t_before < t_after,
            "t_before should be before t_after"
        );
    }
}

#[test]
fn compute_rise_rate_sign() {
    // Brightening (decreasing mag) → negative rise_rate
    let times: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let mags: Vec<f64> = times.iter().map(|&t| 22.0 - 0.1 * t).collect();
    let rate = compute_rise_rate(&times, &mags);
    assert!(rate.is_finite());
    assert!(rate < 0.0, "rise rate for brightening should be negative, got {rate}");
}

#[test]
fn compute_decay_rate_sign() {
    // Fading (increasing mag) → positive decay_rate
    let times: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let mags: Vec<f64> = times.iter().map(|&t| 20.0 + 0.05 * t).collect();
    let rate = compute_decay_rate(&times, &mags);
    assert!(rate.is_finite());
    assert!(rate > 0.0, "decay rate for fading should be positive, got {rate}");
}

#[test]
fn compute_rise_rate_too_few() {
    let rate = compute_rise_rate(&[1.0], &[20.0]);
    assert!(rate.is_nan());
}

#[test]
fn compute_decay_rate_too_few() {
    let rate = compute_decay_rate(&[1.0], &[20.0]);
    assert!(rate.is_nan());
}

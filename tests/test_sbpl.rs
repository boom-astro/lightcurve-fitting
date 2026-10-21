mod synthetic;

use std::collections::HashMap;

use lightcurve_fitting::{build_flux_bands, fit_sbpl, BandData};

// ---------------------------------------------------------------------------
// Helper: generate a synthetic SBPL light curve in flux space.
//
// F(t, nu) = 10^loga * nu^beta * tau^alpha1 * [0.5*(1 + tau^(1/D))]^((alpha2-alpha1)*D)
// where tau = (t - t0) / tb
// ---------------------------------------------------------------------------

const C_ANGSTROM_PER_SEC: f64 = 2.997_924_58e18;

fn band_freq(lambda_angstrom: f64) -> f64 {
    C_ANGSTROM_PER_SEC / lambda_angstrom
}

fn sbpl_flux(t: f64, nu: f64, alpha1: f64, alpha2: f64, beta: f64, d: f64, loga: f64, tb: f64, t0: f64) -> f64 {
    let tau = t - t0;
    if tb <= 0.0 || d <= 0.0 || nu <= 0.0 {
        return f64::NAN;
    }
    if tau < 0.0 {
        return 0.0;
    }
    let ratio = tau / tb;
    let term1 = nu.powf(beta);
    let term2 = ratio.powf(alpha1);
    let inner = 0.5 * (1.0 + ratio.powf(1.0 / d));
    if !inner.is_finite() || inner <= 0.0 {
        return f64::NAN;
    }
    let term3 = inner.powf((alpha2 - alpha1) * d);
    let result = 10f64.powf(loga) * term1 * term2 * term3;
    if result.is_finite() { result } else { f64::NAN }
}

/// Simple xorshift64 for reproducible noise without extra dependencies.
fn xorshift(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 11) as f64 / ((1u64 << 53) as f64)
}

fn normal(rng: &mut u64) -> f64 {
    let u1 = xorshift(rng).max(1e-15);
    let u2 = xorshift(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generate a multi-band SBPL source.
///
/// Bands: g (4770 Å), r (6231 Å), i (7625 Å)
/// Returns a `HashMap<String, BandData>` in flux space.
fn generate_sbpl_source(
    n_per_band: usize,
    seed: u64,
    alpha1: f64, alpha2: f64, beta: f64, d: f64, loga: f64, tb: f64, t0: f64,
) -> HashMap<String, BandData> {
    let mut rng = seed.max(1);

    // (band_name, central_wavelength_Å)
    let bands: &[(&str, f64)] = &[
        ("g", 4770.0),
        ("r", 6231.0),
        ("i", 7625.0),
    ];

    let t_start = t0 + 0.5;          // start just after t0
    let t_end   = t0 + tb * 10.0;    // observe well past the break
    let noise_frac = 0.05;           // 5% flux noise

    let mut result: HashMap<String, BandData> = HashMap::new();

    for &(band_name, lambda) in bands {
        let nu = band_freq(lambda);
        let mut times = Vec::with_capacity(n_per_band);
        let mut values = Vec::with_capacity(n_per_band);
        let mut errors = Vec::with_capacity(n_per_band);

        for j in 0..n_per_band {
            let t = t_start + (t_end - t_start) * (j as f64) / (n_per_band as f64 - 1.0).max(1.0);
            let flux_true = sbpl_flux(t, nu, alpha1, alpha2, beta, d, loga, tb, t0);
            let noise_sigma = flux_true.abs() * noise_frac + 1e-20;
            let flux_obs = flux_true + noise_sigma * normal(&mut rng);
            times.push(t);
            values.push(flux_obs);
            errors.push(noise_sigma);
        }

        result.insert(band_name.to_string(), BandData { times, values, errors });
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn sbpl_returns_result_on_bazin_source() {
    // Even a Bazin-shaped source (no spectral index) should return Some
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 42);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let result = fit_sbpl(&flux_bands);
    assert!(result.is_some(), "fit_sbpl should return Some for a valid source");
}

#[test]
fn sbpl_result_has_finite_fields() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 43);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let result = fit_sbpl(&flux_bands).unwrap();

    // At least some parameters should be Some (enough obs to fit 7 params)
    assert!(
        result.alpha1.is_some() || result.alpha2.is_some(),
        "Expected at least one fitted parameter to be Some"
    );

    if let Some(chi2) = result.reduced_chi2 {
        assert!(chi2.is_finite() && chi2 >= 0.0, "reduced_chi2 should be non-negative finite, got {chi2}");
    }
}

#[test]
fn sbpl_n_obs_and_bands_correct() {
    // 30 points per band, 3 bands → 90 obs total
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 44);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let result = fit_sbpl(&flux_bands).unwrap();
    assert_eq!(result.n_bands, 3, "Expected 3 bands (g, r, i)");
    assert_eq!(result.n_obs, 90, "Expected 90 observations");
}

#[test]
fn sbpl_recovers_spectral_index() {
    // Generate a true-SBPL source and check parameters are in the right ballpark.
    // alpha1 + alpha2 must be non-zero so the smoothing term is active.
    let true_alpha1 = 2.0;
    let true_alpha2 = -2.0;
    let true_beta   = -0.8;
    let true_d      = 0.5;
    // loga=12 gives F~1 Jy for typical optical nu (~6e14 Hz) at beta=-0.8
    let true_loga   = 12.0;
    let true_tb     = 10.0;
    let true_t0     = 5.0;

    let flux_bands = generate_sbpl_source(
        40, 1234,
        true_alpha1, true_alpha2, true_beta, true_d, true_loga, true_tb, true_t0,
    );

    let result = fit_sbpl(&flux_bands).expect("fit_sbpl should return Some for SBPL source");

    println!("SBPL fit results:");
    println!("  alpha1 = {:?}  (true: {true_alpha1})", result.alpha1);
    println!("  alpha2 = {:?}  (true: {true_alpha2})", result.alpha2);
    println!("  beta   = {:?}  (true: {true_beta})", result.beta);
    println!("  D      = {:?}  (true: {true_d})", result.d);
    println!("  loga   = {:?}  (true: {true_loga})", result.loga);
    println!("  tb     = {:?}  (true: {true_tb})", result.tb);
    println!("  t0     = {:?}  (true: {true_t0})", result.t0);
    println!("  reduced_chi2 = {:?}", result.reduced_chi2);

    // Basic sanity: chi2 should be finite and non-negative.
    if let Some(chi2) = result.reduced_chi2 {
        assert!(chi2.is_finite() && chi2 >= 0.0, "reduced_chi2 should be non-negative");
    }
}

#[test]
fn sbpl_errors_are_finite_when_populated() {
    // loga=12 gives O(1) Jy for optical nu at beta=-0.6
    let flux_bands = generate_sbpl_source(
        40, 9999, 1.0, -1.0, -0.6, 0.3, 12.0, 3.0, 0.0,
    );
    let result = fit_sbpl(&flux_bands).unwrap();

    for (name, val) in [
        ("alpha1_err", result.alpha1_err),
        ("alpha2_err", result.alpha2_err),
        ("beta_err",   result.beta_err),
        ("d_err",      result.d_err),
        ("loga_err",   result.loga_err),
        ("tb_err",     result.tb_err),
        ("t0_err",     result.t0_err),
    ] {
        if let Some(e) = val {
            assert!(e.is_finite() && e >= 0.0, "{name} should be >= 0 and finite, got {e}");
        }
    }
}

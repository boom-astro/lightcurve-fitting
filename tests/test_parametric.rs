mod synthetic;

use lightcurve_fitting::parametric::SviModelName;
use lightcurve_fitting::{build_flux_bands, fit_parametric};

#[test]
fn parametric_returns_results() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 100);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric(&flux_bands, false);

    assert!(
        !results.is_empty(),
        "parametric should return at least one band result"
    );
}

#[test]
fn parametric_pso_chi2_finite() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 200);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric(&flux_bands, false);

    for result in &results {
        if let Some(chi2) = result.pso_chi2 {
            assert!(chi2.is_finite(), "pso_chi2 should be finite");
        }
    }
}

#[test]
fn parametric_svi_mu_correct_length() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 300);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric(&flux_bands, false);

    for result in &results {
        let expected_len = match result.model {
            SviModelName::Bazin => 6,
            SviModelName::Villar => 7,
            SviModelName::MetzgerKN => 5,
            SviModelName::Tde => 7,
            SviModelName::Arnett => 5,
            SviModelName::Magnetar => 5,
            SviModelName::ShockCooling => 5,
            SviModelName::Afterglow => 6,
        };
        assert_eq!(
            result.svi_mu.len(),
            expected_len,
            "svi_mu should have {expected_len} params for {:?}, got {}",
            result.model,
            result.svi_mu.len()
        );
        assert_eq!(
            result.svi_log_sigma.len(),
            expected_len,
            "svi_log_sigma should match svi_mu length"
        );
    }
}

#[test]
fn parametric_per_model_chi2_populated() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 400);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric(&flux_bands, false);

    for result in &results {
        // per_model_chi2 should have entries (some may be None due to early stopping)
        assert!(
            !result.per_model_chi2.is_empty(),
            "per_model_chi2 should not be empty"
        );
        // At minimum the selected model should have a chi2
        assert!(
            result.per_model_chi2.contains_key(&result.model),
            "per_model_chi2 should contain the selected model"
        );
    }
}

#[test]
fn parametric_n_obs_correct() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 500);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric(&flux_bands, false);

    for result in &results {
        let band_data = &flux_bands[&result.band];
        assert_eq!(
            result.n_obs,
            band_data.values.len(),
            "n_obs should match band data length"
        );
    }
}

#[test]
fn parametric_empty_bands() {
    let results = fit_parametric(&std::collections::HashMap::new(), false);
    assert!(results.is_empty());
}

#[test]
fn parametric_svi_elbo_finite() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 600);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric(&flux_bands, false);

    for result in &results {
        if let Some(elbo) = result.svi_elbo {
            assert!(elbo.is_finite(), "svi_elbo should be finite, got {elbo}");
        }
    }
}

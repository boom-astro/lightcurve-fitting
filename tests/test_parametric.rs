mod synthetic;

use lightcurve_fitting::parametric::SviModelName;
use lightcurve_fitting::{build_flux_bands, fit_parametric, UncertaintyMethod};

#[test]
fn parametric_returns_results() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 100);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric(&flux_bands, false, UncertaintyMethod::Svi);

    assert!(
        !results.is_empty(),
        "parametric should return at least one band result"
    );
}

#[test]
fn parametric_pso_chi2_finite() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 200);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric(&flux_bands, false, UncertaintyMethod::Svi);

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
    let results = fit_parametric(&flux_bands, false, UncertaintyMethod::Svi);

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
    let results = fit_parametric(&flux_bands, false, UncertaintyMethod::Svi);

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
    let results = fit_parametric(&flux_bands, false, UncertaintyMethod::Svi);

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
    let results = fit_parametric(&std::collections::HashMap::new(), false, UncertaintyMethod::Svi);
    assert!(results.is_empty());
}

#[test]
fn parametric_svi_elbo_finite() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 600);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric(&flux_bands, false, UncertaintyMethod::Svi);

    for result in &results {
        if let Some(elbo) = result.svi_elbo {
            assert!(elbo.is_finite(), "svi_elbo should be finite, got {elbo}");
        }
    }
}

// ---------------------------------------------------------------------------
// Laplace approximation tests
// ---------------------------------------------------------------------------

#[test]
fn parametric_laplace_returns_results() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 700);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric(&flux_bands, false, UncertaintyMethod::Laplace);

    assert!(
        !results.is_empty(),
        "laplace should return at least one band result"
    );
    for result in &results {
        assert_eq!(
            result.uncertainty_method,
            UncertaintyMethod::Laplace,
            "uncertainty_method should be Laplace"
        );
    }
}

#[test]
fn parametric_laplace_mu_correct_length() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 800);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric(&flux_bands, false, UncertaintyMethod::Laplace);

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
            "laplace mu should have {expected_len} params for {:?}, got {}",
            result.model,
            result.svi_mu.len()
        );
        assert_eq!(
            result.svi_log_sigma.len(),
            expected_len,
            "laplace log_sigma should match mu length"
        );
    }
}

#[test]
fn parametric_laplace_log_sigma_finite() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 900);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric(&flux_bands, false, UncertaintyMethod::Laplace);

    for result in &results {
        for (j, &ls) in result.svi_log_sigma.iter().enumerate() {
            assert!(
                ls.is_finite(),
                "laplace log_sigma[{j}] should be finite, got {ls} for {:?}",
                result.model
            );
        }
    }
}

#[test]
fn parametric_laplace_uncertainties_reasonable() {
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 1000);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric(&flux_bands, false, UncertaintyMethod::Laplace);

    for result in &results {
        for (j, &ls) in result.svi_log_sigma.iter().enumerate() {
            let sigma = ls.exp();
            assert!(
                sigma > 0.0 && sigma < 1e4,
                "laplace sigma[{j}] = {sigma} out of (0, 1e4) range for {:?}",
                result.model
            );
        }
    }
}

#[test]
fn metzger_kn_flux_scale_normalizes_detections() {
    use lightcurve_fitting::{eval_model_flux, fit_parametric_model};

    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 1100);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let results = fit_parametric_model(
        &flux_bands,
        false,
        UncertaintyMethod::Laplace,
        Some(SviModelName::MetzgerKN),
    );
    assert!(!results.is_empty());

    let mut n_checked = 0;
    for result in &results {
        assert_eq!(result.model, SviModelName::MetzgerKN);
        let band = &flux_bands[&result.band];
        let det_times: Vec<f64> = band
            .times
            .iter()
            .zip(band.values.iter().zip(band.errors.iter()))
            .filter(|(_, (f, e))| !(**e > 0.0 && **f / **e < 3.0))
            .map(|(t, _)| *t)
            .collect();

        for (params, scale) in [
            (&result.pso_params, result.pso_flux_scale),
            (&result.svi_mu, result.svi_flux_scale),
        ] {
            assert!((0.1..=10.0).contains(&scale), "scale {scale} outside clamp range");
            let peak = eval_model_flux(SviModelName::MetzgerKN, params, &det_times)
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max);
            let raw = 1.0 / peak;
            if (0.1..=10.0).contains(&raw) {
                assert!(
                    (peak * scale - 1.0).abs() < 1e-9,
                    "scaled peak {} should be 1",
                    peak * scale
                );
                n_checked += 1;
            }
        }
    }
    assert!(n_checked > 0, "no fit landed inside the clamp range");
}

#[test]
fn gpu_finalize_pairs_results_by_band_name() {
    use lightcurve_fitting::{finalize_parametric_with_gpu_svi, fit_parametric_model, GpuPsoBandResult};
    use std::collections::HashMap;

    // All bands have the same number of points, so ordering by size is ambiguous.
    let (times, mags, errs, bands) = synthetic::generate_bazin_source(30, 1200);
    let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
    let cpu = fit_parametric_model(&flux_bands, false, UncertaintyMethod::Laplace, Some(SviModelName::Bazin));
    assert!(cpu.len() >= 2);

    let mut band_names = Vec::new();
    let mut gpu_results = Vec::new();
    let mut svi_outputs = Vec::new();
    for r in cpu.iter().rev() {
        band_names.push(r.band.clone());
        gpu_results.push(GpuPsoBandResult {
            model: r.model,
            pso_params: r.pso_params.clone(),
            pso_cost: 0.0,
            per_model_chi2: HashMap::new(),
            per_model_params: HashMap::new(),
            multi_bazin: None,
        });
        svi_outputs.push((r.svi_mu.clone(), r.svi_log_sigma.clone(), 0.0));
    }

    let results = finalize_parametric_with_gpu_svi(&flux_bands, &gpu_results, &svi_outputs, &band_names);
    assert_eq!(results.len(), cpu.len());
    for r in &results {
        let expected = cpu.iter().find(|c| c.band == r.band).unwrap();
        assert_eq!(r.pso_params, expected.pso_params, "band {} got another band's params", r.band);
    }
}

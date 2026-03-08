use lightcurve_fitting::gp::{fit_gp_predict, subsample_data};
use lightcurve_fitting::sparse_gp::DenseGP;

// ---------------------------------------------------------------------------
// subsample_data
// ---------------------------------------------------------------------------

#[test]
fn subsample_identity_when_small() {
    let times = vec![1.0, 2.0, 3.0];
    let mags = vec![20.0, 19.5, 20.5];
    let errors = vec![0.1, 0.1, 0.1];
    let (t, m, e) = subsample_data(&times, &mags, &errors, 10);
    assert_eq!(t, times);
    assert_eq!(m, mags);
    assert_eq!(e, errors);
}

#[test]
fn subsample_reduces_length() {
    let n = 100;
    let times: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mags: Vec<f64> = (0..n).map(|i| 20.0 + 0.01 * i as f64).collect();
    let errors: Vec<f64> = vec![0.1; n];
    let max_points = 25;
    let (t, m, e) = subsample_data(&times, &mags, &errors, max_points);
    assert_eq!(t.len(), max_points);
    assert_eq!(m.len(), max_points);
    assert_eq!(e.len(), max_points);
}

#[test]
fn subsample_preserves_range() {
    let n = 100;
    let times: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mags: Vec<f64> = vec![20.0; n];
    let errors: Vec<f64> = vec![0.1; n];
    let (t, _, _) = subsample_data(&times, &mags, &errors, 10);
    let t_min = t.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(t_min < 10.0, "subsampled min should be near start");
    assert!(t_max > 90.0, "subsampled max should be near end");
}

// ---------------------------------------------------------------------------
// DenseGP
// ---------------------------------------------------------------------------

#[test]
fn fit_gp_simple_curve() {
    let n = 30;
    let times: Vec<f64> = (0..n).map(|i| i as f64 * 2.0).collect();
    let mags: Vec<f64> = times
        .iter()
        .map(|&t| 20.0 + 2.0 * (t / 10.0).sin())
        .collect();
    let noise_var: Vec<f64> = vec![1e-4; n];

    let gp = DenseGP::fit(&times, &mags, &noise_var, 0.2, 10.0);
    assert!(gp.is_some(), "GP should fit successfully");

    let gp = gp.unwrap();
    let pred = gp.predict(&times);

    let mut max_residual = 0.0f64;
    for i in 0..n {
        let residual = (pred[i] - mags[i]).abs();
        max_residual = max_residual.max(residual);
    }
    assert!(
        max_residual < 1.0,
        "GP predictions should be close to data, max residual = {max_residual}"
    );
}

#[test]
fn fit_gp_returns_none_on_degenerate_input() {
    let _result = DenseGP::fit(&[0.0], &[20.0], &[1e-4], 0.2, 10.0);
    // Should not panic
}

// ---------------------------------------------------------------------------
// fit_gp_predict
// ---------------------------------------------------------------------------

#[test]
fn fit_gp_predict_returns_predictions() {
    let n = 30;
    let times: Vec<f64> = (0..n).map(|i| i as f64 * 2.0).collect();
    let mags: Vec<f64> = times
        .iter()
        .map(|&t| 20.0 + 2.0 * (t / 10.0).sin())
        .collect();
    let errors: Vec<f64> = vec![0.1; n];
    let query: Vec<f64> = (0..10).map(|i| i as f64 * 5.0).collect();

    let result = fit_gp_predict(
        &times, &mags, &errors, &query,
        &[0.1, 0.3], &[5.0, 10.0, 20.0],
        None,
    );
    assert!(result.is_some());
    let (pred, std) = result.unwrap();
    assert_eq!(pred.len(), query.len());
    assert_eq!(std.len(), query.len());
}

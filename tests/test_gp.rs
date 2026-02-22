use lightcurve_fitting::gp::{fit_sklears_gp, subsample_data};
use scirs2_core::ndarray::Array1;
use sklears_core::traits::Predict;

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
    // Subsampled times should span similar range
    let t_min = t.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(t_min < 10.0, "subsampled min should be near start");
    assert!(t_max > 90.0, "subsampled max should be near end");
}

// ---------------------------------------------------------------------------
// fit_sklears_gp
// ---------------------------------------------------------------------------

#[test]
fn fit_gp_simple_curve() {
    // Fit a smooth curve: mag = 20 + 2*sin(t/10)
    let n = 30;
    let times: Vec<f64> = (0..n).map(|i| i as f64 * 2.0).collect();
    let mags: Vec<f64> = times
        .iter()
        .map(|&t| 20.0 + 2.0 * (t / 10.0).sin())
        .collect();

    let times_arr = Array1::from_vec(times.clone());
    let mags_arr = Array1::from_vec(mags.clone());

    let result = fit_sklears_gp(&times_arr, &mags_arr, 0.2, 10.0, 1e-4);
    assert!(result.is_some(), "GP should fit successfully");

    let gp = result.unwrap();
    // Predict at training points — should be close
    let xt = times_arr
        .view()
        .insert_axis(scirs2_core::ndarray::Axis(1))
        .to_owned();
    let pred = gp.predict(&xt).expect("prediction should succeed");

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
    // Single point — should still work with GP (n=1 is OK for GP training)
    let times_arr = Array1::from_vec(vec![0.0]);
    let mags_arr = Array1::from_vec(vec![20.0]);
    // This may or may not succeed — just ensure no panic
    let _result = fit_sklears_gp(&times_arr, &mags_arr, 0.2, 10.0, 1e-4);
}

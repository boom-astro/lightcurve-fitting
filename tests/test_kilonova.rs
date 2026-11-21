use std::collections::HashMap;

use lightcurve_fitting::{fit_kilonova, fit_kilonova_batch, kilonova_mags, BandData, KilonovaOptions, KilonovaPrior};

/// A kilonova-like parameter set: (log_l1, a1, a2, log_t1, beta).
const KN: [f64; 5] = [41.6296, 0.2808, -1.8793, 3.7707, 0.3854];

fn band(times: Vec<f64>, mags: Vec<f64>, err: f64) -> BandData {
    let errors = vec![err; times.len()];
    BandData { times, values: mags, errors }
}

/// Noise-free light curve from the model itself, observed `nights` times per band.
fn synthetic(params: &[f64], t_exp: f64, t0: f64, nights: usize, bands: &[&str], dm: f64) -> HashMap<String, BandData> {
    let mut out = HashMap::new();
    for (k, b) in bands.iter().enumerate() {
        let times: Vec<f64> = (0..nights).map(|n| t0 + n as f64 + 0.05 * k as f64).collect();
        let since: Vec<f64> = times.iter().map(|t| t - t_exp).collect();
        let mags = kilonova_mags(params, &since, b).unwrap().iter().map(|m| m + dm).collect();
        out.insert(b.to_string(), band(times, mags, 0.05));
    }
    out
}

/// Days since the GW170817 merger (2017-08-17 12:41:04.4 UTC) for an ISO timestamp.
fn days_since_gw170817(iso: &str) -> f64 {
    fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
        let y = if m <= 2 { y - 1 } else { y };
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146097 + doe - 719468
    }
    let (date, time) = iso.split_once('T').unwrap();
    let ymd: Vec<i64> = date.split('-').map(|x| x.parse().unwrap()).collect();
    let hms: Vec<f64> = time.split(':').map(|x| x.parse().unwrap()).collect();
    let day = days_from_civil(ymd[0], ymd[1], ymd[2]) as f64 + (hms[0] * 3600.0 + hms[1] * 60.0 + hms[2]) / 86400.0;
    let merger = days_from_civil(2017, 8, 17) as f64 + (12.0 * 3600.0 + 41.0 * 60.0 + 4.4) / 86400.0;
    day - merger
}

/// AT2017gfo detections, de-reddened with E(B−V) = 0.105 (Schlafly & Finkbeiner 2011), as absolute mags.
fn at2017gfo() -> HashMap<String, BandData> {
    let ext: HashMap<&str, f64> = [
        ("ps1::g", 3.172), ("ps1::r", 2.271), ("ps1::i", 1.682), ("ps1::z", 1.322), ("ps1::y", 1.087),
        ("2massj", 0.709), ("2massh", 0.449), ("2massks", 0.302),
    ]
    .into_iter()
    .collect();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/AT2017gfo.dat")).unwrap();
    let mut out: HashMap<String, BandData> = HashMap::new();
    for line in text.lines() {
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() < 4 {
            continue;
        }
        let (Some(a), Ok(m), Ok(e)) = (ext.get(p[1]), p[2].parse::<f64>(), p[3].parse::<f64>()) else { continue };
        if !e.is_finite() {
            continue; // upper limit
        }
        let bd = out.entry(p[1].to_string()).or_insert_with(|| band(vec![], vec![], 0.0));
        bd.times.push(days_since_gw170817(p[0]));
        bd.values.push(m - a * 0.105 - 33.05);
        bd.errors.push(e);
    }
    out
}

#[test]
fn model_matches_python_reference() {
    // SciPy/numpy evaluation of the same tables and offsets.
    let expected = [
        (1.0, [-14.93534, -15.38464, -15.65777]),
        (2.0, [-14.25063, -15.0068, -15.51447]),
        (4.0, [-12.22948, -13.41137, -14.22499]),
    ];
    for (t, mags) in expected {
        for (b, m) in ["ztfg", "ztfr", "ztfi"].iter().zip(mags) {
            let got = kilonova_mags(&KN, &[t], b).unwrap()[0];
            assert!((got - m).abs() < 1e-3, "{b} at {t} d: {got} vs {m}");
        }
    }
}

#[test]
fn recovers_parameters_with_known_merger_time() {
    let truth = [41.4, 0.1, -1.5, 3.8, 0.45];
    let data = synthetic(&truth, 0.0, 0.5, 6, &["ztfg", "ztfr", "ztfi"], 0.0);
    let opts = KilonovaOptions { t_exp: Some(0.0), distance_modulus: Some(0.0), ..Default::default() };
    let res = fit_kilonova(&data, &opts).unwrap();
    for (got, want) in res.free.params.iter().zip(truth) {
        assert!((got - want).abs() < 1e-3, "free fit {:?} vs truth {:?}", res.free.params, truth);
    }
    assert!(res.free.chi2 < 1e-6);
    assert!(res.prior_cost < 3.0, "prior cost {} for an in-prior kilonova", res.prior_cost);
    assert!(res.t_exp_fixed);
}

#[test]
fn recovers_free_merger_time() {
    let data = synthetic(&KN, -0.7, 0.0, 5, &["ztfg", "ztfr"], 36.0);
    let res = fit_kilonova(&data, &KilonovaOptions::default()).unwrap();
    assert!((res.free.t_exp + 0.7).abs() < 0.05, "t_exp {}", res.free.t_exp);
    assert!((res.free.params[0] - (KN[0] - 0.4 * 36.0)).abs() < 0.01);
    assert!(res.prior_cost < 1.0, "prior cost {}", res.prior_cost);
}

#[test]
fn hot_non_cooling_source_has_large_prior_cost() {
    // Blackbody that stays at 12,000 K and fades slowly: young-supernova-like colours.
    let hot = [41.0, -0.3, 0.0, 12000f64.log10(), 0.0];
    let data = synthetic(&hot, -1.0, 0.0, 5, &["ztfg", "ztfr"], 36.0);
    let res = fit_kilonova(&data, &KilonovaOptions::default()).unwrap();
    assert!(res.free.chi2 < 1e-3, "free fit should describe a blackbody exactly");
    assert!(res.prior_cost > 20.0, "prior cost {}", res.prior_cost);
}

#[test]
fn power_law_afterglow_has_large_prior_cost() {
    // F_nu ∝ t^-1.2 nu^-0.8: fades fast with constant colour.
    let mut data = HashMap::new();
    for (k, (b, lam)) in [("ztfg", 4746.0f64), ("ztfr", 6366.0)].iter().enumerate() {
        let times: Vec<f64> = (0..5).map(|n| n as f64 + 0.05 * k as f64).collect();
        let mags = times
            .iter()
            .map(|t| 19.0 + 2.5 * 1.2 * (t + 0.5f64).log10() - 2.5 * 0.8 * (lam / 6366.0).log10())
            .collect();
        data.insert(b.to_string(), band(times, mags, 0.05));
    }
    let res = fit_kilonova(&data, &KilonovaOptions::default()).unwrap();
    assert!(res.prior_cost > 10.0, "prior cost {}", res.prior_cost);
}

#[test]
fn at2017gfo_full_fit_matches_reference() {
    // Independent SciPy fit of the same model: free fit, merger time known, 0.4-10 d.
    let mut data = at2017gfo();
    for bd in data.values_mut() {
        let keep: Vec<usize> = (0..bd.times.len()).filter(|&i| bd.times[i] > 0.4 && bd.times[i] < 10.0).collect();
        *bd = BandData {
            times: keep.iter().map(|&i| bd.times[i]).collect(),
            values: keep.iter().map(|&i| bd.values[i]).collect(),
            errors: keep.iter().map(|&i| bd.errors[i]).collect(),
        };
    }
    let opts = KilonovaOptions { t_exp: Some(0.0), distance_modulus: Some(0.0), sys_err: 0.05, ..Default::default() };
    let res = fit_kilonova(&data, &opts).unwrap();
    assert_eq!(res.n_obs, 112);
    let want = [41.6839, -0.5698, -0.3994, 3.7509, 0.4084];
    for (got, w) in res.free.params.iter().zip(want) {
        assert!((got - w).abs() < 0.01, "free params {:?} vs {:?}", res.free.params, want);
    }
    assert!((res.free.rms - 0.3192).abs() < 0.005, "rms {}", res.free.rms);
}

#[test]
fn at2017gfo_ztf_like_subset_matches_reference() {
    // g and r, first point per night over the first 3.5 days, merger time free.
    let full = at2017gfo();
    let mut data = HashMap::new();
    for b in ["ps1::g", "ps1::r"] {
        let bd = &full[b];
        let mut order: Vec<usize> = (0..bd.times.len()).filter(|&i| bd.times[i] < 3.5).collect();
        order.sort_by(|&i, &j| bd.times[i].partial_cmp(&bd.times[j]).unwrap());
        let mut seen = Vec::new();
        let mut keep = Vec::new();
        for i in order {
            let night = (bd.times[i] + 0.3).floor() as i64;
            if !seen.contains(&night) {
                seen.push(night);
                keep.push(i);
            }
        }
        data.insert(b.to_string(), BandData {
            times: keep.iter().map(|&i| bd.times[i]).collect(),
            values: keep.iter().map(|&i| bd.values[i]).collect(),
            errors: keep.iter().map(|&i| bd.errors[i]).collect(),
        });
    }
    let res = fit_kilonova(&data, &KilonovaOptions::default()).unwrap();
    assert_eq!(res.n_obs, 8);
    assert!((res.prior_cost - 4.91).abs() < 0.1, "prior cost {}", res.prior_cost);
    assert!((res.kn_prior.t_exp - 0.371).abs() < 0.02, "t_exp {}", res.kn_prior.t_exp);
}

#[test]
fn too_few_points_or_unknown_bands() {
    let mut data = HashMap::new();
    data.insert("ztfg".to_string(), band(vec![0.0, 1.0], vec![19.0, 19.8], 0.05));
    data.insert("foo".to_string(), band(vec![0.0, 1.0, 2.0], vec![19.0, 19.5, 20.0], 0.05));
    assert!(fit_kilonova(&data, &KilonovaOptions::default()).is_none());
    data.insert("ztfr".to_string(), band(vec![0.1, 1.1], vec![18.8, 19.4], 0.05));
    let res = fit_kilonova(&data, &KilonovaOptions::default()).unwrap();
    assert_eq!(res.bands_skipped, vec!["foo".to_string()]);
    assert_eq!(res.bands_used, vec!["ztfg".to_string(), "ztfr".to_string()]);
    assert!(kilonova_mags(&KN, &[1.0], "u").is_none());
}

#[test]
fn batch_matches_single() {
    let a = synthetic(&KN, -0.5, 0.0, 4, &["ztfg", "ztfr"], 36.0);
    let b = synthetic(&[41.0, -0.3, 0.0, 4.08, 0.0], -1.0, 0.0, 4, &["ztfg", "ztfr"], 36.0);
    let opts = KilonovaOptions::default();
    let batch = fit_kilonova_batch(&[a.clone(), b.clone()], &opts);
    for (src, res) in [a, b].iter().zip(batch) {
        let single = fit_kilonova(src, &opts).unwrap();
        assert!((res.unwrap().prior_cost - single.prior_cost).abs() < 1e-12);
    }
}

#[test]
fn last_nondetection_bounds_free_merger_time() {
    let data = synthetic(&KN, -0.7, 0.0, 5, &["ztfg", "ztfr"], 36.0);
    let opts = KilonovaOptions { t_last_nondet: Some(-0.4), ..Default::default() };
    let res = fit_kilonova(&data, &opts).unwrap();
    for fit in [&res.free, &res.kn_prior] {
        assert!(fit.t_exp >= -0.4 - 1e-9 && fit.t_exp <= -0.05 + 1e-9, "t_exp {}", fit.t_exp);
    }
    // A non-detection before the true merger time leaves the recovery intact.
    let opts = KilonovaOptions { t_last_nondet: Some(-1.2), ..Default::default() };
    let res = fit_kilonova(&data, &opts).unwrap();
    assert!((res.free.t_exp + 0.7).abs() < 0.05, "t_exp {}", res.free.t_exp);
    // One just before the first point keeps a short window.
    let opts = KilonovaOptions { t_last_nondet: Some(-0.02), ..Default::default() };
    let res = fit_kilonova(&data, &opts).unwrap();
    assert!(res.free.t_exp >= -0.02 - 1e-9 && res.free.t_exp < 0.0);
}

#[test]
fn runtime_prior_and_offsets() {
    let data = synthetic(&KN, -0.7, 0.0, 5, &["ztfg", "ztfr", "ztfi"], 36.0);
    let base = fit_kilonova(&data, &KilonovaOptions::default()).unwrap();

    // Passing the built-in prior explicitly changes nothing.
    let explicit = KilonovaOptions { prior: Some(KilonovaPrior::default()), ..Default::default() };
    let same = fit_kilonova(&data, &explicit).unwrap();
    assert!((same.prior_cost - base.prior_cost).abs() < 1e-12);

    // Two identical components give the same prior cost as one.
    let mut two = KilonovaPrior::default();
    two.weights = vec![0.5, 0.5];
    two.means.push(two.means[0].clone());
    two.covs.push(two.covs[0].clone());
    let res = fit_kilonova(&data, &KilonovaOptions { prior: Some(two), ..Default::default() }).unwrap();
    assert!((res.prior_cost - base.prior_cost).abs() < 1e-9, "{} vs {}", res.prior_cost, base.prior_cost);

    // An offsets map without i skips ztfi.
    let mut offsets = std::collections::HashMap::new();
    offsets.insert("g".to_string(), (0.0, 0.0));
    offsets.insert("r".to_string(), (0.0, 0.0));
    let custom = KilonovaPrior { offsets: Some(offsets), ..KilonovaPrior::default() };
    let res = fit_kilonova(&data, &KilonovaOptions { prior: Some(custom), ..Default::default() }).unwrap();
    assert_eq!(res.bands_skipped, vec!["ztfi".to_string()]);

    // A covariance that is not positive definite is rejected.
    let mut bad = KilonovaPrior::default();
    bad.covs[0][1][1] = -1.0;
    assert!(fit_kilonova(&data, &KilonovaOptions { prior: Some(bad), ..Default::default() }).is_none());
}

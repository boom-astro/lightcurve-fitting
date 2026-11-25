"""Tests for all Python-exposed functions in lightcurve_fitting."""

import math
import lightcurve_fitting as lcf


# ---------------------------------------------------------------------------
# Helpers: synthetic light curve data
# ---------------------------------------------------------------------------

def make_bazin_lc(n=50, seed=42):
    """Generate a simple Bazin-like light curve in 3 bands."""
    import random
    rng = random.Random(seed)

    times, mags, errs, bands = [], [], [], []
    for band_name in ["g", "r", "i"]:
        for i in range(n):
            t = -20.0 + 80.0 * i / (n - 1)
            # Bazin-like shape: bright at t=0, fading
            flux = 1.0 * math.exp(-abs(t) / 15.0) + 0.01
            mag = -2.5 * math.log10(flux) + 20.0
            mag += rng.gauss(0, 0.05)
            times.append(t)
            mags.append(mag)
            errs.append(0.05)
            bands.append(band_name)
    return times, mags, errs, bands


# ---------------------------------------------------------------------------
# Tests: Band construction
# ---------------------------------------------------------------------------

def test_build_mag_bands():
    t, m, e, b = make_bazin_lc()
    bands = lcf.build_mag_bands(t, m, e, b)
    assert len(bands) == 3
    d = bands.to_dict()
    assert set(d.keys()) == {"g", "r", "i"}
    for name in d:
        times, vals, errors = d[name]
        assert len(times) == 50
        assert len(vals) == 50
        assert len(errors) == 50


def test_build_flux_bands():
    t, m, e, b = make_bazin_lc()
    bands = lcf.build_flux_bands(t, m, e, b)
    assert len(bands) == 3
    d = bands.to_dict()
    for name in d:
        times, vals, errors = d[name]
        assert len(times) == 50
        # Flux should be positive
        assert all(v > 0 for v in vals)


def test_banddata_from_dict():
    bands = lcf.BandDataMap.from_dict({
        "g": ([1.0, 2.0, 3.0], [20.0, 19.5, 20.0], [0.1, 0.1, 0.1]),
        "r": ([1.0, 2.0, 3.0], [19.0, 18.5, 19.0], [0.1, 0.1, 0.1]),
    })
    assert len(bands) == 2
    assert "BandDataMap" in repr(bands)


# ---------------------------------------------------------------------------
# Tests: Nonparametric fitting
# ---------------------------------------------------------------------------

def test_fit_nonparametric():
    t, m, e, b = make_bazin_lc()
    bands = lcf.build_mag_bands(t, m, e, b)
    results = lcf.fit_nonparametric(bands)
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "band" in r
        assert "peak_mag" in r
        assert "n_obs" in r
        assert r["n_obs"] == 50


def test_fit_nonparametric_with_subsample():
    t, m, e, b = make_bazin_lc(n=200)
    bands = lcf.build_mag_bands(t, m, e, b)
    # Default subsample
    r1 = lcf.fit_nonparametric(bands)
    # Custom subsample
    r2 = lcf.fit_nonparametric(bands, max_subsample=15)
    assert len(r1) > 0
    assert len(r2) > 0


# ---------------------------------------------------------------------------
# Tests: Parametric fitting
# ---------------------------------------------------------------------------

def test_fit_parametric_laplace():
    t, m, e, b = make_bazin_lc()
    bands = lcf.build_flux_bands(t, m, e, b)
    results = lcf.fit_parametric(bands, method="laplace")
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert "model" in r
        assert "pso_chi2" in r
        assert "svi_mu" in r
        assert "svi_log_sigma" in r
        assert r["uncertainty_method"] == "Laplace"


def test_fit_parametric_svi():
    t, m, e, b = make_bazin_lc()
    bands = lcf.build_flux_bands(t, m, e, b)
    results = lcf.fit_parametric(bands, method="svi")
    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert r["uncertainty_method"] == "Svi"


def test_fit_parametric_all_models():
    t, m, e, b = make_bazin_lc()
    bands = lcf.build_flux_bands(t, m, e, b)
    results = lcf.fit_parametric(bands, fit_all_models=True, method="laplace")
    assert len(results) > 0
    for r in results:
        assert "per_model_chi2" in r
        chi2s = r["per_model_chi2"]
        # Should have entries for multiple models
        assert len(chi2s) > 1


def test_fit_parametric_invalid_method():
    t, m, e, b = make_bazin_lc()
    bands = lcf.build_flux_bands(t, m, e, b)
    try:
        lcf.fit_parametric(bands, method="bogus")
        assert False, "Should have raised ValueError"
    except ValueError as ex:
        assert "bogus" in str(ex)


# ---------------------------------------------------------------------------
# Tests: Thermal fitting
# ---------------------------------------------------------------------------

def test_fit_thermal():
    t, m, e, b = make_bazin_lc()
    bands = lcf.build_mag_bands(t, m, e, b)
    result = lcf.fit_thermal(bands)
    # May be None for synthetic data if GP fails, but should not error
    if result is not None:
        assert "log_temp_peak" in result


# ---------------------------------------------------------------------------
# Tests: Combined fit_fast
# ---------------------------------------------------------------------------

def test_fit_fast():
    t, m, e, b = make_bazin_lc()
    bands = lcf.build_mag_bands(t, m, e, b)
    result = lcf.fit_fast(bands)
    assert "nonparametric" in result
    assert "thermal" in result
    assert isinstance(result["nonparametric"], list)
    assert len(result["nonparametric"]) > 0


def test_fit_fast_with_subsample():
    t, m, e, b = make_bazin_lc()
    bands = lcf.build_mag_bands(t, m, e, b)
    result = lcf.fit_fast(bands, max_subsample=15)
    assert "nonparametric" in result
    assert len(result["nonparametric"]) > 0


# ---------------------------------------------------------------------------
# Tests: Model evaluation
# ---------------------------------------------------------------------------

def test_eval_model():
    # Bazin with 6 params
    params = [0.5, 0.1, 10.0, 1.0, 2.0, -2.0]
    times = [0.0, 5.0, 10.0, 20.0, 50.0]
    fluxes = lcf.eval_model("Bazin", params, times)
    assert len(fluxes) == 5
    assert all(math.isfinite(f) for f in fluxes)


def test_eval_model_invalid():
    try:
        lcf.eval_model("NotAModel", [1.0], [0.0])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Tests: GP prediction
# ---------------------------------------------------------------------------

def test_fit_gp_predict():
    train_t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    train_v = [1.0, 1.5, 2.0, 1.8, 1.2, 0.8]
    train_e = [0.1] * 6
    query_t = [0.5, 1.5, 2.5, 3.5, 4.5]
    amps = [0.5, 1.0, 2.0]
    lengths = [0.5, 1.0, 2.0]
    result = lcf.fit_gp_predict(train_t, train_v, train_e, query_t, amps, lengths)
    assert result is not None
    preds, stds = result
    assert len(preds) == 5
    assert len(stds) == 5
    assert all(math.isfinite(p) for p in preds)
    assert all(s > 0 for s in stds)


# ---------------------------------------------------------------------------
# Tests: Kilonova model
# ---------------------------------------------------------------------------

def test_metzger_kn_mags():
    # Physical params: log10(M_ej/Msun), log10(v_ej/c), log10(kappa), t0
    params = [-1.5, -0.7, 1.0, 0.0]
    times = [0.5, 1.0, 2.0, 5.0, 10.0]
    bands = [("g", 6.3e14), ("r", 4.7e14), ("i", 3.9e14)]
    d_l_cm = 1.26e26  # ~40 Mpc

    result = lcf.metzger_kn_mags(params, times, bands, d_l_cm)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"g", "r", "i"}
    for name, mags in result.items():
        assert len(mags) == 5
        # Magnitudes should be finite (may be 99.0 for faint)
        assert all(math.isfinite(m) for m in mags)


# ---------------------------------------------------------------------------
# Tests: Batch functions
# ---------------------------------------------------------------------------

def test_fit_batch_fast():
    sources = []
    for seed in range(5):
        t, m, e, b = make_bazin_lc(seed=seed)
        sources.append(lcf.build_mag_bands(t, m, e, b))
    results = lcf.fit_batch_fast(sources)
    assert len(results) == 5
    for r in results:
        assert "nonparametric" in r
        assert "thermal" in r


def test_fit_batch_nonparametric():
    sources = []
    for seed in range(5):
        t, m, e, b = make_bazin_lc(seed=seed)
        sources.append(lcf.build_mag_bands(t, m, e, b))
    results = lcf.fit_batch_nonparametric(sources)
    assert len(results) == 5
    for r in results:
        assert isinstance(r, list)
        assert len(r) > 0
        assert "band" in r[0]


def test_fit_batch_parametric():
    sources = []
    for seed in range(5):
        t, m, e, b = make_bazin_lc(seed=seed)
        sources.append(lcf.build_flux_bands(t, m, e, b))
    results = lcf.fit_batch_parametric(sources, method="laplace")
    assert len(results) == 5
    for r in results:
        assert isinstance(r, list)
        assert len(r) > 0


# ---------------------------------------------------------------------------
# Run with pytest or directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
        except Exception as ex:
            print(f"  FAIL {test.__name__}: {ex}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------------------
# Tests: Stellar flares
# ---------------------------------------------------------------------------

def make_flare_lc(n=300, cadence=1.0 / 1440, seed=3):
    import random
    rng = random.Random(seed)
    t = [i * cadence for i in range(n)]
    model = lcf.flare_model(t, [0.06], [5.0 / 1440], [0.02], cadence)
    f = [m + rng.gauss(0, 1e-3) for m in model]
    return t, f, [1e-3] * n


def test_flare_model_shape():
    t = [-0.5, 0.0, 0.5, 3.0]
    y = lcf.flare_model(t, [0.0], [1.0], [2.0])
    assert y[1] == 2.0
    assert 0 < y[0] < 2.0 and 0 < y[2] < 2.0 and y[3] < y[2]


def test_fit_flares():
    t, f, e = make_flare_lc()
    res = lcf.fit_flares(t, f, e, exposure_time=1.0 / 1440)
    assert len(res["flares"]) == 1
    fl = res["flares"][0]
    assert abs(fl["t_peak"] - 0.06) < 3 * fl["t_peak_err"]
    assert abs(fl["amplitude"] - 0.02) < 3 * fl["amplitude_err"]
    assert len(res["per_k_bic"]) >= 2


def test_fit_flares_batch():
    windows = [make_flare_lc(seed=s) for s in range(3)]
    out = lcf.fit_flares_batch(windows, exposure_time=1.0 / 1440)
    assert len(out) == 3
    assert all(len(r["flares"]) == 1 for r in out)


def test_fit_flares_too_few_points():
    assert lcf.fit_flares([0.0, 1.0], [0.0, 0.0], [1.0, 1.0]) is None

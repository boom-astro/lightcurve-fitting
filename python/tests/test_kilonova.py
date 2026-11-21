import math

import pytest

import lightcurve_fitting as lcf

PRIOR_MEAN = [41.629614091444985, 0.2808372156230851, -1.8793073325847935, 3.770662925560441, 0.38540599767451444]


def kn_bands(params, t_exp=-0.7, dm=36.0, nights=5):
    times, mags, errs, bands = [], [], [], []
    for k, b in enumerate(["ztfg", "ztfr"]):
        t = [n + 0.05 * k for n in range(nights)]
        m = lcf.kilonova_mags(params, [ti - t_exp for ti in t], b)
        times += t
        mags += [mi + dm for mi in m]
        errs += [0.05] * nights
        bands += [b] * nights
    return lcf.build_mag_bands(times, mags, errs, bands)


def test_kilonova_mags_matches_prototype():
    m = lcf.kilonova_mags(PRIOR_MEAN, [1.0, 2.0], "ztfr")
    assert m[0] == pytest.approx(-15.40666, abs=1e-3)
    assert m[1] == pytest.approx(-14.99759, abs=1e-3)


def test_kilonova_mags_rejects_unknown_band():
    with pytest.raises(ValueError):
        lcf.kilonova_mags(PRIOR_MEAN, [1.0], "u")


def test_fit_kilonova_on_kilonova():
    res = lcf.fit_kilonova(kn_bands(PRIOR_MEAN))
    assert set(res) >= {"kn_prior", "free", "prior_cost", "bands_used", "temperature_1d"}
    assert res["free"]["t_exp"] == pytest.approx(-0.7, abs=0.05)
    assert res["prior_cost"] < 1.0
    assert res["param_names"] == ["log_l1", "a1", "a2", "log_t1", "beta"]


def test_fit_kilonova_on_hot_source():
    hot = [41.0, -0.3, 0.0, math.log10(12000.0), 0.0]
    res = lcf.fit_kilonova(kn_bands(hot, t_exp=-1.0))
    assert res["prior_cost"] > 20.0


def test_fit_kilonova_too_few_points():
    bands = lcf.build_mag_bands([0.0, 1.0], [19.0, 19.5], [0.05, 0.05], ["ztfg", "ztfg"])
    assert lcf.fit_kilonova(bands) is None


def test_fit_kilonova_batch():
    sources = [kn_bands(PRIOR_MEAN), kn_bands([41.0, -0.3, 0.0, 4.08, 0.0], t_exp=-1.0)]
    results = lcf.fit_kilonova_batch(sources)
    assert len(results) == 2
    assert results[0]["prior_cost"] < results[1]["prior_cost"]

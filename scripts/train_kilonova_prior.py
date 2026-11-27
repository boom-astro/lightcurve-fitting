#!/usr/bin/env python3
"""Train the kilonova blackbody model's per-band offsets and mixture prior.

The model (src/kilonova.rs) is a blackbody photosphere with
    log10 L = log_l1 + a1 x + a2 x^2,  log10 T = log_t1 - beta x,  x = log10(t / 1 d),
integrated through each bandpass, plus per-band offsets c0 + c1 x. This script
learns the offsets and a Gaussian-mixture prior on (log_l1, a1, a2, log_t1, beta)
from model grids and writes them, with the bandpass tables, to a JSON file that
scripts/generate_kilonova_tables.py turns into src/kilonova_tables.rs.

Each grid is a directory of light-curve files with a header "# t[days] u g r i z y J H K"
and absolute AB magnitudes in Bulla's filter set (SDSS ugriz, Swope Y, J, H, CSP K).
Every family gets equal weight in the offsets and the prior.

    python scripts/train_kilonova_prior.py --possis DIR/BNS DIR/NSBH --kasen KASEN_DIR \
        --kasen-pairs 600 --out kilonova_training.json

Needs numpy, scipy and sncosmo.
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

GRID_FILTERS = {"u": "sdssu", "g": "sdssg", "r": "sdssr", "i": "sdssi", "z": "sdssz",
                "y": "swope2::y", "J": "swope2::j", "H": "swope2::h", "K": "cspk"}
SURVEY_FILTERS = ["ztfg", "ztfr", "ztfi", "lsstg", "lsstr", "lssti", "lsstz", "lssty",
                  "ps1::g", "ps1::r", "ps1::i", "ps1::z", "ps1::y",
                  "sdssg", "sdssr", "sdssi", "sdssz", "2massj", "2massh", "2massks"]
OFFSET_BANDS = ["g", "r", "i", "z", "y", "J", "H", "K"]
PRIOR_BANDS = ["g", "r", "i"]
LOGT = np.round(np.arange(2.9, 4.8 + 1e-9, 0.01), 3)
H, C, KB, SIGMA_SB, PC10 = 6.62607015e-27, 2.99792458e10, 1.380649e-16, 5.670374e-5, 3.0857e19


def unit_lum_mag(bandpass):
    """AB mag on LOGT of a blackbody with L = 1 erg/s at 10 pc, photon-weighted through `bandpass`."""
    import sncosmo
    bp = sncosmo.get_bandpass(bandpass)
    wave = np.linspace(bp.minwave(), bp.maxwave(), 2000)
    trans = bp(wave)
    nu = C / (wave * 1e-8)
    den = np.trapezoid(3631e-23 * trans / wave, wave)
    out = []
    for lt in LOGT:
        T = 10.0**lt
        bnu = 2 * H * nu**3 / C**2 / np.expm1(np.clip(H * nu / (KB * T), 1e-8, 700))
        fnu = bnu / (4 * SIGMA_SB * T**4 * PC10**2)
        out.append(-2.5 * np.log10(np.trapezoid(fnu * trans / wave, wave) / den))
    return np.array(out)


def load_grid(root):
    out = []
    for fn in sorted(glob.glob(os.path.join(root, "*.dat"))):
        with open(fn) as f:
            cols = f.readline().lstrip("#").split()[1:]
        arr = np.loadtxt(fn)
        out.append(dict(name=os.path.basename(fn), t=arr[:, 0], mags={b: arr[:, i + 1] for i, b in enumerate(cols)}))
    return out


def two_component(singles, n, seed, blue_max=-4.0, red_min=-3.0):
    """Sum a low- and a high-lanthanide Kasen model in flux (Xlan parsed from the file name)."""
    def xlan(m):
        return -float(m["name"].split("Xlan1e-")[1].replace(".dat", ""))
    blue = [m for m in singles if xlan(m) <= blue_max]
    red = [m for m in singles if xlan(m) >= red_min]
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        b, r = blue[rng.integers(len(blue))], red[rng.integers(len(red))]
        t = b["t"][b["t"] <= min(b["t"][-1], r["t"][-1]) + 1e-9]
        mags = {}
        for band in b["mags"]:
            fb = np.interp(t, b["t"], 10 ** (-0.4 * b["mags"][band]), right=np.nan)
            fr = np.interp(t, r["t"], 10 ** (-0.4 * r["mags"][band]), right=np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                mags[band] = -2.5 * np.log10(np.nan_to_num(fb) + np.nan_to_num(fr))
        out.append(dict(name=f"pair{k}", t=t, mags=mags))
    return out


class Model:
    def __init__(self, tables, offsets):
        self.tables, self.offsets = tables, offsets

    def predict(self, theta, t, bands):
        x = np.log10(np.clip(t, 1e-3, None))
        logl = theta[0] + theta[1] * x + theta[2] * x**2
        logt = theta[3] - theta[4] * x
        m = np.empty(len(t))
        for b in np.unique(bands):
            k = bands == b
            c0, c1 = self.offsets.get(b, (0.0, 0.0))
            m[k] = np.interp(logt[k], LOGT, self.tables[GRID_FILTERS[b]]) + c0 + c1 * x[k]
        return m - 2.5 * logl


def fit_lc(model, t, m, bands, err=0.05):
    lo, hi = np.array([-30, -5, -8, 3.0, -1.5]), np.array([60, 5, 3, 4.7, 2.5])

    def resid(p):
        return (model.predict(p, t, bands) - m) / err

    best = None
    for s in ([41.6, 0.28, -1.88, 3.77, 0.385], [41.5, 0.0, -1.5, 3.75, 0.4]):
        s = np.array(s, float)
        s[0] += np.mean(model.predict(s, t, bands) - m) / 2.5
        sol = least_squares(resid, np.clip(s, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), x_scale="jac", max_nfev=600)
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x, float(np.sqrt(np.mean((model.predict(best.x, t, bands) - m) ** 2)))


def epoch_residuals(args):
    lc, tables = args
    U = np.array([tables[GRID_FILTERS[b]] for b in OFFSET_BANDS])
    xs, res = [], []
    for j in np.where((lc["t"] >= 0.5) & (lc["t"] <= 10))[0]:
        mags = np.array([lc["mags"][b][j] for b in OFFSET_BANDS])
        if not np.all(np.isfinite(mags)) or np.any(mags > 0):
            continue
        off = np.mean(mags[:, None] - U, axis=0)
        k = np.argmin(np.sum((mags[:, None] - U - off) ** 2, axis=0))
        xs.append(np.log10(lc["t"][j]))
        res.append(mags - U[:, k] - off[k])
    return np.array(xs), np.array(res).reshape(-1, len(OFFSET_BANDS))


def lc_arrays(lc, bands, tmin, tmax):
    t, m, b = [], [], []
    keep = (lc["t"] >= tmin) & (lc["t"] <= tmax)
    for band in bands:
        y = lc["mags"][band][keep]
        ok = np.isfinite(y) & (y < 0)
        if ok.sum() < 3:
            return None
        t.append(lc["t"][keep][ok]); m.append(y[ok]); b += [band] * ok.sum()
    return np.concatenate(t), np.concatenate(m), np.array(b)


def prior_fit(args):
    lc, tables, offsets = args
    arr = lc_arrays(lc, PRIOR_BANDS, 0.3, 6.0)
    return None if arr is None else fit_lc(Model(tables, offsets), *arr)[0]


def fidelity(args):
    lc, tables, offsets = args
    arr = lc_arrays(lc, PRIOR_BANDS, 0.5, 5.0)
    return None if arr is None else fit_lc(Model(tables, offsets), *arr)[1]


def robust_line(x, r, w):
    A = np.c_[np.ones_like(x), x]
    ww = w.copy()
    for _ in range(20):
        c = np.linalg.lstsq(A * np.sqrt(ww)[:, None], r * np.sqrt(ww), rcond=None)[0]
        ww = w / np.maximum(np.abs(r - A @ c), 0.02)
    res = r - A @ c
    return c, 1.4826 * np.median(np.abs(res - np.median(res)))


def logpdf(X, mu, cov):
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * np.sum(z**2, 0) - np.sum(np.log(np.diag(L))) - 0.5 * len(mu) * np.log(2 * np.pi)


def fit_gmm(X, w, K, seed, n_iter=500, reg=1e-4, floor=0.0):
    """Weighted EM for a full-covariance Gaussian mixture."""
    rng = np.random.default_rng(seed)
    w = w / w.sum()
    mu = X[rng.choice(len(X), K, replace=False, p=w)]
    cov = np.array([np.cov(X.T, aweights=w)] * K)
    pi = np.full(K, 1.0 / K)
    prev = -np.inf
    for _ in range(n_iter):
        logp = np.stack([np.log(pi[k]) + logpdf(X, mu[k], cov[k]) for k in range(K)], 1)
        lse = np.logaddexp.reduce(logp, 1)
        ll = float(np.sum(w * lse))
        resp = np.exp(logp - lse[:, None]) * w[:, None]
        nk = resp.sum(0)
        pi, mu = nk / nk.sum(), (resp.T @ X) / nk[:, None]
        for k in range(K):
            dx = X - mu[k]
            cov[k] = (resp[:, k, None] * dx).T @ dx / nk[k] + reg * np.eye(X.shape[1])
        if ll - prev < 1e-9:
            break
        prev = ll
    cov = cov + floor * np.eye(X.shape[1])
    return dict(weights=pi.tolist(), means=mu.tolist(), covs=cov.tolist(), loglike=ll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--possis", nargs="*", default=[], help="POSSIS light-curve directories (one family)")
    ap.add_argument("--kasen", default=None, help="Kasen et al. (2017) light-curve directory")
    ap.add_argument("--kasen-pairs", type=int, default=600, help="blue+red Kasen combinations (0 to skip)")
    ap.add_argument("--no-kasen-singles", action="store_true", help="train on the blue+red combinations only")
    ap.add_argument("--extra", nargs="*", default=[], help="further families, NAME=DIR")
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--cov-floor", type=float, default=0.0,
                    help="added to every mixture component's variances; smooths over grid sampling")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="kilonova_training.json")
    args = ap.parse_args()

    families = {}
    if args.possis:
        families["POSSIS"] = sum((load_grid(d) for d in args.possis), [])
    if args.kasen:
        kasen = load_grid(args.kasen)
        if not args.no_kasen_singles:
            families["Kasen"] = kasen
        if args.kasen_pairs:
            families["Kasen2"] = two_component(kasen, args.kasen_pairs, args.seed + 1)
    for spec in args.extra:
        name, d = spec.split("=", 1)
        families[name] = load_grid(d)
    rng = np.random.default_rng(args.seed)
    split = {}
    for name, lcs in families.items():
        perm = rng.permutation(len(lcs))
        ntr = int(args.train_frac * len(lcs))
        split[name] = ([lcs[i] for i in perm[:ntr]], [lcs[i] for i in perm[ntr:]])
        print(f"{name}: {ntr} train, {len(lcs) - ntr} test")

    names = sorted(set(GRID_FILTERS.values()) | set(SURVEY_FILTERS))
    with Pool() as p:
        tables = dict(zip(names, p.map(unit_lum_mag, names)))

    # offsets: median departure from a per-epoch blackbody over grizyJHK, linear in log t
    xs, rs, ws = [], [], []
    with Pool() as p:
        for name, (train, _) in split.items():
            outs = p.map(epoch_residuals, [(lc, tables) for lc in train])
            n = sum(len(o[0]) for o in outs)
            for x, r in outs:
                xs.append(x); rs.append(r); ws.append(np.full(len(x), 1.0 / n))
    x, r, w = np.concatenate(xs), np.concatenate(rs), np.concatenate(ws)
    offsets, scatter = {}, {}
    for j, b in enumerate(OFFSET_BANDS):
        c, s = robust_line(x, r[:, j], w)
        offsets[b], scatter[b] = (float(c[0]), float(c[1])), float(s)
        print(f"offset {b}: c0 {c[0]:+.4f}  c1 {c[1]:+.4f}  scatter {s:.3f}")

    # prior: fits to gri over 0.3-6 d with the merger time known
    thetas, wts, fam_labels = [], [], []
    with Pool() as p:
        for name, (train, test) in split.items():
            th = [o for o in p.map(prior_fit, [(lc, tables, offsets) for lc in train]) if o is not None]
            rms = [o for o in p.map(fidelity, [(lc, tables, offsets) for lc in test[:300]]) if o is not None]
            print(f"{name}: {len(th)} prior fits; held-out gri RMS median {np.median(rms):.3f}, p90 {np.percentile(rms, 90):.3f}")
            thetas += th; wts += [1.0 / len(th)] * len(th); fam_labels += [name] * len(th)
    X, W, fam_labels = np.array(thetas), np.array(wts), np.array(fam_labels)
    keep = np.all((X > np.percentile(X, 0.5, 0)) & (X < np.percentile(X, 99.5, 0)), axis=1)
    X, W, fam_labels = X[keep], W[keep], fam_labels[keep]
    gmms = {}
    for K in range(1, args.kmax + 1):
        g = max((fit_gmm(X, W, K, s, floor=args.cov_floor) for s in range(6)), key=lambda g: g["loglike"])
        d = X.shape[1]
        g["bic"] = -2 * g["loglike"] * len(X) + (K - 1 + K * d + K * d * (d + 1) / 2) * np.log(len(X))
        gmms[K] = g
        print(f"mixture K={K}: loglike/n {g['loglike']:.3f}, BIC {g['bic']:.1f}")

    json.dump(dict(logt=LOGT.tolist(), tables={k: v.tolist() for k, v in tables.items()},
                   grid_filters=GRID_FILTERS, offsets=offsets, offset_scatter=scatter,
                   mixtures={str(k): v for k, v in gmms.items()},
                   families={k: len(v) for k, v in families.items()},
                   test_models={k: [lc["name"] for lc in v[1]] for k, v in split.items()},
                   prior_fits=dict(theta=X.tolist(), weight=W.tolist(), family=fam_labels.tolist())),
              open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

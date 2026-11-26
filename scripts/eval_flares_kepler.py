"""Validate fit_flares against the Davenport et al. (2014) GJ 1243 Kepler catalog.

Uses the 11-month short-cadence light curve and 6107-flare catalog from
https://github.com/jradavenport/GJ1243-Flares (downloaded on first run).

Checks:
  1. Classical flares (the cuts used to build the template): peak-time offsets,
     ED agreement, component count, reduced chi2.
  2. Complex flares: number of components chosen by BIC vs the catalog's
     template-component count.
  3. Throughput over all 6107 events.
  4. Cadence degradation: bin the 1-min data to 2/10/30 min, refit with and
     without exposure averaging, and compare to the 1-min fit of the same window.

Usage:
    python scripts/eval_flares_kepler.py --data-dir DATA --out-dir OUT
"""

import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightcurve_fitting as lcf

URL = "https://raw.githubusercontent.com/jradavenport/GJ1243-Flares/master/data/"
FILES = ["gj1243_master_flares.tbl", "gj1243_master_slc.dat"]
CADENCE = 58.8487 / 86400.0  # Kepler short cadence, days
MIN = 1.0 / 1440.0


def load(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if not (data_dir / name).exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(URL + name, data_dir / name)
    lc = np.loadtxt(data_dir / FILES[1])
    cat = np.loadtxt(data_dir / FILES[0])
    return lc, cat


def regularise_times(t):
    """Snap float32-quantised timestamps back onto the cadence grid.

    Cadence indices come from local differences (quantisation is ~5-10 s, far
    below half a cadence); a cubic in index per segment absorbs the slow
    barycentric drift of the cadence.
    """
    out = t.copy()
    breaks = np.concatenate([[0], np.where(np.diff(t) > 0.1)[0] + 1, [len(t)]])
    for a, b in zip(breaks[:-1], breaks[1:]):
        steps = np.maximum(np.round(np.diff(t[a:b]) / CADENCE), 1)
        idx = np.concatenate([[0], np.cumsum(steps)])
        coef = np.polyfit(idx, t[a:b] - t[a], 3)
        out[a:b] = t[a] + np.polyval(coef, idx)
    return out


def window(t, flux, err, lo, hi):
    m = (t >= lo) & (t <= hi)
    f = flux[m]
    med = np.median(f)
    return t[m], f / med - 1.0, err[m] / med


def bin_window(t, f, e, m):
    """Average groups of m consecutive cadences, keeping only complete groups."""
    idx = np.round((t - t[0]) / CADENCE).astype(int) // m
    tb, fb, eb = [], [], []
    for g in np.unique(idx):
        s = idx == g
        if s.sum() == m:
            tb.append(t[s].mean())
            fb.append(f[s].mean())
            eb.append(np.sqrt(np.sum(e[s] ** 2)) / m)
    return np.array(tb), np.array(fb), np.array(eb)


def nearest(flares, tp):
    if not flares:
        return None
    return min(flares, key=lambda c: abs(c["t_peak"] - tp))


def event_ed(flares, lo, hi):
    return sum(c["equivalent_duration"] for c in flares if lo <= c["t_peak"] <= hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("gj1243_data"))
    ap.add_argument("--out-dir", type=Path, default=Path("gj1243_eval"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lc, cat = load(args.data_dir)
    t = regularise_times(lc[:, 0])
    flux, err = lc[:, 3], lc[:, 2]
    (t_start, t_stop, t_peak, ed_cat, cplx, ppl1, ncomp) = (
        cat[:, 2], cat[:, 3], cat[:, 4], cat[:, 8], cat[:, 11], cat[:, 12], cat[:, 14])
    dur = t_stop - t_start
    print(f"{len(t)} cadences, {len(cat)} catalogued flares")

    # Windows padded around the catalogue start/stop.
    windows, bounds = [], []
    for a, b, d in zip(t_start, t_stop, dur):
        lo, hi = a - max(0.5 * d, 10 * MIN), b + max(1.0 * d, 15 * MIN)
        windows.append(window(t, flux, err, lo, hi))
        bounds.append((lo, hi))

    # --- Throughput over every catalogued event --------------------------
    opts = dict(exposure_time=CADENCE, max_flares=5)
    t0 = time.perf_counter()
    results = lcf.fit_flares_batch([tuple(map(list, w)) for w in windows], **opts)
    wall = time.perf_counter() - t0
    t0 = time.perf_counter()
    for w in windows[:300]:
        lcf.fit_flares(*w, **opts)
    serial = (time.perf_counter() - t0) / 300
    npts = np.median([len(w[0]) for w in windows])
    print(f"\n[throughput] {len(windows)} windows (median {npts:.0f} pts) in {wall:.2f} s "
          f"batch; {1e3 * serial:.2f} ms/window serial")

    k_fit = np.array([len(r["flares"]) if r else 0 for r in results])
    # Components peaking inside the catalogued event (neighbours excluded).
    k_in = np.array([
        sum(a - 2 * MIN <= c["t_peak"] <= b for c in r["flares"]) if r else 0
        for r, a, b in zip(results, t_start, t_stop)])
    match = [nearest(r["flares"], tp) if r else None for r, tp in zip(results, t_peak)]
    dtp = np.array([(m["t_peak"] - tp) if m else np.nan for m, tp in zip(match, t_peak)])
    ed_fit = np.array([
        sum(c["equivalent_duration"] for c in r["flares"] if a - 2 * MIN <= c["t_peak"] <= b) * 86400
        if r else np.nan for r, a, b in zip(results, t_start, t_stop)])
    rchi2 = np.array([r["reduced_chi2"] if r else np.nan for r in results])

    # --- Classical template sample --------------------------------------
    classical = ((dur >= 20 * MIN) & (dur <= 75 * MIN) & (cplx < 1.5) & (ppl1 >= 4)
                 & (ed_cat > 0) & (ed_cat <= 100))
    c = classical
    detected = k_in[c] >= 1
    print(f"\n[classical] {c.sum()} flares (template-building cuts)")
    print(f"  detected (>=1 component in event): {detected.mean():.3f}")
    print(f"  components in event: " + ", ".join(
        f"K={k}: {np.mean(k_in[c] == k):.3f}" for k in range(0, 4)))
    off = dtp[c][detected] * 86400
    print(f"  t_peak - catalogue: median {np.median(off):+.1f} s, "
          f"68% within {np.percentile(np.abs(off), 68):.1f} s "
          f"(cadence {CADENCE * 86400:.1f} s)")
    # The catalogue's fractional flux is a fixed multiple of detrended/median - 1;
    # measure it from the peak so EDs can be compared on the same scale.
    peak_ratio, ed_trap = [], []
    for i in np.where(c)[0]:
        tw, fw, ew = windows[i]
        fl = [x for x in results[i]["flares"] if t_start[i] - 2 * MIN <= x["t_peak"] <= t_stop[i]] if results[i] else []
        m = (tw >= t_start[i]) & (tw <= t_stop[i])
        if not fl or m.sum() < 2:
            peak_ratio.append(np.nan)
            ed_trap.append(np.nan)
            continue
        model = lcf.flare_model(list(tw[m]), [x["t_peak"] for x in fl], [x["t_half"] for x in fl],
                                [x["amplitude"] for x in fl], CADENCE)
        peak_ratio.append(max(model) / cat[i, 7])
        ed_trap.append(np.trapezoid(fw[m] - results[i]["baseline"][0], tw[m]) * 86400)
    norm = 1.0 / np.nanmedian(peak_ratio)
    print(f"  catalogue fractional flux = {norm:.3f} x (detrended/median - 1) "
          f"(16-84%: {1 / np.nanpercentile(peak_ratio, 84):.3f}-{1 / np.nanpercentile(peak_ratio, 16):.3f})")
    q = lambda x: np.nanpercentile(x, [50, 16, 84])
    r1 = q(norm * ed_fit[c] / ed_cat[c])
    r2 = q(ed_fit[c] / np.array(ed_trap))
    print(f"  ED_fit / ED_cat (rescaled): {r1[0]:.3f} ({r1[1]:.3f}-{r1[2]:.3f})")
    print(f"  ED_fit / trapezoid ED over t_start..t_stop: {r2[0]:.3f} ({r2[1]:.3f}-{r2[2]:.3f})")
    print(f"  reduced chi2: median {np.nanmedian(rchi2[c]):.2f}, "
          f"90th pct {np.nanpercentile(rchi2[c], 90):.2f}")
    print("  catalogue components (rows) vs fitted components in event (cols K=0..4):")
    for nc in (1, 2, 3):
        s_ = c & (ncomp == nc)
        print(f"    cat={nc} " + " ".join(f"{np.mean(k_in[s_] == k):6.3f}" for k in range(5))
              + f"   (n={s_.sum()})")

    # --- Complex events: component counts --------------------------------
    print("\n[components] rows = catalogue template components, cols = fitted components in event")
    print("        " + " ".join(f"K={k:<4d}" for k in range(0, 6)))
    for nc in range(1, 5):
        s = ncomp == nc
        row = [np.mean(k_in[s] == k) for k in range(0, 6)]
        print(f"  cat={nc} " + " ".join(f"{v:6.3f}" for v in row) + f"   (n={s.sum()})")
    s = cplx > 1.5
    print(f"  human-flagged complex (n={s.sum()}): mean fitted K = {k_in[s].mean():.2f}; "
          f"classical: {k_in[cplx < 1.5].mean():.2f}")

    # --- Cadence degradation ---------------------------------------------
    print("\n[cadence] classical flares with ED_cat > 5 s; window t_peak-0.15 d .. +0.3 d")
    big = np.where(classical & (ed_cat > 5))[0]
    rows = []
    for i in big:
        lo, hi = t_peak[i] - 0.15, t_peak[i] + 0.3
        ws = window(t, flux, err, lo, hi)
        ref = lcf.fit_flares(*ws, exposure_time=CADENCE, max_flares=6)
        ev = [x for x in ref["flares"] if t_start[i] - 2 * MIN <= x["t_peak"] <= t_stop[i]] if ref else []
        if not ev:
            continue
        row = {"ed_ref": event_ed(ev, -np.inf, np.inf),
               "th_ref": max(ev, key=lambda x: x["equivalent_duration"])["t_half"]}
        for m in (2, 10, 30):
            tb, fb, eb = bin_window(*ws, m)
            pad = m * CADENCE
            for tag, exp in (("exp", pad), ("pt", None)):
                r = lcf.fit_flares(tb, fb, eb, exposure_time=exp, max_flares=6)
                evb = [x for x in r["flares"] if t_start[i] - pad <= x["t_peak"] <= t_stop[i] + pad] if r else []
                row[f"ed_{tag}{m}"] = event_ed(evb, -np.inf, np.inf) if evb else np.nan
                row[f"th_{tag}{m}"] = (max(evb, key=lambda x: x["equivalent_duration"])["t_half"]
                                       if evb else np.nan)
        rows.append(row)
    print(f"  {len(rows)} flares with a 1-min fit; median 1-min FWHM of the main component "
          f"{np.median([r['th_ref'] for r in rows]) / MIN:.1f} min")
    print("  bin   model    detected  ED/ED_1min (med, 16-84%)   FWHM/FWHM_1min (med, 16-84%)")
    summary = {}
    for m in (2, 10, 30):
        for tag, label in (("exp", "averaged"), ("pt", "point   ")):
            ed = np.array([r[f"ed_{tag}{m}"] / r["ed_ref"] for r in rows])
            th = np.array([r[f"th_{tag}{m}"] / r["th_ref"] for r in rows])
            e, h = q(ed), q(th)
            summary[(m, tag)] = (ed, th)
            print(f"  {m:2d}m  {label}   {np.isfinite(ed).mean():.3f}     {e[0]:.3f} ({e[1]:.3f}-{e[2]:.3f})"
                  f"          {h[0]:.3f} ({h[1]:.3f}-{h[2]:.3f})")

    # --- Figures ---------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    picks = [np.where(classical & (ed_cat > 10))[0][3],
             np.where(classical & (ed_cat > 3) & (ed_cat < 6))[0][5],
             np.where((ncomp >= 3) & (ed_cat > 20))[0][2]]
    for ax, i in zip(axes[0], picks):
        tw, fw, ew = windows[i]
        r = results[i]
        tt = np.linspace(tw[0], tw[-1], 2000)
        ax.errorbar((tw - t_peak[i]) / MIN, fw, ew, fmt=".", ms=3, color="0.5", lw=0.5)
        if r:
            base = r["baseline"][0]
            for comp in r["flares"]:
                y = lcf.flare_model(list(tt), [comp["t_peak"]], [comp["t_half"]],
                                    [comp["amplitude"]], CADENCE)
                ax.plot((tt - t_peak[i]) / MIN, np.array(y) + base, lw=0.8, ls="--")
            tot = lcf.flare_model(list(tt), [c["t_peak"] for c in r["flares"]],
                                  [c["t_half"] for c in r["flares"]],
                                  [c["amplitude"] for c in r["flares"]], CADENCE)
            ax.plot((tt - t_peak[i]) / MIN, np.array(tot) + base, "k", lw=1.2)
        ax.set_title(f"t={t_peak[i]:.3f}  catalogue K={int(ncomp[i])}  fit K={k_fit[i]}")
        ax.set_xlabel("minutes from catalogue peak")
        ax.set_ylabel("relative flux")

    ax = axes[1, 0]
    ax.loglog(ed_cat[c], norm * ed_fit[c], ".", ms=2, alpha=0.5)
    lim = [0.05, 200]
    ax.plot(lim, lim, "k:", lw=0.8)
    ax.set_xlabel("catalogue ED (s)")
    ax.set_ylabel(f"fitted ED x {norm:.2f} (s)")
    ax.set_title("classical flares")

    ax = axes[1, 1]
    mat = np.array([[np.mean(k_in[ncomp == nc] == k) for k in range(0, 6)] for nc in range(1, 5)])
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, origin="lower", extent=[-0.5, 5.5, 0.5, 4.5])
    for a in range(4):
        for b in range(6):
            ax.text(b, a + 1, f"{mat[a, b]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("fitted components in event")
    ax.set_ylabel("catalogue components")

    ax = axes[1, 2]
    for m, col in zip((2, 10, 30), ("C0", "C1", "C2")):
        for tag, ls in (("exp", "-"), ("pt", ":")):
            ed = summary[(m, tag)][0]
            ed = ed[np.isfinite(ed) & (ed > 0)]
            ax.hist(np.log10(ed), bins=40, range=(-1, 1), histtype="step", color=col, ls=ls,
                    label=f"{m} min {'averaged' if tag == 'exp' else 'point'}")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("log10 ED / ED(1 min)")
    ax.set_title("binned cadence (solid: exposure-averaged)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out_dir / "gj1243_flares.png", dpi=130)
    print(f"\nwrote {args.out_dir / 'gj1243_flares.png'}")


if __name__ == "__main__":
    main()

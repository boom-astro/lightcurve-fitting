#!/usr/bin/env python3
"""Generate documentation plots for nonparametric and parametric GP fits.

Reads real source fixtures and produces per-class images in docs/img/.
Run from the repository root:

    python docs/generate_plots.py
"""

import json
import os
import sys

import numpy as np

import lightcurve_fitting as lcf

# Optional: skip if matplotlib not available
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_PATH = os.path.join(REPO_ROOT, "tests", "fixtures", "real_sources.json")
IMG_DIR = os.path.join(REPO_ROOT, "docs", "img")
os.makedirs(IMG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
BAND_COLORS = {
    "ztfg": "#4daf4a",
    "ztfr": "#e41a1c",
    "ztfi": "#984ea3",
    "g": "#4daf4a",
    "r": "#e41a1c",
    "i": "#984ea3",
}
BAND_ORDER = ["ztfr", "ztfg", "ztfi", "r", "g", "i"]

SNR_THRESHOLD = 3.0
MAX_GP_POINTS = 150  # subsample before DenseGP to avoid fitting noise
MIN_DET_PER_BAND = 5  # skip bands with fewer detections


def detection_mask(mags, errs, zp=23.9, snr_thresh=SNR_THRESHOLD):
    """Return boolean mask: True = detection, False = upper limit."""
    flux = 10 ** (-0.4 * (np.array(mags) - zp))
    factor = 2.5 / np.log(10)
    flux_err = np.array(errs) / factor * flux
    return (flux > 0) & (flux_err > 0) & (flux / flux_err >= snr_thresh)


def subsample(t, v, e, max_points):
    """Uniform-stride subsample, matching the Rust subsample_data logic."""
    if len(t) <= max_points:
        return t, v, e
    step = len(t) / max_points
    indices = [min(int((i + 0.5) * step), len(t) - 1) for i in range(max_points)]
    return t[indices], v[indices], e[indices]


def sorted_bands(band_dict):
    """Return band names sorted by BAND_ORDER, then alphabetically."""
    order = {b: i for i, b in enumerate(BAND_ORDER)}
    return sorted(band_dict.keys(), key=lambda b: (order.get(b, 99), b))


def find_display_window(raw_bands, band_names, is_persistent=False):
    """Find the time window to display.

    For transients: find temporal clusters separated by gaps > 60 days,
    then select the cluster containing the brightest observation.
    For persistent variables (AGN, CV): use the full data range.
    """
    # Collect all detection times and values across bands
    all_det_t = []
    all_det_v = []
    for b in band_names:
        bd = raw_bands[b]
        t = np.array(bd["times"])
        v = np.array(bd["values"])
        e = np.array(bd["errors"])
        mask = detection_mask(v, e)
        if mask.sum() > 0:
            all_det_t.append(t[mask])
            all_det_v.append(v[mask])

    if not all_det_t:
        return None, None

    all_det_t = np.concatenate(all_det_t)
    all_det_v = np.concatenate(all_det_v)

    if is_persistent:
        return all_det_t.min(), all_det_t.max()

    # Sort by time
    order = np.argsort(all_det_t)
    t_sorted = all_det_t[order]
    v_sorted = all_det_v[order]

    # Find gaps > 60 days to split into temporal clusters
    dt = np.diff(t_sorted)
    gap_idx = np.where(dt > 60)[0]
    boundaries = np.concatenate([[0], gap_idx + 1, [len(t_sorted)]])

    # Find which cluster contains the brightest point (min magnitude)
    peak_idx = np.argmin(v_sorted)
    cluster_start, cluster_end = t_sorted[0], t_sorted[-1]
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if s <= peak_idx < e:
            cluster_start = t_sorted[s]
            cluster_end = t_sorted[e - 1]
            break

    duration = max(cluster_end - cluster_start, 20.0)
    pad = max(0.15 * duration, 5.0)
    return cluster_start - pad, cluster_end + pad


# ---------------------------------------------------------------------------
# Nonparametric plot
# ---------------------------------------------------------------------------
def plot_nonparametric(source, outpath, is_persistent=False):
    """Generate a multi-panel nonparametric GP plot for one source."""
    obj_id = source["obj_id"]
    label_name = source["label_name"]
    raw_bands = source["bands"]
    # Only keep bands with enough detections for a meaningful GP fit
    band_names = [
        b for b in sorted_bands(raw_bands)
        if detection_mask(
            np.array(raw_bands[b]["values"]),
            np.array(raw_bands[b]["errors"]),
        ).sum() >= MIN_DET_PER_BAND
    ]
    if not band_names:
        print(f"  Skipping {obj_id}: no bands with >= {MIN_DET_PER_BAND} detections")
        return

    # Fit nonparametric
    bands_map = lcf.BandDataMap.from_dict(
        {b: (raw_bands[b]["times"], raw_bands[b]["values"], raw_bands[b]["errors"]) for b in band_names}
    )
    np_results = lcf.fit_nonparametric(bands_map)
    result_by_band = {r["band"]: r for r in np_results}

    # Find display window from the data
    win_lo, win_hi = find_display_window(raw_bands, band_names, is_persistent)
    if win_lo is None:
        print(f"  Skipping {obj_id}: could not determine display window")
        return

    n_bands = len(band_names)
    fig, axes = plt.subplots(1, n_bands, figsize=(3.5 * n_bands, 3.0), squeeze=False)
    fig.suptitle(f"{obj_id} ({label_name})", fontsize=14, fontweight="bold")

    for col, band_name in enumerate(band_names):
        ax = axes[0, col]
        bd = raw_bands[band_name]
        t = np.array(bd["times"])
        v = np.array(bd["values"])
        e = np.array(bd["errors"])
        color = BAND_COLORS.get(band_name, "#377eb8")
        short_name = band_name.replace("ztf", "")

        mask = detection_mask(v, e)
        n_det = mask.sum()
        n_ul = (~mask).sum()

        r = result_by_band.get(band_name)

        # Plot data in the display window
        in_win = (t >= win_lo) & (t <= win_hi)
        det_in = mask & in_win
        ul_in = (~mask) & in_win

        ax.errorbar(
            t[det_in], v[det_in], yerr=e[det_in],
            fmt="o", color=color, markersize=3, alpha=0.6,
            label=f"{short_name} data ({det_in.sum()})", zorder=3,
        )
        if ul_in.sum() > 0:
            ax.scatter(
                t[ul_in], v[ul_in],
                marker="v", color=color, s=30, alpha=0.5,
                label=f"{short_name} UL ({ul_in.sum()})", zorder=4,
            )

        # GP prediction curve — fit on detections within display window
        if r is not None and n_det >= 5:
            sort_idx = np.argsort(t[mask])
            t_det = t[mask][sort_idx]
            v_det = v[mask][sort_idx]
            e_det = e[mask][sort_idx]

            # Restrict GP training to data in the display window
            win_mask = (t_det >= win_lo) & (t_det <= win_hi)
            if win_mask.sum() >= 5:
                t_det = t_det[win_mask]
                v_det = v_det[win_mask]
                e_det = e_det[win_mask]

            # Subsample for DenseGP
            t_gp, v_gp, e_gp = subsample(t_det, v_det, e_det, MAX_GP_POINTS)

            dt = np.diff(t_gp)
            dt = dt[(dt > 0) & np.isfinite(dt)]
            median_dt = np.median(dt) if len(dt) > 0 else 1.0
            min_lengthscale = max(median_dt * 2.0, 0.1)

            data_dur = t_gp.max() - t_gp.min()
            if data_dur > 0:
                # Predict over the display window
                query_lo = max(win_lo, t_gp.min())
                query_hi = min(win_hi, t_gp.max())
                query_t = np.linspace(query_lo, query_hi, 300).tolist()

                amps = [0.05, 0.1, 0.3, 0.5]
                ls_cands = [
                    ls for f in [4.0, 6.0, 12.0, 24.0, 48.0]
                    if (ls := max(data_dur / f, 0.1)) >= min_lengthscale
                ]
                if not ls_cands:
                    ls_cands = [min_lengthscale]
                gp_result = lcf.fit_gp_predict(
                    t_gp.tolist(), v_gp.tolist(), e_gp.tolist(),
                    query_t, amps, ls_cands,
                )
                if gp_result is not None:
                    pred = np.array(gp_result[0])
                    std = np.array(gp_result[1])
                    ax.plot(query_t, pred, "-", color=color, linewidth=1.5, alpha=0.85, zorder=5)
                    ax.fill_between(
                        query_t, pred - 2 * std, pred + 2 * std,
                        color=color, alpha=0.10, zorder=1,
                    )

            # Per-band features
            peak_mag = r.get("peak_mag")
            b_t0 = r.get("t0")
            b_rise = r.get("rise_time")
            b_decay = r.get("decay_time")
            fwhm = r.get("fwhm")

            # Peak magnitude line and peak time
            if peak_mag is not None:
                ax.axhline(y=peak_mag, color=color, linestyle=":", alpha=0.4, linewidth=1)
            if b_t0 is not None and win_lo <= b_t0 <= win_hi:
                ax.axvline(x=b_t0, color="gray", linestyle="--", alpha=0.4, linewidth=1)

            # FWHM shading
            if fwhm is not None and b_t0 is not None and b_rise is not None and b_decay is not None:
                fwhm_left = b_t0 - min(b_rise, fwhm / 2)
                fwhm_right = b_t0 + min(b_decay, fwhm / 2)
                ax.axvspan(fwhm_left, fwhm_right, color=color, alpha=0.06, zorder=0)

            # Feature annotation box
            lines = []
            if peak_mag is not None:
                lines.append(f"peak={peak_mag:.1f}")
            if b_rise is not None:
                lines.append(f"rise={b_rise:.1f}d")
            if b_decay is not None:
                lines.append(f"decay={b_decay:.1f}d")
            if fwhm is not None:
                lines.append(f"FWHM={fwhm:.1f}d")
            if n_ul > 0:
                lines.append(f"UL={n_ul}")
            if lines:
                ax.text(
                    0.03, 0.03, "\n".join(lines),
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
                )

        # Crop x-axis to display window
        ax.set_xlim(win_lo, win_hi)
        ax.invert_yaxis()
        ax.set_xlabel("Time (days)")
        if col == 0:
            ax.set_ylabel("Magnitude")
        ax.set_title(f"{short_name}-band", fontsize=11)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fig.savefig(outpath, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# Parametric plot
# ---------------------------------------------------------------------------
def plot_parametric(source, outpath, is_persistent=False):
    """Generate a multi-panel parametric model plot for one source."""
    import math

    obj_id = source["obj_id"]
    label_name = source["label_name"]
    raw_bands = source["bands"]
    band_names = [
        b for b in sorted_bands(raw_bands)
        if detection_mask(
            np.array(raw_bands[b]["values"]),
            np.array(raw_bands[b]["errors"]),
        ).sum() >= MIN_DET_PER_BAND
    ]
    if not band_names:
        print(f"  Skipping {obj_id}: no bands with >= {MIN_DET_PER_BAND} detections")
        return

    # Build flux-space bands (strip ztf prefix)
    times_all, mags_all, errs_all, bands_list = [], [], [], []
    short_map = {}  # short_name -> original band name
    for b in band_names:
        short = b.replace("ztf", "")
        short_map[short] = b
        bd = raw_bands[b]
        for t, v, e in zip(bd["times"], bd["values"], bd["errors"]):
            times_all.append(t)
            mags_all.append(v)
            errs_all.append(e)
            bands_list.append(short)

    flux_bands = lcf.build_flux_bands(times_all, mags_all, errs_all, bands_list)
    param_results = lcf.fit_parametric(flux_bands, method="laplace")
    if not param_results:
        print(f"  Skipping {obj_id}: parametric fit returned no results")
        return

    result_by_band = {r["band"]: r for r in param_results}

    # Find display window
    win_lo, win_hi = find_display_window(raw_bands, band_names, is_persistent)
    if win_lo is None:
        print(f"  Skipping {obj_id}: could not determine display window")
        return

    # Only keep bands that have parametric results
    short_names = [b.replace("ztf", "") for b in band_names
                   if b.replace("ztf", "") in result_by_band]
    band_names = [b for b in band_names if b.replace("ztf", "") in result_by_band]

    n_bands = len(short_names)
    if n_bands == 0:
        print(f"  Skipping {obj_id}: no parametric results for any band")
        return
    fig, axes = plt.subplots(1, n_bands, figsize=(3.5 * n_bands, 3.0), squeeze=False)
    fig.suptitle(f"{obj_id} ({label_name}) — Parametric", fontsize=12, fontweight="bold")

    zp = 23.9
    for col, (short_name, orig_name) in enumerate(zip(short_names, band_names)):
        ax = axes[0, col]
        bd = raw_bands[orig_name]
        t = np.array(bd["times"])
        v = np.array(bd["values"])
        e = np.array(bd["errors"])
        color = BAND_COLORS.get(orig_name, BAND_COLORS.get(short_name, "#377eb8"))

        # Convert to flux
        flux = 10 ** ((zp - v) / 2.5)
        flux_err = flux * e * np.log(10) / 2.5

        mask = detection_mask(v, e)
        in_win = (t >= win_lo) & (t <= win_hi)
        det_in = mask & in_win
        ul_in = (~mask) & in_win

        ax.errorbar(
            t[det_in], flux[det_in], yerr=flux_err[det_in],
            fmt="o", color=color, markersize=3, alpha=0.6,
            label=f"{short_name} data ({det_in.sum()})", zorder=3,
        )
        if ul_in.sum() > 0:
            ax.scatter(
                t[ul_in], flux[ul_in],
                marker="v", color=color, s=30, alpha=0.5,
                label=f"{short_name} UL ({ul_in.sum()})", zorder=4,
            )

        # Model prediction curve
        r = result_by_band.get(short_name)
        if r is not None:
            model = r.get("model")
            params = r.get("svi_mu") or r.get("pso_params")
            if model and params and det_in.sum() > 0:
                t_pred = np.linspace(win_lo, win_hi, 300).tolist()
                try:
                    f_pred = np.array(lcf.eval_model(model, params, t_pred))
                    # Scale model flux to match observed flux (least-squares)
                    f_at_obs = np.array(lcf.eval_model(
                        model, params, t[det_in].tolist()))
                    obs_flux = flux[det_in]
                    denom = np.dot(f_at_obs, f_at_obs)
                    if denom > 1e-30:
                        scale = np.dot(obs_flux, f_at_obs) / denom
                    else:
                        scale = 1.0
                    f_scaled = f_pred * scale
                    # Clip to reasonable range to avoid runaway tails
                    flux_lo = -0.1 * np.max(obs_flux)
                    flux_hi = 1.5 * np.max(obs_flux)
                    f_clipped = np.clip(f_scaled, flux_lo, flux_hi)
                    ax.plot(t_pred, f_clipped, "-", color=color,
                            linewidth=1.5, alpha=0.85, zorder=5,
                            label=model)
                except Exception:
                    pass

            # Feature annotation
            info = []
            if model:
                info.append(f"model: {model}")
            chi2 = r.get("pso_chi2")
            if chi2 is not None:
                info.append(f"chi2={chi2:.2f}")
            mag_chi2 = r.get("mag_chi2")
            if mag_chi2 is not None:
                info.append(f"mag_chi2={mag_chi2:.2f}")
            if info:
                ax.text(
                    0.03, 0.97, "\n".join(info),
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.8, edgecolor="gray"),
                )

        ax.set_xlim(win_lo, win_hi)
        ax.set_xlabel("Time (days)")
        if col == 0:
            ax.set_ylabel("Flux")
        ax.set_title(f"{short_name}-band", fontsize=11)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fig.savefig(outpath, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# Curated source list — well-sampled transients from real ZTF data
# ---------------------------------------------------------------------------
PLOT_SOURCES = {
    "sn_ia":       ("ZTF21aaabwzx", False),
    "sn_ib":       ("ZTF18abktmfz", False),
    "sn_ic":       ("ZTF18abfzhct", False),
    "sn_ii":       ("ZTF21abhyqlv", False),
    "sn_iip":      ("ZTF18aawyjjq", False),
    "sn_iin":      ("ZTF19aacjbsj", False),
    "sn_iib":      ("ZTF20abgbuly", False),
    "cataclysmic": ("ZTF18abccqjx", True),
    "agn":         ("ZTF18aalseci", True),
    "tde":         ("ZTF22aadesap", False),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    with open(FIXTURE_PATH) as f:
        sources_by_id = {s["obj_id"]: s for s in json.load(f)}

    for slug, (obj_id, is_persistent) in PLOT_SOURCES.items():
        source = sources_by_id[obj_id]
        label_name = source.get("label_name", slug)
        print(f"Generating {label_name} ({obj_id})...")

        np_path = os.path.join(IMG_DIR, f"np_{slug}.png")
        plot_nonparametric(source, np_path, is_persistent=is_persistent)

        param_path = os.path.join(IMG_DIR, f"param_{slug}.png")
        plot_parametric(source, param_path, is_persistent=is_persistent)


if __name__ == "__main__":
    main()

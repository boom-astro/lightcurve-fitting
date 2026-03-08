#!/usr/bin/env python3
"""Generate throughput plot from benchmark CSVs.

Produces:
  - docs/throughput_points.png  (point-count scaling, all 5 lines)

Reads:
  - benchmarks/nonparametric_points.csv
  - benchmarks/parametric_points.csv
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(SCRIPT_DIR, "..", "docs")

LINES = [
    ("parametric",    "GPU",     "#1f77b4", "-",  "o", 2.5),
    ("nonparametric", "GPU",     "#ff7f0e", "-",  "s", 2.5),
    ("parametric",    "CPU-par", "#1f77b4", ":",  "o", 2),
    ("parametric",    "CPU",     "#1f77b4", "--", "o", 2),
    ("nonparametric", "CPU",     "#ff7f0e", "--", "s", 2),
]


def load_csv(path):
    """Load CSV into dict keyed by (fitter, backend) -> (x[], y[])."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["fitter"], row["backend"])
            if key not in data:
                data[key] = {"x": [], "y": []}
            data[key]["x"].append(int(row["n_points_per_band"]))
            data[key]["y"].append(float(row["throughput_pts_per_sec"]))
    return data


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    # Load both CSVs
    data = {}
    for name in ("nonparametric_points.csv", "parametric_points.csv"):
        path = os.path.join(SCRIPT_DIR, name)
        if os.path.exists(path):
            data.update(load_csv(path))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for fitter, backend, color, ls, marker, lw in LINES:
        key = (fitter, backend)
        if key not in data:
            continue
        d = data[key]
        ax.plot(d["x"], d["y"],
                color=color, linestyle=ls, marker=marker,
                linewidth=lw, markersize=7,
                label=f"{fitter} {backend}")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Points per band", fontsize=13)
    ax.set_ylabel("Throughput (observations / sec)", fontsize=13)
    ax.set_title("Lightcurve-fitting throughput — point-count scaling (100 sources)", fontsize=14)
    ax.grid(True, which="both", alpha=0.3)

    # X-axis ticks at power-of-2 values
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    # Two-part legend
    fitter_handles = [
        Line2D([0], [0], color="#1f77b4", marker="o", linestyle="-", linewidth=2, markersize=7),
        Line2D([0], [0], color="#ff7f0e", marker="s", linestyle="-", linewidth=2, markersize=7),
    ]
    backend_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.5),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2),
    ]
    leg1 = ax.legend(fitter_handles, ["parametric", "nonparametric"],
                     fontsize=10, loc="upper left", title="Fitter", title_fontsize=10)
    ax.add_artist(leg1)
    ax.legend(backend_handles, ["GPU", "CPU-par (32t)", "CPU"],
              fontsize=10, loc="lower right", title="Backend", title_fontsize=10)

    fig.tight_layout()
    out = os.path.join(DOCS_DIR, "throughput_points.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()

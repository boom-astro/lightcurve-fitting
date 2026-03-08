#!/usr/bin/env python3
"""Generate JSON test fixtures from photo_events NPZ files.

Usage:
    python3 tests/gen_test_fixtures.py

Reads NPZ files from /fred/oz480/mcoughli/AppleCider/photo_events
and writes tests/fixtures/real_sources.json (20 sources, small)
and tests/fixtures/real_sources_1000.json (1000 sources, balanced).
"""

import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

PHOTO_EVENTS_DIR = Path("/fred/oz480/mcoughli/AppleCider/photo_events")
MANIFEST = PHOTO_EVENTS_DIR / "manifest_train.csv"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

BAND_NAMES = {0: "ztfg", 1: "ztfr", 2: "ztfi"}

FINE_ID2NAME = [
    "SN Ia",       # 0
    "SN Ib",       # 1
    "SN Ic",       # 2
    "SN II",       # 3
    "SN IIP",      # 4
    "SN IIn",      # 5
    "SN IIb",      # 6
    "Cataclysmic", # 7
    "AGN",         # 8
    "TDE",         # 9
]


def load_source(npz_path):
    """Load one NPZ and return band-separated data as dicts."""
    d = np.load(npz_path)
    data = d["data"]
    columns = list(d["columns"])

    dt_idx = columns.index("dt")
    band_idx = columns.index("band_id")
    logflux_idx = columns.index("logflux")
    logflux_err_idx = columns.index("logflux_err")

    bands = {}
    for i in range(data.shape[0]):
        band_id = int(data[i, band_idx])
        band_name = BAND_NAMES.get(band_id, f"band{band_id}")

        dt = float(data[i, dt_idx])
        logflux = float(data[i, logflux_idx])
        logflux_err = float(data[i, logflux_err_idx])

        if not (np.isfinite(logflux) and np.isfinite(logflux_err)):
            continue

        mag = -2.5 * logflux + 23.9
        mag_err = 2.5 * logflux_err

        if band_name not in bands:
            bands[band_name] = {"times": [], "values": [], "errors": []}
        bands[band_name]["times"].append(dt)
        bands[band_name]["values"].append(mag)
        bands[band_name]["errors"].append(mag_err)

    return bands


def select_best_transients(rows, per_label=2):
    """Select bright, well-sampled transients for showcase fixtures.

    For SN types (0-6) and TDE (9): pick sources with short time spans
    (<200d), large amplitude (>1.5 mag), bright peaks (<19.5 mag), and
    30-300 total events.  Ranked by amplitude then peak brightness.

    For AGN (8) and Cataclysmic (7): pick most-observed sources since
    long baselines are legitimate for these classes.
    """
    by_label = defaultdict(list)
    for row in rows:
        obj_id = row["obj_id"]
        local_path = PHOTO_EVENTS_DIR / "train" / f"{obj_id}.npz"
        if not local_path.exists():
            continue
        by_label[row["label"]].append(
            (obj_id, str(local_path), row["label"], int(row["n_events"]))
        )

    transient_labels = {str(i) for i in list(range(7)) + [9]}
    selected = []

    for label in sorted(by_label.keys()):
        candidates = by_label[label]

        if label not in transient_labels:
            # AGN / Cataclysmic: just pick most observed
            candidates.sort(key=lambda x: x[3], reverse=True)
            selected.extend(candidates[:per_label])
            continue

        # For transients: score by lightcurve quality
        scored = []
        check_pool = [c for c in candidates if 30 <= c[3] <= 300]
        for obj_id, path, lbl, n_events in check_pool[:300]:
            try:
                d = np.load(path)
                data = d["data"]
                cols = list(d["columns"])
                dt_idx = cols.index("dt")
                band_idx = cols.index("band_id")
                logflux_idx = cols.index("logflux")

                mags = -2.5 * data[:, logflux_idx] + 23.9
                times = data[:, dt_idx]
                bands_arr = data[:, band_idx].astype(int)

                best_amp, best_peak = 0.0, 99.0
                short_span = False
                for bid in [0, 1]:  # g, r
                    mask = (bands_arr == bid) & np.isfinite(mags)
                    if mask.sum() < 10:
                        continue
                    t_b = times[mask]
                    m_b = mags[mask]
                    span = t_b.max() - t_b.min()
                    amp = m_b.max() - m_b.min()
                    peak = m_b.min()
                    if span < 200:
                        short_span = True
                    if amp > best_amp:
                        best_amp = amp
                        best_peak = peak

                if short_span and best_amp > 1.5 and best_peak < 19.5:
                    scored.append((obj_id, path, lbl, n_events, best_amp, best_peak))
            except Exception:
                continue

        scored.sort(key=lambda x: (-x[4], x[5]))  # largest amp, brightest peak
        for s in scored[:per_label]:
            selected.append((s[0], s[1], s[2], s[3]))

        # Fallback if not enough quality sources found
        if len(scored) < per_label:
            candidates.sort(key=lambda x: x[3], reverse=True)
            for c in candidates:
                if c[0] not in {s[0] for s in selected}:
                    selected.append(c)
                    if sum(1 for s in selected if s[2] == label) >= per_label:
                        break

    return selected


def select_sources(rows, n_total, min_events=10, prefer_most_points=False,
                    max_events_by_label=None):
    """Select a balanced sample across labels."""
    # Group by label
    by_label = defaultdict(list)
    for row in rows:
        n_events = int(row["n_events"])
        if n_events < min_events:
            continue
        if max_events_by_label and row["label"] in max_events_by_label:
            if n_events > max_events_by_label[row["label"]]:
                continue
        obj_id = row["obj_id"]
        local_path = PHOTO_EVENTS_DIR / "train" / f"{obj_id}.npz"
        if local_path.exists():
            by_label[row["label"]].append((obj_id, str(local_path), row["label"], n_events))

    n_labels = len(by_label)
    per_label = max(n_total // n_labels, 1)
    remainder = n_total - per_label * n_labels

    selected = []
    for label in sorted(by_label.keys()):
        candidates = by_label[label]
        if prefer_most_points:
            # Take the sources with the most observations
            candidates.sort(key=lambda x: x[3], reverse=True)
            n_take = min(per_label, len(candidates))
            if remainder > 0 and len(candidates) > per_label:
                n_take += 1
                remainder -= 1
            selected.extend(candidates[:n_take])
        else:
            # Take evenly spaced samples for diversity
            candidates.sort(key=lambda x: x[3])
            n_take = min(per_label, len(candidates))
            if remainder > 0 and len(candidates) > per_label:
                n_take += 1
                remainder -= 1
            step = max(len(candidates) / n_take, 1.0)
            for i in range(n_take):
                idx = min(int(i * step), len(candidates) - 1)
                selected.append(candidates[idx])

    return selected[:n_total]


def build_fixture(selected):
    """Load and convert selected sources to fixture format."""
    sources = []
    for obj_id, path, label, n_events in selected:
        try:
            bands = load_source(path)
        except Exception as e:
            print(f"  SKIP {obj_id}: {e}")
            continue

        total_obs = sum(len(b["times"]) for b in bands.values())
        label_int = int(label)
        label_name = FINE_ID2NAME[label_int] if label_int < len(FINE_ID2NAME) else f"unknown_{label}"

        sources.append({
            "obj_id": obj_id,
            "label": label_int,
            "label_name": label_name,
            "bands": bands,
        })

    return sources


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))

    # --- Small fixture (20 sources, best quality per class) ---
    # For SN types + TDE: pick bright transients with short spans and large amplitudes.
    # For AGN + Cataclysmic: pick most-observed (long baselines are legitimate).
    selected_small = select_best_transients(rows, per_label=2)
    sources_small = build_fixture(selected_small)

    small_path = FIXTURES_DIR / "real_sources.json"
    with open(small_path, "w") as f:
        json.dump(sources_small, f)

    print(f"Small fixture: {len(sources_small)} sources, {small_path.stat().st_size / 1024:.1f} KB")
    label_counts = defaultdict(int)
    for s in sources_small:
        label_counts[s["label_name"]] += 1
    for name, count in sorted(label_counts.items()):
        print(f"  {name}: {count}")

    # --- Large fixture (1000 sources) ---
    selected_large = select_sources(rows, 1000, min_events=10)
    sources_large = build_fixture(selected_large)

    large_path = FIXTURES_DIR / "real_sources_1000.json"
    with open(large_path, "w") as f:
        json.dump(sources_large, f)

    print(f"\nLarge fixture: {len(sources_large)} sources, {large_path.stat().st_size / 1024 / 1024:.1f} MB")
    label_counts = defaultdict(int)
    for s in sources_large:
        label_counts[s["label_name"]] += 1
    for name, count in sorted(label_counts.items()):
        print(f"  {name}: {count}")

    total_obs = sum(sum(len(b["times"]) for b in s["bands"].values()) for s in sources_large)
    print(f"  Total observations: {total_obs}")


if __name__ == "__main__":
    main()

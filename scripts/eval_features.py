#!/usr/bin/env python3
"""Evaluate nonparametric + thermal features on AppleCider data."""

import sys, os, csv
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import lightcurve_fitting as lcf

DATA_DIR = "/fred/oz480/mcoughli/AppleCider/photo_events/train"
MANIFEST = "/fred/oz480/mcoughli/AppleCider/photo_events/manifest_train.csv"

SUBCLASS_NAMES = [
    "SN Ia", "SN Ib", "SN Ic", "SN II", "SN IIP", "SN IIn", "SN IIb",
    "Cataclysmic", "AGN", "TDE"
]

BAND_MAP = {0: "ztfg", 1: "ztfr", 2: "ztfi"}
ZP = 23.9  # AB magnitude zero point


def logflux_to_mag(logflux, logflux_err):
    mag = ZP - 2.5 * logflux
    mag_err = 2.5 * logflux_err
    return mag, mag_err


def load_source(obj_id):
    path = os.path.join(DATA_DIR, f"{obj_id}.npz")
    d = np.load(path)
    data = d['data']
    label = int(d['label'])

    dt = data[:, 0]
    band_id = data[:, 2].astype(int)
    logflux = data[:, 3]
    logflux_err = data[:, 4]

    valid = np.isfinite(logflux) & np.isfinite(logflux_err) & (logflux_err < 2.0) & (logflux_err > 0)
    dt = dt[valid]
    band_id = band_id[valid]
    logflux = logflux[valid]
    logflux_err = logflux_err[valid]

    mags, mag_errs = logflux_to_mag(logflux, logflux_err)
    bands = [BAND_MAP.get(b, f"band{b}") for b in band_id]
    return dt.tolist(), mags.tolist(), mag_errs.tolist(), bands, label


def read_manifest():
    """Read CSV manifest without pandas."""
    rows = []
    with open(MANIFEST) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['label'] = int(row['label'])
            row['n_events'] = int(row['n_events'])
            rows.append(row)
    return rows


def main():
    manifest = read_manifest()

    N_PER_CLASS = 20
    rng = np.random.RandomState(42)

    # Sample per class
    sampled = []
    n_per_class = {}
    for label_id in range(10):
        subset = [r for r in manifest if r['label'] == label_id and r['n_events'] >= 10]
        if len(subset) > N_PER_CLASS:
            idxs = rng.choice(len(subset), N_PER_CLASS, replace=False)
            subset = [subset[i] for i in idxs]
        n_per_class[label_id] = len(subset)
        sampled.extend(subset)

    print(f"Evaluating {len(sampled)} sources across 10 classes")
    print()

    features_by_class = defaultdict(lambda: defaultdict(list))
    failed_by_class = defaultdict(int)

    for row in sampled:
        obj_id = row['obj_id']
        label = row['label']
        class_name = SUBCLASS_NAMES[label]

        try:
            times, mags, errs, bands, _ = load_source(obj_id)
        except Exception:
            failed_by_class[class_name] += 1
            continue

        if len(times) < 5:
            failed_by_class[class_name] += 1
            continue

        try:
            band_data = lcf.build_mag_bands(times, mags, errs, bands)
            results = lcf.fit_nonparametric(band_data)
        except Exception:
            failed_by_class[class_name] += 1
            continue

        for band_result in results:
            for key in ['peak_mag', 'amplitude', 'duration', 'dm15', 'decay_efold',
                        'decay_halfmax', 'fwhm', 'rise_time', 'rise_halfmax', 'rise_efold',
                        'rise_rate', 'decay_rate', 'near_peak_rise_rate', 'near_peak_decay_rate',
                        'chi2_per_dof', 'post_peak_monotonicity']:
                val = band_result.get(key)
                if val is not None and np.isfinite(val):
                    features_by_class[class_name][key].append(val)

        try:
            thermal = lcf.fit_thermal(band_data)
            if thermal is not None:
                for key in ['log_temp_peak', 'log_temp_latest', 'cooling_rate', 'chi2']:
                    val = thermal.get(key)
                    if val is not None and np.isfinite(val):
                        features_by_class[class_name][f"thermal_{key}"].append(val)
        except Exception:
            pass

    # Print median feature values
    all_features = [
        'peak_mag', 'amplitude', 'duration', 'dm15', 'decay_efold', 'decay_halfmax',
        'fwhm', 'rise_time', 'rise_halfmax', 'rise_efold',
        'rise_rate', 'decay_rate', 'near_peak_rise_rate', 'near_peak_decay_rate',
        'post_peak_monotonicity',
        'thermal_log_temp_peak', 'thermal_log_temp_latest', 'thermal_cooling_rate',
    ]

    short_names = []
    for label_id in range(10):
        name = SUBCLASS_NAMES[label_id]
        short = name.replace("SN ", "").replace("Cataclysmic", "CV")
        short_names.append(short)

    print(f"{'Feature':<28}", end="")
    for s in short_names:
        print(f"  {s:>10}", end="")
    print()
    print("-" * 28 + ("-" * 12) * 10)

    for feat in all_features:
        print(f"{feat:<28}", end="")
        for label_id in range(10):
            class_name = SUBCLASS_NAMES[label_id]
            vals = features_by_class[class_name].get(feat, [])
            if len(vals) >= 2:
                print(f"  {np.median(vals):10.3f}", end="")
            elif len(vals) == 1:
                print(f"  {vals[0]:10.3f}", end="")
            else:
                print(f"  {'---':>10}", end="")
        print()

    # NaN rates
    print()
    print("NaN / missing rates:")
    print(f"{'Feature':<28}", end="")
    for s in short_names:
        print(f"  {s:>10}", end="")
    print()
    print("-" * 28 + ("-" * 12) * 10)

    for feat in all_features:
        print(f"{feat:<28}", end="")
        for label_id in range(10):
            class_name = SUBCLASS_NAMES[label_id]
            vals = features_by_class[class_name].get(feat, [])
            total = len(features_by_class[class_name].get('peak_mag', []))
            if feat.startswith('thermal_'):
                total = n_per_class[label_id]
            if total > 0:
                nan_rate = 1.0 - len(vals) / total
                print(f"  {nan_rate:10.1%}", end="")
            else:
                print(f"  {'---':>10}", end="")
        print()

    print()
    print("Failed sources per class:")
    for label_id in range(10):
        name = SUBCLASS_NAMES[label_id]
        print(f"  {name}: {failed_by_class[name]}")


if __name__ == "__main__":
    main()

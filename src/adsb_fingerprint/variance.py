"""adsb-variance: nested variance decomposition of physical-layer features.

Grounds the replication-vs-coverage sampling question in data. Per-message
features (see features.py) are decomposed, aircraft by aircraft, into
within-window (back-to-back replication, i.e. the measurement noise floor),
between-window (channel/geometry drift within a session), and between-session
components, then pooled across aircraft with the unbalanced-nested method of
moments (Henderson). The components translate directly into policy: the
within-window ICC gives the effective sample size of k reps per window, and
k* = sqrt(var_msg / var_win) is where extra reps stop beating new windows.
"""

import argparse
from collections import (
    Counter,
    defaultdict,
)

import numpy as np
from tqdm import tqdm

from adsb_fingerprint import (
    config,
    db,
    features,
)

ALL_FEATURES = features.FEATURES + ["rssi_db"]

UNITS = {
    "cfo_hz": "Hz",
    "p14_spacing_ns": "ns",
    "preamble_ratio_db": "dB",
    "rise_ns": "ns",
    "rssi_db": "dB",
    "snr_db": "dB",
}


def load_messages(window_seconds):
    """Extract features for every crc-ok message, grouped for the nested design.

    Returns aligned numpy arrays: icao, session, window (wall-clock bucket of
    window_seconds), and a {feature: values} column store (NaN = unmeasurable).
    """
    with db.connect() as conn:
        rows = conn.execute(
            """
            select
                capture_file,
                sample_offset,
                n_samples,
                icao,
                session,
                captured_at,
                rssi_db
            from messages
            where crc_ok
            order by
                capture_file,
                sample_offset
            """
        ).fetchall()

    by_file = defaultdict(list)
    for row in rows:
        by_file[row["capture_file"]].append(row)

    icao, session, window = [], [], []
    cols = {name: [] for name in ALL_FEATURES}
    with tqdm(total=len(rows), unit=" msg", desc="extract features") as bar:
        for capture_file, group in by_file.items():
            path = config.CAPTURE_DIR / capture_file
            if not path.exists():
                bar.update(len(group))
                continue
            data = np.fromfile(path, dtype=np.complex64)
            for row in group:
                snippet = data[row["sample_offset"] : row["sample_offset"] + row["n_samples"]]
                feats = features.extract(snippet, config.SAMPLE_RATE_HZ)
                icao.append(row["icao"])
                session.append(row["session"])
                window.append(int(row["captured_at"].timestamp() // window_seconds))
                for name in features.FEATURES:
                    cols[name].append(feats[name])
                rssi = row["rssi_db"]
                cols["rssi_db"].append(rssi if rssi is not None else float("nan"))
                bar.update(1)

    return (
        np.array(icao),
        np.array(session),
        np.array(window),
        {name: np.array(vals, dtype=float) for name, vals in cols.items()},
    )


def nested_sums(values, sessions, windows):
    """Sums of squares and Henderson coefficients for one aircraft's sample.

    Unbalanced-safe: any window/session sizes. The pieces are additive across
    aircraft (every sum is centered within this aircraft), so pooling is just
    summing dicts before solve_components().
    """
    groups = defaultdict(lambda: defaultdict(list))
    for v, s, w in zip(values, sessions, windows):
        groups[s][w].append(v)
    per_session = [
        [np.asarray(vals, dtype=float) for vals in wins.values()]
        for wins in groups.values()
    ]

    n_total = sum(len(a) for arrays in per_session for a in arrays)
    n_windows = sum(len(arrays) for arrays in per_session)
    n_sessions = len(per_session)
    grand = sum(a.sum() for arrays in per_session for a in arrays) / n_total

    ss_e = ss_w = ss_s = 0.0
    k1 = k2 = k3 = 0.0
    session_means = []
    for arrays in per_session:
        n_s = sum(len(a) for a in arrays)
        m_s = sum(a.sum() for a in arrays) / n_s
        session_means.append(np.mean([a.mean() for a in arrays]))
        ss_s += n_s * (m_s - grand) ** 2
        k1 += sum(len(a) ** 2 for a in arrays) / n_s
        k3 += n_s**2
        for a in arrays:
            ss_e += ((a - a.mean()) ** 2).sum()
            ss_w += len(a) * (a.mean() - m_s) ** 2
            k2 += len(a) ** 2
    k2 /= n_total
    k3 /= n_total

    return {
        "ss_e": ss_e,
        "df_e": n_total - n_windows,
        "ss_w": ss_w,
        "df_w": n_windows - n_sessions,
        "c_w": n_total - k1,
        "ss_s": ss_s,
        "df_s": n_sessions - 1,
        "c_ws": k1 - k2,
        "c_s": n_total - k3,
        "n": n_total,
        "n_windows": n_windows,
        "n_sessions": n_sessions,
        "mean": float(np.mean(session_means)),
    }


def pool_sums(sum_dicts):
    keys = [
        "ss_e",
        "df_e",
        "ss_w",
        "df_w",
        "c_w",
        "ss_s",
        "df_s",
        "c_ws",
        "c_s",
        "n",
        "n_windows",
        "n_sessions",
    ]
    return {key: sum(d[key] for d in sum_dicts) for key in keys}


def solve_components(sums):
    """Solve the expected-mean-square equations for (var_msg, var_win, var_sess).

    Sequential method of moments; negative estimates clamp to zero (standard).
    NaN where the design leaves a level unestimable.
    """
    nan = float("nan")
    var_e = sums["ss_e"] / sums["df_e"] if sums["df_e"] > 0 else nan
    var_w = (
        max(0.0, (sums["ss_w"] - sums["df_w"] * var_e) / sums["c_w"])
        if sums["c_w"] > 0 and np.isfinite(var_e)
        else nan
    )
    var_s = (
        max(0.0, (sums["ss_s"] - sums["df_s"] * var_e - sums["c_ws"] * var_w) / sums["c_s"])
        if sums["c_s"] > 0 and np.isfinite(var_w)
        else nan
    )
    return var_e, var_w, var_s


def _labels(icaos):
    with db.connect() as conn:
        rows = conn.execute(
            """
            select
                icao,
                registration,
                model
            from aircraft
            where icao = any(%(icaos)s)
            """,
            {"icaos": list(icaos)},
        ).fetchall()
    return {
        row["icao"]: " ".join(p for p in (row["registration"], row["model"]) if p)
        for row in rows
    }


def _fmt(value, width=9, prec=3):
    if value is None or not np.isfinite(value):
        return "-".rjust(width)
    return f"{value:{width}.{prec}g}"


def main():
    parser = argparse.ArgumentParser(
        description="Decompose per-message feature variance into within-window / between-window / between-session components.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=10,
        help="Window bucket size for the middle level (default 10, the collect policy window).",
    )
    parser.add_argument(
        "--min-messages",
        type=int,
        default=10,
        help="Min messages per aircraft to enter the pooled estimate.",
    )
    parser.add_argument(
        "--feature",
        choices=ALL_FEATURES,
        default="cfo_hz",
        help="Feature for the per-aircraft table.",
    )
    parser.add_argument(
        "--screen-sd",
        type=float,
        default=10.0,
        help="Drop messages this many robust sd from their aircraft's median (0 = no screening).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Aircraft rows in the per-aircraft table.",
    )
    args = parser.parse_args()

    icao, session, window, cols = load_messages(args.window_seconds)
    if not len(icao):
        print("No messages yet — run adsb-collect / adsb-index first.")
        return

    counts = Counter(icao.tolist())
    keep = sorted(a for a, n in counts.items() if n >= args.min_messages)
    aircraft_idx = {
        a: np.flatnonzero(icao == a)
        for a in keep
    }

    per_aircraft = {name: {} for name in ALL_FEATURES}
    pooled = {}
    screened = Counter()
    for name in ALL_FEATURES:
        finite = np.isfinite(cols[name])
        for a, idx in aircraft_idx.items():
            use = idx[finite[idx]]
            if len(use) < 2:
                continue
            # Gross-error screen: a single corrupted extraction (e.g. an
            # overlapping transmission that still passed CRC) would dominate
            # the plain-moments components, so the components describe the
            # central bulk and the screened count is reported alongside.
            vals = cols[name][use]
            med = np.median(vals)
            scale = 1.4826 * np.median(np.abs(vals - med))
            if args.screen_sd > 0 and scale > 0:
                inlier = np.abs(vals - med) <= args.screen_sd * scale
                screened[name] += int(len(use) - inlier.sum())
                use = use[inlier]
                if len(use) < 2:
                    continue
            per_aircraft[name][a] = nested_sums(
                cols[name][use],
                session[use],
                window[use],
            )
        pooled[name] = pool_sums(per_aircraft[name].values())

    k_policy = max(config.COLLECT_MAX_PER_AIRCRAFT, 1)
    ref = pooled[args.feature]
    print(
        f"=== nested variance decomposition "
        f"(session > {args.window_seconds} s window > message) ==="
    )
    print(
        f"pooled over {len(per_aircraft[args.feature])} aircraft "
        f"(>= {args.min_messages} msgs): "
        f"{ref['n']:,} msgs in {ref['n_windows']:,} windows, "
        f"{ref['n_sessions']:,} aircraft-sessions"
    )

    print(f"\n{'feature':<19}{'unit':<5}{'sd(msg)':>9}{'sd(win)':>9}{'sd(sess)':>9}   {'msg%':>4}{'win%':>6}{'sess%':>7}")
    components = {}
    for name in ALL_FEATURES:
        var_e, var_w, var_s = solve_components(pooled[name])
        components[name] = (var_e, var_w, var_s)
        total = var_e + var_w + var_s
        shares = (
            f"   {var_e / total:4.0%}{var_w / total:6.0%}{var_s / total:7.0%}"
            if np.isfinite(total) and total > 0
            else ""
        )
        print(
            f"{name:<19}{UNITS[name]:<5}"
            f"{_fmt(np.sqrt(var_e))}{_fmt(np.sqrt(var_w))}{_fmt(np.sqrt(var_s))}"
            f"{shares}"
        )
    print("sd(msg) = replication noise floor; sd(win)/sd(sess) = what invariance must absorb")
    if screened:
        drops = ", ".join(f"{name} {n}" for name, n in sorted(screened.items()))
        print(f"screened outliers (>{args.screen_sd:g} robust sd from aircraft median): {drops}")

    print("\n=== sampling policy (replication vs coverage) ===")
    print(f"{'feature':<19}{'icc(win)':>9}{f'ess(k={k_policy})':>9}{'k*':>7}")
    for name in ALL_FEATURES:
        var_e, var_w, var_s = components[name]
        total = var_e + var_w + var_s
        if not (np.isfinite(total) and total > 0):
            continue
        icc = (var_w + var_s) / total
        ess = k_policy / (1.0 + (k_policy - 1) * icc)
        k_star = np.sqrt(var_e / var_w) if var_w > 0 else float("inf")
        k_star_txt = f"{k_star:7.1f}" if np.isfinite(k_star) else "    inf"
        print(f"{name:<19}{icc:9.2f}{ess:9.2f}{k_star_txt}")
    print("icc(win) = correlation of two msgs in one window; ess = what k reps are worth")
    print("k* = sqrt(var_msg/var_win): reps beyond k* re-measure the window, spread instead")

    print("\n=== fingerprint signal (between-aircraft separation) ===")
    print(f"{'feature':<19}{'sd(between)':>12}{'sd(within)':>12}{'ratio':>7}")
    for name in ALL_FEATURES:
        means = [
            s["mean"]
            for s in per_aircraft[name].values()
            if s["n_sessions"] >= 2
        ]
        var_e, var_w, var_s = components[name]
        within = np.sqrt(var_e + var_w + var_s)
        if len(means) < 3 or not np.isfinite(within) or within <= 0:
            continue
        between = float(np.std(means, ddof=1))
        print(f"{name:<19}{_fmt(between, 12)}{_fmt(within, 12)}{between / within:7.1f}")
    print("aircraft means over >= 2 sessions; ratio >> 1 means the feature already separates airframes")

    detail = sorted(
        per_aircraft[args.feature].items(),
        key=lambda item: item[1]["n"],
        reverse=True,
    )[: args.top]
    labels = _labels([a for a, _ in detail])
    print(f"\n=== per aircraft: {args.feature} ({UNITS[args.feature]}) ===")
    print(f"{'icao':<8}{'msgs':>6}{'sess':>5}{'wins':>6}{'mean':>10}{'sd(msg)':>9}{'sd(win)':>9}{'sd(sess)':>9}  label")
    for a, sums in detail:
        var_e, var_w, var_s = solve_components(sums)
        print(
            f"{a:<8}{sums['n']:>6}{sums['n_sessions']:>5}{sums['n_windows']:>6}"
            f"{_fmt(sums['mean'], 10)}"
            f"{_fmt(np.sqrt(var_e))}{_fmt(np.sqrt(var_w))}{_fmt(np.sqrt(var_s))}"
            f"  {labels.get(a, '')}"
        )


if __name__ == "__main__":
    main()

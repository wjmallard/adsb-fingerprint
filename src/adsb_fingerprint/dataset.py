"""Assemble per-message IQ training examples from the messages index.

The captures on disk are the source of truth; this reads each one once, slices
the IQ at every validated message's sample offset, amplitude-normalizes, and
labels by ICAO. Splits are by session (never random) so cross-session accuracy
can expose channel/geometry confounds rather than hardware fingerprints.
"""

import argparse
import re
from collections import Counter, defaultdict

import numpy as np

from adsb_fingerprint import config, db, modes


def load_examples(window=None, pre=0, min_messages=1, min_sessions=1):
    """Return amplitude-normalized per-message IQ windows and their labels.

    Keeps ICAOs with >= min_messages spanning >= min_sessions sessions. Returns
    a dict of aligned numpy arrays: iq (n, window) complex64, icao, session,
    captured_at, rssi_db.
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

    counts = defaultdict(int)
    seen_sessions = defaultdict(set)
    for row in rows:
        counts[row["icao"]] += 1
        seen_sessions[row["icao"]].add(row["session"])
    keep = {
        icao
        for icao in counts
        if counts[icao] >= min_messages and len(seen_sessions[icao]) >= min_sessions
    }
    rows = [row for row in rows if row["icao"] in keep]

    if window is None:
        window = max((row["n_samples"] for row in rows), default=0)

    by_file = defaultdict(list)
    for row in rows:
        by_file[row["capture_file"]].append(row)

    iq, icao, session, captured_at, rssi = [], [], [], [], []
    for capture_file, group in by_file.items():
        data = np.fromfile(config.CAPTURE_DIR / capture_file, dtype=np.complex64)
        for row in group:
            start = max(row["sample_offset"] - pre, 0)
            snippet = data[start : start + window]
            if len(snippet) < window:
                snippet = np.pad(snippet, (0, window - len(snippet)))
            peak = np.abs(snippet).max()
            if peak > 0:
                snippet = snippet / peak
            iq.append(snippet.astype(np.complex64))
            icao.append(row["icao"])
            session.append(row["session"])
            captured_at.append(row["captured_at"])
            rssi.append(row["rssi_db"])

    return {
        "iq": np.array(iq) if iq else np.empty((0, window), np.complex64),
        "icao": np.array(icao),
        "session": np.array(session),
        "captured_at": np.array(captured_at),
        "rssi_db": np.array(rssi, dtype=float),
        "window": window,
    }


def apply_variant(iq, variant, sample_rate=None):
    """Return an ablation view of the IQ windows.

    whole        - unchanged.
    preamble     - keep only the 8 us preamble (the protocol-constant part).
    icao_masked  - zero the ICAO address field (message bits 9-32) so the net
                   can't just read the ID out of the data block.
    """
    if variant == "whole":
        return iq
    sample_rate = sample_rate or config.SAMPLE_RATE_HZ
    spb = sample_rate / 1e6
    out = iq.copy()
    if variant == "preamble":
        out[:, int(round(modes.PREAMBLE_US * spb)):] = 0
    elif variant == "icao_masked":
        start = int(round((modes.PREAMBLE_US + 8) * spb))
        end = int(round((modes.PREAMBLE_US + 32) * spb))
        out[:, start:end] = 0
    else:
        raise ValueError(f"unknown variant: {variant!r}")
    return out


def session_split(sessions, test_sessions):
    """Boolean (train, test) masks holding out the named sessions."""
    test = np.isin(sessions, list(test_sessions))
    return ~test, test


def _labels(icaos):
    with db.connect() as conn:
        rows = conn.execute(
            """
            select
                icao,
                registration,
                manufacturer,
                model,
                type
            from aircraft
            where icao = any(%(icaos)s)
            """,
            {"icaos": list(icaos)},
        ).fetchall()
    return {
        row["icao"]: " ".join(
            part
            for part in (row["registration"], row["manufacturer"], row["model"], row["type"] and f'[{row["type"]}]')
            if part
        )
        for row in rows
    }


def _makers(icaos):
    """Map each ICAO to its maker: the registry manufacturer's first word.

    The FAA/OpenSky manufacturer field spells one brand many ways ("AIRBUS",
    "Airbus Industrie", "AIRBUS S A S", "EMBRAER-EMPRESA BRASILEIRA DE"), so
    the leading word — split on spaces, hyphens and slashes, title-cased —
    collapses them without inventing a taxonomy. Corporate parentage is left
    alone: the registry's own answer stands, so Textron-built Cessnas group
    under whichever name their registration actually carries.
    """
    with db.connect() as conn:
        rows = conn.execute(
            """
            select
                icao,
                manufacturer
            from aircraft
            where icao = any(%(icaos)s)
            """,
            {"icaos": list(icaos)},
        ).fetchall()
    makers = {}
    for row in rows:
        words = re.split(r"[\s\-/]+", (row["manufacturer"] or "").strip())
        if words and words[0]:
            makers[row["icao"]] = words[0].title()
    return makers


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the ADS-B fingerprinting dataset built from the messages index.",
    )
    parser.add_argument("--min-messages", type=int, default=1, help="Min messages per ICAO to keep.")
    parser.add_argument("--min-sessions", type=int, default=1, help="Min distinct sessions per ICAO to keep.")
    parser.add_argument("--window", type=int, default=None, help="IQ window length in samples (default: message length).")
    args = parser.parse_args()

    data = load_examples(
        window=args.window,
        min_messages=args.min_messages,
        min_sessions=args.min_sessions,
    )
    n = len(data["icao"])
    if n == 0:
        print("No examples yet — capture and adsb-index some messages first.")
        return

    classes = sorted(set(data["icao"].tolist()))
    sessions = sorted(set(data["session"].tolist()))
    labels = _labels(classes)

    print(f"examples : {n}")
    print(f"classes  : {len(classes)} ICAOs (>= {args.min_messages} msgs, >= {args.min_sessions} sessions)")
    print(f"sessions : {len(sessions)} ({', '.join(sessions)})")
    print(f"iq shape : {tuple(data['iq'].shape)} complex64, amplitude-normalized")

    print("\nper class:")
    for icao, count in Counter(data["icao"].tolist()).most_common():
        print(f"  {icao}  {count:5d}   {labels.get(icao, '')}")

    print("\nper session:")
    for session, count in sorted(Counter(data["session"].tolist()).items()):
        print(f"  {session}: {count}")


if __name__ == "__main__":
    main()

"""adsb-index: detect Mode S messages in captures and populate the messages index.

Each capture's .cf32 stays on disk (the source of truth); this writes one row
per validated message, anchored to its sample offset in the file, so the IQ can
be re-read later for fingerprinting.
"""

import argparse
import json
from datetime import datetime, timedelta

import numpy as np

from adsb_fingerprint import config, db, modes

COPY_SQL = """
    copy messages (
        capture_file,
        sample_offset,
        n_samples,
        captured_at,
        session,
        df,
        icao,
        type_code,
        crc_ok,
        rssi_db,
        hex,
        altitude_ft,
        latitude,
        longitude,
        callsign,
        ground_speed,
        track,
        vertical_rate
    )
    from stdin
"""


def _iter_sidecars(session=None):
    root = config.CAPTURE_DIR / session if session else config.CAPTURE_DIR
    return sorted(root.glob("**/*.json"))


def index_capture(conn, sidecar_path, reindex=False):
    meta = json.loads(sidecar_path.read_text())
    data_path = sidecar_path.with_suffix(".cf32")
    rel = str(data_path.relative_to(config.CAPTURE_DIR))

    existing = conn.execute(
        "select count(*) as n from messages where capture_file = %(f)s",
        {"f": rel},
    ).fetchone()["n"]
    if existing and not reindex:
        return None

    sample_rate = meta["sample_rate_hz"]
    started = datetime.fromisoformat(meta["captured_at"])
    session = meta["session"]
    reference = (config.RECEIVER_LAT, config.RECEIVER_LON)
    n_samples = int(round(modes.MESSAGE_US * sample_rate / 1e6))

    iq = np.fromfile(data_path, dtype=np.complex64)
    rows = [
        (
            rel,
            msg["sample_offset"],
            n_samples,
            started + timedelta(seconds=msg["sample_offset"] / sample_rate),
            session,
            msg["df"],
            msg["icao"],
            msg["type_code"],
            True,
            msg["rssi_db"],
            msg["hex"],
            msg.get("altitude_ft"),
            msg.get("latitude"),
            msg.get("longitude"),
            msg.get("callsign"),
            msg.get("ground_speed"),
            msg.get("track"),
            msg.get("vertical_rate"),
        )
        for msg in modes.detect_messages(iq, sample_rate, reference=reference)
    ]

    conn.execute("delete from messages where capture_file = %(f)s", {"f": rel})
    with conn.cursor() as cur, cur.copy(COPY_SQL) as copy:
        for row in rows:
            copy.write_row(row)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Detect Mode S ADS-B messages in captures and fill the messages index.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Only index captures in this session (default: all).",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Re-index captures that already have rows (default: skip them).",
    )
    args = parser.parse_args()

    sidecars = _iter_sidecars(args.session)
    if not sidecars:
        raise SystemExit(f"No captures found under {config.CAPTURE_DIR}")

    total = 0
    indexed = 0
    with db.connect() as conn:
        for sidecar in sidecars:
            n = index_capture(conn, sidecar, reindex=args.reindex)
            if n is None:
                print(f"  skip {sidecar.parent.name}/{sidecar.stem} (already indexed)")
                continue
            indexed += 1
            total += n
            print(f"  {sidecar.parent.name}/{sidecar.stem}: {n} messages")
        conn.commit()
    print(f"indexed {total} messages from {indexed} capture(s)")


if __name__ == "__main__":
    main()

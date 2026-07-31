"""Back-fill per-message receiver positions from logged GPS tracks.

Each target message's receiver position is linearly interpolated between
the two track fixes bracketing its captured_at (the photo-geotagging
pattern). Track gaps longer than --max-gap leave the position null —
"unknown", the same semantics as the antenna-transition era. This also
protects sleep-compressed sessions: the logger stops when the machine
sleeps, so compressed stretches have no nearby brackets and stay untagged
rather than being smeared along a bad time axis.
"""

import argparse
import bisect
import csv
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from adsb_fingerprint import config, db


def parse_when(text):
    """Accept YYYY-MM-DD or full ISO 8601; naive values are local time."""
    when = datetime.fromisoformat(text)
    if when.tzinfo is None:
        when = when.astimezone()
    return when


def load_fixes(start_epoch, end_epoch, directory=None):
    """Timestamped (epoch, lat, lon) fixes overlapping the target range."""
    fixes = []
    if directory is None:
        directory = config.GPS_TRACK_DIR
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                epoch = datetime.fromisoformat(row["host_utc"]).timestamp()
                if start_epoch <= epoch <= end_epoch:
                    fixes.append(
                        (
                            epoch,
                            float(row["latitude"]),
                            float(row["longitude"]),
                        )
                    )
    fixes.sort()
    return fixes


def interpolate(fixes, times, epoch, max_gap):
    """Position at epoch, or (None, None) without adequate brackets."""
    i = bisect.bisect_left(times, epoch)
    if i < len(times) and times[i] == epoch:
        _, lat, lon = fixes[i]
        return lat, lon
    if 0 < i < len(times) and times[i] - times[i - 1] <= max_gap:
        t0, lat0, lon0 = fixes[i - 1]
        t1, lat1, lon1 = fixes[i]
        weight = (epoch - t0) / (t1 - t0)
        return (
            lat0 + weight * (lat1 - lat0),
            lon0 + weight * (lon1 - lon0),
        )
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Back-fill receiver positions from GPS track logs.",
    )
    parser.add_argument(
        "sessions",
        nargs="*",
        help="Session names to tag (alternative to --since/--until).",
    )
    parser.add_argument(
        "--since",
        help="Start of the time range (YYYY-MM-DD or ISO 8601; naive = local).",
    )
    parser.add_argument(
        "--until",
        help="End of the time range (YYYY-MM-DD or ISO 8601; naive = local).",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=30,
        help="Bridge track gaps up to this many seconds (default: 30).",
    )
    parser.add_argument(
        "--tracks",
        default=None,
        help=(
            "Track directory to read (default: the puck's gps.tracks; "
            "point at gps.wifi_tracks for Location Services tracks)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without updating the database.",
    )
    args = parser.parse_args()

    by_range = bool(args.since or args.until)
    if args.sessions and by_range:
        raise SystemExit("give session names or --since/--until, not both")
    if not args.sessions and not (args.since and args.until):
        raise SystemExit("give session names, or both --since and --until")

    with db.connect() as conn:
        if args.sessions:
            rows = conn.execute(
                """
                select
                    id,
                    captured_at
                from messages
                where session = any(%(sessions)s)
                order by captured_at
                """,
                {"sessions": args.sessions},
            ).fetchall()
        else:
            rows = conn.execute(
                """
                select
                    id,
                    captured_at
                from messages
                where captured_at >= %(since)s
                  and captured_at < %(until)s
                order by captured_at
                """,
                {
                    "since": parse_when(args.since),
                    "until": parse_when(args.until),
                },
            ).fetchall()
        if not rows:
            raise SystemExit("no messages in the target range")

        first = rows[0]["captured_at"].timestamp()
        last = rows[-1]["captured_at"].timestamp()
        fixes = load_fixes(
            first - args.max_gap,
            last + args.max_gap,
            directory=Path(args.tracks).expanduser() if args.tracks else None,
        )
        print(f"{len(rows)} messages, {len(fixes)} track fixes in range")
        if not fixes:
            raise SystemExit("no track fixes overlap the target range")
        times = [fix[0] for fix in fixes]

        updates = []
        tagged = 0
        for row in tqdm(rows, unit=" msgs", desc="interpolate"):
            lat, lon = interpolate(
                fixes,
                times,
                row["captured_at"].timestamp(),
                args.max_gap,
            )
            if lat is not None:
                tagged += 1
            updates.append((row["id"], lat, lon))

        print(f"{tagged} tagged, {len(updates) - tagged} left null (gaps)")
        if args.dry_run:
            print("dry run — database unchanged")
            return

        conn.execute(
            """
            create temp table geotag_updates (
                id bigint primary key,
                receiver_latitude double precision,
                receiver_longitude double precision
            ) on commit drop
            """
        )
        db.copy_rows(
            conn,
            "copy geotag_updates (id, receiver_latitude, receiver_longitude) from stdin",
            updates,
            total=len(updates),
            desc="load",
        )
        updated = conn.execute(
            """
            update messages
            set receiver_latitude = u.receiver_latitude,
                receiver_longitude = u.receiver_longitude
            from geotag_updates u
            where messages.id = u.id
            """
        ).rowcount
        conn.commit()
        print(f"updated {updated} messages")

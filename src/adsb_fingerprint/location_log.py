"""adsb-location-log: log Location Services fixes to daily track files.

The Wi-Fi twin of adsb-gps-log: one row per interval from the laptop's
own position — reliable indoors where the puck sees no sky, coarser
(±tens of meters) where the puck is sharp. One CSV per UTC day in its
own directory (gps.wifi_tracks) so provenance stays obvious next to the
puck tracks; the shared host_utc/latitude/longitude columns mean
adsb-geotag interpolates either kind via --tracks. Needs the one-time
Location Services grant: run adsb-location first.
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timezone

from adsb_fingerprint import config, location

FIELDS = [
    "host_utc",
    "latitude",
    "longitude",
    "accuracy_m",
]

QUERY_TIMEOUT_S = 5      # per-poll budget; locationd answers from cache in ~0.1 s
HEARTBEAT_SECONDS = 300  # non-TTY reassurance cadence

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


class TrackWriter:
    """Append fixes to one CSV per UTC day, rotating at midnight."""

    def __init__(self, directory, announce=print):
        self.directory = directory
        self.announce = announce
        self.day = None
        self.handle = None
        self.writer = None

    def write(self, row):
        day = row[0][:10].replace("-", "")
        if day != self.day:
            self.close()
            path = self.directory / f"{day}.csv"
            fresh = not path.exists()
            self.handle = path.open("a", newline="")
            self.writer = csv.writer(self.handle)
            if fresh:
                self.writer.writerow(FIELDS)
            self.day = day
            self.announce(f"track file {path}")
        self.writer.writerow(row)
        self.handle.flush()

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None


def main():
    parser = argparse.ArgumentParser(
        description="Log macOS Location Services fixes to daily track files.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=15,
        help="Seconds between fixes (default: 15, comfortably inside adsb-geotag's default 30 s gap bridging).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="Stop after this many seconds (0 = run until interrupted).",
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Watch the fixes without writing track files.",
    )
    args = parser.parse_args()

    if location.current_location(timeout=QUERY_TIMEOUT_S) is None:
        raise SystemExit(
            "no Location Services fix — run adsb-location once to authorize "
            "(and check that Wi-Fi is on)"
        )

    writer = None
    if not args.no_logging:
        directory = config.GPS_WIFI_TRACK_DIR
        directory.mkdir(parents=True, exist_ok=True)
        writer = TrackWriter(directory)
    is_tty = sys.stdout.isatty()
    print(
        f"logging every {args.interval:.0f} s"
        if writer
        else f"watching every {args.interval:.0f} s (not logging)"
    )

    started = time.monotonic()
    n_fixes = 0
    n_misses = 0
    fixing = True
    last_beat = started
    try:
        while not (args.seconds and time.monotonic() - started >= args.seconds):
            loop_start = time.monotonic()
            fix = location.current_location(timeout=QUERY_TIMEOUT_S)
            stamp = datetime.now(timezone.utc)
            clock = stamp.strftime("%H:%M:%SZ")
            if fix is not None:
                n_fixes += 1
                if writer is not None:
                    writer.write(
                        [
                            stamp.isoformat(),
                            f"{fix[0]:.6f}",
                            f"{fix[1]:.6f}",
                            f"{fix[2]:.0f}",
                        ]
                    )
            else:
                n_misses += 1
            if is_tty:
                line = (
                    f"{GREEN}fix{RESET}  {clock}  {fix[0]:.6f}, {fix[1]:.6f}  "
                    f"±{fix[2]:.0f} m · {n_fixes} rows"
                    if fix is not None
                    else f"{RED}no fix{RESET}  {clock} · {n_fixes} rows"
                )
                sys.stdout.write(f"\033[2K\r{line}")
                sys.stdout.flush()
            else:
                if (fix is not None) != fixing:
                    print(f"[{clock}] {'fix restored' if fix else 'fix lost'}")
                elif time.monotonic() - last_beat >= HEARTBEAT_SECONDS:
                    print(f"[{clock}] {n_fixes} fixes, {n_misses} misses")
                    last_beat = time.monotonic()
            fixing = fix is not None
            time.sleep(max(0.0, args.interval - (time.monotonic() - loop_start)))
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
    if is_tty:
        print()
    elapsed = time.monotonic() - started
    print(f"logged {n_fixes} fixes ({n_misses} misses) in {elapsed:.0f} s")


if __name__ == "__main__":
    main()

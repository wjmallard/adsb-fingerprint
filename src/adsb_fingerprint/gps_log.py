"""Log GPS fixes from the USB puck to daily track files.

One CSV per UTC day under the configured tracks directory, one row per
valid GGA fix (1 Hz). host_utc is the join key against messages.captured_at
(both are host-clock times); gps_utc rides along to audit host clock drift.
The device path changes with the physical USB port, so gps.port may be a
glob — re-resolved on every reconnect.
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from glob import glob

import serial

from adsb_fingerprint import config

FIELDS = [
    "host_utc",
    "gps_utc",
    "latitude",
    "longitude",
    "quality",
    "nsat",
    "hdop",
    "alt_m",
    "speed_kt",
    "track_deg",
]

STATUS_EVERY_SECONDS = 300
NO_FIX_EVERY_SECONDS = 30


def nmea_deg(value, hemisphere):
    """ddmm.mmmm / dddmm.mmmm -> signed decimal degrees."""
    dot = value.index(".")
    degrees = float(value[: dot - 2])
    minutes = float(value[dot - 2 :])
    decimal = degrees + minutes / 60.0
    return -decimal if hemisphere in ("S", "W") else decimal


def checksum_ok(line):
    """Validate the NMEA checksum between '$' and '*'."""
    if not line.startswith("$") or "*" not in line:
        return False
    body, _, stated = line[1:].partition("*")
    computed = 0
    for char in body:
        computed ^= ord(char)
    try:
        return computed == int(stated[:2], 16)
    except ValueError:
        return False


def resolve_port(pattern):
    matches = sorted(glob(pattern))
    return matches[0] if matches else None


class TrackWriter:
    """Append fixes to one CSV per UTC day, rotating at midnight."""

    def __init__(self, directory):
        self.directory = directory
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
            print(f"track file {path}")
        self.writer.writerow(row)
        self.handle.flush()

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None


def main():
    parser = argparse.ArgumentParser(
        description="Log GPS fixes from the USB puck to daily track files.",
    )
    parser.add_argument(
        "--port",
        default=config.GPS_PORT,
        help="Serial device path or glob (default: config gps.port).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="Stop after this many seconds (0 = run until interrupted).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every fix as it is logged.",
    )
    args = parser.parse_args()

    config.GPS_TRACK_DIR.mkdir(parents=True, exist_ok=True)
    track = TrackWriter(config.GPS_TRACK_DIR)
    started = time.monotonic()
    last_status = started
    no_fix_since = started
    last_fix = None
    sats_in_view = {}
    n_fixes = 0
    rmc_date = None
    rmc_speed = ""
    rmc_track = ""

    def expired():
        return args.seconds and time.monotonic() - started >= args.seconds

    try:
        while not expired():
            port = resolve_port(args.port)
            if port is None:
                print(f"no device matches {args.port!r} — retrying in 5 s", file=sys.stderr)
                time.sleep(5)
                continue
            try:
                with serial.Serial(port, 9600, timeout=2) as ser:
                    print(f"logging {port}")
                    while not expired():
                        now = time.monotonic()
                        have_fix = no_fix_since is None
                        interval = STATUS_EVERY_SECONDS if have_fix else NO_FIX_EVERY_SECONDS
                        if now - last_status >= interval:
                            if have_fix:
                                print(f"{n_fixes} fixes logged, latest {last_fix[0]}, {last_fix[1]}")
                            else:
                                state = "no fix yet" if n_fixes == 0 else "fix lost"
                                elapsed = int(now - no_fix_since)
                                in_view = sum(sats_in_view.values())
                                print(
                                    f"{state} — {in_view} satellites in view, "
                                    f"{elapsed // 60}m{elapsed % 60:02d}s elapsed"
                                )
                            last_status = now
                        line = ser.readline().decode(errors="replace").strip()
                        if not checksum_ok(line):
                            continue
                        fields = line[1:].partition("*")[0].split(",")
                        sentence = fields[0][2:]
                        if sentence == "RMC" and len(fields) > 9:
                            if fields[2] == "A":
                                rmc_date = fields[9]
                                rmc_speed = fields[7]
                                rmc_track = fields[8]
                        elif sentence == "GGA" and len(fields) > 9:
                            if fields[6] in ("", "0") or not fields[2] or not fields[4]:
                                if no_fix_since is None:
                                    no_fix_since = time.monotonic()
                                continue
                            no_fix_since = None
                            host_utc = datetime.now(timezone.utc).isoformat()
                            gps_utc = ""
                            if rmc_date and fields[1]:
                                gps_utc = (
                                    f"20{rmc_date[4:6]}-{rmc_date[2:4]}-{rmc_date[0:2]}"
                                    f"T{fields[1][0:2]}:{fields[1][2:4]}:{fields[1][4:]}Z"
                                )
                            row = [
                                host_utc,
                                gps_utc,
                                f"{nmea_deg(fields[2], fields[3]):.6f}",
                                f"{nmea_deg(fields[4], fields[5]):.6f}",
                                fields[6],
                                fields[7],
                                fields[8],
                                fields[9],
                                rmc_speed,
                                rmc_track,
                            ]
                            track.write(row)
                            n_fixes += 1
                            last_fix = (row[2], row[3])
                            if args.verbose:
                                print(",".join(row))
                        elif sentence == "GSV" and len(fields) > 3:
                            # One GSV group per talker (constellation); field 3
                            # counts that constellation's satellites in view.
                            if fields[3].isdigit():
                                sats_in_view[fields[0][:2]] = int(fields[3])
            except (serial.SerialException, OSError) as err:
                print(f"serial error: {err} — reconnecting in 5 s", file=sys.stderr)
                time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        track.close()
    print(f"logged {n_fixes} fixes")

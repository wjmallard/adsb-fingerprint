"""Log GPS fixes from the USB puck to daily track files.

One CSV per UTC day under the configured tracks directory, one row per
valid GGA fix (1 Hz). host_utc is the join key against messages.captured_at
(both are host-clock times); gps_utc rides along to audit host clock drift.
The device path changes with the physical USB port, so gps.port may be a
glob — re-resolved on every reconnect.

Display: a full-screen curses dashboard on a terminal (state, satellite
roster, recent events); --plain (or a terminal curses can't drive) falls
back to a one-line self-rewriting status; redirected output gets plain
periodic heartbeat prints.
"""

import argparse
import csv
import curses
import os
import sys
import time
from collections import deque
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
DRAW_EVERY_SECONDS = 0.5
NO_DATA_AFTER_SECONDS = 10

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


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


class GpsState:
    """Everything parsed from the NMEA stream, pruned as it goes stale."""

    def __init__(self, now):
        self.no_fix_since = now
        self.last_sentence = now
        self.last_row = None
        self.n_fixes = 0
        self.counts = {}
        self.sats = {}
        self.used = {}
        self.fix_mode = ""
        self.fix_seen = 0.0
        self.pdop = ""
        self.rmc_date = None
        self.rmc_speed = ""
        self.rmc_track = ""

    def feed(self, fields, now):
        """Digest one checksum-valid sentence; return a CSV row on a fix."""
        self.last_sentence = now
        talker = fields[0][:2]
        sentence = fields[0][2:]
        if sentence == "RMC" and len(fields) > 9:
            if fields[2] == "A":
                self.rmc_date = fields[9]
                self.rmc_speed = fields[7]
                self.rmc_track = fields[8]
        elif sentence == "GGA" and len(fields) > 9:
            if fields[6] in ("", "0") or not fields[2] or not fields[4]:
                if self.no_fix_since is None:
                    self.no_fix_since = now
                return None
            self.no_fix_since = None
            host_utc = datetime.now(timezone.utc).isoformat()
            gps_utc = ""
            if self.rmc_date and fields[1]:
                gps_utc = (
                    f"20{self.rmc_date[4:6]}-{self.rmc_date[2:4]}-{self.rmc_date[0:2]}"
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
                self.rmc_speed,
                self.rmc_track,
            ]
            self.last_row = row
            self.n_fixes += 1
            return row
        elif sentence == "GSV" and len(fields) > 3:
            # One GSV group per talker (constellation); field 3 counts that
            # constellation's satellites in view, then blocks of
            # prn/elevation/azimuth/snr describe each one.
            if fields[3].isdigit():
                self.counts[talker] = (int(fields[3]), now)
            for i in range(4, len(fields) - 3, 4):
                prn, elev, azim, snr = fields[i : i + 4]
                if prn.isdigit():
                    self.sats.setdefault(talker, {})[int(prn)] = (
                        elev,
                        azim,
                        int(snr) if snr.isdigit() else None,
                        now,
                    )
        elif sentence == "GSA" and len(fields) >= 18:
            # Active-satellite list: 2D/3D mode, the SVs used in the
            # solution, and PDOP.
            self.fix_mode = {"2": "2D", "3": "3D"}.get(fields[2], "")
            self.fix_seen = now
            self.pdop = fields[15]
            for prn in fields[3:15]:
                if prn.isdigit():
                    self.used[int(prn)] = now
        return None

    def prune(self, now):
        for talker in list(self.sats):
            fresh = {
                prn: sat
                for prn, sat in self.sats[talker].items()
                if now - sat[3] < NO_DATA_AFTER_SECONDS
            }
            if fresh:
                self.sats[talker] = fresh
            else:
                del self.sats[talker]
        self.counts = {
            talker: entry
            for talker, entry in self.counts.items()
            if now - entry[1] < NO_DATA_AFTER_SECONDS
        }
        self.used = {
            prn: seen
            for prn, seen in self.used.items()
            if now - seen < NO_DATA_AFTER_SECONDS
        }

    def in_view(self, now):
        return sum(count for count, seen in self.counts.values())


class StatusLine:
    """One self-rewriting terminal line; inert when stdout is not a TTY.

    Clipped to the terminal width on every redraw (width is re-measured
    each time, so resizes self-heal) — a line that spills past the edge
    would wrap and scroll instead of rewriting in place.
    """

    def __init__(self):
        self.enabled = sys.stdout.isatty()
        self.active = False

    def update(self, state, color, rest):
        if not self.enabled:
            return
        try:
            width = os.get_terminal_size(sys.stdout.fileno()).columns
        except OSError:
            width = 80
        if width <= 0:
            width = 80
        plain = (state + rest)[: max(10, width - 1)]
        text = color + plain[: len(state)] + RESET + plain[len(state):]
        print(f"\r{text}\033[K", end="", flush=True)
        self.active = True

    def println(self, text, file=sys.stdout):
        """Print a regular line without leaving status-line debris."""
        if self.active:
            print("\r\033[K", end="", flush=True)
            self.active = False
        print(text, file=file)


def format_status(have_fix, last_row, n_fixes, in_view, no_fix_seconds, no_data_seconds):
    """Return (state word, its color, rest of the line) as plain text."""
    if no_data_seconds >= NO_DATA_AFTER_SECONDS:
        elapsed = int(no_data_seconds)
        return (
            "no data from device",
            RED,
            f"  {elapsed // 60}m{elapsed % 60:02d}s",
        )
    if have_fix and last_row:
        clock = (last_row[1] or last_row[0])[11:19]
        sats_used = str(int(last_row[5])) if last_row[5].isdigit() else "?"
        return (
            "fix",
            GREEN,
            f" {clock}Z"
            f"  {last_row[2]}, {last_row[3]}"
            f"  {last_row[7] or '?'}m"
            f"  {sats_used}/{in_view} sats"
            f"  hdop {last_row[6] or '?'}"
            f"  logged {n_fixes}",
        )
    state, color = ("no fix yet", YELLOW) if n_fixes == 0 else ("fix lost", RED)
    elapsed = int(no_fix_seconds)
    return (
        state,
        color,
        f"  {in_view} satellites in view"
        f"  {elapsed // 60}m{elapsed % 60:02d}s",
    )


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


class Dashboard:
    """Full-screen curses dashboard: state, satellite roster, events."""

    NAMES = {
        "GA": "GAL",
        "GB": "BDS",
        "GI": "NavIC",
        "GL": "GLO",
        "GN": "GNSS",
        "GP": "GPS",
        "GQ": "QZSS",
        "QZ": "QZSS",
    }

    def __init__(self, port_pattern):
        self.screen = curses.initscr()
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.noecho()
            curses.cbreak()
            curses.curs_set(0)
            self.screen.keypad(True)
            self.screen.nodelay(True)
        except curses.error:
            curses.endwin()
            raise
        self.port_pattern = port_pattern
        self.events = deque(maxlen=20)
        self.started = time.monotonic()

    def event(self, text):
        self.events.append((datetime.now().strftime("%H:%M:%S"), str(text)))

    def close(self):
        try:
            self.screen.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.endwin()
        except curses.error:
            pass

    def draw(self, gps, port, now):
        while True:
            key = self.screen.getch()
            if key == -1:
                break
            if key == curses.KEY_RESIZE:
                curses.update_lines_cols()
        self.screen.erase()
        height, width = self.screen.getmaxyx()

        def put(y, x, text, attr=0):
            if 0 <= y < height and 0 <= x < width - 1:
                try:
                    self.screen.addnstr(y, x, text, width - x - 1, attr)
                except curses.error:
                    pass

        bold = curses.A_BOLD
        dim = curses.A_DIM
        green = curses.color_pair(1) | bold
        yellow = curses.color_pair(2) | bold
        red = curses.color_pair(3) | bold

        uptime = int(now - self.started)
        put(0, 0, f"adsb-gps-log  {port or self.port_pattern}", dim)
        put(0, max(0, width - 10), f"up {uptime // 3600}h{uptime % 3600 // 60:02d}m", dim)

        no_data = now - gps.last_sentence
        have_fix = gps.no_fix_since is None
        fresh_gsa = now - gps.fix_seen < NO_DATA_AFTER_SECONDS
        if port is None:
            state, attr = "no device", red
            detail = "replug the puck — retrying every 5 s"
        elif no_data >= NO_DATA_AFTER_SECONDS:
            elapsed = int(no_data)
            state, attr = "no data from device", red
            detail = f"{elapsed // 60}m{elapsed % 60:02d}s — dead boot? replug the puck"
        elif have_fix and gps.last_row:
            mode = f" {gps.fix_mode}" if gps.fix_mode and fresh_gsa else ""
            state, attr = f"fix{mode}", green
            detail = f"{(gps.last_row[1] or gps.last_row[0])[11:19]}Z"
        else:
            state, attr = ("no fix yet", yellow) if gps.n_fixes == 0 else ("fix lost", red)
            elapsed = int(now - gps.no_fix_since)
            detail = f"{elapsed // 60}m{elapsed % 60:02d}s"

        put(2, 0, state, attr)
        put(2, len(state) + 2, detail)
        put(2, max(0, width - 16), f"logged {gps.n_fixes}")

        row = gps.last_row
        if row is None:
            put(3, 0, "position -", dim)
        elif have_fix:
            put(3, 0, f"{row[2]}, {row[3]}   {row[7] or '-'} m")
            put(4, 0, f"speed {row[8] or '-'} kt   course {row[9] or '-'}")
            dop = f"hdop {row[6] or '-'}"
            if gps.pdop and fresh_gsa:
                dop += f"   pdop {gps.pdop}"
            put(4, max(0, width - len(dop) - 1), dop)
        else:
            put(3, 0, f"last fix {(row[1] or row[0])[11:19]}Z  {row[2]}, {row[3]}", dim)

        put(6, 0, "satellites", bold)
        header = f"{gps.in_view(now)} in view"
        if have_fix and row is not None and row[5].isdigit():
            header = f"{int(row[5])} used / " + header
        put(6, 12, header)

        # A used PRN is marked only when exactly one constellation reports
        # it in view — GSA lists bare PRNs, so a cross-constellation
        # collision would be a guess.
        seen_in = {}
        for talker, svs in gps.sats.items():
            for prn in svs:
                seen_in[prn] = seen_in.get(prn, 0) + 1

        y = 7
        for talker in sorted(gps.sats):
            if y >= height - 1:
                break
            put(y, 0, f"{self.NAMES.get(talker, talker):<6}", dim)
            x = 7
            ranked = sorted(
                gps.sats[talker].items(),
                key=lambda item: -1 if item[1][2] is None else item[1][2],
                reverse=True,
            )
            for prn, (elev, azim, snr, seen) in ranked:
                token = f"{prn:02d}:{snr if snr is not None else '-'}"
                used = prn in gps.used and seen_in.get(prn) == 1
                put(y, x, token, green if used else 0)
                x += len(token) + 2
                if x >= width:
                    break
            y += 1

        room = height - (y + 2)
        if room >= 1 and self.events:
            put(y + 1, 0, "events", bold)
            for i, (stamp, text) in enumerate(list(self.events)[-room:]):
                put(y + 2 + i, 0, f"{stamp}  {text}", dim)

        self.screen.refresh()


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
        "--plain",
        action="store_true",
        help="One-line status instead of the full-screen dashboard.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every fix as it is logged (plain or non-TTY output only).",
    )
    args = parser.parse_args()

    config.GPS_TRACK_DIR.mkdir(parents=True, exist_ok=True)
    dash = None
    if sys.stdout.isatty() and not args.plain:
        # setupterm raises (initscr would exit the process) on a broken
        # TERM; a terminal with no cursor addressing gets the plain line.
        try:
            curses.setupterm()
            if curses.tigetstr("cup") is not None:
                dash = Dashboard(args.port)
        except Exception:
            dash = None
    status = StatusLine()

    def emit(text, err=False):
        if dash:
            dash.event(text)
        else:
            status.println(text, file=sys.stderr if err else sys.stdout)

    track = TrackWriter(config.GPS_TRACK_DIR, announce=emit)
    started = time.monotonic()
    gps = GpsState(started)
    last_status = started
    last_draw = 0.0
    device_missing = False

    def expired():
        return args.seconds and time.monotonic() - started >= args.seconds

    try:
        while not expired():
            port = resolve_port(args.port)
            if port is None:
                if dash:
                    if not device_missing:
                        dash.event(f"no device matches {args.port!r}")
                    dash.draw(gps, None, time.monotonic())
                else:
                    status.println(
                        f"no device matches {args.port!r} — retrying in 5 s",
                        file=sys.stderr,
                    )
                device_missing = True
                time.sleep(5)
                continue
            device_missing = False
            try:
                with serial.Serial(port, 9600, timeout=2) as ser:
                    emit(f"logging {port}")
                    gps.last_sentence = time.monotonic()
                    while not expired():
                        now = time.monotonic()
                        gps.prune(now)
                        have_fix = gps.no_fix_since is None
                        no_data = now - gps.last_sentence
                        if dash:
                            if now - last_draw >= DRAW_EVERY_SECONDS:
                                dash.draw(gps, port, now)
                                last_draw = now
                        elif status.enabled:
                            if now - last_draw >= DRAW_EVERY_SECONDS:
                                status.update(
                                    *format_status(
                                        have_fix,
                                        gps.last_row,
                                        gps.n_fixes,
                                        gps.in_view(now),
                                        0 if have_fix else now - gps.no_fix_since,
                                        no_data,
                                    )
                                )
                                last_draw = now
                        elif now - last_status >= (
                            STATUS_EVERY_SECONDS
                            if have_fix and no_data < NO_DATA_AFTER_SECONDS
                            else NO_FIX_EVERY_SECONDS
                        ):
                            if no_data >= NO_DATA_AFTER_SECONDS:
                                elapsed = int(no_data)
                                print(
                                    f"no data from device — "
                                    f"{elapsed // 60}m{elapsed % 60:02d}s elapsed"
                                )
                            elif have_fix:
                                print(
                                    f"{gps.n_fixes} fixes logged, "
                                    f"latest {gps.last_row[2]}, {gps.last_row[3]}"
                                )
                            else:
                                state = "no fix yet" if gps.n_fixes == 0 else "fix lost"
                                elapsed = int(now - gps.no_fix_since)
                                print(
                                    f"{state} — {gps.in_view(now)} satellites in view, "
                                    f"{elapsed // 60}m{elapsed % 60:02d}s elapsed"
                                )
                            last_status = now
                        line = ser.readline().decode(errors="replace").strip()
                        if not checksum_ok(line):
                            continue
                        fields = line[1:].partition("*")[0].split(",")
                        row = gps.feed(fields, time.monotonic())
                        if row:
                            track.write(row)
                            if args.verbose and not dash:
                                status.println(",".join(row))
            except (serial.SerialException, OSError) as err:
                emit(f"serial error: {err} — reconnecting in 5 s", err=True)
                if dash:
                    dash.draw(gps, port, time.monotonic())
                time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        track.close()
        if dash:
            dash.close()
    if dash:
        print(f"logged {gps.n_fixes} fixes")
    else:
        status.println(f"logged {gps.n_fixes} fixes")

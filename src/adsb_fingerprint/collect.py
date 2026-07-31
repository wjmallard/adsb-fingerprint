"""adsb-collect: stream from the RTL-SDR, detect Mode S in real time, and
persist only the per-message IQ snippets — never the full raw stream.

Two threads: a reader that calls read_samples() back-to-back so the device is
never starved (which is what causes USB drops), and the main processor that
detects messages, accumulates snippets + index rows in RAM, and flushes them to
the per-session snippet store and the messages index in batches (keeping disk
and commit latency out of the hot path). A small rolling tail is carried across
blocks so a message straddling a boundary survives.

A per-aircraft cap (config.yaml) keeps the first N messages per ICAO per time
window — micro-clusters: back-to-back replicates that pin the per-message
noise floor, in windows spaced apart for fresh geometry — so one nearby
aircraft can't dominate the dataset (or the disk). Ident messages (TC 1-4,
the callsign) are exempt, with their own small allowance per window.

Each session directory gets a session.yaml documenting the radio settings,
receiver location, and sampling policy in effect (plus outcome counters on
exit) — collection design is a covariate in any cross-session analysis.
"""

import argparse
import queue
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import (
    asin,
    cos,
    radians,
    sin,
    sqrt,
)
from statistics import median

import numpy as np
import yaml
from pyModeS import PipeDecoder

from adsb_fingerprint import config, db, modes
from adsb_fingerprint.capture import TUNERS, _apply_gain

BLOCK = 131072           # samples per read (~55 ms at 2.4 MSPS); multiple of 512
CARRY = 1024             # rolling-buffer tail carried between blocks
WINDOW = 384             # samples stored per message (288-sample message + margin)
QUEUE_BLOCKS = 256       # ~256 MB cap; reader only drops (counted) if this fills
FLUSH_S = 1              # seconds between DB/snippet flushes (keeps the index near-real-time)
PRINT_S = 30             # seconds between progress lines
IDENT_ALLOWANCE = 1      # cap-exempt ident (TC 1-4, callsign) messages per aircraft per window

STATION_MIN_AIRCRAFT = 3     # fewest contributing aircraft worth a median
STATION_FREEZE_AIRCRAFT = 20 # stop refining the estimate at this many
STATION_AGREE_KM = 100.0     # config receiver within this of the estimate is trusted

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
        vertical_rate,
        receiver_latitude,
        receiver_longitude
    )
    from stdin
"""

# Rows copied before the station resolves carry null receiver columns; one
# update at resolution time brings them onto the session's stamp.
STATION_BACKFILL_SQL = """
    update messages
    set receiver_latitude = %(latitude)s,
        receiver_longitude = %(longitude)s
    where session = %(session)s
    and receiver_latitude is null
"""

# Every decoded message — capped or not — refreshes live_state, so the
# coalesces keep a field's last known value when this flush has none, and
# msg_heard accumulates across flushes and collector restarts.
STATE_UPSERT_SQL = """
    insert into live_state (
        icao,
        last_seen,
        msg_heard,
        rssi_db,
        callsign,
        callsign_at,
        latitude,
        longitude,
        altitude_ft,
        position_at,
        ground_speed,
        track,
        vertical_rate,
        velocity_at
    )
    values (
        %(icao)s,
        %(last_seen)s,
        %(msg_heard)s,
        %(rssi_db)s,
        %(callsign)s,
        %(callsign_at)s,
        %(latitude)s,
        %(longitude)s,
        %(altitude_ft)s,
        %(position_at)s,
        %(ground_speed)s,
        %(track)s,
        %(vertical_rate)s,
        %(velocity_at)s
    )
    on conflict (icao) do update set
        last_seen = excluded.last_seen,
        msg_heard = live_state.msg_heard + excluded.msg_heard,
        rssi_db = excluded.rssi_db,
        callsign = coalesce(excluded.callsign, live_state.callsign),
        callsign_at = coalesce(excluded.callsign_at, live_state.callsign_at),
        latitude = coalesce(excluded.latitude, live_state.latitude),
        longitude = coalesce(excluded.longitude, live_state.longitude),
        altitude_ft = coalesce(excluded.altitude_ft, live_state.altitude_ft),
        position_at = coalesce(excluded.position_at, live_state.position_at),
        ground_speed = coalesce(excluded.ground_speed, live_state.ground_speed),
        track = coalesce(excluded.track, live_state.track),
        vertical_rate = coalesce(excluded.vertical_rate, live_state.vertical_rate),
        velocity_at = coalesce(excluded.velocity_at, live_state.velocity_at)
"""

STATE_FIELDS = (
    "rssi_db",
    "callsign",
    "callsign_at",
    "latitude",
    "longitude",
    "altitude_ft",
    "position_at",
    "ground_speed",
    "track",
    "vertical_rate",
    "velocity_at",
)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two WGS84 points, in km."""
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = (
        sin((lat2 - lat1) / 2) ** 2
        + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    )
    return 2.0 * 6371.0088 * asin(sqrt(a))


class StationEstimator:
    """Receiver position inferred from the traffic itself.

    Each aircraft's first resolved fix enters the pool (everything heard is
    within radio range of the antenna); the per-axis median is the station
    estimate — good to tens of km, plenty inside CPR's 180 NM local-decode
    validity and enough to catch a surveyed config gone stale after a move.
    One estimate per session: a session is assumed stationary (mobile
    collection gets true per-message positions from adsb-geotag instead).
    """

    def __init__(self):
        self.first_fixes = {}

    def add(self, icao, latitude, longitude):
        if icao not in self.first_fixes and len(self.first_fixes) < STATION_FREEZE_AIRCRAFT:
            self.first_fixes[icao] = (latitude, longitude)

    @property
    def n(self):
        return len(self.first_fixes)

    def estimate(self):
        fixes = self.first_fixes.values()
        return (
            median(lat for lat, _ in fixes),
            median(lon for _, lon in fixes),
        )


def resolve_station(estimator):
    """Pick the session's receiver stamp: (latitude, longitude, source).

    The surveyed config position wins while it agrees with the traffic
    estimate; a config left over from a previous location shows up as
    disagreement and is ignored with a warning.
    """
    est_lat, est_lon = estimator.estimate()
    print(
        f"\nstation estimate: {est_lat:.3f}, {est_lon:.3f} "
        f"(median of {estimator.n} aircraft)"
    )
    if config.RECEIVER_LAT is None:
        return est_lat, est_lon, "estimated"
    offset_km = _haversine_km(
        config.RECEIVER_LAT,
        config.RECEIVER_LON,
        est_lat,
        est_lon,
    )
    if offset_km <= STATION_AGREE_KM:
        print(f"  config receiver agrees ({offset_km:.0f} km) — using its surveyed position")
        return config.RECEIVER_LAT, config.RECEIVER_LON, "config"
    print(
        f"  config receiver is {offset_km:.0f} km away — stale (moved since it was "
        f"set?) — ignoring it"
    )
    return est_lat, est_lon, "estimated"


def track_state(state, dirty, icao, stamp, msg):
    """Fold one decoded message into the per-airframe latest-state buffer."""
    entry = state.setdefault(icao, {"msg_heard": 0})
    entry["msg_heard"] += 1
    entry["last_seen"] = stamp
    entry["rssi_db"] = msg["rssi_db"]
    if msg.get("callsign"):
        entry["callsign"] = msg["callsign"]
        entry["callsign_at"] = stamp
    if msg.get("latitude") is not None:
        entry["latitude"] = msg["latitude"]
        entry["longitude"] = msg["longitude"]
        entry["position_at"] = stamp
    if msg.get("altitude_ft") is not None:
        entry["altitude_ft"] = msg["altitude_ft"]
    if msg.get("ground_speed") is not None or msg.get("vertical_rate") is not None:
        for key in ("ground_speed", "track", "vertical_rate"):
            if msg.get(key) is not None:
                entry[key] = msg[key]
        entry["velocity_at"] = stamp
    dirty.add(icao)


def _reader(device, block_queue, stop, stats):
    while not stop.is_set():
        try:
            block = device.read_samples(BLOCK).astype(np.complex64)
        except Exception:
            break
        stats["blocks_read"] += 1
        try:
            block_queue.put(block, timeout=1.0)
        except queue.Full:
            stats["blocks_dropped"] += 1


def collect(
    seconds,
    session,
    device_index,
    center_freq_hz,
    sample_rate_hz,
    gain,
    ppm,
    max_per_aircraft,
    window_seconds,
):
    from rtlsdr import RtlSdr

    device = RtlSdr(device_index)
    device.sample_rate = sample_rate_hz
    device.center_freq = center_freq_hz
    if ppm != 0:
        device.freq_correction = ppm
    applied_gain = _apply_gain(device, gain)
    tuner = TUNERS.get(device.get_tuner_type(), "unknown")

    # Stateful decoder: positions resolve from each aircraft's own CPR frame
    # pairs, so collection needs no receiver location — it works anywhere.
    # The station's own position falls out of the traffic instead.
    pipe = PipeDecoder()
    estimator = StationEstimator()
    station = None
    started = datetime.now(timezone.utc)
    session = session or started.strftime("%Y%m%dT%H%M%SZ")
    session_dir = config.CAPTURE_DIR / session
    session_dir.mkdir(parents=True, exist_ok=True)
    snippet_path = session_dir / "snippets.cf32"
    snippet_rel = str(snippet_path.relative_to(config.CAPTURE_DIR))
    meta_path = session_dir / "session.yaml"      # not .json — adsb-index globs captures/**/*.json
    guard = int(round(modes.PREAMBLE_US * sample_rate_hz / 1e6))
    is_tty = sys.stdout.isatty()

    def write_meta(outcome=None):
        meta = {
            "session": session,
            "tool": "adsb-collect",
            "started_at": started.isoformat(),
            "radio": {
                "center_freq_hz": int(center_freq_hz),
                "sample_rate_hz": int(sample_rate_hz),
                "gain": "auto" if applied_gain is None else float(gain),
                "actual_gain_db": applied_gain,
                "freq_correction_ppm": int(ppm),
                "tuner": tuner,
                "device_index": int(device_index),
            },
            "receiver": (
                {
                    "latitude": station[0],
                    "longitude": station[1],
                    "source": station[2],
                }
                if station
                else {
                    "source": "unresolved",
                }
            ),
            "policy": {
                "max_per_aircraft": max_per_aircraft,
                "window_seconds": window_seconds,
                "ident_allowance": IDENT_ALLOWANCE,
            },
        }
        if outcome is not None:
            meta["outcome"] = outcome
        meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    dur = f"{seconds:.0f}s" if seconds else "until Ctrl-C"
    cap = (
        f"cap {max_per_aircraft}/aircraft/{window_seconds}s (+{IDENT_ALLOWANCE} ident)"
        if max_per_aircraft > 0
        else "no cap"
    )
    print(
        f"collecting {dur} @ {center_freq_hz / 1e6:.3f} MHz, "
        f"{sample_rate_hz / 1e6:.3f} MSPS, "
        f"gain={'auto' if applied_gain is None else applied_gain}, "
        f"tuner={tuner} · reader+processor · {cap}"
    )
    print(f"  snippets -> {snippet_path}")
    print(f"  meta     -> {meta_path}")
    write_meta()

    device.read_samples(1024)                     # discard first read (PLL settle)

    block_queue = queue.Queue(maxsize=QUEUE_BLOCKS)
    stop = threading.Event()
    stats = {"blocks_read": 0, "blocks_dropped": 0, "maxq": 0}
    reader = threading.Thread(
        target=_reader,
        args=(device, block_queue, stop, stats),
        daemon=True,
    )

    carry = np.empty(0, dtype=np.complex64)
    abs_base = 0
    last_abs = -guard
    n_detected = 0
    n_stored = 0
    n_capped = 0
    seen = set()
    window_counts = defaultdict(int)
    ident_counts = defaultdict(int)
    current_bucket = -1
    snips = []
    rows = []
    state = {}
    dirty = set()
    start = time.perf_counter()
    last_flush = start
    last_print = start

    def flush(store, conn):
        if snips:
            np.concatenate(snips).tofile(store)
            snips.clear()
        if rows:
            with conn.cursor() as cur, cur.copy(COPY_SQL) as copy:
                for row in rows:
                    copy.write_row(row)
            rows.clear()
        if dirty:
            with conn.cursor() as cur:
                cur.executemany(
                    STATE_UPSERT_SQL,
                    [
                        {
                            "icao": icao,
                            "last_seen": state[icao]["last_seen"],
                            "msg_heard": state[icao]["msg_heard"],
                            **{key: state[icao].get(key) for key in STATE_FIELDS},
                        }
                        for icao in dirty
                    ],
                )
            for icao in dirty:
                state[icao]["msg_heard"] = 0
            dirty.clear()
        conn.commit()

    with open(snippet_path, "wb") as store, db.connect() as conn:
        conn.execute("delete from messages where capture_file = %(f)s", {"f": snippet_rel})
        conn.commit()
        reader.start()
        try:
            while not (seconds and time.perf_counter() - start >= seconds):
                try:
                    block = block_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                stats["maxq"] = max(stats["maxq"], block_queue.qsize())
                buf = np.concatenate([carry, block])
                for msg in modes.detect_messages(buf, sample_rate_hz):
                    offset = msg["sample_offset"]
                    if offset + WINDOW > len(buf):
                        continue
                    abs_off = abs_base + offset
                    if abs_off <= last_abs + guard:
                        continue
                    last_abs = abs_off
                    icao = msg["icao"]
                    n_detected += 1
                    stamp = started + timedelta(seconds=abs_off / sample_rate_hz)
                    msg.update(
                        modes.decode_message(
                            pipe,
                            msg["hex"],
                            abs_off / sample_rate_hz,
                        ),
                    )
                    if msg.get("latitude") is not None:
                        estimator.add(icao, msg["latitude"], msg["longitude"])
                    track_state(state, dirty, icao, stamp, msg)
                    if icao not in seen:
                        seen.add(icao)
                        print(f"\n  ✈  {icao}   (aircraft #{len(seen)})")
                    elif is_tty:
                        sys.stdout.write(".")
                        sys.stdout.flush()

                    if max_per_aircraft > 0:
                        bucket = int(abs_off / sample_rate_hz / window_seconds)
                        if bucket != current_bucket:
                            current_bucket = bucket
                            window_counts.clear()
                            ident_counts.clear()
                        if 1 <= msg["type_code"] <= 4 and ident_counts[icao] < IDENT_ALLOWANCE:
                            ident_counts[icao] += 1
                        elif window_counts[icao] >= max_per_aircraft:
                            n_capped += 1
                            continue
                        else:
                            window_counts[icao] += 1

                    snips.append(buf[offset : offset + WINDOW].copy())
                    rows.append(
                        (
                            snippet_rel,
                            n_stored * WINDOW,
                            WINDOW,
                            stamp,
                            session,
                            msg["df"],
                            icao,
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
                            station[0] if station else None,
                            station[1] if station else None,
                        )
                    )
                    n_stored += 1
                carry = buf[-CARRY:]
                abs_base += len(buf) - len(carry)
                now = time.perf_counter()
                if now - last_flush >= FLUSH_S:
                    flush(store, conn)
                    last_flush = now
                    if station is None and estimator.n >= STATION_FREEZE_AIRCRAFT:
                        station = resolve_station(estimator)
                        conn.execute(
                            STATION_BACKFILL_SQL,
                            {
                                "latitude": station[0],
                                "longitude": station[1],
                                "session": session,
                            },
                        )
                        conn.commit()
                if now - last_print >= PRINT_S:
                    elapsed = now - start
                    print(
                        f"\n[{elapsed:4.0f}s] det {n_detected} · kept {n_stored} · "
                        f"capped {n_capped} · {n_detected / elapsed:.1f}/s · "
                        f"{len(seen)} aircraft  |  read {stats['blocks_read']} blks  "
                        f"dropped {stats['blocks_dropped']}  qmax {stats['maxq']}"
                    )
                    last_print = now
        except KeyboardInterrupt:
            print("\ninterrupted.")
        finally:
            stop.set()
            reader.join(timeout=2.0)
            flush(store, conn)
            device.close()

        # A session too short to hit the freeze count still gets a stamp
        # from whatever pool accumulated — unless even that is too thin,
        # in which case rows honestly stay "position unknown".
        if station is None and estimator.n >= STATION_MIN_AIRCRAFT:
            station = resolve_station(estimator)
        if station is None:
            print(
                f"\nstation: unresolved ({estimator.n} aircraft with positions, "
                f"need {STATION_MIN_AIRCRAFT}) — receiver stamps left null"
            )
        else:
            conn.execute(
                STATION_BACKFILL_SQL,
                {
                    "latitude": station[0],
                    "longitude": station[1],
                    "session": session,
                },
            )
            conn.commit()

    elapsed = time.perf_counter() - start
    expected_blocks = elapsed * sample_rate_hz / BLOCK
    print(
        f"\ncollected {n_detected} messages ({n_detected / elapsed:.1f}/s) in {elapsed:.0f}s, "
        f"{len(seen)} aircraft — kept {n_stored}, capped {n_capped}"
    )
    print(
        f"  reader: {stats['blocks_read']} blocks read (~{expected_blocks:.0f} expected), "
        f"{stats['blocks_dropped']} queue-full drops, qmax {stats['maxq']}/{QUEUE_BLOCKS}"
    )
    if snippet_path.exists():
        print(f"  snippet store: {snippet_path.stat().st_size / 1e6:.1f} MB")
    write_meta(
        outcome={
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(elapsed, 1),
            "detected": n_detected,
            "stored": n_stored,
            "capped": n_capped,
            "aircraft": len(seen),
            "blocks_read": stats["blocks_read"],
            "blocks_dropped": stats["blocks_dropped"],
            "qmax": stats["maxq"],
        },
    )
    return n_stored


def main():
    parser = argparse.ArgumentParser(
        description="Stream the RTL-SDR and persist only detected Mode S message snippets.",
    )
    parser.add_argument("--seconds", type=float, default=None, help="Duration (default: until Ctrl-C).")
    parser.add_argument("--session", default=None, help="Session label (default: start timestamp).")
    parser.add_argument("--gain", default=config.RADIO_GAIN, help='Tuner gain in dB or "auto".')
    parser.add_argument("--freq-hz", type=float, default=config.CENTER_FREQ_HZ, help="Center frequency (Hz).")
    parser.add_argument("--sample-rate-hz", type=float, default=config.SAMPLE_RATE_HZ, help="Sample rate (Hz).")
    parser.add_argument("--ppm", type=int, default=config.FREQ_CORRECTION_PPM, help="Frequency correction (PPM).")
    parser.add_argument("--device-index", type=int, default=0, help="RTL-SDR device index.")
    parser.add_argument(
        "--max-per-aircraft",
        type=int,
        default=config.COLLECT_MAX_PER_AIRCRAFT,
        help="Max messages kept per aircraft per window (0 = no cap; default: config.yaml).",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=config.COLLECT_WINDOW_SECONDS,
        help="Cap window length in seconds (default: config.yaml).",
    )
    args = parser.parse_args()

    collect(
        seconds=args.seconds,
        session=args.session,
        device_index=args.device_index,
        center_freq_hz=args.freq_hz,
        sample_rate_hz=args.sample_rate_hz,
        gain=args.gain,
        ppm=args.ppm,
        max_per_aircraft=args.max_per_aircraft,
        window_seconds=args.window_seconds,
    )


if __name__ == "__main__":
    main()

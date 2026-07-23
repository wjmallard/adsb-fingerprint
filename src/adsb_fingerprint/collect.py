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

import numpy as np
import yaml

from adsb_fingerprint import config, db, modes
from adsb_fingerprint.capture import TUNERS, _apply_gain

BLOCK = 131072           # samples per read (~55 ms at 2.4 MSPS); multiple of 512
CARRY = 1024             # rolling-buffer tail carried between blocks
WINDOW = 384             # samples stored per message (288-sample message + margin)
QUEUE_BLOCKS = 256       # ~256 MB cap; reader only drops (counted) if this fills
FLUSH_S = 1              # seconds between DB/snippet flushes (keeps the index near-real-time)
PRINT_S = 30             # seconds between progress lines
IDENT_ALLOWANCE = 1      # cap-exempt ident (TC 1-4, callsign) messages per aircraft per window

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

    reference = (config.RECEIVER_LAT, config.RECEIVER_LON)
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
            "receiver": {
                "latitude": config.RECEIVER_LAT,
                "longitude": config.RECEIVER_LON,
            },
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
            conn.commit()
            rows.clear()

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
                for msg in modes.detect_messages(buf, sample_rate_hz, reference=reference):
                    offset = msg["sample_offset"]
                    if offset + WINDOW > len(buf):
                        continue
                    abs_off = abs_base + offset
                    if abs_off <= last_abs + guard:
                        continue
                    last_abs = abs_off
                    icao = msg["icao"]
                    n_detected += 1
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
                            started + timedelta(seconds=abs_off / sample_rate_hz),
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
                        )
                    )
                    n_stored += 1
                carry = buf[-CARRY:]
                abs_base += len(buf) - len(carry)
                now = time.perf_counter()
                if now - last_flush >= FLUSH_S:
                    flush(store, conn)
                    last_flush = now
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

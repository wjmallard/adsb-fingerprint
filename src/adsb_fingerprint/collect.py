"""adsb-collect: stream from the RTL-SDR, detect Mode S in real time, and
persist only the per-message IQ snippets — never the full raw stream.

Two threads: a reader that calls read_samples() back-to-back so the device is
never starved (which is what causes USB drops), and the main processor that
detects messages, accumulates snippets + index rows in RAM, and flushes them to
the per-session snippet store and the messages index in batches (keeping disk
and commit latency out of the hot path). A small rolling tail is carried across
blocks so a message straddling a boundary survives.
"""

import argparse
import queue
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from adsb_fingerprint import config, db, modes
from adsb_fingerprint.capture import TUNERS, _apply_gain

BLOCK = 131072           # samples per read (~55 ms at 2.4 MSPS); multiple of 512
CARRY = 1024             # rolling-buffer tail carried between blocks
WINDOW = 384             # samples stored per message (288-sample message + margin)
QUEUE_BLOCKS = 256       # ~256 MB cap; reader only drops (counted) if this fills
CHECKPOINT_S = 15        # seconds between progress/flush checkpoints

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
        rssi_db
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
):
    from rtlsdr import RtlSdr

    device = RtlSdr(device_index)
    device.sample_rate = sample_rate_hz
    device.center_freq = center_freq_hz
    if ppm != 0:
        device.freq_correction = ppm
    applied_gain = _apply_gain(device, gain)
    tuner = TUNERS.get(device.get_tuner_type(), "unknown")

    started = datetime.now(timezone.utc)
    session = session or started.strftime("%Y%m%dT%H%M%SZ")
    session_dir = config.CAPTURE_DIR / session
    session_dir.mkdir(parents=True, exist_ok=True)
    snippet_path = session_dir / "snippets.cf32"
    snippet_rel = str(snippet_path.relative_to(config.CAPTURE_DIR))
    guard = int(round(modes.PREAMBLE_US * sample_rate_hz / 1e6))
    is_tty = sys.stdout.isatty()

    dur = f"{seconds:.0f}s" if seconds else "until Ctrl-C"
    print(
        f"collecting {dur} @ {center_freq_hz / 1e6:.3f} MHz, "
        f"{sample_rate_hz / 1e6:.3f} MSPS, "
        f"gain={'auto' if applied_gain is None else applied_gain}, "
        f"tuner={tuner} (reader + processor threads)"
    )
    print(f"  snippets -> {snippet_path}")

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
    n_msgs = 0
    seen = set()
    snips = []
    rows = []
    start = time.perf_counter()
    last_ckpt = start

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
                for msg in modes.detect_messages(buf, sample_rate_hz):
                    offset = msg["sample_offset"]
                    if offset + WINDOW > len(buf):
                        continue
                    abs_off = abs_base + offset
                    if abs_off <= last_abs + guard:
                        continue
                    last_abs = abs_off
                    snips.append(buf[offset : offset + WINDOW].copy())
                    rows.append(
                        (
                            snippet_rel,
                            n_msgs * WINDOW,
                            WINDOW,
                            started + timedelta(seconds=abs_off / sample_rate_hz),
                            session,
                            msg["df"],
                            msg["icao"],
                            msg["type_code"],
                            True,
                            msg["rssi_db"],
                        )
                    )
                    n_msgs += 1
                    if msg["icao"] not in seen:
                        seen.add(msg["icao"])
                        print(f"\n  ✈  {msg['icao']}   (aircraft #{len(seen)})")
                    elif is_tty:
                        sys.stdout.write(".")
                        sys.stdout.flush()
                carry = buf[-CARRY:]
                abs_base += len(buf) - len(carry)
                now = time.perf_counter()
                if now - last_ckpt >= CHECKPOINT_S:
                    flush(store, conn)
                    elapsed = now - start
                    print(
                        f"\n[{elapsed:4.0f}s] {n_msgs} msgs  {n_msgs / elapsed:.1f}/s  "
                        f"{len(seen)} aircraft  |  read {stats['blocks_read']} blks  "
                        f"dropped {stats['blocks_dropped']}  qmax {stats['maxq']}"
                    )
                    last_ckpt = now
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
        f"\ncollected {n_msgs} messages in {elapsed:.0f}s "
        f"({n_msgs / elapsed:.1f} msg/s), {len(seen)} aircraft"
    )
    print(
        f"  reader: {stats['blocks_read']} blocks read (~{expected_blocks:.0f} expected), "
        f"{stats['blocks_dropped']} queue-full drops, qmax {stats['maxq']}/{QUEUE_BLOCKS}"
    )
    if snippet_path.exists():
        print(f"  snippet store: {snippet_path.stat().st_size / 1e6:.1f} MB")
    return n_msgs


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
    args = parser.parse_args()

    collect(
        seconds=args.seconds,
        session=args.session,
        device_index=args.device_index,
        center_freq_hz=args.freq_hz,
        sample_rate_hz=args.sample_rate_hz,
        gain=args.gain,
        ppm=args.ppm,
    )


if __name__ == "__main__":
    main()

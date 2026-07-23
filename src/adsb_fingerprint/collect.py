"""adsb-collect: stream from the RTL-SDR, detect Mode S messages in real time,
and persist only the per-message IQ snippets — never the full raw stream.

A small rolling tail is carried between reads so a message straddling a block
boundary survives; the modes detector runs on each block, and every validated
message's IQ window is appended to a per-session snippet store plus a row in
the messages index. A multi-hour run costs a few hundred MB, not 100+ GB.

This is the single-thread (synchronous) version. If it drops samples, the
reader and processor split into two threads.
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from adsb_fingerprint import config, db, modes
from adsb_fingerprint.capture import TUNERS, _apply_gain

BLOCK = 131072           # samples per read (~55 ms at 2.4 MSPS); multiple of 512
CARRY = 1024             # rolling-buffer tail carried between blocks
WINDOW = 384             # samples stored per message (288-sample message + margin)

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

    dur = f"{seconds:.0f}s" if seconds else "until Ctrl-C"
    print(
        f"collecting {dur} @ {center_freq_hz / 1e6:.3f} MHz, "
        f"{sample_rate_hz / 1e6:.3f} MSPS, "
        f"gain={'auto' if applied_gain is None else applied_gain}, tuner={tuner}"
    )
    print(f"  snippets -> {snippet_path}")

    device.read_samples(1024)                     # discard first read (PLL settle)

    carry = np.empty(0, dtype=np.complex64)
    abs_base = 0
    last_abs = -guard
    n_msgs = 0
    proc_time = 0.0
    start_perf = time.perf_counter()
    deadline = start_perf + seconds if seconds else None

    with open(snippet_path, "wb") as store, db.connect() as conn:
        conn.execute("delete from messages where capture_file = %(f)s", {"f": snippet_rel})
        conn.commit()
        try:
            while deadline is None or time.perf_counter() < deadline:
                block = device.read_samples(BLOCK).astype(np.complex64)
                mark = time.perf_counter()
                buf = np.concatenate([carry, block])
                rows = []
                for msg in modes.detect_messages(buf, sample_rate_hz):
                    offset = msg["sample_offset"]
                    if offset + WINDOW > len(buf):
                        continue                  # straddles the end; defer to next block
                    abs_off = abs_base + offset
                    if abs_off <= last_abs + guard:
                        continue                  # already emitted in the overlap
                    last_abs = abs_off
                    buf[offset : offset + WINDOW].astype(np.complex64).tofile(store)
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
                if rows:
                    with conn.cursor() as cur, cur.copy(COPY_SQL) as copy:
                        for row in rows:
                            copy.write_row(row)
                    conn.commit()
                carry = buf[-CARRY:]
                abs_base += len(buf) - len(carry)
                proc_time += time.perf_counter() - mark
        except KeyboardInterrupt:
            print("\ninterrupted — finalizing.")
        finally:
            device.close()

    elapsed = time.perf_counter() - start_perf
    duty = proc_time / elapsed if elapsed else 0.0
    rate = n_msgs / elapsed if elapsed else 0.0
    print(
        f"collected {n_msgs} messages in {elapsed:.1f}s "
        f"({rate:.1f} msg/s), processing duty {duty * 100:.0f}%"
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

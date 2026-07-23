"""adsb-capture: stream RTL-SDR IQ into timestamped .cf32 capture files."""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

from adsb_fingerprint import config

TUNERS = {
    1: "E4000",
    2: "FC0012",
    3: "FC0013",
    4: "FC2580",
    5: "R820T",
    6: "R828D",
}

CHUNK_SECONDS = 0.5


def _round_512(n):
    # librtlsdr reads must be a multiple of 512 samples.
    return ((int(n) + 511) // 512) * 512


def _apply_gain(device, gain):
    # "auto" enables the tuner AGC. Otherwise switch to manual mode (required
    # for the gain to take effect) and snap to the nearest supported value.
    # Returns the applied gain in dB, or None for auto. The pyrtlsdr `gain`
    # getter reads 0 on this librtlsdr build, so we return what we applied
    # rather than trust a readback.
    if isinstance(gain, str) and gain.lower() == "auto":
        device.set_manual_gain_enabled(False)
        return None
    device.set_manual_gain_enabled(True)
    valid_db = [g / 10 for g in device.get_gains()]
    applied = min(valid_db, key=lambda g: abs(g - float(gain)))
    device.gain = applied
    return applied


def capture(
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

    n_written = 0
    try:
        tuner = TUNERS.get(device.get_tuner_type(), "unknown")
        started_at = datetime.now(timezone.utc)
        stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
        session = session or stamp

        session_dir = config.CAPTURE_DIR / session
        session_dir.mkdir(parents=True, exist_ok=True)
        data_path = session_dir / f"{stamp}.cf32"
        meta_path = session_dir / f"{stamp}.json"

        dur_label = f"{seconds:.1f}s" if seconds else "until Ctrl-C"
        print(
            f"capturing {dur_label} @ {center_freq_hz / 1e6:.3f} MHz, "
            f"{sample_rate_hz / 1e6:.3f} MSPS, gain={gain}, tuner={tuner}"
        )
        print(f"  -> {data_path}")

        device.read_samples(1024)  # discard first read (PLL settle)

        target = _round_512(sample_rate_hz * seconds) if seconds else None
        chunk = _round_512(sample_rate_hz * CHUNK_SECONDS)

        bar = tqdm(
            total=target,
            unit="S",
            unit_scale=True,
            desc="capturing",
        )
        try:
            with open(data_path, "wb") as f:
                while target is None or n_written < target:
                    n = chunk if target is None else min(chunk, target - n_written)
                    samples = device.read_samples(_round_512(n))
                    samples.astype(np.complex64).tofile(f)
                    n_written += len(samples)
                    bar.update(len(samples))
        except KeyboardInterrupt:
            print("\ninterrupted — finalizing.")
        finally:
            bar.close()

        meta = {
            "session": session,
            "captured_at": started_at.isoformat(),
            "center_freq_hz": int(center_freq_hz),
            "sample_rate_hz": int(sample_rate_hz),
            "gain": "auto" if applied_gain is None else float(gain),
            "actual_gain_db": applied_gain,
            "freq_correction_ppm": int(ppm),
            "tuner": tuner,
            "device_index": int(device_index),
            "sample_format": "cf32",
            "n_samples": int(n_written),
            "duration_s": n_written / sample_rate_hz,
            "data_file": data_path.name,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    finally:
        device.close()

    size_mb = data_path.stat().st_size / 1e6
    print(
        f"wrote {n_written} samples ({n_written / sample_rate_hz:.3f} s, {size_mb:.1f} MB)\n"
        f"  data: {data_path}\n"
        f"  meta: {meta_path}"
    )
    return data_path


def main():
    parser = argparse.ArgumentParser(
        description="Stream RTL-SDR IQ into timestamped .cf32 capture files.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Capture duration in seconds (default: until Ctrl-C).",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session label / subdirectory (default: start timestamp).",
    )
    parser.add_argument(
        "--gain",
        default=config.RADIO_GAIN,
        help='Tuner gain in dB, or "auto" (default: config.yaml).',
    )
    parser.add_argument(
        "--freq-hz",
        type=float,
        default=config.CENTER_FREQ_HZ,
        help="Center frequency in Hz (default: config.yaml).",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=config.SAMPLE_RATE_HZ,
        help="Sample rate in Hz (default: config.yaml).",
    )
    parser.add_argument(
        "--ppm",
        type=int,
        default=config.FREQ_CORRECTION_PPM,
        help="Frequency correction in PPM (default: config.yaml).",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="RTL-SDR device index (default: 0).",
    )
    args = parser.parse_args()

    capture(
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

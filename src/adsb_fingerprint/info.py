"""adsb-info: print RTL-SDR device details and the configured capture target."""

from adsb_fingerprint import config

TUNERS = {
    1: "E4000",
    2: "FC0012",
    3: "FC0013",
    4: "FC2580",
    5: "R820T",
    6: "R828D",
}


def main():
    from rtlsdr import RtlSdr

    try:
        device = RtlSdr(0)
    except Exception as e:
        print(f"Error opening RTL-SDR device: {e}")
        raise SystemExit(1)

    try:
        tuner = TUNERS.get(device.get_tuner_type(), "unknown")
        gains = device.get_gains()  # tenths of dB
        print("RTL-SDR device 0")
        print(f"  tuner       : {tuner}")
        print(f"  valid gains : {', '.join(f'{g / 10:.1f}' for g in gains)} dB")
        print(f"  gain range  : {min(gains) / 10:.1f} - {max(gains) / 10:.1f} dB")
    finally:
        device.close()

    print()
    print("Configured capture target (config.yaml):")
    print(f"  center freq : {config.CENTER_FREQ_HZ / 1e6:.3f} MHz")
    print(f"  sample rate : {config.SAMPLE_RATE_HZ / 1e6:.3f} MSPS")
    print(f"  gain        : {config.RADIO_GAIN}")
    print(f"  freq corr   : {config.FREQ_CORRECTION_PPM} ppm")
    print(f"  captures -> : {config.CAPTURE_DIR}")


if __name__ == "__main__":
    main()

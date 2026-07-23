"""Cheap deterministic physical-layer features from stored IQ snippets.

Each feature is a scalar measured with a few numpy ops (no learning), chosen to
be interpretable and mostly alignment-invariant: carrier frequency offset from
the lag-1 phase of the data block, preamble pulse amplitude ratio and pulse-1
to pulse-4 spacing from an oversampled envelope, first-pulse rise time, and
in-band SNR against the post-message tail. Snippets are stored with the
preamble starting at sample 0 (see adsb-collect), so all geometry derives from
the sample rate alone. Unmeasurable values come back as NaN, never raise.
"""

import numpy as np

from adsb_fingerprint import modes

_OSR = 8                 # envelope oversampling, matches modes._demod's grid

# Preamble pulse centres relative to pulse 1's centre, in microseconds
# (pulses occupy 0-0.5, 1.0-1.5, 3.5-4.0, 4.5-5.0 us).
_PULSE_OFFSET_US = (0.0, 1.0, 3.5, 4.5)
_PULSE_HALF_US = 0.3

FEATURES = [
    "cfo_hz",
    "p14_spacing_ns",
    "preamble_ratio_db",
    "rise_ns",
    "snr_db",
]

_NANS = {name: float("nan") for name in FEATURES}


def _pulse_stats(env, fine_us, center_us):
    mask = (fine_us >= center_us - _PULSE_HALF_US) & (fine_us <= center_us + _PULSE_HALF_US)
    w = env[mask]
    total = w.sum()
    if not len(w) or total <= 0:
        return float("nan"), float("nan")
    return float(w.mean()), float((fine_us[mask] * w).sum() / total)


def _rise_ns(env, fine_us, lo_us, hi_us):
    seg = np.where((fine_us >= lo_us) & (fine_us <= hi_us))[0]
    if not len(seg):
        return float("nan")
    e = env[seg]
    t = fine_us[seg]
    peak_idx = int(np.argmax(e))
    peak = e[peak_idx]
    if peak <= 0 or peak_idx == 0:
        return float("nan")
    lo, hi = 0.1 * peak, 0.9 * peak
    below = np.where(e[:peak_idx] < lo)[0]
    if not len(below):
        return float("nan")                     # no quiet baseline before the edge
    i = below[-1]                               # last sample under 10% ...
    j = i + np.argmax(e[i : peak_idx + 1] >= hi)  # ... first at/over 90% after it
    if e[j] < hi:
        return float("nan")
    t10 = np.interp(lo, e[i : i + 2], t[i : i + 2])
    t90 = np.interp(hi, e[j - 1 : j + 1], t[j - 1 : j + 1])
    return float(t90 - t10) * 1e3


def extract(snippet, sample_rate):
    """Return {feature_name: float} for one message snippet (NaN = unmeasurable)."""
    spb = sample_rate / 1e6
    msg_end = int(round(modes.MESSAGE_US * spb))
    if len(snippet) < msg_end:
        return dict(_NANS)

    # The RTL-SDR's zero-IF DC spur sits in-band at 1090 MHz; estimate it from
    # the quietest quarter of samples (PPM off-chips + tail) and remove it.
    x = snippet.astype(np.complex128)
    mag0 = np.abs(x)
    x -= x[mag0 <= np.quantile(mag0, 0.25)].mean()
    mag = np.abs(x)

    out = dict(_NANS)

    # CFO: the aperture is what buys precision — lag-1 products span 0.42 us
    # and scatter tens of kHz, so the estimate is refined in stages, each
    # unambiguous far beyond the previous stage's residual: a coarse lag-1
    # seed (+/- fs/2), then 1 us segment phasors compared at 1/8/32 us gaps
    # (+/- 500/62/16 kHz), then a slip-safe weighted phase regression over the
    # full 112 us data block (~ +/-0.3 kHz per message on real captures).
    d0 = int(round(modes.PREAMBLE_US * spb))
    data = x[d0:msg_end]
    dmag = mag[d0:msg_end]
    level = np.quantile(dmag, 0.75)               # ~ typical pulse-top level
    on = dmag >= 0.5 * level
    pair = on[:-1] & on[1:]
    strong = np.flatnonzero(dmag >= 0.7 * level)
    if pair.sum() >= 8 and len(strong) >= 16:
        acc = np.sum(data[1:][pair] * np.conj(data[:-1][pair]))
        if abs(acc) > 0:
            cfo = float(np.angle(acc)) * sample_rate / (2.0 * np.pi)
            t = strong / sample_rate
            seg = (strong / spb).astype(int)      # 1 us (= 1 bit) segments
            u = np.zeros(int(seg[-1]) + 1, dtype=complex)
            for gap_us in (1, 8, 32):
                if len(u) <= gap_us:
                    continue
                u[:] = 0
                np.add.at(u, seg, data[strong] * np.exp(-2j * np.pi * cfo * t))
                step = (u[gap_us:] * np.conj(u[:-gap_us])).sum()
                if abs(step) > 0:
                    cfo += float(np.angle(step)) / (2.0 * np.pi * gap_us * 1e-6)
            z = data[strong] * np.exp(-2j * np.pi * cfo * t)
            slope = np.polyfit(
                t,
                np.unwrap(np.angle(z)),
                1,
                w=dmag[strong],
            )[0]
            out["cfo_hz"] = cfo + slope / (2.0 * np.pi)

    # Oversampled preamble envelope, self-aligned on pulse 1's centroid so the
    # detector's +/-1-sample alignment slop cancels out of the shape features.
    fine_us = np.arange(0.0, modes.PREAMBLE_US, 1.0 / _OSR)
    env = np.interp(fine_us * spb, np.arange(len(mag)), mag)
    head = env[fine_us <= 0.9]
    if head.sum() > 0:
        c1_us = float((fine_us[: len(head)] * head).sum() / head.sum())
        amps, cents = zip(
            *(
                _pulse_stats(env, fine_us, c1_us + offset)
                for offset in _PULSE_OFFSET_US
            )
        )
        early, late = amps[0] + amps[1], amps[2] + amps[3]
        if early > 0 and late > 0:
            out["preamble_ratio_db"] = 20.0 * np.log10(late / early)
        out["p14_spacing_ns"] = (cents[3] - cents[0]) * 1e3
        # Rise time is measured on pulse 3: it is the only early pulse with a
        # guaranteed quiet microsecond before it (the snippet starts mid-rise
        # of pulse 1, so pulse 1 has no baseline to cross from).
        out["rise_ns"] = _rise_ns(
            env,
            fine_us,
            c1_us + 2.4,
            c1_us + 3.85,
        )

    # SNR: message power over the quiet tail after the data block.
    tail = mag[msg_end + 4 :]
    if len(tail) >= 16:
        signal = float(np.mean(mag[:msg_end] ** 2))
        noise = float(np.quantile(tail**2, 0.25))
        if signal > 0 and noise > 0:
            out["snr_db"] = 10.0 * np.log10(signal / noise)

    return out

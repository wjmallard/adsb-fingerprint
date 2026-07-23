"""Mode S / ADS-B detection and demodulation from complex IQ.

Finds DF17/DF18 extended-squitter preambles by magnitude correlation,
PPM-demodulates each candidate on an oversampled grid, and keeps the ones that
pass the pyModeS CRC. The CRC is the real filter, so the preamble detector is
deliberately permissive.
"""

import numpy as np
from pyModeS.util import crc, df, hex2bin, icao

PREAMBLE_US = 8          # preamble length before the data block starts
MESSAGE_US = 120         # 8 us preamble + 112 us data
DATA_BITS = 112

# Preamble pulse centres (high) and gap/quiet points (low), in microseconds.
_PULSE_US = np.array([0.0, 1.0, 3.5, 4.5])
_GAP_US = np.array([0.5, 1.5, 2.0, 2.5, 3.0, 4.0, 5.5, 6.5, 7.5])

_OSR = 8                 # per-candidate oversampling for clean bit timing


def _bits_to_hex(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return "%028X" % value


def _demod(mag, start, spb):
    span = int(round(MESSAGE_US * spb)) + 4
    window = mag[start : start + span]
    if len(window) < span:
        return None
    fine_us = np.arange(0.0, MESSAGE_US, 1.0 / _OSR)
    m = np.interp(fine_us * spb, np.arange(len(window)), window)
    data0 = PREAMBLE_US * _OSR
    seg = m[data0 : data0 + DATA_BITS * _OSR].reshape(DATA_BITS, _OSR)
    half = _OSR // 2
    bits = seg[:, :half].sum(axis=1) > seg[:, half:].sum(axis=1)
    return _bits_to_hex(bits)


def detect_messages(iq, sample_rate):
    """Yield validated DF17/DF18 messages found in complex IQ.

    Each result is a dict: sample_offset, df, icao, type_code, hex, rssi_db.
    """
    spb = sample_rate / 1e6                       # samples per microsecond (per bit)
    mag = np.abs(iq).astype(np.float32)

    pulse_idx = np.round(_PULSE_US * spb).astype(int)
    gap_idx = np.round(_GAP_US * spb).astype(int)
    limit = len(mag) - int(round(MESSAGE_US * spb)) - 4
    if limit <= 0:
        return

    pulse_mean = sum(mag[p : p + limit] for p in pulse_idx) / len(pulse_idx)
    gap_mean = sum(mag[g : g + limit] for g in gap_idx) / len(gap_idx)
    noise = float(np.median(mag))
    score = pulse_mean - gap_mean
    mask = (pulse_mean > 2.0 * gap_mean) & (pulse_mean > 3.0 * noise)

    peaks = (
        np.where(
            mask[1:-1]
            & (score[1:-1] > score[:-2])
            & (score[1:-1] >= score[2:])
        )[0]
        + 1
    )

    guard = int(round(PREAMBLE_US * spb))
    last = -guard
    for peak in peaks:
        if peak - last < guard:
            continue
        for start in (peak - 1, peak, peak + 1):     # ±1 sample alignment search
            if start < 0:
                continue
            msg = _demod(mag, int(start), spb)
            if msg is None:
                continue
            downlink = df(msg)
            if downlink in (17, 18) and crc(msg) == 0:
                last = peak
                level = float(mag[int(start) + pulse_idx].mean())
                yield {
                    "sample_offset": int(start),
                    "df": downlink,
                    "icao": icao(msg).upper(),
                    "type_code": int(hex2bin(msg)[32:37], 2),
                    "hex": msg,
                    "rssi_db": 20.0 * np.log10(level) if level > 0 else None,
                }
                break

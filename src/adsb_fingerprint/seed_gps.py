"""Seed the GPS puck with host time and position for a faster first fix.

A restart after travel is the slow case: the puck's battery-backed RAM
still holds the previous city, so it predicts Doppler for the wrong sky
and burns minutes discovering that. This pushes UBX-MGA aiding over the
serial port — UTC from the NTP-disciplined host clock (immune to the
displayed timezone) and a rough position from Location Services — so
the search starts on the sky actually overhead. It does not shortcut
the per-satellite ephemeris decode, so the payoff is a warm start, not
a hot one; without a Location Services fix it still seeds time alone.

adsb-log-gps runs the same seeding at every connect (--no-seed skips),
so the standalone command is for aiding without starting the logger —
and doubles as a liveness probe: a puck that answers nothing is in the
known dead-boot mode (replug). It cannot run while the logger holds the
port; serial opens are exclusive.
"""

import argparse
import struct
import sys
import time
from datetime import datetime, timezone
from glob import glob

import serial

from adsb_fingerprint import config, location

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

TONE_COLORS = {
    "good": GREEN,
    "warn": YELLOW,
    "bad": RED,
}

UBX_SYNC = b"\xb5\x62"
CLS_ACK = 0x05
CLS_CFG = 0x06
CLS_MGA = 0x13
ID_ACK_NAK = 0x00
ID_ACK_ACK = 0x01
ID_CFG_NAVX5 = 0x23
ID_MGA_INI = 0x40
ID_MGA_ACK = 0x60
TYPE_INI_POS_LLH = 0x01
TYPE_INI_TIME_UTC = 0x10

# GPS-UTC leap seconds as of 2026; a stale value is absorbed by the
# claimed time accuracy.
LEAP_SECONDS = 18
TIME_ACCURACY_S = 2

# Never claim better than this: Wi-Fi accuracy is optimistic, altitude
# goes unstated, and sub-kilometer precision buys the search nothing.
POSITION_ACCURACY_FLOOR_M = 1000.0

ACK_TIMEOUT_S = 3.0

REJECT_REASONS = {
    1: "receiver has no time",
    2: "message version unsupported",
    3: "payload size mismatch",
    4: "could not be stored",
    5: "receiver not ready",
    6: "message type unknown",
}


def paint(text, color):
    return f"{color}{text}{RESET}" if sys.stdout.isatty() else text


def resolve_port(pattern):
    matches = sorted(glob(pattern))
    return matches[0] if matches else None


def fletcher(data):
    """The UBX 8-bit Fletcher checksum."""
    ck_a = ck_b = 0
    for byte in data:
        ck_a = (ck_a + byte) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return bytes((ck_a, ck_b))


def ubx_frame(msg_class, msg_id, payload):
    body = bytes((msg_class, msg_id)) + len(payload).to_bytes(2, "little") + payload
    return UBX_SYNC + body + fletcher(body)


def cfg_navx5_ack_aiding():
    """CFG-NAVX5 applying only the ackAiding bit, in RAM — the puck then
    acknowledges each aiding message, and the setting reverts at power
    cycle."""
    payload = bytearray(40)
    struct.pack_into(
        "<HH",
        payload,
        0,
        2,       # message version
        0x0400,  # mask1: apply the ackAid field only
    )
    payload[17] = 1
    return ubx_frame(CLS_CFG, ID_CFG_NAVX5, bytes(payload))


def mga_ini_time_utc(now):
    """MGA-INI-TIME_UTC: UTC that is valid on this message's arrival."""
    payload = struct.pack(
        "<BBBbHBBBBBBIHHI",
        TYPE_INI_TIME_UTC,
        0,                        # message version
        0,                        # reference: arrival of this message
        LEAP_SECONDS,
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second,
        0,                        # reserved
        now.microsecond * 1000,   # nanosecond part
        TIME_ACCURACY_S,
        0,                        # reserved
        0,                        # accuracy fraction, ns
    )
    return ubx_frame(CLS_MGA, ID_MGA_INI, payload)


def mga_ini_pos_llh(latitude, longitude, accuracy_m):
    """MGA-INI-POS_LLH: a rough position with its standard deviation."""
    payload = struct.pack(
        "<BBBBiiiI",
        TYPE_INI_POS_LLH,
        0,                        # message version
        0,                        # reserved
        0,                        # reserved
        round(latitude * 1e7),
        round(longitude * 1e7),
        0,                        # altitude, cm — unknown, inside the floor
        round(max(accuracy_m, POSITION_ACCURACY_FLOOR_M) * 100),
    )
    return ubx_frame(CLS_MGA, ID_MGA_INI, payload)


def send(ser, frame):
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()


def read_frames(ser, deadline):
    """Checksum-valid UBX (class, id, payload) frames until the deadline,
    skipping the interleaved NMEA stream."""
    while time.monotonic() < deadline:
        if ser.read(1) != UBX_SYNC[:1]:
            continue
        if ser.read(1) != UBX_SYNC[1:]:
            continue
        header = ser.read(4)
        if len(header) < 4:
            continue
        length = int.from_bytes(header[2:], "little")
        if length > 512:
            continue               # desync, not a real frame header
        rest = ser.read(length + 2)
        if len(rest) < length + 2:
            continue
        if fletcher(header + rest[:length]) != rest[length:]:
            continue
        yield header[0], header[1], rest[:length]


def await_cfg_ack(ser):
    """True/False for ACK/NAK of CFG-NAVX5; None if the puck stays silent."""
    for msg_class, msg_id, payload in read_frames(ser, time.monotonic() + ACK_TIMEOUT_S):
        if msg_class == CLS_ACK and payload[:2] == bytes((CLS_CFG, ID_CFG_NAVX5)):
            return msg_id == ID_ACK_ACK
    return None


def await_mga_ack(ser, ini_type):
    """(accepted, reason) from the matching aiding ack; (None, _) on silence."""
    for msg_class, msg_id, payload in read_frames(ser, time.monotonic() + ACK_TIMEOUT_S):
        if (
            msg_class == CLS_MGA
            and msg_id == ID_MGA_ACK
            and len(payload) >= 8
            and payload[3] == ID_MGA_INI
            and payload[4] == ini_type
        ):
            reason = REJECT_REASONS.get(payload[2], f"info code {payload[2]}")
            return payload[0] == 1, reason
    return None, "no acknowledgment"


def outcome(ser, acks, ini_type):
    """(verdict, tone) for one aiding message just sent."""
    if not acks:
        return "sent (unconfirmed)", "warn"
    accepted, reason = await_mga_ack(ser, ini_type)
    if accepted:
        return "accepted", "good"
    if accepted is None:
        return "no acknowledgment", "bad"
    return f"rejected: {reason}", "bad"


def summarize(items):
    for name, _, verdict, tone in items:
        if tone == "bad":
            if verdict == "no acknowledgment":
                return f"seed: {name} unacknowledged"
            return f"seed: {name} {verdict}"
    verdicts = [verdict for _, _, verdict, _ in items]
    if "sent (unconfirmed)" in verdicts:
        return "seeded (unconfirmed)"
    if "seeded time only" in verdicts:
        return "seeded time only (no Location Services fix)"
    return "seeded time + position"


def seed(ser, place):
    """Push aiding over an already-open port; (summary, items).

    summary is a one-line outcome for event logs; items are
    (name, detail, verdict, tone) rows for line-by-line display. Only
    serial errors raise — a silent or refusing puck just reports as
    such.
    """
    send(ser, cfg_navx5_ack_aiding())
    acks = await_cfg_ack(ser)
    if acks is None:
        # One retry: a flaky USB moment can eat the first exchange even
        # while NMEA keeps flowing (seen live 2026-08-05).
        send(ser, cfg_navx5_ack_aiding())
        acks = await_cfg_ack(ser)
    if acks is None:
        return (
            "seed: no UBX reply — dead boot? replug",
            [("puck", "no UBX reply", "dead boot? replug and rerun", "bad")],
        )
    items = []
    now = datetime.now(timezone.utc)
    send(ser, mga_ini_time_utc(now))
    detail = f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')} ±{TIME_ACCURACY_S} s"
    items.append(("time", detail, *outcome(ser, acks, TYPE_INI_TIME_UTC)))
    if place is None:
        items.append(("position", "no Location Services fix", "seeded time only", "warn"))
    else:
        latitude, longitude, accuracy = place
        send(ser, mga_ini_pos_llh(latitude, longitude, accuracy))
        claimed_km = max(accuracy, POSITION_ACCURACY_FLOOR_M) / 1000
        detail = f"{latitude:.5f}, {longitude:.5f} ±{claimed_km:.0f} km"
        items.append(("position", detail, *outcome(ser, acks, TYPE_INI_POS_LLH)))
    return summarize(items), items


def main():
    parser = argparse.ArgumentParser(
        description="Seed the GPS puck with host time and a Location Services "
        "position for a faster first fix.",
    )
    parser.add_argument(
        "--port",
        default=config.GPS_PORT,
        help="Serial device path or glob (default: config gps.port).",
    )
    args = parser.parse_args()

    port = resolve_port(args.port)
    if port is None:
        raise SystemExit(f"no device matches {args.port!r}")

    place = location.current_location()

    try:
        ser = serial.Serial(port, 9600, timeout=0.2)
    except (serial.SerialException, OSError) as err:
        hint = (
            " — adsb-log-gps holds the port and already seeds at connect"
            if "busy" in str(err).lower()
            else ""
        )
        raise SystemExit(f"cannot open {port}: {err}{hint}")

    print(f"{'port':<10}{port}")
    with ser:
        _, items = seed(ser, place)
    for name, detail, verdict, tone in items:
        print(f"{name:<10}{detail} — " + paint(verdict, TONE_COLORS[tone]))
    if any(tone == "bad" for _, _, _, tone in items):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

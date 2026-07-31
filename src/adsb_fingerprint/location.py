"""Query macOS Location Services for the laptop's own position.

Wi-Fi positioning works indoors where a GPS puck sees no sky, and the
laptop running the collector is always present — so this is the live
first guess for the station's whereabouts, cross-checked against the
traffic estimate before anything final trusts it (see adsb-collect).

Authorization is one-time per terminal app but awkward: recent macOS
often shows no dialog for CLI tools — the request just registers the
terminal in System Settings -> Privacy & Security -> Location Services
for the user to enable by hand. `adsb-location` (main below) exists to
walk that dance: it fires the request, narrates the authorization state
as it changes, and prints the fix the moment the grant lands. The
collector's own startup query stays quiet and bounded instead.

macOS-only by nature: pyobjc is a darwin-marked dependency, and its
import here is deliberately lazy so the rest of the package works
without it.
"""

import argparse
import time

FIX_TIMEOUT_S = 8.0      # give CoreLocation this long to produce a fix
FIX_MAX_AGE_S = 60.0     # ignore cached fixes older than this
PROMPT_TIMEOUT_S = 30.0  # give a human this long to answer the one-time prompt


def current_location(timeout=FIX_TIMEOUT_S):
    """The laptop's (latitude, longitude, accuracy_m), or None.

    None covers every unavailable case alike: pyobjc not installed,
    location services off or denied for this terminal, or no fresh fix
    inside the timeout.
    """
    try:
        from CoreLocation import (
            CLLocationManager,
            kCLAuthorizationStatusDenied,
            kCLAuthorizationStatusNotDetermined,
            kCLAuthorizationStatusRestricted,
        )
        from Foundation import (
            NSDate,
            NSRunLoop,
        )
    except ImportError:
        return None

    manager = CLLocationManager.alloc().init()
    prompted = False
    if manager.authorizationStatus() == kCLAuthorizationStatusNotDetermined:
        # The explicit request is what raises the one-time permission
        # prompt (attributed to the hosting terminal app) — merely
        # starting updates no longer does on current macOS. Some macOS
        # versions never show a dialog for CLI tools at all and only
        # register the terminal in the Location Services list, so say
        # what's happening instead of appearing to hang.
        manager.requestWhenInUseAuthorization()
        prompted = True
        print(
            "location services: asking macOS for permission — approve the "
            "prompt if one appears; if none does, enable this terminal under "
            "System Settings -> Privacy & Security -> Location Services "
            f"(waiting up to {PROMPT_TIMEOUT_S:.0f} s)"
        )
    manager.startUpdatingLocation()
    try:
        deadline = time.monotonic() + (PROMPT_TIMEOUT_S if prompted else timeout)
        runloop = NSRunLoop.currentRunLoop()
        while time.monotonic() < deadline:
            status = manager.authorizationStatus()
            if status in (
                kCLAuthorizationStatusDenied,
                kCLAuthorizationStatusRestricted,
            ):
                return None
            if prompted and status != kCLAuthorizationStatusNotDetermined:
                # Prompt answered — the fix itself gets the normal window.
                deadline = time.monotonic() + timeout
                prompted = False
            fix = manager.location()
            if fix is not None and fix.horizontalAccuracy() >= 0:
                age = -fix.timestamp().timeIntervalSinceNow()
                if age <= FIX_MAX_AGE_S:
                    coordinate = fix.coordinate()
                    return (
                        float(coordinate.latitude),
                        float(coordinate.longitude),
                        float(fix.horizontalAccuracy()),
                    )
            runloop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.2))
    finally:
        manager.stopUpdatingLocation()
    return None


def _status_name(status):
    return {
        0: "undetermined",
        1: "restricted",
        2: "denied",
        3: "authorized (always)",
        4: "authorized",
    }.get(status, f"unknown ({status})")


def main():
    parser = argparse.ArgumentParser(
        description="Authorize and test macOS Location Services for the station position.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Give up after this many seconds (default: 300; Ctrl-C stops sooner).",
    )
    args = parser.parse_args()

    try:
        from CoreLocation import (
            CLLocationManager,
            kCLAuthorizationStatusDenied,
            kCLAuthorizationStatusNotDetermined,
            kCLAuthorizationStatusRestricted,
        )
        from Foundation import (
            NSDate,
            NSRunLoop,
        )
    except ImportError:
        raise SystemExit("pyobjc is not installed — Location Services is macOS-only")

    manager = CLLocationManager.alloc().init()
    status = manager.authorizationStatus()
    print(f"authorization: {_status_name(status)}")
    if status == kCLAuthorizationStatusNotDetermined:
        manager.requestWhenInUseAuthorization()
        print(
            "requested — approve the macOS dialog if one appears. On recent\n"
            "macOS none does: this terminal instead appears under\n"
            "System Settings -> Privacy & Security -> Location Services\n"
            "within a few seconds — toggle it on there. Waiting..."
        )
    manager.startUpdatingLocation()

    runloop = NSRunLoop.currentRunLoop()
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            now_status = manager.authorizationStatus()
            if now_status != status:
                status = now_status
                print(f"authorization: {_status_name(status)}")
            if status in (
                kCLAuthorizationStatusDenied,
                kCLAuthorizationStatusRestricted,
            ):
                raise SystemExit(
                    "denied — enable this terminal under System Settings -> "
                    "Privacy & Security -> Location Services, then rerun"
                )
            fix = manager.location()
            if fix is not None and fix.horizontalAccuracy() >= 0:
                age = -fix.timestamp().timeIntervalSinceNow()
                stale = f", {age:.0f} s old" if age > FIX_MAX_AGE_S else ""
                coordinate = fix.coordinate()
                print(
                    f"fix: {coordinate.latitude:.5f}, {coordinate.longitude:.5f} "
                    f"(±{fix.horizontalAccuracy():.0f} m{stale})"
                )
                print("authorized — adsb-collect picks this up at every launch now.")
                return
            runloop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.2))
        raise SystemExit(f"no fix within {args.timeout:.0f} s")
    except KeyboardInterrupt:
        raise SystemExit("\nstopped")
    finally:
        manager.stopUpdatingLocation()


if __name__ == "__main__":
    main()

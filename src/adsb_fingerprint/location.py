"""Query macOS Location Services for the laptop's own position.

Wi-Fi positioning works indoors where a GPS puck sees no sky, and the
laptop running the collector is always present — so this is the live
first guess for the station's whereabouts, cross-checked against the
traffic estimate before anything final trusts it (see adsb-collect).
The first query from a given terminal app raises the macOS permission
prompt; approve it once (System Settings -> Privacy & Security ->
Location Services) and later queries answer immediately.

macOS-only by nature: pyobjc is a darwin-marked dependency, and its
import here is deliberately lazy so the rest of the package works
without it.
"""

import time

FIX_TIMEOUT_S = 8.0      # give CoreLocation this long to produce a fix
FIX_MAX_AGE_S = 60.0     # ignore cached fixes older than this


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
            kCLAuthorizationStatusRestricted,
        )
        from Foundation import (
            NSDate,
            NSRunLoop,
        )
    except ImportError:
        return None

    manager = CLLocationManager.alloc().init()
    # Starting updates is also what raises the one-time permission
    # prompt while authorization is still undetermined.
    manager.startUpdatingLocation()
    try:
        deadline = time.monotonic() + timeout
        runloop = NSRunLoop.currentRunLoop()
        while time.monotonic() < deadline:
            if manager.authorizationStatus() in (
                kCLAuthorizationStatusDenied,
                kCLAuthorizationStatusRestricted,
            ):
                return None
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

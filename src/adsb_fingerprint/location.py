"""Query macOS Location Services for the laptop's own position.

Wi-Fi positioning works indoors where a GPS puck sees no sky, and the
laptop running the collector is always present — so this is a live
first guess for the station's whereabouts, cross-checked against the
traffic estimate before anything final trusts it (see adsb-collect).

The asking is delegated to a tiny Swift helper compiled on first use
(location_helper.swift, Info.plist embedded, ad-hoc signed): macOS
ignores authorization requests from binaries without an embedded usage
description, which an interpreter can never carry — a bare python
request raises no dialog and registers nothing. The helper appears as
"adsb-location" in System Settings -> Privacy & Security -> Location
Services; `adsb-location` (main below) walks the one-time authorization
dance interactively, while the collector's startup query stays quiet
and bounded.

The pyobjc path (darwin-marked dependency, lazily imported) remains as
a fallback for environments without the Swift toolchain where python
itself has somehow been granted access.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

FIX_TIMEOUT_S = 8.0      # give CoreLocation this long to produce a fix
FIX_MAX_AGE_S = 60.0     # ignore cached fixes older than this
PROMPT_TIMEOUT_S = 30.0  # give a human this long to answer the one-time prompt


def _helper_path():
    """Path to the compiled helper, building it on first use (or None).

    Built into the environment's bin directory beside the console
    scripts; rebuilt whenever the source or embedded plist is newer.
    None when the Swift toolchain is unavailable.
    """
    helper = Path(sys.prefix) / "bin" / "adsb-location-helper"
    source = Path(__file__).with_name("location_helper.swift")
    plist = Path(__file__).with_name("location_helper.plist")
    fresh = max(source.stat().st_mtime, plist.stat().st_mtime)
    if helper.exists() and helper.stat().st_mtime >= fresh:
        return helper
    try:
        subprocess.run(
            [
                "xcrun",
                "swiftc",
                "-swift-version", "5",
                "-O",
                str(source),
                "-o", str(helper),
                "-Xlinker", "-sectcreate",
                "-Xlinker", "__TEXT",
                "-Xlinker", "__info_plist",
                "-Xlinker", str(plist),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        subprocess.run(
            ["codesign", "--force", "--sign", "-", str(helper)],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except Exception:
        return None
    return helper


def current_location(timeout=FIX_TIMEOUT_S):
    """The laptop's (latitude, longitude, accuracy_m), or None.

    None covers every unavailable case alike: no way to ask (toolchain
    and pyobjc both missing), location services off or not authorized,
    or no fresh fix inside the timeout.
    """
    helper = _helper_path()
    if helper is not None:
        try:
            result = subprocess.run(
                [str(helper), str(timeout)],
                capture_output=True,
                text=True,
                timeout=max(timeout, PROMPT_TIMEOUT_S) + 15,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        try:
            latitude, longitude, accuracy = (float(v) for v in result.stdout.split())
        except ValueError:
            return None
        return latitude, longitude, accuracy
    return _pyobjc_location(timeout)


def _pyobjc_location(timeout):
    """In-process CoreLocation query — works only if python itself is
    somehow authorized; its requests raise no prompt (no embedded usage
    description), so this is strictly a fallback."""
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
        manager.requestWhenInUseAuthorization()
        prompted = True
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

    helper = _helper_path()
    if helper is None:
        raise SystemExit(
            "could not build the location helper — the Swift toolchain is "
            "needed (xcode-select --install). Without it, python itself "
            "cannot be granted location access."
        )
    print(
        'asking macOS for a fix — approve the dialog if one appears; if none\n'
        'does, enable "adsb-location" under System Settings -> Privacy &\n'
        'Security -> Location Services (the entry exists once the request fires)'
    )
    try:
        result = subprocess.run(
            [str(helper), str(args.timeout)],
            stdout=subprocess.PIPE,
            text=True,
        )
    except KeyboardInterrupt:
        raise SystemExit("\nstopped")
    if result.returncode == 0:
        latitude, longitude, accuracy = (float(v) for v in result.stdout.split())
        print(f"fix: {latitude:.5f}, {longitude:.5f} (±{accuracy:.0f} m)")
        print("authorized — adsb-collect picks this up at every launch now.")
        return
    raise SystemExit("no fix — see the guidance above, then rerun")


if __name__ == "__main__":
    main()

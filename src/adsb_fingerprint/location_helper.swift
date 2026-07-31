// Location helper for adsb-fingerprint.
//
// CoreLocation ignores authorization requests from processes without an
// embedded usage description — which an interpreter can never carry — so
// this tiny binary does the asking. location.py builds it on demand
// (Info.plist embedded via -sectcreate, ad-hoc signed); it appears as
// "adsb-location" in System Settings -> Privacy & Security -> Location
// Services.
//
// Usage: adsb-location-helper [timeout_seconds]
// stdout on success: "<latitude> <longitude> <accuracy_m>"; stderr
// narrates authorization. Exit 0 fix, 1 denied, 2 no fix in time.

import CoreLocation
import Foundation

func note(_ line: String) {
    FileHandle.standardError.write((line + "\n").data(using: .utf8)!)
}

func statusName(_ status: CLAuthorizationStatus) -> String {
    switch status {
    case .notDetermined: return "undetermined"
    case .restricted: return "restricted"
    case .denied: return "denied"
    case .authorizedAlways: return "authorized"
    case .authorizedWhenInUse: return "authorized (when in use)"
    @unknown default: return "unknown"
    }
}

final class Listener: NSObject, CLLocationManagerDelegate {
    let manager = CLLocationManager()
    let timeout: TimeInterval
    var deadline: Date
    var lastStatus: CLAuthorizationStatus?

    init(timeout: TimeInterval) {
        self.timeout = timeout
        // Leave room for a human to answer the one-time permission
        // dialog; an already-authorized run shrinks this on the first
        // status callback below.
        self.deadline = Date(timeIntervalSinceNow: max(timeout, 30))
        super.init()
    }

    func start() {
        manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
        manager.delegate = self   // fires the status callback with current state
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        if status != lastStatus {
            lastStatus = status
            note("authorization: \(statusName(status))")
        }
        switch status {
        case .notDetermined:
            note("requested — approve the dialog if one appears, or enable "
                + "\"adsb-location\" under System Settings -> Privacy & Security "
                + "-> Location Services")
            manager.requestWhenInUseAuthorization()
        case .denied, .restricted:
            exit(1)
        default:
            deadline = Date(timeIntervalSinceNow: timeout)
            manager.startUpdatingLocation()
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let fix = locations.last,
              fix.horizontalAccuracy >= 0,
              -fix.timestamp.timeIntervalSinceNow <= 60 else { return }
        print("\(fix.coordinate.latitude) \(fix.coordinate.longitude) \(fix.horizontalAccuracy)")
        exit(0)
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        note("error: \(error.localizedDescription)")
    }
}

let timeout = CommandLine.arguments.count > 1
    ? (Double(CommandLine.arguments[1]) ?? 8.0)
    : 8.0
let listener = Listener(timeout: timeout)
listener.start()
while Date() < listener.deadline {
    RunLoop.main.run(mode: .default, before: Date(timeIntervalSinceNow: 0.2))
}
note("no fix within the window")
exit(2)

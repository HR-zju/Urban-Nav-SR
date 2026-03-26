"""macOS current-location helpers backed by CoreLocation."""

from __future__ import annotations

from typing import Any
import sys
import time


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("macOS location is only available on Darwin/macOS.")


def get_current_location(timeout_s: float = 10.0) -> dict[str, Any]:
    """Return the current macOS device location using CoreLocation.

    This requires:
    - running on macOS
    - Terminal / host app to have Location Services permission
    - PyObjC CoreLocation packages installed
    """

    _require_macos()

    try:
        from Foundation import NSObject, NSRunLoop, NSDate
        from CoreLocation import CLLocationManager
    except ImportError as exc:
        raise RuntimeError(
            "macOS location support requires PyObjC. "
            "Install the optional extra: pip install -e '.[macos]'"
        ) from exc

    state: dict[str, Any] = {"location": None, "error": None, "authorized": None}

    class LocationDelegate(NSObject):
        def init(self):
            self = super().init()
            return self

        def locationManager_didUpdateLocations_(self, manager, locations):
            if not locations:
                return
            location = locations[-1]
            coordinate = location.coordinate()
            state["location"] = {
                "latitude": float(coordinate.latitude),
                "longitude": float(coordinate.longitude),
                "accuracy_m": float(location.horizontalAccuracy()),
                "altitude_m": float(location.altitude()),
                "timestamp": str(location.timestamp()),
                "source": "corelocation",
                "platform": "macos",
            }

        def locationManager_didFailWithError_(self, manager, error):
            state["error"] = str(error)

        def locationManagerDidChangeAuthorization_(self, manager):
            status = int(manager.authorizationStatus())
            state["authorized"] = status

        def locationManager_didChangeAuthorizationStatus_(self, manager, status):
            state["authorized"] = int(status)

    delegate = LocationDelegate.alloc().init()
    manager = CLLocationManager.alloc().init()
    manager.setDelegate_(delegate)

    request_auth = getattr(manager, "requestWhenInUseAuthorization", None)
    if callable(request_auth):
        request_auth()
    request_location = getattr(manager, "requestLocation", None)
    if callable(request_location):
        request_location()
    else:
        manager.startUpdatingLocation()

    deadline = time.time() + max(timeout_s, 1.0)
    run_loop = NSRunLoop.currentRunLoop()
    while time.time() < deadline:
        run_loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
        if state["location"] is not None:
            stop = getattr(manager, "stopUpdatingLocation", None)
            if callable(stop):
                stop()
            return state["location"]
        if state["error"] is not None:
            break

    stop = getattr(manager, "stopUpdatingLocation", None)
    if callable(stop):
        stop()

    if state["error"]:
        raise RuntimeError(f"CoreLocation failed: {state['error']}")
    raise TimeoutError(f"Timed out after {timeout_s:.1f}s waiting for macOS location.")


__all__ = ["get_current_location"]

import Foundation
import CoreLocation
import MapKit

private enum CoordinateTransform {
    private static let earthA = 6378245.0
    private static let ee = 0.00669342162296594323

    static func wgs84ToGCJ02(_ coordinate: CLLocationCoordinate2D) -> CLLocationCoordinate2D {
        guard !isOutsideChina(coordinate) else {
            return coordinate
        }

        let delta = delta(for: coordinate)
        return CLLocationCoordinate2D(
            latitude: coordinate.latitude + delta.latitude,
            longitude: coordinate.longitude + delta.longitude
        )
    }

    private static func isOutsideChina(_ coordinate: CLLocationCoordinate2D) -> Bool {
        coordinate.longitude < 72.004 ||
        coordinate.longitude > 137.8347 ||
        coordinate.latitude < 0.8293 ||
        coordinate.latitude > 55.8271
    }

    private static func delta(for coordinate: CLLocationCoordinate2D) -> CLLocationCoordinate2D {
        var dLat = transformLat(x: coordinate.longitude - 105.0, y: coordinate.latitude - 35.0)
        var dLon = transformLon(x: coordinate.longitude - 105.0, y: coordinate.latitude - 35.0)
        let radLat = coordinate.latitude / 180.0 * Double.pi
        var magic = sin(radLat)
        magic = 1 - ee * magic * magic
        let sqrtMagic = sqrt(magic)
        dLat = (dLat * 180.0) / ((earthA * (1 - ee)) / (magic * sqrtMagic) * Double.pi)
        dLon = (dLon * 180.0) / (earthA / sqrtMagic * cos(radLat) * Double.pi)
        return CLLocationCoordinate2D(latitude: dLat, longitude: dLon)
    }

    private static func transformLat(x: Double, y: Double) -> Double {
        var value = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * sqrt(abs(x))
        value += (20.0 * sin(6.0 * x * Double.pi) + 20.0 * sin(2.0 * x * Double.pi)) * 2.0 / 3.0
        value += (20.0 * sin(y * Double.pi) + 40.0 * sin(y / 3.0 * Double.pi)) * 2.0 / 3.0
        value += (160.0 * sin(y / 12.0 * Double.pi) + 320.0 * sin(y * Double.pi / 30.0)) * 2.0 / 3.0
        return value
    }

    private static func transformLon(x: Double, y: Double) -> Double {
        var value = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * sqrt(abs(x))
        value += (20.0 * sin(6.0 * x * Double.pi) + 20.0 * sin(2.0 * x * Double.pi)) * 2.0 / 3.0
        value += (20.0 * sin(x * Double.pi) + 40.0 * sin(x / 3.0 * Double.pi)) * 2.0 / 3.0
        value += (150.0 * sin(x / 12.0 * Double.pi) + 300.0 * sin(x / 30.0 * Double.pi)) * 2.0 / 3.0
        return value
    }
}

@MainActor
final class RouteMapViewModel: NSObject, ObservableObject {
    @Published var userCoordinate: CLLocationCoordinate2D?
    @Published var destinationCoordinate: CLLocationCoordinate2D?
    @Published var routeCoordinates: [CLLocationCoordinate2D] = []
    @Published var routeSummary: MCPRouteSummary?
    @Published var statusText = "等待定位"
    @Published var destinationName = ""

    private let locationManager = CLLocationManager()
    private var isAwaitingLocation = false

    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
    }

    func requestLocationPermission() {
        locationManager.requestWhenInUseAuthorization()
    }

    func locateCurrentPosition() {
        isAwaitingLocation = true
        statusText = "正在定位当前位置..."
        locationManager.requestLocation()
    }

    func resetAll() {
        isAwaitingLocation = false
        userCoordinate = nil
        destinationCoordinate = nil
        routeCoordinates = []
        routeSummary = nil
        statusText = "等待定位"
        destinationName = ""
    }

    func applyRecommendationResult(_ result: MCPRecommendationResponse) {
        let chosen = result.chosenDestination ?? result.recommendations.first
        destinationName = chosen?.name ?? ""
        if let chosen {
            destinationCoordinate = CLLocationCoordinate2D(
                latitude: chosen.coordinates.latitude,
                longitude: chosen.coordinates.longitude
            )
        }
        routeCoordinates = result.routePolyline.map {
            CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude)
        }
        if routeCoordinates.isEmpty {
            routeCoordinates = result.keyWaypoints.map {
                CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude)
            }
        }
        routeSummary = result.routeSummary ?? chosen?.routeSummary
        statusText = routeCoordinates.count >= 2 ? "路线已更新" : "推荐已更新"
    }

    func applyRoutePlan(_ routePlan: MCPRoutePlan, destinationName: String?) {
        self.destinationName = destinationName ?? self.destinationName
        routeSummary = routePlan.routeSummary
        routeCoordinates = (routePlan.routePolyline ?? []).map {
            CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude)
        }
        if routeCoordinates.isEmpty {
            routeCoordinates = (routePlan.coarseWaypoints ?? []).map {
                CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude)
            }
        }
        if let lastStop = routePlan.stopSequence?.last?.coordinate {
            destinationCoordinate = lastStop
        } else if let last = routeCoordinates.last {
            destinationCoordinate = last
        }
        statusText = routeCoordinates.count >= 2 ? "路线已更新" : "路线数据不足"
    }

    func boundingRegion() -> MKCoordinateRegion? {
        let points = [userCoordinate, destinationCoordinate].compactMap { $0 } + routeCoordinates
        guard !points.isEmpty else { return nil }

        var minLat = points[0].latitude
        var maxLat = points[0].latitude
        var minLon = points[0].longitude
        var maxLon = points[0].longitude

        for point in points {
            minLat = min(minLat, point.latitude)
            maxLat = max(maxLat, point.latitude)
            minLon = min(minLon, point.longitude)
            maxLon = max(maxLon, point.longitude)
        }

        let center = CLLocationCoordinate2D(
            latitude: (minLat + maxLat) / 2.0,
            longitude: (minLon + maxLon) / 2.0
        )
        let span = MKCoordinateSpan(
            latitudeDelta: max((maxLat - minLat) * 1.4, 0.01),
            longitudeDelta: max((maxLon - minLon) * 1.4, 0.01)
        )
        return MKCoordinateRegion(center: center, span: span)
    }
}

extension RouteMapViewModel: CLLocationManagerDelegate {
    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            switch manager.authorizationStatus {
            case .authorizedWhenInUse, .authorizedAlways:
                self.statusText = "定位权限已授权"
                self.locateCurrentPosition()
            case .denied, .restricted:
                self.statusText = "定位权限被拒绝"
            case .notDetermined:
                self.statusText = "等待定位权限"
            @unknown default:
                self.statusText = "未知定位权限状态"
            }
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        Task { @MainActor in
            guard self.isAwaitingLocation else { return }
            guard let location = locations.last else { return }
            self.isAwaitingLocation = false
            self.userCoordinate = CoordinateTransform.wgs84ToGCJ02(location.coordinate)
            self.statusText = "当前位置已更新"
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        Task { @MainActor in
            guard self.isAwaitingLocation else { return }
            self.isAwaitingLocation = false
            self.statusText = "定位失败: \(error.localizedDescription)"
        }
    }
}

import Foundation
import CoreLocation
import MapKit

@MainActor
final class MapNavigationViewModel: NSObject, ObservableObject {
    @Published var targetAddress = ""
    @Published var statusText = "等待定位"
    @Published var userCoordinate: CLLocationCoordinate2D?
    @Published var destinationCoordinate: CLLocationCoordinate2D?
    @Published var route: MKRoute?
    @Published var routeWaypoints: [CLLocationCoordinate2D] = []
    @Published var routeSummary: MCPRouteSummary?

    private let locationManager = CLLocationManager()

    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
    }

    func requestLocationPermission() {
        locationManager.requestWhenInUseAuthorization()
    }

    func locateCurrentPosition() {
        statusText = "正在定位当前位置..."
        locationManager.requestLocation()
    }

    func searchAddressAndRoute() async {
        let keyword = targetAddress.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !keyword.isEmpty else {
            statusText = "请输入目标地址"
            return
        }

        do {
            routeWaypoints = []
            routeSummary = nil
            let geocoder = CLGeocoder()
            let places = try await geocoder.geocodeAddressString(keyword)
            guard let coordinate = places.first?.location?.coordinate else {
                statusText = "未找到目标地址"
                return
            }
            destinationCoordinate = coordinate
            statusText = "已定位目标地址"

            guard let userCoordinate else {
                statusText = "已定位目标地址，等待当前位置"
                return
            }
            route = try await calculateRoute(from: userCoordinate, to: coordinate)
            routeSummary = MCPRouteSummary(
                distanceM: route?.distance,
                durationS: route?.expectedTravelTime,
                durationMin: (route?.expectedTravelTime ?? 0) / 60.0
            )
            statusText = "路径规划完成"
        } catch {
            statusText = "路径规划失败: \(error.localizedDescription)"
        }
    }

    func applySelectedCandidate(_ candidate: AddressCandidate) async {
        targetAddress = "\(candidate.title) \(candidate.subtitle)"
        routeWaypoints = []
        routeSummary = MCPRouteSummary(
            distanceM: candidate.distanceM,
            durationS: candidate.durationMin.map { $0 * 60.0 },
            durationMin: candidate.durationMin
        )
        if let coordinate = candidate.coordinate {
            destinationCoordinate = coordinate
            if let userCoordinate {
                do {
                    route = try await calculateRoute(from: userCoordinate, to: coordinate)
                    if candidate.distanceM == nil && candidate.durationMin == nil {
                        routeSummary = MCPRouteSummary(
                            distanceM: route?.distance,
                            durationS: route?.expectedTravelTime,
                            durationMin: (route?.expectedTravelTime ?? 0) / 60.0
                        )
                    }
                    statusText = "已更新至候选地址并完成路径规划"
                } catch {
                    statusText = "候选地址路径规划失败: \(error.localizedDescription)"
                }
            } else {
                statusText = "候选地址已选择，等待当前位置"
            }
            return
        }
        await searchAddressAndRoute()
    }

    func applyPlanningResult(_ result: MCPRecommendationResponse) async {
        let chosen = result.chosenDestination ?? result.recommendations.first
        guard let chosen else {
            statusText = "规划完成，但未返回可用目标"
            return
        }

        let destination = CLLocationCoordinate2D(
            latitude: chosen.coordinates.latitude,
            longitude: chosen.coordinates.longitude
        )
        destinationCoordinate = destination
        targetAddress = chosen.name
        routeSummary = result.routeSummary ?? chosen.routeSummary

        let finePolyline = result.routePolyline.map {
            CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude)
        }
        let coarsePolyline = result.keyWaypoints.map {
            CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude)
        }
        routeWaypoints = finePolyline.count >= 2 ? finePolyline : coarsePolyline

        if routeWaypoints.count >= 2 {
            route = nil
            if routeSummary == nil {
                let distance = polylineDistanceMeters(routeWaypoints)
                routeSummary = MCPRouteSummary(
                    distanceM: distance,
                    durationS: nil,
                    durationMin: nil
                )
            }
            statusText = finePolyline.count >= 2
                ? "规划完成，已使用 MCP 细粒度 polyline 展示"
                : "规划完成，已使用 MCP 路线关键点展示"
            return
        }

        guard let userCoordinate else {
            statusText = "规划完成，等待当前位置后进行路径兜底"
            return
        }

        do {
            route = try await calculateRoute(from: userCoordinate, to: destination)
            if routeSummary == nil {
                routeSummary = MCPRouteSummary(
                    distanceM: route?.distance,
                    durationS: route?.expectedTravelTime,
                    durationMin: (route?.expectedTravelTime ?? 0) / 60.0
                )
            }
            statusText = "规划完成，已使用本地路径兜底"
        } catch {
            statusText = "规划结果已应用，但本地路径兜底失败: \(error.localizedDescription)"
        }
    }

    private func polylineDistanceMeters(_ coordinates: [CLLocationCoordinate2D]) -> Double {
        guard coordinates.count >= 2 else { return 0 }
        var total: Double = 0
        for index in 1..<coordinates.count {
            let from = coordinates[index - 1]
            let to = coordinates[index]
            let start = CLLocation(latitude: from.latitude, longitude: from.longitude)
            let end = CLLocation(latitude: to.latitude, longitude: to.longitude)
            total += start.distance(from: end)
        }
        return total
    }

    private func calculateRoute(from source: CLLocationCoordinate2D, to destination: CLLocationCoordinate2D) async throws -> MKRoute {
        let sourceItem = MKMapItem(placemark: MKPlacemark(coordinate: source))
        let destinationItem = MKMapItem(placemark: MKPlacemark(coordinate: destination))

        let request = MKDirections.Request()
        request.source = sourceItem
        request.destination = destinationItem
        request.transportType = .automobile

        let directions = MKDirections(request: request)
        let response = try await directions.calculate()
        guard let first = response.routes.first else {
            throw NSError(domain: "MapNavigation", code: -1, userInfo: [NSLocalizedDescriptionKey: "无可用路线"])
        }
        return first
    }

    private func refreshRouteAfterLocationUpdateIfNeeded() async {
        guard let userCoordinate, let destinationCoordinate else { return }

        // MCP 已提供关键路径点时，优先保留 MCP 路线展示，仅更新状态。
        if routeWaypoints.count >= 2 {
            statusText = "当前位置已更新（MCP 路线已就绪）"
            return
        }

        do {
            route = try await calculateRoute(from: userCoordinate, to: destinationCoordinate)
            if routeSummary == nil {
                routeSummary = MCPRouteSummary(
                    distanceM: route?.distance,
                    durationS: route?.expectedTravelTime,
                    durationMin: (route?.expectedTravelTime ?? 0) / 60.0
                )
            }
            statusText = "当前位置已更新，路径已刷新"
        } catch {
            statusText = "当前位置已更新，但路径刷新失败: \(error.localizedDescription)"
        }
    }
}

extension MapNavigationViewModel: CLLocationManagerDelegate {
    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            switch manager.authorizationStatus {
            case .authorizedWhenInUse, .authorizedAlways:
                self.statusText = "定位权限已授权"
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
            guard let location = locations.last else { return }
            self.userCoordinate = location.coordinate
            await self.refreshRouteAfterLocationUpdateIfNeeded()
            if self.destinationCoordinate == nil {
                self.statusText = "当前位置已更新"
            }
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: any Error) {
        Task { @MainActor in
            self.statusText = "定位失败: \(error.localizedDescription)"
        }
    }
}

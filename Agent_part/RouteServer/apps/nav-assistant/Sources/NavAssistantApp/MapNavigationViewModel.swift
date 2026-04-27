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
    @Published var routeSegments: [[CLLocationCoordinate2D]] = []
    @Published var routeSummary: MCPRouteSummary?
    @Published var viaPoints: [ViaPoint] = []
    @Published var nextWaypointName: String?
    @Published var nextWaypointDistance: String?
    @Published var nextWaypointDuration: String?

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

    func resetAll() {
        targetAddress = ""
        statusText = "等待定位"
        userCoordinate = nil
        destinationCoordinate = nil
        route = nil
        routeWaypoints = []
        routeSegments = []
        routeSummary = nil
        viaPoints = []
        nextWaypointName = nil
        nextWaypointDistance = nil
        nextWaypointDuration = nil
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
                requestLocationPermission()
                locationManager.requestLocation()
                statusText = "已定位目标地址，正在自动定位当前位置..."
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

    // MARK: - Route Session Result Handling

    func applyRouteSessionResult(_ result: MCPRouteSessionResult) {
        guard let routePlan = result.routePlan else {
            statusText = "Route session 响应中缺少 route_plan"
            return
        }

        // Update via points from session
        if let session = result.session, let viaPointsList = session.viaPoints {
            viaPoints = viaPointsList
        }

        // Extract next waypoint info from stop sequence (index 1 = first via point)
        nextWaypointName = nil
        nextWaypointDistance = nil
        nextWaypointDuration = nil
        if let stopSequence = routePlan.stopSequence, stopSequence.count > 1,
           let firstVia = stopSequence[1] as StopSequenceItem?,
           firstVia.stopType == "via" {
            nextWaypointName = firstVia.name
            if let leg = firstVia.routeLegSummary {
                nextWaypointDistance = leg.distanceText
                nextWaypointDuration = leg.durationText
            }
        }

        // Update route summary
        routeSummary = routePlan.routeSummary

        // Prefer explicit route geometry from route_plan. Using only stop_sequence
        // (origin/destination/via) can degrade into straight segments.
        let finePolyline = (routePlan.routePolyline ?? []).map {
            CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude)
        }
        let coarsePolyline = (routePlan.coarseWaypoints ?? []).map {
            CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude)
        }

        // Build segments for colored multi-leg rendering.
        // With via points, use buildSegmentsFromStopSequence for per-leg colored segments.
        // With no via points, use the full polyline as a single segment.
        let viaCount = (routePlan.stopSequence ?? []).filter { $0.stopType == "via" }.count
        let hasMultipleLegs = viaCount > 0

        if hasMultipleLegs {
            // Multi-colored segment rendering: split the actual polyline at via point boundaries.
            // This preserves the real OSRM route geometry for each leg instead of straight lines.
            if finePolyline.count >= 2 {
                let segments = buildPolylineSegments(finePolyline, stopSequence: routePlan.stopSequence ?? [])
                routeSegments = segments
                routeWaypoints = finePolyline

                if let lastStop = (routePlan.stopSequence ?? []).last, let coord = lastStop.coordinate {
                    destinationCoordinate = coord
                }
                statusText = "途经点路线已更新，共 \(viaCount) 个途经点（polyline 分段）"
                return
            }

            // Fallback: no polyline, use straight-line segments between stops
            if let stopSequence = routePlan.stopSequence, !stopSequence.isEmpty {
                let segments = buildSegmentsFromStopSequence(stopSequence, routePlan: routePlan)
                routeSegments = segments
                routeWaypoints = segments.flatMap { $0 }

                if let lastStop = stopSequence.last, let coord = lastStop.coordinate {
                    destinationCoordinate = coord
                }
                statusText = "途经点路线已更新，共 \(viaCount) 个途经点（直线分段）"
                return
            }
        }

        // Single-leg path (no via points): use the full polyline
        routeWaypoints = finePolyline.count >= 2 ? finePolyline : coarsePolyline
        routeSegments = routeWaypoints.count >= 2 ? [routeWaypoints] : []

        if routeWaypoints.count >= 2 {
            route = nil

            if let stopSequence = routePlan.stopSequence,
               let lastStop = stopSequence.last,
               let coord = lastStop.coordinate {
                destinationCoordinate = coord
            } else if let last = routeWaypoints.last {
                destinationCoordinate = last
            }

            if routeSummary == nil {
                let distance = polylineDistanceMeters(routeWaypoints)
                routeSummary = MCPRouteSummary(
                    distanceM: distance,
                    durationS: nil,
                    durationMin: nil
                )
            }

            statusText = finePolyline.count >= 2
                ? "路线已更新，已使用 route_session polyline"
                : "路线已更新，已使用路线关键点"
            return
        }

        statusText = "Route session 响应缺少路线坐标"
    }

    /// Splits the full route polyline into per-leg segments using via point coordinates as boundaries.
    /// Each segment contains the actual OSRM route coordinates for one leg, preserving the real
    /// route geometry instead of drawing straight lines between stops.
    private func buildPolylineSegments(
        _ polyline: [CLLocationCoordinate2D],
        stopSequence: [StopSequenceItem]
    ) -> [[CLLocationCoordinate2D]] {
        guard polyline.count >= 2 else { return [] }

        // Find the index in polyline nearest to a given coordinate
        func nearestPolylineIndex(to coord: CLLocationCoordinate2D) -> Int {
            var bestIdx = 0
            var bestDist = Double.infinity
            for (i, pt) in polyline.enumerated() {
                let loc1 = CLLocation(latitude: coord.latitude, longitude: coord.longitude)
                let loc2 = CLLocation(latitude: pt.latitude, longitude: pt.longitude)
                let d = loc1.distance(from: loc2)
                if d < bestDist {
                    bestDist = d
                    bestIdx = i
                }
            }
            return bestIdx
        }

        // For each stop (except origin), find its nearest polyline index
        var stopPolylineIndices: [Int] = []
        for stop in stopSequence {
            if let coord = stop.coordinate {
                stopPolylineIndices.append(nearestPolylineIndex(to: coord))
            }
        }

        // Build segments: polyline[0..stop1] for leg 0, polyline[stop1..stop2] for leg 1, etc.
        var segments: [[CLLocationCoordinate2D]] = []
        var start = 0

        for endIdx in stopPolylineIndices {
            guard start <= endIdx && endIdx < polyline.count else {
                start += 1
                continue
            }
            let segment = Array(polyline[start...endIdx])
            if !segment.isEmpty {
                segments.append(segment)
            }
            start = endIdx
        }

        // Add remaining polyline points after the last via (up to the destination)
        if start < polyline.count {
            let lastSegment = Array(polyline[start...])
            if !lastSegment.isEmpty {
                // Avoid duplicating the last point if already added
                let sameAsLast = segments.last?.last.map { last in
                    lastSegment.first.map { $0.latitude == last.latitude && $0.longitude == last.longitude } ?? false
                } ?? false
                if !sameAsLast {
                    segments.append(lastSegment)
                }
            }
        }

        // Ensure we have at least one segment
        if segments.isEmpty && !polyline.isEmpty {
            segments.append(polyline)
        }

        return segments
    }

    private func buildSegmentsFromStopSequence(
        _ stopSequence: [StopSequenceItem],
        routePlan: MCPRoutePlan
    ) -> [[CLLocationCoordinate2D]] {
        var segments: [[CLLocationCoordinate2D]] = []
        var currentSegment: [CLLocationCoordinate2D] = []

        for (index, stop) in stopSequence.enumerated() {
            if let coord = stop.coordinate {
                currentSegment.append(coord)
            }

            // End of a leg: stop is destination or via (end of segment)
            let isLegEnd = stop.stopType == "destination" ||
                (stop.stopType == "via" && index + 1 < stopSequence.count)
            let isLast = index == stopSequence.count - 1

            if isLegEnd && !currentSegment.isEmpty {
                segments.append(currentSegment)
                // Don't clear currentSegment if we're at the last stop
                // (it's already included in the segment we just added)
                if !isLast {
                    // Start a new segment with the current stop as the beginning
                    if let coord = stop.coordinate {
                        currentSegment = [coord]
                    } else {
                        currentSegment = []
                    }
                }
            }
        }

        return segments
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
        request.transportType = .walking

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
            self.userCoordinate = CoordinateTransform.wgs84ToGCJ02(location.coordinate)
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

import SwiftUI
import MapKit

struct MapPanelView: View {
    @ObservedObject var viewModel: MapNavigationViewModel
    let isVoiceOnlyMode: Bool

    @State private var initialCameraPosition: MapCameraPosition = .automatic
    @State private var mapReloadToken = UUID()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("地图导航")
                .font(.headline)

            if isVoiceOnlyMode {
                Text("盲人语音模式已启用，地图仅保留为可视化参考，主要交互请在左侧通过语音完成。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                HStack {
                    TextField("输入目标地址", text: $viewModel.targetAddress)
                        .textFieldStyle(.roundedBorder)
                    Button("路径规划") {
                        Task { await viewModel.searchAddressAndRoute() }
                    }
                }

                HStack {
                    Button("请求定位权限") {
                        viewModel.requestLocationPermission()
                    }
                    Button("定位当前位置") {
                        viewModel.locateCurrentPosition()
                    }
                }
            }

            Text(viewModel.statusText)
                .font(.caption)
                .foregroundStyle(.secondary)

            if shouldShowSummary {
                let distanceKm = (viewModel.routeSummary?.distanceM ?? viewModel.route?.distance ?? 0) / 1000.0
                let durationMinValue = viewModel.routeSummary?.durationMin
                    ?? ((viewModel.routeSummary?.durationS ?? viewModel.route?.expectedTravelTime ?? 0) / 60.0)
                let durationText = durationMinValue > 0 ? String(format: "%.1f 分钟", durationMinValue) : "--"
                Text("路线摘要：\(String(format: "%.2f km", distanceKm)) / \(durationText)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Map(initialPosition: initialCameraPosition) {
                if let userCoordinate = viewModel.userCoordinate {
                    Annotation("我", coordinate: userCoordinate) {
                        Circle()
                            .fill(.blue)
                            .frame(width: 10, height: 10)
                    }
                }

                if let destinationCoordinate = viewModel.destinationCoordinate {
                    Annotation("目标", coordinate: destinationCoordinate) {
                        Circle()
                            .fill(.red)
                            .frame(width: 10, height: 10)
                    }
                }

                if viewModel.routeWaypoints.count >= 2 {
                    MapPolyline(
                        MKPolyline(
                            coordinates: viewModel.routeWaypoints,
                            count: viewModel.routeWaypoints.count
                        )
                    )
                    .stroke(.green, lineWidth: 5)
                }

                if let route = viewModel.route {
                    MapPolyline(route.polyline)
                        .stroke(.green, lineWidth: 5)
                }
            }
            .id(mapReloadToken)
            .onReceive(viewModel.$destinationCoordinate) { newValue in
                guard let destination = newValue else { return }
                focusMap(
                    on: destination,
                    span: MKCoordinateSpan(latitudeDelta: 0.05, longitudeDelta: 0.05)
                )
            }
            .onReceive(viewModel.$userCoordinate) { newValue in
                guard viewModel.destinationCoordinate == nil, let user = newValue else { return }
                focusMap(
                    on: user,
                    span: MKCoordinateSpan(latitudeDelta: 0.02, longitudeDelta: 0.02)
                )
            }
            .frame(minHeight: 300)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .padding()
    }

    private var shouldShowSummary: Bool {
        viewModel.routeSummary != nil
            || viewModel.route != nil
            || viewModel.routeWaypoints.count >= 2
    }

    private func focusMap(on coordinate: CLLocationCoordinate2D, span: MKCoordinateSpan) {
        initialCameraPosition = .region(
            MKCoordinateRegion(center: coordinate, span: span)
        )
        // Recreate the map instead of mutating a live camera binding, which can
        // trigger AttributeGraph cycles on macOS during SwiftUI layout.
        mapReloadToken = UUID()
    }
}

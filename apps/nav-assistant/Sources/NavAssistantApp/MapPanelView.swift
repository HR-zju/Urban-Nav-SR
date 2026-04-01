import SwiftUI
import MapKit

struct MapPanelView: View {
    @ObservedObject var viewModel: MapNavigationViewModel
    let isVoiceOnlyMode: Bool

    @State private var initialCameraPosition: MapCameraPosition = .automatic
    @State private var mapReloadToken = UUID()
    @State private var isManualLookupExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            routeOverviewSection
            mapSection
            if !isVoiceOnlyMode {
                manualLookupSection
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Color.navWindow)
    }

    private var routeOverviewSection: some View {
        NavPanel(
            title: "地图与路线",
            subtitle: isVoiceOnlyMode
                ? "地图仅保留为视觉参考，主要交互已经集中到左侧语音指挥台。"
                : "右侧聚焦当前位置、目标与路线反馈，避免与左侧需求录入流程竞争。",
            accessory: {
                NavStatusBadge(text: routeStatusText, tone: routeStatusTone, systemImage: routeStatusSymbol)
            }
        ) {
            VStack(alignment: .leading, spacing: 14) {
                LazyVGrid(
                    columns: [
                        GridItem(.flexible(), spacing: 10),
                        GridItem(.flexible(), spacing: 10)
                    ],
                    spacing: 10
                ) {
                    NavMetricBadge(label: "当前位置", value: viewModel.userCoordinate == nil ? "未定位" : "已获取")
                    NavMetricBadge(label: "目的地", value: currentDestinationTitle, emphasized: viewModel.destinationCoordinate != nil)
                    NavMetricBadge(label: "路线距离", value: routeDistanceText)
                    NavMetricBadge(label: "预计耗时", value: routeDurationText)
                }

                if !viewModel.targetAddress.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    NavInfoBanner(text: "当前目标：\(viewModel.targetAddress)", tone: .accent)
                }

                NavInfoBanner(text: viewModel.statusText, tone: statusTone)

                HStack(spacing: 10) {
                    Button("请求定位权限") {
                        viewModel.requestLocationPermission()
                    }

                    Button("定位当前位置") {
                        viewModel.locateCurrentPosition()
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
    }

    private var mapSection: some View {
        NavPanel(
            title: "路线预览",
            subtitle: "系统会在识别到当前位置或目标后自动聚焦地图。"
        ) {
            ZStack(alignment: .bottomLeading) {
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
                .frame(minHeight: 430)
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(Color.navBorder.opacity(0.55), lineWidth: 1)
                )
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

                if viewModel.userCoordinate == nil && viewModel.destinationCoordinate == nil {
                    ContentUnavailableView(
                        "等待位置或目标",
                        systemImage: "map",
                        description: Text("先定位当前位置，再确认目的地，地图会自动聚焦路线。")
                    )
                    .padding(28)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .padding(24)
                }

                HStack(spacing: 8) {
                    if viewModel.userCoordinate != nil {
                        NavStatusBadge(text: "我的位置", tone: .accent, systemImage: "location.fill")
                    }
                    if viewModel.destinationCoordinate != nil {
                        NavStatusBadge(text: "目的地", tone: .danger, systemImage: "mappin.circle.fill")
                    }
                    if hasActiveRoute {
                        NavStatusBadge(text: "路线", tone: .success, systemImage: "arrow.triangle.branch")
                    }
                }
                .padding(16)
            }
        }
    }

    private var manualLookupSection: some View {
        NavPanel(
            title: "手动地图检索",
            subtitle: "保留备用的直接地址输入能力，但弱化为次要入口，避免和主流程重复。"
        ) {
            DisclosureGroup("展开备用地址检索", isExpanded: $isManualLookupExpanded) {
                VStack(alignment: .leading, spacing: 12) {
                    TextField("输入目标地址", text: $viewModel.targetAddress)
                        .textFieldStyle(.roundedBorder)

                    HStack(spacing: 10) {
                        Button("路径规划") {
                            Task { await viewModel.searchAddressAndRoute() }
                        }
                        .buttonStyle(.borderedProminent)

                        if !viewModel.targetAddress.isEmpty {
                            Button("清空地址") {
                                viewModel.targetAddress = ""
                            }
                        }
                    }
                }
                .padding(.top, 12)
            }
        }
    }

    private var currentDestinationTitle: String {
        let text = viewModel.targetAddress.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? "未选择" : text
    }

    private var routeDistanceText: String {
        if let distance = viewModel.routeSummary?.distanceText {
            return distance
        }
        guard let route = viewModel.route else { return "--" }
        if route.distance >= 1000 {
            return String(format: "%.2f km", route.distance / 1000.0)
        }
        return String(format: "%.0f m", route.distance)
    }

    private var routeDurationText: String {
        if let duration = viewModel.routeSummary?.durationText {
            return duration
        }
        guard let route = viewModel.route else { return "--" }
        return String(format: "%.0f 分钟", route.expectedTravelTime / 60.0)
    }

    private var hasActiveRoute: Bool {
        viewModel.routeSummary != nil || viewModel.route != nil || viewModel.routeWaypoints.count >= 2
    }

    private var routeStatusText: String {
        if hasActiveRoute {
            return "路线已就绪"
        }
        if viewModel.destinationCoordinate != nil {
            return "目标已锁定"
        }
        if viewModel.userCoordinate != nil {
            return "等待目的地"
        }
        return "等待定位"
    }

    private var routeStatusTone: NavSurfaceTone {
        if viewModel.statusText.contains("失败") || viewModel.statusText.contains("拒绝") {
            return .danger
        }
        if hasActiveRoute {
            return .success
        }
        if viewModel.destinationCoordinate != nil {
            return .accent
        }
        return .neutral
    }

    private var routeStatusSymbol: String {
        if viewModel.statusText.contains("失败") || viewModel.statusText.contains("拒绝") {
            return "xmark.octagon.fill"
        }
        if hasActiveRoute {
            return "checkmark.circle.fill"
        }
        if viewModel.destinationCoordinate != nil {
            return "mappin.circle.fill"
        }
        return "location.slash"
    }

    private var statusTone: NavSurfaceTone {
        if viewModel.statusText.contains("失败") || viewModel.statusText.contains("拒绝") {
            return .danger
        }
        if viewModel.statusText.contains("等待") {
            return .neutral
        }
        if viewModel.statusText.contains("正在") {
            return .warning
        }
        return hasActiveRoute ? .success : .accent
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

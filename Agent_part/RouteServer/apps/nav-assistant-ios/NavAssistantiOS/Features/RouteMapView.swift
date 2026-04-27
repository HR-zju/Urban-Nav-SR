import SwiftUI
import MapKit

struct RouteMapView: View {
    @ObservedObject var viewModel: RouteMapViewModel
    @ObservedObject var navigationViewModel: NavigationFlowViewModel
    @State private var cameraPosition: MapCameraPosition = .automatic

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            RouteMapHeaderView(statusText: activeSessionStatusText)
            routeCanvas
            RouteMapDetailsView(
                destinationName: viewModel.destinationName,
                sessionID: navigationViewModel.activeRouteSessionID,
                heartbeatText: navigationViewModel.routeSessionHeartbeatText,
                changeReason: navigationViewModel.lastRouteSessionChangeReason,
                routePlanCacheHit: navigationViewModel.lastRoutePlanCacheHit,
                routePlanUpdateMode: navigationViewModel.lastRoutePlanUpdateMode,
                routeSummary: viewModel.routeSummary
            )

            if hasActiveSession {
                SessionActionSection(
                    isLoading: navigationViewModel.isLoading,
                    onRefresh: { Task { await navigationViewModel.refreshActiveRouteSession() } },
                    onForceRefresh: { Task { await navigationViewModel.refreshActiveRouteSession(forceReplan: true) } },
                    onReplan: { Task { await navigationViewModel.replanActiveRouteSessionFromCurrentLocation() } }
                )

                ViaPointSectionView(
                    viaPoints: navigationViewModel.currentViaPoints,
                    viaPointPrompt: $navigationViewModel.viaPointPrompt,
                    isLoading: navigationViewModel.isLoading,
                    onAdd: { Task { await navigationViewModel.addViaPointFromPrompt() } }
                )

                WaypointPushSectionView(
                    robotHost: $navigationViewModel.robotHost,
                    robotPortText: $navigationViewModel.robotPortText,
                    isLoading: navigationViewModel.isLoading,
                    statusMessage: navigationViewModel.waypointPushStatusMessage,
                    lastResponse: navigationViewModel.lastWaypointPushResponse,
                    onPush: { Task { await navigationViewModel.pushWaypointsForActiveSession() } }
                )
            }

            if let routeSessionErrorMessage = navigationViewModel.routeSessionErrorMessage,
               !routeSessionErrorMessage.isEmpty {
                Text(routeSessionErrorMessage)
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color(uiColor: .systemBackground))
                .shadow(color: Color.black.opacity(0.06), radius: 14, y: 6)
        )
    }

    private func refreshCamera() {
        if let region = viewModel.boundingRegion() {
            cameraPosition = .region(region)
        }
    }

    private var hasActiveSession: Bool {
        navigationViewModel.activeRouteSessionID != nil
    }

    private var activeSessionStatusText: String? {
        guard hasActiveSession else { return nil }
        return navigationViewModel.routeSessionStatusText
    }

    private var routeCanvas: some View {
        Group {
            if shouldRenderMap {
                Map(position: $cameraPosition) {
                    if let userCoordinate = viewModel.userCoordinate {
                        Annotation("我", coordinate: userCoordinate) {
                            Circle()
                                .fill(.blue)
                                .frame(width: 12, height: 12)
                        }
                    }

                    if let destinationCoordinate = viewModel.destinationCoordinate {
                        Annotation("目标", coordinate: destinationCoordinate) {
                            Circle()
                                .fill(.red)
                                .frame(width: 12, height: 12)
                        }
                    }

                    if viewModel.routeCoordinates.count >= 2 {
                        MapPolyline(
                            MKPolyline(
                                coordinates: viewModel.routeCoordinates,
                                count: viewModel.routeCoordinates.count
                            )
                        )
                        .stroke(.green, lineWidth: 5)
                    }
                }
            } else {
                VStack(spacing: 10) {
                    Image(systemName: "map")
                        .font(.system(size: 28))
                        .foregroundStyle(.secondary)
                    Text("地图将在定位成功或生成路线后显示")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(uiColor: .secondarySystemBackground))
            }
        }
        .frame(height: 320)
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .onAppear(perform: refreshCamera)
        .onChange(of: userCoordinateKey) { _, _ in refreshCamera() }
        .onChange(of: destinationCoordinateKey) { _, _ in refreshCamera() }
        .onChange(of: routeCoordinatesKey) { _, _ in refreshCamera() }
    }

    private var userCoordinateKey: String {
        coordinateKey(viewModel.userCoordinate)
    }

    private var destinationCoordinateKey: String {
        coordinateKey(viewModel.destinationCoordinate)
    }

    private var routeCoordinatesKey: String {
        viewModel.routeCoordinates.map(coordinateKey).joined(separator: "|")
    }

    private var shouldRenderMap: Bool {
        viewModel.userCoordinate != nil
            || viewModel.destinationCoordinate != nil
            || !viewModel.routeCoordinates.isEmpty
    }

    private func coordinateKey(_ coordinate: CLLocationCoordinate2D?) -> String {
        guard let coordinate else { return "nil" }
        return coordinateKey(coordinate)
    }

    private func coordinateKey(_ coordinate: CLLocationCoordinate2D) -> String {
        "\(coordinate.latitude),\(coordinate.longitude)"
    }
}

private struct RouteMapHeaderView: View {
    let statusText: String?

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("路线预览")
                .font(.title3.bold())
            Spacer()
            if let statusText {
                Text(statusText)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.blue)
            }
        }
    }
}

private struct RouteMapDetailsView: View {
    let destinationName: String
    let sessionID: String?
    let heartbeatText: String?
    let changeReason: String?
    let routePlanCacheHit: Bool?
    let routePlanUpdateMode: String?
    let routeSummary: MCPRouteSummary?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !destinationName.isEmpty {
                Text("目的地：\(destinationName)")
                    .font(.subheadline)
            }

            if let sessionID {
                Text("Session ID：\(sessionID)")
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
            }

            if let heartbeatText {
                Text(heartbeatText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            if let changeReason, !changeReason.isEmpty {
                Text("最近变更：\(changeReason)")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            if let routePlanCacheHit {
                Text(routePlanCacheHit ? "当前路线来自缓存" : "当前路线为最新计算")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            if let routePlanUpdateMode, !routePlanUpdateMode.isEmpty {
                Text("更新方式：\(routePlanUpdateMode)")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 16) {
                if let distance = routeSummary?.distanceText {
                    Label(distance, systemImage: "arrow.triangle.swap")
                }
                if let duration = routeSummary?.durationText {
                    Label(duration, systemImage: "clock")
                }
            }
            .font(.footnote)
            .foregroundStyle(.secondary)
        }
    }
}

private struct SessionActionSection: View {
    let isLoading: Bool
    let onRefresh: () -> Void
    let onForceRefresh: () -> Void
    let onReplan: () -> Void

    var body: some View {
        ViewThatFits(in: .vertical) {
            HStack(spacing: 12) {
                actionButtons
            }

            VStack(alignment: .leading, spacing: 10) {
                actionButtons
            }
        }
    }

    @ViewBuilder
    private var actionButtons: some View {
        Button("刷新会话", action: onRefresh)
            .buttonStyle(.bordered)
            .disabled(isLoading)

        Button("强制重算", action: onForceRefresh)
            .buttonStyle(.bordered)
            .disabled(isLoading)

        Button("当前位置重规划", action: onReplan)
            .buttonStyle(.borderedProminent)
            .disabled(isLoading)
    }
}

private struct ViaPointSectionView: View {
    let viaPoints: [ViaPoint]
    @Binding var viaPointPrompt: String
    let isLoading: Bool
    let onAdd: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("途经点")
                .font(.headline)

            if viaPoints.isEmpty {
                Text("当前没有途经点。输入一个地点需求后可搜索并插入。")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(viaPoints) { viaPoint in
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "mappin.and.ellipse")
                            .foregroundStyle(.orange)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(viaPoint.name)
                                .font(.subheadline.weight(.medium))
                            if let location = viaPoint.location {
                                Text(location)
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }

            TextField("新增途经点，例如：便利店、咖啡店、药店", text: $viaPointPrompt)
                .textInputAutocapitalization(.never)
                .padding(10)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color(uiColor: .secondarySystemBackground))
                )

            Button(isLoading ? "添加中…" : "搜索并添加途经点", action: onAdd)
                .buttonStyle(.bordered)
                .disabled(isLoading)
        }
    }
}

private struct WaypointPushSectionView: View {
    @Binding var robotHost: String
    @Binding var robotPortText: String
    let isLoading: Bool
    let statusMessage: String?
    let lastResponse: WaypointPushResponse?
    let onPush: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Waypoint 下发")
                .font(.headline)

            TextField("机器人地址", text: $robotHost)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .padding(10)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color(uiColor: .secondarySystemBackground))
                )

            TextField("机器人端口", text: $robotPortText)
                .keyboardType(.numberPad)
                .padding(10)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color(uiColor: .secondarySystemBackground))
                )

            Button(isLoading ? "下发中…" : "下发当前 Waypoints", action: onPush)
                .buttonStyle(.borderedProminent)
                .disabled(isLoading)

            if let statusMessage, !statusMessage.isEmpty {
                Text(statusMessage)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            if let lastResponse {
                Text("最近一次下发：状态 \(lastResponse.status)，数量 \(lastResponse.totalCount ?? 0)")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

import Foundation
import CoreLocation
import Combine

@MainActor
final class NavigationFlowViewModel: ObservableObject {
    private static let defaultRobotHost = "192.168.168.110"
    private static let defaultRobotPortText = "8765"

    @Published var flowStage: NavigationFlowStage = .idle
    @Published var promptText = ""
    @Published var outputMode: RecognitionOutputMode = .streaming
    @Published var candidates: [AddressCandidate] = []
    @Published var selectedCandidate: AddressCandidate?
    @Published var recommendationResponse: MCPRecommendationResponse?
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var showingVoiceAssistant = false
    @Published var activeRouteSession: RouteSession?
    @Published var activeRoutePlan: MCPRoutePlan?
    @Published var isRouteSessionConnected = false
    @Published var lastRouteSessionHeartbeatAt: Date?
    @Published var routeSessionErrorMessage: String?
    @Published var lastRouteSessionChangeReason: String?
    @Published var lastRoutePlanCacheHit: Bool?
    @Published var lastRoutePlanUpdateMode: String?
    @Published var viaPointPrompt = ""
    @Published var robotHost = NavigationFlowViewModel.defaultRobotHost
    @Published var robotPortText = NavigationFlowViewModel.defaultRobotPortText
    @Published var lastWaypointPushResponse: WaypointPushResponse?
    @Published var waypointPushStatusMessage: String?

    let speech = SpeechRecognitionManager()
    let voice = VoiceInteractionManager()
    let mapViewModel = RouteMapViewModel()

    private let recommendationService: RecommendationService
    private let routeService: RouteSessionServiceProtocol
    private let waypointPushService: WaypointPushService
    private let webSocketRouteSessionClient: WebSocketRouteSessionClient
    private var cancellables: Set<AnyCancellable> = []
    private var resetGeneration = UUID()

    init(
        recommendationService: RecommendationService,
        routeService: RouteSessionServiceProtocol,
        waypointPushService: WaypointPushService,
        webSocketRouteSessionClient: WebSocketRouteSessionClient
    ) {
        self.recommendationService = recommendationService
        self.routeService = routeService
        self.waypointPushService = waypointPushService
        self.webSocketRouteSessionClient = webSocketRouteSessionClient
        bindNestedState()
    }

    func bootstrap() async {
        flowStage = .idle
    }

    func syncPromptFromSpeech() {
        let text = speech.combinedText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
            promptText = text
        }
    }

    func toggleSpeechRecognition() async {
        if speech.isRunning {
            speech.stopRecognition()
            syncPromptFromSpeech()
            return
        }
        let granted = await speech.requestPermissionsIfNeeded()
        guard granted else { return }
        speech.startRecognition(mode: outputMode)
    }

    func requestRecommendations() async {
        let generation = resetGeneration
        let prompt = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            errorMessage = "请输入导航需求。"
            flowStage = .error
            return
        }
        guard let currentLocation = mapViewModel.userCoordinate else {
            errorMessage = "请先定位当前位置。"
            flowStage = .error
            return
        }

        flowStage = .recommending
        isLoading = true
        defer { isLoading = false }
        errorMessage = nil
        do {
            let result = try await recommendationService.recommend(
                prompt: prompt,
                currentLocation: currentLocation,
                mode: "foot",
                maxCandidates: 5,
                searchRadius: 3000
            )
            guard generation == resetGeneration else { return }
            recommendationResponse = result
            candidates = result.recommendations.map { $0.toAddressCandidate() }
            selectedCandidate = nil
            clearRouteSessionState(disconnect: true)
            mapViewModel.applyRecommendationResult(result)
            voice.announceCandidates(from: result)
            flowStage = .candidateSelection
        } catch {
            guard generation == resetGeneration else { return }
            errorMessage = error.localizedDescription
            flowStage = .error
        }
    }

    func chooseCandidate(_ candidate: AddressCandidate) async {
        let generation = resetGeneration
        selectedCandidate = candidate
        errorMessage = nil

        guard let currentLocation = mapViewModel.userCoordinate else {
            errorMessage = "缺少当前位置。"
            flowStage = .error
            return
        }

        let destinationLocation: String
        if let locationText = candidate.locationText, locationText.contains(",") {
            destinationLocation = locationText
        } else if let coordinate = candidate.coordinate {
            destinationLocation = "\(coordinate.longitude),\(coordinate.latitude)"
        } else {
            errorMessage = "候选缺少可用坐标。"
            flowStage = .error
            return
        }

        clearRouteSessionState(disconnect: true)
        flowStage = .creatingSession
        isLoading = true
        defer { isLoading = false }
        do {
            let sessionResult = try await routeService.createSession(
                origin: "\(currentLocation.longitude),\(currentLocation.latitude)",
                destination: destinationLocation,
                viaPoints: nil,
                mode: "foot",
                sessionName: candidate.title
            )
            guard generation == resetGeneration else { return }
            applyRouteSessionResult(sessionResult, destinationName: candidate.title)
            if let sessionID = sessionResult.session?.sessionId {
                connectRouteSession(sessionID: sessionID)
            }
            if let routePlan = sessionResult.routePlan {
                voice.announceRoute(plan: routePlan, destinationName: candidate.title)
            } else if let recommendationResponse,
                      recommendationResponse.chosenDestination?.id == candidate.providerId {
                voice.announceRoute(from: recommendationResponse)
            }
            flowStage = .sessionActive
        } catch {
            guard generation == resetGeneration else { return }
            errorMessage = error.localizedDescription
            flowStage = .error
        }
    }

    func announceCandidates() {
        guard let recommendationResponse else { return }
        voice.announceCandidates(from: recommendationResponse)
    }

    func announceCurrentRoute() {
        if let activeRoutePlan {
            voice.announceRoute(plan: activeRoutePlan, destinationName: selectedCandidate?.title ?? mapViewModel.destinationName)
            return
        }
        if let recommendationResponse,
           recommendationResponse.chosenDestination?.id == selectedCandidate?.providerId {
            voice.announceRoute(from: recommendationResponse)
            return
        }
        guard let summary = mapViewModel.routeSummary else { return }
        let plan = MCPRoutePlan(
            provider: nil,
            origin: nil,
            destination: nil,
            mode: "foot",
            viaPoints: nil,
            stopSequence: nil,
            routeSummary: summary,
            routePolyline: mapViewModel.routeCoordinates.map {
                MCPCoordinates(latitude: $0.latitude, longitude: $0.longitude)
            },
            coarseWaypoints: nil,
            legCount: nil
        )
        voice.announceRoute(plan: plan, destinationName: selectedCandidate?.title ?? mapViewModel.destinationName)
    }

    func pushWaypoints(sessionID: String, robotHost: String, robotPort: Int) async {
        do {
            _ = try await waypointPushService.pushWaypoints(
                sessionID: sessionID,
                robotHost: robotHost,
                robotPort: robotPort
            )
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func addViaPointFromPrompt() async {
        let generation = resetGeneration
        guard let sessionID = activeRouteSessionID else {
            routeSessionErrorMessage = "当前没有可编辑的路线会话。"
            flowStage = .error
            return
        }
        guard let currentLocation = mapViewModel.userCoordinate else {
            routeSessionErrorMessage = "缺少当前位置，无法搜索途经点。"
            flowStage = .error
            return
        }

        let prompt = viaPointPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            routeSessionErrorMessage = "请输入途经点需求，例如便利店、咖啡店。"
            flowStage = .error
            return
        }

        flowStage = .addingViaPoint
        isLoading = true
        defer { isLoading = false }
        routeSessionErrorMessage = nil
        do {
            let recommendation = try await recommendationService.recommend(
                prompt: prompt,
                currentLocation: currentLocation,
                mode: activeRouteSession?.mode ?? "foot",
                maxCandidates: 1,
                searchRadius: 2000
            )
            guard generation == resetGeneration else { return }

            guard let candidate = recommendation.recommendations.first else {
                routeSessionErrorMessage = "未找到可添加的途经点，请换个说法。"
                flowStage = .error
                return
            }

            let locationText = candidate.location?.contains(",") == true
                ? candidate.location!
                : "\(candidate.coordinates.longitude),\(candidate.coordinates.latitude)"
            let result = try await routeService.addViaPoint(
                sessionID: sessionID,
                viaPoint: ViaPoint(
                    location: locationText,
                    name: candidate.name,
                    stopType: "via",
                    insertBeforeIndex: nil,
                    coordinates: candidate.coordinates
                ),
                insertBeforeIndex: nil
            )
            guard generation == resetGeneration else { return }
            applyRouteSessionResult(result, destinationName: selectedCandidate?.title ?? mapViewModel.destinationName)
            if let returnedSessionID = result.session?.sessionId,
               returnedSessionID != sessionID {
                connectRouteSession(sessionID: returnedSessionID)
            }
            viaPointPrompt = ""
            flowStage = .sessionActive
        } catch {
            guard generation == resetGeneration else { return }
            routeSessionErrorMessage = error.localizedDescription
            flowStage = .error
        }
    }

    func refreshActiveRouteSession(forceReplan: Bool = false) async {
        let generation = resetGeneration
        guard let sessionID = activeRouteSessionID else {
            routeSessionErrorMessage = "当前没有可刷新的路线会话。"
            flowStage = .error
            return
        }

        flowStage = .replanning
        isLoading = true
        defer { isLoading = false }
        routeSessionErrorMessage = nil
        do {
            let result = try await routeService.getSession(sessionID: sessionID, forceReplan: forceReplan)
            guard generation == resetGeneration else { return }
            applyRouteSessionResult(result, destinationName: selectedCandidate?.title ?? mapViewModel.destinationName)
            flowStage = .sessionActive
        } catch {
            guard generation == resetGeneration else { return }
            routeSessionErrorMessage = error.localizedDescription
            flowStage = .error
        }
    }

    func replanActiveRouteSessionFromCurrentLocation() async {
        let generation = resetGeneration
        guard let sessionID = activeRouteSessionID else {
            routeSessionErrorMessage = "当前没有可重规划的路线会话。"
            flowStage = .error
            return
        }
        guard let currentLocation = mapViewModel.userCoordinate else {
            routeSessionErrorMessage = "缺少当前位置，无法重规划。"
            flowStage = .error
            return
        }

        flowStage = .replanning
        isLoading = true
        defer { isLoading = false }
        routeSessionErrorMessage = nil
        do {
            let result = try await routeService.replan(
                sessionID: sessionID,
                origin: "\(currentLocation.longitude),\(currentLocation.latitude)"
            )
            guard generation == resetGeneration else { return }
            applyRouteSessionResult(result, destinationName: selectedCandidate?.title ?? mapViewModel.destinationName)
            flowStage = .sessionActive
        } catch {
            guard generation == resetGeneration else { return }
            routeSessionErrorMessage = error.localizedDescription
            flowStage = .error
        }
    }

    func pushWaypointsForActiveSession() async {
        let generation = resetGeneration
        guard let sessionID = activeRouteSessionID else {
            waypointPushStatusMessage = "当前没有可下发的路线会话。"
            flowStage = .error
            return
        }

        let host = robotHost.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !host.isEmpty else {
            waypointPushStatusMessage = "请输入机器人地址。"
            flowStage = .error
            return
        }

        guard let port = Int(robotPortText.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            waypointPushStatusMessage = "机器人端口必须是整数。"
            flowStage = .error
            return
        }

        flowStage = .pushingWaypoints
        isLoading = true
        defer { isLoading = false }
        waypointPushStatusMessage = nil
        do {
            let response = try await waypointPushService.pushWaypoints(
                sessionID: sessionID,
                robotHost: host,
                robotPort: port
            )
            guard generation == resetGeneration else { return }
            lastWaypointPushResponse = response
            if let totalCount = response.totalCount, let wsInfo = response.wsInfo {
                waypointPushStatusMessage = "已向 \(wsInfo) 下发 \(totalCount) 个 waypoint。"
            } else {
                waypointPushStatusMessage = "waypoint 下发完成。"
            }
            flowStage = .sessionActive
        } catch {
            guard generation == resetGeneration else { return }
            waypointPushStatusMessage = error.localizedDescription
            flowStage = .error
        }
    }

    func resetAll() {
        resetGeneration = UUID()
        speech.resetAll()
        voice.resetAll()
        clearRouteSessionState(disconnect: true)
        mapViewModel.resetAll()

        flowStage = .idle
        promptText = ""
        outputMode = .streaming
        candidates = []
        selectedCandidate = nil
        recommendationResponse = nil
        isLoading = false
        errorMessage = nil
        showingVoiceAssistant = false
        robotHost = Self.defaultRobotHost
        robotPortText = Self.defaultRobotPortText
    }

    func disconnectRouteSession() {
        webSocketRouteSessionClient.disconnect()
    }

    var activeRouteSessionID: String? {
        activeRouteSession?.sessionId
    }

    var currentViaPoints: [ViaPoint] {
        if let viaPoints = activeRouteSession?.viaPoints, !viaPoints.isEmpty {
            return viaPoints
        }
        return activeRoutePlan?.viaPoints ?? []
    }

    var routeSessionStatusText: String {
        guard let session = activeRouteSession else {
            return "未建立路线会话"
        }
        if let routeSessionErrorMessage, !routeSessionErrorMessage.isEmpty {
            return "会话 \(session.sessionId) 连接异常"
        }
        if isRouteSessionConnected {
            return "会话 \(session.sessionId) 已连接"
        }
        return "会话 \(session.sessionId) 已创建，等待连接"
    }

    var flowStageSummary: String {
        flowStage.summary
    }

    var routeSessionHeartbeatText: String? {
        guard let lastRouteSessionHeartbeatAt else { return nil }
        return "最近心跳 \(Self.heartbeatTimeFormatter.string(from: lastRouteSessionHeartbeatAt))"
    }

    private func bindNestedState() {
        speech.objectWillChange
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                self?.objectWillChange.send()
            }
            .store(in: &cancellables)

        voice.objectWillChange
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                self?.objectWillChange.send()
            }
            .store(in: &cancellables)

        webSocketRouteSessionClient.onConnectionStateChange = { [weak self] isConnected in
            self?.isRouteSessionConnected = isConnected
        }
        webSocketRouteSessionClient.onHeartbeat = { [weak self] heartbeatAt in
            self?.lastRouteSessionHeartbeatAt = heartbeatAt
        }
        webSocketRouteSessionClient.onError = { [weak self] message in
            self?.routeSessionErrorMessage = message
            self?.flowStage = .error
        }
        webSocketRouteSessionClient.onSnapshot = { [weak self] snapshot in
            guard let self else { return }
            self.applyRouteSessionResult(
                snapshot,
                destinationName: self.selectedCandidate?.title ?? self.mapViewModel.destinationName
            )
        }
    }

    private func clearRouteSessionState(disconnect: Bool) {
        if disconnect {
            webSocketRouteSessionClient.disconnect()
        }
        activeRouteSession = nil
        activeRoutePlan = nil
        isRouteSessionConnected = false
        lastRouteSessionHeartbeatAt = nil
        routeSessionErrorMessage = nil
        lastRouteSessionChangeReason = nil
        lastRoutePlanCacheHit = nil
        lastRoutePlanUpdateMode = nil
        viaPointPrompt = ""
        lastWaypointPushResponse = nil
        waypointPushStatusMessage = nil
    }

    private func connectRouteSession(sessionID: String) {
        routeSessionErrorMessage = nil
        lastRouteSessionHeartbeatAt = nil
        webSocketRouteSessionClient.connect(sessionID: sessionID)
    }

    private func applyRouteSessionResult(_ result: MCPRouteSessionResult, destinationName: String?) {
        activeRouteSession = result.session ?? activeRouteSession
        activeRoutePlan = result.routePlan ?? activeRoutePlan
        lastRouteSessionChangeReason = result.changeReason
        lastRoutePlanCacheHit = result.routePlanCacheHit
        lastRoutePlanUpdateMode = result.routePlanUpdateMode
        routeSessionErrorMessage = nil

        if let routePlan = result.routePlan {
            mapViewModel.applyRoutePlan(
                routePlan,
                destinationName: destinationName ?? result.session?.sessionName ?? mapViewModel.destinationName
            )
        }
        if activeRouteSession != nil {
            flowStage = .sessionActive
        }
    }

    private static let heartbeatTimeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "zh_CN")
        formatter.dateFormat = "HH:mm:ss"
        return formatter
    }()
}

import Foundation
import CoreLocation

// MARK: - Process Runner Errors

enum RealtimeProcessError: Error, LocalizedError {
    case missingScript
    case timeout
    case processFailed(String)
    case invalidResponse(String)

    var errorDescription: String? {
        switch self {
        case .missingScript:
            return "未找到 realtime_instruction_session_client.py 脚本"
        case .timeout:
            return "realtime_instruction 脚本调用超时"
        case .processFailed(let detail):
            return "realtime_instruction 调用失败: \(detail)"
        case .invalidResponse(let detail):
            return "realtime_instruction 响应解析失败: \(detail)"
        }
    }
}

@MainActor
final class CandidateSelectionViewModel: ObservableObject {
    @Published var promptInput = ""
    @Published var dialogHistory: [String] = []
    @Published var candidates: [AddressCandidate] = []
    @Published var selectedCandidate: AddressCandidate?
    @Published var statusText = "等待输入导航需求"
    @Published var isPlanning = false
    @Published var planningError: String?
    @Published var lastPlanningResult: MCPRecommendationResponse?
    @Published var lastRouteSessionResult: MCPRouteSessionResult?

    private let service: AddressCandidateService
    private let routeSessionService: RouteSessionService
    private var lastConfirmedText: String = ""
    private var lastLocation: CLLocationCoordinate2D?
    private var lastPlanningPrompt: String = ""
    private var refinementMessages: [String] = []
    private var activeSessionId: String?
    private var activeDestinationLocation: String?

    init(
        service: AddressCandidateService = MCPAddressCandidateService(),
        routeSessionService: RouteSessionService = MCPRouteSessionService()
    ) {
        self.service = service
        self.routeSessionService = routeSessionService
    }

    func fetchCandidates() async {
        let prompt = promptInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            statusText = "请输入地址相关 prompt"
            return
        }
        do {
            let result = try await service.fetchCandidates(prompt: prompt, history: dialogHistory)
            candidates = result
            selectedCandidate = nil
            dialogHistory.append("用户: \(prompt)")
            statusText = "收到 \(result.count) 个候选地址"
        } catch {
            statusText = "候选地址查询失败: \(error.localizedDescription)"
        }
    }

    func chooseCandidate(_ candidate: AddressCandidate) async {
        selectedCandidate = candidate
        statusText = "已选择: \(candidate.title)"

        guard let currentLocation = lastLocation else {
            statusText = "已选择: \(candidate.title)，等待定位后创建路线会话..."
            return
        }

        // Store destination location for via point use
        if let coord = candidate.coordinate {
            activeDestinationLocation = "\(coord.longitude),\(coord.latitude)"
        }

        isPlanning = true
        planningError = nil
        statusText = "已选择: \(candidate.title)，正在创建路线会话..."
        defer { isPlanning = false }

        do {
            let sessionResult = try await routeSessionService.createSession(
                origin: "\(currentLocation.longitude),\(currentLocation.latitude)",
                destination: candidate.coordinate.map { "\($0.longitude),\($0.latitude)" } ?? "",
                mode: "foot"
            )

            // Persist session ID for via-point flow
            if let sessionId = sessionResult.session?.sessionId {
                activeSessionId = sessionId
                NSLog("[CandidateVM] Created route session: %@", sessionId)

                // Auto-send waypoints to robot via WebSocket
                let origin = "\(currentLocation.longitude),\(currentLocation.latitude)"
                let destination = candidate.coordinate.map { "\($0.longitude),\($0.latitude)" } ?? ""
                do {
                    try await routeSessionService.sendWaypoints(
                        sessionId: sessionId,
                        origin: origin,
                        destination: destination,
                        mode: "foot"
                    )
                    // Update candidates to show the confirmed session route
                    if let session = sessionResult.session, let viaPointsList = session.viaPoints {
                        statusText = "路线已创建，waypoints 已发送到机器人，途经点: \(viaPointsList.count) 个"
                    } else {
                        statusText = "路线已创建，waypoints 已发送到机器人"
                    }
                } catch {
                    NSLog("[CandidateVM] sendWaypoints failed: %@", error.localizedDescription)
                    // Update candidates to show the confirmed session route (even if WS failed)
                    if let session = sessionResult.session, let viaPointsList = session.viaPoints {
                        statusText = "路线已创建，waypoints 发送失败: \(error.localizedDescription)，途经点: \(viaPointsList.count) 个"
                    } else {
                        statusText = "路线已创建，waypoints 发送失败: \(error.localizedDescription)"
                    }
                }
            } else {
                // No session ID — store result but skip waypoint sending
                lastRouteSessionResult = sessionResult
                if let session = sessionResult.session, let viaPointsList = session.viaPoints {
                    statusText = "路线会话已创建，途经点: \(viaPointsList.count) 个"
                } else {
                    statusText = "路线会话已创建"
                }
            }

            // Store result for ContentView → MapNavigationViewModel sync
            lastRouteSessionResult = sessionResult
        } catch {
            planningError = error.localizedDescription
            statusText = "已选择: \(candidate.title)，但路线会话创建失败: \(error.localizedDescription)"
            NSLog("[CandidateVM] Session creation failed: %@", error.localizedDescription)
        }
    }

    private func createSessionForDestination(
        origin: CLLocationCoordinate2D,
        destination: CLLocationCoordinate2D?,
        mode: String
    ) async {
        guard let destCoord = destination else { return }
        let originStr = "\(origin.longitude),\(origin.latitude)"
        let destStr = "\(destCoord.longitude),\(destCoord.latitude)"

        do {
            let result = try await routeSessionService.createSession(
                origin: originStr,
                destination: destStr,
                mode: mode
            )
            if let sessionId = result.session?.sessionId {
                activeSessionId = sessionId
                NSLog("[CandidateVM] Created route session: %@", sessionId)
            }
        } catch {
            NSLog("[CandidateVM] Failed to create session: %@", error.localizedDescription)
        }
    }

    func continueDialog(_ message: String) {
        let text = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        dialogHistory.append("继续对话: \(text)")
        statusText = "已记录多轮上下文"
    }

    private func isViaPointIntent(_ message: String) -> Bool {
        let viaKeywords = [
            "经过", "路过", "顺路", "途经", "中途", "绕道",
            "经过一下", "路过一下", "顺路经过", "顺路去", "绕路去",
            "中途去", "中间停一下", "中途停一下"
        ]
        let lowercased = message.lowercased()
        return viaKeywords.contains { lowercased.contains($0) }
    }

    private func addViaPointFlow(instruction: String) async {
        guard let sessionId = activeSessionId,
              let currentLocation = lastLocation else {
            statusText = "无活跃路线会话，无法添加途经点"
            return
        }

        isPlanning = true
        planningError = nil
        statusText = "正在添加途经点: \(instruction)..."
        defer { isPlanning = false }

        do {
            // Directly call realtime_instruction_session_client.py (bypasses MCPRouteSessionService bridge)
            let result = try await runRealtimeInstructionProcess(
                sessionId: sessionId,
                instruction: instruction,
                currentLat: currentLocation.latitude,
                currentLon: currentLocation.longitude,
                mode: "foot",
                radius: 3000,
                maxCandidates: 3
            )

            // Store result so ContentView → MapNavigationViewModel propagates the new route_plan
            lastRouteSessionResult = result

            // Store updated session ID if returned
            let resolvedSessionId: String?
            if let newSessionId = result.session?.sessionId {
                activeSessionId = newSessionId
                resolvedSessionId = newSessionId
            } else {
                resolvedSessionId = sessionId
            }

            // Auto-send updated waypoints to robot via WebSocket
            if let resolvedId = resolvedSessionId,
               let currentLoc = lastLocation,
               let destLoc = activeDestinationLocation {
                do {
                    try await routeSessionService.sendWaypoints(
                        sessionId: resolvedId,
                        origin: "\(currentLoc.longitude),\(currentLoc.latitude)",
                        destination: destLoc,
                        mode: "foot"
                    )
                    if let session = result.session, let viaPointsList = session.viaPoints {
                        candidates = []
                        statusText = "途经点【\(instruction)】已添加，waypoints 已更新到机器人，途经点: \(viaPointsList.count) 个"
                    } else {
                        statusText = "途经点【\(instruction)】已添加，waypoints 已更新到机器人"
                    }
                } catch {
                    NSLog("[CandidateVM] sendWaypoints failed after addViaPoint: %@", error.localizedDescription)
                    if let session = result.session, let viaPointsList = session.viaPoints {
                        candidates = []
                        statusText = "途经点已添加，waypoints 更新失败: \(error.localizedDescription)，途经点: \(viaPointsList.count) 个"
                    } else {
                        statusText = "途经点【\(instruction)】已添加，waypoints 更新失败: \(error.localizedDescription)"
                    }
                }
            } else {
                // Fallback: update status without waypoint sending
                if let session = result.session, let viaPointsList = session.viaPoints {
                    candidates = []
                    statusText = "途经点【\(instruction)】已添加，途经点: \(viaPointsList.count) 个"
                } else {
                    statusText = "途经点【\(instruction)】已添加"
                }
            }

            dialogHistory.append("添加途经点: \(instruction)")
        } catch {
            planningError = error.localizedDescription
            statusText = "途经点添加失败: \(error.localizedDescription)"
        }
    }

    func planFromConfirmedText(
        _ text: String,
        currentLocation: CLLocationCoordinate2D
    ) async {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else {
            statusText = "确认文本为空，无法规划"
            return
        }

        isPlanning = true
        planningError = nil
        statusText = "正在调用 MCP 规划..."
        selectedCandidate = nil
        lastConfirmedText = normalized
        refinementMessages = []
        let planningPrompt = buildPlanningPrompt(baseRequirement: normalized, refinements: [])
        lastPlanningPrompt = planningPrompt
        NSLog("[CandidateVM] Initial planning prompt=%@", planningPrompt)
        lastLocation = currentLocation
        defer { isPlanning = false }

        do {
            let result = try await service.planFromConfirmedText(
                confirmedText: planningPrompt,
                currentLocation: currentLocation,
                mode: "foot",
                maxCandidates: 5,
                searchRadius: 3000,
                selectedIndex: 1
            )
            dialogHistory.append("确认文本: \(normalized)")
            applyPlanningResult(result, successText: "规划完成")
        } catch {
            planningError = error.localizedDescription
            statusText = "规划失败: \(error.localizedDescription)"
        }
    }

    func continueDialogAndReplan(_ message: String) async {
        let text = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        guard let currentLocation = lastLocation else {
            statusText = "缺少定位信息，请先执行一次确认文本规划"
            return
        }
        guard !lastConfirmedText.isEmpty else {
            statusText = "缺少初始确认文本，请先执行一次确认文本规划"
            return
        }

        // Route via point intent to dedicated flow
        if isViaPointIntent(text) {
            await addViaPointFlow(instruction: text)
            return
        }

        continueDialog(text)
        refinementMessages.append(text)
        isPlanning = true
        planningError = nil
        statusText = "正在根据继续对话更新候选..."
        selectedCandidate = nil
        defer { isPlanning = false }

        let mergedPrompt = buildPlanningPrompt(
            baseRequirement: lastConfirmedText,
            refinements: refinementMessages
        )
        lastPlanningPrompt = mergedPrompt
        NSLog("[CandidateVM] Continued planning prompt=%@", mergedPrompt)

        do {
            let result = try await service.planFromConfirmedText(
                confirmedText: mergedPrompt,
                currentLocation: currentLocation,
                mode: "foot",
                maxCandidates: 5,
                searchRadius: 3000,
                selectedIndex: 1
            )
            applyPlanningResult(result, successText: "继续对话完成")
        } catch {
            planningError = error.localizedDescription
            statusText = "继续对话规划失败: \(error.localizedDescription)"
        }
    }

    private func buildPlanningPrompt(baseRequirement: String, refinements: [String]) -> String {
        guard !refinements.isEmpty else {
            return baseRequirement
        }
        let refinementText = refinements.enumerated().map { index, text in
            "\(index + 1). \(text)"
        }.joined(separator: "\n")
        return """
        原始导航需求：
        \(baseRequirement)

        补充约束（后面的约束优先级更高）：
        \(refinementText)

        请严格根据以上需求返回候选目的地；若约束冲突，以补充约束为准。
        """
    }

    // MARK: - Realtime Instruction Process (Third-click bypass)

    private func runRealtimeInstructionProcess(
        sessionId: String,
        instruction: String,
        currentLat: Double,
        currentLon: Double,
        mode: String,
        radius: Double,
        maxCandidates: Int
    ) async throws -> MCPRouteSessionResult {
        let realtimeTimeoutSeconds: TimeInterval = 120.0

        let scriptCandidates = Self.resolveRealtimeScriptCandidates()
        guard let scriptPath = scriptCandidates.first(where: { FileManager.default.fileExists(atPath: $0) }) else {
            NSLog("[CandidateVM] realtime script not found, candidates: %@", scriptCandidates.joined(separator: ", "))
            throw RealtimeProcessError.missingScript
        }

        let scriptURL = URL(fileURLWithPath: scriptPath)
        let openStreetRoot = scriptURL.deletingLastPathComponent().deletingLastPathComponent().path
        let env = buildRealtimeEnvironment()
        let pythonCandidates = [
            ProcessInfo.processInfo.environment["PYTHON_BIN"],
            "/opt/anaconda3/envs/openstreet/bin/python3",
            "/opt/anaconda3/envs/openstreet/bin/python"
        ]
        .compactMap { $0 }
        .filter { !$0.isEmpty }

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let args = [
                    "--lat", String(currentLat),
                    "--lon", String(currentLon),
                    "--mode", mode,
                    "--session-id", sessionId,
                    "--append-instruction", instruction,
                    "--radius", String(radius),
                    "--max-candidates", String(maxCandidates),
                    "--non-interactive"
                ]
                let pythonRunners = pythonCandidates.map { [$0, scriptPath] + args }
                NSLog("[CandidateVM] runRealtimeInstructionProcess: script=%@", scriptPath)
                NSLog("[CandidateVM] runRealtimeInstructionProcess: args=%@", args.joined(separator: " "))
                NSLog("[CandidateVM] runRealtimeInstructionProcess: mode=%@, radius=%@, maxCandidates=%@", mode, String(radius), String(maxCandidates))

                var runnerErrors: [String] = []
                for runner in pythonRunners {
                    let process = Process()
                    let outputPipe = Pipe()
                    let errorPipe = Pipe()
                    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
                    process.arguments = runner
                    process.environment = env
                    process.currentDirectoryURL = URL(fileURLWithPath: openStreetRoot)
                    process.standardOutput = outputPipe
                    process.standardError = errorPipe

                    NSLog("[CandidateVM] trying runner: %@", runner.prefix(4).joined(separator: " "))

                    do {
                        try process.run()
                    } catch {
                        runnerErrors.append("启动失败(\(runner.first ?? "unknown")): \(error.localizedDescription)")
                        continue
                    }

                    let deadline = Date().addingTimeInterval(realtimeTimeoutSeconds)
                    while process.isRunning && Date() < deadline {
                        Thread.sleep(forTimeInterval: 0.05)
                    }

                    if process.isRunning {
                        process.terminate()
                        NSLog("[CandidateVM] runner TIMED OUT after %@s: %@", String(realtimeTimeoutSeconds), runner.first ?? "unknown")
                        runnerErrors.append("超时(\(runner.first ?? "unknown"))")
                        continue
                    }

                    process.waitUntilExit()

                    let stdout = String(
                        data: outputPipe.fileHandleForReading.readDataToEndOfFile(),
                        encoding: .utf8
                    ) ?? ""
                    let stderr = String(
                        data: errorPipe.fileHandleForReading.readDataToEndOfFile(),
                        encoding: .utf8
                    ) ?? ""

                    guard process.terminationStatus == 0 else {
                        let reason = stderr.isEmpty ? stdout : stderr
                        NSLog("[CandidateVM] runner failed stderr: %@", String(reason.prefix(500)))
                        runnerErrors.append("退出非0(\(runner.first ?? "unknown")): \(reason)")
                        continue
                    }

                    do {
                        let jsonText = try Self.extractTopLevelJSONObjectForRealtime(from: stdout)
                        guard let jsonData = jsonText.data(using: .utf8) else {
                            runnerErrors.append("JSON 编码失败(\(runner.first ?? "unknown"))")
                            continue
                        }

                        let result = try Self.parseRealtimeResult(jsonData: jsonData, sessionId: sessionId)
                        NSLog("[CandidateVM] SUCCESS with runner: %@", runner.first ?? "unknown")
                        continuation.resume(returning: result)
                        return
                    } catch {
                        NSLog("[CandidateVM] parse/JSON error: %@", error.localizedDescription)
                        runnerErrors.append("解析失败(\(runner.first ?? "unknown")): \(error.localizedDescription)")
                        continue
                    }
                }

                let error = RealtimeProcessError.processFailed(runnerErrors.joined(separator: "\n"))
                continuation.resume(throwing: error)
            }
        }
    }

    nonisolated private static func extractTopLevelJSONObjectForRealtime(from output: String) throws -> String {
        // The script outputs "---RESULT---" before the JSON block
        if let resultStart = output.range(of: "---RESULT---") {
            let jsonStart = resultStart.upperBound
            let afterMarker = String(output[jsonStart...]).trimmingCharacters(in: .whitespacesAndNewlines)
            return try Self.extractJSONObject(from: afterMarker)
        }
        // Fallback: find first { and extract
        return try Self.extractJSONObject(from: output)
    }

    nonisolated private static func extractJSONObject(from text: String) throws -> String {
        guard let startIndex = text.firstIndex(of: "{") else {
            throw RealtimeProcessError.invalidResponse("未找到 JSON 起始")
        }

        var depth = 0
        var inString = false
        var escaping = false
        var endIndex: String.Index?

        var index = startIndex
        while index < text.endIndex {
            let char = text[index]
            if inString {
                if escaping {
                    escaping = false
                } else if char == "\\" {
                    escaping = true
                } else if char == "\"" {
                    inString = false
                }
            } else {
                if char == "\"" {
                    inString = true
                } else if char == "{" {
                    depth += 1
                } else if char == "}" {
                    depth -= 1
                    if depth == 0 {
                        endIndex = index
                        break
                    }
                }
            }
            index = text.index(after: index)
        }

        guard let endIndex else {
            throw RealtimeProcessError.invalidResponse("JSON 不完整")
        }
        return String(text[startIndex...endIndex])
    }

    nonisolated private static func parseRealtimeResult(jsonData: Data, sessionId: String) throws -> MCPRouteSessionResult {
        guard let jsonObject = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw RealtimeProcessError.invalidResponse("JSON 非对象")
        }

        // Parse session (may come from root-level "session" or nested)
        var session: RouteSession?
        if let sessionDict = jsonObject["session"] as? [String: Any] {
            session = Self.parseRouteSession(sessionDict, sessionId: sessionId)
        } else {
            // Build from top-level fields
            session = RouteSession(
                sessionId: sessionId,
                sessionName: jsonObject["session_name"] as? String,
                origin: jsonObject["origin"] as? String,
                destination: jsonObject["destination"] as? String,
                mode: jsonObject["mode"] as? String,
                viaPoints: nil,
                createdAt: nil
            )
        }

        // Parse route_plan from the new structure
        var routePlan: MCPRoutePlan?
        if let routePlanDict = jsonObject["route_plan"] as? [String: Any] {
            routePlan = Self.parseRoutePlan(routePlanDict)
        }

        // Also support legacy top-level via_points and route_summary
        var viaPoints: [ViaPoint] = []
        if let viaPointsArray = jsonObject["via_points"] as? [[String: Any]] {
            viaPoints = viaPointsArray.compactMap { Self.parseViaPoint($0) }
        }

        // Merge via_points into session if not already set
        if let existingSession = session, let existingViaPoints = existingSession.viaPoints, existingViaPoints.isEmpty {
            session = RouteSession(
                sessionId: existingSession.sessionId,
                sessionName: existingSession.sessionName,
                origin: existingSession.origin,
                destination: existingSession.destination,
                mode: existingSession.mode,
                viaPoints: viaPoints.isEmpty ? nil : viaPoints,
                createdAt: existingSession.createdAt
            )
        }

        // Parse resolved_via if present
        var resolvedVia: ResolvedVia?
        if let resolvedDict = jsonObject["resolved_via"] as? [String: Any] {
            resolvedVia = Self.parseResolvedVia(resolvedDict)
        }

        return MCPRouteSessionResult(
            session: session,
            routePlan: routePlan,
            resolvedVia: resolvedVia,
            addResult: nil,
            replanResult: nil
        )
    }

    nonisolated private static func parseRouteSession(_ dict: [String: Any], sessionId: String) -> RouteSession? {
        var viaPoints: [ViaPoint] = []
        if let viaPointsArray = dict["via_points"] as? [[String: Any]] {
            viaPoints = viaPointsArray.compactMap { Self.parseViaPoint($0) }
        }

        return RouteSession(
            sessionId: dict["session_id"] as? String ?? sessionId,
            sessionName: dict["session_name"] as? String,
            origin: dict["origin"] as? String,
            destination: dict["destination"] as? String,
            mode: dict["mode"] as? String,
            viaPoints: viaPoints.isEmpty ? nil : viaPoints,
            createdAt: dict["created_at"] as? String
        )
    }

    nonisolated private static func parseRoutePlan(_ dict: [String: Any]) -> MCPRoutePlan {
        // Parse route_summary
        var routeSummary: MCPRouteSummary?
        if let summaryDict = dict["route_summary"] as? [String: Any] {
            routeSummary = MCPRouteSummary(
                distanceM: summaryDict["distance_m"] as? Double,
                durationS: summaryDict["duration_s"] as? Double,
                durationMin: summaryDict["duration_min"] as? Double
            )
        }

        // Parse stop_sequence
        var stopSequence: [StopSequenceItem] = []
        if let stopArray = dict["stop_sequence"] as? [[String: Any]] {
            stopSequence = stopArray.compactMap { Self.parseStopSequenceItem($0) }
        }

        // Parse route_polyline
        var routePolyline: [MCPCoordinates] = []
        if let polylineArray = dict["route_polyline"] as? [[String: Any]] {
            routePolyline = polylineArray.compactMap { Self.parseCoordinates($0) }
        }

        // Parse coarse_waypoints
        var coarseWaypoints: [MCPWaypoint] = []
        if let waypointsArray = dict["coarse_waypoints"] as? [[String: Any]] {
            coarseWaypoints = waypointsArray.compactMap { Self.parseMCPWaypoint($0) }
        }

        let legCount = dict["leg_count"] as? Int

        return MCPRoutePlan(
            routeSummary: routeSummary,
            stopSequence: stopSequence.isEmpty ? nil : stopSequence,
            routePolyline: routePolyline.isEmpty ? nil : routePolyline,
            coarseWaypoints: coarseWaypoints.isEmpty ? nil : coarseWaypoints,
            legCount: legCount
        )
    }

    nonisolated private static func parseStopSequenceItem(_ dict: [String: Any]) -> StopSequenceItem? {
        // Parse route_leg_summary
        var legSummary: RouteLegSummary?
        if let legDict = dict["route_leg_summary"] as? [String: Any] {
            legSummary = RouteLegSummary(
                distanceM: legDict["distance_m"] as? Double,
                durationMin: legDict["duration_min"] as? Double,
                fromStop: legDict["from_stop"] as? String,
                toStop: legDict["to_stop"] as? String
            )
        }

        return StopSequenceItem(
            stopType: dict["stop_type"] as? String ?? dict["type"] as? String,
            name: dict["name"] as? String,
            location: dict["location"] as? String,
            routeLegSummary: legSummary
        )
    }

    nonisolated private static func parseCoordinates(_ dict: [String: Any]) -> MCPCoordinates? {
        if let lat = dict["latitude"] as? Double, let lon = dict["longitude"] as? Double {
            return MCPCoordinates(latitude: lat, longitude: lon)
        }
        return nil
    }

    nonisolated private static func parseMCPWaypoint(_ dict: [String: Any]) -> MCPWaypoint? {
        var longitude: Double = 0
        var latitude: Double = 0

        if let lon = dict["longitude"] as? Double, let lat = dict["latitude"] as? Double {
            longitude = lon
            latitude = lat
        } else if let coordDict = dict["coordinate"] as? [String: Any],
                  let lon = coordDict["longitude"] as? Double,
                  let lat = coordDict["latitude"] as? Double {
            longitude = lon
            latitude = lat
        } else if let location = dict["location"] as? String {
            let parts = location.split(separator: ",")
            if parts.count == 2, let lon = Double(parts[0]), let lat = Double(parts[1]) {
                longitude = lon
                latitude = lat
            }
        }

        guard longitude != 0 || latitude != 0 else { return nil }

        return MCPWaypoint(
            longitude: longitude,
            latitude: latitude,
            instruction: dict["instruction"] as? String,
            road: dict["road"] as? String,
            action: dict["action"] as? String
        )
    }

    nonisolated private static func parseViaPoint(_ dict: [String: Any]) -> ViaPoint? {
        guard let name = dict["name"] as? String else { return nil }

        var coords: MCPCoordinates?
        if let coordsDict = dict["coordinates"] as? [String: Any],
           let lat = coordsDict["latitude"] as? Double,
           let lon = coordsDict["longitude"] as? Double {
            coords = MCPCoordinates(latitude: lat, longitude: lon)
        }

        return ViaPoint(
            location: dict["location"] as? String,
            name: name,
            stopType: dict["stop_type"] as? String,
            insertBeforeIndex: dict["insert_before_index"] as? Int,
            coordinates: coords
        )
    }

    nonisolated private static func parseResolvedVia(_ dict: [String: Any]) -> ResolvedVia? {
        var rawItem: MCPRecommendationItem?
        if let rawDict = dict["raw"] as? [String: Any] {
            var coords = MCPCoordinates(latitude: 0, longitude: 0)
            if let coordsDict = rawDict["coordinates"] as? [String: Any],
               let lat = coordsDict["latitude"] as? Double,
               let lon = coordsDict["longitude"] as? Double {
                coords = MCPCoordinates(latitude: lat, longitude: lon)
            }
            rawItem = MCPRecommendationItem(
                id: rawDict["id"] as? String ?? "",
                name: rawDict["name"] as? String ?? "",
                coordinates: coords,
                score: rawDict["score"] as? Double,
                routeSummary: nil,
                location: rawDict["location"] as? String,
                tags: nil
            )
        }

        return ResolvedVia(
            name: dict["name"] as? String,
            location: dict["location"] as? String,
            raw: rawItem
        )
    }

    private func buildRealtimeEnvironment() -> [String: String] {
        var env = [String: String]()
        for (key, value) in ProcessInfo.processInfo.environment {
            env[key] = value
        }

        let fallbackPath = [
            "/opt/anaconda3/envs/openstreet/bin",
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin"
        ].joined(separator: ":")

        let existingPath = env["PATH"] ?? ""
        env["PATH"] = "\(fallbackPath):\(existingPath)"
        env["CONDA_PREFIX"] = "/opt/anaconda3/envs/openstreet"
        env["CONDA_DEFAULT_ENV"] = "openstreet"
        env["PYTHONPATH"] = "/opt/anaconda3/envs/openstreet/lib/python3.12/site-packages:/Users/minruiliu/Nav/OpenStreetMcp/src"

        return env
    }

    private static func resolveRealtimeScriptCandidates() -> [String] {
        let envScript = ProcessInfo.processInfo.environment["REALTIME_INSTRUCTION_SCRIPT_PATH"]
        let relativeCandidates = [
            "../../examples/realtime_instruction_session_client.py",
            "examples/realtime_instruction_session_client.py"
        ]

        var rawCandidates: [String] = []
        if let envScript, !envScript.isEmpty {
            rawCandidates.append(envScript)
        }

        for baseURL in candidateBaseDirectoriesForRealtime() {
            for relativePath in relativeCandidates {
                rawCandidates.append(
                    baseURL.appendingPathComponent(relativePath).standardizedFileURL.path
                )
            }
        }

        var resolved: [String] = []
        var seen: Set<String> = []
        for candidate in rawCandidates where !candidate.isEmpty {
            let normalized = URL(fileURLWithPath: candidate).standardizedFileURL.path
            if seen.insert(normalized).inserted {
                resolved.append(normalized)
            }
        }
        NSLog("[CandidateVM] resolveRealtimeScriptCandidates: resolved=%@", resolved.joined(separator: ", "))
        return resolved
    }

    private static func candidateBaseDirectoriesForRealtime() -> [URL] {
        var directories: [URL] = []
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        directories.append(cwd)

        if Bundle.main.bundleURL.pathExtension == "app" {
            let bundleURL = Bundle.main.bundleURL
            let buildDirectoryURL = bundleURL.deletingLastPathComponent()
            let appProjectRootURL = buildDirectoryURL.deletingLastPathComponent()
            let openStreetRootURL = appProjectRootURL
                .deletingLastPathComponent()
                .deletingLastPathComponent()

            directories.append(appProjectRootURL)
            directories.append(openStreetRootURL)
        }

        var uniqueDirectories: [URL] = []
        var seen: Set<String> = []
        for directory in directories {
            let normalized = directory.standardizedFileURL.path
            if seen.insert(normalized).inserted {
                uniqueDirectories.append(directory.standardizedFileURL)
            }
        }
        return uniqueDirectories
    }

    func resetAll() {
        promptInput = ""
        dialogHistory = []
        candidates = []
        selectedCandidate = nil
        statusText = "等待输入导航需求"
        isPlanning = false
        planningError = nil
        lastPlanningResult = nil
        lastRouteSessionResult = nil
        lastConfirmedText = ""
        lastLocation = nil
        lastPlanningPrompt = ""
        refinementMessages = []
        activeSessionId = nil
        activeDestinationLocation = nil
    }

    private func applyPlanningResult(_ result: MCPRecommendationResponse, successText: String) {
        lastPlanningResult = result
        candidates = result.recommendations.map { $0.toAddressCandidate() }
        planningError = nil
        statusText = "\(successText)，收到 \(candidates.count) 个候选地址"
    }
}

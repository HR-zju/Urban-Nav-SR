import Foundation
import CoreLocation

// MARK: - Route Session Service Protocol

protocol RouteSessionService {
    /// Create a new route session (origin → destination).
    func createSession(
        origin: String,
        destination: String,
        mode: String
    ) async throws -> MCPRouteSessionResult

    /// Add a via point to an existing session using direct coordinates.
    func addViaPoint(
        sessionId: String,
        viaLocation: String,
        viaName: String
    ) async throws -> MCPRouteSessionResult

    /// Replan route from a new origin (e.g., user moved).
    func replan(
        sessionId: String,
        origin: String
    ) async throws -> MCPRouteSessionResult

    /// Resolve natural-language instruction to a location and add as via point,
    /// then replan. This is the primary "add via point" flow for user dialogs.
    func resolveAndAddVia(
        sessionId: String,
        instruction: String,
        currentLat: Double,
        currentLon: Double,
        origin: String,
        mode: String,
        radius: Int,
        maxCandidates: Int
    ) async throws -> MCPRouteSessionResult

    /// Extract waypoints from an existing route session and send to the robot
    /// via WebSocket. Uses the MCP session's cached route_plan to avoid
    /// re-creating the session (zero QPS cost).
    func sendWaypoints(
        sessionId: String,
        origin: String,
        destination: String,
        mode: String
    ) async throws
}

// MARK: - Errors

enum RouteSessionServiceError: Error, LocalizedError {
    case missingBridgeScript
    case processFailed(String)
    case timeout
    case invalidResponse(String)
    case noSessionId
    case noRoutePlan
    case sendWaypointsFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingBridgeScript:
            return "未找到 route_session_bridge.py，请检查路径配置。"
        case .processFailed(let detail):
            return "Route session 调用失败: \(detail)"
        case .timeout:
            return "Route session 调用超时。"
        case .invalidResponse(let detail):
            return "Route session 响应解析失败: \(detail)"
        case .noSessionId:
            return "响应中缺少 session_id。"
        case .noRoutePlan:
            return "响应中缺少 route_plan。"
        case .sendWaypointsFailed(let detail):
            return "Waypoints 发送失败: \(detail)"
        }
    }
}

// MARK: - MCP Route Session Service Implementation

struct MCPRouteSessionService: RouteSessionService {
    private let timeoutSeconds: TimeInterval
    private let bridgeScriptCandidates: [String]

    init(timeoutSeconds: TimeInterval = 180.0) {
        let envScript = ProcessInfo.processInfo.environment["ROUTE_SESSION_BRIDGE_SCRIPT_PATH"]
        self.timeoutSeconds = timeoutSeconds
        self.bridgeScriptCandidates = Self.resolveBridgeScriptCandidates(envScript: envScript)
    }

    // MARK: - Protocol Methods

    func createSession(
        origin: String,
        destination: String,
        mode: String
    ) async throws -> MCPRouteSessionResult {
        try await runBridgeProcess(
            action: "create",
            args: [
                "--origin", origin,
                "--destination", destination,
                "--mode", mode,
            ]
        )
    }

    func addViaPoint(
        sessionId: String,
        viaLocation: String,
        viaName: String
    ) async throws -> MCPRouteSessionResult {
        try await runBridgeProcess(
            action: "add_via",
            args: [
                "--session-id", sessionId,
                "--via-location", viaLocation,
                "--via-name", viaName,
            ]
        )
    }

    func replan(
        sessionId: String,
        origin: String
    ) async throws -> MCPRouteSessionResult {
        try await runBridgeProcess(
            action: "replan",
            args: [
                "--session-id", sessionId,
                "--origin", origin,
            ]
        )
    }

    func resolveAndAddVia(
        sessionId: String,
        instruction: String,
        currentLat: Double,
        currentLon: Double,
        origin: String,
        mode: String,
        radius: Int,
        maxCandidates: Int
    ) async throws -> MCPRouteSessionResult {
        try await runBridgeProcess(
            action: "resolve_and_add_via",
            args: [
                "--session-id", sessionId,
                "--instruction", instruction,
                "--current-lat", String(currentLat),
                "--current-lon", String(currentLon),
                "--origin", origin,
                "--mode", mode,
                "--radius", String(radius),
                "--max-candidates", String(maxCandidates),
            ]
        )
    }

    func sendWaypoints(
        sessionId: String,
        origin: String,
        destination: String,
        mode: String
    ) async throws {
        guard let scriptPath = bridgeScriptCandidates.first(where: { FileManager.default.fileExists(atPath: $0) }) else {
            throw RouteSessionServiceError.missingBridgeScript
        }
        let scriptURL = URL(fileURLWithPath: scriptPath)
        let openStreetRoot = scriptURL.deletingLastPathComponent().deletingLastPathComponent().path
        let env = buildBridgeEnvironment()
        let pythonCandidates = [
            ProcessInfo.processInfo.environment["PYTHON_BIN"],
            "/opt/anaconda3/envs/openstreet/bin/python3"
        ]
            .compactMap { $0 }
            .filter { !$0.isEmpty }
        let primaryPython = pythonCandidates.first ?? "/opt/anaconda3/envs/openstreet/bin/python3"

        let robotHost = ProcessInfo.processInfo.environment["ROBOT_WS_HOST"] ?? "192.168.168.110"
        let robotPort = ProcessInfo.processInfo.environment["ROBOT_WS_PORT"] ?? "8765"

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                let args = [
                    "--action", "send_ws",
                    "--session-id", sessionId,
                    "--origin", origin,
                    "--destination", destination,
                    "--mode", mode,
                    "--robot-host", robotHost,
                    "--robot-port", robotPort,
                ]

                NSLog("[MCPRouteSessionService] sendWaypoints: script=%@", scriptPath)
                NSLog("[MCPRouteSessionService] sendWaypoints: sessionId=%@, host=%@:%@", sessionId, robotHost, robotPort)

                let process = Process()
                let outputPipe = Pipe()
                let errorPipe = Pipe()
                let runner = [primaryPython, scriptPath] + args
                process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
                process.arguments = runner
                process.environment = env
                process.currentDirectoryURL = URL(fileURLWithPath: openStreetRoot)
                process.standardOutput = outputPipe
                process.standardError = errorPipe

                do {
                    try process.run()
                } catch {
                    continuation.resume(throwing: RouteSessionServiceError.processFailed("启动失败: \(error.localizedDescription)"))
                    return
                }

                let deadline = Date().addingTimeInterval(60.0)
                while process.isRunning && Date() < deadline {
                    Thread.sleep(forTimeInterval: 0.05)
                }

                if process.isRunning {
                    process.terminate()
                    continuation.resume(throwing: RouteSessionServiceError.timeout)
                    return
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
                    NSLog("[MCPRouteSessionService] sendWaypoints failed: %@", String(reason.prefix(500)))
                    continuation.resume(throwing: RouteSessionServiceError.sendWaypointsFailed(reason))
                    return
                }

                // Parse the response to extract status
                do {
                    let jsonText = try self.extractTopLevelJSONObject(from: stdout)
                    guard let jsonData = jsonText.data(using: .utf8),
                          let json = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
                        continuation.resume(throwing: RouteSessionServiceError.invalidResponse("非 JSON 响应"))
                        return
                    }

                    let status = json["status"] as? String ?? ""
                    let wsInfo = json["ws_info"] as? String ?? ""
                    let totalCount = json["total_count"] as? Int ?? 0

                    if status == "sent" {
                        NSLog("[MCPRouteSessionService] sendWaypoints: sent %d waypoints to %@", totalCount, wsInfo)
                        continuation.resume()
                    } else {
                        NSLog("[MCPRouteSessionService] sendWaypoints failed: status=%@, ws_info=%@", status, wsInfo)
                        continuation.resume(throwing: RouteSessionServiceError.sendWaypointsFailed("发送失败: \(status) - \(wsInfo)"))
                    }
                } catch {
                    continuation.resume(throwing: RouteSessionServiceError.invalidResponse("响应解析失败: \(error.localizedDescription)"))
                }
            }
        }
    }

    // MARK: - Bridge Process

    private func runBridgeProcess(action: String, args: [String]) async throws -> MCPRouteSessionResult {
        guard let scriptPath = bridgeScriptCandidates.first(where: { FileManager.default.fileExists(atPath: $0) }) else {
            throw RouteSessionServiceError.missingBridgeScript
        }
        let scriptURL = URL(fileURLWithPath: scriptPath)
        let openStreetRoot = scriptURL.deletingLastPathComponent().deletingLastPathComponent().path
        let env = buildBridgeEnvironment()
        let pythonCandidates = [
            ProcessInfo.processInfo.environment["PYTHON_BIN"],
            "/opt/anaconda3/envs/openstreet/bin/python3"
        ]
            .compactMap { $0 }
            .filter { !$0.isEmpty }
        let primaryPython = pythonCandidates.first ?? "/opt/anaconda3/envs/openstreet/bin/python3"

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let commonArguments = ["--action", action] + args
                    let runners: [[String]] = [[primaryPython, scriptPath] + commonArguments]

                    NSLog("[MCPRouteSessionService] action=%@, script=%@", action, scriptPath)
                    NSLog("[MCPRouteSessionService] args=%@", args.joined(separator: ", "))

                    var runnerErrors: [String] = []
                    for runner in runners {
                        let process = Process()
                        let outputPipe = Pipe()
                        let errorPipe = Pipe()
                        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
                        process.arguments = runner
                        process.environment = env
                        process.currentDirectoryURL = URL(fileURLWithPath: openStreetRoot)
                        process.standardOutput = outputPipe
                        process.standardError = errorPipe
                        NSLog("[MCPRouteSessionService] trying runner: %@", runner.prefix(4).joined(separator: " "))

                        do {
                            try process.run()
                        } catch {
                            runnerErrors.append("启动失败(\(runner.first ?? "unknown")): \(error.localizedDescription)")
                            continue
                        }

                        let deadline = Date().addingTimeInterval(self.timeoutSeconds)
                        while process.isRunning && Date() < deadline {
                            Thread.sleep(forTimeInterval: 0.05)
                        }

                        if process.isRunning {
                            process.terminate()
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
                            runnerErrors.append("退出非0(\(runner.first ?? "unknown")): \(reason)")
                            continue
                        }

                        do {
                            let jsonText = try self.extractTopLevelJSONObject(from: stdout)
                            guard let jsonData = jsonText.data(using: .utf8) else {
                                runnerErrors.append("JSON 编码失败(\(runner.first ?? "unknown"))")
                                continue
                            }
                            let decoded = try JSONDecoder().decode(MCPRouteSessionResult.self, from: jsonData)
                            continuation.resume(returning: decoded)
                            return
                        } catch {
                            runnerErrors.append("解析失败(\(runner.first ?? "unknown")): \(error.localizedDescription)")
                            continue
                        }
                    }

                    throw RouteSessionServiceError.processFailed(runnerErrors.joined(separator: "\n"))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func extractTopLevelJSONObject(from output: String) throws -> String {
        guard let startIndex = output.firstIndex(of: "{") else {
            throw RouteSessionServiceError.invalidResponse("未找到 JSON 起始")
        }

        var depth = 0
        var inString = false
        var escaping = false
        var endIndex: String.Index?

        var index = startIndex
        while index < output.endIndex {
            let char = output[index]
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
            index = output.index(after: index)
        }

        guard let endIndex else {
            throw RouteSessionServiceError.invalidResponse("JSON 不完整")
        }
        return String(output[startIndex...endIndex])
    }

    private func buildBridgeEnvironment() -> [String: String] {
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

        NSLog("[MCPRouteSessionService] buildBridgeEnvironment: final PATH=%@", env["PATH"] ?? "nil")
        NSLog("[MCPRouteSessionService] buildBridgeEnvironment: CONDA_PREFIX=%@", env["CONDA_PREFIX"] ?? "nil")
        NSLog("[MCPRouteSessionService] buildBridgeEnvironment: PYTHONPATH=%@", env["PYTHONPATH"] ?? "nil")
        return env
    }

    private static func resolveBridgeScriptCandidates(envScript: String?) -> [String] {
        var rawCandidates: [String] = []

        if let envScript, !envScript.isEmpty {
            rawCandidates.append(envScript)
        }

        let relativeCandidates = [
            "../../examples/route_session_bridge.py",
            "examples/route_session_bridge.py"
        ]

        for baseURL in candidateBaseDirectories() {
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
        return resolved
    }

    private static func candidateBaseDirectories() -> [URL] {
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
}

// MARK: - Mock Route Session Service (for development/testing)

final class MockRouteSessionService: RouteSessionService {
    private var sessions: [String: RouteSession] = [:]
    private var nextSessionId = 1

    func createSession(
        origin: String,
        destination: String,
        mode: String
    ) async throws -> MCPRouteSessionResult {
        let sessionId = "mock-session-\(nextSessionId)"
        nextSessionId += 1

        let session = RouteSession(
            sessionId: sessionId,
            sessionName: "nav_app_session",
            origin: origin,
            destination: destination,
            mode: mode,
            viaPoints: [],
            createdAt: ISO8601DateFormatter().string(from: Date())
        )
        sessions[sessionId] = session

        return MCPRouteSessionResult(
            session: session,
            routePlan: MCPRoutePlan(
                routeSummary: MCPRouteSummary(distanceM: 1500, durationS: 900, durationMin: 15),
                stopSequence: [
                    StopSequenceItem(stopType: "origin", name: "起点", location: origin, routeLegSummary: nil),
                    StopSequenceItem(stopType: "destination", name: "目的地", location: destination, routeLegSummary: RouteLegSummary(distanceM: 1500, durationMin: 15, fromStop: "起点", toStop: "目的地"))
                ],
                routePolyline: [
                    MCPCoordinates(latitude: 31.2304, longitude: 121.4737)
                ],
                coarseWaypoints: [],
                legCount: 1
            ),
            resolvedVia: nil,
            addResult: nil,
            replanResult: nil
        )
    }

    func addViaPoint(
        sessionId: String,
        viaLocation: String,
        viaName: String
    ) async throws -> MCPRouteSessionResult {
        guard let session = sessions[sessionId] else {
            throw RouteSessionServiceError.processFailed("Session not found: \(sessionId)")
        }

        let newViaPoint = ViaPoint(
            location: viaLocation,
            name: viaName,
            stopType: "via",
            insertBeforeIndex: nil,
            coordinates: nil
        )

        var viaPoints = session.viaPoints ?? []
        viaPoints.append(newViaPoint)

        let updatedSession = RouteSession(
            sessionId: session.sessionId,
            sessionName: session.sessionName,
            origin: session.origin,
            destination: session.destination,
            mode: session.mode,
            viaPoints: viaPoints,
            createdAt: session.createdAt
        )
        sessions[sessionId] = updatedSession

        return MCPRouteSessionResult(
            session: updatedSession,
            routePlan: MCPRoutePlan(
                routeSummary: MCPRouteSummary(distanceM: 2200, durationS: 1320, durationMin: 22),
                stopSequence: [
                    StopSequenceItem(stopType: "origin", name: "起点", location: session.origin, routeLegSummary: nil),
                    StopSequenceItem(stopType: "via", name: viaName, location: viaLocation, routeLegSummary: RouteLegSummary(distanceM: 800, durationMin: 8, fromStop: "起点", toStop: viaName)),
                    StopSequenceItem(stopType: "destination", name: "目的地", location: session.destination, routeLegSummary: RouteLegSummary(distanceM: 1400, durationMin: 14, fromStop: viaName, toStop: "目的地"))
                ],
                routePolyline: [
                    MCPCoordinates(latitude: 31.2304, longitude: 121.4737),
                    MCPCoordinates(latitude: 31.2310, longitude: 121.4750),
                    MCPCoordinates(latitude: 31.2350, longitude: 121.4800)
                ],
                coarseWaypoints: [],
                legCount: 2
            ),
            resolvedVia: nil,
            addResult: nil,
            replanResult: nil
        )
    }

    func replan(
        sessionId: String,
        origin: String
    ) async throws -> MCPRouteSessionResult {
        guard let session = sessions[sessionId] else {
            throw RouteSessionServiceError.processFailed("Session not found: \(sessionId)")
        }

        let updatedSession = RouteSession(
            sessionId: session.sessionId,
            sessionName: session.sessionName,
            origin: origin,
            destination: session.destination,
            mode: session.mode,
            viaPoints: session.viaPoints,
            createdAt: session.createdAt
        )
        sessions[sessionId] = updatedSession

        return MCPRouteSessionResult(
            session: updatedSession,
            routePlan: MCPRoutePlan(
                routeSummary: MCPRouteSummary(distanceM: 1800, durationS: 1080, durationMin: 18),
                stopSequence: [
                    StopSequenceItem(stopType: "origin", name: "起点(新)", location: origin, routeLegSummary: nil),
                    StopSequenceItem(stopType: "destination", name: "目的地", location: session.destination, routeLegSummary: RouteLegSummary(distanceM: 1800, durationMin: 18, fromStop: "起点(新)", toStop: "目的地"))
                ],
                routePolyline: [
                    MCPCoordinates(latitude: 31.2280, longitude: 121.4710),
                    MCPCoordinates(latitude: 31.2350, longitude: 121.4800)
                ],
                coarseWaypoints: [],
                legCount: 1
            ),
            resolvedVia: nil,
            addResult: nil,
            replanResult: nil
        )
    }

    func resolveAndAddVia(
        sessionId: String,
        instruction: String,
        currentLat: Double,
        currentLon: Double,
        origin: String,
        mode: String,
        radius: Int,
        maxCandidates: Int
    ) async throws -> MCPRouteSessionResult {
        // Simulate resolving a natural-language instruction to a location
        let resolvedLocation = "\(currentLon + 0.002),\(currentLat + 0.001)"
        let resolvedName = instruction

        guard let session = sessions[sessionId] else {
            throw RouteSessionServiceError.processFailed("Session not found: \(sessionId)")
        }

        let newViaPoint = ViaPoint(
            location: resolvedLocation,
            name: resolvedName,
            stopType: "via",
            insertBeforeIndex: nil,
            coordinates: nil
        )

        var viaPoints = session.viaPoints ?? []
        viaPoints.append(newViaPoint)

        let updatedSession = RouteSession(
            sessionId: session.sessionId,
            sessionName: session.sessionName,
            origin: session.origin,
            destination: session.destination,
            mode: session.mode,
            viaPoints: viaPoints,
            createdAt: session.createdAt
        )
        sessions[sessionId] = updatedSession

        return MCPRouteSessionResult(
            session: updatedSession,
            routePlan: MCPRoutePlan(
                routeSummary: MCPRouteSummary(distanceM: 2500, durationS: 1500, durationMin: 25),
                stopSequence: [
                    StopSequenceItem(stopType: "origin", name: "起点", location: session.origin, routeLegSummary: nil),
                    StopSequenceItem(stopType: "via", name: resolvedName, location: resolvedLocation, routeLegSummary: RouteLegSummary(distanceM: 1000, durationMin: 10, fromStop: "起点", toStop: resolvedName)),
                    StopSequenceItem(stopType: "destination", name: "目的地", location: session.destination, routeLegSummary: RouteLegSummary(distanceM: 1500, durationMin: 15, fromStop: resolvedName, toStop: "目的地"))
                ],
                routePolyline: [
                    MCPCoordinates(latitude: currentLat, longitude: currentLon),
                    MCPCoordinates(latitude: currentLat + 0.001, longitude: currentLon + 0.002),
                    MCPCoordinates(latitude: 31.2350, longitude: 121.4800)
                ],
                coarseWaypoints: [],
                legCount: 2
            ),
            resolvedVia: ResolvedVia(
                name: resolvedName,
                location: resolvedLocation,
                raw: nil
            ),
            addResult: nil,
            replanResult: nil
        )
    }

    func sendWaypoints(
        sessionId: String,
        origin: String,
        destination: String,
        mode: String
    ) async throws {
        // Mock: do nothing, simulating successful waypoint transmission
        NSLog("[MockRouteSessionService] sendWaypoints: sessionId=%@, origin=%@, destination=%@", sessionId, origin, destination)
    }
}

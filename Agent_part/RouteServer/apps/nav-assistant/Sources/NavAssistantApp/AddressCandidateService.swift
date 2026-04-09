import Foundation
import CoreLocation

protocol AddressCandidateService {
    func fetchCandidates(prompt: String, history: [String]) async throws -> [AddressCandidate]
    func planFromConfirmedText(
        confirmedText: String,
        currentLocation: CLLocationCoordinate2D,
        mode: String,
        maxCandidates: Int,
        searchRadius: Int,
        selectedIndex: Int
    ) async throws -> MCPRecommendationResponse
}

enum PlanningServiceError: Error, LocalizedError {
    case missingBridgeScript
    case processFailed(String)
    case timeout
    case invalidResponse(String)
    case emptyRecommendations

    var errorDescription: String? {
        switch self {
        case .missingBridgeScript:
            return "未找到 MCP 桥接脚本，请检查路径配置。"
        case .processFailed(let detail):
            return "MCP 调用失败: \(detail)"
        case .timeout:
            return "MCP 调用超时。"
        case .invalidResponse(let detail):
            return "MCP 响应解析失败: \(detail)"
        case .emptyRecommendations:
            return "MCP 返回为空候选。"
        }
    }
}

struct MCPAddressCandidateService: AddressCandidateService {
    private let timeoutSeconds: TimeInterval
    private let bridgeScriptCandidates: [String]
    private let fallback = MockAddressCandidateService()

    init(timeoutSeconds: TimeInterval = 20.0) {
        let envScript = ProcessInfo.processInfo.environment["MCP_BRIDGE_SCRIPT_PATH"]
        self.timeoutSeconds = timeoutSeconds
        self.bridgeScriptCandidates = Self.resolveBridgeScriptCandidates(envScript: envScript)
    }

    func fetchCandidates(prompt: String, history: [String]) async throws -> [AddressCandidate] {
        try await fallback.fetchCandidates(prompt: prompt, history: history)
    }

    func planFromConfirmedText(
        confirmedText: String,
        currentLocation: CLLocationCoordinate2D,
        mode: String,
        maxCandidates: Int,
        searchRadius: Int,
        selectedIndex: Int
    ) async throws -> MCPRecommendationResponse {
        do {
            let response = try await runBridgeProcess(
                confirmedText: confirmedText,
                currentLocation: currentLocation,
                mode: mode,
                maxCandidates: maxCandidates,
                searchRadius: searchRadius,
                selectedIndex: selectedIndex
            )
            guard !response.recommendations.isEmpty else {
                throw PlanningServiceError.emptyRecommendations
            }
            return response
        } catch {
            let fallbackCandidates = try await fallback.fetchCandidates(prompt: confirmedText, history: [])
            let fallbackRecommendations = fallbackCandidates.enumerated().map { index, candidate in
                MCPRecommendationItem(
                    id: candidate.providerId ?? "fallback-\(index)",
                    name: candidate.title,
                    coordinates: MCPCoordinates(
                        latitude: candidate.coordinate?.latitude ?? currentLocation.latitude,
                        longitude: candidate.coordinate?.longitude ?? currentLocation.longitude
                    ),
                    score: candidate.score,
                    routeSummary: MCPRouteSummary(
                        distanceM: candidate.distanceM,
                        durationS: nil,
                        durationMin: candidate.durationMin
                    ),
                    location: nil,
                    tags: MCPRecommendationTags(type: nil, address: candidate.subtitle)
                )
            }
            return MCPRecommendationResponse(
                provider: "fallback",
                recommendations: fallbackRecommendations,
                chosenDestination: fallbackRecommendations.first,
                keyWaypoints: [],
                routePolyline: [],
                routeSummary: fallbackRecommendations.first?.routeSummary,
                message: error.localizedDescription
            )
        }
    }

    private func runBridgeProcess(
        confirmedText: String,
        currentLocation: CLLocationCoordinate2D,
        mode: String,
        maxCandidates: Int,
        searchRadius: Int,
        selectedIndex: Int
    ) async throws -> MCPRecommendationResponse {
        guard let scriptPath = bridgeScriptCandidates.first(where: { FileManager.default.fileExists(atPath: $0) }) else {
            throw PlanningServiceError.missingBridgeScript
        }
        let scriptURL = URL(fileURLWithPath: scriptPath)
        let openStreetRoot = scriptURL.deletingLastPathComponent().deletingLastPathComponent().path
        let env = buildBridgeEnvironment()
        let pythonCandidates = [
            ProcessInfo.processInfo.environment["PYTHON_BIN"],
            "/opt/anaconda3/envs/openstreet/bin/python3",
            "/opt/anaconda3/envs/openstreet/bin/python",
            "/Library/Developer/CommandLineTools/usr/bin/python3",
            "/usr/bin/python3",
            "python3"
        ]
            .compactMap { $0 }
            .filter { !$0.isEmpty }
        let uvCandidates = [
            ProcessInfo.processInfo.environment["UV_BIN"],
            "/opt/homebrew/bin/uv",
            "/usr/local/bin/uv",
            "uv"
        ]
            .compactMap { $0 }
            .filter { !$0.isEmpty }

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let commonArguments = [
                        "--prompt", confirmedText,
                        "--non-interactive",
                        "--json-only",
                        "--select-index", String(selectedIndex),
                        "--lat", String(currentLocation.latitude),
                        "--lon", String(currentLocation.longitude),
                        "--mode", mode,
                        "--max-candidates", String(maxCandidates),
                        "--radius", String(searchRadius)
                    ]
                    let pythonRunners = pythonCandidates.map { [$0, scriptPath] + commonArguments }
                    let uvRunners = uvCandidates.map { [$0, "--directory", openStreetRoot, "run", "python", scriptPath] + commonArguments }
                    let runners: [[String]] = pythonRunners + uvRunners

                    let promptPreview = confirmedText.replacingOccurrences(of: "\n", with: "\\n")
                    NSLog("[MCPAddressCandidateService] Running script: %@", scriptPath)
                    NSLog(
                        "[MCPAddressCandidateService] mode=%@, maxCandidates=%@, radius=%@, selectedIndex=%@",
                        String(mode),
                        String(maxCandidates),
                        String(searchRadius),
                        String(selectedIndex)
                    )
                    NSLog(
                        "[MCPAddressCandidateService] lat=%@, lon=%@",
                        String(currentLocation.latitude),
                        String(currentLocation.longitude)
                    )
                    NSLog("[MCPAddressCandidateService] prompt=%@", promptPreview)

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
                        NSLog("[MCPAddressCandidateService] trying runner: %@", runner.prefix(4).joined(separator: " "))

                        do {
                            try process.run()
                        } catch {
                            runnerErrors.append("启动失败(\(runner.first ?? "unknown")): \(error.localizedDescription)")
                            continue
                        }

                        let deadline = Date().addingTimeInterval(timeoutSeconds)
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
                            let jsonText = try extractTopLevelJSONObject(from: stdout)
                            guard let jsonData = jsonText.data(using: .utf8) else {
                                runnerErrors.append("JSON 编码失败(\(runner.first ?? "unknown"))")
                                continue
                            }
                            let decoded = try JSONDecoder().decode(MCPRecommendationResponse.self, from: jsonData)
                            continuation.resume(returning: decoded)
                            return
                        } catch {
                            runnerErrors.append("解析失败(\(runner.first ?? "unknown")): \(error.localizedDescription)")
                            continue
                        }
                    }

                    throw PlanningServiceError.processFailed(runnerErrors.joined(separator: "\n"))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func extractTopLevelJSONObject(from output: String) throws -> String {
        guard let startIndex = output.firstIndex(of: "{") else {
            throw PlanningServiceError.invalidResponse("未找到 JSON 起始")
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
            throw PlanningServiceError.invalidResponse("JSON 不完整")
        }
        return String(output[startIndex...endIndex])
    }

    private func buildBridgeEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let fallbackPath = [
            "/opt/anaconda3/envs/openstreet/bin",
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin"
        ].joined(separator: ":")

        if let existingPath = env["PATH"], !existingPath.isEmpty {
            if !existingPath.contains("/opt/anaconda3/envs/openstreet/bin") {
                env["PATH"] = "\(fallbackPath):\(existingPath)"
            }
        } else {
            env["PATH"] = fallbackPath
        }

        return env
    }

    private static func resolveBridgeScriptCandidates(envScript: String?) -> [String] {
        var rawCandidates: [String] = []

        if let envScript, !envScript.isEmpty {
            rawCandidates.append(envScript)
        }

        let relativeCandidates = [
            "../../examples/macos_prompt_recommendation_client.py",
            "examples/macos_prompt_recommendation_client.py"
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

struct MockAddressCandidateService: AddressCandidateService {
    func fetchCandidates(prompt: String, history: [String]) async throws -> [AddressCandidate] {
        let normalized = prompt.lowercased()
        if normalized.contains("上海") {
            return [
                AddressCandidate(title: "上海虹桥站", subtitle: "闵行区申贵路 1500 号", coordinate: CLLocationCoordinate2D(latitude: 31.1944, longitude: 121.3275), providerId: "mock-sh-hq", score: 0.8, distanceM: nil, durationMin: nil),
                AddressCandidate(title: "上海南站", subtitle: "徐汇区沪闵路 9001 号", coordinate: CLLocationCoordinate2D(latitude: 31.1547, longitude: 121.4296), providerId: "mock-sh-nz", score: 0.7, distanceM: nil, durationMin: nil),
                AddressCandidate(title: "上海站", subtitle: "静安区秣陵路 303 号", coordinate: CLLocationCoordinate2D(latitude: 31.2492, longitude: 121.4557), providerId: "mock-sh-z", score: 0.6, distanceM: nil, durationMin: nil)
            ]
        }

        if normalized.contains("北京") {
            return [
                AddressCandidate(title: "北京南站", subtitle: "丰台区永外大街车站路 12 号", coordinate: CLLocationCoordinate2D(latitude: 39.8652, longitude: 116.3787), providerId: "mock-bj-nz", score: 0.8, distanceM: nil, durationMin: nil),
                AddressCandidate(title: "北京西站", subtitle: "丰台区莲花池东路 118 号", coordinate: CLLocationCoordinate2D(latitude: 39.8952, longitude: 116.3218), providerId: "mock-bj-xz", score: 0.7, distanceM: nil, durationMin: nil),
                AddressCandidate(title: "北京站", subtitle: "东城区毛家湾胡同甲 13 号", coordinate: CLLocationCoordinate2D(latitude: 39.9031, longitude: 116.4273), providerId: "mock-bj-z", score: 0.6, distanceM: nil, durationMin: nil)
            ]
        }

        return [
            AddressCandidate(title: "杭州东站", subtitle: "上城区天城路 1 号", coordinate: CLLocationCoordinate2D(latitude: 30.2923, longitude: 120.2122), providerId: "mock-hz-dz", score: 0.8, distanceM: nil, durationMin: nil),
            AddressCandidate(title: "杭州站", subtitle: "上城区环城东路 1 号", coordinate: CLLocationCoordinate2D(latitude: 30.2449, longitude: 120.1827), providerId: "mock-hz-z", score: 0.7, distanceM: nil, durationMin: nil),
            AddressCandidate(title: "杭州西站", subtitle: "余杭区仓前街道", coordinate: CLLocationCoordinate2D(latitude: 30.2928, longitude: 119.9868), providerId: "mock-hz-xz", score: 0.6, distanceM: nil, durationMin: nil)
        ]
    }

    func planFromConfirmedText(
        confirmedText: String,
        currentLocation: CLLocationCoordinate2D,
        mode: String,
        maxCandidates: Int,
        searchRadius: Int,
        selectedIndex: Int
    ) async throws -> MCPRecommendationResponse {
        let candidates = try await fetchCandidates(prompt: confirmedText, history: [])
        let recommendations = candidates.prefix(maxCandidates).enumerated().map { index, candidate in
            MCPRecommendationItem(
                id: candidate.providerId ?? "mock-\(index)",
                name: candidate.title,
                coordinates: MCPCoordinates(
                    latitude: candidate.coordinate?.latitude ?? currentLocation.latitude,
                    longitude: candidate.coordinate?.longitude ?? currentLocation.longitude
                ),
                score: candidate.score,
                routeSummary: MCPRouteSummary(
                    distanceM: candidate.distanceM,
                    durationS: nil,
                    durationMin: candidate.durationMin
                ),
                location: nil,
                tags: MCPRecommendationTags(type: nil, address: candidate.subtitle)
            )
        }
        let normalizedIndex = max(1, min(selectedIndex, recommendations.count))
        return MCPRecommendationResponse(
            provider: "mock",
            recommendations: recommendations,
            chosenDestination: recommendations.isEmpty ? nil : recommendations[normalizedIndex - 1],
            keyWaypoints: [],
            routePolyline: [],
            routeSummary: recommendations.isEmpty ? nil : recommendations[normalizedIndex - 1].routeSummary,
            message: nil
        )
    }
}

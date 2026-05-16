import Foundation

struct WaypointPushService {
    private let apiClient: APIClient

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    func pushWaypoints(
        sessionID: String,
        robotHost: String,
        robotPort: Int
    ) async throws -> WaypointPushResponse {
        let rawSnapshot = try await apiClient.getJSONObject(
            "api/route-sessions/\(sessionID)",
            queryItems: [URLQueryItem(name: "force_replan", value: "false")]
        )
        guard let snapshot = rawSnapshot as? [String: Any] else {
            throw WaypointPushServiceError.invalidSnapshot
        }
        guard let routePlan = snapshot["route_plan"] as? [String: Any], !routePlan.isEmpty else {
            throw WaypointPushServiceError.missingRoutePlan
        }

        let session = snapshot["session"] as? [String: Any]
        let payload = try Self.buildWaypointPushPayload(
            routePlan: routePlan,
            session: session,
            sessionID: sessionID,
            routePlanCacheHit: snapshot["route_plan_cache_hit"]
        )

        try await Self.sendPayload(payload, robotHost: robotHost, robotPort: robotPort)

        return WaypointPushResponse(
            status: "sent",
            wsInfo: "ws://\(robotHost):\(robotPort)",
            routeId: payload["route_id"] as? String,
            totalCount: payload["total_count"] as? Int,
            routePolylineCount: payload["route_polyline_count"] as? Int
        )
    }
}

private enum WaypointPushServiceError: LocalizedError {
    case invalidSnapshot
    case missingRoutePlan
    case invalidRobotURL
    case invalidPayload

    var errorDescription: String? {
        switch self {
        case .invalidSnapshot:
            return "路线会话快照格式无效。"
        case .missingRoutePlan:
            return "当前路线会话缺少 route_plan。"
        case .invalidRobotURL:
            return "机器人 WebSocket 地址无效。"
        case .invalidPayload:
            return "waypoint JSON 组包失败。"
        }
    }
}

private extension WaypointPushService {
    static func buildWaypointPushPayload(
        routePlan: [String: Any],
        session: [String: Any]?,
        sessionID: String,
        routePlanCacheHit: Any?
    ) throws -> [String: Any] {
        let coarseWaypoints = routePlan["coarse_waypoints"] as? [Any] ?? []
        let routePolyline = routePlan["route_polyline"] as? [Any] ?? []
        var waypoints: [[String: Any]] = []
        waypoints.reserveCapacity(coarseWaypoints.count)

        for item in coarseWaypoints {
            guard let waypoint = item as? [String: Any],
                  let location = waypoint["location"] as? String,
                  let parsed = parseLocationText(location) else {
                continue
            }

            let waypointIndex = (waypoint["global_waypoint_index"] as? NSNumber)?.intValue ?? 0
            let isTerminal = (waypoint["is_leg_terminal"] as? Bool) ?? false
            let isViaStop = (waypoint["is_via_stop"] as? Bool) ?? false

            waypoints.append(
                [
                    "waypoint_index": waypointIndex,
                    "longitude": parsed.longitude,
                    "latitude": parsed.latitude,
                    "is_terminal": isTerminal,
                    "is_via_stop": isViaStop,
                ]
            )
        }

        var payload: [String: Any] = [
            "route_id": buildRouteID(),
            "total_count": waypoints.count,
            "route_polyline_count": routePolyline.count,
            "waypoints": waypoints,
            "_raw_route_plan": routePlan,
        ]

        if let session {
            payload["_session_id"] = (session["session_id"] as? String) ?? sessionID
            payload["_session_name"] = (session["session_name"] as? String) ?? ""
            payload["_origin"] = (session["origin"] as? String) ?? ""
            payload["_destination"] = (session["destination"] as? String) ?? ""
            payload["_route_plan_cache_hit"] = routePlanCacheHit as? Bool
        }

        return payload
    }

    static func sendPayload(
        _ payload: [String: Any],
        robotHost: String,
        robotPort: Int
    ) async throws {
        guard let url = URL(string: "ws://\(robotHost):\(robotPort)") else {
            throw WaypointPushServiceError.invalidRobotURL
        }
        guard JSONSerialization.isValidJSONObject(payload) else {
            throw WaypointPushServiceError.invalidPayload
        }

        let text = try makeASCIIJSONString(from: payload)

        let task = URLSession.shared.webSocketTask(with: url)
        task.resume()
        defer {
            task.cancel(with: .normalClosure, reason: nil)
        }

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            task.send(.string(text)) { error in
                if let error {
                    continuation.resume(throwing: APIClientError.transport(error.localizedDescription))
                } else {
                    continuation.resume()
                }
            }
        }
    }

    static func parseLocationText(_ text: String) -> (longitude: Double, latitude: Double)? {
        let parts = text.split(separator: ",", maxSplits: 1).map { String($0) }
        guard parts.count == 2,
              let longitude = Double(parts[0]),
              let latitude = Double(parts[1]) else {
            return nil
        }
        return (longitude, latitude)
    }

    static func buildRouteID(now: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone.current
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let secondsText = formatter.string(from: now)
        let milliseconds = Calendar.current.component(.nanosecond, from: now) / 1_000_000
        return "route_\(secondsText)_\(String(format: "%03d", milliseconds))"
    }

    static func makeASCIIJSONString(from object: [String: Any]) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        guard let text = String(data: data, encoding: .utf8) else {
            throw WaypointPushServiceError.invalidPayload
        }

        var escaped = ""
        escaped.reserveCapacity(text.count)

        for scalar in text.unicodeScalars {
            if scalar.value <= 0x7F {
                escaped.unicodeScalars.append(scalar)
                continue
            }

            if scalar.value <= 0xFFFF {
                escaped += String(format: "\\u%04X", scalar.value)
                continue
            }

            let value = scalar.value - 0x10000
            let high = 0xD800 + ((value >> 10) & 0x3FF)
            let low = 0xDC00 + (value & 0x3FF)
            escaped += String(format: "\\u%04X\\u%04X", high, low)
        }

        return escaped
    }
}

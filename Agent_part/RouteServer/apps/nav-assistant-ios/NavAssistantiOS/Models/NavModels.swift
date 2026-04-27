import Foundation
import CoreLocation

enum RecognitionOutputMode: String, CaseIterable, Identifiable {
    case streaming = "流式"
    case sentence = "句子级"

    var id: String { rawValue }
}

enum NavigationFlowStage: String {
    case idle
    case locating
    case recommending
    case candidateSelection
    case creatingSession
    case sessionActive
    case addingViaPoint
    case replanning
    case pushingWaypoints
    case error

    var title: String {
        switch self {
        case .idle:
            return "待命"
        case .locating:
            return "定位中"
        case .recommending:
            return "推荐中"
        case .candidateSelection:
            return "待选点"
        case .creatingSession:
            return "建路线"
        case .sessionActive:
            return "导航中"
        case .addingViaPoint:
            return "加途经点"
        case .replanning:
            return "重规划"
        case .pushingWaypoints:
            return "下发路径"
        case .error:
            return "异常"
        }
    }

    var systemImage: String {
        switch self {
        case .idle:
            return "pause.circle"
        case .locating:
            return "location.circle"
        case .recommending:
            return "sparkles"
        case .candidateSelection:
            return "list.bullet"
        case .creatingSession:
            return "point.topleft.down.curvedto.point.bottomright.up"
        case .sessionActive:
            return "point.3.connected.trianglepath.dotted"
        case .addingViaPoint:
            return "plus.circle"
        case .replanning:
            return "arrow.trianglehead.clockwise"
        case .pushingWaypoints:
            return "paperplane"
        case .error:
            return "exclamationmark.triangle"
        }
    }

    var summary: String {
        switch self {
        case .idle:
            return "等待输入需求。"
        case .locating:
            return "正在获取当前位置和权限。"
        case .recommending:
            return "正在请求目的地推荐。"
        case .candidateSelection:
            return "请选择目的地候选。"
        case .creatingSession:
            return "正在创建路线会话。"
        case .sessionActive:
            return "路线会话已建立，可刷新、加途经点或下发 waypoint。"
        case .addingViaPoint:
            return "正在搜索并插入新的途经点。"
        case .replanning:
            return "正在根据当前位置重规划。"
        case .pushingWaypoints:
            return "正在向机器人下发 waypoint。"
        case .error:
            return "操作失败，请查看错误信息。"
        }
    }

    var isBusy: Bool {
        switch self {
        case .locating, .recommending, .creatingSession, .addingViaPoint, .replanning, .pushingWaypoints:
            return true
        case .idle, .candidateSelection, .sessionActive, .error:
            return false
        }
    }
}

struct AddressCandidate: Identifiable, Equatable {
    let title: String
    let subtitle: String
    let coordinate: CLLocationCoordinate2D?
    let providerId: String?
    let locationText: String?
    let score: Double?
    let distanceM: Double?
    let durationMin: Double?

    var id: String {
        if let providerId, !providerId.isEmpty {
            return providerId
        }
        if let locationText, !locationText.isEmpty {
            return "\(title)|\(locationText)"
        }
        if let coordinate {
            return "\(title)|\(coordinate.latitude)|\(coordinate.longitude)"
        }
        return "\(title)|\(subtitle)"
    }

    static func == (lhs: AddressCandidate, rhs: AddressCandidate) -> Bool {
        lhs.id == rhs.id
    }

    var distanceText: String? {
        guard let distanceM, distanceM > 0 else { return nil }
        if distanceM >= 1000 {
            return String(format: "%.1f km", distanceM / 1000.0)
        }
        return String(format: "%.0f m", distanceM)
    }

    var durationText: String? {
        guard let durationMin, durationMin > 0 else { return nil }
        return String(format: "%.0f 分钟", durationMin)
    }
}

struct APIHealthResponse: Decodable {
    let status: String
    let provider: String
    let timestamp: String
}

struct MCPRecommendationResponse: Codable {
    let provider: String?
    let recommendations: [MCPRecommendationItem]
    let chosenDestination: MCPRecommendationItem?
    let keyWaypoints: [MCPWaypoint]
    let routePolyline: [MCPCoordinates]
    let routeSummary: MCPRouteSummary?
    let message: String?

    enum CodingKeys: String, CodingKey {
        case provider
        case recommendations
        case chosenDestination = "chosen_destination"
        case keyWaypoints = "key_waypoints"
        case routePolyline = "route_polyline"
        case routeSummary = "route_summary"
        case message
    }
}

struct MCPRecommendationItem: Codable, Identifiable {
    let id: String
    let name: String
    let coordinates: MCPCoordinates
    let score: Double?
    let routeSummary: MCPRouteSummary?
    let location: String?
    let tags: MCPRecommendationTags?

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case coordinates
        case score
        case routeSummary = "route_summary"
        case location
        case tags
    }

    func toAddressCandidate() -> AddressCandidate {
        AddressCandidate(
            title: name,
            subtitle: tags?.address ?? tags?.type ?? "",
            coordinate: CLLocationCoordinate2D(latitude: coordinates.latitude, longitude: coordinates.longitude),
            providerId: id,
            locationText: location,
            score: score,
            distanceM: routeSummary?.distanceM,
            durationMin: routeSummary?.durationMin
        )
    }
}

struct MCPCoordinates: Codable {
    let latitude: Double
    let longitude: Double
}

struct MCPRecommendationTags: Codable {
    let type: String?
    let address: String?
}

struct MCPWaypoint: Codable {
    let longitude: Double
    let latitude: Double
    let instruction: String?
    let road: String?
    let action: String?

    enum CodingKeys: String, CodingKey {
        case longitude
        case latitude
        case instruction
        case road
        case action
        case location
        case coordinate
    }

    enum CoordinateCodingKeys: String, CodingKey {
        case longitude
        case latitude
    }

    init(longitude: Double, latitude: Double, instruction: String? = nil, road: String? = nil, action: String? = nil) {
        self.longitude = longitude
        self.latitude = latitude
        self.instruction = instruction
        self.road = road
        self.action = action
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        if let lon = try container.decodeIfPresent(Double.self, forKey: .longitude),
           let lat = try container.decodeIfPresent(Double.self, forKey: .latitude) {
            longitude = lon
            latitude = lat
        } else if let coordinateContainer = try? container.nestedContainer(keyedBy: CoordinateCodingKeys.self, forKey: .coordinate),
                  let lon = try coordinateContainer.decodeIfPresent(Double.self, forKey: .longitude),
                  let lat = try coordinateContainer.decodeIfPresent(Double.self, forKey: .latitude) {
            longitude = lon
            latitude = lat
        } else if let location = try container.decodeIfPresent(String.self, forKey: .location) {
            let parts = location.split(separator: ",")
            guard parts.count == 2,
                  let lon = Double(parts[0]),
                  let lat = Double(parts[1]) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .location,
                    in: container,
                    debugDescription: "Invalid location format, expected 'lon,lat'."
                )
            }
            longitude = lon
            latitude = lat
        } else {
            throw DecodingError.keyNotFound(
                CodingKeys.longitude,
                DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Missing waypoint coordinates.")
            )
        }

        instruction = try container.decodeIfPresent(String.self, forKey: .instruction)
        road = try container.decodeIfPresent(String.self, forKey: .road)
        action = try container.decodeIfPresent(String.self, forKey: .action)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(longitude, forKey: .longitude)
        try container.encode(latitude, forKey: .latitude)
        try container.encodeIfPresent(instruction, forKey: .instruction)
        try container.encodeIfPresent(road, forKey: .road)
        try container.encodeIfPresent(action, forKey: .action)
    }
}

struct MCPRouteSummary: Codable {
    let distanceM: Double?
    let durationS: Double?
    let durationMin: Double?

    enum CodingKeys: String, CodingKey {
        case distanceM = "distance_m"
        case durationS = "duration_s"
        case durationMin = "duration_min"
    }

    var distanceText: String? {
        guard let distanceM, distanceM > 0 else { return nil }
        if distanceM >= 1000 {
            return String(format: "%.2f km", distanceM / 1000.0)
        }
        return String(format: "%.0f m", distanceM)
    }

    var durationText: String? {
        let minutes = durationMin ?? durationS.map { $0 / 60.0 }
        guard let minutes, minutes > 0 else { return nil }
        return String(format: "%.0f 分钟", minutes)
    }
}

struct RouteSession: Codable {
    let sessionId: String
    let sessionName: String?
    let provider: String?
    let origin: String?
    let destination: String?
    let mode: String?
    let viaPoints: [ViaPoint]?
    let createdAt: String?
    let updatedAt: String?
    let routePlanCachedAt: String?

    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case sessionName = "session_name"
        case provider
        case origin
        case destination
        case mode
        case viaPoints = "via_points"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case routePlanCachedAt = "route_plan_cached_at"
    }
}

struct ViaPoint: Codable, Identifiable {
    var id: String { name }
    let location: String?
    let name: String
    let stopType: String?
    let insertBeforeIndex: Int?
    let coordinates: MCPCoordinates?

    enum CodingKeys: String, CodingKey {
        case location
        case name
        case stopType = "stop_type"
        case insertBeforeIndex = "insert_before_index"
        case coordinates
    }

    var coordinate: CLLocationCoordinate2D? {
        if let coordinates {
            return CLLocationCoordinate2D(latitude: coordinates.latitude, longitude: coordinates.longitude)
        }
        guard let location else { return nil }
        let parts = location.split(separator: ",")
        guard parts.count == 2,
              let lon = Double(parts[0]),
              let lat = Double(parts[1]) else {
            return nil
        }
        return CLLocationCoordinate2D(latitude: lat, longitude: lon)
    }
}

struct RouteLegSummary: Codable {
    let distanceM: Double?
    let durationMin: Double?

    enum CodingKeys: String, CodingKey {
        case distanceM = "distance_m"
        case durationMin = "duration_min"
    }
}

struct StopSequenceItem: Codable, Identifiable {
    let id = UUID()
    let stopType: String?
    let name: String?
    let location: String?
    let routeLegSummary: RouteLegSummary?

    enum CodingKeys: String, CodingKey {
        case stopType = "stop_type"
        case fallbackType = "type"
        case name
        case location
        case routeLegSummary = "route_leg_summary"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        stopType = try container.decodeIfPresent(String.self, forKey: .stopType)
            ?? container.decodeIfPresent(String.self, forKey: .fallbackType)
        name = try container.decodeIfPresent(String.self, forKey: .name)
        location = try container.decodeIfPresent(String.self, forKey: .location)
        routeLegSummary = try container.decodeIfPresent(RouteLegSummary.self, forKey: .routeLegSummary)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(stopType, forKey: .stopType)
        try container.encodeIfPresent(name, forKey: .name)
        try container.encodeIfPresent(location, forKey: .location)
        try container.encodeIfPresent(routeLegSummary, forKey: .routeLegSummary)
    }

    var coordinate: CLLocationCoordinate2D? {
        guard let location else { return nil }
        let parts = location.split(separator: ",")
        guard parts.count == 2,
              let lon = Double(parts[0]),
              let lat = Double(parts[1]) else {
            return nil
        }
        return CLLocationCoordinate2D(latitude: lat, longitude: lon)
    }
}

struct MCPRoutePlan: Codable {
    let provider: String?
    let origin: String?
    let destination: String?
    let mode: String?
    let viaPoints: [ViaPoint]?
    let stopSequence: [StopSequenceItem]?
    let routeSummary: MCPRouteSummary?
    let routePolyline: [MCPCoordinates]?
    let coarseWaypoints: [MCPWaypoint]?
    let legCount: Int?

    enum CodingKeys: String, CodingKey {
        case provider
        case origin
        case destination
        case mode
        case viaPoints = "via_points"
        case stopSequence = "stop_sequence"
        case routeSummary = "route_summary"
        case routePolyline = "route_polyline"
        case coarseWaypoints = "coarse_waypoints"
        case legCount = "leg_count"
    }
}

struct HTTPRoutePlanResponse: Codable {
    let provider: String
    let routePlan: MCPRoutePlan

    enum CodingKeys: String, CodingKey {
        case provider
        case routePlan = "route_plan"
    }
}

struct MCPRouteSessionResult: Codable {
    let session: RouteSession?
    let routePlan: MCPRoutePlan?
    let routePlanCacheHit: Bool?
    let sessionChanged: Bool?
    let changeReason: String?
    let routePlanUpdateMode: String?

    enum CodingKeys: String, CodingKey {
        case session
        case routePlan = "route_plan"
        case routePlanCacheHit = "route_plan_cache_hit"
        case sessionChanged = "session_changed"
        case changeReason = "change_reason"
        case routePlanUpdateMode = "route_plan_update_mode"
    }
}

struct WaypointPushResponse: Codable {
    let status: String
    let wsInfo: String?
    let routeId: String?
    let totalCount: Int?

    enum CodingKeys: String, CodingKey {
        case status
        case wsInfo = "ws_info"
        case routeId = "route_id"
        case totalCount = "total_count"
    }
}

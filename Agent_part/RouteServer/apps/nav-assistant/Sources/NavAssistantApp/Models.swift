import Foundation
import CoreLocation

enum RecognitionOutputMode: String, CaseIterable, Identifiable {
    case streaming = "流式"
    case sentence = "句子级"

    var id: String { rawValue }

    var summary: String {
        switch self {
        case .streaming:
            return "边说边显示，适合快速试探和修正。"
        case .sentence:
            return "整句收敛后再提交，适合盲人按钮式录音。"
        }
    }
}

enum AppInteractionMode: String, CaseIterable, Identifiable {
    case standard = "标准模式"
    case voiceOnly = "盲人语音模式"

    var id: String { rawValue }

    var systemImage: String {
        switch self {
        case .standard:
            return "sidebar.left"
        case .voiceOnly:
            return "waveform.circle"
        }
    }

    var summary: String {
        switch self {
        case .standard:
            return "保留文本编辑、候选筛选和地图联动，适合桌面场景下精细确认。"
        case .voiceOnly:
            return "采用按钮触发录音与语音播报，减少视觉依赖，突出可达性和顺序引导。"
        }
    }
}

struct ConfirmedTextRecord: Identifiable {
    let id = UUID()
    let text: String
    let createdAt: Date
}

struct AddressCandidate: Identifiable {
    let id = UUID()
    let title: String
    let subtitle: String
    let coordinate: CLLocationCoordinate2D?
    let providerId: String?
    let score: Double?
    let distanceM: Double?
    let durationMin: Double?

    var displaySubtitle: String {
        let trimmed = subtitle.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "暂无详细地址信息" : trimmed
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

    var scoreText: String? {
        guard let score else { return nil }
        return String(format: "评分 %.2f", score)
    }

    var metricPairs: [(label: String, value: String)] {
        [
            ("距离", distanceText),
            ("预计耗时", durationText),
            ("匹配度", scoreText)
        ].compactMap { label, value in
            guard let value else { return nil }
            return (label, value)
        }
    }
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
            coordinate: CLLocationCoordinate2D(
                latitude: coordinates.latitude,
                longitude: coordinates.longitude
            ),
            providerId: id,
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
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Missing waypoint coordinates."
                )
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

// MARK: - Route Session Models

struct RouteSession: Codable {
    let sessionId: String
    let sessionName: String?
    let origin: String?
    let destination: String?
    let mode: String?
    let viaPoints: [ViaPoint]?
    let createdAt: String?

    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case sessionName = "session_name"
        case origin
        case destination
        case mode
        case viaPoints = "via_points"
        case createdAt = "created_at"
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
        guard let coords = coordinates else { return nil }
        return CLLocationCoordinate2D(latitude: coords.latitude, longitude: coords.longitude)
    }

    var locationCoordinate: CLLocationCoordinate2D? {
        guard let loc = location, loc.contains(",") else { return nil }
        let parts = loc.split(separator: ",")
        guard parts.count == 2,
              let lon = Double(parts[0]),
              let lat = Double(parts[1]) else { return nil }
        return CLLocationCoordinate2D(latitude: lat, longitude: lon)
    }
}

struct RouteLegSummary: Codable {
    let distanceM: Double?
    let durationMin: Double?
    let fromStop: String?
    let toStop: String?

    enum CodingKeys: String, CodingKey {
        case distanceM = "distance_m"
        case durationMin = "duration_min"
        case fromStop = "from_stop"
        case toStop = "to_stop"
    }

    var distanceText: String? {
        guard let distanceM, distanceM > 0 else { return nil }
        if distanceM >= 1000 {
            return String(format: "%.2f km", distanceM / 1000.0)
        }
        return String(format: "%.0f m", distanceM)
    }

    var durationText: String? {
        guard let minutes = durationMin, minutes > 0 else { return nil }
        return String(format: "%.0f 分钟", minutes)
    }
}

struct StopSequenceItem: Codable {
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

    init(
        stopType: String?,
        name: String?,
        location: String?,
        routeLegSummary: RouteLegSummary?
    ) {
        self.stopType = stopType
        self.name = name
        self.location = location
        self.routeLegSummary = routeLegSummary
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
        guard let loc = location, loc.contains(",") else { return nil }
        let parts = loc.split(separator: ",")
        guard parts.count == 2,
              let lon = Double(parts[0]),
              let lat = Double(parts[1]) else { return nil }
        return CLLocationCoordinate2D(latitude: lat, longitude: lon)
    }
}

struct MCPRoutePlan: Codable {
    let routeSummary: MCPRouteSummary?
    let stopSequence: [StopSequenceItem]?
    let routePolyline: [MCPCoordinates]?
    let coarseWaypoints: [MCPWaypoint]?
    let legCount: Int?

    enum CodingKeys: String, CodingKey {
        case routeSummary = "route_summary"
        case stopSequence = "stop_sequence"
        case routePolyline = "route_polyline"
        case coarseWaypoints = "coarse_waypoints"
        case legCount = "leg_count"
    }
}

/// Raw JSON sub-result stored as a dictionary (used to break struct recursion).
/// Decode manually after confirming the full object structure.
struct RawJSONResult: Codable {
    let data: [String: AnyCodableValue]

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.data = (try? container.decode([String: AnyCodableValue].self)) ?? [:]
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(data)
    }
}

/// Type-erased Codable wrapper so any JSON value can be stored.
enum AnyCodableValue: Codable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case array([AnyCodableValue])
    case dictionary([String: AnyCodableValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let v = try? container.decode(String.self) { self = .string(v); return }
        if let v = try? container.decode(Int.self) { self = .int(v); return }
        if let v = try? container.decode(Double.self) { self = .double(v); return }
        if let v = try? container.decode(Bool.self) { self = .bool(v); return }
        if let v = try? container.decode([AnyCodableValue].self) { self = .array(v); return }
        if let v = try? container.decode([String: AnyCodableValue].self) { self = .dictionary(v); return }
        self = .null
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .int(let v): try container.encode(v)
        case .double(let v): try container.encode(v)
        case .bool(let v): try container.encode(v)
        case .array(let v): try container.encode(v)
        case .dictionary(let v): try container.encode(v)
        case .null: try container.encodeNil()
        }
    }
}

struct MCPRouteSessionResult: Codable {
    let session: RouteSession?
    let routePlan: MCPRoutePlan?
    let resolvedVia: ResolvedVia?
    let addResult: RawJSONResult?
    let replanResult: RawJSONResult?

    enum CodingKeys: String, CodingKey {
        case session
        case routePlan = "route_plan"
        case resolvedVia = "resolved_via"
        case addResult = "add_result"
        case replanResult = "replan_result"
    }
}

struct ResolvedVia: Codable {
    let name: String?
    let location: String?
    let raw: MCPRecommendationItem?
}

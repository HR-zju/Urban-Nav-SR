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
    let routeSummary: MCPRouteSummary?
    let message: String?

    enum CodingKeys: String, CodingKey {
        case provider
        case recommendations
        case chosenDestination = "chosen_destination"
        case keyWaypoints = "key_waypoints"
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

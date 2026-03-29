import Foundation
import CoreLocation

enum RecognitionOutputMode: String, CaseIterable, Identifiable {
    case streaming = "流式"
    case sentence = "句子级"

    var id: String { rawValue }
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
}

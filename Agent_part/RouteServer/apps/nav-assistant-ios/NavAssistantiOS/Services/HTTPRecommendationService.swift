import Foundation
import CoreLocation

protocol RecommendationService {
    func recommend(
        prompt: String,
        currentLocation: CLLocationCoordinate2D,
        mode: String,
        maxCandidates: Int,
        searchRadius: Int,
        resolutionScope: String?,
        anchorLocation: String?
    ) async throws -> MCPRecommendationResponse
}

struct HTTPRecommendationService: RecommendationService {
    private let apiClient: APIClient

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    func recommend(
        prompt: String,
        currentLocation: CLLocationCoordinate2D,
        mode: String,
        maxCandidates: Int,
        searchRadius: Int,
        resolutionScope: String? = nil,
        anchorLocation: String? = nil
    ) async throws -> MCPRecommendationResponse {
        try await apiClient.post(
            "api/recommendations",
            body: RecommendationRequest(
                userInstruction: prompt,
                currentLatitude: currentLocation.latitude,
                currentLongitude: currentLocation.longitude,
                maxCandidates: maxCandidates,
                searchRadius: searchRadius,
                mode: mode,
                resolutionScope: resolutionScope,
                anchorLocation: anchorLocation
            )
        )
    }
}

private struct RecommendationRequest: Encodable {
    let userInstruction: String
    let currentLatitude: Double
    let currentLongitude: Double
    let maxCandidates: Int
    let searchRadius: Int
    let mode: String
    let resolutionScope: String?
    let anchorLocation: String?

    enum CodingKeys: String, CodingKey {
        case userInstruction = "user_instruction"
        case currentLatitude = "current_latitude"
        case currentLongitude = "current_longitude"
        case maxCandidates = "max_candidates"
        case searchRadius = "search_radius"
        case mode
        case resolutionScope = "resolution_scope"
        case anchorLocation = "anchor_location"
    }
}

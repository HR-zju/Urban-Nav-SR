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
        try await apiClient.post(
            "api/waypoints/push",
            body: PushWaypointsRequest(
                sessionID: sessionID,
                robotHost: robotHost,
                robotPort: robotPort
            )
        )
    }
}

private struct PushWaypointsRequest: Encodable {
    let sessionID: String
    let robotHost: String
    let robotPort: Int

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case robotHost = "robot_host"
        case robotPort = "robot_port"
    }
}

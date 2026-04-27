import Foundation

protocol RouteSessionServiceProtocol {
    func planRoute(
        origin: String,
        destination: String,
        viaPoints: [ViaPoint]?,
        mode: String
    ) async throws -> HTTPRoutePlanResponse

    func createSession(
        origin: String,
        destination: String,
        viaPoints: [ViaPoint]?,
        mode: String,
        sessionName: String?
    ) async throws -> MCPRouteSessionResult

    func getSession(sessionID: String, forceReplan: Bool) async throws -> MCPRouteSessionResult

    func addViaPoint(sessionID: String, viaPoint: ViaPoint, insertBeforeIndex: Int?) async throws -> MCPRouteSessionResult

    func replan(sessionID: String, origin: String) async throws -> MCPRouteSessionResult
}

struct HTTPRouteSessionService: RouteSessionServiceProtocol {
    private let apiClient: APIClient

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    func planRoute(
        origin: String,
        destination: String,
        viaPoints: [ViaPoint]?,
        mode: String
    ) async throws -> HTTPRoutePlanResponse {
        try await apiClient.post(
            "api/routes/plan",
            body: RoutePlanRequest(origin: origin, destination: destination, viaPoints: viaPoints, mode: mode)
        )
    }

    func createSession(
        origin: String,
        destination: String,
        viaPoints: [ViaPoint]?,
        mode: String,
        sessionName: String?
    ) async throws -> MCPRouteSessionResult {
        try await apiClient.post(
            "api/route-sessions",
            body: CreateRouteSessionRequest(
                origin: origin,
                destination: destination,
                viaPoints: viaPoints,
                mode: mode,
                sessionName: sessionName
            )
        )
    }

    func getSession(sessionID: String, forceReplan: Bool = false) async throws -> MCPRouteSessionResult {
        try await apiClient.get(
            "api/route-sessions/\(sessionID)",
            queryItems: [URLQueryItem(name: "force_replan", value: forceReplan ? "true" : "false")]
        )
    }

    func addViaPoint(sessionID: String, viaPoint: ViaPoint, insertBeforeIndex: Int?) async throws -> MCPRouteSessionResult {
        try await apiClient.post(
            "api/route-sessions/\(sessionID)/via-points",
            body: AddViaPointRequest(newViaPoint: viaPoint, insertBeforeIndex: insertBeforeIndex)
        )
    }

    func replan(sessionID: String, origin: String) async throws -> MCPRouteSessionResult {
        try await apiClient.post(
            "api/route-sessions/\(sessionID)/replan",
            body: ReplanRequest(origin: origin)
        )
    }
}

private struct RoutePlanRequest: Encodable {
    let origin: String
    let destination: String
    let viaPoints: [ViaPoint]?
    let mode: String

    enum CodingKeys: String, CodingKey {
        case origin
        case destination
        case viaPoints = "via_points"
        case mode
    }
}

private struct CreateRouteSessionRequest: Encodable {
    let origin: String
    let destination: String
    let viaPoints: [ViaPoint]?
    let mode: String
    let sessionName: String?

    enum CodingKeys: String, CodingKey {
        case origin
        case destination
        case viaPoints = "via_points"
        case mode
        case sessionName = "session_name"
    }
}

private struct AddViaPointRequest: Encodable {
    let newViaPoint: ViaPoint
    let insertBeforeIndex: Int?

    enum CodingKeys: String, CodingKey {
        case newViaPoint = "new_via_point"
        case insertBeforeIndex = "insert_before_index"
    }
}

private struct ReplanRequest: Encodable {
    let origin: String
}

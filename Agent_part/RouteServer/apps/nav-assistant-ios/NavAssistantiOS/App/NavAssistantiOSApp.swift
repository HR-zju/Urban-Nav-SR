import SwiftUI

@main
struct NavAssistantiOSApp: App {
    @StateObject private var flowViewModel = NavigationFlowViewModel(
        recommendationService: HTTPRecommendationService(apiClient: APIClient(baseURL: AppConfiguration.shared.apiBaseURL)),
        routeService: HTTPRouteSessionService(apiClient: APIClient(baseURL: AppConfiguration.shared.apiBaseURL)),
        waypointPushService: WaypointPushService(apiClient: APIClient(baseURL: AppConfiguration.shared.apiBaseURL)),
        webSocketRouteSessionClient: WebSocketRouteSessionClient(baseURL: AppConfiguration.shared.apiBaseURL)
    )

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: flowViewModel)
        }
    }
}

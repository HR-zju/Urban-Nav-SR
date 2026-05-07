import SwiftUI

@main
struct NavAssistantiOSApp: App {
    @StateObject private var appConfiguration: AppConfiguration
    @StateObject private var flowViewModel: NavigationFlowViewModel

    init() {
        let appConfiguration = AppConfiguration.shared
        let apiClient = APIClient(baseURLProvider: { appConfiguration.apiBaseURL })
        _appConfiguration = StateObject(wrappedValue: appConfiguration)
        _flowViewModel = StateObject(
            wrappedValue: NavigationFlowViewModel(
                recommendationService: HTTPRecommendationService(apiClient: apiClient),
                routeService: HTTPRouteSessionService(apiClient: apiClient),
                waypointPushService: WaypointPushService(apiClient: apiClient),
                webSocketRouteSessionClient: WebSocketRouteSessionClient(
                    baseURLProvider: { appConfiguration.apiBaseURL }
                )
            )
        )
    }

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: flowViewModel, appConfiguration: appConfiguration)
        }
    }
}

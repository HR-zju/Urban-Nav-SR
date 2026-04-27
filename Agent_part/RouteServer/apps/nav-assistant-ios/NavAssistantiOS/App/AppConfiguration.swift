import Foundation

struct AppConfiguration {
    let apiBaseURL: URL

    static let shared = AppConfiguration.load()

    private static func load() -> AppConfiguration {
        let value = (Bundle.main.object(forInfoDictionaryKey: "APIBaseURL") as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
        let baseURL = URL(string: value ?? "") ?? URL(string: "http://127.0.0.1:8080")!
        return AppConfiguration(apiBaseURL: baseURL)
    }
}

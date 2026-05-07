import Foundation

@MainActor
final class AppConfiguration: ObservableObject {
    static let shared = AppConfiguration()

    @Published private(set) var apiBaseURLText: String

    let defaultAPIBaseURLText: String

    var apiBaseURL: URL {
        Self.normalizedURL(from: apiBaseURLText) ?? Self.normalizedURL(from: defaultAPIBaseURLText) ?? URL(string: "http://127.0.0.1:8080")!
    }

    private static let overrideKey = "nav_assistant.api_base_url_override"
    private let userDefaults: UserDefaults

    private init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
        let infoValue = (Bundle.main.object(forInfoDictionaryKey: "APIBaseURL") as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let fallbackValue = Self.normalizedURLString(from: infoValue) ?? "http://127.0.0.1:8080"
        self.defaultAPIBaseURLText = fallbackValue

        let overrideValue = userDefaults.string(forKey: Self.overrideKey)
        self.apiBaseURLText = Self.normalizedURLString(from: overrideValue) ?? fallbackValue
    }

    @discardableResult
    func updateAPIBaseURL(_ value: String) -> Bool {
        guard let normalized = Self.normalizedURLString(from: value) else {
            return false
        }
        apiBaseURLText = normalized
        userDefaults.set(normalized, forKey: Self.overrideKey)
        return true
    }

    func resetAPIBaseURL() {
        apiBaseURLText = defaultAPIBaseURLText
        userDefaults.removeObject(forKey: Self.overrideKey)
    }

    private static func normalizedURLString(from rawValue: String?) -> String? {
        guard let normalizedURL = normalizedURL(from: rawValue) else {
            return nil
        }
        return normalizedURL.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    private static func normalizedURL(from rawValue: String?) -> URL? {
        let trimmed = rawValue?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmed.isEmpty else { return nil }

        let candidate = trimmed.contains("://") ? trimmed : "http://\(trimmed)"
        guard let url = URL(string: candidate),
              let scheme = url.scheme?.lowercased(),
              ["http", "https"].contains(scheme),
              let host = url.host,
              !host.isEmpty else {
            return nil
        }

        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        components?.path = ""
        components?.query = nil
        components?.fragment = nil
        return components?.url
    }
}

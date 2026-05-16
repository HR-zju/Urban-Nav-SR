import Foundation

@MainActor
final class WebSocketRouteSessionClient: ObservableObject {
    @Published var lastSnapshot: MCPRouteSessionResult?
    @Published var lastHeartbeatAt: Date?
    @Published var lastError: String?
    @Published var isConnected = false

    private var task: URLSessionWebSocketTask?
    private let session = URLSession(configuration: .default)
    private let baseURLProvider: () -> URL
    var onSnapshot: ((MCPRouteSessionResult) -> Void)?
    var onHeartbeat: ((Date) -> Void)?
    var onConnectionStateChange: ((Bool) -> Void)?
    var onError: ((String) -> Void)?

    init(baseURL: URL) {
        self.baseURLProvider = { baseURL }
    }

    init(baseURLProvider: @escaping () -> URL) {
        self.baseURLProvider = baseURLProvider
    }

    func connect(sessionID: String) {
        disconnect()
        lastSnapshot = nil
        lastHeartbeatAt = nil
        lastError = nil
        let baseURL = baseURLProvider()
        guard var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false) else {
            lastError = "无法构造 WebSocket 地址。"
            onError?(lastError ?? "无法构造 WebSocket 地址。")
            return
        }
        components.scheme = baseURL.scheme == "https" ? "wss" : "ws"
        components.path = "/ws/route-sessions/\(sessionID)"
        guard let url = components.url else {
            lastError = "WebSocket 地址无效。"
            onError?(lastError ?? "WebSocket 地址无效。")
            return
        }

        let task = session.webSocketTask(with: url)
        self.task = task
        task.resume()
        isConnected = true
        onConnectionStateChange?(true)
        listen()
    }

    func requestRefresh(forceReplan: Bool = false) {
        let payload = ["action": "refresh", "force_replan": forceReplan] as [String : Any]
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let text = String(data: data, encoding: .utf8) else {
            return
        }
        task?.send(.string(text)) { [weak self] error in
            if let error {
                Task { @MainActor in
                    self?.lastError = error.localizedDescription
                }
            }
        }
    }

    func disconnect() {
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        isConnected = false
        onConnectionStateChange?(false)
    }

    private func listen() {
        task?.receive { [weak self] result in
            guard let self else { return }
            Task { @MainActor in
                switch result {
                case .failure(let error):
                    self.lastError = error.localizedDescription
                    self.isConnected = false
                    self.onError?(error.localizedDescription)
                    self.onConnectionStateChange?(false)
                case .success(let message):
                    self.handle(message)
                    self.listen()
                }
            }
        }
    }

    private func handle(_ message: URLSessionWebSocketTask.Message) {
        let data: Data
        switch message {
        case .data(let rawData):
            data = rawData
        case .string(let text):
            data = Data(text.utf8)
        @unknown default:
            return
        }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }

        let type = json["type"] as? String
        if type == "heartbeat" {
            lastHeartbeatAt = Date()
            onHeartbeat?(lastHeartbeatAt ?? Date())
            return
        }

        if type == "snapshot",
           let payload = json["data"],
           let snapshotData = try? JSONSerialization.data(withJSONObject: payload),
           let snapshot = try? JSONDecoder().decode(MCPRouteSessionResult.self, from: snapshotData) {
            lastSnapshot = snapshot
            onSnapshot?(snapshot)
            lastError = nil
            return
        }

        if type == "deleted" {
            lastError = "当前路线会话已不存在。"
            onError?(lastError ?? "当前路线会话已不存在。")
            disconnect()
            return
        }

        if let error = json["error"] as? [String: Any] {
            lastError = error["message"] as? String
            onError?(lastError ?? "WebSocket 返回错误。")
        }
    }
}

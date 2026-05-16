import Foundation

struct EmptyRequest: Encodable {}

struct APIErrorPayload: Decodable {
    struct APIError: Decodable {
        let code: String
        let message: String
        let detail: String?
    }

    let error: APIError
}

enum APIClientError: Error, LocalizedError {
    case invalidResponse
    case server(String)
    case transport(String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "服务返回了无效响应。"
        case .server(let message):
            return message
        case .transport(let message):
            return message
        }
    }
}

struct APIClient {
    private let baseURLProvider: () -> URL
    let session: URLSession

    init(baseURL: URL, session: URLSession = .shared) {
        self.baseURLProvider = { baseURL }
        self.session = session
    }

    init(baseURLProvider: @escaping () -> URL, session: URLSession = .shared) {
        self.baseURLProvider = baseURLProvider
        self.session = session
    }

    func get<Response: Decodable>(
        _ path: String,
        queryItems: [URLQueryItem] = []
    ) async throws -> Response {
        try await request(path: path, method: "GET", queryItems: queryItems, body: nil as EmptyRequest?)
    }

    func post<Body: Encodable, Response: Decodable>(
        _ path: String,
        body: Body
    ) async throws -> Response {
        try await request(path: path, method: "POST", queryItems: [], body: body)
    }

    func getJSONObject(
        _ path: String,
        queryItems: [URLQueryItem] = []
    ) async throws -> Any {
        let data = try await requestData(path: path, method: "GET", queryItems: queryItems, body: nil as EmptyRequest?)
        do {
            return try JSONSerialization.jsonObject(with: data)
        } catch {
            throw APIClientError.transport("响应 JSON 解析失败: \(error.localizedDescription)")
        }
    }

    private func request<Body: Encodable, Response: Decodable>(
        path: String,
        method: String,
        queryItems: [URLQueryItem],
        body: Body?
    ) async throws -> Response {
        let data = try await requestData(path: path, method: method, queryItems: queryItems, body: body)
        do {
            return try JSONDecoder().decode(Response.self, from: data)
        } catch {
            throw APIClientError.transport("响应解析失败: \(error.localizedDescription)")
        }
    }

    private func requestData<Body: Encodable>(
        path: String,
        method: String,
        queryItems: [URLQueryItem],
        body: Body?
    ) async throws -> Data {
        let baseURL = baseURLProvider()
        guard var components = URLComponents(url: baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false) else {
            throw APIClientError.transport("无法构造请求地址。")
        }
        if !queryItems.isEmpty {
            components.queryItems = queryItems
        }
        guard let url = components.url else {
            throw APIClientError.transport("请求地址无效。")
        }

        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let body {
            request.httpBody = try JSONEncoder().encode(body)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw APIClientError.transport(error.localizedDescription)
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIClientError.invalidResponse
        }

        guard (200..<300).contains(httpResponse.statusCode) else {
            if let apiError = try? JSONDecoder().decode(APIErrorPayload.self, from: data) {
                throw APIClientError.server(apiError.error.message)
            }
            let message = String(data: data, encoding: .utf8) ?? "服务调用失败。"
            throw APIClientError.server(message)
        }
        return data
    }
}

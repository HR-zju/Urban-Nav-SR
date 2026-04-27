import Foundation
import AVFoundation

@MainActor
final class VoiceInteractionManager: NSObject, ObservableObject {
    @Published var isSpeaking = false
    @Published var lastSpokenText = ""
    @Published var lastError: String?

    private let synthesizer = AVSpeechSynthesizer()
    private let voice = AVSpeechSynthesisVoice(language: "zh-CN")
    private var completionHandlers: [ObjectIdentifier: () -> Void] = [:]

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func speak(_ text: String, interrupt: Bool = true, onFinish: (() -> Void)? = nil) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if interrupt && synthesizer.isSpeaking {
            completionHandlers.removeAll()
            synthesizer.stopSpeaking(at: .immediate)
        }

        lastSpokenText = trimmed
        lastError = nil

        let utterance = AVSpeechUtterance(string: trimmed)
        utterance.voice = voice
        utterance.rate = 0.46
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0
        if let onFinish {
            completionHandlers[ObjectIdentifier(utterance)] = onFinish
        }
        synthesizer.speak(utterance)
    }

    func stopSpeaking() {
        completionHandlers.removeAll()
        guard synthesizer.isSpeaking else { return }
        synthesizer.stopSpeaking(at: .immediate)
    }

    func resetAll() {
        stopSpeaking()
        lastSpokenText = ""
        lastError = nil
    }

    func announceCandidates(from response: MCPRecommendationResponse, onFinish: (() -> Void)? = nil) {
        speak(Self.buildCandidateAnnouncement(from: response), onFinish: onFinish)
    }

    func announceRoute(from response: MCPRecommendationResponse, onFinish: (() -> Void)? = nil) {
        speak(Self.buildRouteAnnouncement(from: response), onFinish: onFinish)
    }

    func announceRoute(plan: MCPRoutePlan, destinationName: String?, onFinish: (() -> Void)? = nil) {
        speak(Self.buildRouteAnnouncement(from: plan, destinationName: destinationName), onFinish: onFinish)
    }

    private static func buildCandidateAnnouncement(from response: MCPRecommendationResponse) -> String {
        guard !response.recommendations.isEmpty else {
            return "没有找到合适的候选目的地，请换一种说法再试一次。"
        }

        let candidateText = response.recommendations.enumerated().map { index, item in
            let address = item.tags?.address?.trimmingCharacters(in: .whitespacesAndNewlines)
            let distance = formatDistance(item.routeSummary?.distanceM)
            let duration = formatDuration(item.routeSummary?.durationMin ?? item.routeSummary?.durationS.map { $0 / 60.0 })
            let details = [address, distance, duration].compactMap { $0 }.joined(separator: "，")
            if details.isEmpty {
                return "候选\(index + 1)，\(item.name)。"
            }
            return "候选\(index + 1)，\(item.name)，\(details)。"
        }.joined(separator: " ")

        return "已找到 \(response.recommendations.count) 个候选目的地。\(candidateText)"
    }

    private static func buildRouteAnnouncement(from response: MCPRecommendationResponse) -> String {
        let chosen = response.chosenDestination ?? response.recommendations.first
        let destinationName = chosen?.name ?? "目的地"
        let summary = response.routeSummary ?? chosen?.routeSummary
        let distance = formatDistance(summary?.distanceM) ?? "距离信息暂缺"
        let duration = formatDuration(summary?.durationMin ?? summary?.durationS.map { $0 / 60.0 }) ?? "耗时信息暂缺"
        return "已为你选择 \(destinationName)。路线摘要：\(distance)，\(duration)。"
    }

    private static func buildRouteAnnouncement(from plan: MCPRoutePlan, destinationName: String?) -> String {
        let distance = formatDistance(plan.routeSummary?.distanceM) ?? "距离信息暂缺"
        let duration = formatDuration(plan.routeSummary?.durationMin ?? plan.routeSummary?.durationS.map { $0 / 60.0 }) ?? "耗时信息暂缺"
        return "当前路线前往 \(destinationName ?? "目的地")。路线摘要：\(distance)，\(duration)。"
    }

    private static func formatDistance(_ meters: Double?) -> String? {
        guard let meters, meters > 0 else { return nil }
        if meters >= 1000 {
            return String(format: "约 %.1f 公里", meters / 1000.0)
        }
        return String(format: "约 %.0f 米", meters)
    }

    private static func formatDuration(_ minutes: Double?) -> String? {
        guard let minutes, minutes > 0 else { return nil }
        return String(format: "约 %.0f 分钟", minutes)
    }
}

extension VoiceInteractionManager: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = true
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            let completion = self.completionHandlers.removeValue(forKey: ObjectIdentifier(utterance))
            self.isSpeaking = false
            completion?()
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.completionHandlers.removeValue(forKey: ObjectIdentifier(utterance))
            self.isSpeaking = false
        }
    }
}

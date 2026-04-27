import Foundation
import AVFoundation

enum BlindVoiceCommand {
    case selectCandidate(Int)
    case repeatCandidates
    case repeatRoute
    case retryLocation
    case exitVoiceMode
    case refineRequest(String)
}

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

    func announceText(_ text: String, onFinish: (() -> Void)? = nil) {
        speak(text, onFinish: onFinish)
    }

    static func parseCommand(_ text: String, candidateCount: Int) -> BlindVoiceCommand {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let compact = trimmed.replacingOccurrences(of: " ", with: "")

        if compact.contains("重播候选") || compact.contains("再说候选") || compact.contains("重复候选") {
            return .repeatCandidates
        }

        if compact.contains("重新定位")
            || compact.contains("重新获取位置")
            || compact.contains("重新获取定位")
            || compact.contains("定位当前位置")
            || compact.contains("刷新定位")
            || compact.contains("刷新位置") {
            return .retryLocation
        }

        if compact.contains("退出盲人模式")
            || compact.contains("退出语音模式")
            || compact.contains("切换标准模式") {
            return .exitVoiceMode
        }

        if compact.contains("重播路线")
            || compact.contains("重播路径")
            || compact.contains("再说路线")
            || compact.contains("重复路线")
            || compact.contains("重复路径") {
            return .repeatRoute
        }

        if let index = extractCandidateIndex(from: compact, candidateCount: candidateCount) {
            return .selectCandidate(index)
        }

        return .refineRequest(trimmed)
    }

    private static func extractCandidateIndex(from text: String, candidateCount: Int) -> Int? {
        guard candidateCount > 0 else { return nil }

        let hasSelectionCue = text.contains("选")
            || text.contains("第")
            || text.contains("候选")
            || text.contains("号")
            || text.contains("个")

        if hasSelectionCue, let index = extractDigits(from: text), (1...candidateCount).contains(index) {
            return index
        }

        let chineseForms = [
            1: ["第一个", "第一", "一号", "1号", "候选一", "候选1", "选一", "选1"],
            2: ["第二个", "第二", "二号", "2号", "两号", "候选二", "候选2", "选二", "选2", "选两"],
            3: ["第三个", "第三", "三号", "3号", "候选三", "候选3", "选三", "选3"],
            4: ["第四个", "第四", "四号", "4号", "候选四", "候选4", "选四", "选4"],
            5: ["第五个", "第五", "五号", "5号", "候选五", "候选5", "选五", "选5"]
        ]

        for index in 1...candidateCount {
            if chineseForms[index]?.contains(where: { text.contains($0) }) == true {
                return index
            }
        }

        let standaloneForms = [
            1: ["一", "1"],
            2: ["二", "两", "2"],
            3: ["三", "3"],
            4: ["四", "4"],
            5: ["五", "5"]
        ]

        if text.count <= 2 {
            for index in 1...candidateCount {
                if standaloneForms[index]?.contains(text) == true {
                    return index
                }
            }
        }

        return nil
    }

    private static func extractDigits(from text: String) -> Int? {
        let pattern = #"\d+"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, range: range),
              let numberRange = Range(match.range, in: text) else {
            return nil
        }
        return Int(text[numberRange])
    }

    private static func buildCandidateAnnouncement(from response: MCPRecommendationResponse) -> String {
        guard !response.recommendations.isEmpty else {
            return "没有找到合适的候选目的地，请换一种说法再试一次。"
        }

        let candidateText = response.recommendations.enumerated().map { index, item in
            let address = item.tags?.address?.trimmingCharacters(in: .whitespacesAndNewlines)
            let distance = formatDistance(item.routeSummary?.distanceM)
            let duration = formatDuration(item.routeSummary?.durationMin ?? item.routeSummary?.durationS.map { $0 / 60.0 })

            let details = [address, distance, duration]
                .compactMap { value in
                    guard let value, !value.isEmpty else { return nil }
                    return value
                }
                .joined(separator: "，")

            if details.isEmpty {
                return "候选\(index + 1)，\(item.name)。"
            }
            return "候选\(index + 1)，\(item.name)，\(details)。"
        }.joined(separator: " ")

        return """
        已找到 \(response.recommendations.count) 个候选目的地。\(candidateText) 请说选择第几个；如果想继续缩小范围，也可以直接补充要求，例如更近一点、不要商场，或者重播候选。
        """
    }

    private static func buildRouteAnnouncement(from response: MCPRecommendationResponse) -> String {
        let chosen = response.chosenDestination ?? response.recommendations.first
        guard let chosen else {
            return "已经完成规划，但当前没有可播报的路线。"
        }

        let summary = response.routeSummary ?? chosen.routeSummary
        let distance = formatDistance(summary?.distanceM) ?? "距离信息暂缺"
        let duration = formatDuration(summary?.durationMin ?? summary?.durationS.map { $0 / 60.0 }) ?? "耗时信息暂缺"

        let waypointTexts = response.keyWaypoints.prefix(5).enumerated().compactMap { index, waypoint in
            let rawInstruction = waypoint.instruction?.trimmingCharacters(in: .whitespacesAndNewlines)
            if let rawInstruction, !rawInstruction.isEmpty {
                return "第\(index + 1)步，\(rawInstruction)。"
            }

            let fallback = [waypoint.action, waypoint.road]
                .compactMap { value in
                    guard let value, !value.isEmpty else { return nil }
                    return value
                }
                .joined(separator: "，")
            guard !fallback.isEmpty else { return nil }
            return "第\(index + 1)步，\(fallback)。"
        }

        let waypointAnnouncement: String
        if waypointTexts.isEmpty {
            waypointAnnouncement = "当前只拿到了路线摘要，还没有更细的关键路径播报。"
        } else {
            waypointAnnouncement = "关键路径如下。\(waypointTexts.joined(separator: " "))"
        }

        return """
        已为你选择 \(chosen.name)。路线摘要：\(distance)，\(duration)。\(waypointAnnouncement) 如果你想重新筛选候选，请直接说新的要求；如果想再次收听路线，请说重播路线。
        """
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

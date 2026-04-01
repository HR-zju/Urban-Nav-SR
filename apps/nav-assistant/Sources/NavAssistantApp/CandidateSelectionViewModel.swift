import Foundation
import CoreLocation

@MainActor
final class CandidateSelectionViewModel: ObservableObject {
    @Published var promptInput = ""
    @Published var dialogHistory: [String] = []
    @Published var candidates: [AddressCandidate] = []
    @Published var selectedCandidate: AddressCandidate?
    @Published var statusText = "等待输入 prompt"
    @Published var isPlanning = false
    @Published var planningError: String?
    @Published var lastPlanningResult: MCPRecommendationResponse?

    private let service: AddressCandidateService
    private var lastConfirmedText: String = ""
    private var lastLocation: CLLocationCoordinate2D?
    private var lastPlanningPrompt: String = ""
    private var refinementMessages: [String] = []

    init(service: AddressCandidateService = MCPAddressCandidateService()) {
        self.service = service
    }

    func fetchCandidates() async {
        let prompt = promptInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            statusText = "请输入地址相关 prompt"
            return
        }
        do {
            let result = try await service.fetchCandidates(prompt: prompt, history: dialogHistory)
            candidates = result
            selectedCandidate = nil
            dialogHistory.append("用户: \(prompt)")
            statusText = "收到 \(result.count) 个候选地址"
        } catch {
            statusText = "候选地址查询失败: \(error.localizedDescription)"
        }
    }

    func chooseCandidate(_ candidate: AddressCandidate) async {
        selectedCandidate = candidate
        statusText = "已选择: \(candidate.title)"

        guard let selectedIndex = candidates.firstIndex(where: { $0.id == candidate.id }).map({ $0 + 1 }) else {
            return
        }
        guard let currentLocation = lastLocation, !lastPlanningPrompt.isEmpty else {
            return
        }

        isPlanning = true
        planningError = nil
        statusText = "已选择: \(candidate.title)，正在更新该候选路线..."
        defer { isPlanning = false }
        NSLog("[CandidateVM] Replan with selectedIndex=%@, prompt=%@", String(selectedIndex), lastPlanningPrompt)

        do {
            let result = try await service.planFromConfirmedText(
                confirmedText: lastPlanningPrompt,
                currentLocation: currentLocation,
                mode: "car",
                maxCandidates: 5,
                searchRadius: 3000,
                selectedIndex: selectedIndex
            )
            applyPlanningResult(result, successText: "已按所选候选更新路线")
        } catch {
            planningError = error.localizedDescription
            statusText = "已选择: \(candidate.title)，但路线更新失败: \(error.localizedDescription)"
        }
    }

    func continueDialog(_ message: String) {
        let text = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        dialogHistory.append("继续对话: \(text)")
        statusText = "已记录多轮上下文"
    }

    func planFromConfirmedText(
        _ text: String,
        currentLocation: CLLocationCoordinate2D
    ) async {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else {
            statusText = "确认文本为空，无法规划"
            return
        }

        isPlanning = true
        planningError = nil
        statusText = "正在调用 MCP 规划..."
        selectedCandidate = nil
        lastConfirmedText = normalized
        refinementMessages = []
        let planningPrompt = buildPlanningPrompt(baseRequirement: normalized, refinements: [])
        lastPlanningPrompt = planningPrompt
        NSLog("[CandidateVM] Initial planning prompt=%@", planningPrompt)
        lastLocation = currentLocation
        defer { isPlanning = false }

        do {
            let result = try await service.planFromConfirmedText(
                confirmedText: planningPrompt,
                currentLocation: currentLocation,
                mode: "car",
                maxCandidates: 5,
                searchRadius: 3000,
                selectedIndex: 1
            )
            dialogHistory.append("确认文本: \(normalized)")
            applyPlanningResult(result, successText: "规划完成")
        } catch {
            planningError = error.localizedDescription
            statusText = "规划失败: \(error.localizedDescription)"
        }
    }

    func continueDialogAndReplan(_ message: String) async {
        let text = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        guard let currentLocation = lastLocation else {
            statusText = "缺少定位信息，请先执行一次确认文本规划"
            return
        }
        guard !lastConfirmedText.isEmpty else {
            statusText = "缺少初始确认文本，请先执行一次确认文本规划"
            return
        }

        continueDialog(text)
        refinementMessages.append(text)
        isPlanning = true
        planningError = nil
        statusText = "正在根据继续对话更新候选..."
        selectedCandidate = nil
        defer { isPlanning = false }

        let mergedPrompt = buildPlanningPrompt(
            baseRequirement: lastConfirmedText,
            refinements: refinementMessages
        )
        lastPlanningPrompt = mergedPrompt
        NSLog("[CandidateVM] Continued planning prompt=%@", mergedPrompt)

        do {
            let result = try await service.planFromConfirmedText(
                confirmedText: mergedPrompt,
                currentLocation: currentLocation,
                mode: "car",
                maxCandidates: 5,
                searchRadius: 3000,
                selectedIndex: 1
            )
            applyPlanningResult(result, successText: "继续对话完成")
        } catch {
            planningError = error.localizedDescription
            statusText = "继续对话规划失败: \(error.localizedDescription)"
        }
    }

    private func buildPlanningPrompt(baseRequirement: String, refinements: [String]) -> String {
        guard !refinements.isEmpty else {
            return baseRequirement
        }
        let refinementText = refinements.enumerated().map { index, text in
            "\(index + 1). \(text)"
        }.joined(separator: "\n")
        return """
        原始导航需求：
        \(baseRequirement)
        
        补充约束（后面的约束优先级更高）：
        \(refinementText)
        
        请严格根据以上需求返回候选目的地；若约束冲突，以补充约束为准。
        """
    }

    private func applyPlanningResult(_ result: MCPRecommendationResponse, successText: String) {
        lastPlanningResult = result
        candidates = result.recommendations.map { $0.toAddressCandidate() }
        planningError = nil
        statusText = "\(successText)，收到 \(candidates.count) 个候选地址"
    }
}

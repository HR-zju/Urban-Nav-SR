import SwiftUI

struct VoiceAssistantSheet: View {
    @ObservedObject var viewModel: NavigationFlowViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("播报") {
                    Button("播报候选列表") {
                        viewModel.announceCandidates()
                    }
                    .disabled(viewModel.recommendationResponse == nil)

                    Button("播报当前路线") {
                        viewModel.announceCurrentRoute()
                    }
                    .disabled(viewModel.mapViewModel.routeSummary == nil)

                    Button("停止播报") {
                        viewModel.voice.stopSpeaking()
                    }
                }

                Section("当前状态") {
                    LabeledContent("是否正在播报", value: viewModel.voice.isSpeaking ? "是" : "否")
                    LabeledContent("最近一次播报", value: viewModel.voice.lastSpokenText.isEmpty ? "无" : viewModel.voice.lastSpokenText)
                    LabeledContent("当前阶段", value: viewModel.flowStage.title)
                    LabeledContent("路线会话", value: viewModel.routeSessionStatusText)
                    if let sessionID = viewModel.activeRouteSessionID {
                        LabeledContent("Session ID", value: sessionID)
                    }
                    if let waypointPushStatusMessage = viewModel.waypointPushStatusMessage, !waypointPushStatusMessage.isEmpty {
                        LabeledContent("Waypoint 下发", value: waypointPushStatusMessage)
                    }
                }
            }
            .navigationTitle("语音助手")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("关闭") {
                        dismiss()
                    }
                }
            }
        }
    }
}

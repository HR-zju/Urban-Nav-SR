import SwiftUI

struct ContentView: View {
    @ObservedObject var viewModel: NavigationFlowViewModel
    @State private var showingResetConfirmation = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    RequestView(viewModel: viewModel)

                    if !viewModel.candidates.isEmpty {
                        CandidateListView(viewModel: viewModel)
                    }

                    RouteMapView(
                        viewModel: viewModel.mapViewModel,
                        navigationViewModel: viewModel
                    )
                }
                .padding(20)
            }
            .navigationTitle("Nav Assistant")
            .toolbar {
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button {
                        showingResetConfirmation = true
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                    }

                    Button {
                        viewModel.showingVoiceAssistant = true
                    } label: {
                        Image(systemName: "waveform")
                    }
                }
            }
            .confirmationDialog("重置当前导航状态？", isPresented: $showingResetConfirmation, titleVisibility: .visible) {
                Button("重置当前状态", role: .destructive) {
                    viewModel.resetAll()
                }
                Button("取消", role: .cancel) {
                }
            } message: {
                Text("会清空导航需求、候选列表、路线会话、语音播报和地图状态。")
            }
            .sheet(isPresented: $viewModel.showingVoiceAssistant) {
                VoiceAssistantSheet(viewModel: viewModel)
            }
            .onChange(of: viewModel.speech.combinedText) { _, newValue in
                guard !newValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
                viewModel.promptText = newValue
            }
            .onDisappear {
                viewModel.disconnectRouteSession()
            }
        }
    }
}

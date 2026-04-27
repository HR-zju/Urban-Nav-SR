import SwiftUI

struct RequestView: View {
    @ObservedObject var viewModel: NavigationFlowViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                Text("导航需求")
                    .font(.title2.bold())
                Spacer()
                Label(viewModel.flowStage.title, systemImage: viewModel.flowStage.systemImage)
                    .font(.caption.weight(.medium))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(
                        Capsule(style: .continuous)
                            .fill(viewModel.flowStage == .error ? Color.red.opacity(0.12) : Color.blue.opacity(0.12))
                    )
                    .foregroundStyle(viewModel.flowStage == .error ? .red : .blue)
            }

            Text(viewModel.mapViewModel.statusText)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Text(viewModel.flowStageSummary)
                .font(.footnote)
                .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                Button("请求定位权限") {
                    viewModel.mapViewModel.requestLocationPermission()
                }
                Button("定位当前位置") {
                    viewModel.mapViewModel.locateCurrentPosition()
                }
                .buttonStyle(.borderedProminent)
            }

            Picker("识别输出", selection: $viewModel.outputMode) {
                ForEach(RecognitionOutputMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            TextEditor(text: $viewModel.promptText)
                .frame(minHeight: 120)
                .padding(10)
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(Color(uiColor: .secondarySystemBackground))
                )

            HStack(spacing: 12) {
                Button(viewModel.speech.isRunning ? "停止录音" : "开始录音") {
                    Task { await viewModel.toggleSpeechRecognition() }
                }
                .buttonStyle(.bordered)

                Button("同步识别文本") {
                    viewModel.syncPromptFromSpeech()
                }

                Spacer()

                Button(viewModel.isLoading ? "请求中…" : "推荐目的地") {
                    Task { await viewModel.requestRecommendations() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isLoading)
            }

            if let errorMessage = viewModel.errorMessage, !errorMessage.isEmpty {
                Text(errorMessage)
                    .font(.footnote)
                    .foregroundStyle(.red)
            }

            if let speechError = viewModel.speech.lastError, !speechError.isEmpty {
                Text(speechError)
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color(uiColor: .systemBackground))
                .shadow(color: Color.black.opacity(0.06), radius: 14, y: 6)
        )
    }
}

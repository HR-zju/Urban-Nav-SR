import SwiftUI

struct RequestView: View {
    @ObservedObject var viewModel: NavigationFlowViewModel
    @ObservedObject var appConfiguration: AppConfiguration
    @FocusState private var isPromptFocused: Bool
    @State private var apiBaseURLDraft = ""
    @State private var apiBaseURLErrorMessage: String?

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

            VStack(alignment: .leading, spacing: 8) {
                Text("后端地址")
                    .font(.headline)

                TextField("例如：http://10.116.95.131:8080", text: $apiBaseURLDraft)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.URL)
                    .submitLabel(.done)
                    .onSubmit {
                        applyAPIBaseURL()
                    }
                    .padding(10)
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(Color(uiColor: .secondarySystemBackground))
                    )

                HStack(spacing: 12) {
                    Button("应用地址") {
                        applyAPIBaseURL()
                    }
                    .buttonStyle(.bordered)

                    Button("恢复默认") {
                        appConfiguration.resetAPIBaseURL()
                        apiBaseURLDraft = appConfiguration.apiBaseURLText
                        apiBaseURLErrorMessage = nil
                        hideKeyboard()
                    }
                    .buttonStyle(.bordered)
                }

                Text("当前生效：\(appConfiguration.apiBaseURLText)")
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                if let apiBaseURLErrorMessage, !apiBaseURLErrorMessage.isEmpty {
                    Text(apiBaseURLErrorMessage)
                        .font(.footnote)
                        .foregroundStyle(.orange)
                }
            }

            TextEditor(text: $viewModel.promptText)
                .focused($isPromptFocused)
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
                    isPromptFocused = false
                    hideKeyboard()
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
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("完成") {
                    isPromptFocused = false
                    hideKeyboard()
                }
            }
        }
        .onAppear {
            apiBaseURLDraft = appConfiguration.apiBaseURLText
        }
        .onChange(of: appConfiguration.apiBaseURLText) { _, newValue in
            apiBaseURLDraft = newValue
        }
    }

    private func applyAPIBaseURL() {
        if appConfiguration.updateAPIBaseURL(apiBaseURLDraft) {
            apiBaseURLDraft = appConfiguration.apiBaseURLText
            apiBaseURLErrorMessage = nil
            hideKeyboard()
            return
        }
        apiBaseURLErrorMessage = "后端地址无效，请输入 http://IP:端口 或 https://域名。"
    }
}

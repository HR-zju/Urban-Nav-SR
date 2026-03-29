import SwiftUI

struct ContentView: View {
    @StateObject private var speech = SpeechRecognitionManager()
    @StateObject private var mapVM = MapNavigationViewModel()
    @StateObject private var candidateVM = CandidateSelectionViewModel()

    @State private var outputMode: RecognitionOutputMode = .streaming
    @State private var editableText = ""
    @State private var confirmedRecords: [ConfirmedTextRecord] = []
    @State private var continueDialogInput = ""
    @State private var recordingTarget: RecordingTarget = .mainText
    @State private var continueDialogBaselineText = ""

    var body: some View {
        HSplitView {
            leftPanel
                .frame(minWidth: 520)
            MapPanelView(viewModel: mapVM)
                .frame(minWidth: 420)
        }
        .frame(minWidth: 980, minHeight: 680)
        .onChange(of: speech.combinedText) { _, newValue in
            switch recordingTarget {
            case .continueDialog:
                continueDialogInput = extractContinueDialogText(from: newValue)
            case .mainText:
                editableText = newValue
            }
        }
        .onReceive(candidateVM.$selectedCandidate) { newValue in
            guard let selected = newValue else { return }
            Task { await mapVM.applySelectedCandidate(selected) }
        }
        .onReceive(candidateVM.$lastPlanningResult) { newValue in
            guard let planning = newValue else { return }
            Task { await mapVM.applyPlanningResult(planning) }
        }
    }

    // MARK: - Layout

    private var leftPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                speechSection
                editableTextSection
                continueDialogSection
                candidateSection
            }
            .padding()
        }
    }

    // MARK: - Speech

    private var speechSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("语音识别")
                    .font(.headline)
                Spacer()
                if speech.isRunning {
                    Text(recordingTarget == .mainText ? "● 主识别录音中" : "● 对话录音中")
                        .font(.caption2.bold())
                        .foregroundStyle(.orange)
                }
            }

            HStack {
                Picker("输出模式", selection: $outputMode) {
                    ForEach(RecognitionOutputMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 180)

                Button(speech.isRunning && recordingTarget == .mainText ? "停止录音" : "开始录音") {
                    toggleMainRecording()
                }
            }

            if let error = speech.lastError {
                Text(error).font(.caption).foregroundStyle(.red)
            }

            Text(speech.authorizationSummary)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Editable Text (compact)

    private var editableTextSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("识别结果（可编辑）")
                .font(.subheadline)

            TextEditor(text: $editableText)
                .font(.body.monospaced())
                .frame(minHeight: 60, maxHeight: 90)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.quaternary, lineWidth: 1)
                )

            HStack(spacing: 10) {
                Button("同步识别文本") {
                    editableText = speech.combinedText
                }
                .controlSize(.small)

                Button(candidateVM.isPlanning ? "规划中…" : "确认文本并规划") {
                    let normalized = editableText.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !normalized.isEmpty else { return }
                    confirmedRecords.insert(
                        ConfirmedTextRecord(text: normalized, createdAt: Date()),
                        at: 0
                    )
                    guard let current = mapVM.userCoordinate else {
                        mapVM.statusText = "请先点击「定位当前位置」后再进行规划"
                        return
                    }
                    Task {
                        await candidateVM.planFromConfirmedText(normalized, currentLocation: current)
                    }
                }
                .disabled(candidateVM.isPlanning || editableText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .controlSize(.small)
            }

            if let latest = confirmedRecords.first {
                Text("最近确认：\(latest.text)")
                    .lineLimit(1)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Continue Dialog (above candidates)

    private var continueDialogSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("继续对话")
                .font(.subheadline)

            TextField("例如：更近一点、不要商场、优先地铁站附近", text: $continueDialogInput)
                .textFieldStyle(.roundedBorder)

            HStack(spacing: 10) {
                Button(speech.isRunning && recordingTarget == .continueDialog ? "停止对话录音" : "对话录音") {
                    toggleContinueDialogRecording()
                }
                .controlSize(.small)

                Button("继续对话并更新候选") {
                    let message = continueDialogInput
                    continueDialogInput = ""
                    Task {
                        await candidateVM.continueDialogAndReplan(message)
                    }
                }
                .controlSize(.small)
                .disabled(
                    candidateVM.isPlanning
                    || continueDialogInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    || candidateVM.candidates.isEmpty
                )
            }

            if !candidateVM.dialogHistory.isEmpty {
                ScrollView {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(Array(candidateVM.dialogHistory.enumerated()), id: \.offset) { _, msg in
                            Text(msg)
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(.background.secondary)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                .frame(maxHeight: 110)
            }
        }
        .padding(10)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Candidate Addresses (bottom)

    private var candidateSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("候选地址")
                .font(.headline)

            Text(candidateVM.statusText)
                .font(.caption)
                .foregroundStyle(.secondary)

            if let planningError = candidateVM.planningError {
                Text(planningError).font(.caption).foregroundStyle(.red)
            }

            if candidateVM.candidates.isEmpty && !candidateVM.isPlanning {
                Text("点击「确认文本并规划」自动获取候选地址")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            ForEach(candidateVM.candidates) { candidate in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(candidate.title).font(.body.bold())
                        Text(candidate.subtitle).font(.caption).foregroundStyle(.secondary)
                        HStack(spacing: 8) {
                            if let d = candidate.distanceM {
                                Text(String(format: "%.2f km", d / 1000.0))
                            }
                            if let t = candidate.durationMin {
                                Text(String(format: "%.1f 分钟", t))
                            }
                            if let s = candidate.score {
                                Text(String(format: "score %.2f", s))
                            }
                        }
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                    }
                    Spacer()
                    Button("选择") {
                        Task {
                            await candidateVM.chooseCandidate(candidate)
                        }
                    }
                    .disabled(candidateVM.isPlanning)
                }
                .padding(8)
                .background(.background.secondary)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
        .padding(12)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Recording Control

    private func toggleMainRecording() {
        if speech.isRunning && recordingTarget == .mainText {
            speech.stopRecognition()
            return
        }
        if speech.isRunning {
            speech.stopRecognition()
        }
        recordingTarget = .mainText
        Task {
            guard await speech.requestPermissionsIfNeeded() else { return }
            speech.startRecognition(mode: outputMode)
        }
    }

    private func toggleContinueDialogRecording() {
        if speech.isRunning && recordingTarget == .continueDialog {
            speech.stopRecognition()
            recordingTarget = .mainText
            return
        }
        if speech.isRunning {
            speech.stopRecognition()
        }
        recordingTarget = .continueDialog
        continueDialogBaselineText = speech.combinedText
        Task {
            guard await speech.requestPermissionsIfNeeded() else { return }
            speech.startRecognition(mode: .streaming)
        }
    }

    private func extractContinueDialogText(from combined: String) -> String {
        if combined.hasPrefix(continueDialogBaselineText) {
            let idx = combined.index(combined.startIndex, offsetBy: continueDialogBaselineText.count)
            return String(combined[idx...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return combined.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private enum RecordingTarget {
    case mainText
    case continueDialog
}

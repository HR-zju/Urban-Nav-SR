import SwiftUI
import CoreLocation

struct ContentView: View {
    @StateObject private var speech = SpeechRecognitionManager()
    @StateObject private var voice = VoiceInteractionManager()
    @StateObject private var mapVM = MapNavigationViewModel()
    @StateObject private var candidateVM = CandidateSelectionViewModel()

    @State private var interactionMode: AppInteractionMode = .standard
    @State private var outputMode: RecognitionOutputMode = .streaming
    @State private var editableText = ""
    @State private var confirmedRecords: [ConfirmedTextRecord] = []
    @State private var continueDialogInput = ""
    @State private var recordingTarget: RecordingTarget = .mainText
    @State private var continueDialogBaselineText = ""
    @State private var blindRecordingContext: BlindRecordingContext?
    @State private var blindPhase: BlindWorkflowPhase = .inactive
    @State private var pendingBlindNavigationRequest: String?
    @State private var suppressBlindRecognitionResult = false
    @State private var lastBlindLocationAlert = ""

    var body: some View {
        HSplitView {
            leftPanel
                .frame(minWidth: 520)
            MapPanelView(viewModel: mapVM, isVoiceOnlyMode: interactionMode == .voiceOnly)
                .frame(minWidth: 420)
        }
        .frame(minWidth: 980, minHeight: 680)
        .onChange(of: speech.combinedText) { _, newValue in
            guard interactionMode == .standard else { return }
            switch recordingTarget {
            case .continueDialog:
                continueDialogInput = extractContinueDialogText(from: newValue)
            case .mainText:
                editableText = newValue
            }
        }
        .onChange(of: speech.isRunning) { wasRunning, isRunning in
            guard interactionMode == .voiceOnly, wasRunning, !isRunning else { return }

            if suppressBlindRecognitionResult {
                suppressBlindRecognitionResult = false
                blindRecordingContext = nil
                speech.resetTranscript()
                return
            }

            guard let context = blindRecordingContext else { return }
            let transcript = speech.combinedText.trimmingCharacters(in: .whitespacesAndNewlines)
            blindRecordingContext = nil
            speech.resetTranscript()

            Task {
                await handleBlindVoiceInput(transcript, context: context)
            }
        }
        .onChange(of: interactionMode) { _, newValue in
            cancelBlindListening()
            voice.stopSpeaking()
            pendingBlindNavigationRequest = nil
            lastBlindLocationAlert = ""

            if newValue == .voiceOnly {
                beginBlindVoiceMode()
            } else {
                blindPhase = .inactive
            }
        }
        .onChange(of: mapVM.statusText) { _, newValue in
            guard interactionMode == .voiceOnly, mapVM.userCoordinate == nil else { return }

            let alertMessage: String?
            if newValue.contains("定位权限被拒绝") {
                alertMessage = "定位权限被拒绝。请先在系统设置允许定位，允许后点击定位当前位置；如果你已经录过需求，定位成功后我会自动规划。"
            } else if newValue.contains("定位失败") {
                alertMessage = "当前位置定位失败。请点击定位当前位置重试；如果你已经录过需求，定位成功后我会自动规划。"
            } else {
                alertMessage = nil
            }

            guard let alertMessage, lastBlindLocationAlert != newValue else { return }
            lastBlindLocationAlert = newValue
            blindSpeak(alertMessage)
        }
        .onReceive(candidateVM.$selectedCandidate) { newValue in
            guard let selected = newValue else { return }
            Task { await mapVM.applySelectedCandidate(selected) }
        }
        .onReceive(mapVM.$userCoordinate) { newValue in
            guard interactionMode == .voiceOnly, newValue != nil else { return }
            Task {
                await handleBlindLocationReady()
            }
        }
        .onReceive(candidateVM.$lastPlanningResult) { newValue in
            guard let planning = newValue else { return }
            Task { await mapVM.applyPlanningResult(planning) }
        }
    }

    private var leftPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                modeSection
                if interactionMode == .voiceOnly {
                    blindVoiceSection
                } else {
                    speechSection
                    editableTextSection
                    continueDialogSection
                    candidateSection
                }
            }
            .padding()
        }
    }

    private var modeSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("交互模式")
                .font(.headline)

            Picker("交互模式", selection: $interactionMode) {
                ForEach(AppInteractionMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            Text(interactionMode == .voiceOnly
                 ? "盲人语音模式使用按钮触发录音，系统会把候选和路线结果用语音播报出来。"
                 : "标准模式保留文本编辑、候选列表和地图联动。")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private var blindVoiceSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("语音交互模式已开启")
                    .font(.headline)
                Spacer()
                if speech.isRunning {
                    Text("● 正在录音")
                        .font(.caption2.bold())
                        .foregroundStyle(.orange)
                } else if voice.isSpeaking {
                    Text("● 正在播报")
                        .font(.caption2.bold())
                        .foregroundStyle(.green)
                }
            }

            Text("盲人模式下由用户主动按按钮录音。你可以先定位，再按录音按钮说出目的地；有候选后也可以再次按录音按钮说“选择第一个”或补充要求。")
                .font(.caption)
                .foregroundStyle(.secondary)

            statusBlock(title: "当前阶段", text: blindPhase.description)
            statusBlock(title: "候选状态", text: candidateVM.statusText)
            statusBlock(title: "定位状态", text: mapVM.statusText)

            HStack(spacing: 10) {
                Button("请求定位权限") {
                    mapVM.requestLocationPermission()
                    if interactionMode == .voiceOnly {
                        blindSpeak("已请求定位权限。")
                    }
                }

                Button("定位当前位置") {
                    blindPhase = .waitingForLocation
                    mapVM.locateCurrentPosition()
                    if interactionMode == .voiceOnly {
                        blindSpeak("正在定位当前位置。")
                    }
                }
            }

            HStack(spacing: 10) {
                Button(speech.isRunning ? "结束录音" : "按按钮开始录音") {
                    toggleBlindRecording()
                }
                .controlSize(.large)
                .disabled(candidateVM.isPlanning || voice.isSpeaking)

                Button("停止播报") {
                    voice.stopSpeaking()
                    if blindPhase == .announcingCandidates || blindPhase == .announcingRoute {
                        blindPhase = .readyToRecord
                    }
                }
                .disabled(!voice.isSpeaking)
            }

            Text("按录音按钮后可直接说：带我去最近的地铁站、选择第一个、更近一点、重播候选、重播路线、重新定位、退出盲人模式。")
                .font(.caption)
                .foregroundStyle(.secondary)

            if let pendingBlindNavigationRequest, !pendingBlindNavigationRequest.isEmpty {
                statusBlock(title: "已缓存需求", text: pendingBlindNavigationRequest)
            }

            if let planningError = candidateVM.planningError {
                Text(planningError)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            if let speechError = speech.lastError {
                Text(speechError)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            if !voice.lastSpokenText.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("最近播报内容")
                        .font(.caption.bold())
                    Text(voice.lastSpokenText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
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

    private func statusBlock(title: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption.bold())
            Text(text)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(.background.secondary)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

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
                    Task { await submitConfirmedText(editableText) }
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
                        await replanWithDialog(message)
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
                            await selectCandidate(candidate)
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

    private func beginBlindVoiceMode() {
        blindPhase = .readyToRecord
        let intro = mapVM.userCoordinate == nil
            ? "已进入盲人语音模式。请先点击定位当前位置，然后按录音按钮说出导航需求。说完后我会用语音播报结果。"
            : "已进入盲人语音模式。当前位置已就绪。请按录音按钮说出导航需求，说完后我会用语音播报结果。"
        blindSpeak(intro)
    }

    private func handleBlindLocationReady() async {
        guard interactionMode == .voiceOnly else { return }
        lastBlindLocationAlert = ""

        if let pending = pendingBlindNavigationRequest,
           !pending.isEmpty,
           !voice.isSpeaking,
           !speech.isRunning,
           !candidateVM.isPlanning {
            await submitConfirmedText(pending)
            return
        }

        if blindPhase == .waitingForLocation,
           !voice.isSpeaking,
           !speech.isRunning,
           !candidateVM.isPlanning {
            blindPhase = .readyToRecord
            blindSpeak("当前位置已确认。现在可以按录音按钮说出导航需求。")
        }
    }

    private func handleBlindVoiceInput(_ transcript: String, context: BlindRecordingContext) async {
        guard !transcript.isEmpty else {
            blindPhase = .readyToRecord
            blindSpeak("没有听清楚，请再按一次录音按钮。")
            return
        }

        switch context {
        case .voiceInput:
            await handleBlindVoiceCommand(transcript)
        }
    }

    private func handleBlindVoiceCommand(_ transcript: String) async {
        let command = VoiceInteractionManager.parseCommand(transcript, candidateCount: candidateVM.candidates.count)
        switch command {
        case .repeatCandidates:
            announceCandidateList()
        case .repeatRoute:
            announceRouteSummary()
        case .retryLocation:
            retryBlindLocation()
        case .exitVoiceMode:
            interactionMode = .standard
        case .selectCandidate(let index):
            guard candidateVM.candidates.indices.contains(index - 1) else {
                blindPhase = .readyToRecord
                blindSpeak("候选编号超出范围，请再按录音按钮重新说选择第几个。")
                return
            }
            await selectCandidate(candidateVM.candidates[index - 1])
        case .refineRequest(let message):
            if shouldTreatAsFreshNavigationRequest(message) {
                await submitConfirmedText(message)
            } else {
                await replanWithDialog(message)
            }
        }
    }

    private func submitConfirmedText(_ text: String) async {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return }

        rememberConfirmedText(normalized)

        if interactionMode == .voiceOnly {
            pendingBlindNavigationRequest = normalized
        }

        guard let current = mapVM.userCoordinate else {
            if interactionMode == .voiceOnly {
                blindPhase = .waitingForLocation
                blindSpeak("已记住你的导航需求。请点击定位当前位置；定位成功后我会自动规划并播报候选。")
            } else {
                mapVM.statusText = "请先点击「定位当前位置」后再进行规划"
            }
            return
        }

        pendingBlindNavigationRequest = nil
        if interactionMode == .voiceOnly {
            blindPhase = .planning
        }

        await candidateVM.planFromConfirmedText(normalized, currentLocation: current)
        if interactionMode == .voiceOnly {
            announceCandidateList()
        }
    }

    private func rememberConfirmedText(_ normalized: String) {
        editableText = normalized
        guard confirmedRecords.first?.text != normalized else { return }
        confirmedRecords.insert(
            ConfirmedTextRecord(text: normalized, createdAt: Date()),
            at: 0
        )
    }

    private func replanWithDialog(_ message: String) async {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        continueDialogInput = trimmed
        if interactionMode == .voiceOnly {
            blindPhase = .planning
        }
        await candidateVM.continueDialogAndReplan(trimmed)
        if interactionMode == .voiceOnly {
            announceCandidateList()
        }
    }

    private func selectCandidate(_ candidate: AddressCandidate) async {
        if interactionMode == .voiceOnly {
            blindPhase = .planning
        }
        await candidateVM.chooseCandidate(candidate)
        if interactionMode == .voiceOnly {
            announceRouteSummary()
        }
    }

    private func announceCandidateList() {
        if let error = candidateVM.planningError {
            blindPhase = .readyToRecord
            blindSpeak("规划失败。\(error)")
            return
        }

        guard let result = candidateVM.lastPlanningResult else {
            blindPhase = .readyToRecord
            blindSpeak(candidateVM.statusText)
            return
        }

        guard interactionMode == .voiceOnly else {
            voice.announceCandidates(from: result)
            return
        }

        blindPhase = .announcingCandidates
        voice.announceCandidates(from: result) {
            Task { @MainActor in
                self.blindPhase = .readyToRecord
            }
        }
    }

    private func announceRouteSummary() {
        if let error = candidateVM.planningError {
            blindPhase = .readyToRecord
            blindSpeak("路线更新失败。\(error)")
            return
        }

        guard let result = candidateVM.lastPlanningResult else {
            blindPhase = .readyToRecord
            blindSpeak("当前还没有可播报的路线。")
            return
        }

        guard interactionMode == .voiceOnly else {
            voice.announceRoute(from: result)
            return
        }

        blindPhase = .announcingRoute
        voice.announceRoute(from: result) {
            Task { @MainActor in
                self.blindPhase = .readyToRecord
            }
        }
    }

    private func retryBlindLocation() {
        blindPhase = .waitingForLocation
        lastBlindLocationAlert = ""
        mapVM.requestLocationPermission()
        mapVM.locateCurrentPosition()
        blindSpeak("正在重新定位当前位置。定位完成后如果已有需求，我会自动规划并播报候选。")
    }

    private func toggleBlindRecording() {
        if speech.isRunning {
            speech.stopRecognition()
            return
        }

        if voice.isSpeaking {
            voice.stopSpeaking()
        }

        speech.resetTranscript()
        blindRecordingContext = .voiceInput
        blindPhase = .recording
        recordingTarget = .mainText

        Task {
            guard await speech.requestPermissionsIfNeeded() else {
                blindRecordingContext = nil
                blindPhase = .readyToRecord
                blindSpeak(speech.lastError ?? "语音权限不可用。")
                return
            }
            speech.startRecognition(mode: .sentence)
        }
    }

    private func blindSpeak(_ text: String) {
        guard interactionMode == .voiceOnly else {
            voice.announceText(text)
            return
        }
        voice.announceText(text)
    }

    private func cancelBlindListening() {
        if speech.isRunning {
            suppressBlindRecognitionResult = true
            speech.stopRecognition()
        }
        blindRecordingContext = nil
        speech.resetTranscript()
    }

    private func shouldTreatAsFreshNavigationRequest(_ text: String) -> Bool {
        if candidateVM.candidates.isEmpty {
            return true
        }

        let compact = text.replacingOccurrences(of: " ", with: "")
        let freshRequestHints = [
            "带我去", "导航到", "我要去", "送我去", "前往",
            "去最近的", "去附近的", "新的目的地", "重新开始", "换个地方"
        ]

        return freshRequestHints.contains(where: { compact.contains($0) })
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

private enum BlindRecordingContext {
    case voiceInput
}

private enum BlindWorkflowPhase {
    case inactive
    case readyToRecord
    case waitingForLocation
    case recording
    case planning
    case announcingCandidates
    case announcingRoute

    var description: String {
        switch self {
        case .inactive:
            return "盲人模式未启用"
        case .readyToRecord:
            return "等待用户按按钮开始录音"
        case .waitingForLocation:
            return "等待定位，定位成功后会自动规划已记录的需求"
        case .recording:
            return "正在录音，等待用户说完导航需求或语音指令"
        case .planning:
            return "正在根据本次录音内容处理导航请求"
        case .announcingCandidates:
            return "正在播报候选目的地"
        case .announcingRoute:
            return "正在播报路线摘要和关键路径"
        }
    }
}

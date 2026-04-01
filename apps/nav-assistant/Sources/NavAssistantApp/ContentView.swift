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
                .frame(minWidth: 500, idealWidth: 560)
            MapPanelView(viewModel: mapVM, isVoiceOnlyMode: isVoiceOnlyMode)
                .frame(minWidth: 500)
        }
        .background(Color.navWindow)
        .frame(minWidth: 1180, minHeight: 760)
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

    private var isVoiceOnlyMode: Bool {
        interactionMode == .voiceOnly
    }

    private var leftPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                overviewSection
                modeSection
                if isVoiceOnlyMode {
                    blindVoiceSection
                    blindCommandSection
                } else {
                    requestSection
                    continueDialogSection
                    candidateSection
                }
            }
            .padding(20)
        }
        .background(Color.navWindow)
    }

    private var overviewSection: some View {
        NavPanel(
            title: isVoiceOnlyMode ? "语音导航指挥台" : "桌面导航工作台",
            subtitle: isVoiceOnlyMode
                ? "把定位、录音、候选播报和路线播报集中到一个可预测区域，降低视觉切换成本。"
                : "围绕“说出需求 -> 确认候选 -> 查看路线”展开，保留桌面端精细编辑与地图联动。",
            tone: isVoiceOnlyMode ? .accent : .neutral,
            accessory: {
                NavStatusBadge(text: overallStatusText, tone: overallStatusTone, systemImage: overallStatusSymbol)
            }
        ) {
            HStack(spacing: 12) {
                NavMetricBadge(label: "当前模式", value: interactionMode.rawValue, emphasized: true)
                NavMetricBadge(label: "当前位置", value: mapVM.userCoordinate == nil ? "未定位" : "已获取")
                NavMetricBadge(
                    label: isVoiceOnlyMode ? "当前阶段" : "路线状态",
                    value: isVoiceOnlyMode ? blindPhase.shortTitle : routeReadinessText
                )
            }
        }
    }

    private var modeSection: some View {
        NavPanel(
            title: "模式与可达性",
            subtitle: "保留标准桌面工作流，同时把盲人友好模式提升为独立而清晰的操作入口。"
        ) {
            Picker("交互模式", selection: $interactionMode) {
                ForEach(AppInteractionMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            HStack(alignment: .top, spacing: 12) {
                ForEach(AppInteractionMode.allCases) { mode in
                    ModeSummaryCard(mode: mode, isSelected: interactionMode == mode)
                }
            }
        }
    }

    private var requestSection: some View {
        NavPanel(
            title: "导航需求",
            subtitle: "先录入本次目的地需求，再确认规划。键盘和语音可以随时混合使用。",
            accessory: {
                NavStatusBadge(text: speechStatusText, tone: speechStatusTone, systemImage: speechStatusSymbol)
            }
        ) {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .top, spacing: 12) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("识别输出")
                            .font(.caption.weight(.semibold))
                        Text(outputMode.summary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    Picker("输出模式", selection: $outputMode) {
                        ForEach(RecognitionOutputMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 210)

                    Spacer()

                    Button(speech.isRunning && recordingTarget == .mainText ? "停止录音" : "开始录音") {
                        toggleMainRecording()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(candidateVM.isPlanning)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("导航需求文本")
                        .font(.caption.weight(.semibold))

                    TextEditor(text: $editableText)
                        .font(.body)
                        .scrollContentBackground(.hidden)
                        .frame(minHeight: 110, maxHeight: 150)
                        .padding(8)
                        .background(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(Color.navRaisedBackground)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .stroke(Color.navBorder.opacity(0.6), lineWidth: 1)
                        )
                }

                HStack(spacing: 10) {
                    Button("同步最新识别文本") {
                        editableText = speech.combinedText
                    }

                    Spacer()

                    Button(candidateVM.isPlanning ? "正在规划…" : "确认并规划") {
                        Task { await submitConfirmedText(editableText) }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(candidateVM.isPlanning || editableText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }

                if let latest = confirmedRecords.first {
                    NavTranscriptBubble(title: "最近一次确认", text: latest.text)
                }

                if let error = speech.lastError {
                    NavInfoBanner(text: error, tone: .danger)
                }

                NavInfoBanner(text: speech.authorizationSummary, tone: .neutral)
            }
        }
    }

    private var blindVoiceSection: some View {
        NavPanel(
            title: "盲人友好模式",
            subtitle: "用更大的主要操作、更清晰的阶段反馈和更完整的播报记录，让无障碍流程更可预测。",
            tone: .accent,
            accessory: {
                NavStatusBadge(text: blindPhase.shortTitle, tone: blindPhase.tone, systemImage: blindPhase.systemImage)
            }
        ) {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 12) {
                    NavMetricBadge(label: "当前阶段", value: blindPhase.shortTitle, emphasized: true)
                    NavMetricBadge(label: "候选结果", value: candidateVM.candidates.isEmpty ? "等待候选" : "已有 \(candidateVM.candidates.count) 项")
                    NavMetricBadge(label: "定位状态", value: mapVM.userCoordinate == nil ? "未获取位置" : "位置已就绪")
                }

                Button {
                    toggleBlindRecording()
                } label: {
                    Label(
                        speech.isRunning ? "结束录音" : "按按钮开始录音",
                        systemImage: speech.isRunning ? "stop.circle.fill" : "waveform.circle.fill"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(speech.isRunning ? .orange : Color.accentColor)
                .controlSize(.large)
                .disabled(candidateVM.isPlanning || voice.isSpeaking)

                HStack(spacing: 10) {
                    Button("停止播报") {
                        voice.stopSpeaking()
                        if blindPhase == .announcingCandidates || blindPhase == .announcingRoute {
                            blindPhase = .readyToRecord
                        }
                    }
                    .disabled(!voice.isSpeaking)

                    Button("请求定位权限") {
                        mapVM.requestLocationPermission()
                        if isVoiceOnlyMode {
                            blindSpeak("已请求定位权限。")
                        }
                    }

                    Button("定位当前位置") {
                        blindPhase = .waitingForLocation
                        mapVM.locateCurrentPosition()
                        if isVoiceOnlyMode {
                            blindSpeak("正在定位当前位置。")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }

                NavTranscriptBubble(title: "候选状态", text: candidateVM.statusText)
                NavTranscriptBubble(title: "定位状态", text: mapVM.statusText)

                if let pendingBlindNavigationRequest, !pendingBlindNavigationRequest.isEmpty {
                    NavInfoBanner(text: "已缓存导航需求：\(pendingBlindNavigationRequest)", tone: .accent)
                }

                if let planningError = candidateVM.planningError {
                    NavInfoBanner(text: planningError, tone: .danger)
                }

                if let speechError = speech.lastError {
                    NavInfoBanner(text: speechError, tone: .danger)
                }

                if !voice.lastSpokenText.isEmpty {
                    NavTranscriptBubble(title: "最近一次播报", text: voice.lastSpokenText)
                }

                NavInfoBanner(text: speech.authorizationSummary, tone: .neutral)
            }
        }
    }

    private var blindCommandSection: some View {
        NavPanel(
            title: "可用语音命令",
            subtitle: "关键命令按“定位、选项、重播、细化需求”分组，减少记忆负担。"
        ) {
            VStack(alignment: .leading, spacing: 10) {
                NavCommandHintRow(title: "开始导航", example: "带我去最近的地铁站")
                NavCommandHintRow(title: "选择候选", example: "选择第一个")
                NavCommandHintRow(title: "继续缩小范围", example: "更近一点，不要商场")
                NavCommandHintRow(title: "重播与恢复", example: "重播候选、重播路线、重新定位、退出盲人模式")
            }
        }
    }

    private var continueDialogSection: some View {
        NavPanel(
            title: "继续筛选",
            subtitle: "当候选已经出现后，用补充偏好快速缩小范围，而不是重新开始。"
        ) {
            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("补充约束")
                        .font(.caption.weight(.semibold))
                    TextField("例如：更近一点、不要商场、优先地铁站附近", text: $continueDialogInput)
                        .textFieldStyle(.roundedBorder)
                }

                HStack(spacing: 10) {
                    Button(speech.isRunning && recordingTarget == .continueDialog ? "停止对话录音" : "对话录音") {
                        toggleContinueDialogRecording()
                    }
                    .disabled(candidateVM.isPlanning)

                    Spacer()

                    Button(candidateVM.isPlanning ? "正在更新…" : "更新候选") {
                        let message = continueDialogInput
                        continueDialogInput = ""
                        Task {
                            await replanWithDialog(message)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(
                        candidateVM.isPlanning
                        || continueDialogInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        || candidateVM.candidates.isEmpty
                    )
                }

                if !candidateVM.dialogHistory.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("最近上下文")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        ForEach(Array(candidateVM.dialogHistory.suffix(4).enumerated()), id: \.offset) { _, message in
                            NavTranscriptBubble(title: "上下文记录", text: message)
                        }
                    }
                } else {
                    ContentUnavailableView(
                        "等待补充约束",
                        systemImage: "ellipsis.message",
                        description: Text("先完成一次候选规划，随后可以继续用自然语言补充条件。")
                    )
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 4)
                }
            }
        }
    }

    private var candidateSection: some View {
        NavPanel(
            title: "候选目的地",
            subtitle: "优先显示最符合当前需求的选项，并突出距离、耗时和匹配度。",
            accessory: {
                NavStatusBadge(text: candidateStatusText, tone: candidateStatusTone, systemImage: candidateStatusSymbol)
            }
        ) {
            VStack(alignment: .leading, spacing: 14) {
                NavTranscriptBubble(title: "规划状态", text: candidateVM.statusText)

                if let planningError = candidateVM.planningError {
                    NavInfoBanner(text: planningError, tone: .danger)
                }

                if candidateVM.candidates.isEmpty && !candidateVM.isPlanning {
                    ContentUnavailableView(
                        "尚未生成候选",
                        systemImage: "mappin.slash",
                        description: Text("先确认导航需求，系统会返回候选目的地与路线摘要。")
                    )
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                } else {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(Array(candidateVM.candidates.enumerated()), id: \.element.id) { index, candidate in
                            CandidateCard(
                                candidate: candidate,
                                rank: index + 1,
                                isSelected: candidateVM.selectedCandidate?.id == candidate.id,
                                isBusy: candidateVM.isPlanning
                            ) {
                                Task {
                                    await selectCandidate(candidate)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private var overallStatusText: String {
        if speech.isRunning {
            return recordingTarget == .continueDialog ? "对话录音中" : "录音中"
        }
        if voice.isSpeaking {
            return "正在播报"
        }
        if candidateVM.isPlanning {
            return "规划中"
        }
        return "已就绪"
    }

    private var overallStatusTone: NavSurfaceTone {
        if speech.isRunning {
            return .warning
        }
        if voice.isSpeaking {
            return .success
        }
        if candidateVM.isPlanning {
            return .accent
        }
        return .neutral
    }

    private var overallStatusSymbol: String {
        if speech.isRunning {
            return "mic.fill"
        }
        if voice.isSpeaking {
            return "speaker.wave.2.fill"
        }
        if candidateVM.isPlanning {
            return "arrow.trianglehead.2.clockwise"
        }
        return "checkmark.circle.fill"
    }

    private var speechStatusText: String {
        if speech.isRunning {
            return recordingTarget == .continueDialog ? "对话录音中" : "主需求录音中"
        }
        return "等待开始录音"
    }

    private var speechStatusTone: NavSurfaceTone {
        speech.isRunning ? .warning : .neutral
    }

    private var speechStatusSymbol: String {
        speech.isRunning ? "waveform" : "mic"
    }

    private var routeReadinessText: String {
        if mapVM.routeSummary != nil || mapVM.route != nil || mapVM.routeWaypoints.count >= 2 {
            return "路线已生成"
        }
        if mapVM.destinationCoordinate != nil {
            return "目的地已锁定"
        }
        return "等待规划"
    }

    private var candidateStatusText: String {
        if candidateVM.isPlanning {
            return "正在规划"
        }
        if candidateVM.candidates.isEmpty {
            return "等待候选"
        }
        return "\(candidateVM.candidates.count) 个候选"
    }

    private var candidateStatusTone: NavSurfaceTone {
        if candidateVM.planningError != nil {
            return .danger
        }
        if candidateVM.isPlanning {
            return .accent
        }
        return candidateVM.candidates.isEmpty ? .neutral : .success
    }

    private var candidateStatusSymbol: String {
        if candidateVM.planningError != nil {
            return "xmark.octagon.fill"
        }
        if candidateVM.isPlanning {
            return "arrow.trianglehead.2.clockwise"
        }
        return candidateVM.candidates.isEmpty ? "mappin.slash" : "mappin.and.ellipse"
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

private struct ModeSummaryCard: View {
    let mode: AppInteractionMode
    let isSelected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(mode.rawValue, systemImage: mode.systemImage)
                .font(.subheadline.weight(.semibold))
            Text(mode.summary)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(isSelected ? Color.accentColor.opacity(0.12) : Color.navRaisedBackground)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(isSelected ? Color.accentColor.opacity(0.35) : Color.navBorder.opacity(0.55), lineWidth: 1)
        )
    }
}

private struct CandidateCard: View {
    let candidate: AddressCandidate
    let rank: Int
    let isSelected: Bool
    let isBusy: Bool
    let onSelect: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        NavStatusBadge(
                            text: rank == 1 ? "最佳匹配" : "候选 \(rank)",
                            tone: rank == 1 ? .accent : .neutral,
                            systemImage: rank == 1 ? "sparkles" : "mappin.circle"
                        )
                        if isSelected {
                            NavStatusBadge(text: "当前路线", tone: .success, systemImage: "checkmark.circle.fill")
                        }
                    }

                    Text(candidate.title)
                        .font(.headline)

                    Label(candidate.displaySubtitle, systemImage: "mappin.and.ellipse")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if isSelected {
                    Button("已选中") {}
                        .buttonStyle(.bordered)
                        .disabled(true)
                } else {
                    Button("选择此项") {
                        onSelect()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isBusy)
                }
            }

            if !candidate.metricPairs.isEmpty {
                HStack(spacing: 8) {
                    ForEach(candidate.metricPairs, id: \.label) { pair in
                        NavMetricBadge(label: pair.label, value: pair.value)
                    }
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color.navRaisedBackground)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(isSelected ? Color.accentColor.opacity(0.35) : Color.navBorder.opacity(0.55), lineWidth: 1)
        )
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

    var shortTitle: String {
        switch self {
        case .inactive:
            return "未启用"
        case .readyToRecord:
            return "等待录音"
        case .waitingForLocation:
            return "等待定位"
        case .recording:
            return "正在录音"
        case .planning:
            return "正在规划"
        case .announcingCandidates:
            return "播报候选"
        case .announcingRoute:
            return "播报路线"
        }
    }

    var systemImage: String {
        switch self {
        case .inactive:
            return "eye.slash"
        case .readyToRecord:
            return "waveform.circle"
        case .waitingForLocation:
            return "location.circle"
        case .recording:
            return "mic.circle.fill"
        case .planning:
            return "arrow.trianglehead.2.clockwise.circle"
        case .announcingCandidates:
            return "list.bullet.circle"
        case .announcingRoute:
            return "point.topleft.down.curvedto.point.bottomright.up.circle"
        }
    }

    var tone: NavSurfaceTone {
        switch self {
        case .inactive:
            return .neutral
        case .readyToRecord:
            return .accent
        case .waitingForLocation, .recording:
            return .warning
        case .planning:
            return .accent
        case .announcingCandidates, .announcingRoute:
            return .success
        }
    }

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

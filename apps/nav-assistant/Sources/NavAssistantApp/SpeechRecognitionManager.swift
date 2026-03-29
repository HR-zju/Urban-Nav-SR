import Foundation
import AVFoundation
import Speech

@MainActor
final class SpeechRecognitionManager: ObservableObject {
    @Published var authorizationSummary: String = "未请求权限"
    @Published var isRunning = false
    @Published var committedText = ""
    @Published var liveText = ""
    @Published var lastError: String?

    private let audioEngine = AVAudioEngine()
    private var speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh_CN"))
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    private var hasRequiredUsageDescriptions: Bool {
        let info = Bundle.main.infoDictionary ?? [:]
        let speech = (info["NSSpeechRecognitionUsageDescription"] as? String)?.isEmpty == false
        let mic = (info["NSMicrophoneUsageDescription"] as? String)?.isEmpty == false
        return speech && mic
    }

    @discardableResult
    func requestPermissionsIfNeeded() async -> Bool {
        guard hasRequiredUsageDescriptions else {
            authorizationSummary = "当前运行方式缺少权限描述"
            lastError = "请使用 .app Bundle 启动，或在 Xcode 中运行 App target。"
            return false
        }

        let speechStatus = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }

        let micGranted = await AVCaptureDevice.requestAccess(for: .audio)

        switch (speechStatus, micGranted) {
        case (.authorized, true):
            authorizationSummary = "语音识别与麦克风权限已授权"
            lastError = nil
            return true
        case (.authorized, false):
            authorizationSummary = "缺少麦克风权限"
            lastError = "请在系统设置中开启麦克风权限"
            return false
        default:
            authorizationSummary = "缺少语音识别权限"
            lastError = "请在系统设置中开启语音识别权限"
            return false
        }
    }

    func startRecognition(mode: RecognitionOutputMode) {
        guard !isRunning else { return }
        guard hasRequiredUsageDescriptions else {
            lastError = "缺少权限描述，无法启动语音识别。请改为 .app Bundle 启动。"
            return
        }
        lastError = nil

        recognitionTask?.cancel()
        recognitionTask = nil

        request = SFSpeechAudioBufferRecognitionRequest()
        guard let request else {
            lastError = "无法创建语音识别请求"
            return
        }
        request.shouldReportPartialResults = true

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.removeTap(onBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            self?.request?.append(buffer)
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
        } catch {
            lastError = "启动录音失败: \(error.localizedDescription)"
            return
        }

        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }

            if let error {
                Task { @MainActor in
                    self.lastError = "识别失败: \(error.localizedDescription)"
                    self.stopRecognition()
                }
                return
            }

            guard let result else { return }

            Task { @MainActor in
                let spoken = result.bestTranscription.formattedString
                switch mode {
                case .streaming:
                    self.liveText = spoken
                case .sentence:
                    if result.isFinal {
                        if !self.committedText.isEmpty {
                            self.committedText += "\n"
                        }
                        self.committedText += spoken
                        self.liveText = ""
                    } else {
                        self.liveText = spoken
                    }
                }
            }
        }

        isRunning = true
    }

    func stopRecognition() {
        guard isRunning else { return }
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        request = nil
        isRunning = false
    }

    var combinedText: String {
        if liveText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return committedText
        }
        if committedText.isEmpty {
            return liveText
        }
        return "\(committedText)\n\(liveText)"
    }
}

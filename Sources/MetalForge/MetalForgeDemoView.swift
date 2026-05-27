#if canImport(SwiftUI)
import SwiftUI
import AVFoundation
import CoreVideo
@preconcurrency import Metal
@preconcurrency import CoreMedia

// ===========================================================================
// MetalForgeDemoController
//
// Owns engine + pipeline + view + capture manager + (optional) recorder.
// Glues them together with the correct concurrency boundaries:
//
//   captureQueue (background, AV delegate)
//       │  pixelBuffer + PTS  →  pipeline.process()  →  MTLTexture
//       │
//       ├──► DispatchQueue.main ──► MetalForgeView.present(texture:)   (display)
//       │
//       └──► recorder.appendVideoFrame(texture:, presentationTime:)    (record)
//
// Both consumers eventually fire `recycleHandler` once their GPU work
// completes. Since the same texture is shared by two consumers, naive
// pipeline.recycle on the first completion would race: the pool could re-hand
// the texture to the next pipeline.process call while the second consumer is
// still reading it. We solve this with a per-texture refcount keyed by
// ObjectIdentifier of the MTLTexture — both consumers call the same internal
// recycle handler; only the LAST one returns the texture to the pool.
// ===========================================================================

/// Glue controller that drives the demo SwiftUI view. Subclasses of
/// `ObservableObject` so SwiftUI re-renders when published state changes.
@MainActor
public final class MetalForgeDemoController: ObservableObject {

    // MARK: - Published state (SwiftUI-bound)

    @Published public private(set) var isRecording = false
    @Published public private(set) var permissionsGranted = false
    @Published public private(set) var lastRecordingURL: URL?
    @Published public private(set) var setupError: String?

    /// Which filter is currently driving the user-controllable parameter.
    /// Changing this re-routes the `filterIntensity` slider; all six filters
    /// stay in the pipeline simultaneously (inactive ones at zero intensity)
    /// so we don't have to mutate `pipeline.filters` from a non-pipeline thread.
    @Published public var activeFilter: FilterChoice = .glitch {
        didSet {
            // Temporal filters carry per-frame state that would "leak" old
            // imagery into the chain when re-selected. Clear their history on
            // every filter switch — but NOT on intensity changes, which is
            // why the clears live here (didSet of activeFilter) instead of
            // inside applyActiveFilter().
            if oldValue != activeFilter {
                motionBlurFilter.clearHistory()
                neonTrailsFilter.clearHistory()
            }
            applyActiveFilter()
        }
    }

    /// Single normalised intensity in `[0, 1]` controlling whichever filter is
    /// active. Each filter maps `[0, 1]` to its own natural range internally.
    @Published public var filterIntensity: Float = 0.5 {
        didSet { applyActiveFilter() }
    }

    // ----- ColorGrading-specific bindings -----
    // These are exposed because `filterIntensity` is a single value and the
    // ColorCorrection filter exposes three independent parameters (exposure,
    // contrast, saturation). LUT has its own intensity to keep the slider
    // label semantically distinct from the generic `filterIntensity`.
    @Published public var exposure:    Float = 0.0 { didSet { applyActiveFilter() } }
    @Published public var contrast:    Float = 1.0 { didSet { applyActiveFilter() } }
    @Published public var saturation:  Float = 1.0 { didSet { applyActiveFilter() } }
    @Published public var lutIntensity: Float = 1.0 { didSet { applyActiveFilter() } }

    public enum FilterChoice: String, CaseIterable, Identifiable, Sendable {
        case glitch          = "Glitch"
        case chromatic       = "Chroma"
        case noise           = "Noise"
        case jitter          = "Jitter"
        case motionBlur      = "Motion"
        case neonTrails      = "Trails"
        case lutGrading      = "LUT"
        case colorCorrection = "Color"
        public var id: String { rawValue }
    }

    // MARK: - MetalForge components

    public let engine: MetalForgeEngine
    public let view: MetalForgeView
    private let pipeline: MetalForgePipeline
    private let captureManager: MetalForgeCaptureManager

    // All four filters live permanently in the pipeline chain. The active one
    // is driven by `filterIntensity`; the others have intensity-equivalents
    // pinned to zero so they are functional no-ops. This avoids any need to
    // mutate `pipeline.filters` concurrently with frame processing.
    private let glitchFilter:          GlitchFilter
    private let chromaticFilter:       ChromaticAberrationFilter
    private let noiseFilter:           AnalogNoiseFilter
    private let jitterFilter:          HorizontalJitterFilter
    private let motionBlurFilter:      MotionBlurFilter
    private let neonTrailsFilter:      NeonTrailsFilter
    private let lutFilter:             MetalForgeLUTFilter
    private let colorCorrectionFilter: ColorCorrectionFilter

    // Reference to the active recorder (nil when not recording). Protected by
    // `stateLock` because handleVideoFrame on captureQueue needs to read it.
    private var recorder: MetalForgeRecorder?
    private var currentRecordingURL: URL?

    // MARK: - Cross-thread state

    /// Lock guarding `_recorderSnapshot` and `_pendingCount`. Held very briefly
    /// (microseconds) — never blocks GPU or capture.
    private let stateLock = NSLock()
    /// Snapshot of `recorder` readable from `captureQueue` without hopping to
    /// main. Written from the main actor in start/stop.
    /// `nonisolated(unsafe)` because we *do* protect it with `stateLock` — the
    /// compiler just can't prove the invariant. Every access in this file is
    /// inside `stateLock.lock()/unlock()` or `stateLock.withLock`.
    nonisolated(unsafe) private var _recorderSnapshot: MetalForgeRecorder?
    /// Refcount of in-flight consumers per texture, keyed by identity.
    /// Initialised at dispatch time; decremented in `handleRecycle`.
    /// Same `nonisolated(unsafe)` rationale: lock-protected.
    nonisolated(unsafe) private var _pendingCount: [ObjectIdentifier: Int] = [:]

    // MARK: - Init

    public init() {
        // For a demo, init failure → fatal. Production apps should surface this
        // as an error UI rather than crashing.
        do {
            self.engine                = try MetalForgeEngine()
            self.view                  = try MetalForgeView(engine: engine)
            self.pipeline              = try MetalForgePipeline(engine: engine)
            self.glitchFilter          = try GlitchFilter(engine: engine)
            self.chromaticFilter       = try ChromaticAberrationFilter(engine: engine)
            self.noiseFilter           = try AnalogNoiseFilter(engine: engine)
            self.jitterFilter          = try HorizontalJitterFilter(engine: engine)
            self.motionBlurFilter      = try MotionBlurFilter(engine: engine)
            self.neonTrailsFilter      = try NeonTrailsFilter(engine: engine)
            // The 32³ warm preset is generated on the fly — no external assets.
            self.lutFilter             = try MetalForgeLUTFilter(engine: engine, preset: .warm, size: 32)
            self.colorCorrectionFilter = try ColorCorrectionFilter(engine: engine)
        } catch {
            fatalError("MetalForge core initialisation failed: \(error)")
        }
        self.captureManager = MetalForgeCaptureManager()

        // All eight filters live in the chain. Inactive ones are pinned to
        // their no-op state by `applyActiveFilter` (intensity = 0, contrast = 1,
        // etc.). Temporal filters' history buffers are kept fresh by
        // `clearHistory()` on every activeFilter switch.
        pipeline.append(glitchFilter)
        pipeline.append(chromaticFilter)
        pipeline.append(noiseFilter)
        pipeline.append(jitterFilter)
        pipeline.append(motionBlurFilter)
        pipeline.append(neonTrailsFilter)
        pipeline.append(lutFilter)
        pipeline.append(colorCorrectionFilter)

        applyActiveFilter()   // sets initial active filter intensity, others zero

        // ----- Wire the display recycle path -----
        // View calls this on Metal's completion thread after each draw.
        view.recycleHandler = { [weak self] texture in
            self?.handleRecycle(texture: texture)
        }
    }

    // MARK: - Filter routing

    /// Apply the current `activeFilter` and `filterIntensity` to the four
    /// pipeline filters. The active one gets the normalised intensity mapped
    /// to its natural range; all others are pinned to their zero state.
    private func applyActiveFilter() {
        // ----- Reset all to no-op state -----
        glitchFilter.intensity             = 0
        chromaticFilter.redShift           = SIMD2(0, 0)
        chromaticFilter.greenShift         = SIMD2(0, 0)
        noiseFilter.noiseIntensity         = 0
        jitterFilter.jitterIntensity       = 0
        // MotionBlur: alpha = 1.0 means "pass through current frame unchanged".
        motionBlurFilter.accumulationAlpha = 1.0
        // NeonTrails: intensity 0 → no new glow, but `decay` could still keep
        // an old trail alive on the destination. We don't touch decay here
        // because clearHistory() already dropped the buffer on filter switch.
        neonTrailsFilter.intensity         = 0
        // ColorGrading no-ops: LUT intensity 0 ≡ pass-through (mix yields base);
        // colorCorrection at identity (exposure 0, contrast 1, saturation 1)
        // produces source == output exactly.
        lutFilter.intensity                = 0
        colorCorrectionFilter.exposure     = 0
        colorCorrectionFilter.contrast     = 1
        colorCorrectionFilter.saturation   = 1
        colorCorrectionFilter.temperatureShift = 0

        // ----- Drive the active one -----
        switch activeFilter {
        case .glitch:
            glitchFilter.intensity = filterIntensity

        case .chromatic:
            // Map slider [0, 1] → symmetric ±2 % horizontal shifts.
            let shift = filterIntensity * 0.02
            chromaticFilter.redShift   = SIMD2( shift, 0)
            chromaticFilter.greenShift = SIMD2(-shift, 0)

        case .noise:
            // Up to ±0.25 luminance grain at full intensity.
            noiseFilter.noiseIntensity = filterIntensity * 0.5

        case .jitter:
            // Up to ±5 % horizontal row-shift at full intensity.
            jitterFilter.jitterIntensity = filterIntensity * 0.05

        case .motionBlur:
            // Slider 0 → alpha = 1.0 (sharp), Slider 1 → alpha = 0.1 (heavy blur).
            // We avoid alpha = 0 (full freeze) to keep the preview responsive
            // even at the slider extreme — there's always at least 10 % new
            // signal mixed in.
            motionBlurFilter.accumulationAlpha = 1.0 - filterIntensity * 0.9

        case .neonTrails:
            // Slider drives glow intensity. Trail decay is fixed at 0.9 for
            // a noticeable but not overwhelming persistence; tweakable in code.
            neonTrailsFilter.intensity = filterIntensity * 1.5
            neonTrailsFilter.decay     = 0.9

        case .lutGrading:
            // LUT has its own dedicated intensity binding (not shared with
            // the generic `filterIntensity` slider).
            lutFilter.intensity = lutIntensity

        case .colorCorrection:
            // Three independent parameters from their dedicated bindings.
            colorCorrectionFilter.exposure   = exposure
            colorCorrectionFilter.contrast   = contrast
            colorCorrectionFilter.saturation = saturation
        }
    }

    // MARK: - Setup (called once on view appear)

    public func setup() async {
        let cameraOK = await MetalForgeCaptureManager.requestCameraAccess()
        let micOK    = await MetalForgeCaptureManager.requestMicrophoneAccess()
        permissionsGranted = cameraOK   // mic optional — recording works without audio
        if !micOK {
            // Just note it; we still proceed with video-only capture.
            print("MetalForgeDemo: microphone access denied — recordings will be silent.")
        }
        guard cameraOK else {
            setupError = "Camera access required."
            return
        }

        do {
            try captureManager.configure(position: .back, preferHDR: false)
        } catch {
            setupError = "Camera configure failed: \(error.localizedDescription)"
            return
        }

        // ----- Wire capture callbacks (fire on captureQueue) -----
        captureManager.onVideoFrame = { [weak self] pixelBuffer, pts in
            self?.handleVideoFrame(pixelBuffer: pixelBuffer, pts: pts)
        }
        captureManager.onAudioSample = { [weak self] sampleBuffer in
            self?.handleAudioSample(sampleBuffer)
        }

        captureManager.startCapture()
    }

    // MARK: - Frame handlers (nonisolated — run on captureQueue)

    nonisolated func handleVideoFrame(pixelBuffer: CVPixelBuffer, pts: CMTime) {
        // Heavy autoreleased traffic from CV/AV/Metal — pool drain per frame
        // keeps memory flat on a long-running capture queue with no runloop.
        autoreleasepool {
            guard let texture = pipeline.process(pixelBuffer: pixelBuffer) else { return }

            // Atomic snapshot: decide consumer set AND ref count under the same
            // lock so we never under-count or over-count for the same texture.
            stateLock.lock()
            let activeRecorder = _recorderSnapshot
            let willRecord     = (activeRecorder?.state == .recording)
            let refs           = willRecord ? 2 : 1
            _pendingCount[ObjectIdentifier(texture as AnyObject)] = refs
            stateLock.unlock()

            // ----- Display path -----
            // MetalForgeView is @MainActor — hop to main for present(texture:).
            // It internally schedules an MTKView draw which triggers our
            // recycleHandler when the GPU completes.
            DispatchQueue.main.async { [weak self] in
                self?.view.present(texture: texture)
            }

            // ----- Recording path (background) -----
            // MetalForgeRecorder.appendVideoFrame is safe from any thread; it
            // dispatches to its own videoQueue and always recycles via
            // recycleHandler (which we wired in startRecording).
            if willRecord, let activeRecorder {
                activeRecorder.appendVideoFrame(texture: texture, presentationTime: pts)
            }
        }
    }

    nonisolated func handleAudioSample(_ sampleBuffer: CMSampleBuffer) {
        stateLock.lock()
        let activeRecorder = _recorderSnapshot
        stateLock.unlock()
        activeRecorder?.appendAudioSample(sampleBuffer)
    }

    /// Refcount-aware texture recycle. Both `view.recycleHandler` and
    /// `recorder.recycleHandler` are pointed at this single closure. The
    /// actual pool recycle only happens when the LAST consumer finishes.
    nonisolated func handleRecycle(texture: MTLTexture) {
        let key = ObjectIdentifier(texture as AnyObject)
        stateLock.lock()
        let current = _pendingCount[key] ?? 1
        let next    = current - 1
        let done    = next <= 0
        if done {
            _pendingCount.removeValue(forKey: key)
        } else {
            _pendingCount[key] = next
        }
        stateLock.unlock()

        if done {
            // pipeline.recycle delegates to TexturePool, which is internally
            // thread-safe — fine to call from any consumer's completion thread.
            pipeline.recycle(texture)
        }
    }

    // MARK: - Recording control

    public func toggleRecording() {
        if isRecording {
            Task { await stopRecording() }
        } else {
            startRecording()
        }
    }

    private func startRecording() {
        let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let url = docsDir.appendingPathComponent("MetalForge-\(Int(Date().timeIntervalSince1970)).mp4")

        let videoSize = captureManager.videoDimensions
        guard videoSize.width > 0, videoSize.height > 0 else {
            setupError = "Camera not running yet."
            return
        }

        do {
            let r = try MetalForgeRecorder(
                engine: engine,
                videoSize: videoSize,
                workingColorSpace: captureManager.workingColorSpace,
                frameRate: 30
            )
            // Same recycle handler as the view → unified refcount path.
            r.recycleHandler = { [weak self] texture in
                self?.handleRecycle(texture: texture)
            }
            r.errorHandler = { error in
                print("MetalForgeDemo: recorder error \(error)")
            }
            try r.startRecording(outputURL: url)

            // Publish to main-actor state and to the capture-queue snapshot.
            recorder            = r
            currentRecordingURL = url
            stateLock.withLock { _recorderSnapshot = r }
            isRecording = true
        } catch {
            setupError = "startRecording failed: \(error.localizedDescription)"
        }
    }

    private func stopRecording() async {
        guard let r = recorder else { return }
        let savedURL = currentRecordingURL

        // Clear the capture-queue snapshot first so no new frames are routed to
        // the recorder while finalisation is in flight.
        stateLock.withLock { _recorderSnapshot = nil }

        do {
            try await r.stopRecording()
            lastRecordingURL = savedURL
        } catch {
            print("MetalForgeDemo: stopRecording failed: \(error)")
        }
        recorder            = nil
        currentRecordingURL = nil
        isRecording         = false
    }
}

// ===========================================================================
// SwiftUI view
// ===========================================================================

/// Drop-in demo view. Embed in an iOS / macOS app target that declares the
/// required `Info.plist` keys:
///
///   - `NSCameraUsageDescription`     ("Used to capture video.")
///   - `NSMicrophoneUsageDescription` ("Used to capture audio for recording.")
public struct MetalForgeDemoView: View {

    @StateObject private var controller = MetalForgeDemoController()

    public init() {}

    public var body: some View {
        ZStack {
            // ----- Camera preview / fallback -----
            if controller.permissionsGranted {
                MetalForgeViewRepresentable(view: controller.view)
                    .ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
                VStack(spacing: 12) {
                    Image(systemName: "video.slash")
                        .font(.system(size: 56))
                        .foregroundStyle(.white.opacity(0.6))
                    Text("Camera access required")
                        .foregroundStyle(.white)
                }
            }

            // ----- Controls overlay -----
            VStack {
                Spacer()
                controlPanel
            }

            // ----- Toasts -----
            if let url = controller.lastRecordingURL {
                VStack {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                        Text("Saved: \(url.lastPathComponent)")
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    .font(.caption)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    Spacer()
                }
                .padding(.top, 50)
            }

            if let err = controller.setupError {
                VStack {
                    Spacer()
                    Text(err)
                        .font(.caption)
                        .foregroundStyle(.white)
                        .padding(10)
                        .background(.red.opacity(0.8), in: RoundedRectangle(cornerRadius: 8))
                        .padding(.bottom, 200)
                }
            }
        }
        .task { await controller.setup() }
    }

    private var controlPanel: some View {
        VStack(spacing: 20) {
            // ----- Filter picker -----
            // Segmented style for at-a-glance switching. All four filters are
            // live in the pipeline simultaneously; the picker just routes the
            // intensity slider below to whichever one is selected.
            Picker("Filter", selection: $controller.activeFilter) {
                ForEach(MetalForgeDemoController.FilterChoice.allCases) { choice in
                    Text(choice.rawValue).tag(choice)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)

            // ----- Active filter controls -----
            // ColorCorrection needs three sliders (exposure / contrast / sat);
            // LUT has its own intensity binding; all other filters share the
            // generic `filterIntensity`.
            filterControls
                .padding(.horizontal)

            // Record button
            Button {
                controller.toggleRecording()
            } label: {
                ZStack {
                    Circle()
                        .stroke(.white, lineWidth: 4)
                        .frame(width: 72, height: 72)
                    if controller.isRecording {
                        RoundedRectangle(cornerRadius: 6)
                            .fill(.red)
                            .frame(width: 30, height: 30)
                    } else {
                        Circle()
                            .fill(.red)
                            .frame(width: 56, height: 56)
                    }
                }
            }
            .buttonStyle(.plain)
            .accessibilityLabel(controller.isRecording ? "Stop recording" : "Start recording")
        }
        .padding(.vertical, 16)
        .background(.ultraThinMaterial)
    }

    /// SF Symbol that visually hints at the active filter type.
    private func iconName(for filter: MetalForgeDemoController.FilterChoice) -> String {
        switch filter {
        case .glitch:          return "waveform.path"
        case .chromatic:       return "circle.lefthalf.filled.righthalf.striped.horizontal"
        case .noise:           return "dot.radiowaves.left.and.right"
        case .jitter:          return "arrow.left.arrow.right"
        case .motionBlur:      return "wind"
        case .neonTrails:      return "sparkles"
        case .lutGrading:      return "paintpalette"
        case .colorCorrection: return "slider.horizontal.3"
        }
    }

    /// Controls strip that adapts to the active filter. Most filters expose a
    /// single normalised intensity; LUT has its own intensity binding; the
    /// color-correction filter spreads across three independent sliders.
    @ViewBuilder
    private var filterControls: some View {
        switch controller.activeFilter {
        case .colorCorrection:
            VStack(spacing: 10) {
                LabelledSlider(
                    label: "Exposure",
                    iconName: "sun.max",
                    value: $controller.exposure,
                    range: -2...2,
                    format: "%+.1f EV",
                    displayMultiplier: 1
                )
                LabelledSlider(
                    label: "Contrast",
                    iconName: "circle.righthalf.filled",
                    value: $controller.contrast,
                    range: 0.5...2,
                    format: "%.2f×",
                    displayMultiplier: 1
                )
                LabelledSlider(
                    label: "Saturation",
                    iconName: "drop.fill",
                    value: $controller.saturation,
                    range: 0...2,
                    format: "%.0f%%",
                    displayMultiplier: 100
                )
            }

        case .lutGrading:
            LabelledSlider(
                label: "LUT Intensity",
                iconName: iconName(for: .lutGrading),
                value: $controller.lutIntensity,
                range: 0...1,
                format: "%.0f%%",
                displayMultiplier: 100
            )

        default:
            LabelledSlider(
                label: "\(controller.activeFilter.rawValue) intensity",
                iconName: iconName(for: controller.activeFilter),
                value: $controller.filterIntensity,
                range: 0...1,
                format: "%.0f%%",
                displayMultiplier: 100
            )
        }
    }
}

// MARK: - Reusable labelled slider

/// Single-line slider with leading icon, label, and trailing formatted value.
/// Used throughout `filterControls` so the visual cadence stays consistent
/// no matter how many parameters the active filter exposes.
private struct LabelledSlider: View {
    let label: String
    let iconName: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let format: String
    /// Multiplier applied to the *displayed* value (e.g. ×100 to show a
    /// `0…1` binding as a percentage). The underlying `value` binding is
    /// untouched — the multiplier is presentation only.
    let displayMultiplier: Float

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: iconName)
                    .foregroundStyle(.white)
                    .imageScale(.small)
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.white)
                Spacer()
                Text(String(format: format, value * displayMultiplier))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.white.opacity(0.7))
            }
            Slider(value: $value, in: range)
                .tint(.white)
        }
    }
}

// MARK: - NSLock async-safe withLock helper (only on Swift toolchains that
// don't synthesise it). The MetalForgeRecorder already uses NSLock.withLock,
// so on the supported targets this is already available — no helper needed.

#endif // canImport(SwiftUI)

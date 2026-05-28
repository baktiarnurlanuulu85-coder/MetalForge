import SwiftUI
import AVFoundation
import CoreVideo
import CoreMedia
@preconcurrency import Metal
import MetalForge

// MARK: - Filter choices exposed to the UI

/// The set of demo filters the example app exposes.
///
/// All filters live in the `MetalForgePipeline` simultaneously; switching the
/// active choice just re-targets the inactive ones to their no-op state. This
/// is the same race-free pattern used by `MetalForgeDemoController` in the
/// library, scaled down to four demo entries.
enum FilterChoice: String, CaseIterable, Identifiable, Sendable {
    case original     = "Original"
    case warm         = "Warm"
    case cool         = "Cool"
    case highContrast = "High Contrast"

    var id: String { rawValue }

    /// SF Symbol used in the segmented picker.
    var iconName: String {
        switch self {
        case .original:     return "circle"
        case .warm:         return "sun.max.fill"
        case .cool:         return "snowflake"
        case .highContrast: return "circle.righthalf.filled"
        }
    }

    /// Whether the intensity slider has any effect on this choice.
    var supportsIntensity: Bool {
        self != .original
    }
}

// MARK: - View model

/// SwiftUI-bound controller that owns the MetalForge engine + pipeline and
/// drives the camera preview.
///
/// Threading:
/// - `@MainActor` for `@Published` state and UI access.
/// - Frames arrive on `MetalForgeCaptureManager.captureQueue` (background) and
///   are forwarded into `MetalForgePipeline.process(pixelBuffer:)` directly,
///   without hopping to the main actor (the pipeline does its own GPU sync).
/// - `view.present(texture:)` runs back on the main actor.
@MainActor
final class CameraViewModel: ObservableObject {

    // MARK: Published UI state

    @Published var permissionsGranted = false
    @Published var setupError: String?

    /// Currently selected demo filter.
    @Published var activeFilter: FilterChoice = .original {
        didSet { applyActiveFilter() }
    }

    /// Effect strength in `[0, 1]`. Mapped to each filter's own natural range
    /// inside `applyActiveFilter()`.
    @Published var filterIntensity: Float = 1.0 {
        didSet { applyActiveFilter() }
    }

    /// When `true`, the entire processed look is bypassed and the pipeline
    /// renders the raw camera signal (all filters set to identity).
    @Published var showOriginal: Bool = false {
        didSet { applyActiveFilter() }
    }

    /// Frames per second over a rolling 1-second window. Updated on the main
    /// actor from the capture callback.
    @Published private(set) var fps: Double = 0

    // MARK: MetalForge components

    /// Exposed so `CameraPreviewView` can hand the underlying `MetalForgeView`
    /// to SwiftUI through `MetalForgeViewRepresentable`.
    let engine: MetalForgeEngine
    let view: MetalForgeView

    private let pipeline: MetalForgePipeline
    private let capture: MetalForgeCaptureManager

    // Permanent filter chain — see the `MetalForgeDemoController` pattern.
    private let warmLUT:         MetalForgeLUTFilter
    private let coolLUT:         MetalForgeLUTFilter
    private let colorCorrection: ColorCorrectionFilter

    // MARK: FPS bookkeeping

    private var fpsFrameCount: Int = 0
    private var fpsWindowStart: CFTimeInterval = CACurrentMediaTime()

    // MARK: Init

    init() {
        // Init failure is fatal for the demo. A production app should surface
        // this through a proper error UI.
        do {
            let engine = try MetalForgeEngine()
            self.engine = engine
            self.view = try MetalForgeView(engine: engine)
            self.pipeline = try MetalForgePipeline(engine: engine)

            self.warmLUT = try MetalForgeLUTFilter(engine: engine, preset: .warm, size: 32)
            self.coolLUT = try MetalForgeLUTFilter(engine: engine, preset: .cool, size: 32)
            self.colorCorrection = try ColorCorrectionFilter(engine: engine)
        } catch {
            fatalError("MetalForge initialisation failed: \(error)")
        }

        self.capture = MetalForgeCaptureManager()

        // All three filters live permanently in the chain. We just rewrite
        // their parameters between frames depending on `activeFilter`.
        pipeline.append(warmLUT)
        pipeline.append(coolLUT)
        pipeline.append(colorCorrection)

        applyActiveFilter()

        // Recycle each presented texture back into the pool after the GPU
        // finishes the draw. Capturing `pipeline` weakly avoids any retain
        // cycle through the closure.
        view.recycleHandler = { [weak pipeline] texture in
            pipeline?.recycle(texture)
        }
    }

    // MARK: Setup

    /// Request camera permission and start the capture session.
    /// Call from a SwiftUI `.task { ... }`.
    func setup() async {
        let granted = await MetalForgeCaptureManager.requestCameraAccess()
        permissionsGranted = granted
        guard granted else {
            setupError = "Camera access denied. Enable it in Settings to use the demo."
            return
        }

        do {
            // The demo runs in SDR — keeps the example simple. The library
            // supports HDR via `preferHDR: true`; see MetalForge README.
            try capture.configure(position: .back, preferHDR: false)
        } catch {
            setupError = "Camera configure failed: \(error.localizedDescription)"
            return
        }

        capture.onVideoFrame = { [weak self] pixelBuffer, _ in
            self?.handleVideoFrame(pixelBuffer)
        }
        // No recording in this initial example, so no onAudioSample handler.

        capture.startCapture()
    }

    // MARK: Frame ingest (captureQueue)

    /// Called on `MetalForgeCaptureManager`'s background capture queue.
    nonisolated private func handleVideoFrame(_ pixelBuffer: CVPixelBuffer) {
        // The pipeline drives its own GPU sync internally; we can call it
        // straight from the capture queue.
        guard let texture = pipeline.process(pixelBuffer: pixelBuffer) else { return }

        // Hop to the main actor to present + bump FPS. The actor isolation
        // means the view's `present(texture:)` call is safe.
        Task { @MainActor [weak self] in
            self?.present(texture: texture)
        }
    }

    @MainActor
    private func present(texture: MTLTexture) {
        view.present(texture: texture)
        tickFPS()
    }

    private func tickFPS() {
        fpsFrameCount += 1
        let now = CACurrentMediaTime()
        let elapsed = now - fpsWindowStart
        if elapsed >= 1.0 {
            fps = Double(fpsFrameCount) / elapsed
            fpsFrameCount = 0
            fpsWindowStart = now
        }
    }

    // MARK: Filter routing

    /// Configure every filter in the pipeline for the current `activeFilter`
    /// + `filterIntensity`. Inactive filters are pinned to their identity
    /// state so they encode but make no visible change.
    private func applyActiveFilter() {
        // Identity defaults for everything.
        warmLUT.intensity = 0
        coolLUT.intensity = 0
        colorCorrection.exposure = 0
        colorCorrection.contrast = 1
        colorCorrection.saturation = 1
        colorCorrection.temperatureShift = 0

        guard !showOriginal else { return }

        switch activeFilter {
        case .original:
            // Everything stays at identity — pure camera signal.
            break

        case .warm:
            warmLUT.intensity = filterIntensity

        case .cool:
            coolLUT.intensity = filterIntensity

        case .highContrast:
            // Map slider [0, 1] to a perceptually useful contrast range.
            // 1.0 = identity (slider 0), 2.0 = strong punch (slider 1).
            colorCorrection.contrast = 1.0 + filterIntensity
            // Slight saturation bump rides along with contrast for a more
            // cinematic look. Pure contrast on its own often feels flat.
            colorCorrection.saturation = 1.0 + 0.25 * filterIntensity
        }
    }
}

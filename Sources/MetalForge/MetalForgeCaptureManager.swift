import Foundation
import AVFoundation
import CoreVideo
@preconcurrency import CoreMedia

/// Wraps `AVCaptureSession` to deliver video frames (as `CVPixelBuffer`) and
/// audio sample buffers (as `CMSampleBuffer`) on a dedicated background queue,
/// ready for `MetalForgePipeline` and `MetalForgeRecorder` to consume.
///
/// ## Camera Selection
/// - **iOS / visionOS**: `.builtInWideAngleCamera` on the requested position.
/// - **macOS**: the system default video device.
///
/// ## HDR
/// If `preferHDR` is true on iOS, the manager scans the device's `formats` for
/// one supporting `isVideoHDRSupported` and switches to it. The output pixel
/// format then becomes `kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange` —
/// exactly what `MetalForgePipeline` auto-routes through the HDR path. If HDR
/// isn't available, the manager falls back to 8-bit 4:2:0 YUV (BT.709).
///
/// ## Concurrency
/// `captureOutput(_:didOutput:from:)` fires on the dedicated `captureQueue`
/// (serial, `userInitiated` QoS). All `onVideoFrame` / `onAudioSample` handler
/// invocations happen on that same queue — handlers must be safe to call from
/// a non-main thread. The pipeline's `process(pixelBuffer:)` is fine because
/// it does its own GPU sync internally (`waitUntilCompleted`).
public final class MetalForgeCaptureManager: NSObject, @unchecked Sendable {

    public enum ConfigurationError: Error, LocalizedError {
        case noVideoDevice
        case cannotAddInput(String)
        case cannotAddOutput(String)
        case deviceConfigurationFailed(String)

        public var errorDescription: String? {
            switch self {
            case .noVideoDevice:               return "No video capture device available."
            case .cannotAddInput(let media):   return "Capture session refused the \(media) input."
            case .cannotAddOutput(let media):  return "Capture session refused the \(media) output."
            case .deviceConfigurationFailed(let d): return "Device configuration failed: \(d)"
            }
        }
    }

    // MARK: - Public state

    /// Underlying capture session — exposed so callers can inspect / extend it
    /// (additional outputs, preview layers, etc.). Mutating session topology
    /// outside `configure(...)` should be wrapped in `beginConfiguration` /
    /// `commitConfiguration` on the caller's side.
    public let session = AVCaptureSession()

    /// The pixel format the camera was configured to emit. Useful for
    /// downstream code that needs to derive `workingColorSpace`.
    public private(set) var sourcePixelFormat: OSType = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange

    /// Native dimensions of the active format (post-`configure`). Use this to
    /// size the `MetalForgeRecorder`.
    public private(set) var videoDimensions: CGSize = .zero

    /// Resolved working colour space based on the selected pixel format —
    /// SDR for 8-bit YUV, HDR10/PQ for 10-bit YUV.
    public var workingColorSpace: MetalForgeColorSpace {
        switch sourcePixelFormat {
        case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
             kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
            return .hdr10PQ
        default:
            return .sdr
        }
    }

    // MARK: - Callbacks

    /// Fires on `captureQueue` for every video frame. Pixel buffer + PTS.
    public var onVideoFrame: (@Sendable (CVPixelBuffer, CMTime) -> Void)?

    /// Fires on `captureQueue` for every audio sample. Passthrough to recorder.
    public var onAudioSample: (@Sendable (CMSampleBuffer) -> Void)?

    // MARK: - Private

    private let captureQueue = DispatchQueue(
        label: "com.metalforge.capture",
        qos: .userInitiated
    )
    private let videoOutput = AVCaptureVideoDataOutput()
    private let audioOutput = AVCaptureAudioDataOutput()

    // MARK: - Permissions

    /// `await`-friendly camera permission request. Returns `true` if access was
    /// previously granted or the user grants it in response to the prompt.
    public static func requestCameraAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:           return true
        case .denied, .restricted:  return false
        case .notDetermined:        return await AVCaptureDevice.requestAccess(for: .video)
        @unknown default:           return false
        }
    }

    public static func requestMicrophoneAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:           return true
        case .denied, .restricted:  return false
        case .notDetermined:        return await AVCaptureDevice.requestAccess(for: .audio)
        @unknown default:           return false
        }
    }

    // MARK: - Configuration

    /// Build the capture topology: video input + Metal-compatible video output,
    /// audio input + audio output (if a microphone is available). Idempotent
    /// per session — call once after permission is granted.
    public func configure(
        position: AVCaptureDevice.Position = .back,
        preferHDR: Bool = false
    ) throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .high

        // ---------- Video input ----------
        #if os(macOS)
        guard let videoDevice = AVCaptureDevice.default(for: .video) else {
            throw ConfigurationError.noVideoDevice
        }
        #else
        guard let videoDevice = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: position
        ) else {
            throw ConfigurationError.noVideoDevice
        }
        #endif

        #if os(iOS) || os(visionOS)
        if preferHDR {
            // Best-effort HDR switch; failures fall back to SDR silently.
            try? configureHDRFormat(on: videoDevice)
        }
        #endif

        let videoInput = try AVCaptureDeviceInput(device: videoDevice)
        guard session.canAddInput(videoInput) else {
            throw ConfigurationError.cannotAddInput("video")
        }
        session.addInput(videoInput)

        // ---------- Decide pixel format ----------
        // We picked the active format above; now derive the matching CV pixel
        // format. The 10-bit format requires HDR-capable hardware (iPhone 12+).
        let pixelFormat: OSType
        #if os(iOS) || os(visionOS)
        if preferHDR && videoDevice.activeFormat.isVideoHDRSupported && videoDevice.isVideoHDREnabled {
            pixelFormat = kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        } else {
            pixelFormat = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        }
        #else
        pixelFormat = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        #endif
        sourcePixelFormat = pixelFormat

        // ---------- Video output ----------
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey   as String: pixelFormat,
            // Metal compatibility unlocks the zero-copy CVMetalTextureCache path
            // for `engine.makeTextures(from:colorSpace:)`.
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        // Drop frames the captureQueue can't drain fast enough — better visible
        // stutter than rising A/V drift.
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: captureQueue)
        guard session.canAddOutput(videoOutput) else {
            throw ConfigurationError.cannotAddOutput("video")
        }
        session.addOutput(videoOutput)

        // ---------- Orientation ----------
        if let connection = videoOutput.connection(with: .video) {
            #if os(iOS) || os(visionOS)
            // Portrait — most demos run on phones held vertically.
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
            #endif
        }

        // ---------- Audio input + output (optional) ----------
        if let audioDevice = AVCaptureDevice.default(for: .audio),
           let audioInput = try? AVCaptureDeviceInput(device: audioDevice),
           session.canAddInput(audioInput)
        {
            session.addInput(audioInput)

            audioOutput.setSampleBufferDelegate(self, queue: captureQueue)
            if session.canAddOutput(audioOutput) {
                session.addOutput(audioOutput)
            }
        }

        // ---------- Cached dimensions ----------
        let dims = CMVideoFormatDescriptionGetDimensions(videoDevice.activeFormat.formatDescription)
        videoDimensions = CGSize(width: CGFloat(dims.width), height: CGFloat(dims.height))
    }

    #if os(iOS) || os(visionOS)
    /// Find the highest-resolution HDR-capable format and activate it on the
    /// device, enabling 10-bit HDR signalling. Silent fallback to default
    /// format on any failure.
    private func configureHDRFormat(on device: AVCaptureDevice) throws {
        let hdrFormats = device.formats.filter { $0.isVideoHDRSupported }
        let bestHDR = hdrFormats.max { a, b in
            let da = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
            let db = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
            return Int(da.width) * Int(da.height) < Int(db.width) * Int(db.height)
        }
        guard let format = bestHDR else { return }

        do {
            try device.lockForConfiguration()
            device.activeFormat = format
            // Take manual control so the system doesn't toggle HDR off.
            device.automaticallyAdjustsVideoHDREnabled = false
            device.isVideoHDREnabled = true
            device.unlockForConfiguration()
        } catch {
            throw ConfigurationError.deviceConfigurationFailed(error.localizedDescription)
        }
    }
    #endif

    // MARK: - Run control

    public func startCapture() {
        guard !session.isRunning else { return }
        // session.startRunning() is a blocking call (sometimes hundreds of ms);
        // dispatch to a global queue so the caller's queue (often main) stays
        // responsive.
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
        }
    }

    public func stopCapture() {
        guard session.isRunning else { return }
        session.stopRunning()
    }
}

// MARK: - Sample buffer delegate

extension MetalForgeCaptureManager:
    AVCaptureVideoDataOutputSampleBufferDelegate,
    AVCaptureAudioDataOutputSampleBufferDelegate
{
    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Identify which output produced this sample by reference identity —
        // this is the canonical Apple pattern.
        if output === videoOutput {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            onVideoFrame?(pixelBuffer, pts)
        } else if output === audioOutput {
            onAudioSample?(sampleBuffer)
        }
    }
}

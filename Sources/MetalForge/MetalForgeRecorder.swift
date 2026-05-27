import Foundation
import AVFoundation
import VideoToolbox
import CoreVideo
// MTLTexture / CMSampleBuffer / MTLDevice etc. aren't yet annotated Sendable
// in Apple's headers; @preconcurrency downgrades Swift 6 strict-mode warnings
// while we cross-queue these reference types.
@preconcurrency import Metal
@preconcurrency import CoreMedia

/// Trivial @unchecked Sendable box used to carry not-yet-Sendable Apple types
/// (CMSampleBuffer, …) across queue boundaries without compiler warnings.
private struct UncheckedSendableBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

/// Records the processed output of `MetalForgePipeline` (a stream of
/// `MTLTexture`s) plus a pristine audio passthrough into a single MP4/MOV file
/// via `AVAssetWriter`.
///
/// ## Architecture
///
/// ```
///                     ┌─ videoQueue ─► [Pull CVPixelBuffer from adaptor pool]
///                     │                ► [Wrap as MTLTexture(s) via 2nd cache]
///   appendVideoFrame ─┤                ► [GPU encode: BGRA blit | RGB→YUV compute]
///                     │                ► [adaptor.append(buffer, withPresentationTime: pts)]
///                     │
///                     └─ audioQueue ─► [audioInput.append(sampleBuffer)] (passthrough)
/// ```
///
/// ## Synchronisation handshake
///
/// AVAssetWriter syncs audio and video by **PTS only** — not by call order.
/// Our contract:
///
/// 1. Both video and audio inputs receive samples with the **original** PTS from
///    `AVCaptureSession` (or whatever upstream source produced them). We never
///    rewrite PTS.
/// 2. `startSession(atSourceTime:)` is called on the *first* video frame's PTS.
///    Any audio sample with an earlier PTS arrives before the session window and
///    is dropped.
/// 3. Video work is async-dispatched to `videoQueue` so the GPU encode + adaptor
///    append doesn't block the pipeline's processing thread. Audio is dispatched
///    to its own `audioQueue` — audio frames go through immediately because
///    there's no GPU work.
/// 4. **Frame-drop policy**: before any work, `appendVideoFrame` checks
///    `videoInput.isReadyForMoreMediaData`. If `false`, the frame is recycled and
///    dropped — better a visible video stutter than rising memory and growing
///    A/V drift. Audio is **never** dropped under back-pressure (it would cause
///    audible clicks); if the audio input's queue is saturated the sample is
///    skipped with a warning, but in practice audio bandwidth is so small this
///    never happens.
/// 5. On `stopRecording()`: both queues are drained synchronously, both inputs
///    marked finished, then `finishWriting` is awaited through a continuation.
///
/// ## Zero-copy output path
///
/// A **second** `CVMetalTextureCacheRef` (separate from the one in
/// `MetalForgeEngine`) is bound to the adaptor's `CVPixelBufferPool`. Each
/// frame:
///   1. Pull a CVPixelBuffer from `adaptor.pixelBufferPool` (IOSurface-backed,
///      Metal-compatible, format matches our recording profile).
///   2. Wrap the buffer's plane(s) as writable `MTLTexture` through the second
///      cache — no allocation, no byte copy.
///   3. Encode either a `MTLBlitCommandEncoder` copy (SDR) or the
///      `RGBToYUVConverter` compute pair (HDR 10-bit YUV) **into** the wrapped
///      texture(s).
///   4. After `commandBuffer.waitUntilCompleted()`, append the CVPixelBuffer to
///      the adaptor with the original PTS.
public final class MetalForgeRecorder: @unchecked Sendable {

    // MARK: - Public types

    /// Recorder lifecycle state.
    public enum State: Equatable, Sendable {
        case idle
        case recording
        case paused
        case finalizing
        case finished
        case failed(String)
    }

    // MARK: - Public configuration

    /// Called after the recorder has finished consuming a texture passed to
    /// `appendVideoFrame`. Typically wired to `MetalForgePipeline.recycle(_:)`
    /// so the pool reclaims the texture. **Fires on `videoQueue`**.
    public var recycleHandler: (@Sendable (MTLTexture) -> Void)?

    /// Called when the underlying `AVAssetWriter` transitions to `.failed`.
    /// Fires on `videoQueue`.
    public var errorHandler: (@Sendable (Error) -> Void)?

    // MARK: - Configuration (immutable after init)

    private let engine: MetalForgeEngine
    private let videoSize: CGSize
    private let frameRate: Int
    private let workingColorSpace: MetalForgeColorSpace
    private let rgbToYUV: RGBToYUVConverter?    // only built for HDR

    // MARK: - Queues

    /// Serial queue for video encode + append. Off-pipeline so processing doesn't
    /// block. QoS `.userInitiated` matches the real-time intent.
    private let videoQueue: DispatchQueue
    /// Serial queue for audio passthrough. Independent of `videoQueue` so a
    /// video encode hiccup never delays an audio sample append.
    private let audioQueue: DispatchQueue

    // MARK: - Lock-protected state

    private let stateLock = NSLock()
    private var _state: State = .idle
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var audioInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var outputTextureCache: CVMetalTextureCache?
    private var sessionStarted: Bool = false

    // MARK: - Init

    /// Construct a recorder. The video codec, pixel format, and HDR colour
    /// metadata are derived from `workingColorSpace`:
    /// - `.sdr`     → H.264 + BT.709 + `kCVPixelFormatType_32BGRA`
    /// - `.hdr10PQ` → HEVC Main10 + BT.2020/PQ + `kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange`
    /// - `.hlg`     → HEVC Main10 + BT.2020/HLG + 10-bit YUV bi-planar
    ///
    /// - Parameters:
    ///   - engine: Shared engine — the recorder reuses its `MTLDevice` and
    ///             command queue for the output encode pass.
    ///   - videoSize: Output frame dimensions in pixels (must match the
    ///                texture you'll pass to `appendVideoFrame`).
    ///   - workingColorSpace: Must match the pipeline's working colour space.
    ///   - frameRate: Hint to the encoder for rate-control tuning.
    public init(
        engine: MetalForgeEngine,
        videoSize: CGSize,
        workingColorSpace: MetalForgeColorSpace,
        frameRate: Int = 60
    ) throws {
        self.engine            = engine
        self.videoSize         = videoSize
        self.workingColorSpace = workingColorSpace
        self.frameRate         = frameRate
        // The RGB→YUV converter is only needed when we write 10-bit YUV
        // bi-planar (HDR). Skip compilation for SDR pipelines.
        self.rgbToYUV          = workingColorSpace.isHDR
            ? try RGBToYUVConverter(engine: engine)
            : nil
        self.videoQueue = DispatchQueue(
            label: "com.metalforge.recorder.video",
            qos: .userInitiated
        )
        self.audioQueue = DispatchQueue(
            label: "com.metalforge.recorder.audio",
            qos: .userInitiated
        )
    }

    // MARK: - Public state accessors

    /// Current lifecycle state. Read-thread-safe.
    public var state: State {
        stateLock.lock(); defer { stateLock.unlock() }
        return _state
    }

    // MARK: - Lifecycle

    /// Begin recording to `outputURL`. Returns immediately after the writer is
    /// configured; the first call to `appendVideoFrame` will start the session
    /// timeline at that frame's PTS.
    ///
    /// - Throws: `MetalForgeError.recorderInvalidState` if not currently `.idle`,
    ///           or `recorderAssetWriterFailed` / `recorderCannotAddInput` if
    ///           AVFoundation refuses the configuration.
    public func startRecording(outputURL: URL) throws {
        stateLock.lock()
        guard _state == .idle || _state == .finished else {
            let badState = "\(_state)"
            stateLock.unlock()
            throw MetalForgeError.recorderInvalidState(
                "startRecording requires .idle, currently \(badState)"
            )
        }
        stateLock.unlock()

        // ----- AVAssetWriter -----
        let fileType: AVFileType = (outputURL.pathExtension.lowercased() == "mov")
            ? .mov : .mp4
        // Remove any stale file at the URL — AVAssetWriter refuses to overwrite.
        try? FileManager.default.removeItem(at: outputURL)

        let writer: AVAssetWriter
        do {
            writer = try AVAssetWriter(outputURL: outputURL, fileType: fileType)
        } catch {
            throw MetalForgeError.recorderAssetWriterFailed(error.localizedDescription)
        }

        // ----- Video input -----
        let videoSettings = makeVideoOutputSettings()
        let vInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        vInput.expectsMediaDataInRealTime = true

        let sourceAttributes = makePixelBufferAttributes()
        let pba = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: vInput,
            sourcePixelBufferAttributes: sourceAttributes
        )

        guard writer.canAdd(vInput) else {
            throw MetalForgeError.recorderCannotAddInput("video")
        }
        writer.add(vInput)

        // ----- Audio input (passthrough) -----
        // `outputSettings: nil` tells AVAssetWriter to mux the audio sample
        // buffers without decoding or re-encoding — original quality, original
        // PTS, sample-accurate alignment.
        let aInput = AVAssetWriterInput(mediaType: .audio, outputSettings: nil)
        aInput.expectsMediaDataInRealTime = true

        guard writer.canAdd(aInput) else {
            throw MetalForgeError.recorderCannotAddInput("audio")
        }
        writer.add(aInput)

        // ----- Second CVMetalTextureCache for the output pool -----
        // This cache wraps the IOSurfaces that the adaptor's pool allocates so
        // the GPU can write directly into them — same zero-copy mechanism as on
        // the input side, but in reverse.
        var cache: CVMetalTextureCache?
        let cacheStatus = CVMetalTextureCacheCreate(
            kCFAllocatorDefault, nil, engine.device, nil, &cache
        )
        guard cacheStatus == kCVReturnSuccess, let outputCache = cache else {
            throw MetalForgeError.textureCacheCreationFailed(cacheStatus)
        }

        // ----- Start the writer -----
        guard writer.startWriting() else {
            let detail = writer.error?.localizedDescription ?? "unknown"
            throw MetalForgeError.recorderAssetWriterFailed("startWriting failed: \(detail)")
        }

        // ----- Commit state -----
        stateLock.lock()
        self.assetWriter        = writer
        self.videoInput         = vInput
        self.audioInput         = aInput
        self.adaptor            = pba
        self.outputTextureCache = outputCache
        self.sessionStarted     = false
        self._state             = .recording
        stateLock.unlock()
    }

    /// Suspend frame intake. Subsequent `appendVideoFrame` and
    /// `appendAudioSample` calls are dropped (and textures recycled) until
    /// `resumeRecording` is called.
    ///
    /// Note: PTS gaps will appear in the output file. True pause-without-gaps
    /// requires PTS offset bookkeeping, which is out of scope for this initial
    /// implementation.
    public func pauseRecording() {
        stateLock.lock(); defer { stateLock.unlock() }
        if _state == .recording { _state = .paused }
    }

    public func resumeRecording() {
        stateLock.lock(); defer { stateLock.unlock() }
        if _state == .paused { _state = .recording }
    }

    /// Stop recording and finalise the output file. Drains both internal queues,
    /// marks both inputs as finished, awaits `AVAssetWriter.finishWriting`, then
    /// releases all per-session resources.
    ///
    /// Safe to call from any thread.
    public func stopRecording() async throws {
        // NSLock.withLock is the async-safe locking primitive (introduced in
        // Swift 5.9). It guarantees the lock is released even on throw, and
        // unlike raw lock()/unlock() it's callable from async contexts.
        let snapshot: (writer: AVAssetWriter?, vInput: AVAssetWriterInput?, aInput: AVAssetWriterInput?)?
        snapshot = try stateLock.withLock { () -> (AVAssetWriter?, AVAssetWriterInput?, AVAssetWriterInput?) in
            guard _state == .recording || _state == .paused else {
                throw MetalForgeError.recorderInvalidState(
                    "stopRecording requires .recording or .paused, currently \(_state)"
                )
            }
            _state = .finalizing
            return (assetWriter, videoInput, audioInput)
        }
        guard let (writer, vInput, aInput) = snapshot else { return }

        // ----- Drain pending work on both queues -----
        // Synchronous barrier sync waits for all previously async-dispatched
        // closures to complete. After this returns, no more video encodes are
        // in flight and no more sampleBuffer appends are queued.
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            videoQueue.async { cont.resume() }
        }
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            audioQueue.async { cont.resume() }
        }

        // ----- Mark inputs finished -----
        // Per AVAssetWriter docs, both inputs must be marked finished before
        // finishWriting is called. Marking is idempotent and thread-safe.
        vInput?.markAsFinished()
        aInput?.markAsFinished()

        // ----- Await muxer finalisation -----
        if let writer {
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                writer.finishWriting {
                    cont.resume()
                }
            }
        }

        // ----- Capture final status before clearing -----
        let finalStatus: AVAssetWriter.Status = writer?.status ?? .unknown
        let finalError: Error?                = writer?.error

        // ----- Release all per-session resources under the lock -----
        stateLock.withLock {
            self.assetWriter        = nil
            self.videoInput         = nil
            self.audioInput         = nil
            self.adaptor            = nil
            self.outputTextureCache = nil
            self.sessionStarted     = false
            _state = (finalStatus == .completed)
                ? .finished
                : .failed(finalError?.localizedDescription ?? "unknown")
        }

        if finalStatus != .completed {
            throw MetalForgeError.recorderAssetWriterFailed(
                finalError?.localizedDescription ?? "finalisation failed with status \(finalStatus.rawValue)"
            )
        }
    }

    // MARK: - Video frame ingest

    /// Submit a single processed frame for recording.
    ///
    /// Drops the frame (and recycles the texture) immediately if:
    /// - the recorder is not in `.recording` state, or
    /// - the video input's internal buffer is full (`isReadyForMoreMediaData == false`).
    ///
    /// Otherwise async-dispatches to `videoQueue` for GPU encode + append. The
    /// texture is captured in the dispatch closure (which retains it) and
    /// released through `recycleHandler` after the GPU completes its work.
    public func appendVideoFrame(texture: MTLTexture, presentationTime: CMTime) {
        // ----- Fast-path drop on caller thread -----
        // Grab atomic state snapshot under the lock.
        stateLock.lock()
        let recording  = (_state == .recording)
        let vInput     = videoInput
        stateLock.unlock()

        guard recording else {
            recycleHandler?(texture)
            return
        }
        // isReadyForMoreMediaData is documented thread-safe to read.
        if let vInput, !vInput.isReadyForMoreMediaData {
            // Back-pressure drop — the encoder can't keep up. Recycle now,
            // saving a dispatch + GPU encode we'd waste only to drop later.
            recycleHandler?(texture)
            return
        }

        videoQueue.async { [weak self] in
            self?.processVideoFrame(texture: texture, presentationTime: presentationTime)
        }
    }

    /// Main video frame work — runs on `videoQueue`. Always recycles the texture
    /// (via `defer`) so the pipeline pool stays in balance regardless of error
    /// path taken.
    private func processVideoFrame(texture: MTLTexture, presentationTime: CMTime) {
        // Recycle unconditionally on exit. `recycleHandler` is captured at the
        // moment of read; if the user nils it out mid-flight, the previous
        // value still fires for the texture currently in flight.
        let recycler = self.recycleHandler
        defer { recycler?(texture) }

        // ----- Re-snapshot under the lock (state may have changed since dispatch) -----
        stateLock.lock()
        let recording  = (_state == .recording)
        let writer     = assetWriter
        let vInput     = videoInput
        let pba        = adaptor
        let cache      = outputTextureCache
        let sessionUp  = sessionStarted
        stateLock.unlock()

        guard recording, let writer, let vInput, let pba, let cache else { return }
        guard vInput.isReadyForMoreMediaData else { return }

        // ----- Start session on first valid frame -----
        // Session timeline anchored to the original upstream PTS — this is what
        // keeps audio and video aligned, since audio samples carry the same
        // timebase from the AVCaptureSession.
        if !sessionUp {
            writer.startSession(atSourceTime: presentationTime)
            stateLock.lock()
            sessionStarted = true
            stateLock.unlock()
        }

        // autoreleasepool: AVAssetWriter and CVPixelBuffer machinery allocate
        // significant Obj-C heap per frame; without this pool the autoreleased
        // objects accumulate until the next runloop iteration of the calling
        // thread — which, on a dedicated dispatch queue, can be very far away.
        autoreleasepool {
            // ----- Pull buffer from the adaptor's pool (zero-allocation hot path) -----
            guard let pool = pba.pixelBufferPool else { return }
            var pbOpt: CVPixelBuffer?
            let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pbOpt)
            guard status == kCVReturnSuccess, let pixelBuffer = pbOpt else { return }

            // ----- GPU copy: blit (SDR) or RGB→YUV compute (HDR) -----
            let copied: Bool = workingColorSpace.isHDR
                ? copyToYUVPixelBuffer(source: texture, destination: pixelBuffer, cache: cache)
                : copyToBGRAPixelBuffer(source: texture, destination: pixelBuffer, cache: cache)
            guard copied else { return }

            // ----- Append with original PTS -----
            // PTS-preserving handshake: the timestamp we pass here is the same
            // one the upstream produced. AVAssetWriter muxes by PTS, not by
            // call order — so audio samples submitted on audioQueue with their
            // own original timestamps will end up perfectly interleaved at the
            // correct presentation times in the muxed file.
            let appended = pba.append(pixelBuffer, withPresentationTime: presentationTime)
            if !appended {
                handleAppendFailure(writer: writer)
            }
        }
    }

    // MARK: - Audio sample ingest

    /// Submit one audio sample buffer for passthrough recording. Dispatches to
    /// `audioQueue` (independent of video). Drops silently if not in
    /// `.recording`, before the session starts, or if the audio input is full.
    public func appendAudioSample(_ sampleBuffer: CMSampleBuffer) {
        // Quick state check on caller thread to avoid a dispatch when stopped.
        stateLock.lock()
        let recording = (_state == .recording)
        stateLock.unlock()

        guard recording else { return }

        // Box CMSampleBuffer in an explicit @unchecked Sendable wrapper so the
        // closure can capture it under Swift 6 strict concurrency.
        // CMSampleBuffer is documented thread-safe for the operations we
        // perform (append + read PTS); it simply isn't annotated yet.
        let boxed = UncheckedSendableBox(sampleBuffer)
        audioQueue.async { [weak self] in
            self?.processAudioSample(boxed.value)
        }
    }

    private func processAudioSample(_ sampleBuffer: CMSampleBuffer) {
        stateLock.lock()
        let recording = (_state == .recording)
        let aInput    = audioInput
        let sessionUp = sessionStarted
        stateLock.unlock()

        guard recording, let aInput else { return }
        // Drop audio that predates the session start — the writer rejects samples
        // earlier than `startSession(atSourceTime:)` and would log an error.
        // Once the first video frame starts the session, subsequent audio flows.
        guard sessionUp else { return }
        guard aInput.isReadyForMoreMediaData else { return }

        // Passthrough — no decoding, no re-encoding, original PTS preserved
        // by AVAssetWriter because the sample buffer already carries it.
        // Wrap in autoreleasepool because append() autoreleases internal
        // CMBlockBuffer references that would otherwise pile up on a dispatch
        // queue with no runloop drain.
        _ = autoreleasepool {
            aInput.append(sampleBuffer)
        }
    }

    // MARK: - GPU copies

    /// SDR path: simple blit from the pipeline RGB texture to the BGRA-backed
    /// output `CVPixelBuffer`. Both textures share the same pixel format so a
    /// `MTLBlitCommandEncoder` is sufficient — no shader needed.
    private func copyToBGRAPixelBuffer(
        source: MTLTexture,
        destination pixelBuffer: CVPixelBuffer,
        cache: CVMetalTextureCache
    ) -> Bool {
        let width  = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        var cvTex: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pixelBuffer, nil,
            .bgra8Unorm, width, height, 0, &cvTex
        )
        guard status == kCVReturnSuccess,
              let cvTex,
              let destTexture = CVMetalTextureGetTexture(cvTex)
        else { return false }

        guard let commandBuffer = engine.commandQueue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder()
        else { return false }
        commandBuffer.label = "MetalForgeRecorder.BGRABlit"

        // Source and destination dimensions must match; if a user feeds an
        // unexpected size, clamp to the smaller of the two so we don't crash.
        let copyWidth  = min(source.width,  destTexture.width)
        let copyHeight = min(source.height, destTexture.height)
        blit.copy(
            from:               source,
            sourceSlice:        0,
            sourceLevel:        0,
            sourceOrigin:       MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize:         MTLSize(width: copyWidth, height: copyHeight, depth: 1),
            to:                 destTexture,
            destinationSlice:   0,
            destinationLevel:   0,
            destinationOrigin:  MTLOrigin(x: 0, y: 0, z: 0)
        )
        blit.endEncoding()
        commandBuffer.commit()
        // waitUntilCompleted is necessary: the adaptor.append happens
        // immediately after, and the encoder must see fully-written data.
        commandBuffer.waitUntilCompleted()
        return true
    }

    /// HDR path: wrap both planes of the 10-bit YUV bi-planar `CVPixelBuffer`
    /// as `.r16Unorm` / `.rg16Unorm` textures, then run the `RGBToYUVConverter`
    /// luma + chroma compute passes directly into them.
    private func copyToYUVPixelBuffer(
        source: MTLTexture,
        destination pixelBuffer: CVPixelBuffer,
        cache: CVMetalTextureCache
    ) -> Bool {
        guard let rgbToYUV else { return false }

        // ----- Wrap luma plane (.r16Unorm) -----
        let lumaWidth  = CVPixelBufferGetWidthOfPlane(pixelBuffer,  0)
        let lumaHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        var lumaCV: CVMetalTexture?
        let lumaStatus = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pixelBuffer, nil,
            .r16Unorm, lumaWidth, lumaHeight, 0, &lumaCV
        )
        guard lumaStatus == kCVReturnSuccess,
              let lumaCV,
              let lumaTexture = CVMetalTextureGetTexture(lumaCV)
        else { return false }

        // ----- Wrap chroma plane (.rg16Unorm, half-res) -----
        let chromaWidth  = CVPixelBufferGetWidthOfPlane(pixelBuffer,  1)
        let chromaHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1)
        var chromaCV: CVMetalTexture?
        let chromaStatus = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pixelBuffer, nil,
            .rg16Unorm, chromaWidth, chromaHeight, 1, &chromaCV
        )
        guard chromaStatus == kCVReturnSuccess,
              let chromaCV,
              let chromaTexture = CVMetalTextureGetTexture(chromaCV)
        else { return false }

        // ----- Encode the two compute passes -----
        guard let commandBuffer = engine.commandQueue.makeCommandBuffer() else { return false }
        commandBuffer.label = "MetalForgeRecorder.RGBToYUV"

        // Always BT.2020 video range for HDR HEVC HDR10/HLG output — this is
        // what VideoToolbox and downstream players expect. The transfer
        // function (PQ vs HLG) is already baked into the source texture's
        // values by HDREncodeFilter; the matrix just maps RGB → YCbCr.
        rgbToYUV.encode(
            source:        source,
            luma:          lumaTexture,
            chroma:        chromaTexture,
            commandBuffer: commandBuffer,
            matrix:        YUVColorMatrices.bt2020VideoRangeEncode,
            rangeFlg:      false   // video range — what the encoder expects
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return true
    }

    // MARK: - Failure handling

    private func handleAppendFailure(writer: AVAssetWriter) {
        // adaptor.append == false can mean either "transient back-pressure"
        // (rare given we already checked isReadyForMoreMediaData) or "writer
        // has failed permanently". Inspect status to decide.
        guard writer.status == .failed, let err = writer.error else { return }
        stateLock.lock()
        _state = .failed(err.localizedDescription)
        stateLock.unlock()
        errorHandler?(err)
    }

    // MARK: - Output settings builders

    private func makeVideoOutputSettings() -> [String: Any] {
        let pixelsPerFrame  = videoSize.width * videoSize.height
        let pixelsPerSecond = pixelsPerFrame * CGFloat(frameRate)
        // Bits-per-pixel rule of thumb: ~0.08 for H.264 SDR, ~0.12 for HEVC HDR.
        // Tuned for "high quality, manageable file size" — users wanting
        // archival quality should override or expose a knob in a future API.
        let bitsPerPixel: Double = workingColorSpace.isHDR ? 0.12 : 0.08
        let bitRate: Int = Int(Double(pixelsPerSecond) * bitsPerPixel)

        var compression: [String: Any] = [
            AVVideoAverageBitRateKey:           bitRate,
            AVVideoExpectedSourceFrameRateKey:  frameRate
        ]

        var settings: [String: Any] = [
            AVVideoCodecKey:  workingColorSpace.isHDR ? AVVideoCodecType.hevc : AVVideoCodecType.h264,
            AVVideoWidthKey:  Int(videoSize.width),
            AVVideoHeightKey: Int(videoSize.height)
        ]

        // ----- HDR-specific colour metadata -----
        // This is what tells players (QuickTime, Photos, web) to interpret the
        // file as BT.2020 / PQ-or-HLG and route through the HDR display path.
        if workingColorSpace.isHDR {
            // HEVC profile must be Main10 for 10-bit HDR. AVAssetWriter selects
            // it automatically when colour properties + pixel format both
            // request 10-bit, but we make it explicit for clarity.
            compression[AVVideoProfileLevelKey] = kVTProfileLevel_HEVC_Main10_AutoLevel as String

            settings[AVVideoColorPropertiesKey] = [
                AVVideoColorPrimariesKey:    AVVideoColorPrimaries_ITU_R_2020,
                AVVideoTransferFunctionKey:  workingColorSpace == .hlg
                    ? AVVideoTransferFunction_ITU_R_2100_HLG
                    : AVVideoTransferFunction_SMPTE_ST_2084_PQ,
                AVVideoYCbCrMatrixKey:       AVVideoYCbCrMatrix_ITU_R_2020
            ]
        } else {
            settings[AVVideoColorPropertiesKey] = [
                AVVideoColorPrimariesKey:    AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey:  AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey:       AVVideoYCbCrMatrix_ITU_R_709_2
            ]
        }

        settings[AVVideoCompressionPropertiesKey] = compression
        return settings
    }

    private func makePixelBufferAttributes() -> [String: Any] {
        // The format here drives what the adaptor's `pixelBufferPool` allocates.
        //   • SDR: BGRA (matches our pipeline's intermediate format — blit copy)
        //   • HDR: 10-bit YUV bi-planar video range (HEVC HDR10 / HLG native input)
        let pixelFormat: OSType = workingColorSpace.isHDR
            ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
            : kCVPixelFormatType_32BGRA

        return [
            kCVPixelBufferPixelFormatTypeKey  as String: pixelFormat,
            kCVPixelBufferWidthKey            as String: Int(videoSize.width),
            kCVPixelBufferHeightKey           as String: Int(videoSize.height),
            // The two attributes that together unlock the zero-copy output path:
            // IOSurface backing so the buffer is sharable with the GPU, and
            // Metal compatibility so CVMetalTextureCacheCreateTextureFromImage
            // can wrap it without an extra allocation.
            kCVPixelBufferMetalCompatibilityKey   as String: true,
            kCVPixelBufferIOSurfacePropertiesKey  as String: [:] as CFDictionary
        ]
    }
}

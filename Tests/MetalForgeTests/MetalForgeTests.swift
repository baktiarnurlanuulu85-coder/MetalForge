import XCTest
import Metal
@testable import MetalForge

final class MetalForgeEngineTests: XCTestCase {

    // Most GPU tests are guarded — CI machines may not have a Metal device.
    private var engine: MetalForgeEngine?

    override func setUp() async throws {
        engine = try? MetalForgeEngine()
    }

    func testEngineInitialises() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("No Metal device available in this environment.")
        }
        XCTAssertNoThrow(try MetalForgeEngine())
    }

    func testTextureCacheFlushDoesNotCrash() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        // Flushing an empty cache must not crash or assert.
        engine.flushTextureCache()
    }

    func testTexturePoolAcquireRecycle() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let pool = TexturePool(device: engine.device)

        let tex1 = pool.acquire(width: 256, height: 256, pixelFormat: .bgra8Unorm)
        XCTAssertNotNil(tex1)

        if let tex1 {
            pool.recycle(tex1)
            // After recycling, the next acquire of the same spec should return the
            // same MTLTexture object (pointer equality).
            let tex2 = pool.acquire(width: 256, height: 256, pixelFormat: .bgra8Unorm)
            XCTAssertTrue(tex1 === tex2, "Pool should return the recycled texture.")
        }
    }

    func testPipelineIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        // Now throws because it eagerly compiles the embedded YUVToRGBConverter PSO.
        XCTAssertNoThrow(try MetalForgePipeline(engine: engine))
    }

    func testPipelineProcessesYUV8BitPixelBuffer() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let pipeline = try MetalForgePipeline(engine: engine)

        // Build an empty 8-bit YUV bi-planar buffer marked Metal-compatible.
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, 128, 128,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            attrs as CFDictionary, &pb
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        guard let buffer = pb else { return }

        // No downstream filters: result is the YUV-converter output.
        let result = pipeline.process(pixelBuffer: buffer)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.pixelFormat, .bgra8Unorm)
        if let result { pipeline.recycle(result) }
    }

    func testPipelineProcessesYUV10BitPixelBufferAsHDR() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let pipeline = try MetalForgePipeline(engine: engine)

        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, 128, 128,
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            attrs as CFDictionary, &pb
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        guard let buffer = pb else { return }

        let result = pipeline.process(pixelBuffer: buffer)
        XCTAssertNotNil(result)
        // 10-bit YUV should auto-route through the HDR working space.
        XCTAssertEqual(result?.pixelFormat, .rgba16Float)
        if let result { pipeline.recycle(result) }
    }

    func testAdjustmentFilterIdentityIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        // This will fail in CI if the .metallib isn't bundled — that's intentional.
        // Run locally with `swift test` in an environment with Metal support.
        XCTAssertNoThrow(try AdjustmentFilter(engine: engine))
    }

    func testGlitchFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try GlitchFilter(engine: engine))
    }

    // MARK: - YUV converter

    func testYUVToRGBConverterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try YUVToRGBConverter(engine: engine))
    }

    func testMatrixSelectionFallsBackToBT709() {
        // A synthetic CVPixelBuffer with no colour attachments should fall back
        // to BT.709 video range (the standard SDR default).
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, 64, 64,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            nil, &pb
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        guard let buffer = pb else { return }

        let result = YUVColorMatrices.matrix(for: buffer)
        XCTAssertFalse(result.isFullRange)
        XCTAssertEqual(result.matrix.columns.0.x, 1.16438, accuracy: 0.0001)
    }

    func testMatrixSelectionDetectsFullRange() {
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, 64, 64,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            nil, &pb
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        guard let buffer = pb else { return }

        let result = YUVColorMatrices.matrix(for: buffer)
        XCTAssertTrue(result.isFullRange)
        // Full-range BT.709 has Y coefficient = 1.0 (no range expansion).
        XCTAssertEqual(result.matrix.columns.0.x, 1.0, accuracy: 0.0001)
    }

    func testColorSpaceIsHDRReport() {
        XCTAssertFalse(MetalForgeColorSpace.sdr.isHDR)
        XCTAssertTrue(MetalForgeColorSpace.hdr10PQ.isHDR)
        XCTAssertTrue(MetalForgeColorSpace.hlg.isHDR)
    }

    // MARK: - HDR transfer

    func testHDRDecodeFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try HDRDecodeFilter(engine: engine))
    }

    func testHDREncodeFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try HDREncodeFilter(engine: engine))
    }

    func testColorSpaceDetectionDefaultsTo10BitAsPQ() {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, 64, 64,
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            attrs as CFDictionary, &pb
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        guard let buffer = pb else { return }

        // No transfer function attachment → PQ default.
        XCTAssertEqual(MetalForgeColorSpace.detect(from: buffer), .hdr10PQ)
    }

    // MARK: - MetalForgeView

    @MainActor
    func testMetalForgeViewIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let view = try MetalForgeView(engine: engine, frame: CGRect(x: 0, y: 0, width: 256, height: 256))
        XCTAssertEqual(view.colorPixelFormat, .bgra8Unorm)
        XCTAssertTrue(view.isPaused)
        XCTAssertFalse(view.enableSetNeedsDisplay)
    }

    @MainActor
    func testMetalForgeViewSwitchesColorSpaceForHDR() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let view = try MetalForgeView(engine: engine, frame: CGRect(x: 0, y: 0, width: 256, height: 256))

        view.workingColorSpace = .hdr10PQ
        XCTAssertEqual(view.colorPixelFormat, .bgr10a2Unorm)

        view.workingColorSpace = .hlg
        XCTAssertEqual(view.colorPixelFormat, .bgr10a2Unorm)

        view.workingColorSpace = .sdr
        XCTAssertEqual(view.colorPixelFormat, .bgra8Unorm)
    }

    // MARK: - Analog distortion pack

    func testChromaticAberrationFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try ChromaticAberrationFilter(engine: engine))
    }

    func testAnalogNoiseFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try AnalogNoiseFilter(engine: engine))
    }

    func testHorizontalJitterFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try HorizontalJitterFilter(engine: engine))
    }

    // MARK: - Temporal effects pack

    func testMotionBlurFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try MotionBlurFilter(engine: engine))
    }

    func testNeonTrailsFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try NeonTrailsFilter(engine: engine))
    }

    // MARK: - Color grading pack

    func testColorCorrectionFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try ColorCorrectionFilter(engine: engine))
    }

    func testLUTFilterWithIdentityPresetIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try MetalForgeLUTFilter(engine: engine, preset: .identity, size: 16))
    }

    func testLUTFilterWithWarmPresetIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try MetalForgeLUTFilter(engine: engine, preset: .warm, size: 32))
    }

    func testLUTFilterRejectsWrongDataSize() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let badData = Data(count: 100)   // 16³×4 = 16384, definitely not 100
        XCTAssertThrowsError(
            try MetalForgeLUTFilter(engine: engine, lutSize: 16, lutData: badData)
        )
    }

    func testLUTPresetDataMatchesExpectedSize() {
        let data = MetalForgeLUTFilter.makePresetLUTData(preset: .warm, size: 32)
        XCTAssertEqual(data.count, 32 * 32 * 32 * 4)
    }

    func testMotionBlurClearHistoryDoesNotCrash() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let filter = try MotionBlurFilter(engine: engine)
        // Clearing without ever encoding is a valid no-op.
        filter.clearHistory()
        XCTAssertTrue(true, "clearHistory() should be safe to call before first encode")
    }

    // MARK: - Recorder

    func testRGBToYUVConverterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try RGBToYUVConverter(engine: engine))
    }

    func testRecorderSDRIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let recorder = try MetalForgeRecorder(
            engine: engine,
            videoSize: CGSize(width: 1280, height: 720),
            workingColorSpace: .sdr,
            frameRate: 30
        )
        XCTAssertEqual(recorder.state, .idle)
    }

    func testRecorderHDRIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let recorder = try MetalForgeRecorder(
            engine: engine,
            videoSize: CGSize(width: 3840, height: 2160),
            workingColorSpace: .hdr10PQ,
            frameRate: 60
        )
        XCTAssertEqual(recorder.state, .idle)
    }

    func testRecorderStartStopRoundTrip() async throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let recorder = try MetalForgeRecorder(
            engine: engine,
            videoSize: CGSize(width: 640, height: 480),
            workingColorSpace: .sdr,
            frameRate: 30
        )

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalforge-test-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        try recorder.startRecording(outputURL: tmpURL)
        XCTAssertEqual(recorder.state, .recording)

        // Stop without any frames — writer should fail finalisation because
        // there's no session, which is the expected behaviour we want to verify.
        do {
            try await recorder.stopRecording()
            // If it succeeds, state should be .finished
            XCTAssertEqual(recorder.state, .finished)
        } catch {
            // Or it fails — also acceptable for an empty session.
            if case .failed = recorder.state {} else {
                XCTFail("Expected .failed state after failed finalisation, got \(recorder.state)")
            }
        }
    }

    func testColorSpaceDetectionRecognisesHLG() {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, 64, 64,
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            attrs as CFDictionary, &pb
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        guard let buffer = pb else { return }

        CVBufferSetAttachment(
            buffer,
            kCVImageBufferTransferFunctionKey,
            kCVImageBufferTransferFunction_ITU_R_2100_HLG,
            .shouldPropagate
        )
        XCTAssertEqual(MetalForgeColorSpace.detect(from: buffer), .hlg)
    }
}

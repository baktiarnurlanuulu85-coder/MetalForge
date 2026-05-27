import Metal
import CoreVideo
import Foundation

/// Executes a complete MetalForge processing chain on a `CVPixelBuffer`.
///
/// ## Auto-detection of input format
/// `process(pixelBuffer:)` inspects `CVPixelBufferGetPixelFormatType` and chooses
/// the working colour space and intermediate texture format automatically:
///
/// | Input format                                | Working space | Intermediate format |
/// |---------------------------------------------|---------------|---------------------|
/// | `kCVPixelFormatType_32BGRA`                 | SDR           | `.bgra8Unorm`       |
/// | `420YpCbCr8BiPlanar{Full,Video}Range`       | SDR           | `.bgra8Unorm`       |
/// | `420YpCbCr10BiPlanar{Full,Video}Range`      | HDR           | `.rgba16Float`      |
///
/// For YUV inputs, a `YUVToRGBConverter` is automatically inserted as stage 0 of
/// the chain. The colour matrix and range flag are read from the pixel buffer's
/// `kCVImageBufferYCbCrMatrixKey` attachment via `YUVColorMatrices.matrix(for:)`.
///
/// Downstream `MetalForgeFilter`s receive the already-RGB intermediate texture
/// and, since each filter checks `source.pixelFormat`, automatically select
/// their HDR or SDR specialised pipeline.
///
/// ## Data Flow
/// ```
/// CVPixelBuffer ─┬─[BGRA]─► sourceTexture ──► [filter 0] ──► … ──► result
///                │
///                └─[YUV ]─► (luma, chroma) ─► [YUVConverter] ─► rgb0 ─► [filter 0] ─► …
/// ```
/// Intermediate textures come from `TexturePool`; the final result is owned by
/// the caller until passed back via `recycle(_:)`.
///
/// ## Concurrency
/// `process(pixelBuffer:)` is safe to call from any serial queue; it **blocks**
/// until the GPU completes (`waitUntilCompleted`). For production 60 fps work,
/// replace with `addCompletedHandler` and signal via `DispatchSemaphore` or
/// `CheckedContinuation`.
public final class MetalForgePipeline: @unchecked Sendable {

    private let engine: MetalForgeEngine
    private let pool: TexturePool
    private let yuvConverter: YUVToRGBConverter
    private let hdrDecode: HDRDecodeFilter
    private let hdrEncode: HDREncodeFilter
    // Mutation safe only between frames or with external synchronisation.
    private var filters: [any MetalForgeFilter] = []

    /// Initialise the pipeline. Throws if any built-in MSL kernel
    /// (YUV converter, PQ/HLG decode + encode) cannot be loaded or specialised
    /// — this should only happen if the package's `.metallib` is corrupted.
    public init(engine: MetalForgeEngine) throws {
        self.engine       = engine
        self.pool         = TexturePool(device: engine.device)
        self.yuvConverter = try YUVToRGBConverter(engine: engine)
        self.hdrDecode    = try HDRDecodeFilter(engine: engine)
        self.hdrEncode    = try HDREncodeFilter(engine: engine)
    }

    // MARK: - Filter management

    public func append(_ filter: any MetalForgeFilter) {
        filters.append(filter)
    }

    public func removeAllFilters() {
        filters.removeAll()
    }

    // MARK: - Processing

    /// Process a single `CVPixelBuffer` end-to-end.
    ///
    /// - Returns: The final processed `MTLTexture` (pool-owned), or `nil` if the
    ///   input format is unsupported, plane extraction fails, or texture
    ///   allocation fails. Caller **must** pass the returned texture to
    ///   `recycle(_:)` once it has been presented / encoded downstream.
    public func process(pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)

        switch format {
        case kCVPixelFormatType_32BGRA:
            return processBGRA(pixelBuffer: pixelBuffer)

        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
             kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            return processYUV(
                pixelBuffer: pixelBuffer,
                colorSpace: .sdr,
                intermediateFormat: .bgra8Unorm
            )

        case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
             kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
            // 10-bit YUV → HDR. Distinguish PQ vs HLG from the buffer's transfer
            // function attachment so the decode/encode stages use the correct math.
            let detected = MetalForgeColorSpace.detect(from: pixelBuffer)
            return processYUV(
                pixelBuffer: pixelBuffer,
                colorSpace: detected,
                intermediateFormat: .rgba16Float
            )

        default:
            // Unsupported pixel format: caller should arrange a CPU pre-conversion.
            return nil
        }
    }

    // MARK: - BGRA fast path

    private func processBGRA(pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        // No filters configured: hand back the raw CVPixelBuffer-backed texture.
        // Caller MUST NOT recycle this — it is owned by the CVMetalTextureCache.
        if filters.isEmpty {
            return engine.makeTexture(from: pixelBuffer)
        }

        return autoreleasepool {
            guard
                let source = engine.makeTexture(from: pixelBuffer),
                let commandBuffer = engine.commandQueue.makeCommandBuffer()
            else { return nil }
            commandBuffer.label = "MetalForgePipeline(BGRA)"

            return runFilterChain(
                source: source,
                width: source.width,
                height: source.height,
                intermediateFormat: source.pixelFormat,
                commandBuffer: commandBuffer
            )
        }
    }

    // MARK: - YUV path (8-bit + 10-bit)

    private func processYUV(
        pixelBuffer: CVPixelBuffer,
        colorSpace: MetalForgeColorSpace,
        intermediateFormat: MTLPixelFormat
    ) -> MTLTexture? {
        return autoreleasepool {
            guard
                let planes = engine.makeTextures(from: pixelBuffer, colorSpace: colorSpace),
                let commandBuffer = engine.commandQueue.makeCommandBuffer()
            else { return nil }
            commandBuffer.label = "MetalForgePipeline(YUV)"

            let width  = planes.luma.width
            let height = planes.luma.height

            // ----- Stage layout -----
            //  0           : YUV converter output (PQ/HLG-encoded RGB for HDR; sRGB-ish for SDR)
            //  1           : HDR decode → linear light  (only for HDR; absent for SDR)
            //  2 .. 1+N    : downstream user filters    (in linear light for HDR)
            //  2+N         : HDR encode → display PQ/HLG (only for HDR; absent for SDR)
            //
            // We pre-acquire every intermediate up front so the texture pool
            // doesn't contend during encode and we can roll back cleanly on failure.
            let isHDR = colorSpace.isHDR
            // Configure the HDR filters' transfer function for this frame.
            // Safe to write here — the previous frame already completed (we
            // waitUntilCompleted at the bottom).
            if isHDR {
                hdrDecode.colorSpace = colorSpace
                hdrEncode.colorSpace = colorSpace
            }

            let userFilterCount = filters.count
            let extraHDRStages  = isHDR ? 2 : 0   // decode + encode
            let stageCount      = 1 + userFilterCount + extraHDRStages

            var outputs: [MTLTexture] = []
            outputs.reserveCapacity(stageCount)
            for _ in 0..<stageCount {
                guard let tex = pool.acquire(
                    width: width,
                    height: height,
                    pixelFormat: intermediateFormat
                ) else {
                    outputs.forEach { pool.recycle($0) }
                    return nil
                }
                outputs.append(tex)
            }

            // ----- Stage 0: YUV → RGB -----
            // For HDR the result is still PQ/HLG-encoded (non-linear); the
            // decode stage below converts to linear light. For SDR we hand
            // the BT.709-ish RGB straight to user filters.
            let colorInfo = YUVColorMatrices.matrix(for: pixelBuffer)
            yuvConverter.encode(
                luma:          planes.luma,
                chroma:        planes.chroma,
                destination:   outputs[0],
                commandBuffer: commandBuffer,
                matrix:        colorInfo.matrix,
                rangeFlg:      colorInfo.isFullRange
            )

            // Index of the first slot user filters will write into.
            // For HDR: outputs[0] (YUV) → outputs[1] (decode) → outputs[2] (filter 0) → …
            // For SDR: outputs[0] (YUV) → outputs[1] (filter 0) → …
            var nextIndex = 1

            // ----- Stage 1: HDR decode (HDR only) -----
            if isHDR {
                hdrDecode.encode(
                    source:        outputs[0],
                    destination:   outputs[1],
                    commandBuffer: commandBuffer
                )
                nextIndex = 2
            }

            // ----- Downstream user filters (linear light for HDR) -----
            for (i, filter) in filters.enumerated() {
                filter.encode(
                    source:        outputs[nextIndex - 1 + i],
                    destination:   outputs[nextIndex + i],
                    commandBuffer: commandBuffer
                )
            }

            // ----- Final stage: HDR encode (HDR only) -----
            if isHDR {
                let lastUserIdx = nextIndex - 1 + userFilterCount
                hdrEncode.encode(
                    source:        outputs[lastUserIdx],
                    destination:   outputs[lastUserIdx + 1],
                    commandBuffer: commandBuffer
                )
            }

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            // Recycle intermediates; caller owns the last (final result).
            for i in 0 ..< outputs.count - 1 {
                pool.recycle(outputs[i])
            }
            return outputs.last
        }
    }

    // MARK: - Shared filter-chain encoder

    /// Encode the downstream `MetalForgeFilter` chain into `commandBuffer`,
    /// allocating intermediates from the texture pool. Used by the BGRA path
    /// (the YUV path inlines its own variant because it has an extra stage 0).
    private func runFilterChain(
        source: MTLTexture,
        width: Int,
        height: Int,
        intermediateFormat: MTLPixelFormat,
        commandBuffer: MTLCommandBuffer
    ) -> MTLTexture? {
        var outputs: [MTLTexture] = []
        outputs.reserveCapacity(filters.count)
        for _ in filters {
            guard let tex = pool.acquire(
                width: width,
                height: height,
                pixelFormat: intermediateFormat
            ) else {
                outputs.forEach { pool.recycle($0) }
                return nil
            }
            outputs.append(tex)
        }

        for (i, filter) in filters.enumerated() {
            let input  = (i == 0) ? source : outputs[i - 1]
            let output = outputs[i]
            filter.encode(source: input, destination: output, commandBuffer: commandBuffer)
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        for i in 0 ..< outputs.count - 1 {
            pool.recycle(outputs[i])
        }
        return outputs.last
    }

    // MARK: - Resource management

    /// Return the final output texture to the pool once consumed.
    public func recycle(_ texture: MTLTexture) {
        pool.recycle(texture)
    }

    /// Release all pooled intermediates and flush the texture cache.
    /// Call on memory pressure.
    public func purgeTexturePool() {
        pool.purge()
        engine.flushTextureCache()
    }
}

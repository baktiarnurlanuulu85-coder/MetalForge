import Metal
import CoreVideo
import Foundation

/// Central Metal context: one device, one command queue, one texture cache.
///
/// ## Lifecycle
/// Create once per app and share the same instance across the pipeline.
/// The `CVMetalTextureCache` is a critical resource — call `flushTextureCache()`
/// in response to `UIApplication.didReceiveMemoryWarningNotification` to release
/// stale texture wrappers and prevent OOM on 4K 60 fps streams.
///
/// ## Thread Safety
/// `makeTexture(from:)` and `flushTextureCache()` may be called from any thread.
/// The underlying CVMetalTextureCache is itself thread-safe per Apple's documentation.
public final class MetalForgeEngine: @unchecked Sendable {

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // CVMetalTextureCache wraps IOSurface/CVPixelBuffer memory as MTLTexture
    // WITHOUT copying. The cache holds a reference to each live texture wrapper;
    // flush it regularly to reclaim memory from completed frames.
    private let textureCache: CVMetalTextureCache

    public init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw MetalForgeError.deviceNotAvailable
        }
        guard let queue = dev.makeCommandQueue() else {
            throw MetalForgeError.commandQueueCreationFailed
        }
        device = dev
        commandQueue = queue

        // kCVMetalTextureCacheMaximumTextureAgeKey: evict wrappers older than 1 s.
        // The producer (AVFoundation / VideoToolbox) must set
        // kCVPixelBufferMetalCompatibilityKey = true on CVPixelBuffers for the
        // zero-copy path to succeed.
        let attrs: [String: Any] = [
            kCVMetalTextureCacheMaximumTextureAgeKey as String: 1.0
        ]
        var cache: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(
            kCFAllocatorDefault,
            attrs as CFDictionary,
            dev,
            nil,
            &cache
        )
        guard status == kCVReturnSuccess, let resolvedCache = cache else {
            throw MetalForgeError.textureCacheCreationFailed(status)
        }
        textureCache = resolvedCache
    }

    // MARK: - Zero-copy texture extraction

    /// Wraps a `CVPixelBuffer` plane as an `MTLTexture` via the texture cache.
    ///
    /// **Zero-copy guarantee**: no pixel data is moved. The returned `MTLTexture`
    /// shares the same backing IOSurface as the `CVPixelBuffer`. The texture is
    /// valid only while the `CVPixelBuffer` is retained by the caller.
    ///
    /// Supported formats:
    /// - `kCVPixelFormatType_32BGRA`        → `.bgra8Unorm`
    /// - `kCVPixelFormatType_420YpCbCr8BiPlanarFullRange` / `VideoRange`
    ///   - plane 0 (luma)   → `.r8Unorm`
    ///   - plane 1 (chroma) → `.rg8Unorm`
    ///
    /// - Parameter planeIndex: Ignored for BGRA; selects the YUV plane for YCbCr.
    public func makeTexture(from pixelBuffer: CVPixelBuffer, planeIndex: Int = 0) -> MTLTexture? {
        let formatType = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let width:  Int
        let height: Int
        let pixelFormat: MTLPixelFormat

        switch formatType {
        case kCVPixelFormatType_32BGRA:
            width       = CVPixelBufferGetWidth(pixelBuffer)
            height      = CVPixelBufferGetHeight(pixelBuffer)
            pixelFormat = .bgra8Unorm

        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
             kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            width       = CVPixelBufferGetWidthOfPlane(pixelBuffer,  planeIndex)
            height      = CVPixelBufferGetHeightOfPlane(pixelBuffer, planeIndex)
            // Luma plane is single-channel R; chroma plane is two-channel RG (Cb, Cr).
            pixelFormat = (planeIndex == 0) ? .r8Unorm : .rg8Unorm

        default:
            return nil
        }

        var cvTexture: CVMetalTexture?
        // This call does NOT schedule any GPU work — it registers a mapping between
        // the IOSurface and a new MTLTexture descriptor inside the cache.
        let result = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            pixelFormat,
            width,
            height,
            planeIndex,
            &cvTexture
        )
        guard result == kCVReturnSuccess, let tex = cvTexture else { return nil }
        return CVMetalTextureGetTexture(tex)
    }

    // MARK: - YUV Bi-Planar zero-copy extraction

    /// Extracts both luma and chroma planes of a YUV bi-planar `CVPixelBuffer` as
    /// separate `MTLTexture`s, **without copying** any pixel data.
    ///
    /// Returns `nil` if the pixel format is not a recognised YUV bi-planar layout
    /// or if the requested precision does not match the buffer's actual bit depth.
    ///
    /// ## Plane formats
    /// - **SDR (8-bit YUV)** — `kCVPixelFormatType_420YpCbCr8BiPlanar{Full,Video}Range`
    ///   - luma   → `.r8Unorm`   (1 byte/texel)
    ///   - chroma → `.rg8Unorm`  (2 bytes/texel, half resolution per axis)
    /// - **HDR (10-bit YUV packed in 16-bit)** —
    ///   `kCVPixelFormatType_420YpCbCr10BiPlanar{Full,Video}Range`
    ///   - luma   → `.r16Unorm`  (10 bits stored in the high bits of a 16-bit word)
    ///   - chroma → `.rg16Unorm`
    ///
    /// ## Apple's 10-bit convention
    /// 10-bit samples are left-shifted by 6 bits into a 16-bit container. When the
    /// GPU reads a `.r16Unorm` texel, it divides by 65535 — yielding a value
    /// approximately equal to `sample10bit / 1023.984`. The ~0.1 % offset versus
    /// ideal 10-bit normalisation is absorbed into the colour matrix in
    /// downstream `YUVToRGBConverter`; do not pre-scale here.
    ///
    /// - Parameters:
    ///   - pixelBuffer: A bi-planar YUV buffer marked `kCVPixelBufferMetalCompatibilityKey = true`.
    ///   - colorSpace:  The expected colour space; selects the precision of plane textures.
    /// - Returns: `(luma, chroma)` tuple, or `nil` on format/bit-depth mismatch.
    public func makeTextures(
        from pixelBuffer: CVPixelBuffer,
        colorSpace: MetalForgeColorSpace
    ) -> (luma: MTLTexture, chroma: MTLTexture)? {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)

        // --- Validate format matches the requested colour-space precision ---
        let lumaFormat:   MTLPixelFormat
        let chromaFormat: MTLPixelFormat
        switch (format, colorSpace.isHDR) {
        case (kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,  false),
             (kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, false):
            lumaFormat   = .r8Unorm
            chromaFormat = .rg8Unorm

        case (kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,  true),
             (kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, true):
            lumaFormat   = .r16Unorm
            chromaFormat = .rg16Unorm

        default:
            // Mismatch (e.g. 8-bit buffer with .hdr10PQ requested, or BGRA input).
            return nil
        }

        guard
            let luma   = makeCVTexture(from: pixelBuffer, planeIndex: 0, format: lumaFormat),
            let chroma = makeCVTexture(from: pixelBuffer, planeIndex: 1, format: chromaFormat)
        else { return nil }

        return (luma, chroma)
    }

    /// Internal helper: build an `MTLTexture` for a specific plane and pixel format.
    private func makeCVTexture(
        from pixelBuffer: CVPixelBuffer,
        planeIndex: Int,
        format: MTLPixelFormat
    ) -> MTLTexture? {
        let width  = CVPixelBufferGetWidthOfPlane(pixelBuffer,  planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, planeIndex)

        var cvTexture: CVMetalTexture?
        let result = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            format,
            width,
            height,
            planeIndex,
            &cvTexture
        )
        guard result == kCVReturnSuccess, let tex = cvTexture else { return nil }
        return CVMetalTextureGetTexture(tex)
    }

    // MARK: - Memory management

    /// Evict all cached texture wrappers whose reference age exceeds the maximum.
    /// Call from a `UIApplication.didReceiveMemoryWarningNotification` handler.
    public func flushTextureCache() {
        CVMetalTextureCacheFlush(textureCache, 0)
    }
}

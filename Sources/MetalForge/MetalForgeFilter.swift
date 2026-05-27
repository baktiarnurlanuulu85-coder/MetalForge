import Metal
import simd

// MARK: - Standard downstream filter

/// A single-input GPU processing pass operating on already-converted RGB textures.
///
/// Implement this protocol for downstream effects (colour grading, blur, glitch,
/// stylisation, etc.). The source and destination textures share the same pixel
/// format, dimensions, and colour space.
///
/// ## Lifecycle Contract
/// - `encode` is called on the pipeline's processing queue, never the main thread.
/// - The implementation must **not** commit or wait on `commandBuffer`; the
///   pipeline owns its full lifecycle.
/// - Configuration properties (intensity, etc.) should be written by the caller
///   between frames, not during an in-flight encode.
public protocol MetalForgeFilter: AnyObject, Sendable {

    /// Encode the filter's compute or render pass.
    ///
    /// - Parameters:
    ///   - source:        Read-only input texture (typically `.private` storage).
    ///   - destination:   Write target; already allocated by the pipeline.
    ///   - commandBuffer: Pipeline-owned command buffer. Do not commit.
    func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    )
}

// MARK: - Source-stage filter (YUV → RGB and similar)

/// A multi-plane "source" filter that ingests separate luma and chroma textures
/// (typically from a YUV bi-planar `CVPixelBuffer`) and writes a single RGB
/// destination texture suitable for the standard `MetalForgeFilter` chain.
///
/// Source filters sit at index 0 of the pipeline. The `matrix` and `rangeFlg`
/// parameters carry the YCbCr → RGB conversion description; the caller is
/// expected to derive these from the originating `CVPixelBuffer`'s colour
/// attachments (see `YUVColorMatrices.matrix(for:)`).
public protocol MetalForgeSourceFilter: AnyObject, Sendable {

    /// Encode a luma + chroma → RGB compute pass.
    ///
    /// - Parameters:
    ///   - luma:          Single-channel luma plane (`.r8Unorm` or `.r16Unorm`).
    ///   - chroma:        Two-channel CbCr plane at half resolution per axis.
    ///   - destination:   Full-resolution RGB target (`.bgra8Unorm` or `.rgba16Float`).
    ///   - commandBuffer: Pipeline-owned command buffer. Do not commit.
    ///   - matrix:        Pre-scaled YCbCr→RGB matrix (3×3) including any range expansion.
    ///   - rangeFlg:      `true` for full-range YCbCr, `false` for video (limited) range.
    func encode(
        luma: MTLTexture,
        chroma: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer,
        matrix: matrix_float3x3,
        rangeFlg: Bool
    )
}

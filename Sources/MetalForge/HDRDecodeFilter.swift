import Metal
import Foundation

/// Applies the inverse transfer function (EOTF) for an HDR signal, converting
/// non-linear PQ or HLG-encoded pixel values into linear-light values suitable
/// for mathematically correct downstream processing.
///
/// ## What it does
/// Implements the exact BT.2100 EOTFs:
/// - **PQ (`.hdr10PQ`)**: SMPTE ST.2084 EOTF. Output `1.0 ≡ 10,000 cd/m²`.
///   SDR diffuse white sits at ≈ `0.01` after decode.
/// - **HLG (`.hlg`)**: BT.2100 inverse OETF (per channel). The OOTF is
///   intentionally **not** applied — it is a display-side scaling that depends
///   on the actual peak luminance of the target and must run on the
///   presentation stage, not in the intermediate processing chain.
///
/// ## Usage
/// Typically owned and configured by `MetalForgePipeline`, but can also be
/// dropped into a custom pipeline. Configure `colorSpace` before each frame:
///
/// ```swift
/// let decoder = try HDRDecodeFilter(engine: engine)
/// decoder.colorSpace = MetalForgeColorSpace.detect(from: pixelBuffer)
/// decoder.encode(source: pqTexture, destination: linearTexture, commandBuffer: cb)
/// ```
///
/// ## Thread Safety
/// `@unchecked Sendable`: `colorSpace` is settable, but expected to be written
/// between frames only. The pipeline serial queue provides this guarantee.
public final class HDRDecodeFilter: @unchecked Sendable, MetalForgeFilter {

    /// Active transfer function. Setting this to `.sdr` is a programming error;
    /// the encode call falls through to the PQ pipeline in that case to avoid
    /// crashing, but the result is meaningless.
    public var colorSpace: MetalForgeColorSpace = .hdr10PQ

    private let pqPSO: MTLComputePipelineState
    private let hlgPSO: MTLComputePipelineState

    public init(engine: MetalForgeEngine) throws {
        pqPSO  = try Self.makePSO(engine: engine, kernel: "pqDecodeKernel")
        hlgPSO = try Self.makePSO(engine: engine, kernel: "hlgDecodeKernel")
    }

    private static func makePSO(
        engine: MetalForgeEngine,
        kernel: String
    ) throws -> MTLComputePipelineState {
        let function = try engine.makeFunction(name: kernel)
        do {
            return try engine.device.makeComputePipelineState(function: function)
        } catch {
            throw MetalForgeError.pipelineStateCreationFailed(error.localizedDescription)
        }
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        let pso = (colorSpace == .hlg) ? hlgPSO : pqPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "HDRDecodeFilter(\(colorSpace == .hlg ? "HLG" : "PQ"))"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)

        // SIMD-aligned threadgroup shape; non-uniform dispatch covers exact dims.
        let simdWidth = pso.threadExecutionWidth
        let groupH    = pso.maxTotalThreadsPerThreadgroup / simdWidth
        encoder.dispatchThreads(
            MTLSize(width: destination.width, height: destination.height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: simdWidth, height: groupH, depth: 1)
        )
        encoder.endEncoding()
    }
}

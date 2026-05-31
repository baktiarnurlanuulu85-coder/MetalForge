import Metal
import Foundation

/// Applies the forward transfer function (OETF) for an HDR signal, packing
/// linear-light pixel values back into a PQ or HLG-encoded representation
/// suitable for storage, transmission, or display.
///
/// ## What it does
/// Implements the exact BT.2100 OETFs:
/// - **PQ (`.hdr10PQ`)**: SMPTE ST.2084 OETF. Expects input `1.0 ≡ 10,000 cd/m²`.
///   Output is the PQ-encoded `[0, 1]` range ready for `.bgr10a2Unorm` /
///   `.rgba16Float` display surfaces with `CGColorSpace.itur_2100_PQ`.
/// - **HLG (`.hlg`)**: BT.2100 OETF (per channel). Scene-relative input;
///   no OOTF is applied — same reasoning as in `HDRDecodeFilter`.
///
/// `HDREncodeFilter(decoded)` and `HDRDecodeFilter(encoded)` round-trip exactly,
/// modulo float-precision noise.
///
/// ## Usage
/// Place this filter at the *very end* of the chain, after any linear-light
/// effects, so the output is display-ready non-linear values.
///
/// ## Thread Safety
/// `@unchecked Sendable` for the same reasons as `HDRDecodeFilter`.
public final class HDREncodeFilter: @unchecked Sendable, MetalForgeFilter {

    public var colorSpace: MetalForgeColorSpace = .hdr10PQ

    private let pqPSO: MTLComputePipelineState
    private let hlgPSO: MTLComputePipelineState

    public init(engine: MetalForgeEngine) throws {
        pqPSO  = try Self.makePSO(engine: engine, kernel: "pqEncodeKernel")
        hlgPSO = try Self.makePSO(engine: engine, kernel: "hlgEncodeKernel")
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
        encoder.label = "HDREncodeFilter(\(colorSpace == .hlg ? "HLG" : "PQ"))"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)

        let simdWidth = pso.threadExecutionWidth
        let groupH    = pso.maxTotalThreadsPerThreadgroup / simdWidth
        encoder.dispatchThreads(
            MTLSize(width: destination.width, height: destination.height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: simdWidth, height: groupH, depth: 1)
        )
        encoder.endEncoding()
    }
}

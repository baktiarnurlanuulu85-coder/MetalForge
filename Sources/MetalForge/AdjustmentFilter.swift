import Metal
import Foundation

/// Uniform layout — must match `AdjustmentUniforms` in `AdjustmentShader.metal`.
private struct AdjustmentUniforms {
    /// Additive brightness offset: -1.0 (black) … 0.0 (no change) … 1.0 (white)
    var brightness: Float
    /// Contrast multiplier around mid-gray: 0.0 (flat grey) … 1.0 (identity) … 4.0 (crushed)
    var contrast: Float
}

/// Applies a per-pixel brightness and contrast adjustment using a compute kernel.
///
/// The filter manages **two** specialised `MTLComputePipelineState` objects, one
/// compiled with `isHDR = false` (final clamp to [0,1]) and one with `isHDR = true`
/// (no clamp). At `encode` time the variant is chosen automatically based on the
/// `source.pixelFormat` — `.rgba16Float` selects the HDR PSO, everything else
/// selects the SDR PSO.
///
/// Configure `brightness` and `contrast` between frames. Reads happen inside
/// `encode`; concurrent writes during an in-flight encode require external sync.
// @unchecked Sendable: mutable configuration properties (brightness, contrast)
// are expected to be written by the caller before the pipeline processes a frame.
public final class AdjustmentFilter: @unchecked Sendable, MetalForgeFilter {

    /// Brightness offset in [-1, 1]. Default: `0` (no change).
    public var brightness: Float = 0.0

    /// Contrast scale around 0.5. Default: `1.0` (no change).
    public var contrast: Float = 1.0

    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    public init(engine: MetalForgeEngine) throws {
        sdrPSO = try Self.makePSO(engine: engine, isHDR: false)
        hdrPSO = try Self.makePSO(engine: engine, isHDR: true)
    }

    /// Compile a specialised PSO with a single `bool` function constant at slot 0.
    private static func makePSO(
        engine: MetalForgeEngine,
        isHDR: Bool
    ) throws -> MTLComputePipelineState {
        // MTLFunctionConstantValues carries the values for [[function_constant(N)]]
        // declarations the function references. The constant is *baked into AIR*
        // when makeFunction(constantValues:) is called, so the resulting MTLFunction
        // is a fully specialised variant with the `if (!isHDR)` branch eliminated.
        let constants = MTLFunctionConstantValues()
        var flag = isHDR
        constants.setConstantValue(&flag, type: .bool, index: 0)

        let function = try engine.makeFunction(
            name: "adjustmentKernel",
            constantValues: constants
        )

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
        // .rgba16Float is the HDR working format chosen by MetalForgePipeline for
        // 10-bit YUV input. Anything else (bgra8Unorm, etc.) is treated as SDR.
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "AdjustmentFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)

        var uniforms = AdjustmentUniforms(brightness: brightness, contrast: contrast)
        encoder.setBytes(&uniforms, length: MemoryLayout<AdjustmentUniforms>.stride, index: 0)

        // --- Threadgroup sizing on the *selected* PSO ---
        // threadExecutionWidth = SIMD lane width (32 on Apple GPU).
        // Both PSO variants will report identical values in practice — the
        // function constant doesn't affect occupancy — but reading off the PSO
        // we actually dispatch is defensive against future shader changes.
        let simdWidth = pso.threadExecutionWidth
        let groupH    = pso.maxTotalThreadsPerThreadgroup / simdWidth
        let groupSize = MTLSize(width: simdWidth, height: groupH, depth: 1)

        let gridSize = MTLSize(width: destination.width, height: destination.height, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
        encoder.endEncoding()
    }
}

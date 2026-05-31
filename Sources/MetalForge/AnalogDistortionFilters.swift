import Metal
import simd
import Foundation

// ===========================================================================
// AnalogDistortionPack — three filters that share AnalogKernels.metal.
//
// Each filter follows the established MetalForge pattern:
//   • Conforms to MetalForgeFilter with the (source, destination, commandBuffer)
//     encode signature.
//   • Pre-compiles two specialised MTLComputePipelineState instances via the
//     `isHDR` function constant — selected at encode time by the source
//     texture's pixel format.
//   • Uses non-uniform `dispatchThreads` with the SIMD-aligned threadgroup
//     shape (width = threadExecutionWidth, height = remaining occupancy).
// ===========================================================================

// MARK: - Shared PSO compilation

/// Compile one specialised PSO for an Analog-pack kernel.
/// Factored out so all three filters share the function-constant + error-wrapping logic.
private func makeAnalogPSO(
    engine: MetalForgeEngine,
    kernel: String,
    isHDR: Bool
) throws -> MTLComputePipelineState {
    let constants = MTLFunctionConstantValues()
    var flag = isHDR
    constants.setConstantValue(&flag, type: .bool, index: 0)

    let function = try engine.makeFunction(name: kernel, constantValues: constants)
    do {
        return try engine.device.makeComputePipelineState(function: function)
    } catch {
        throw MetalForgeError.pipelineStateCreationFailed(error.localizedDescription)
    }
}

/// Shared threadgroup dispatch used by all three Analog filters.
/// Apple GPU SIMD width is 32; the function constant doesn't change occupancy,
/// so both SDR and HDR PSO variants report the same execution width.
private func dispatchAnalog(
    encoder: MTLComputeCommandEncoder,
    pso: MTLComputePipelineState,
    width: Int,
    height: Int
) {
    let simdWidth = pso.threadExecutionWidth
    let groupH    = pso.maxTotalThreadsPerThreadgroup / simdWidth
    encoder.dispatchThreads(
        MTLSize(width: width, height: height, depth: 1),
        threadsPerThreadgroup: MTLSize(width: simdWidth, height: groupH, depth: 1)
    )
}

// ===========================================================================
// 1. ChromaticAberrationFilter
// ===========================================================================

/// Uniform layout — must mirror `ChromaticAberrationUniforms` in MSL.
/// Stride: 16 bytes (two `float2` columns, 8 bytes each).
private struct ChromaticAberrationUniforms {
    var redShift:   SIMD2<Float>
    var greenShift: SIMD2<Float>
}

/// Per-channel chromatic dispersion: R sampled at `uv + redShift`, G at
/// `uv + greenShift`, B at centre. The shifts are in normalised UV space
/// (i.e. `0.01` ≈ 1 % of frame width). Typical values: 0.002 … 0.02.
///
/// Setting both shifts to zero is a pass-through.
public final class ChromaticAberrationFilter: @unchecked Sendable, MetalForgeFilter {

    /// Offset applied to the R-channel sample. Default `(0.005, 0)` ≈ subtle horizontal red fringe.
    public var redShift: SIMD2<Float> = SIMD2(0.005, 0.0)

    /// Offset applied to the G-channel sample. Default `(-0.005, 0)` for symmetric dispersion.
    public var greenShift: SIMD2<Float> = SIMD2(-0.005, 0.0)

    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    public init(engine: MetalForgeEngine) throws {
        sdrPSO = try makeAnalogPSO(engine: engine,
                                   kernel: "chromaticAberrationKernel", isHDR: false)
        hdrPSO = try makeAnalogPSO(engine: engine,
                                   kernel: "chromaticAberrationKernel", isHDR: true)
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        // Select the function-constant-specialised PSO by destination format.
        // .rgba16Float is the marker for the HDR working space.
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ChromaticAberrationFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)

        var uniforms = ChromaticAberrationUniforms(redShift: redShift, greenShift: greenShift)
        encoder.setBytes(
            &uniforms,
            length: MemoryLayout<ChromaticAberrationUniforms>.stride,
            index: 0
        )

        dispatchAnalog(encoder: encoder, pso: pso,
                       width: destination.width, height: destination.height)
        encoder.endEncoding()
    }
}

// ===========================================================================
// 2. AnalogNoiseFilter
// ===========================================================================

private struct NoiseUniforms {
    var intensity: Float
    var timeSeed:  Float
}

/// Additive mean-zero grain. `noiseIntensity` controls the amplitude of the
/// per-pixel additive offset (sensible range 0 … 0.5; 0 is bypass).
/// `timeSeed` is added on top of an internal auto-advancing time so the grain
/// flickers naturally between frames — set it to a constant to "freeze" the
/// noise pattern.
public final class AnalogNoiseFilter: @unchecked Sendable, MetalForgeFilter {

    public var noiseIntensity: Float = 0.15
    /// Additional time offset added to the internal frame-clock. Default 0
    /// means pure auto-animation. Set to a fixed value (e.g. -elapsed) to
    /// freeze the noise pattern, or use it as a seed if you want a different
    /// per-instance phase for stacked grain.
    public var timeSeed: Float = 0.0

    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState
    /// Internal time anchor — drives the temporal evolution of the noise.
    private let startDate = Date()

    public init(engine: MetalForgeEngine) throws {
        sdrPSO = try makeAnalogPSO(engine: engine,
                                   kernel: "analogNoiseKernel", isHDR: false)
        hdrPSO = try makeAnalogPSO(engine: engine,
                                   kernel: "analogNoiseKernel", isHDR: true)
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "AnalogNoiseFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)

        // Internal elapsed time + user-provided seed offset.
        let elapsed = Float(Date().timeIntervalSince(startDate))
        var uniforms = NoiseUniforms(
            intensity: noiseIntensity,
            timeSeed:  elapsed + timeSeed
        )
        encoder.setBytes(&uniforms, length: MemoryLayout<NoiseUniforms>.stride, index: 0)

        dispatchAnalog(encoder: encoder, pso: pso,
                       width: destination.width, height: destination.height)
        encoder.endEncoding()
    }
}

// ===========================================================================
// 3. HorizontalJitterFilter
// ===========================================================================

private struct JitterUniforms {
    var intensity: Float
    var timeSeed:  Float
}

/// Per-row horizontal displacement. `jitterIntensity` is the max absolute
/// offset in normalised UV (0 … ~0.1; 0.05 ≈ 5 % of frame width). The
/// displacement is randomised per row per frame; combined with VCR-style hue
/// shifts (via `ChromaticAberrationFilter`) you get a convincing dead-tape look.
public final class HorizontalJitterFilter: @unchecked Sendable, MetalForgeFilter {

    public var jitterIntensity: Float = 0.02
    /// Same semantics as `AnalogNoiseFilter.timeSeed` — offset added to the
    /// internal frame clock.
    public var timeSeed: Float = 0.0

    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState
    private let startDate = Date()

    public init(engine: MetalForgeEngine) throws {
        sdrPSO = try makeAnalogPSO(engine: engine,
                                   kernel: "horizontalJitterKernel", isHDR: false)
        hdrPSO = try makeAnalogPSO(engine: engine,
                                   kernel: "horizontalJitterKernel", isHDR: true)
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "HorizontalJitterFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)

        let elapsed = Float(Date().timeIntervalSince(startDate))
        var uniforms = JitterUniforms(
            intensity: jitterIntensity,
            timeSeed:  elapsed + timeSeed
        )
        encoder.setBytes(&uniforms, length: MemoryLayout<JitterUniforms>.stride, index: 0)

        dispatchAnalog(encoder: encoder, pso: pso,
                       width: destination.width, height: destination.height)
        encoder.endEncoding()
    }
}

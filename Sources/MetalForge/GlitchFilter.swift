import Metal
import Foundation

/// Uniform layout — must match `GlitchUniforms` in `GlitchShader.metal`.
private struct GlitchUniforms {
    /// Elapsed seconds since filter creation; drives all procedural noise variation.
    var time: Float
    /// Effect strength: 0.0 = bypass, 1.0 = maximum glitch. Default: 0.5.
    var intensity: Float
    /// Monotonically increasing frame counter; seeds per-frame hash variation.
    var frameIndex: UInt32
    // 4 bytes of implicit padding here to satisfy Metal's alignment rules —
    // the struct is 12 bytes, which is already 4-byte aligned; no extra needed.
}

/// Applies a **Cinematic Glitch + Chromatic Aberration** effect.
///
/// ## Effect breakdown
/// - **Horizontal glitch bands**: the image is divided into ~24 independent horizontal
///   bands. Each band has a per-frame probability of "firing" (sliding horizontally).
///   Higher `intensity` raises the probability and the maximum displacement.
/// - **Chromatic aberration**: the R and B channels are sampled at slightly offset UVs
///   in opposite horizontal directions, recreating the look of a lens with uncorrected
///   lateral chromatic aberration or an analog signal with colour channel skew.
/// - **CRT scanlines**: a subtle periodic darkening every few rows modelling a CRT
///   phosphor line structure, computed mathematically (no texture lookup).
/// - **Vignette**: soft edge darkening, scaled by `intensity`.
/// - **Noise flash**: random whole-frame signal corruption spikes.
///
/// All effects are entirely procedural — no lookup textures required.
// @unchecked Sendable: `intensity` is a configuration knob written before frame
// processing begins. `frameIndex` is only mutated inside `encode`, which the
// pipeline calls from a single serial queue. No concurrent mutation occurs in
// normal use; document this constraint for integrators.
public final class GlitchFilter: @unchecked Sendable, MetalForgeFilter {

    /// Effect strength in [0, 1]. Thread-safe to write between frames.
    public var intensity: Float = 0.5

    // Two specialised PSOs — SDR variant clamps the final color to [0,1],
    // HDR variant preserves values above 1.0 for .rgba16Float targets.
    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    // frameIndex uses wrapping arithmetic (&+=) so overflow is safe.
    private var frameIndex: UInt32 = 0
    private let startDate = Date()

    public init(engine: MetalForgeEngine) throws {
        let library = try engine.device.makeDefaultLibrary(bundle: Bundle.module)
        sdrPSO = try Self.makePSO(library: library, device: engine.device, isHDR: false)
        hdrPSO = try Self.makePSO(library: library, device: engine.device, isHDR: true)
    }

    private static func makePSO(
        library: MTLLibrary,
        device: MTLDevice,
        isHDR: Bool
    ) throws -> MTLComputePipelineState {
        let constants = MTLFunctionConstantValues()
        var flag = isHDR
        constants.setConstantValue(&flag, type: .bool, index: 0)

        let function: MTLFunction
        do {
            function = try library.makeFunction(
                name: "glitchKernel",
                constantValues: constants
            )
        } catch {
            throw MetalForgeError.shaderFunctionNotFound("glitchKernel")
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw MetalForgeError.pipelineStateCreationFailed(error.localizedDescription)
        }
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        // Select PSO by destination format — .rgba16Float means we're in the HDR
        // working space and must not clamp highlights.
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "GlitchFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)

        var uniforms = GlitchUniforms(
            time:       Float(Date().timeIntervalSince(startDate)),
            intensity:  intensity,
            frameIndex: frameIndex
        )
        encoder.setBytes(&uniforms, length: MemoryLayout<GlitchUniforms>.stride, index: 0)

        // &+= wraps safely at UInt32.max without trapping in release or debug builds.
        frameIndex &+= 1

        let simdWidth = pso.threadExecutionWidth
        let groupH    = pso.maxTotalThreadsPerThreadgroup / simdWidth
        encoder.dispatchThreads(
            MTLSize(width: destination.width, height: destination.height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: simdWidth, height: groupH, depth: 1)
        )
        encoder.endEncoding()
    }
}

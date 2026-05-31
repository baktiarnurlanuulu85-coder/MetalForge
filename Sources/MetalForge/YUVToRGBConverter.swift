@preconcurrency import Metal
import simd
import Foundation

/// Uniform layout — must mirror `YUVConverterUniforms` in `YUVConverterShader.metal`.
///
/// Layout (Swift / MSL both):
/// ```
/// offset  0 .. 47   : float3x3 colorMatrix  (3 cols × 16 B per column)
/// offset 48 .. 51   : uint     isFullRange
/// offset 52 .. 63   : trailing padding to 16-byte alignment
/// total stride       : 64 bytes
/// ```
private struct YUVConverterUniforms {
    var colorMatrix: matrix_float3x3
    var isFullRange: UInt32
}

/// Source-stage filter that converts a YUV bi-planar input into an RGB
/// destination texture using a compute kernel.
///
/// ## How to use
/// ```swift
/// let engine    = try MetalForgeEngine()
/// let converter = try YUVToRGBConverter(engine: engine)
/// let pool      = TexturePool(device: engine.device)
///
/// // From a CMSampleBuffer:
/// guard let planes = engine.makeTextures(from: pixelBuffer, colorSpace: .sdr) else { return }
/// let (matrix, isFull) = YUVColorMatrices.matrix(for: pixelBuffer)
/// let dest = pool.acquire(width:  planes.luma.width,
///                         height: planes.luma.height,
///                         pixelFormat: .bgra8Unorm)!
///
/// let cmd = engine.commandQueue.makeCommandBuffer()!
/// converter.encode(luma: planes.luma,
///                  chroma: planes.chroma,
///                  destination: dest,
///                  commandBuffer: cmd,
///                  matrix: matrix,
///                  rangeFlg: isFull)
/// cmd.commit()
/// ```
///
/// Thread safety: the converter holds only an immutable `MTLComputePipelineState`,
/// so it is fully `Sendable` and safe to share across processing queues.
public final class YUVToRGBConverter: MetalForgeSourceFilter {

    private let pipelineState: MTLComputePipelineState

    public init(engine: MetalForgeEngine) throws {
        let library = try engine.device.makeDefaultLibrary(bundle: Bundle.module)
        guard let function = library.makeFunction(name: "yuvToRgbCompute") else {
            throw MetalForgeError.shaderFunctionNotFound("yuvToRgbCompute")
        }
        do {
            pipelineState = try engine.device.makeComputePipelineState(function: function)
        } catch {
            throw MetalForgeError.pipelineStateCreationFailed(error.localizedDescription)
        }
    }

    public func encode(
        luma: MTLTexture,
        chroma: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer,
        matrix: matrix_float3x3,
        rangeFlg: Bool
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "YUVToRGBConverter"
        encoder.setComputePipelineState(pipelineState)
        encoder.setTexture(luma,        index: 0)
        encoder.setTexture(chroma,      index: 1)
        encoder.setTexture(destination, index: 2)

        var uniforms = YUVConverterUniforms(
            colorMatrix: matrix,
            isFullRange: rangeFlg ? 1 : 0
        )
        encoder.setBytes(
            &uniforms,
            length: MemoryLayout<YUVConverterUniforms>.stride,
            index: 0
        )

        // --- Threadgroup math for Apple GPU (SIMD width = 32) ---
        // threadExecutionWidth tells us the SIMD lane count (32 on Apple GPU).
        // maxTotalThreadsPerThreadgroup divided by lane count gives the optimal
        // 2-D shape: a wide+short threadgroup that fills the warp on every issue.
        // Typical result on A14+: width = 32, height = 32 → 1024 threads/group.
        let simdWidth   = pipelineState.threadExecutionWidth
        let groupHeight = pipelineState.maxTotalThreadsPerThreadgroup / simdWidth
        let threadsPerGroup = MTLSize(width: simdWidth, height: groupHeight, depth: 1)

        // dispatchThreads (non-uniform) launches exactly destination.width ×
        // destination.height threads. The shader still needs its bounds guard
        // because the *hardware* always issues whole threadgroups; the runtime
        // simply masks out-of-grid threads from completion, but they still
        // execute the program — hence the in-shader if (gid >= …) check.
        let threadsPerGrid = MTLSize(
            width:  destination.width,
            height: destination.height,
            depth:  1
        )
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
    }
}

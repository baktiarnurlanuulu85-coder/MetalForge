import Metal
import simd
import Foundation

// ===========================================================================
// ColorGradingPack — two filters that share ColorGradingKernels.metal:
//
//   • MetalForgeLUTFilter    — 3D LUT (typically 32³) sampled per pixel via
//                              Metal's hardware trilinear interpolation.
//   • ColorCorrectionFilter  — exposure / contrast (0.18 pivot) / saturation
//                              / temperature, all in linear-light math.
// ===========================================================================

// MARK: - Shared PSO compilation helper

/// Compile one specialised PSO for a ColorGrading kernel.
private func makeColorGradingPSO(
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

/// Shared SIMD-aligned threadgroup dispatch (same pattern as every other
/// MetalForge filter).
private func dispatchColorGrading(
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
// 1. MetalForgeLUTFilter
// ===========================================================================

/// Uniform layout — must match `LUTUniforms` in `ColorGradingKernels.metal`.
private struct LUTUniforms {
    var intensity: Float
    var lutSize:   Float
}

/// Per-pixel 3D LUT colour grading.
///
/// The LUT is a cube texture (`textureType = .type3D`) typically `32³` or `16³`
/// in `.rgba8Unorm` storage. Each pixel's RGB value is treated as a 3D
/// coordinate into the cube, and the hardware trilinear sampler interpolates
/// the 8 surrounding cube vertices for a smooth grading response.
///
/// ## HDR handling
/// Working space `.rgba16Float` can carry values above 1.0. The shader splits
/// the source into a `base` (clamped to `[0, 1]`) and a `highlight` (the
/// above-white residual), samples the LUT only on `base`, and adds the
/// highlight back unchanged — so HDR tonality is preserved while the
/// authored-for-SDR LUT cube does its grading on the visible range.
public final class MetalForgeLUTFilter: @unchecked Sendable, MetalForgeFilter {

    /// Built-in cube generators usable without external asset files. The demo
    /// app uses `.warm` so it can compile and run with zero resources.
    public enum PresetLUT: String, CaseIterable, Sendable {
        case identity   // pass-through cube; useful for testing intensity blending
        case warm       // boost R, slightly reduce B (incandescent / fire look)
        case cool       // boost B, slightly reduce R (mercury / moonlight look)
        case sepia      // luma-driven monochrome with warm tint
    }

    /// Edge length of the cube — read-only, fixed by init.
    public let lutSize: Int

    /// Blend between identity (`0`) and full LUT effect (`1`). Sensible range
    /// `[0, 1]`. Default `1.0` = full effect.
    public var intensity: Float = 1.0

    private let lutTexture: MTLTexture
    private let sdrPSO:    MTLComputePipelineState
    private let hdrPSO:    MTLComputePipelineState

    // MARK: Init

    /// Construct with a raw cube data buffer.
    ///
    /// - Parameters:
    ///   - lutSize: Edge length of the cube (e.g. 32 for a 32³ LUT).
    ///   - lutData: Exactly `lutSize × lutSize × lutSize × 4` bytes of
    ///     `.rgba8Unorm` data in `B → G → R` outer-to-inner ordering (Metal's
    ///     natural 3D-texture layout).
    public init(engine: MetalForgeEngine, lutSize: Int, lutData: Data) throws {
        self.lutSize = lutSize

        // ----- Validate buffer size -----
        let expectedBytes = lutSize * lutSize * lutSize * 4
        guard lutData.count == expectedBytes else {
            throw MetalForgeError.pipelineStateCreationFailed(
                "LUT data size mismatch: expected \(expectedBytes) bytes for \(lutSize)³ cube, got \(lutData.count)"
            )
        }

        // ----- Allocate the 3D texture -----
        let desc = MTLTextureDescriptor()
        desc.textureType = .type3D
        desc.pixelFormat = .rgba8Unorm
        desc.width       = lutSize
        desc.height      = lutSize
        desc.depth       = lutSize
        desc.usage       = .shaderRead
        // .shared on Apple-Silicon-only platforms (iOS / visionOS / tvOS) —
        // unified memory, no CPU↔GPU sync cost. On macOS we use .managed so
        // the code runs unchanged on Intel Macs too; Metal handles the sync
        // automatically through `replace()`.
        #if os(macOS)
        desc.storageMode = .managed
        #else
        desc.storageMode = .shared
        #endif

        guard let texture = engine.device.makeTexture(descriptor: desc) else {
            throw MetalForgeError.textureAllocationFailed
        }

        // ----- Upload cube data -----
        // 3D texture replace requires both bytesPerRow (one 2D row) and
        // bytesPerImage (one Z-slice). The source data must be laid out
        // contiguously in B → G → R order.
        let bytesPerRow   = lutSize * 4
        let bytesPerImage = bytesPerRow * lutSize
        let region        = MTLRegionMake3D(0, 0, 0, lutSize, lutSize, lutSize)
        lutData.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else { return }
            texture.replace(
                region:         region,
                mipmapLevel:    0,
                slice:          0,
                withBytes:      baseAddress,
                bytesPerRow:    bytesPerRow,
                bytesPerImage:  bytesPerImage
            )
        }

        self.lutTexture = texture

        // ----- Compile both SDR and HDR PSO variants -----
        self.sdrPSO = try makeColorGradingPSO(
            engine: engine,
            kernel: "lut3DColorKernel", isHDR: false)
        self.hdrPSO = try makeColorGradingPSO(
            engine: engine,
            kernel: "lut3DColorKernel", isHDR: true)
    }

    /// Convenience initialiser using one of the built-in preset cubes.
    /// No external asset files required.
    public convenience init(
        engine: MetalForgeEngine,
        preset: PresetLUT = .identity,
        size: Int = 32
    ) throws {
        let data = Self.makePresetLUTData(preset: preset, size: size)
        try self.init(engine: engine, lutSize: size, lutData: data)
    }

    // MARK: Preset generators

    /// Generate one of the built-in preset LUT cubes as raw `.rgba8Unorm` data.
    /// Layout follows the convention required by `init(engine:lutSize:lutData:)`:
    /// outer B → middle G → inner R.
    public static func makePresetLUTData(preset: PresetLUT, size: Int) -> Data {
        precondition(size >= 2, "LUT size must be at least 2.")
        var bytes = [UInt8](repeating: 0, count: size * size * size * 4)
        var idx = 0

        // Triple-nested loop: B (slowest, slice index) → G (row) → R (column).
        // Generating in this order matches Metal's 3D texture memory layout
        // exactly, so the buffer can be uploaded via `replace()` directly.
        for b in 0..<size {
            for g in 0..<size {
                for r in 0..<size {
                    // Normalise indices to [0, 1].
                    let rN = Float(r) / Float(size - 1)
                    let gN = Float(g) / Float(size - 1)
                    let bN = Float(b) / Float(size - 1)

                    let rOut: Float, gOut: Float, bOut: Float
                    switch preset {
                    case .identity:
                        // Pass-through cube. lut(rgb) == rgb. Useful for
                        // testing intensity blending isolation.
                        rOut = rN
                        gOut = gN
                        bOut = bN

                    case .warm:
                        // Push reds toward white, dim blues — incandescent /
                        // golden-hour look. Coefficients tuned for visible
                        // but not overpowering effect.
                        rOut = min(rN + (1.0 - rN) * 0.15, 1.0)
                        gOut = gN
                        bOut = max(bN * 0.85, 0.0)

                    case .cool:
                        // Symmetric to .warm — push blues, dim reds.
                        rOut = max(rN * 0.85, 0.0)
                        gOut = gN
                        bOut = min(bN + (1.0 - bN) * 0.15, 1.0)

                    case .sepia:
                        // Classical sepia: collapse to luminance, then re-tint.
                        // BT.601 weights (close enough for this look) keep the
                        // brightness perceptually plausible.
                        let luma = 0.299 * rN + 0.587 * gN + 0.114 * bN
                        rOut = min(luma * 1.10, 1.0)
                        gOut = min(luma * 0.95, 1.0)
                        bOut = max(luma * 0.70, 0.0)
                    }

                    bytes[idx + 0] = UInt8((rOut * 255.0).rounded())
                    bytes[idx + 1] = UInt8((gOut * 255.0).rounded())
                    bytes[idx + 2] = UInt8((bOut * 255.0).rounded())
                    bytes[idx + 3] = 255
                    idx += 4
                }
            }
        }
        return Data(bytes)
    }

    // MARK: Encode

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "MetalForgeLUTFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)
        encoder.setTexture(lutTexture,  index: 2)

        var uniforms = LUTUniforms(intensity: intensity, lutSize: Float(lutSize))
        encoder.setBytes(&uniforms, length: MemoryLayout<LUTUniforms>.stride, index: 0)

        dispatchColorGrading(
            encoder: encoder, pso: pso,
            width: destination.width, height: destination.height
        )
        encoder.endEncoding()
    }
}

// ===========================================================================
// 2. ColorCorrectionFilter
// ===========================================================================

/// Uniform layout — must match `ColorCorrectionUniforms` in MSL.
private struct ColorCorrectionUniforms {
    var exposure:         Float
    var contrast:         Float
    var saturation:       Float
    var temperatureShift: Float
}

/// Linear-light colour correction. Each parameter has a clearly-defined
/// identity value — set all to identity and the filter is a no-op:
///
///   - `exposure`         = 0       (stops; identity)
///   - `contrast`         = 1.0     (multiplier around 0.18; identity)
///   - `saturation`       = 1.0     (1.0 = original; 0 = grey; >1 = boosted)
///   - `temperatureShift` = 0       (-1 = cool, +1 = warm)
///
/// Contrast pivot is **0.18** (scene-referred middle grey), not 0.5. See the
/// MSL comments for why.
public final class ColorCorrectionFilter: @unchecked Sendable, MetalForgeFilter {

    public var exposure:         Float = 0.0
    public var contrast:         Float = 1.0
    public var saturation:       Float = 1.0
    public var temperatureShift: Float = 0.0

    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    public init(engine: MetalForgeEngine) throws {
        self.sdrPSO = try makeColorGradingPSO(
            engine: engine,
            kernel: "colorCorrectionKernel", isHDR: false)
        self.hdrPSO = try makeColorGradingPSO(
            engine: engine,
            kernel: "colorCorrectionKernel", isHDR: true)
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ColorCorrectionFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)

        var uniforms = ColorCorrectionUniforms(
            exposure:         exposure,
            contrast:         contrast,
            saturation:       saturation,
            temperatureShift: temperatureShift
        )
        encoder.setBytes(
            &uniforms,
            length: MemoryLayout<ColorCorrectionUniforms>.stride,
            index: 0
        )

        dispatchColorGrading(
            encoder: encoder, pso: pso,
            width: destination.width, height: destination.height
        )
        encoder.endEncoding()
    }
}

import Metal
import simd
import Foundation

// ===========================================================================
// TemporalEffectsPack — frame-to-frame effects.
//
// Each filter owns a persistent `MTLTexture` that mirrors this frame's output
// for use as "previous frame" data in the next call. Texture lifecycle:
//
//   1. encode(): if no buffer exists (or its dims/format don't match source),
//                allocate a new .private MTLTexture matching source. Seed it
//                with a blit copy of source so the first frame's result == source.
//   2. encode(): kernel reads source + previousBuffer, writes destination.
//   3. encode(): blit destination → previousBuffer for next frame.
//   4. clearHistory(): drop the buffer reference (next encode re-seeds).
//   5. deinit: ARC releases the buffer.
//
// Concurrency: encode() runs on the pipeline's processing queue (serialised).
// clearHistory() may be called from any thread (typically main, when the user
// switches filters in the demo). Both serialise through `bufferLock`. The blit
// inside encode uses a *local snapshot* of the buffer, so even if clearHistory
// nils the property mid-encode, ARC keeps the in-flight texture alive until
// the command buffer completes.
// ===========================================================================

// MARK: - Shared infrastructure

/// Common base helpers — DRY for the two temporal filters' PSO compilation
/// and persistent-buffer plumbing.
private enum TemporalSharedHelpers {

    /// Compile one isHDR-specialised PSO for a temporal kernel.
    static func makePSO(
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

    /// Allocate a private-storage texture matching another's dims and format.
    /// `.shaderRead` for kernel sampling; blit destination doesn't need a
    /// usage flag, but `.shaderWrite` is included so the same texture could
    /// be used as a compute output if a future kernel needs it.
    static func makePersistentTexture(
        device: MTLDevice,
        matching reference: MTLTexture
    ) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: reference.pixelFormat,
            width:       reference.width,
            height:      reference.height,
            mipmapped:   false
        )
        desc.storageMode = .private
        desc.usage       = [.shaderRead, .shaderWrite]
        return device.makeTexture(descriptor: desc)
    }

    /// Threadgroup dispatch shape — same SIMD-aligned pattern as every other
    /// MetalForge filter.
    static func dispatch(
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
}

// ===========================================================================
// 1. MotionBlurFilter
// ===========================================================================

private struct MotionBlurUniforms {
    var accumulationAlpha: Float
}

/// Exponential temporal smoothing. `accumulationAlpha`:
///   - `1.0` → no blur, output == source.
///   - `0.5` → equal blend with previous frame.
///   - `0.1` → very heavy persistence (10-frame half-life).
///   - `0.0` → freeze: output stays at first seeded frame forever.
public final class MotionBlurFilter: @unchecked Sendable, MetalForgeFilter {

    /// Blend weight of the current frame against the trail buffer.
    /// Sensible range: `[0.05, 1.0]`. Default `0.5` ≈ 1-frame motion smoothing.
    public var accumulationAlpha: Float = 0.5

    private let device: MTLDevice
    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    /// Lock guards `previousTexture` and its associated `previousFormat`/`previousSize`.
    /// Held only across stored-property reads/writes, never across GPU encodes
    /// (those use a local snapshot of the reference).
    private let bufferLock = NSLock()
    private var previousTexture: MTLTexture?
    private var previousFormat:  MTLPixelFormat = .invalid
    private var previousWidth:   Int = 0
    private var previousHeight:  Int = 0

    public init(engine: MetalForgeEngine) throws {
        device = engine.device
        sdrPSO = try TemporalSharedHelpers.makePSO(
            engine: engine, kernel: "motionBlurKernel", isHDR: false)
        hdrPSO = try TemporalSharedHelpers.makePSO(
            engine: engine, kernel: "motionBlurKernel", isHDR: true)
    }

    /// Drop the persistent previous-frame buffer. The next `encode` call will
    /// re-allocate and re-seed it from the source — so output == source for
    /// that single frame, then accumulation restarts.
    ///
    /// Call this when:
    ///   • The user switches away from this filter (so re-selecting it doesn't
    ///     "leak" old frames from before).
    ///   • The capture resolution or colour space is about to change.
    public func clearHistory() {
        bufferLock.lock()
        previousTexture = nil
        previousFormat  = .invalid
        previousWidth   = 0
        previousHeight  = 0
        bufferLock.unlock()
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        // ----- Acquire / re-acquire persistent buffer under the lock -----
        bufferLock.lock()
        let needsAlloc =
            previousTexture == nil
            || previousFormat != source.pixelFormat
            || previousWidth  != source.width
            || previousHeight != source.height

        if needsAlloc {
            previousTexture = TemporalSharedHelpers.makePersistentTexture(
                device: device, matching: source)
            previousFormat  = source.pixelFormat
            previousWidth   = source.width
            previousHeight  = source.height
        }
        // Local snapshot — used for the GPU work even if `clearHistory` runs
        // mid-encode and nils the property.
        let prev = previousTexture
        bufferLock.unlock()

        guard let prev else { return }

        // ----- Seed buffer on first frame (or after a reallocation) -----
        // Blit the source into the freshly-allocated previous so the kernel's
        // mix() produces source==source==source for this single frame. Without
        // the seed, the kernel would read uninitialised .private memory.
        if needsAlloc {
            if let seedBlit = commandBuffer.makeBlitCommandEncoder() {
                seedBlit.label = "MotionBlurFilter.seedPrevious"
                seedBlit.copy(from: source, to: prev)
                seedBlit.endEncoding()
            }
        }

        // ----- Compute pass: mix(prev, curr, alpha) → destination -----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "MotionBlurFilter"
            enc.setComputePipelineState(pso)
            enc.setTexture(source,      index: 0)
            enc.setTexture(prev,        index: 1)
            enc.setTexture(destination, index: 2)

            var u = MotionBlurUniforms(accumulationAlpha: accumulationAlpha)
            enc.setBytes(&u, length: MemoryLayout<MotionBlurUniforms>.stride, index: 0)

            TemporalSharedHelpers.dispatch(
                encoder: enc, pso: pso,
                width: destination.width, height: destination.height
            )
            enc.endEncoding()
        }

        // ----- Save destination → previousTexture for next frame -----
        // Same command buffer, so this blit is serialised after the compute
        // pass by Metal's intra-buffer ordering guarantees. No fences needed.
        if let saveBlit = commandBuffer.makeBlitCommandEncoder() {
            saveBlit.label = "MotionBlurFilter.savePrevious"
            saveBlit.copy(from: destination, to: prev)
            saveBlit.endEncoding()
        }
    }
}

// ===========================================================================
// 2. NeonTrailsFilter
// ===========================================================================

private struct NeonTrailsUniforms {
    var neonColor: SIMD3<Float>   // 16-byte aligned (3 floats + 4 pad)
    var intensity: Float
    var decay:     Float
}

/// Motion-driven glow that leaves persistent trails behind moving subjects.
/// `intensity` scales the brightness of newly-formed trails; `decay` controls
/// how long existing trails persist (0.85 ≈ 4 frames, 0.95 ≈ 14 frames).
public final class NeonTrailsFilter: @unchecked Sendable, MetalForgeFilter {

    /// Brightness multiplier for new trail glow. Sensible range `[0, 2.0]`.
    public var intensity: Float = 1.0

    /// Per-frame multiplicative decay of the trail buffer. Sensible range
    /// `[0.5, 0.99]`; values close to 1 give very long persistence.
    public var decay: Float = 0.85

    /// RGB tint of the trail glow. Default: bright cyan.
    public var neonColor: SIMD3<Float> = SIMD3(0.0, 0.8, 1.0)

    private let device: MTLDevice
    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    private let bufferLock = NSLock()
    private var previousTexture: MTLTexture?
    private var previousFormat:  MTLPixelFormat = .invalid
    private var previousWidth:   Int = 0
    private var previousHeight:  Int = 0

    public init(engine: MetalForgeEngine) throws {
        device = engine.device
        sdrPSO = try TemporalSharedHelpers.makePSO(
            engine: engine, kernel: "neonTrailsKernel", isHDR: false)
        hdrPSO = try TemporalSharedHelpers.makePSO(
            engine: engine, kernel: "neonTrailsKernel", isHDR: true)
    }

    /// Drop the persistent previous-frame buffer. Same semantics as
    /// `MotionBlurFilter.clearHistory` — call when switching away from this
    /// filter or when capture format changes.
    public func clearHistory() {
        bufferLock.lock()
        previousTexture = nil
        previousFormat  = .invalid
        previousWidth   = 0
        previousHeight  = 0
        bufferLock.unlock()
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        bufferLock.lock()
        let needsAlloc =
            previousTexture == nil
            || previousFormat != source.pixelFormat
            || previousWidth  != source.width
            || previousHeight != source.height

        if needsAlloc {
            previousTexture = TemporalSharedHelpers.makePersistentTexture(
                device: device, matching: source)
            previousFormat  = source.pixelFormat
            previousWidth   = source.width
            previousHeight  = source.height
        }
        let prev = previousTexture
        bufferLock.unlock()

        guard let prev else { return }

        // First frame seeding — neon trails start from a fresh state with no
        // accumulated glow, so motion detection against the seeded buffer
        // produces zero (no diff) → no spurious trails appear at filter
        // activation. Subsequent frames build up trails legitimately.
        if needsAlloc {
            if let seedBlit = commandBuffer.makeBlitCommandEncoder() {
                seedBlit.label = "NeonTrailsFilter.seedPrevious"
                seedBlit.copy(from: source, to: prev)
                seedBlit.endEncoding()
            }
        }

        // ----- Compute pass -----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "NeonTrailsFilter"
            enc.setComputePipelineState(pso)
            enc.setTexture(source,      index: 0)
            enc.setTexture(prev,        index: 1)
            enc.setTexture(destination, index: 2)

            var u = NeonTrailsUniforms(
                neonColor: neonColor,
                intensity: intensity,
                decay:     decay
            )
            enc.setBytes(&u, length: MemoryLayout<NeonTrailsUniforms>.stride, index: 0)

            TemporalSharedHelpers.dispatch(
                encoder: enc, pso: pso,
                width: destination.width, height: destination.height
            )
            enc.endEncoding()
        }

        // ----- Save this frame's output to the previous buffer -----
        // The compute pass wrote `destination`, which now contains source +
        // trail composite. Blitting this into prev means next frame's
        // motion-detection sees the *trail-augmented* image — that's what
        // keeps trails alive across many frames despite a single history texture.
        if let saveBlit = commandBuffer.makeBlitCommandEncoder() {
            saveBlit.label = "NeonTrailsFilter.savePrevious"
            saveBlit.copy(from: destination, to: prev)
            saveBlit.endEncoding()
        }
    }
}

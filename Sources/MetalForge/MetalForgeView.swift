#if os(iOS) || os(tvOS) || os(visionOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

// MTLTexture is not yet marked Sendable in Apple's Metal headers, even though
// it is documented as thread-safe for the operations we perform (capturing in
// a completion handler). @preconcurrency downgrades the related warnings until
// Apple annotates the protocol in a future SDK.
@preconcurrency import Metal
import MetalKit
import QuartzCore
import CoreGraphics
import simd

/// Uniform layout — must mirror `DisplayUniforms` in `DisplayRenderShader.metal`.
private struct DisplayUniforms {
    var srcRectMin: SIMD2<Float>
    var srcRectMax: SIMD2<Float>
    var viewportRectMin: SIMD2<Float>
    var viewportRectMax: SIMD2<Float>
}

/// Cross-platform Metal-backed view that renders the output of a
/// `MetalForgePipeline` to the screen with full EDR / HDR support.
///
/// `MetalForgeView` subclasses `MTKView` directly — which is itself a `UIView`
/// on iOS / tvOS / visionOS and an `NSView` on macOS — so it integrates with
/// either platform's view hierarchy with no wrapper layer.
///
/// ## Threading
/// Inherits `@MainActor` isolation from the platform view classes. All public
/// API must be called from the main actor. From a background processing queue:
///
/// ```swift
/// DispatchQueue.main.async { [weak view] in
///     view?.present(texture: outputTexture)
/// }
/// ```
///
/// ## Texture Lifecycle
/// The view stores the most recently presented texture in `pendingTexture`
/// (overwriting any previous unrendered frame — "latest wins" backpressure).
/// After the GPU finishes drawing, the texture is handed to `recycleHandler`,
/// which the integrator typically points at `MetalForgePipeline.recycle(_:)`.
///
/// ## EDR & HDR Configuration
/// The view configures its underlying `CAMetalLayer` for high-precision output
/// based on `workingColorSpace`:
///
/// | Working space | colorPixelFormat   | layer.colorspace               | EDR on |
/// |---------------|--------------------|--------------------------------|--------|
/// | `.sdr`        | `.bgra8Unorm`      | `sRGB`                         | no     |
/// | `.hdr10PQ`    | `.bgr10a2Unorm`    | `itur_2100_PQ`                 | yes    |
/// | `.hlg`        | `.bgr10a2Unorm`    | `itur_2100_HLG`                | yes    |
///
/// When `wantsExtendedDynamicRangeContent = true` and the layer's colorspace is
/// a true HDR colorspace, CoreAnimation routes the layer through the EDR
/// composition path on Liquid Retina XDR (iPad Pro M-series, MacBook Pro 14"/16"
/// M-series) and Pro Display XDR. On these panels the system uses local
/// dimming + dual-modulation backlight to deliver up to 1600 nits sustained
/// (1000 nits full-screen), reproducing PQ-encoded values up to ~1.0 (= 10,000
/// nits in BT.2100) tone-mapped to the panel's actual capability. On non-EDR
/// displays the same content path is honoured but the system tone-maps to the
/// panel's SDR range automatically.
public final class MetalForgeView: MTKView {

    // MARK: - Public configuration

    /// Aspect-ratio behaviour when source and viewport aspect ratios differ.
    public enum ScalingMode: Sendable {
        /// Source fits entirely inside the viewport; bars on the short axis.
        case aspectFit
        /// Source fills the viewport entirely; cropped on the long axis.
        case aspectFill
        /// Source stretched to fill the viewport, ignoring aspect ratio.
        case stretch
    }

    /// Aspect-ratio strategy applied during the display blit. Default: `.aspectFit`.
    public var scalingMode: ScalingMode = .aspectFit {
        didSet { if pendingTexture != nil { self.draw() } }
    }

    /// Drives `colorPixelFormat`, layer `colorspace`, and EDR mode.
    /// Default: `.sdr`. Set this once your pipeline has detected the input.
    public var workingColorSpace: MetalForgeColorSpace = .sdr {
        didSet { applyColorSpace(workingColorSpace) }
    }

    /// Called once per presented texture after the GPU finishes drawing it
    /// (or immediately when a newer frame supersedes a pending one). Typically
    /// wired to `MetalForgePipeline.recycle(_:)` so pool textures cycle back.
    ///
    /// Fires on Metal's internal completion thread — must be `@Sendable`.
    public var recycleHandler: (@Sendable (MTLTexture) -> Void)?

    // MARK: - Internal state

    private let engine: MetalForgeEngine
    // One render PSO per supported colour pixel format; swapped without
    // recompilation when `workingColorSpace` changes.
    private let pipelineStates: [MTLPixelFormat: MTLRenderPipelineState]
    private let samplerState: MTLSamplerState
    private var pendingTexture: MTLTexture?

    // MARK: - Init

    public init(engine: MetalForgeEngine, frame: CGRect = .zero) throws {
        // All `self` stored properties must be set BEFORE super.init.
        self.engine = engine

        self.pipelineStates = try Self.compilePipelineStates(engine: engine)

        let samplerDesc = MTLSamplerDescriptor()
        samplerDesc.label                 = "MetalForgeView.displaySampler"
        samplerDesc.minFilter             = .linear
        samplerDesc.magFilter             = .linear
        samplerDesc.sAddressMode          = .clampToEdge
        samplerDesc.tAddressMode          = .clampToEdge
        samplerDesc.normalizedCoordinates = true
        guard let sampler = engine.device.makeSamplerState(descriptor: samplerDesc) else {
            throw MetalForgeError.pipelineStateCreationFailed("MTLSamplerState creation failed")
        }
        self.samplerState = sampler

        super.init(frame: frame, device: engine.device)

        // ----- MTKView configuration -----
        self.sampleCount             = 1
        self.clearColor              = MTLClearColorMake(0, 0, 0, 1)
        self.framebufferOnly         = true
        // Explicit-render mode: the view only redraws when we call `draw()`.
        // Frame production is driven by MetalForgePipeline, not by a display
        // link, so paused mode gives us deterministic latency and no wasted
        // draws between frames.
        self.isPaused                = true
        self.enableSetNeedsDisplay   = false
        self.delegate                = self

        applyColorSpace(.sdr)
    }

    @available(*, unavailable)
    required init(coder: NSCoder) {
        fatalError("MetalForgeView does not support initialisation from a coder. Use init(engine:frame:).")
    }

    // MARK: - PSO compilation

    private static func compilePipelineStates(
        engine: MetalForgeEngine
    ) throws -> [MTLPixelFormat: MTLRenderPipelineState] {
        let device = engine.device
        let vertexFn   = try engine.makeFunction(name: "displayVertex")
        let fragmentFn = try engine.makeFunction(name: "displayFragment")

        // Pre-compile for every format we expose through `workingColorSpace`,
        // plus `.rgba16Float` for EDR paths on supported displays.
        let formats: [MTLPixelFormat] = [.bgra8Unorm, .bgr10a2Unorm, .rgba16Float]
        var states: [MTLPixelFormat: MTLRenderPipelineState] = [:]
        states.reserveCapacity(formats.count)

        for format in formats {
            let desc = MTLRenderPipelineDescriptor()
            desc.label                            = "MetalForgeView.\(format)"
            desc.vertexFunction                   = vertexFn
            desc.fragmentFunction                 = fragmentFn
            desc.colorAttachments[0].pixelFormat  = format
            do {
                states[format] = try device.makeRenderPipelineState(descriptor: desc)
            } catch {
                throw MetalForgeError.pipelineStateCreationFailed(error.localizedDescription)
            }
        }
        return states
    }

    // MARK: - Colour space + EDR

    private func applyColorSpace(_ cs: MetalForgeColorSpace) {
        // CAMetalLayer cast — guaranteed by MTKView's contract on both platforms.
        let metalLayer = self.layer as? CAMetalLayer

        switch cs {
        case .sdr:
            self.colorPixelFormat                       = .bgra8Unorm
            metalLayer?.colorspace                      = CGColorSpace(name: CGColorSpace.sRGB)
            metalLayer?.wantsExtendedDynamicRangeContent = false

        case .hdr10PQ:
            // .bgr10a2Unorm is the standard HDR10 surface format: 10 bits per
            // channel, 2-bit alpha — half the memory of .rgba16Float and
            // sufficient precision for PQ-encoded display values.
            self.colorPixelFormat                       = .bgr10a2Unorm
            metalLayer?.colorspace                      = CGColorSpace(name: CGColorSpace.itur_2100_PQ)
            metalLayer?.wantsExtendedDynamicRangeContent = true

        case .hlg:
            self.colorPixelFormat                       = .bgr10a2Unorm
            metalLayer?.colorspace                      = CGColorSpace(name: CGColorSpace.itur_2100_HLG)
            metalLayer?.wantsExtendedDynamicRangeContent = true
        }

        // Force re-render with the newly selected colour pipeline if we have
        // a frame in hand. Without this, EDR-mode changes only take effect on
        // the NEXT pipeline frame, which can be noticeable on toggle.
        if pendingTexture != nil { self.draw() }
    }

    // MARK: - Frame presentation

    /// Submit a new processed texture for display. Replaces any previously
    /// queued but undrawn texture (back-pressure: latest wins).
    ///
    /// - Important: Must be invoked on the main actor.
    public func present(texture: MTLTexture) {
        // Drop the previous frame if a new one arrived first — the GPU never
        // saw it, so recycle synchronously now to keep the pool in balance.
        if let displaced = pendingTexture, displaced !== texture {
            recycleHandler?(displaced)
        }
        pendingTexture = texture
        self.draw()    // Calls into MTKViewDelegate.draw(in:) below.
    }

    // MARK: - Display transform math (CPU-side)

    private func computeDisplayTransform(
        sourceSize: CGSize,
        viewportSize: CGSize,
        scalingMode: ScalingMode
    ) -> DisplayUniforms {
        // Degenerate sizes → identity (shader will just sample whole source).
        guard sourceSize.width   > 0, sourceSize.height   > 0,
              viewportSize.width > 0, viewportSize.height > 0
        else {
            return DisplayUniforms(
                srcRectMin:      SIMD2(0, 0),
                srcRectMax:      SIMD2(1, 1),
                viewportRectMin: SIMD2(0, 0),
                viewportRectMax: SIMD2(1, 1)
            )
        }

        let srcAspect = Float(sourceSize.width   / sourceSize.height)
        let dstAspect = Float(viewportSize.width / viewportSize.height)

        switch scalingMode {
        case .stretch:
            return DisplayUniforms(
                srcRectMin:      SIMD2(0, 0),
                srcRectMax:      SIMD2(1, 1),
                viewportRectMin: SIMD2(0, 0),
                viewportRectMax: SIMD2(1, 1)
            )

        case .aspectFit:
            // Full source rect; viewport rect shrinks on the *non-matching* axis.
            // Letterbox / pillarbox bars appear in the cleared margin.
            let scale: SIMD2<Float> = (srcAspect > dstAspect)
                ? SIMD2(1, dstAspect / srcAspect)      // src wider  → vertical letterbox
                : SIMD2(srcAspect / dstAspect, 1)      // src taller → horizontal pillarbox
            let half = (SIMD2<Float>(1, 1) - scale) * 0.5
            return DisplayUniforms(
                srcRectMin:      SIMD2(0, 0),
                srcRectMax:      SIMD2(1, 1),
                viewportRectMin: half,
                viewportRectMax: half + scale
            )

        case .aspectFill:
            // Full viewport rect; source rect shrinks (we sample only the inner crop).
            let scale: SIMD2<Float> = (srcAspect > dstAspect)
                ? SIMD2(dstAspect / srcAspect, 1)      // src wider  → horizontal crop
                : SIMD2(1, srcAspect / dstAspect)      // src taller → vertical crop
            let half = (SIMD2<Float>(1, 1) - scale) * 0.5
            return DisplayUniforms(
                srcRectMin:      half,
                srcRectMax:      half + scale,
                viewportRectMin: SIMD2(0, 0),
                viewportRectMax: SIMD2(1, 1)
            )
        }
    }
}

// MARK: - MTKViewDelegate

extension MetalForgeView: MTKViewDelegate {

    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Re-render the current frame at the new drawable size on the *next*
        // present; we don't have a fresh texture so just trigger a draw if
        // something is pending.
        if pendingTexture != nil { self.draw() }
    }

    public func draw(in view: MTKView) {
        // No work if no frame is queued.
        guard let texture = pendingTexture else { return }

        // Skip silently if the view isn't on screen yet — drawable not yet
        // allocated. Keep `pendingTexture` so it lands on the next display refresh.
        guard
            let drawable      = view.currentDrawable,
            let descriptor    = view.currentRenderPassDescriptor,
            let commandBuffer = engine.commandQueue.makeCommandBuffer(),
            let pso           = pipelineStates[view.colorPixelFormat]
        else { return }

        // Claim the texture — anything that arrives after this consumes a
        // fresh pendingTexture slot.
        pendingTexture = nil

        // The descriptor's clearColor is taken from `self.clearColor`, but we
        // re-apply explicitly to defend against external mutation.
        descriptor.colorAttachments[0].clearColor  = self.clearColor
        descriptor.colorAttachments[0].loadAction  = .clear
        descriptor.colorAttachments[0].storeAction = .store

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else {
            // Encoder couldn't be created — recycle the texture so the pool
            // doesn't leak it on a degenerate frame.
            recycleHandler?(texture)
            return
        }
        renderEncoder.label = "MetalForgeView.displayPass"
        renderEncoder.setRenderPipelineState(pso)
        renderEncoder.setFragmentTexture(texture, index: 0)
        renderEncoder.setFragmentSamplerState(samplerState, index: 0)

        var uniforms = computeDisplayTransform(
            sourceSize:   CGSize(width: texture.width, height: texture.height),
            viewportSize: view.drawableSize,
            scalingMode:  scalingMode
        )
        renderEncoder.setFragmentBytes(
            &uniforms,
            length: MemoryLayout<DisplayUniforms>.stride,
            index: 0
        )

        // Fullscreen triangle — no vertex buffer, vertex shader generates positions.
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        renderEncoder.endEncoding()

        // Capture the @Sendable recycler value at the moment of commit, so the
        // completion handler closes over a Sendable closure and Sendable
        // texture — no MainActor capture required.
        let recycler      = self.recycleHandler
        let recycleTarget = texture
        commandBuffer.addCompletedHandler { _ in
            recycler?(recycleTarget)
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

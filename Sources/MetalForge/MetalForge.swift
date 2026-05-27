/// MetalForge — High-performance Metal-based video processing engine.
///
/// Typical usage:
/// ```swift
/// let engine   = try MetalForgeEngine()
/// let pipeline = MetalForgePipeline(engine: engine)
/// try pipeline.append(AdjustmentFilter(engine: engine))
/// try pipeline.append(GlitchFilter(engine: engine))
///
/// // In your AVCaptureVideoDataOutput delegate (background queue):
/// if let output = pipeline.process(pixelBuffer: pixelBuffer) {
///     // present or encode `output`, then:
///     pipeline.recycle(output)
/// }
/// ```
public enum MetalForge {}

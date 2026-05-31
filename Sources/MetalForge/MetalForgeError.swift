import CoreVideo

public enum MetalForgeError: Error, LocalizedError, Sendable {
    case deviceNotAvailable
    case commandQueueCreationFailed
    case textureCacheCreationFailed(CVReturn)
    case pipelineStateCreationFailed(String)
    case shaderFunctionNotFound(String)
    case shaderLibraryUnavailable(String)
    case textureAllocationFailed
    case recorderInvalidState(String)
    case recorderAssetWriterFailed(String)
    case recorderCannotAddInput(String)

    public var errorDescription: String? {
        switch self {
        case .deviceNotAvailable:
            return "No Metal-capable GPU found on this device."
        case .commandQueueCreationFailed:
            return "Failed to create MTLCommandQueue."
        case .textureCacheCreationFailed(let code):
            return "CVMetalTextureCache creation failed with CVReturn \(code)."
        case .pipelineStateCreationFailed(let msg):
            return "MTLComputePipelineState error: \(msg)"
        case .shaderFunctionNotFound(let name):
            return "MSL kernel '\(name)' not found in the compiled .metallib."
        case .shaderLibraryUnavailable(let detail):
            return "Failed to load the MetalForge shader library: \(detail)"
        case .textureAllocationFailed:
            return "MTLTexture allocation failed — device may be out of VRAM."
        case .recorderInvalidState(let detail):
            return "MetalForgeRecorder is in an invalid state for this operation: \(detail)"
        case .recorderAssetWriterFailed(let detail):
            return "AVAssetWriter error: \(detail)"
        case .recorderCannotAddInput(let media):
            return "AVAssetWriter rejected the \(media) input."
        }
    }
}

import Metal
import Foundation

/// A thread-safe pool that recycles `MTLTexture` instances by (width, height, pixelFormat).
///
/// ## Why this matters
/// Allocating a new `MTLTexture` on every frame at 4K 60 fps triggers the GPU memory
/// allocator ~60 times/second per filter pass. The allocator is not free — it takes
/// lock-protected IOSurface bookkeeping under the hood. Pooling reduces this to a few
/// allocs at startup and near-zero steady-state cost.
///
/// ## Storage Mode
/// All pooled textures use `.private` storage (GPU-only). They are never mapped to
/// CPU-accessible memory, which makes them the fastest possible intermediate surfaces.
/// Do not use them for readback; use a separate staging buffer for that.
///
/// ## Thread Safety
/// `acquire` and `recycle` are safe to call concurrently from multiple threads.
public final class TexturePool: @unchecked Sendable {

    private struct Key: Hashable {
        let width: Int
        let height: Int
        let pixelFormat: MTLPixelFormat
    }

    private let device: MTLDevice
    private var buckets: [Key: [MTLTexture]] = [:]
    private let lock = NSLock()

    public init(device: MTLDevice) {
        self.device = device
    }

    // MARK: - Acquire / Recycle

    /// Return a pooled texture matching the spec, or allocate a new one.
    ///
    /// - Returns: An `MTLTexture` with `.shaderRead | .shaderWrite` usage, or `nil`
    ///   if the Metal device fails to allocate (extremely unlikely, indicates OOM).
    public func acquire(
        width: Int,
        height: Int,
        pixelFormat: MTLPixelFormat = .bgra8Unorm
    ) -> MTLTexture? {
        let key = Key(width: width, height: height, pixelFormat: pixelFormat)

        lock.lock()
        if var bucket = buckets[key], !bucket.isEmpty {
            let texture = bucket.removeLast()
            buckets[key] = bucket
            lock.unlock()
            return texture
        }
        lock.unlock()

        // Nothing in the pool — allocate.
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width:       width,
            height:      height,
            mipmapped:   false
        )
        desc.usage       = [.shaderRead, .shaderWrite]
        // .private = GPU-only memory; fastest for compute intermediates; not CPU-readable.
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }

    /// Return a texture to the pool so a future `acquire` can reuse it.
    ///
    /// Only recycle textures that were obtained from *this* pool and whose
    /// GPU work (command buffer) has already completed.
    public func recycle(_ texture: MTLTexture) {
        let key = Key(
            width:       texture.width,
            height:      texture.height,
            pixelFormat: texture.pixelFormat
        )
        lock.lock()
        buckets[key, default: []].append(texture)
        lock.unlock()
    }

    // MARK: - Memory pressure

    /// Release all pooled textures immediately.
    /// Call in response to a system memory-pressure notification.
    public func purge() {
        lock.lock()
        buckets.removeAll(keepingCapacity: false)
        lock.unlock()
    }
}

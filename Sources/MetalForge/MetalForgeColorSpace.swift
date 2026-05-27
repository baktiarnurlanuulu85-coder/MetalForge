import Foundation
import CoreVideo

/// The working colour space of a MetalForge processing chain.
///
/// Determines the pixel format of intermediate textures and the colour
/// transfer characteristics applied by source-stage filters.
///
/// - `sdr`:     BT.709 primaries, gamma 2.2 / sRGB transfer, 8-bit per channel.
///              Intermediate textures use `.bgra8Unorm`.
/// - `hdr10PQ`: BT.2020 primaries, SMPTE ST.2084 (PQ) transfer, 10-bit input.
///              Intermediate textures use `.rgba16Float` to preserve linear
///              scene-referred values above 1.0 (highlights up to 10 000 nits).
/// - `hlg`:     BT.2020 primaries, ARIB STD-B67 (Hybrid Log-Gamma) transfer.
///              Same intermediate format as PQ; differs only in EOTF.
public enum MetalForgeColorSpace: Sendable, Equatable {
    case sdr
    case hdr10PQ
    case hlg

    /// Whether this colour space requires high-precision floating-point intermediates.
    public var isHDR: Bool {
        switch self {
        case .sdr:              return false
        case .hdr10PQ, .hlg:    return true
        }
    }

    /// Detect the working colour space from a `CVPixelBuffer`'s format and
    /// transfer-function attachment.
    ///
    /// Rules:
    /// 1. **8-bit YUV / BGRA** → `.sdr` (no transfer-function lookup needed).
    /// 2. **10-bit YUV** → inspect `kCVImageBufferTransferFunctionKey`:
    ///    - `…_ITU_R_2100_HLG`  → `.hlg`
    ///    - anything else, or attachment missing → `.hdr10PQ` (HDR10 default).
    ///
    /// This matches what `AVAssetReader` and `AVCaptureVideoDataOutput` populate
    /// on their CMSampleBuffers for HEVC HDR sources on iPhone 12+/Apple silicon.
    public static func detect(from pixelBuffer: CVPixelBuffer) -> MetalForgeColorSpace {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let is10bit: Bool
        switch format {
        case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
             kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
            is10bit = true
        default:
            is10bit = false
        }

        if !is10bit { return .sdr }

        let transferID = CVBufferCopyAttachment(
            pixelBuffer,
            kCVImageBufferTransferFunctionKey,
            nil
        ) as? String

        if transferID == (kCVImageBufferTransferFunction_ITU_R_2100_HLG as String) {
            return .hlg
        }
        // HDR10 / PQ is the default for any 10-bit content lacking explicit HLG marking.
        return .hdr10PQ
    }
}

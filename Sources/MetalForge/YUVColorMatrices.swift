import CoreVideo
import simd

/// Pre-baked YCbCr → RGB conversion matrices for the standard ITU broadcast
/// colour spaces, plus a helper that picks the correct one from a
/// `CVPixelBuffer`'s colour attachments.
///
/// ## Matrix Convention
/// All matrices are arranged so that:
/// ```
///   rgb = M * (ycbcr - offset)
/// ```
/// where for **full-range** input `offset = (0, 0.5, 0.5)` and for **video-range**
/// input `offset = (16/255, 128/255, 128/255)`. The range-expansion scaling
/// (×1.16438 for Y, etc.) is **already baked** into the video-range matrices, so
/// the shader only needs to subtract offsets — no extra scaling.
///
/// ## 10-bit note
/// The 10-bit-packed-in-16-bit normalisation used by Metal's `.r16Unorm` / `.rg16Unorm`
/// differs from ideal 10-bit by ~0.1 %; this delta is below visible threshold for
/// all downstream operations and is absorbed into the same matrices.
public enum YUVColorMatrices {

    // MARK: BT.601 (legacy SD video)

    public static let bt601FullRange = matrix_float3x3(rows: [
        SIMD3<Float>(1.0,  0.0,         1.402     ),
        SIMD3<Float>(1.0, -0.344136,   -0.714136  ),
        SIMD3<Float>(1.0,  1.772,       0.0       )
    ])

    public static let bt601VideoRange = matrix_float3x3(rows: [
        SIMD3<Float>(1.16438,  0.0,       1.59603 ),
        SIMD3<Float>(1.16438, -0.39176,  -0.81297 ),
        SIMD3<Float>(1.16438,  2.01723,   0.0     )
    ])

    // MARK: BT.709 (HD SDR — default for most modern 1080p content)

    public static let bt709FullRange = matrix_float3x3(rows: [
        SIMD3<Float>(1.0,  0.0,         1.5748    ),
        SIMD3<Float>(1.0, -0.1873,     -0.4681    ),
        SIMD3<Float>(1.0,  1.8556,      0.0       )
    ])

    public static let bt709VideoRange = matrix_float3x3(rows: [
        SIMD3<Float>(1.16438,  0.0,       1.79274 ),
        SIMD3<Float>(1.16438, -0.21325,  -0.53291 ),
        SIMD3<Float>(1.16438,  2.11240,   0.0     )
    ])

    // MARK: BT.2020 (UHD / HDR10 / Dolby Vision)

    public static let bt2020FullRange = matrix_float3x3(rows: [
        SIMD3<Float>(1.0,  0.0,         1.4746    ),
        SIMD3<Float>(1.0, -0.16455,    -0.57135   ),
        SIMD3<Float>(1.0,  1.8814,      0.0       )
    ])

    public static let bt2020VideoRange = matrix_float3x3(rows: [
        SIMD3<Float>(1.16438,  0.0,       1.67867 ),
        SIMD3<Float>(1.16438, -0.18733,  -0.65042 ),
        SIMD3<Float>(1.16438,  2.14177,   0.0     )
    ])

    // MARK: RGB → YCbCr encode matrices (inverse, with range compression baked in)
    //
    // Layout: rows are (Y, Cb, Cr) coefficients applied to (R, G, B). Offsets
    // (16/255 for Y video range, 128/255 for CbCr) are added by the shader after
    // the matrix multiplication.

    /// BT.709 RGB → YCbCr (video range). Used for SDR recording.
    public static let bt709VideoRangeEncode = matrix_float3x3(rows: [
        SIMD3<Float>( 0.18259,  0.61423,  0.06200),   // Y
        SIMD3<Float>(-0.10064, -0.33857,  0.43922),   // Cb
        SIMD3<Float>( 0.43922, -0.39894, -0.04027)    // Cr
    ])

    /// BT.2020 RGB → YCbCr (video range). Used for HDR10 / HLG recording.
    /// Same matrix for both PQ and HLG — they share BT.2020 primaries and the
    /// non-constant-luminance YCbCr basis. The transfer function (PQ vs HLG)
    /// affects only the *values* fed to this matrix, not the matrix itself.
    public static let bt2020VideoRangeEncode = matrix_float3x3(rows: [
        SIMD3<Float>( 0.22569,  0.58228,  0.05093),   // Y
        SIMD3<Float>(-0.12266, -0.31649,  0.43922),   // Cb
        SIMD3<Float>( 0.43922, -0.40392, -0.03533)    // Cr
    ])

    // MARK: Selection

    /// Resolves the YCbCr→RGB matrix and range flag from a `CVPixelBuffer`.
    ///
    /// Reads:
    /// 1. `kCVImageBufferYCbCrMatrixKey` attachment (BT.601 / BT.709 / BT.2020).
    /// 2. The pixel format type to detect full vs video range.
    ///
    /// Fallback policy:
    /// - If the matrix attachment is missing, choose by bit depth:
    ///   10-bit → BT.2020, 8-bit → BT.709 (the safe modern defaults).
    /// - Unknown matrix identifiers fall back to BT.709 (most common SDR matrix).
    ///
    /// - Parameter pixelBuffer: The YUV bi-planar source buffer.
    /// - Returns: A tuple of the pre-scaled 3×3 matrix and the range flag
    ///   (`true` = full range, `false` = video range).
    public static func matrix(for pixelBuffer: CVPixelBuffer)
        -> (matrix: matrix_float3x3, isFullRange: Bool)
    {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)

        // ----- Range from format type (authoritative source) -----
        let isFullRange: Bool
        switch format {
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
             kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
            isFullRange = true
        default:
            isFullRange = false
        }

        // ----- Matrix from YCbCr matrix attachment -----
        // The Swift overlay automatically bridges CVBufferCopyAttachment's
        // +1 CF retain into ARC; we receive a CFTypeRef? directly.
        let attachmentID = CVBufferCopyAttachment(
            pixelBuffer,
            kCVImageBufferYCbCrMatrixKey,
            nil
        ) as? String

        let matrix: matrix_float3x3
        if let id = attachmentID {
            if id == (kCVImageBufferYCbCrMatrix_ITU_R_2020 as String) {
                matrix = isFullRange ? bt2020FullRange : bt2020VideoRange
            } else if id == (kCVImageBufferYCbCrMatrix_ITU_R_601_4 as String) {
                matrix = isFullRange ? bt601FullRange : bt601VideoRange
            } else {
                // BT.709 — most common SDR matrix; also our fallback for unknown IDs.
                matrix = isFullRange ? bt709FullRange : bt709VideoRange
            }
        } else {
            // No attachment — guess by bit depth.
            switch format {
            case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
                 kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
                matrix = isFullRange ? bt2020FullRange : bt2020VideoRange
            default:
                matrix = isFullRange ? bt709FullRange : bt709VideoRange
            }
        }

        return (matrix, isFullRange)
    }
}

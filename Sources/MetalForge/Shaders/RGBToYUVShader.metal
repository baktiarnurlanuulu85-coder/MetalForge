#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// RGB → YCbCr 4:2:0 bi-planar encoder.
//
// Mirror of YUVToRGBConverter, used at the recorder output stage to pack the
// pipeline's RGB texture into the two-plane CVPixelBuffer that VideoToolbox
// expects for HEVC HDR10 (10-bit) or H.264 (8-bit) encoding.
//
// Two-pass design:
//   • Pass 1 (`rgbToLumaKernel`)   runs at LUMA resolution. One thread per Y
//     texel — straight matrix multiply on the source RGB sample.
//   • Pass 2 (`rgbToChromaKernel`) runs at CHROMA resolution (half-res per axis).
//     Each thread samples the full-res RGB texture through a `filter::linear`
//     sampler at the chroma texel centre — the TMU averages a 2×2 RGB block in
//     hardware, giving correct 4:2:0 downsampling for free, no manual averaging
//     and no threadgroup-shared memory needed.
//
// Range / offset convention:
//   The colour matrix already includes range compression (×219/255 for Y,
//   ×224/255 for CbCr in video range). The shader only adds offsets after the
//   matrix multiply: 16/255 for Y in video range, 128/255 for CbCr (centred at
//   neutral). For full range Y the offset is 0.
// ===========================================================================

struct RGBToYUVUniforms {
    float3x3 colorMatrix;   // BT.709 or BT.2020, video-range compression baked in
    uint     isFullRange;   // 1 = full range, 0 = video range (Y has 16/255 offset)
};

constexpr sampler chromaSampler(
    coord::normalized,
    filter::linear,
    address::clamp_to_edge
);

// ===========================================================================
// Pass 1: Luma
//   Reads RGB at full resolution, writes Y to luma plane.
// ===========================================================================
kernel void rgbToLumaKernel(
    texture2d<float, access::read>  rgbTex  [[texture(0)]],
    texture2d<float, access::write> lumaTex [[texture(1)]],
    constant RGBToYUVUniforms&      u       [[buffer(0)]],
    uint2                           gid     [[thread_position_in_grid]])
{
    if (gid.x >= lumaTex.get_width() || gid.y >= lumaTex.get_height()) return;

    // PQ/HLG-encoded values from HDREncodeFilter are in [0,1].
    // SDR values from the standard chain are also in [0,1]. Clamp defends
    // against rogue out-of-range floats from custom user filters.
    float3 rgb = clamp(rgbTex.read(gid).rgb, 0.0f, 1.0f);

    // Matrix gives (Y, Cb, Cr) before offset. We only need Y for the luma plane.
    const float y = (u.colorMatrix * rgb).x;

    // Add range offset: 0 for full range, 16/255 for video (limited) range.
    const float yOffset = (u.isFullRange != 0u) ? 0.0f : (16.0f / 255.0f);

    // Clamp to valid [0,1] storage range. For .r16Unorm with 10-bit-packed
    // semantics, writing 0.0625 stores exactly 10-bit value 64 (the video-range
    // black point), and writing 0.918 stores 940 (video-range white point).
    const float yOut = clamp(y + yOffset, 0.0f, 1.0f);

    lumaTex.write(float4(yOut, 0.0f, 0.0f, 1.0f), gid);
}

// ===========================================================================
// Pass 2: Chroma (4:2:0 subsampled)
//   Runs at half-resolution. Samples RGB with hardware bilinear filtering at
//   the chroma texel centre — TMU averages the 2×2 RGB neighbourhood in one
//   fetch. Writes (Cb, Cr) to the .rg16Unorm chroma plane.
// ===========================================================================
kernel void rgbToChromaKernel(
    texture2d<float, access::sample> rgbTex    [[texture(0)]],
    texture2d<float, access::write>  chromaTex [[texture(1)]],
    constant RGBToYUVUniforms&       u         [[buffer(0)]],
    uint2                            gid       [[thread_position_in_grid]])
{
    if (gid.x >= chromaTex.get_width() || gid.y >= chromaTex.get_height()) return;

    // Normalised UV at chroma texel centre. With filter::linear on the full-res
    // RGB texture, each sample averages the 2×2 source block — exactly the
    // 4:2:0 chroma subsampling specified by JPEG / MPEG / BT.601 / 709 / 2020.
    const float2 chromaSize = float2(float(chromaTex.get_width()),
                                     float(chromaTex.get_height()));
    const float2 uv         = (float2(gid) + 0.5f) / chromaSize;

    float3 rgb = clamp(rgbTex.sample(chromaSampler, uv).rgb, 0.0f, 1.0f);

    const float3 yuv = u.colorMatrix * rgb;

    // CbCr neutral position. 128/255 is the canonical 8-bit value; for our
    // 10-bit-in-16-bit storage (.rg16Unorm) the rounding to neutral is exact
    // enough for visual quality. Both full and video range use the same offset.
    const float cbOffset = 128.0f / 255.0f;

    const float cb = clamp(yuv.y + cbOffset, 0.0f, 1.0f);
    const float cr = clamp(yuv.z + cbOffset, 0.0f, 1.0f);

    chromaTex.write(float4(cb, cr, 0.0f, 1.0f), gid);
}

#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// Uniform layout — must mirror YUVConverterUniforms in YUVToRGBConverter.swift.
// In MSL, float3x3 is stored as 3 columns of float3 (each 16-byte aligned),
// total 48 B. Trailing uint + 12 B padding gives a 64 B stride.
// ===========================================================================
struct YUVConverterUniforms {
    float3x3 colorMatrix;   // BT.601 / 709 / 2020, range-expansion baked in
    uint     isFullRange;   // 1 = full range, 0 = video (limited) range
};

// ===========================================================================
// Inline sampler.
// `coord::normalized + filter::linear` is what gives us hardware-accelerated
// 4:2:0 → 4:4:4 chroma upsampling on the TMU — the four nearest chroma texels
// are bilinearly blended in a single instruction, no manual lerp needed.
// `address::clamp_to_edge` prevents wrap-around artifacts on right/bottom edges.
// ===========================================================================
constexpr sampler chromaSampler(
    coord::normalized,
    filter::linear,
    address::clamp_to_edge
);

// ===========================================================================
// yuvToRgbCompute
//
// One thread per destination texel. Apple GPU dispatches in SIMD groups of 32;
// `dispatchThreads` (non-uniform) avoids manual ceil() rounding while the
// in-shader bounds guard handles the final partial tile at texture edges.
//
// NOTE on luma vs chroma access modes:
// - luma uses `access::read` because the destination is at luma resolution,
//   so `gid` maps 1:1 to luma texels — no filtering needed.
// - chroma uses `access::sample` (the source spec said `read`, but `sample`
//   is mandatory to bind a `sampler`; this is exactly how we get the free TMU
//   bilinear upsample described in the requirements).
// ===========================================================================
kernel void yuvToRgbCompute(
    texture2d<float, access::read>   luma        [[texture(0)]],
    texture2d<float, access::sample> chroma      [[texture(1)]],
    texture2d<float, access::write>  destination [[texture(2)]],
    constant YUVConverterUniforms&   u           [[buffer(0)]],
    uint2                            gid         [[thread_position_in_grid]])
{
    const uint W = destination.get_width();
    const uint H = destination.get_height();
    if (gid.x >= W || gid.y >= H) return;

    // -----------------------------------------------------------------------
    // Read luma (Y) at full resolution.
    // For .r8Unorm  : value ∈ [0, 1]  representing 8-bit  Y
    // For .r16Unorm : value ≈ Y_10bit / 1023.98 (10-bit packed in 16-bit container)
    // -----------------------------------------------------------------------
    float y = luma.read(gid).r;

    // -----------------------------------------------------------------------
    // Sample chroma (Cb, Cr) with normalised UV. The +0.5 half-pixel offset
    // places the sample at the texel centre, which — combined with linear
    // filtering on a half-resolution texture — produces the correct
    // upsampled chroma value at the destination pixel.
    // -----------------------------------------------------------------------
    const float2 destSize = float2(float(W), float(H));
    const float2 uv       = (float2(gid) + 0.5f) / destSize;
    const float2 cbcr     = chroma.sample(chromaSampler, uv).rg;

    // -----------------------------------------------------------------------
    // Build the YCbCr vector with range-appropriate offsets.
    // The colour matrix already includes any range-expansion scaling.
    // -----------------------------------------------------------------------
    float3 ycbcr;
    if (u.isFullRange != 0u) {
        // Full range: Y ∈ [0, 1], CbCr centred at 0.5
        ycbcr = float3(y,
                       cbcr.x - 0.5f,
                       cbcr.y - 0.5f);
    } else {
        // Video range: Y starts at 16/255, CbCr at 128/255. Matrix scales internally.
        ycbcr = float3(y       - (16.0f  / 255.0f),
                       cbcr.x  - (128.0f / 255.0f),
                       cbcr.y  - (128.0f / 255.0f));
    }

    // -----------------------------------------------------------------------
    // Matrix multiply gives linear-light RGB in the input's primaries.
    // For SDR (BT.709) the result is in sRGB-relative linear (or close enough
    // for typical processing pipelines without gamma decode).
    // For HDR (BT.2020) the result is still in the source's transfer space
    // (PQ or HLG); a separate EOTF stage is required for true linear-light.
    // No clamp here — downstream HDR filters may rely on out-of-[0,1] values.
    // -----------------------------------------------------------------------
    const float3 rgb = u.colorMatrix * ycbcr;

    destination.write(float4(rgb, 1.0f), gid);
}

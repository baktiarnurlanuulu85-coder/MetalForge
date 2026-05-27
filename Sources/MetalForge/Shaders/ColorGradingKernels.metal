#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// ColorGradingPack — professional colour pipeline operations.
//
//   • lut3DColorKernel        Trilinear sampling of a `texture3d` LUT cube,
//                             HDR-safe via base/highlight splitting.
//   • colorCorrectionKernel   Exposure (stops), contrast around scene-referred
//                             0.18 grey, BT.709/2020-aware saturation, R/B
//                             temperature push.
//
// Both kernels are function-constant specialised on `isHDR` — SDR variant
// clamps to [0, 1], HDR variant lets highlights propagate above 1.0 (with
// only a max(_, 0) negative-luminance guard).
// ===========================================================================

constant bool isHDR [[function_constant(0)]];

// ---------------------------------------------------------------------------
// Luma coefficient sets. The Y' weights for BT.709 (SDR) and BT.2020 (HDR)
// match the ones our YCbCr conversion uses upstream — keeping the saturation
// pivot luminance-perceptually consistent with the input's encoding chain.
// ---------------------------------------------------------------------------
constant float3 kBT709Luma  = float3(0.2126f, 0.7152f, 0.0722f);
constant float3 kBT2020Luma = float3(0.2627f, 0.6780f, 0.0593f);

// ===========================================================================
// 1. 3D LUT colour grading
// ===========================================================================

struct LUTUniforms {
    float intensity;   // 0 = bypass (output = source), 1 = full LUT effect
    float lutSize;     // cube edge length (16, 32, 64 typical). Float for math.
};

// Inline sampler with bilinear filtering (trilinear on 3D textures) and
// clamp-to-edge — same pattern as our 2D samplers. Address mode matters here:
// any UV that lands outside [0,1] is clamped to the nearest face of the cube,
// which is what we want once we've already remapped input into the padded range.
constexpr sampler lutSampler(
    coord::normalized,
    filter::linear,
    address::clamp_to_edge
);

kernel void lut3DColorKernel(
    texture2d<float, access::read>    src    [[texture(0)]],
    texture2d<float, access::write>   dst    [[texture(1)]],
    texture3d<float, access::sample>  lutTex [[texture(2)]],
    constant LUTUniforms&             u      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    const float4 color   = src.read(gid);
    const float3 srcRGB  = color.rgb;

    // ---------- HDR-safe input mapping ----------
    // A 3D LUT cube is only defined for inputs in [0, 1]. In HDR working space
    // pixel values can exceed 1.0 (highlights up to ~10× SDR-white for PQ).
    //
    // Strategy:
    //   • base       = clamp(src, 0, 1)   — feed this into the LUT
    //   • highlight  = max(src - 1, 0)    — the "above-white" portion
    //   • output     = mix(base, lut(base), intensity) + highlight
    //
    // This grades only the visible range and lets highlights propagate
    // unchanged — the perceptually correct compromise for "apply LUT in HDR".
    float3 base;
    float3 highlight;
    if (isHDR) {
        base      = clamp(srcRGB, 0.0f, 1.0f);
        highlight = max(srcRGB - 1.0f, 0.0f);
    } else {
        base      = saturate(srcRGB);
        highlight = float3(0.0f);
    }

    // ---------- Padded LUT coordinate mapping ----------
    // For an N×N×N LUT, naive `uvw = base` would have the sampler reach
    // outside the cube on the edges due to bilinear interpolation, fetching
    // clamp-edge texels with reduced precision.
    //
    // The correct mapping shifts the range to centre on texels:
    //   [0, 1] → [0.5/N, (N - 0.5)/N]
    // ⇒  uvw = base × (N-1)/N + 0.5/N
    //
    // At base=0 we sample the centre of the first texel; at base=1, the
    // centre of the last. Anywhere in between, trilinear interpolation gives
    // smooth response across the full cube.
    const float  scale  = (u.lutSize - 1.0f) / u.lutSize;
    const float  offset = 0.5f / u.lutSize;
    const float3 uvw    = base * scale + offset;

    // Trilinear sample of the 3D cube. The hardware reads 8 surrounding
    // texels and interpolates in a single instruction.
    const float3 graded = lutTex.sample(lutSampler, uvw).rgb;

    // ---------- Blend identity ↔ LUT ----------
    const float3 mixed = mix(base, graded, u.intensity);

    // Add HDR highlights back unchanged. For SDR `highlight` is the zero
    // vector, so this is a no-op there.
    float3 result = mixed + highlight;

    if (!isHDR) {
        result = saturate(result);
    }

    dst.write(float4(result, color.a), gid);
}

// ===========================================================================
// 2. Linear-space colour correction
// ===========================================================================

struct ColorCorrectionUniforms {
    float exposure;          // in stops (EV); 0 = identity, +1 = 2× brightness
    float contrast;          // multiplier around 0.18 pivot; 1 = identity
    float saturation;        // 0 = grayscale, 1 = original, >1 = boost
    float temperatureShift;  // -1 cool ↔ +1 warm; 0 = identity
};

kernel void colorCorrectionKernel(
    texture2d<float, access::read>    src [[texture(0)]],
    texture2d<float, access::write>   dst [[texture(1)]],
    constant ColorCorrectionUniforms& u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    const float4 color = src.read(gid);
    float3 rgb = color.rgb;

    // ---------- 1. Exposure ----------
    // exposure = stops; the linear-light multiplier is 2^stops.
    // exp2(x) is one HW instruction on Apple GPU — faster than pow(2, x).
    rgb *= exp2(u.exposure);

    // ---------- 2. Contrast around scene-referred 0.18 ----------
    // Crucial: the pivot is **0.18**, not 0.5.
    //   • 0.18 linear is the BT.709/BT.2020 reference "middle grey" — what
    //     an 18 % grey card reflects, and the canonical anchor for linear
    //     exposure metering.
    //   • 0.5 would be wrong because in linear-light space, 0.5 ≈ 73 % sRGB,
    //     deep into the highlights perceptually. Using 0.5 as the pivot
    //     would crush shadows under any contrast boost.
    //   • In linear, doubling contrast from grey doubles the *distance*
    //     from grey in linear units — which is roughly perceptually uniform
    //     when paired with a proper EOTF on output.
    rgb = (rgb - 0.18f) * u.contrast + 0.18f;
    // Guard before saturation — negative linear luminance is non-physical and
    // would compute a bogus luma scalar in the next step.
    rgb = max(rgb, 0.0f);

    // ---------- 3. Saturation around per-pixel luminance ----------
    // Standard "weight × pixel ⇒ luma scalar, mix toward grey-of-same-luma".
    // BT.709 weights for SDR, BT.2020 for HDR — these are exactly the weights
    // the YCbCr decode uses upstream, so the saturation pivot is colorimetrically
    // consistent with the input encoding.
    const float3 lumaCoef = isHDR ? kBT2020Luma : kBT709Luma;
    const float  luma     = dot(rgb, lumaCoef);
    rgb = mix(float3(luma), rgb, u.saturation);

    // ---------- 4. Temperature shift ----------
    // Simple R/B push. Not a Planckian colour-temperature curve, just a
    // perceptually-mild red-vs-blue balance for the demo. The 0.3 cap keeps
    // ±1 slider extremes from blowing out into pure red or pure blue.
    rgb.r *= 1.0f + u.temperatureShift * 0.3f;
    rgb.b *= 1.0f - u.temperatureShift * 0.3f;
    rgb = max(rgb, 0.0f);

    // ---------- 5. SDR-only highlight clip ----------
    if (!isHDR) {
        rgb = min(rgb, 1.0f);
    }

    dst.write(float4(rgb, color.a), gid);
}

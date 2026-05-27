#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// AnalogDistortionPack — Chromatic aberration + analog noise + horizontal jitter.
//
// Shared design:
//   • Each kernel reads a function-constant `isHDR` that gates the final clamp.
//     With function-constant specialisation the Metal compiler emits two
//     distinct PSO variants per kernel — the unused branch is eliminated as
//     dead code, so there is zero per-thread cost for the conditional.
//   • SDR path clamps output to [0, 1]; HDR path lets highlights propagate
//     above 1.0 untouched, preserving the linear scene-referred range expected
//     by .rgba16Float intermediates.
//   • All three use non-uniform `dispatchThreads`, so the in-shader bounds
//     guard `gid >= width/height` is mandatory on edge tiles.
// ===========================================================================

constant bool isHDR [[function_constant(0)]];

// ---------------------------------------------------------------------------
// Hash functions — "Hash Without Sine" family (David Hoskins). Cheap, good
// quality, no transcendentals on the hot path of analogNoise / jitter.
// ---------------------------------------------------------------------------
static inline float hash21(float2 p) {
    p = fract(p * float2(123.34f, 456.21f));
    p += dot(p, p + 78.233f);
    return fract(p.x * p.y);
}

static inline float hash11(float n) {
    // Single transcendental, still cheap on Apple GPU. Used only for per-row
    // hashing in horizontalJitter where we want strong serial correlation.
    return fract(sin(n * 12.9898f) * 43758.5453f);
}

// ---------------------------------------------------------------------------
// Inline sampler — bilinear + clamp-to-edge. Used by chromaticAberration
// (offset sampling) and horizontalJitter (row-shifted sampling). Clamp-to-edge
// means out-of-bounds reads return edge texels instead of black/wrap.
// ---------------------------------------------------------------------------
constexpr sampler texSampler(
    coord::normalized,
    filter::linear,
    address::clamp_to_edge
);

// ===========================================================================
// 1. Chromatic Aberration
//
// Three samples of the source: red channel at uv + redShift, green at
// uv + greenShift, blue at uv (centre). Combined into a single RGB texel.
// Alpha taken from the centre sample. This recreates the chromatic
// dispersion of an uncorrected lens (lateral CA), or the colour-skew you
// see in old NTSC / VHS chains.
// ===========================================================================
struct ChromaticAberrationUniforms {
    float2 redShift;     // normalised UV offset for R sample
    float2 greenShift;   // normalised UV offset for G sample
};

kernel void chromaticAberrationKernel(
    texture2d<float, access::sample>  source [[texture(0)]],
    texture2d<float, access::write>   dest   [[texture(1)]],
    constant ChromaticAberrationUniforms& u  [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint W = dest.get_width();
    const uint H = dest.get_height();
    if (gid.x >= W || gid.y >= H) return;

    const float2 uv = (float2(gid) + 0.5f) / float2(float(W), float(H));

    // Three samples; only one component used from each, but the GPU likely
    // coalesces these into a single TMU burst because the offsets are tiny.
    const float r     = source.sample(texSampler, uv + u.redShift  ).r;
    const float g     = source.sample(texSampler, uv + u.greenShift).g;
    const float4 cb   = source.sample(texSampler, uv);   // blue + alpha from centre
    const float  b    = cb.b;
    const float  a    = cb.a;

    float4 color = float4(r, g, b, a);

    if (!isHDR) {
        // SDR target (.bgra8Unorm) — clamp to displayable range.
        color = clamp(color, 0.0f, 1.0f);
    }
    // HDR path: highlights above 1.0 pass through unchanged.

    dest.write(color, gid);
}

// ===========================================================================
// 2. Analog Noise
//
// Additive grain seeded by a 2D hash of pixel position + time. Models the
// gaussian-ish noise of film grain or magnetic tape, NOT the saturated digital
// noise of broken codecs. The grain is centred at 0 (mean-zero) so it preserves
// average luminance — only local variance increases.
// ===========================================================================
struct NoiseUniforms {
    float intensity;   // 0 = bypass, sensible range up to ~0.5
    float timeSeed;    // monotonically advancing scalar — drives temporal noise
};

kernel void analogNoiseKernel(
    texture2d<float, access::read>  source [[texture(0)]],
    texture2d<float, access::write> dest   [[texture(1)]],
    constant NoiseUniforms&         u      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint W = dest.get_width();
    const uint H = dest.get_height();
    if (gid.x >= W || gid.y >= H) return;

    float4 color = source.read(gid);

    // Per-pixel hash, advanced over time by the time seed. The 73.0 multiplier
    // breaks correlation between successive frames — without it the noise
    // pattern would only shift one pixel per second.
    const float noise = hash21(float2(gid) + u.timeSeed * 73.0f);
    // Centre the noise on 0 then scale by intensity. Same offset added to R,
    // G, B so the grain is luminance-only (no colour drift).
    const float grain = (noise - 0.5f) * u.intensity;

    if (isHDR) {
        // HDR target: highlights can rise above 1.0; we still clamp from below
        // at 0 because negative-luminance is non-physical and would confuse
        // any downstream tone-mapping.
        color.rgb = max(color.rgb + float3(grain), 0.0f);
    } else {
        color.rgb = clamp(color.rgb + float3(grain), 0.0f, 1.0f);
    }

    dest.write(color, gid);
}

// ===========================================================================
// 3. Horizontal Jitter
//
// Each row gets a per-frame random horizontal shift. Emulates VHS head-switch
// noise, weak sync, or magnetic tape stretch. Implemented as a normalised-UV
// offset on the X axis only, sampled with the shared clamp-to-edge sampler so
// off-screen pixels at the swing extreme become edge-extended (no black bars).
// ===========================================================================
struct JitterUniforms {
    float intensity;   // 0 = bypass; sensible up to ~0.05 (5% of frame width)
    float timeSeed;
};

kernel void horizontalJitterKernel(
    texture2d<float, access::sample> source [[texture(0)]],
    texture2d<float, access::write>  dest   [[texture(1)]],
    constant JitterUniforms&         u      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint W = dest.get_width();
    const uint H = dest.get_height();
    if (gid.x >= W || gid.y >= H) return;

    // Per-row, per-frame hash. The 0.137 multiplier on y de-correlates adjacent
    // rows (so they don't all shift the same direction). The 13.7 multiplier
    // on timeSeed gives strong temporal evolution between frames.
    const float rowHash = hash11(float(gid.y) * 0.137f + u.timeSeed * 13.7f);
    // Map [0, 1] → [-1, 1] for signed displacement.
    const float offset  = (rowHash * 2.0f - 1.0f) * u.intensity;

    const float2 destSize = float2(float(W), float(H));
    const float2 uv       = (float2(gid) + 0.5f) / destSize;
    const float2 sampleUV = float2(uv.x + offset, uv.y);

    // Sampler is clamp_to_edge — when offset pushes UV outside [0, 1] we read
    // the nearest edge pixel rather than wrapping or zero-padding.
    float4 color = source.sample(texSampler, sampleUV);

    if (!isHDR) {
        color = clamp(color, 0.0f, 1.0f);
    }

    dest.write(color, gid);
}

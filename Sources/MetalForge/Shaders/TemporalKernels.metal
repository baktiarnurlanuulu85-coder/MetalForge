#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// TemporalEffectsPack — frame-coherent effects that need previous-frame state.
//
// Both kernels take three textures:
//   • currentSource   — this frame's input (read-only)
//   • previousFrame   — last frame's *output* (read-only, persistent buffer
//                       owned by the Swift filter, .private storage)
//   • destination     — this frame's output (write)
//
// The Swift wrapper is responsible for:
//   1. Allocating `previousFrame` lazily, matching source dims + format.
//   2. Seeding it with the source on the first frame (so result == source).
//   3. Blitting `destination → previousFrame` after the compute pass so the
//      next frame sees this frame's result as "previous".
//
// As with all our compute kernels, `isHDR` is a function constant that swaps
// between clamped (SDR target) and unclamped (HDR target, .rgba16Float)
// specialisations at PSO compile time — zero per-thread cost.
// ===========================================================================

constant bool isHDR [[function_constant(0)]];

// ===========================================================================
// 1. Motion Blur
//
// Exponential temporal accumulation. Each frame's output is a linear blend
// between the current source and the last frame's *output*. Because the
// previous output already contains a blend, the blur tail is a geometric
// series — small alphas (~0.1) give very long persistence (~10 frames of
// half-life), large alphas (~0.9) give crisp motion with subtle trails.
//
//   result = mix(previous, current, alpha)
//          = previous + (current - previous) * alpha
//
//   alpha = 1.0  →  pure current (no blur)
//   alpha = 0.0  →  freeze frame (output stays = previous forever)
//   alpha = 0.5  →  classic 2-frame box blur
// ===========================================================================
struct MotionBlurUniforms {
    float accumulationAlpha;   // 0..1
};

kernel void motionBlurKernel(
    texture2d<float, access::read>   currentSource  [[texture(0)]],
    texture2d<float, access::read>   previousFrame  [[texture(1)]],
    texture2d<float, access::write>  destination    [[texture(2)]],
    constant MotionBlurUniforms&     u              [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint W = destination.get_width();
    const uint H = destination.get_height();
    if (gid.x >= W || gid.y >= H) return;

    const float4 curr = currentSource.read(gid);
    const float4 prev = previousFrame.read(gid);

    // Linear blend in colour space. For HDR this works correctly because both
    // inputs are in the same linear space and mix() preserves the dynamic range.
    float4 color;
    color.rgb = mix(prev.rgb, curr.rgb, u.accumulationAlpha);
    color.a   = curr.a;     // preserve current alpha; trails don't fade alpha

    // Always clamp below zero — negative luminance is non-physical and would
    // corrupt the accumulation buffer over time.
    color.rgb = max(color.rgb, 0.0f);

    // Above-1.0 clamp only in SDR. HDR keeps highlights for downstream encode.
    if (!isHDR) {
        color.rgb = min(color.rgb, 1.0f);
    }

    destination.write(color, gid);
}

// ===========================================================================
// 2. Neon Trails
//
// Motion-driven additive glow with self-decaying history. Per pixel:
//
//   motion     = |current - previous_output|
//   motionMag  = average channel magnitude  (cheap luminance approximation)
//   newGlow    = neonColor * motionMag * intensity
//   fadedTrail = previous_output * decay
//   output     = current + fadedTrail * trailMix + newGlow
//
// Because `previousFrame` is the previous *output* (not previous source),
// the motion mask is "contaminated" by the existing trail — which is exactly
// what makes trails persist for several frames before fading below the
// detection threshold. Single-texture history budget achieves what would
// normally require two buffers (previous source + accumulator).
// ===========================================================================
struct NeonTrailsUniforms {
    float3 neonColor;   // 12B + 4B pad. Bright cyan default in Swift wrapper.
    float  intensity;   // 0..1.5, slider-driven
    float  decay;       // per-frame multiplicative decay, e.g. 0.85
};

kernel void neonTrailsKernel(
    texture2d<float, access::read>   currentSource  [[texture(0)]],
    texture2d<float, access::read>   previousFrame  [[texture(1)]],
    texture2d<float, access::write>  destination    [[texture(2)]],
    constant NeonTrailsUniforms&     u              [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint W = destination.get_width();
    const uint H = destination.get_height();
    if (gid.x >= W || gid.y >= H) return;

    const float3 curr = currentSource.read(gid).rgb;
    const float3 prev = previousFrame.read(gid).rgb;

    // ----- Motion detection -----
    // Per-channel absolute difference, then averaged to a scalar magnitude.
    // Averaging is faster than `length(motion)` (no sqrt) and is visually
    // indistinguishable for the small motion values we typically see.
    const float3 motion    = abs(curr - prev);
    const float  motionMag = (motion.r + motion.g + motion.b) * (1.0f / 3.0f);

    // ----- New glow contribution -----
    // Neon colour modulated by the motion magnitude and user intensity.
    const float3 newGlow = u.neonColor * (motionMag * u.intensity);

    // ----- Decay previous trail -----
    // Multiplicative fade: each frame the trail is `decay` of its previous
    // value. decay=0.85 ⇒ half-life ≈ 4 frames; decay=0.95 ⇒ ≈ 14 frames.
    const float3 fadedTrail = prev * u.decay;

    // ----- Composite -----
    // Live source dominates; the trail buffer + new glow are additive on top.
    // The 0.3 mix factor for fadedTrail prevents the buffer from saturating
    // to white when integrated over many frames of motion.
    float3 result = curr + fadedTrail * 0.3f + newGlow;

    // Negative-luminance guard (might happen if `prev` was clamped below 0
    // somehow in a previous pass — defensive).
    result = max(result, 0.0f);

    if (!isHDR) {
        // SDR target: clip highlights to displayable range so additive
        // accumulation doesn't blow out the .bgra8Unorm output.
        result = min(result, 1.0f);
    }
    // HDR target: highlights up to several × reference white are preserved.

    destination.write(float4(result, 1.0f), gid);
}

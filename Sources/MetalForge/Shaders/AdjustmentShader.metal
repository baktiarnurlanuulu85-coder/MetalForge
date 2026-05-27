#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Function constant — resolved at PSO compile time, not per-thread.
// The Metal compiler emits two distinct AIR specialisations (SDR vs HDR)
// when the AdjustmentFilter creates its two MTLComputePipelineState objects.
// All `if (!isHDR)` branches below are eliminated as dead code in each variant,
// so there is zero runtime branching cost.
// ---------------------------------------------------------------------------
constant bool isHDR [[function_constant(0)]];

// Must mirror AdjustmentUniforms in AdjustmentFilter.swift (same field order, same types).
struct AdjustmentUniforms {
    float brightness;   // Additive offset: -1.0 … 0.0 (identity) … 1.0
    float contrast;     // Scale around 0.5:  0.0 … 1.0 (identity) … 4.0
};

// ---------------------------------------------------------------------------
// adjustmentKernel
// One thread per output texel. Dispatched with nonUniform threadgroups so the
// bounds check below only fires for the partial tile at the texture edge.
// ---------------------------------------------------------------------------
kernel void adjustmentKernel(
    texture2d<float, access::read>  inTexture  [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant AdjustmentUniforms&    u          [[buffer(0)]],
    uint2                           gid        [[thread_position_in_grid]])
{
    // Guard: nonUniform dispatch may launch threads outside texture bounds.
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    float4 color = inTexture.read(gid);

    // --- Brightness ---
    // Simple additive shift in linear light (no gamma conversion for perf).
    // For colour-accurate work, linearise before shift and re-apply gamma after.
    color.rgb += u.brightness;

    // --- Contrast ---
    // Scale around mid-gray 0.5:  out = (in − 0.5) × contrast + 0.5
    // At contrast = 1.0 this is the identity. At 0 everything maps to 0.5 (grey).
    color.rgb = (color.rgb - 0.5f) * u.contrast + 0.5f;

    // --- Range clamp (SDR only) ---
    // In SDR mode, clamp to [0, 1] to match the .bgra8Unorm output range.
    // In HDR mode, leave highlights above 1.0 intact — the .rgba16Float
    // destination preserves them and downstream filters / tone-mapping rely on
    // the linear scene-referred values. The compiler removes the dead branch
    // for whichever variant of the PSO we are running.
    if (!isHDR) {
        color = clamp(color, 0.0f, 1.0f);
    }

    outTexture.write(color, gid);
}

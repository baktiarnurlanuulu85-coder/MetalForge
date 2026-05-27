#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// Uniforms — must mirror DisplayUniforms in MetalForgeView.swift.
//
// The transform from viewport texCoord → source texCoord is expressed as a
// pair of rectangles. This single formulation covers all three scaling modes:
//
//   • aspectFit  : srcRect = [0,1]², viewportRect is a *sub-rectangle*
//                  (outside viewportRect → black letter/pillarbox bars)
//   • aspectFill : viewportRect = [0,1]², srcRect is a *sub-rectangle*
//                  (we sample only the inner crop of the source)
//   • stretch    : both rects are [0,1]² (identity)
// ===========================================================================
struct DisplayUniforms {
    float2 srcRectMin;
    float2 srcRectMax;
    float2 viewportRectMin;
    float2 viewportRectMax;
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// ===========================================================================
// Fullscreen triangle vertex shader.
//
// Three vertices, no vertex buffer needed — clip positions emitted from vid.
//   vid 0: clip (-1, -1)  texCoord (0, 1)   ← bottom-left of viewport
//   vid 1: clip ( 3, -1)  texCoord (2, 1)
//   vid 2: clip (-1,  3)  texCoord (0,-1)
//
// The visible [-1,1]² region of clip space rasterises to texCoord ∈ [0,1]² with
// Y flipped (Metal's texture coordinate origin is top-left, not bottom-left).
// This is the standard "AMD" / Sebastian Aaltonen fullscreen-tri pattern: a
// single triangle is faster than two (no edge shared, no extra rasteriser pass).
// ===========================================================================
vertex VertexOut displayVertex(uint vid [[vertex_id]]) {
    VertexOut out;
    float2 pos = float2(
        (vid == 1) ? 3.0f : -1.0f,
        (vid == 2) ? 3.0f : -1.0f
    );
    out.position = float4(pos, 0.0f, 1.0f);
    out.texCoord = float2(
        (vid == 1) ?  2.0f : 0.0f,
        (vid == 2) ? -1.0f : 1.0f
    );
    return out;
}

// ===========================================================================
// Display fragment shader.
//
// 1. If viewport texCoord is outside viewportRect → return black (letterbox).
// 2. Otherwise, remap viewport texCoord into srcRect via the rect transform
//    and sample the source texture with a linear sampler.
// ===========================================================================
fragment float4 displayFragment(
    VertexOut                         in        [[stage_in]],
    texture2d<float, access::sample>  sourceTex [[texture(0)]],
    constant DisplayUniforms&         u         [[buffer(0)]],
    sampler                           samp      [[sampler(0)]])
{
    const float2 vp = in.texCoord;

    // Outside-viewport rejection. Using a single component-wise compare avoids
    // branch divergence on Apple GPUs when the entire warp is inside the rect.
    if (vp.x < u.viewportRectMin.x || vp.x > u.viewportRectMax.x ||
        vp.y < u.viewportRectMin.y || vp.y > u.viewportRectMax.y) {
        // Opaque black — matches the MTKView clear colour exactly so there is
        // no visible seam between the cleared area and the fragment-rendered area.
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    // Map viewport UV → source UV via the rect transform.
    //   t ∈ [0,1]² is the position within viewportRect.
    //   srcUV ∈ srcRect (also inside [0,1]² for any valid scaling mode).
    const float2 t     = (vp - u.viewportRectMin) / (u.viewportRectMax - u.viewportRectMin);
    const float2 srcUV = u.srcRectMin + t * (u.srcRectMax - u.srcRectMin);

    return sourceTex.sample(samp, srcUV);
}

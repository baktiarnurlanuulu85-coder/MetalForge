#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// HDR Transfer Functions — Rec. ITU-R BT.2100
//
// This file implements the exact mathematical EOTFs (decoding) and OETFs
// (encoding) for the two HDR transfer functions standardised in BT.2100:
//
//   • PQ (Perceptual Quantizer)     — SMPTE ST.2084
//   • HLG (Hybrid Log-Gamma)        — ARIB STD-B67
//
// All formulas use the exact constants from BT.2100 Table 4 — no truncation,
// no curve fits. The literal definitions are:
//
//   PQ EOTF:  L = ( (max(V^(1/m₂) - c₁, 0)) / (c₂ - c₃·V^(1/m₂)) )^(1/m₁)
//   PQ OETF:  V = ( (c₁ + c₂·L^m₁) / (1 + c₃·L^m₁) )^m₂
//
//   HLG OETF^-1:
//             E = (E')² / 3                            for  0 ≤ E' ≤ 0.5
//             E = (exp((E' − c)/a) + b) / 12           for  0.5 < E' ≤ 1
//
//   HLG OETF:
//             E' = 0.5 · sqrt(12·E)                    for  0 ≤ E ≤ 1/12
//             E' = a · ln(12·E − b) + c                for  1/12 < E ≤ 1
//
// Scale conventions in this file (literal BT.2100):
//   • PQ:  L = 1.0 corresponds to 10,000 cd/m² (system peak). SDR diffuse white
//          (100 nits) sits at L ≈ 0.01.
//   • HLG: scene-relative, L = 1.0 corresponds to the signal peak. The OOTF
//          (which would scale scene→display luminance) is *not* applied here —
//          it is a display-side step and must run on the presentation stage,
//          not inside the intermediate processing pipeline.
// ===========================================================================

// -------------------- PQ constants (BT.2100 Table 4) --------------------
constant float kPQ_m1 = 0.1593017578125f;   // = 2610  / 16384
constant float kPQ_m2 = 78.84375f;          // = 2523  / 4096  × 128
constant float kPQ_c1 = 0.8359375f;         // = 3424  / 4096
constant float kPQ_c2 = 18.8515625f;        // = 2413  / 4096  × 32
constant float kPQ_c3 = 18.6875f;           // = 2392  / 4096  × 32

// -------------------- HLG constants (BT.2100 Table 5) -------------------
constant float kHLG_a = 0.17883277f;
constant float kHLG_b = 0.28466892f;        // = 1 − 4a
constant float kHLG_c = 0.55991073f;        // = 0.5 − a·ln(4a)

// ===========================================================================
// PQ EOTF (decode): non-linear V → linear L
// ===========================================================================
static inline float3 pqEOTF(float3 v) {
    // Clamp to [0,1]; outside that range the inverse pow becomes ill-defined
    // (den could change sign for V > c₂/c₃ ≈ 1.009).
    v = clamp(v, 0.0f, 1.0f);
    float3 vp = pow(v, 1.0f / kPQ_m2);
    // num: enforce non-negativity for V near 0 (numerical noise can yield −ε).
    float3 num = max(vp - kPQ_c1, 0.0f);
    float3 den = kPQ_c2 - kPQ_c3 * vp;       // strictly positive on [0,1]
    return pow(num / den, 1.0f / kPQ_m1);
}

// ===========================================================================
// PQ OETF (encode): linear L → non-linear V
// ===========================================================================
static inline float3 pqOETF(float3 l) {
    l = max(l, 0.0f);                         // fractional pow needs L ≥ 0
    float3 lp = pow(l, kPQ_m1);
    return pow((kPQ_c1 + kPQ_c2 * lp) / (1.0f + kPQ_c3 * lp), kPQ_m2);
}

// ===========================================================================
// HLG inverse OETF (decode): non-linear E' → scene-linear E (per channel)
// ===========================================================================
static inline float hlgInverseOETFScalar(float v) {
    v = clamp(v, 0.0f, 1.0f);
    float low  = (v * v) * (1.0f / 3.0f);
    // Use safe domain for log arg in the high branch — even though `select`
    // discards the unused branch, computing exp on a valid value is cheap and
    // avoids potential NaN bit-leaks on some hardware.
    float high = (exp((v - kHLG_c) / kHLG_a) + kHLG_b) * (1.0f / 12.0f);
    return select(low, high, v > 0.5f);
}

static inline float3 hlgInverseOETF(float3 v) {
    return float3(hlgInverseOETFScalar(v.r),
                  hlgInverseOETFScalar(v.g),
                  hlgInverseOETFScalar(v.b));
}

// ===========================================================================
// HLG OETF (encode): scene-linear E → non-linear E' (per channel)
// ===========================================================================
static inline float hlgOETFScalar(float e) {
    e = max(e, 0.0f);
    float low  = 0.5f * sqrt(12.0f * e);
    // log arg in the high branch must be > 0. For e ≤ 1/12, (12e − b) can be
    // negative — we clamp to a small positive to guarantee a finite result,
    // even though the select discards this branch in that domain.
    float logArg = max(12.0f * e - kHLG_b, 1e-6f);
    float high   = kHLG_a * log(logArg) + kHLG_c;
    return select(low, high, e > (1.0f / 12.0f));
}

static inline float3 hlgOETF(float3 l) {
    return float3(hlgOETFScalar(l.r),
                  hlgOETFScalar(l.g),
                  hlgOETFScalar(l.b));
}

// ===========================================================================
// COMPUTE KERNELS
//
// All four kernels share the same shape:
//   • one thread per destination texel
//   • non-uniform dispatch (in-shader bounds guard required)
//   • preserve alpha untouched
//   • read/write through .rgba16Float (HDR working format)
// ===========================================================================

kernel void pqDecodeKernel(
    texture2d<float, access::read>  inTex  [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    uint2                           gid    [[thread_position_in_grid]])
{
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;
    float4 color = inTex.read(gid);
    color.rgb = pqEOTF(color.rgb);
    outTex.write(color, gid);
}

kernel void pqEncodeKernel(
    texture2d<float, access::read>  inTex  [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    uint2                           gid    [[thread_position_in_grid]])
{
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;
    float4 color = inTex.read(gid);
    color.rgb = pqOETF(color.rgb);
    outTex.write(color, gid);
}

kernel void hlgDecodeKernel(
    texture2d<float, access::read>  inTex  [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    uint2                           gid    [[thread_position_in_grid]])
{
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;
    float4 color = inTex.read(gid);
    color.rgb = hlgInverseOETF(color.rgb);
    outTex.write(color, gid);
}

kernel void hlgEncodeKernel(
    texture2d<float, access::read>  inTex  [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    uint2                           gid    [[thread_position_in_grid]])
{
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;
    float4 color = inTex.read(gid);
    color.rgb = hlgOETF(color.rgb);
    outTex.write(color, gid);
}

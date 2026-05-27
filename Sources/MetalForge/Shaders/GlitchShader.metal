#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Function constant — bound at PSO specialisation time.
// SDR PSO is compiled with isHDR = false (final clamp to [0,1] kept);
// HDR PSO is compiled with isHDR = true (clamp removed, highlights preserved).
// ---------------------------------------------------------------------------
constant bool isHDR [[function_constant(0)]];

// Must mirror GlitchUniforms in GlitchFilter.swift (field order and types match).
struct GlitchUniforms {
    float time;         // Elapsed seconds — drives all temporal variation
    float intensity;    // 0.0 = bypass, 1.0 = full effect
    uint  frameIndex;   // Per-frame counter — used as an additional seed
};

// ===========================================================================
// PSEUDO-RANDOM NUMBER GENERATION
// Wang hash family — integer-domain, no texture lookup, GPU-friendly.
// ===========================================================================

// Core Wang hash: maps any uint to a well-distributed uint in full [0, 2^32) range.
static inline uint wangHash(uint n) {
    n = (n ^ 61u) ^ (n >> 16u);
    n = n + (n << 3u);
    n = n ^ (n >> 4u);
    n = n * 0x27d4eb2du;
    n = n ^ (n >> 15u);
    return n;
}

// Map a float's bit-pattern through the Wang hash, return [0, 1).
static inline float hashF(float x) {
    return float(wangHash(as_type<uint>(x))) * (1.0f / 4294967296.0f);
}

// 2-D hash: combine two floats, return [0, 1).
static inline float hash2F(float x, float y) {
    // Shift y into a non-overlapping mantissa region before adding.
    return hashF(x + y * 127.1f + 311.7f);
}

// ===========================================================================
// CRT SCANLINE MODEL
// Mimics horizontal phosphor line structure of a CRT display.
// Returns a [0, 1] multiplier (< 1 at scanline boundaries).
// ===========================================================================
static inline float scanlineMultiplier(float pixelY) {
    // Period = 3 pixels: two "phosphor" rows (bright) + one gap row (dimmer).
    // smoothstep avoids aliasing at the gap edge.
    float t = fmod(pixelY, 3.0f);
    float gap = smoothstep(1.5f, 2.5f, t);   // 0 during phosphor, 1 at gap
    return 1.0f - 0.14f * gap;
}

// ===========================================================================
// VIGNETTE
// Soft edge darkening using a product of two parabolas (one per axis).
// ===========================================================================
static inline float vignette(float2 uv, float strength) {
    // uv * (1 - uv) peaks at 0.25; scale to [0, 1].
    float2 v = uv * (1.0f - uv);
    // pow with small exponent (0.25) gives a gentle, wide falloff.
    float vig = pow(v.x * v.y * 16.0f, 0.25f);
    return mix(1.0f, vig, strength);
}

// ===========================================================================
// UV → CLAMPED TEXEL COORDINATE
// Clamps to [0, W-1] × [0, H-1] instead of repeating or leaving undefined.
// ===========================================================================
static inline uint2 uvToTexel(float2 uv, uint W, uint H) {
    int2 c = int2(uv * float2(float(W), float(H)));
    c = clamp(c, int2(0, 0), int2(int(W) - 1, int(H) - 1));
    return uint2(c);
}

// ===========================================================================
// glitchKernel — Cinematic Glitch + Chromatic Aberration
// One thread per output texel. Dispatched non-uniformly; bounds guard below.
// ===========================================================================
kernel void glitchKernel(
    texture2d<float, access::read>  inTexture  [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant GlitchUniforms&        u          [[buffer(0)]],
    uint2                           gid        [[thread_position_in_grid]])
{
    const uint W = outTexture.get_width();
    const uint H = outTexture.get_height();
    if (gid.x >= W || gid.y >= H) return;

    const float2 texSize = float2(float(W), float(H));
    const float2 uv      = float2(gid) / texSize;   // Normalised [0, 1]

    // -----------------------------------------------------------------------
    // HORIZONTAL GLITCH BANDS
    // Divide the image into `numBands` horizontal slices. Each slice gets an
    // independent random offset that changes ~`tempo` times per second.
    // -----------------------------------------------------------------------
    const float numBands = 24.0f;
    const float tempo    = 7.0f;    // How often bands switch (Hz)

    float bandRow  = floor(uv.y * numBands);
    // Quantise time so all pixels in a band see the same offset this frame.
    float timeTick = floor(u.time * tempo);
    // Each band gets a unique seed by combining its row index and the time tick.
    float bandSeed  = hashF(bandRow * 17.3f + timeTick * 83.7f);

    // A band "fires" when its random value exceeds the no-glitch threshold.
    // Higher intensity lowers the threshold → more bands fire per frame.
    float fireThreshold = 1.0f - u.intensity * 0.55f;
    float fired = step(fireThreshold, bandSeed);

    // Displacement in normalised UV space (max ±4 % of texture width when fired).
    float bandShift = (hashF(bandSeed * 31.4f) * 2.0f - 1.0f) * 0.04f * fired;

    // -----------------------------------------------------------------------
    // CHROMATIC ABERRATION
    // R channel shifted right; B channel shifted left; G follows the band shift.
    // `caBase` provides a baseline separation even when no band fires.
    // -----------------------------------------------------------------------
    float caBase = u.intensity * 0.007f;

    float2 uvR = float2(uv.x + caBase + bandShift, uv.y);
    float2 uvG = float2(uv.x          + bandShift, uv.y);
    float2 uvB = float2(uv.x - caBase + bandShift, uv.y);

    // Sample each channel separately at its offset position.
    float r = inTexture.read(uvToTexel(uvR, W, H)).r;
    float g = inTexture.read(uvToTexel(uvG, W, H)).g;
    float b = inTexture.read(uvToTexel(uvB, W, H)).b;
    float a = inTexture.read(gid).a;                    // Alpha unmodified

    float4 color = float4(r, g, b, a);

    // -----------------------------------------------------------------------
    // CRT SCANLINE OVERLAY
    // Multiplicative darkening; independent of intensity so the CRT feel is
    // always present when the filter is active. Scale by intensity to taste.
    // -----------------------------------------------------------------------
    float sl = scanlineMultiplier(float(gid.y));
    // Blend between identity (1.0) and full scanline at `intensity`.
    color.rgb *= mix(1.0f, sl, u.intensity);

    // -----------------------------------------------------------------------
    // VIGNETTE
    // -----------------------------------------------------------------------
    color.rgb *= vignette(uv, u.intensity * 0.7f);

    // -----------------------------------------------------------------------
    // SPORADIC NOISE FLASH
    // ~3 % of frames trigger a brief full-frame brightness spike that overlays
    // high-frequency pixel noise, simulating an analogue signal burst.
    // -----------------------------------------------------------------------
    // flashClock changes 12 times per second; step fires when > 97 th-percentile.
    float flashClock   = floor(u.time * 12.0f);
    float flashTrigger = step(0.97f, hashF(flashClock * 4.3f + float(u.frameIndex) * 0.001f));
    // Per-pixel noise varies spatially and temporally.
    float pixelNoise   = hash2F(float(gid.x) * 0.0073f, float(gid.y) * 0.0137f + u.time);
    color.rgb = mix(color.rgb, float3(pixelNoise), flashTrigger * u.intensity * 0.18f);

    // -----------------------------------------------------------------------
    // VERTICAL HOLD ROLL
    // Occasionally shifts a thin horizontal strip up or down by a few pixels,
    // like a de-synced vertical hold on analogue tape.
    // -----------------------------------------------------------------------
    float rollClock    = floor(u.time * 2.0f);
    float rollFire     = step(0.92f, hashF(rollClock * 7.1f)) * u.intensity;
    float rollCenter   = hashF(rollClock * 3.7f);              // [0, 1] — strip centre
    float rollWidth    = 0.04f;                                 // ±4 % of height
    float inRoll       = step(0.0f, rollWidth - abs(uv.y - rollCenter));
    float rollShift    = (hashF(rollClock) * 2.0f - 1.0f) * 0.015f * rollFire * inRoll;
    // Re-read green with the vertical shift applied (colour shift inside the roll).
    float2 uvRoll = float2(uv.x, uv.y + rollShift);
    color.g = mix(color.g,
                  inTexture.read(uvToTexel(uvRoll, W, H)).g,
                  inRoll * rollFire);

    // --- Range clamp (SDR only) ---
    // The chromatic aberration, scanline, vignette, flash and roll passes are
    // all multiplicative or mix-based — they preserve the HDR character of
    // values > 1.0. The only HDR-killing op is the final clamp, which we gate
    // on the function constant so the HDR PSO emits an unclamped write.
    if (!isHDR) {
        color = clamp(color, 0.0f, 1.0f);
    }
    outTexture.write(color, gid);
}

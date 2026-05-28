# MetalForge

**A zero-copy, HDR-native video processing engine for iOS and macOS — built on Metal compute shaders, Apple's hardware video pipeline, and strict linear-light mathematics.**

MetalForge ingests live camera frames or compressed video streams, runs an arbitrary chain of GPU compute filters over them in either SDR (BT.709) or true HDR (BT.2100 PQ / HLG) working space, and feeds the result simultaneously to an EDR-aware on-screen view and an `AVAssetWriter`-backed recorder — without a single CPU-side pixel copy from capture to display to file.

---

## Key Highlights

- **Zero-copy throughout.** Camera `CVPixelBuffer`s are wrapped as `MTLTexture` via `CVMetalTextureCache`. The recorder owns a second cache attached to the `AVAssetWriterInputPixelBufferAdaptor`'s pool, so processed frames are written by the GPU directly into the encoder's input surfaces. No `memcpy` exists anywhere on the live frame path.
- **Hardware-accelerated 3D LUTs.** `texture3d<float, access::sample>` with `filter::linear` performs trilinear interpolation across 8 cube vertices in a single TMU instruction. Padded edge-coordinate mapping eliminates boundary artefacts at the cube faces.
- **Strict thread safety with custom refcounting.** Shared textures consumed by both display and recorder are tracked by an `NSLock`-guarded `[ObjectIdentifier: Int]` map. A texture returns to the pool only after the last consumer's GPU completion handler fires.
- **Linear-light mathematics throughout the HDR path.** PQ and HLG signals are decoded via literal BT.2100 EOTFs (no approximations), processed in linear scene-referred space, and re-encoded for display or file output. Contrast pivots at scene-referred `0.18` middle grey, not the perceptual `0.5` that would crush shadows under linear math.
- **Cross-platform `UIView` / `NSView` architecture.** `MetalForgeView` subclasses `MTKView` directly — itself cross-platform — and ships with SwiftUI `UIViewRepresentable` / `NSViewRepresentable` wrappers in a single conditional file. No platform-specific control-flow leaks into user code.
- **Function-constant PSO specialisation.** Every filter compiles two `MTLComputePipelineState` variants on construction (`isHDR = false` and `isHDR = true`) via `[[function_constant(0)]]`. The unused clamp / luma-weights branch is eliminated as dead code at AIR specialisation time. Zero per-thread cost for the SDR/HDR distinction.

---

## Architecture & Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│  AVCaptureSession                                            │
│  • 10-bit 4:2:0 YUV BiPlanar (HDR HEVC) or 8-bit / BGRA      │
│  • alwaysDiscardsLateVideoFrames = true                      │
│  • dedicated captureQueue (userInitiated QoS)                │
└──────────────────────────┬───────────────────────────────────┘
                           │ CMSampleBuffer + PTS
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  MetalForgeCaptureManager                                    │
│  • AVCaptureVideoDataOutputSampleBufferDelegate              │
│  • AVCaptureAudioDataOutputSampleBufferDelegate              │
└──────────────────────────┬───────────────────────────────────┘
                           │ CVPixelBuffer
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  MetalForgeEngine.makeTextures(from: pixelBuffer)            │
│  • CVMetalTextureCacheCreateTextureFromImage (zero-copy)     │
│  • Luma plane → .r8Unorm  or  .r16Unorm                      │
│  • Chroma plane → .rg8Unorm or .rg16Unorm                    │
└──────────────────────────┬───────────────────────────────────┘
                           │ (luma, chroma) MTLTextures
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  MetalForgePipeline.process()                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  0.  YUVToRGBConverter                                 │  │
│  │      • Matrix from kCVImageBufferYCbCrMatrixKey        │  │
│  │      • Chroma via filter::linear → free 4:2:0 → 4:4:4  │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │  1.  HDRDecodeFilter         (HDR sources only)        │  │
│  │      • PQ / HLG EOTF → linear scene light              │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │  2…N. User filters         (operate in linear light)   │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │  N+1. HDREncodeFilter        (HDR sources only)        │  │
│  │      • Linear → PQ / HLG OETF (display-ready)          │  │
│  └────────────────────────────────────────────────────────┘  │
│  • TexturePool recycles .rgba16Float / .bgra8Unorm           │
│  • commandBuffer.waitUntilCompleted (sync barrier)           │
└──────────────────────────┬───────────────────────────────────┘
                           │ MTLTexture
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  _pendingCount[ObjectIdentifier(texture)] = N                │
│  N = 2 if recording, else 1   (NSLock-protected)             │
└──────────┬───────────────────────────────────┬───────────────┘
           │ DispatchQueue.main.async          │ appendVideoFrame
           ▼                                   ▼
┌──────────────────────────┐  ┌───────────────────────────────┐
│  MetalForgeView          │  │  MetalForgeRecorder           │
│  (MTKView, paused mode)  │  │  (own videoQueue + audioQueue)│
│  • CAMetalLayer:         │  │  • 2nd CVMetalTextureCache    │
│      wantsEDRContent     │  │  • AVAssetWriterInputPBA      │
│      itur_2100_PQ / sRGB │  │  • SDR: MTLBlit (BGRA)        │
│  • Aspect-rect transform │  │  • HDR: RGBToYUVConverter     │
│  • Fullscreen triangle   │  │       (luma + chroma compute) │
│  • addCompletedHandler   │  │  • adaptor.append(buf, pts)   │
│      → recycleHandler    │  │  • audioInput passthrough     │
└────────────┬─────────────┘  └──────────────┬────────────────┘
             │ recycleHandler(texture)       │ recycleHandler
             └────────────────┬──────────────┘
                              ▼
            ┌──────────────────────────────────────┐
            │  handleRecycle(texture):             │
            │  decrement _pendingCount; when == 0  │
            │  pipeline.recycle(texture)           │
            │    → TexturePool reuses next frame   │
            └──────────────────────────────────────┘
```

### The "Fake No-Op Bypass" Filter-Switching Pattern

A naive demo would switch between filters by mutating `pipeline.filters` from the main thread when the user taps a UI control. This races with the background `captureQueue`, which is iterating that same array inside `pipeline.process(pixelBuffer:)`. Swift arrays are copy-on-write, so the iteration won't crash — but the GPU encode for the in-flight frame may include the wrong filter, or skip one entirely.

MetalForge sidesteps the race entirely by keeping **every filter in the pipeline chain simultaneously, permanently**, and switching effects by zeroing the parameters of inactive filters:

| Filter | No-op parameter setting | Why it's identity |
|--------|-------------------------|-------------------|
| `GlitchFilter` | `intensity = 0` | Internal `if (intensity == 0) return source;` early-out in MSL |
| `ChromaticAberrationFilter` | `redShift = greenShift = (0,0)` | All three TMU samples land at the same UV |
| `AnalogNoiseFilter` | `noiseIntensity = 0` | Additive `grain = 0` → output = source |
| `HorizontalJitterFilter` | `jitterIntensity = 0` | Sample offset is zero → uv unchanged |
| `MotionBlurFilter` | `accumulationAlpha = 1.0` | `mix(prev, curr, 1.0) = curr` |
| `NeonTrailsFilter` | `intensity = 0` (+ `clearHistory()`) | No new glow contribution; trail buffer dropped |
| `MetalForgeLUTFilter` | `intensity = 0` | `mix(base, graded, 0) = base`; highlights re-added |
| `ColorCorrectionFilter` | `exposure 0 / contrast 1 / saturation 1` | All operations collapse to identity |

The cost is one extra compute dispatch per inactive filter — at 1080p on Apple Silicon each pass runs in ≈0.3 ms, so eight passes total fit in ≈2.4 ms, comfortably inside a 16.6 ms 60 Hz budget. This is the price paid for guaranteed-race-free filter switching with zero pipeline-mutation synchronisation primitives.

---

## Feature & Shader Library

| Feature / Filter | Core Optimisation / Math Principle | HDR vs SDR Behaviour |
|------------------|-----------------------------------|----------------------|
| **Zero-copy YUV ingestion** | `CVMetalTextureCacheCreateTextureFromImage` wraps the camera's IOSurface as two `MTLTexture` views — luma `.r8Unorm` / `.r16Unorm`, chroma `.rg8Unorm` / `.rg16Unorm`. No staging buffer, no `replace()`. | 8-bit YUV → SDR working space (`.bgra8Unorm` intermediate). 10-bit YUV → HDR working space (`.rgba16Float` intermediate). 10-bit samples are stored in the high 10 bits of a 16-bit container per Apple's convention; the colour matrix absorbs the ~0.1 % normalisation delta. |
| **YUV → RGB hardware chroma upsample** | Chroma plane sampled via `texture2d<float, access::sample>` with an inline `filter::linear, address::clamp_to_edge` sampler. The TMU performs full 4:2:0 → 4:4:4 bilinear upsampling in a single instruction per pixel. Colour matrix selected at runtime from `kCVImageBufferYCbCrMatrixKey` (BT.601, BT.709, or BT.2020); range from the pixel-format type. | Same kernel path for both ranges. BT.2020 video-range matrix carries the ×219/255 luma and ×224/255 chroma compression internally — the shader only subtracts the `(16/255, 128/255, 128/255)` offset vector. |
| **PQ / HLG transfer functions** | Literal BT.2100 EOTFs and OETFs — no curve fits. PQ uses the five canonical constants (`m₁`, `m₂`, `c₁`, `c₂`, `c₃`); HLG uses (`a`, `b`, `c`). Piecewise HLG branches resolved via `select()` for SIMD-divergence-free execution. Detection of PQ vs HLG comes from `kCVImageBufferTransferFunctionKey` on the pixel buffer. | Runs only when `workingColorSpace.isHDR`. PQ outputs linear values where `1.0 ≡ 10 000 cd/m²` (literal BT.2100 scale). HLG outputs scene-relative linear values; OOTF is deliberately omitted — it is a display-side scaling that depends on the target panel's peak luminance. |
| **Chromatic Aberration** *(Analog Pack)* | Three TMU samples per pixel: R at `uv + redShift`, G at `uv + greenShift`, B+alpha at `uv`. Shift vectors are normalised UVs (typical range ±0.02). Inline `filter::linear, address::clamp_to_edge` sampler. | HDR variant skips the `clamp(color, 0, 1)` so above-white dispersion propagates as HDR highlights. SDR variant clamps. |
| **Analog Noise** *(Analog Pack)* | Hoskins' "Hash Without Sine" (`hash21`) per pixel, seeded by `gid + timeSeed × 73.0`. Mean-zero additive grain — preserves average luminance, only adds local variance. Auto-advancing internal `startDate`-driven time, with an optional user-provided `timeSeed` offset for deterministic playback. | HDR: `max(color + grain, 0)` — only floor-clamps at zero (no above-1.0 ceiling). SDR: full `clamp` to `[0, 1]`. |
| **Horizontal Jitter** *(Analog Pack)* | Per-row `hash11(y × 0.137 + time × 13.7)` produces a signed normalised X offset. Sampled with `clamp_to_edge` so the swing extremes pull edge pixels rather than wrap or zero-pad — avoids the dark-stripe seam that wrap would cause. | Same kernel path; SDR adds a final `[0, 1]` clamp on the sampled colour. |
| **Motion Blur** *(Temporal Pack)* | `mix(previousOutput, current, accumulationAlpha)`. Persistent `.private` storage `MTLTexture` owned by the filter, allocated lazily at first encode, dimensions / format matched to source. Lifecycle: seed via `MTLBlitCommandEncoder` on first frame → compute pass → second blit saves `destination → previousTexture`. `NSLock` guards the texture reference against concurrent `clearHistory()` calls. | Both variants perform the same linear blend. HDR keeps highlights above 1.0; SDR clamps. Negative-luminance protection (`max(color, 0)`) is always applied to prevent accumulation buffer corruption. |
| **Neon Trails** *(Temporal Pack)* | Single-buffer compromise: `previousTexture` stores the previous output (not previous source). `motion = abs(current - previousOutput)` is "contaminated" by existing trails — which is exactly what makes trails persist across multiple frames before fading below the detection threshold. Decay multiplier per frame; additive composite `current + decayedTrail × 0.3 + neonColor × motionMag × intensity`. | HDR: additive glow can blow into HDR highlights — physically and aesthetically correct for neon. SDR: clamps the final composite to `[0, 1]`. |
| **Color Correction** *(Color Grading Pack)* | **Exposure**: `rgb *= exp2(stops)` — a single hardware instruction on Apple GPU, faster than `pow(2, stops)`. **Contrast**: `(rgb − 0.18) × contrast + 0.18` — pivots at scene-referred 18 % middle grey, not 0.5. (In linear space, 0.5 is perceptually ≈73 %; pivoting there crushes shadows under any boost.) **Saturation**: `mix(luma, rgb, saturation)` with luma weights `(0.2126, 0.7152, 0.0722)` for BT.709 or `(0.2627, 0.6780, 0.0593)` for BT.2020 — selected at PSO compile time via `[[function_constant(0)]]`, so the unused branch is dead-stripped from AIR. **Temperature**: simple R/B push, ±30 % at full deflection. | The function-constant specialisation is the *entire* HDR difference — BT.2020 luma weights are perceptually correct for the wide gamut. All other operations are identical in both variants; the SDR variant adds a final `min(rgb, 1.0)` highlight clip. |
| **3D LUT** *(Color Grading Pack)* | `texture3d<float, access::sample>` with inline `filter::linear` sampler — hardware trilinear interpolation across 8 cube vertices in a single TMU instruction. Padded edge-coordinate mapping: `uvw = rgb × (N − 1)/N + 0.5/N`, so sample positions land on texel centres at the cube boundaries rather than reaching beyond them and pulling clamp-to-edge values. Cube uploaded once via `texture.replace()`; `.shared` storage on iOS / visionOS / tvOS, `.managed` on macOS (Metal handles CPU↔GPU sync automatically for `replace()` writes). Four built-in preset generators (`.identity`, `.warm`, `.cool`, `.sepia`) eliminate any external `.cube` asset dependency. | HDR uses a **base/highlight split**: `base = clamp(rgb, 0, 1)` is graded by the LUT; `highlight = max(rgb − 1, 0)` bypasses the cube entirely and is added back to the graded result. The LUT — defined only over the `[0, 1]` cube — affects the visible range while HDR peak highlights propagate untouched. SDR variant uses `saturate(rgb)` and `highlight = 0`. |

---

## Multi-Threaded Sync & Lifecycle Management

### Texture Refcount for Shared Consumers

Each processed `MTLTexture` returned by `pipeline.process(pixelBuffer:)` is consumed by **two** independent GPU pipelines when recording is active:

1. **`MetalForgeView`** — schedules a render pass that *reads* the texture as a fragment-shader source. Completion is signalled by `MTLCommandBuffer.addCompletedHandler` firing on Metal's completion thread.
2. **`MetalForgeRecorder`** — schedules either a `MTLBlitCommandEncoder` copy (SDR → BGRA pool buffer) or an `RGBToYUVConverter` compute pair (HDR → 10-bit YUV bi-planar pool buffer). Completion is signalled by a separate command buffer on the recorder's `videoQueue`.

Naive recycling would race: whichever consumer finishes first would return the texture to `TexturePool`, which would re-hand it to the next `pipeline.process` call for write — corrupting the in-flight read by the slower consumer.

MetalForge resolves this with a counter dictionary on the demo controller:

```swift
private let stateLock = NSLock()
nonisolated(unsafe) private var _pendingCount: [ObjectIdentifier: Int] = [:]

func dispatchFrame(texture: MTLTexture, recording: Bool) {
    stateLock.lock()
    _pendingCount[ObjectIdentifier(texture as AnyObject)] = recording ? 2 : 1
    stateLock.unlock()
    // both consumers receive the same `texture` reference
}

nonisolated func handleRecycle(texture: MTLTexture) {
    let key = ObjectIdentifier(texture as AnyObject)
    stateLock.lock()
    let remaining = (_pendingCount[key] ?? 1) - 1
    if remaining <= 0 {
        _pendingCount.removeValue(forKey: key)
        stateLock.unlock()
        pipeline.recycle(texture)   // only now does the pool reclaim
    } else {
        _pendingCount[key] = remaining
        stateLock.unlock()
    }
}
```

Both `view.recycleHandler` and `recorder.recycleHandler` are wired to `handleRecycle`. The reference count is initialised *atomically with the dispatch decision* under the same lock, so we never end up with a count of 2 but only one consumer actually firing back, or vice versa.

### The Snapshot Pattern with `nonisolated(unsafe)`

`MetalForgeDemoController` is `@MainActor` because it conforms to `ObservableObject` and exposes `@Published` properties to SwiftUI. But the capture pipeline runs on a background dispatch queue and needs to read the active `MetalForgeRecorder` reference on every frame, without paying the cost of a `Task { @MainActor in … }` hop (which would add hundreds of microseconds and burn through the GCD scheduler).

The pattern:

```swift
@MainActor
public final class MetalForgeDemoController: ObservableObject {

    private let stateLock = NSLock()

    // Compiler can't prove the invariant; we hold it manually with stateLock.
    nonisolated(unsafe) private var _recorderSnapshot: MetalForgeRecorder?

    // Main-actor mutation (writing the property is naturally synchronised
    // by the actor; we additionally lock to also serialise with the reader).
    private func startRecording() {
        // … construct recorder …
        stateLock.withLock { _recorderSnapshot = recorder }
    }

    // Background-thread read (nonisolated context).
    nonisolated func handleVideoFrame(pixelBuffer: CVPixelBuffer, pts: CMTime) {
        stateLock.lock()
        let active = _recorderSnapshot
        stateLock.unlock()
        active?.appendVideoFrame(texture: ..., presentationTime: pts)
    }
}
```

`nonisolated(unsafe)` tells the Swift 6 strict-concurrency checker "I am taking responsibility for the synchronisation invariant here." The lock makes it actually safe — read and write are both serialised through `stateLock`. The cost is a single uncontended `os_unfair_lock` acquire on each frame, which Apple's locks resolve in nanoseconds.

### Three-Level Self-Regulating Frame-Drop Policy

A real-time capture-to-display-to-record pipeline must drop frames when GPU work falls behind, otherwise memory grows unbounded and audio/video drift opens up. MetalForge does this at three independent points:

| Level | Mechanism | What it protects |
|-------|-----------|------------------|
| **1. Capture** | `AVCaptureVideoDataOutput.alwaysDiscardsLateVideoFrames = true` | If the delegate (running on `captureQueue`) hasn't returned by the next frame's arrival, AVFoundation drops the new frame internally before it even reaches us. |
| **2. Pipeline** | `commandBuffer.waitUntilCompleted()` at the end of `pipeline.process` | Synchronously stalls the captureQueue until the GPU finishes. If the GPU is slow, subsequent capture frames pile up briefly and then get dropped by level 1 — the system self-regulates. No explicit logic required. |
| **3. Recorder** | `videoInput.isReadyForMoreMediaData` check inside `appendVideoFrame` | Checked twice: once on the caller thread (fast-path early drop, before any GPU encode), and again on the recorder's videoQueue right before the append. Drops the frame if the AVAssetWriter's internal encoder queue is saturated. Audio is *never* dropped here — audible clicks are worse than visual stutter. |

Together, these three guards ensure the system degrades gracefully under load: frames silently drop, but timing stays accurate (because PTS values never get rewritten) and memory never balloons.

---

## Quick Start

### Engine + Pipeline

```swift
import MetalForge

let engine   = try MetalForgeEngine()
let pipeline = try MetalForgePipeline(engine: engine)

let glitch = try GlitchFilter(engine: engine)
glitch.intensity = 0.4
pipeline.append(glitch)

let colorCorrection = try ColorCorrectionFilter(engine: engine)
colorCorrection.exposure   = +0.5    // half a stop brighter
colorCorrection.contrast   = 1.2     // 20 % more contrast around 0.18 grey
colorCorrection.saturation = 1.15
pipeline.append(colorCorrection)
```

Pipeline auto-detects the input pixel format and inserts the YUV converter, HDR decode, and HDR encode stages as needed. Your filters always see RGB in the working colour space (linear scene-referred for HDR, sRGB-ish for SDR).

### Integration in an `AVCaptureVideoDataOutputSampleBufferDelegate`

```swift
import AVFoundation
import MetalForge

final class CaptureController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    let engine:   MetalForgeEngine
    let pipeline: MetalForgePipeline
    let view:     MetalForgeView

    init() throws {
        self.engine   = try MetalForgeEngine()
        self.pipeline = try MetalForgePipeline(engine: engine)
        self.view     = try MetalForgeView(engine: engine)

        // Single sink → single recycler. Pipeline owns the pool.
        view.recycleHandler = { [weak pipeline] texture in
            pipeline?.recycle(texture)
        }
    }

    func captureOutput(
        _ output:        AVCaptureOutput,
        didOutput sample: CMSampleBuffer,
        from connection:  AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { return }
        guard let processed   = pipeline.process(pixelBuffer: pixelBuffer) else { return }

        // MetalForgeView is @MainActor — hop to main for the present call.
        DispatchQueue.main.async { [weak view] in
            view?.present(texture: processed)
        }
    }
}
```

For HDR capture, set `AVCaptureDeviceFormat.isVideoHDRSupported`-capable format on your device and request `kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange` as the output format. `MetalForgePipeline` will route it through the HDR decode/encode chain automatically.

### SwiftUI Layout

```swift
import SwiftUI
import MetalForge

struct CameraView: View {

    @StateObject private var controller = MetalForgeDemoController()

    var body: some View {
        ZStack {
            MetalForgeViewRepresentable(view: controller.view)
                .ignoresSafeArea()

            VStack {
                Spacer()
                VStack(spacing: 16) {
                    Picker("Filter", selection: $controller.activeFilter) {
                        ForEach(MetalForgeDemoController.FilterChoice.allCases) {
                            Text($0.rawValue).tag($0)
                        }
                    }
                    .pickerStyle(.segmented)

                    Slider(value: $controller.filterIntensity, in: 0...1)
                        .tint(.white)
                }
                .padding()
                .background(.ultraThinMaterial)
            }
        }
        .task { await controller.setup() }
    }
}
```

Add `NSCameraUsageDescription` and `NSMicrophoneUsageDescription` to your app's `Info.plist`. The controller wires up permission requests, capture configuration, the recorder, and the texture-refcount dispatch — see `MetalForgeDemoView.swift` for the full reference implementation.

### Recording with PTS-Preserving Audio Passthrough

```swift
let recorder = try MetalForgeRecorder(
    engine:            engine,
    videoSize:         CGSize(width: 3840, height: 2160),
    workingColorSpace: .hdr10PQ,
    frameRate:         60
)

recorder.recycleHandler = { [weak pipeline] tex in pipeline?.recycle(tex) }

try recorder.startRecording(outputURL: outputURL)

// On every video frame from the capture delegate:
recorder.appendVideoFrame(texture: processed, presentationTime: pts)

// On every audio sample (passthrough — no decoding / re-encoding):
recorder.appendAudioSample(sampleBuffer)

// When done:
try await recorder.stopRecording()
```

The recorder writes 10-bit YUV bi-planar HEVC for HDR (BT.2020 + PQ/HLG metadata correctly attached) and H.264 BGRA for SDR. Audio is muxed without re-encoding, preserving the original PTS from `AVCaptureSession`.

---

## Example App

**MetalForgeCamera** is a minimal SwiftUI demo app located in
[`Examples/MetalForgeCamera`](Examples/MetalForgeCamera/README.md).

It demonstrates:

- live camera preview backed by `AVCaptureSession`
- `CVPixelBuffer` → `MetalForgePipeline` integration through the zero-copy
  `MetalForgeCaptureManager`
- real-time GPU filter switching (Original / Warm / Cool / High Contrast)
- SwiftUI controls for filter intensity and before/after comparison
- live FPS readout

To run it:

```sh
open Examples/MetalForgeCamera/MetalForgeCamera.xcodeproj
```

The project references this package via a relative `../..` local package
reference — no manual wiring required. Pick a physical iOS device (the
simulator has no camera) and build. See the example's own
[README](Examples/MetalForgeCamera/README.md) for details on architecture,
permissions, and current limitations.

---

## Module Layout

| Domain | Files |
|--------|-------|
| **Core engine** | `MetalForgeEngine.swift`, `TexturePool.swift`, `MetalForgePipeline.swift`, `MetalForgeError.swift`, `MetalForgeColorSpace.swift` |
| **Filter protocols** | `MetalForgeFilter.swift` (incl. `MetalForgeSourceFilter`) |
| **YUV ↔ RGB** | `YUVToRGBConverter.swift`, `RGBToYUVConverter.swift`, `YUVColorMatrices.swift`, `Shaders/YUVConverterShader.metal`, `Shaders/RGBToYUVShader.metal` |
| **HDR transfer** | `HDRDecodeFilter.swift`, `HDREncodeFilter.swift`, `Shaders/HDRTransferShader.metal` |
| **Display** | `MetalForgeView.swift`, `MetalForgeViewRepresentable.swift`, `Shaders/DisplayRenderShader.metal` |
| **Recording** | `MetalForgeRecorder.swift` |
| **Capture** | `MetalForgeCaptureManager.swift` |
| **Effects — analog pack** | `AnalogDistortionFilters.swift`, `Shaders/AnalogKernels.metal` |
| **Effects — temporal pack** | `TemporalEffectsFilters.swift`, `Shaders/TemporalKernels.metal` |
| **Effects — color grading** | `ColorGradingFilters.swift`, `Shaders/ColorGradingKernels.metal` |
| **Built-in adjustment / glitch** | `AdjustmentFilter.swift`, `GlitchFilter.swift`, `Shaders/AdjustmentShader.metal`, `Shaders/GlitchShader.metal` |
| **Demo** | `MetalForgeDemoView.swift` (SwiftUI controller + view) |

---

## Platform Requirements

- **iOS 17+** / **macOS 14+** / **tvOS 16+** / **visionOS 1+**
- **Swift 6.0+** (strict concurrency)
- Metal-capable device (every Apple Silicon Mac and every iOS device since A7)
- For HDR capture: iPhone 12+ or iPad Pro M-series; macOS HDR sources via `AVAssetReader`

---

## License

MetalForge is released under the **MIT License**.

```
Copyright (c) 2026 MetalForge contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

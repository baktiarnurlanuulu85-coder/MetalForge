# MetalForge

**Real-time GPU video processing for Apple platforms, built on Metal and AVFoundation.**

![Swift](https://img.shields.io/badge/Swift-6.0-orange.svg)
![Platforms](https://img.shields.io/badge/platform-iOS%2017%2B%20%7C%20macOS%2014%2B-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![CI](https://github.com/letbakty/MetalForge/actions/workflows/ci.yml/badge.svg)

---

## Overview

MetalForge is a Swift Package for processing live camera frames and video streams
on the GPU. It wraps `CVPixelBuffer`s as Metal textures, runs them through a chain
of compute-shader filters in either an SDR or HDR-oriented working space, and feeds
the result to an on-screen view and/or an `AVAssetWriter`-backed recorder.

Reach for MetalForge when you need a Metal compute filter chain wired into Apple's
capture and recording stack without writing the `CVPixelBuffer` ↔ `MTLTexture`
plumbing yourself.

> **Status:** early `0.1.x` development. APIs may change before `1.0`.

---

## Features

- **Real-time GPU video processing** with Metal compute shaders
- **`CVPixelBuffer` → `MTLTexture` pipeline** backed by `CVMetalTextureCache`
- **Composable filter chain** — append filters and process frame by frame
- **3D LUT color grading** with hardware trilinear interpolation and built-in presets
- **SDR / HDR-oriented pipeline architecture** (BT.709 and BT.2100 PQ / HLG paths)
- **SwiftUI / UIKit / AppKit preview** via `MetalForgeView` and representable wrappers
- **AVFoundation capture & recording building blocks** (`MetalForgeCaptureManager`, `MetalForgeRecorder`)
- **Swift Package Manager** support
- **Example camera app** under `Examples/MetalForgeCamera`
- **GitHub Actions CI** running `swift build` and `swift test`

---

## Demo

> **Demo video coming soon.**

The bundled [`MetalForgeCamera`](Examples/MetalForgeCamera) example app shows the
library running end to end on a device:

- live camera preview
- real-time filter switching
- intensity control
- before/after comparison

---

## Use Cases

MetalForge is a building block rather than a finished app. It fits well when you need:

- **Live camera filters** — apply GPU effects to a capture feed in real time
- **Video editor preview pipeline** — drive an on-screen preview from a filter chain
- **LUT-based color grading** — load 3D LUTs for consistent looks across footage
- **HDR-aware video effects** — process BT.2100 PQ / HLG content in linear-ish space
- **Custom camera / recorder prototypes** — wire capture, processing, and recording together
- **GPU processing experiments** — a small, readable base for trying out Metal compute kernels

---

## Requirements

- iOS 17+
- macOS 14+
- Swift 6 / Xcode 16+
- A Metal-capable Apple device
- Swift Package Manager

---

## Installation

Add MetalForge as a Swift Package dependency. There is no tagged release yet, so
track the `develop` branch for now (a tagged `v0.1.0` is coming soon):

```swift
.package(
    url: "https://github.com/letbakty/MetalForge.git",
    branch: "develop"
)
```

Then add `"MetalForge"` to your target's dependencies.

For production use, prefer a tagged release once `v0.1.0` is published.

---

## Quick Start

```swift
import MetalForge

// 1. Create the engine and pipeline.
let engine   = try MetalForgeEngine()
let pipeline = try MetalForgePipeline(engine: engine)

// 2. Append one or more filters.
let colorCorrection = try ColorCorrectionFilter(engine: engine)
colorCorrection.exposure   = 0.5    // half a stop brighter
colorCorrection.contrast   = 1.2
colorCorrection.saturation = 1.15
pipeline.append(colorCorrection)

// 3. Wire the view's recycle handler back to the pool so processed textures
//    are reclaimed instead of leaked. (Also set recorder.recycleHandler when
//    recording.)
view.recycleHandler = { [weak pipeline] texture in
    pipeline?.recycle(texture)
}

// 4. Process a CVPixelBuffer (e.g. from a capture delegate).
//    The pipeline auto-detects the pixel format and inserts the YUV→RGB,
//    HDR-decode, and HDR-encode stages as needed.
guard let processed = pipeline.process(pixelBuffer: pixelBuffer) else { return }

// 5. Present the result on screen (MetalForgeView is @MainActor).
DispatchQueue.main.async {
    view.present(texture: processed)
}
```

See
[`MetalForgeDemoView.swift`](Sources/MetalForge/MetalForgeDemoView.swift) for a full
capture → display → record reference implementation.

---

## Example App

[`Examples/MetalForgeCamera`](Examples/MetalForgeCamera) is a minimal SwiftUI iOS app
that runs a live camera feed through the filter chain. It demonstrates:

- live camera preview backed by `AVCaptureSession`
- real-time filter switching (Original / Warm / Cool / High Contrast)
- a filter intensity slider
- a before/after ("Show Original") toggle
- SwiftUI integration via `MetalForgeViewRepresentable`
- a local SwiftPM dependency on this package (relative `../..` path — no manual wiring)

Open it with:

```bash
open Examples/MetalForgeCamera/MetalForgeCamera.xcodeproj
```

Run on a real iPhone — the iOS simulator has no camera. See the example's own
[README](Examples/MetalForgeCamera/README.md) for permissions and details.

---

## Supported Filters & Stages

All filters conform to `MetalForgeFilter` and are backed by a Metal compute kernel.
Conversion and HDR stages are inserted automatically by the pipeline based on the
input pixel format.

**Color correction**
- `ColorCorrectionFilter` — `exposure`, `contrast`, `saturation`, `temperatureShift`
- `AdjustmentFilter` — `brightness`, `contrast`

**3D LUT grading**
- `MetalForgeLUTFilter` — hardware trilinear interpolation with built-in presets
  (`.identity`, `.warm`, `.cool`, `.sepia`) and an adjustable `intensity`

**Glitch effects**
- `GlitchFilter` — `intensity`

**Analog effects**
- `ChromaticAberrationFilter` — `redShift`, `greenShift`
- `AnalogNoiseFilter` — `noiseIntensity`, `timeSeed`
- `HorizontalJitterFilter` — `jitterIntensity`, `timeSeed`

**Temporal effects**
- `MotionBlurFilter` — `accumulationAlpha`
- `NeonTrailsFilter` — `intensity`, `decay`, `neonColor`

**YUV ↔ RGB conversion** *(automatic stages)*
- `YUVToRGBConverter` — wraps bi-planar camera YUV into RGB working space
- `RGBToYUVConverter` — converts processed RGB back to YUV for recording

**HDR decode / encode** *(automatic, HDR sources only)*
- `HDRDecodeFilter` — PQ / HLG → linear scene light
- `HDREncodeFilter` — linear → PQ / HLG for display or file output

---

## Architecture

```
CVPixelBuffer ─► MetalForgeEngine.makeTextures(from:) ─► MetalForgePipeline.process()
                                                              │
                                                              ▼
                              YUV→RGB ─► [HDR decode] ─► user filters ─► [HDR encode]
                                                              │
                                              ┌───────────────┴───────────────┐
                                              ▼                               ▼
                                       MetalForgeView                 MetalForgeRecorder
                                       (MTKView preview)              (AVAssetWriter)
```

- **`MetalForgeEngine`** — owns the `MTLDevice`, command queue, texture caches, and
  centralized shader-library loading.
- **`MetalForgePipeline`** — holds the ordered filter chain and drives per-frame
  processing, inserting color-conversion and HDR transfer stages automatically.
- **Filters** — small types conforming to `MetalForgeFilter` (e.g. `GlitchFilter`,
  `ColorCorrectionFilter`, `MetalForgeLUTFilter`, the analog and temporal packs), each
  backed by a Metal compute kernel.
- **`TexturePool`** — recycles intermediate `MTLTexture`s to avoid per-frame allocation.
- **Shader library loading** — the engine loads a precompiled `default.metallib` when
  present (Xcode builds) and otherwise compiles the bundled `.metal` sources at runtime,
  so the package works under both SwiftPM and Xcode.
- **Capture & recording** — `MetalForgeCaptureManager` adapts `AVCaptureSession` output.
  `MetalForgeRecorder` writes processed frames via `AVAssetWriter` and supports optional
  passthrough audio when available.

---

## Design Notes

A few choices worth calling out for anyone reading the source:

- **`CVPixelBuffer` → `MTLTexture` wrapping** — camera buffers are wrapped as Metal
  textures rather than copied, keeping the per-frame path off the CPU.
- **`CVMetalTextureCache`** — backs that wrapping so the GPU reads the same IOSurface
  the camera produced; the recorder uses a second cache for its output pool.
- **`TexturePool` reuse** — intermediate textures are pooled and reused across frames
  instead of being allocated and freed every frame.
- **Explicit texture recycling** — processed textures return to the pool through
  `recycleHandler` callbacks, so a texture is only reused once its consumers are done
  with it (important when preview and recorder share the same frame).
- **SwiftPM / Xcode shader library loading** — the engine loads a precompiled
  `default.metallib` when present (Xcode) and otherwise compiles the bundled `.metal`
  sources at runtime, so the same package works under both toolchains and in CI.
- **Separation of concerns** — capture (`MetalForgeCaptureManager`), processing
  (`MetalForgePipeline`), preview (`MetalForgeView`), and recording
  (`MetalForgeRecorder`) are independent pieces you can use together or on their own.

---

## Testing

```bash
swift build
swift test
```

GitHub Actions runs both on every push and pull request targeting `main` and `develop`
via [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Benchmarks

Benchmarks are planned for `v0.2.0`. The benchmark runner will measure:

- average GPU frame time
- end-to-end frame latency
- 1080p / 4K pipeline cost
- LUT + color correction cost
- memory behavior under sustained preview

---

## Current Status

MetalForge is currently in early `0.1.x` development. The core engine, filter chain,
preview components, and recording path are covered by build and unit tests, but APIs
may change before the `1.0` release.

---

## Limitations

Honest about where the project is today:

- The API is early `0.1.x` and may change before `1.0`.
- Benchmarks are not published yet — performance numbers will come with the `v0.2.0`
  benchmark runner.
- The example app focuses on live preview and filtering.
- A physical iPhone is required for camera preview (the iOS simulator has no camera).
- A dedicated recording-example UI is planned but not built yet; recording is available
  as an API (`MetalForgeRecorder`) rather than a finished screen in the example app.

---

## Roadmap

- `v0.1.0` initial public release (tagged)
- More pixel-accuracy tests
- A benchmark runner with real measurements
- Example app screenshots / videos
- A dedicated recording example
- More built-in LUT presets
- Expanded documentation

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — high-level architecture and component graph
- [`docs/PIPELINE.md`](docs/PIPELINE.md) — per-frame processing flow
- [`docs/FILE_MAP.md`](docs/FILE_MAP.md) — file-by-file project map

---

## Contributing

Issues and pull requests are welcome. Please open an issue to discuss substantial
changes first, keep PRs focused, and make sure `swift build` and `swift test` pass
before submitting.

---

## License

MetalForge is released under the MIT License. See [LICENSE](LICENSE) for the full text.

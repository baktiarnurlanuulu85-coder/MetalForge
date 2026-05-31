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

## ✨ Features

- ⚡ **Real-time GPU video processing** with Metal compute shaders
- 🎥 **`CVPixelBuffer` → `MTLTexture` pipeline** backed by `CVMetalTextureCache`
- 🧩 **Composable filter chain** — append filters and process frame by frame
- 🎨 **3D LUT color grading** with hardware trilinear interpolation and built-in presets
- 🌈 **SDR / HDR-oriented pipeline architecture** (BT.709 and BT.2100 PQ / HLG paths)
- 🖥️ **SwiftUI / UIKit / AppKit preview** via `MetalForgeView` and representable wrappers
- 📼 **AVFoundation capture & recording building blocks** (`MetalForgeCaptureManager`, `MetalForgeRecorder`)
- 🧱 **Swift Package Manager** support
- 📱 **Example camera app** under `Examples/MetalForgeCamera`
- 🧪 **GitHub Actions CI** running `swift build` and `swift test`

---

## Requirements

- iOS 17+
- macOS 14+
- Swift 6 / Xcode 16+
- A Metal-capable Apple device
- Swift Package Manager

---

## 🧱 Installation

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

## 🚀 Quick Start

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

## 📱 Example App

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

## 🏗️ Architecture

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

## 🧪 Testing

```bash
swift build
swift test
```

GitHub Actions runs both on every push and pull request targeting `main` and `develop`
via [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## 📊 Benchmarks

Benchmarks are planned for `v0.2.0`. The benchmark runner will measure:

- average GPU frame time
- end-to-end frame latency
- 1080p / 4K pipeline cost
- LUT + color correction cost
- memory behavior under sustained preview

---

## 📌 Current Status

MetalForge is currently in early `0.1.x` development. The core engine, filter chain,
preview components, and recording path are covered by build and unit tests, but APIs
may change before the `1.0` release.

---

## 🗺️ Roadmap

- `v0.1.0` initial public release (tagged)
- More pixel-accuracy tests
- A benchmark runner with real measurements
- Example app screenshots / videos
- A dedicated recording example
- More built-in LUT presets
- Expanded documentation

---

## 🤝 Contributing

Issues and pull requests are welcome. Please open an issue to discuss substantial
changes first, keep PRs focused, and make sure `swift build` and `swift test` pass
before submitting.

---

## 📄 License

MetalForge is released under the MIT License. See [LICENSE](LICENSE) for the full text.

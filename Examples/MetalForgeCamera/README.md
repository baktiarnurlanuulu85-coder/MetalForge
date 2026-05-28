# MetalForgeCamera

A minimal SwiftUI iOS app that demonstrates the [MetalForge](../../README.md)
library running a live camera feed through a GPU filter chain.

## What this app demonstrates

- Live `AVCaptureSession` preview wired into `MetalForgePipeline`
- Zero-copy `CVPixelBuffer` → `MTLTexture` ingestion via the library's built-in
  `MetalForgeCaptureManager`
- Real-time filter switching using `MetalForgeLUTFilter` (warm / cool presets)
  and `ColorCorrectionFilter` (high-contrast preset)
- SwiftUI controls bound to the pipeline (filter picker, intensity slider,
  before/after toggle)
- Clean Apple-style UI with a translucent bottom control panel and a floating
  FPS indicator

## How to open and run

1. Open `Examples/MetalForgeCamera/MetalForgeCamera.xcodeproj` in Xcode 16 or
   later.
2. The project references the parent MetalForge package via the relative path
   `../..` — no extra setup is required. If Xcode does not pick it up
   automatically, select the project, go to **Package Dependencies**, and
   verify that the local package at `../..` is listed.
3. Pick an iOS device (the iOS simulator has no camera) and run.
4. Grant camera permission when prompted.

> **iOS deployment target:** 17.0
>
> **Xcode:** 16.0 or later (Swift 6 strict-concurrency toolchain matches the
> MetalForge package manifest).

## Features

| UI element              | What it does                                                       |
|-------------------------|--------------------------------------------------------------------|
| Filter picker           | Segmented control switching between Original / Warm / Cool / High Contrast |
| Intensity slider        | Drives the active filter's strength in `[0, 1]`                    |
| "Show Original" toggle  | Bypasses all filters (before/after comparison)                     |
| FPS pill                | Rolling 1-second average of pipeline output rate                   |

The architecture follows the "permanent filter chain" pattern from the
library's `MetalForgeDemoController`: all filters live in the pipeline at all
times, and the active selection is enforced by zeroing the inactive filters'
intensity parameters. This avoids any race between UI taps on the main thread
and frame processing on the capture queue.

## Current limitations

- SDR only. The app calls `capture.configure(position: .back, preferHDR: false)`.
  HDR (PQ / HLG) is supported by the library and can be enabled by flipping
  the `preferHDR` flag — the rest of the pipeline configures itself
  automatically.
- No recording. The Info.plist declares `NSCameraUsageDescription` plus
  `NSMicrophoneUsageDescription` — the latter is required because
  `MetalForgeCaptureManager.configure()` unconditionally attaches a microphone
  input to the `AVCaptureSession`, even when audio is not consumed. Add
  `NSPhotoLibraryAddUsageDescription` if you wire up `MetalForgeRecorder`
  and want to save clips to the Photos library.
- Single rear camera. The capture manager exposes `position:`; flip it to
  `.front` to use the selfie camera.
- The intensity slider is inert for **Original** — there is nothing to
  modulate. The slider is still rendered (per spec) but disabled in that mode.

## File layout

```
MetalForgeCamera/
  MetalForgeCameraApp.swift   – @main SwiftUI app
  ContentView.swift           – root layout (preview + HUD + controls)
  CameraPreviewView.swift     – thin MetalForgeViewRepresentable wrapper
  CameraViewModel.swift       – owns engine + pipeline + capture; UI state
  FilterControlPanel.swift    – bottom translucent control panel
  Assets.xcassets/            – AppIcon + AccentColor placeholders
  Info.plist                  – NSCameraUsageDescription declared here
```

## See also

- [MetalForge root README](../../README.md) — architecture, HDR pipeline, the
  `MetalForgeRecorder` API, and the full demo controller pattern.

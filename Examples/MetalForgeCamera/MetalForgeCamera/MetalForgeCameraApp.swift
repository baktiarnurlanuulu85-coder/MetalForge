import SwiftUI

/// Entry point for the MetalForgeCamera demo.
///
/// This example app demonstrates how to wire a live `AVCaptureSession`
/// pipeline into the MetalForge library and render the result with
/// `MetalForgeView`, all from a small SwiftUI surface.
@main
struct MetalForgeCameraApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
                .statusBarHidden()
        }
    }
}

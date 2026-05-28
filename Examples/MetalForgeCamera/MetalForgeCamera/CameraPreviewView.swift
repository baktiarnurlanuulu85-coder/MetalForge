import SwiftUI
import MetalForge

/// Thin SwiftUI wrapper around the shared `MetalForgeView`.
///
/// The underlying view is owned by `CameraViewModel` so SwiftUI re-renders
/// never re-create the Metal resources. `MetalForgeViewRepresentable` is the
/// official bridge shipped with the library.
struct CameraPreviewView: View {
    let view: MetalForgeView

    var body: some View {
        MetalForgeViewRepresentable(view: view)
            .ignoresSafeArea()
    }
}

import SwiftUI

/// Root layout: full-screen camera preview with a floating control panel.
struct ContentView: View {

    @StateObject private var viewModel = CameraViewModel()

    var body: some View {
        ZStack {
            // ----- Live preview / permission fallback -----
            if viewModel.permissionsGranted {
                CameraPreviewView(view: viewModel.view)
            } else {
                permissionsPlaceholder
            }

            // ----- HUD: FPS pill at the top -----
            VStack {
                HStack {
                    Spacer()
                    fpsPill
                        .padding(.trailing, 16)
                        .padding(.top, 12)
                }
                Spacer()
            }

            // ----- Bottom controls -----
            VStack {
                Spacer()
                FilterControlPanel(viewModel: viewModel)
            }

            // ----- Error toast -----
            if let err = viewModel.setupError {
                errorToast(err)
            }
        }
        .task { await viewModel.setup() }
    }

    // MARK: - Subviews

    private var permissionsPlaceholder: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 12) {
                Image(systemName: "video.slash")
                    .font(.system(size: 56))
                    .foregroundStyle(.white.opacity(0.7))
                Text("Camera access required")
                    .font(.headline)
                    .foregroundStyle(.white)
                Text("Enable camera access in Settings to use the demo.")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.6))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
            }
        }
    }

    /// Compact pill displaying the current FPS.
    private var fpsPill: some View {
        HStack(spacing: 6) {
            Image(systemName: "speedometer")
                .imageScale(.small)
            Text(String(format: "%.0f FPS", viewModel.fps))
                .font(.caption.monospacedDigit())
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial, in: Capsule())
        .opacity(viewModel.fps > 0 ? 1 : 0)
        .animation(.easeInOut(duration: 0.2), value: viewModel.fps > 0)
    }

    private func errorToast(_ message: String) -> some View {
        VStack {
            Spacer()
            Text(message)
                .font(.caption)
                .foregroundStyle(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.red.opacity(0.85), in: RoundedRectangle(cornerRadius: 10))
                .padding(.bottom, 220)
                .padding(.horizontal, 24)
        }
    }
}

#Preview {
    ContentView()
        .preferredColorScheme(.dark)
}

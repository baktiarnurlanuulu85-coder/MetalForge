import SwiftUI

/// Bottom control panel: filter picker + intensity slider + before/after toggle.
///
/// Pure-UI; all state is bound to `CameraViewModel`.
struct FilterControlPanel: View {

    @ObservedObject var viewModel: CameraViewModel

    var body: some View {
        VStack(spacing: 16) {
            // ----- Filter picker -----
            Picker("Filter", selection: $viewModel.activeFilter) {
                ForEach(FilterChoice.allCases) { choice in
                    Label(choice.rawValue, systemImage: choice.iconName)
                        .tag(choice)
                }
            }
            .pickerStyle(.segmented)
            .disabled(viewModel.showOriginal)

            // ----- Intensity slider -----
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Image(systemName: "dial.medium")
                        .imageScale(.small)
                    Text("Intensity")
                        .font(.caption)
                    Spacer()
                    Text("\(Int(viewModel.filterIntensity * 100))%")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.white.opacity(0.7))
                }
                .foregroundStyle(.white)

                Slider(value: $viewModel.filterIntensity, in: 0...1)
                    .tint(.white)
                    // The "Original" choice has nothing to dial, so the slider
                    // is visually present (per spec) but inert.
                    .disabled(
                        viewModel.showOriginal ||
                        !viewModel.activeFilter.supportsIntensity
                    )
            }

            // ----- Before / After toggle -----
            HStack {
                Image(systemName: "rectangle.righthalf.inset.filled.arrow.right")
                Toggle("Show Original (Before)", isOn: $viewModel.showOriginal)
                    .font(.subheadline)
            }
            .foregroundStyle(.white)
            .tint(.white)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 18)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
        .padding(.horizontal, 12)
        .padding(.bottom, 12)
    }
}

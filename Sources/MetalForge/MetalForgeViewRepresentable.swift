#if canImport(SwiftUI)
import SwiftUI

/// SwiftUI wrapper for `MetalForgeView`. Cross-platform: `UIViewRepresentable`
/// on UIKit-based platforms (iOS / tvOS / visionOS) and `NSViewRepresentable`
/// on AppKit (macOS).
///
/// The wrapped `MetalForgeView` is owned externally — typically by an
/// `ObservableObject` controller that also owns the engine, pipeline, and
/// recorder. SwiftUI never recreates the underlying Metal resources; it just
/// hands the existing view to the system view hierarchy.
///
/// Lifetime: the view stays alive as long as the owning controller does.
/// Multiple representable instances created across SwiftUI re-renders all
/// return the same wrapped view, so there are no flashes / re-allocations.

#if canImport(UIKit)
import UIKit

public struct MetalForgeViewRepresentable: UIViewRepresentable {
    public let view: MetalForgeView

    public init(view: MetalForgeView) {
        self.view = view
    }

    public func makeUIView(context: Context) -> MetalForgeView {
        view
    }

    public func updateUIView(_ uiView: MetalForgeView, context: Context) {
        // No-op. View configuration (workingColorSpace, scalingMode, etc.)
        // is driven by the owning controller via direct property mutation.
    }
}

#elseif canImport(AppKit)
import AppKit

public struct MetalForgeViewRepresentable: NSViewRepresentable {
    public let view: MetalForgeView

    public init(view: MetalForgeView) {
        self.view = view
    }

    public func makeNSView(context: Context) -> MetalForgeView {
        view
    }

    public func updateNSView(_ nsView: MetalForgeView, context: Context) {
        // Same rationale as iOS — externally driven configuration.
    }
}

#endif
#endif

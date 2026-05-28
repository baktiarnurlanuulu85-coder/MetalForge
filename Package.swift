// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MetalForge",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(name: "MetalForge", targets: ["MetalForge"]),
    ],
    targets: [
        .target(
            name: "MetalForge",
            path: "Sources/MetalForge",
            resources: [
                // .process compiles *.metal files into a .metallib and bundles it
                // alongside the module; accessible at runtime via Bundle.module.
                .process("Shaders")
            ]
        ),
        .testTarget(
            name: "MetalForgeTests",
            dependencies: ["MetalForge"],
            path: "Tests/MetalForgeTests"
        )
    ]
)

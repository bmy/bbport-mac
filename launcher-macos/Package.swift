// swift-tools-version:5.9
// bbport launcher for macOS: a SwiftUI front end for tools/macos/run.sh.
// Built and bundled into out/bbport.app by tools/macos/build_launcher.sh.
import PackageDescription

let package = Package(
    name: "BBLauncher",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "BBLauncher",
            path: "Sources/BBLauncher"
        ),
    ]
)

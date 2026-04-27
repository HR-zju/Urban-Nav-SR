// swift-tools-version: 5.10
import PackageDescription
import Foundation

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let infoPlistPath = "\(packageRoot)/Sources/NavAssistantApp/Info.plist"

let package = Package(
    name: "NavAssistantApp",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "NavAssistantApp",
            targets: ["NavAssistantApp"]
        )
    ],
    targets: [
        .executableTarget(
            name: "NavAssistantApp",
            exclude: ["Info.plist"],
            resources: [.process("Assets.xcassets")],
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("Speech"),
                .linkedFramework("MapKit"),
                .linkedFramework("CoreLocation"),
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", infoPlistPath
                ])
            ]
        )
    ]
)

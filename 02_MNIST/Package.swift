// swift-tools-version:4.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "02_MNIST",
    dependencies: [
        .package(url: "https://github.com/Octadero/MNISTKit.git", from: "0.0.7"),
        .package(url: "https://github.com/Octadero/TensorFlow.git", from: "0.0.5")
    ],
    targets: [
        .target(
            name: "02_MNIST",
            dependencies: [ "CAPI", "Proto", "TensorFlowKit", "MNISTKit"])
        ]
)

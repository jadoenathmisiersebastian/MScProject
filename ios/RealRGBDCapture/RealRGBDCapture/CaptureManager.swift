import ARKit
import Combine
import CoreImage
import UIKit

private struct CaptureMetadata: Codable {
    let schema_version: Int
    let frame_id: String
    let timestamp: TimeInterval

    let rgb_filename: String
    let rgb_size: [Int]

    let raw_depth_filename: String
    let smoothed_depth_filename: String?
    let depth_size: [Int]
    let depth_storage: String
    let depth_measurement_strategy: String

    let confidence_filename: String?
    let confidence_storage: String

    let camera_intrinsics_row_major: [[Float]]
    let intrinsics_resolution: [Int]
    let camera_transform_row_major: [[Float]]

    let captured_image_orientation: String
    let configured_interface_orientation: String
}

private enum CaptureError: LocalizedError {
    case noRGBImage
    case unsupportedDepthFormat(OSType)
    case inaccessiblePixelBuffer
    case couldNotEncodePNG

    var errorDescription: String? {
        switch self {
        case .noRGBImage:
            return "Could not convert the RGB camera buffer."
        case .unsupportedDepthFormat(let format):
            return "Unsupported depth pixel format: \(format)."
        case .inaccessiblePixelBuffer:
            return "Could not access a pixel buffer."
        case .couldNotEncodePNG:
            return "Could not encode the RGB image as PNG."
        }
    }
}

@MainActor
final class CaptureManager: ObservableObject {
    @Published private(set) var statusMessage = "Starting ARKit..."
    @Published private(set) var isDepthReady = false
    @Published private(set) var isSaving = false
    @Published private(set) var capturedFrameCount = 0

    weak var session: ARSession?

    private let fileManager = FileManager.default
    private let ciContext = CIContext()

    func attach(session: ARSession) {
        self.session = session
        statusMessage = "ARKit session active"
    }

    func inspectCurrentFrame() {
        guard let frame = session?.currentFrame else {
            statusMessage = "No ARKit frame available"
            isDepthReady = false
            return
        }

        guard let depthData = frame.sceneDepth else {
            statusMessage = "RGB available, but depth is unavailable"
            isDepthReady = false
            return
        }

        let rgbWidth = CVPixelBufferGetWidth(frame.capturedImage)
        let rgbHeight = CVPixelBufferGetHeight(frame.capturedImage)
        let depthWidth = CVPixelBufferGetWidth(depthData.depthMap)
        let depthHeight = CVPixelBufferGetHeight(depthData.depthMap)

        statusMessage =
            "Synchronized RGB \(rgbWidth)x\(rgbHeight), " +
            "depth \(depthWidth)x\(depthHeight)"

        isDepthReady = true
    }

    func captureCurrentFrame() {
        guard !isSaving else {
            return
        }

        guard let frame = session?.currentFrame else {
            statusMessage = "Capture failed: no ARKit frame"
            return
        }

        guard case .normal = frame.camera.trackingState else {
            statusMessage = "Capture rejected: tracking is not normal"
            return
        }

        guard let rawDepth = frame.sceneDepth else {
            statusMessage = "Capture failed: scene depth unavailable"
            return
        }

        isSaving = true
        var incompleteDirectory: URL?

        defer {
            isSaving = false
        }

        do {
            let rootDirectory = try captureRootDirectory()
            let frameID = try nextFrameID(in: rootDirectory)
            let frameDirectory = rootDirectory.appendingPathComponent(
                frameID,
                isDirectory: true
            )

            incompleteDirectory = frameDirectory

            try fileManager.createDirectory(
                at: frameDirectory,
                withIntermediateDirectories: false
            )

            let rgbURL = frameDirectory.appendingPathComponent("rgb.png")
            try writeRGB(
                pixelBuffer: frame.capturedImage,
                to: rgbURL
            )

            let rawDepthURL = frameDirectory.appendingPathComponent(
                "depth_raw.bin"
            )
            try writeFloat32PixelBuffer(
                rawDepth.depthMap,
                to: rawDepthURL
            )

            var smoothedDepthFilename: String?

            if let smoothedDepth = frame.smoothedSceneDepth {
                let smoothedDepthURL = frameDirectory.appendingPathComponent(
                    "depth_smoothed.bin"
                )

                try writeFloat32PixelBuffer(
                    smoothedDepth.depthMap,
                    to: smoothedDepthURL
                )

                smoothedDepthFilename = "depth_smoothed.bin"
            }

            var confidenceFilename: String?

            if let confidenceMap = rawDepth.confidenceMap {
                let confidenceURL = frameDirectory.appendingPathComponent(
                    "confidence.bin"
                )

                try writeUInt8PixelBuffer(
                    confidenceMap,
                    to: confidenceURL
                )

                confidenceFilename = "confidence.bin"
            }

            let rgbWidth = CVPixelBufferGetWidth(frame.capturedImage)
            let rgbHeight = CVPixelBufferGetHeight(frame.capturedImage)
            let depthWidth = CVPixelBufferGetWidth(rawDepth.depthMap)
            let depthHeight = CVPixelBufferGetHeight(rawDepth.depthMap)

            let metadata = CaptureMetadata(
                schema_version: 1,
                frame_id: frameID,
                timestamp: frame.timestamp,
                rgb_filename: "rgb.png",
                rgb_size: [rgbWidth, rgbHeight],
                raw_depth_filename: "depth_raw.bin",
                smoothed_depth_filename: smoothedDepthFilename,
                depth_size: [depthWidth, depthHeight],
                depth_storage:
                    "Float32 little-endian row-major metres",
                depth_measurement_strategy: "camera_z",
                confidence_filename: confidenceFilename,
                confidence_storage:
                    "UInt8 row-major ARConfidenceLevel raw values",
                camera_intrinsics_row_major: rows(
                    of: frame.camera.intrinsics
                ),
                intrinsics_resolution: [
                    Int(frame.camera.imageResolution.width),
                    Int(frame.camera.imageResolution.height)
                ],
                camera_transform_row_major: rows(
                    of: frame.camera.transform
                ),
                captured_image_orientation: "arkit_native_sensor",
                configured_interface_orientation: "landscape_left"
            )

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

            let metadataData = try encoder.encode(metadata)
            let metadataURL = frameDirectory.appendingPathComponent(
                "metadata.json"
            )

            try metadataData.write(
                to: metadataURL,
                options: .atomic
            )

            incompleteDirectory = nil
            capturedFrameCount += 1
            isDepthReady = true
            statusMessage = "Saved \(frameID)"
        } catch {
            if let incompleteDirectory {
                try? fileManager.removeItem(at: incompleteDirectory)
            }

            statusMessage =
                "Capture failed: \(error.localizedDescription)"
        }
    }

    private func captureRootDirectory() throws -> URL {
        let documents = fileManager.urls(
            for: .documentDirectory,
            in: .userDomainMask
        )[0]

        let root = documents.appendingPathComponent(
            "real_rgbd_raw",
            isDirectory: true
        )

        try fileManager.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )

        return root
    }

    private func nextFrameID(in root: URL) throws -> String {
        let existing = try fileManager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )

        let maximumIndex = existing.compactMap { url -> Int? in
            let name = url.lastPathComponent

            guard name.hasPrefix("frame_") else {
                return nil
            }

            return Int(name.dropFirst("frame_".count))
        }
        .max() ?? 0

        return String(
            format: "frame_%06d",
            maximumIndex + 1
        )
    }

    private func writeRGB(
        pixelBuffer: CVPixelBuffer,
        to url: URL
    ) throws {
        let image = CIImage(cvPixelBuffer: pixelBuffer)

        guard let cgImage = ciContext.createCGImage(
            image,
            from: image.extent
        ) else {
            throw CaptureError.noRGBImage
        }

        guard let pngData = UIImage(cgImage: cgImage).pngData() else {
            throw CaptureError.couldNotEncodePNG
        }

        try pngData.write(to: url, options: .atomic)
    }

    private func writeFloat32PixelBuffer(
        _ pixelBuffer: CVPixelBuffer,
        to url: URL
    ) throws {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)

        guard format == kCVPixelFormatType_DepthFloat32 else {
            throw CaptureError.unsupportedDepthFormat(format)
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw CaptureError.inaccessiblePixelBuffer
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let outputBytesPerRow = width * MemoryLayout<Float32>.size

        var data = Data(capacity: outputBytesPerRow * height)

        for row in 0..<height {
            let rowAddress = baseAddress.advanced(
                by: row * bytesPerRow
            )

            data.append(
                rowAddress.assumingMemoryBound(to: UInt8.self),
                count: outputBytesPerRow
            )
        }

        try data.write(to: url, options: .atomic)
    }

    private func writeUInt8PixelBuffer(
        _ pixelBuffer: CVPixelBuffer,
        to url: URL
    ) throws {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw CaptureError.inaccessiblePixelBuffer
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        var data = Data(capacity: width * height)

        for row in 0..<height {
            let rowAddress = baseAddress.advanced(
                by: row * bytesPerRow
            )

            data.append(
                rowAddress.assumingMemoryBound(to: UInt8.self),
                count: width
            )
        }

        try data.write(to: url, options: .atomic)
    }

    private func rows(
        of matrix: simd_float3x3
    ) -> [[Float]] {
        [
            [
                matrix.columns.0.x,
                matrix.columns.1.x,
                matrix.columns.2.x
            ],
            [
                matrix.columns.0.y,
                matrix.columns.1.y,
                matrix.columns.2.y
            ],
            [
                matrix.columns.0.z,
                matrix.columns.1.z,
                matrix.columns.2.z
            ]
        ]
    }

    private func rows(
        of matrix: simd_float4x4
    ) -> [[Float]] {
        [
            [
                matrix.columns.0.x,
                matrix.columns.1.x,
                matrix.columns.2.x,
                matrix.columns.3.x
            ],
            [
                matrix.columns.0.y,
                matrix.columns.1.y,
                matrix.columns.2.y,
                matrix.columns.3.y
            ],
            [
                matrix.columns.0.z,
                matrix.columns.1.z,
                matrix.columns.2.z,
                matrix.columns.3.z
            ],
            [
                matrix.columns.0.w,
                matrix.columns.1.w,
                matrix.columns.2.w,
                matrix.columns.3.w
            ]
        ]
    }
}

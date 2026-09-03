import ARKit
import SwiftUI

struct ContentView: View {
    @StateObject private var captureManager = CaptureManager()

    private let supportsSceneDepth =
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)

    var body: some View {
        ZStack {
            ARCameraView(captureManager: captureManager)
                .ignoresSafeArea()

            VStack {
                HStack {
                    Text(
                        supportsSceneDepth
                            ? "LiDAR Scene Depth Enabled"
                            : "Scene Depth Unsupported"
                    )
                    .font(.headline)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(.black.opacity(0.7))
                    .clipShape(RoundedRectangle(cornerRadius: 6))

                    Spacer()
                }

                Spacer()

                VStack(spacing: 10) {
                    Text(captureManager.statusMessage)
                        .font(.subheadline)
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.center)

                    Button {
                        captureManager.inspectCurrentFrame()
                    } label: {
                        Label(
                            "Check RGB-D Frame",
                            systemImage:
                                "camera.metering.center.weighted"
                        )
                        .font(.headline)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!supportsSceneDepth)

                    Button {
                        captureManager.captureCurrentFrame()
                    } label: {
                        Label(
                            captureManager.isSaving
                                ? "Saving..."
                                : "Save RGB-D Frame",
                            systemImage: "camera.fill"
                        )
                        .font(.headline)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                    .disabled(
                        !supportsSceneDepth ||
                        captureManager.isSaving
                    )

                    Text(
                        "Saved frames: " +
                        "\(captureManager.capturedFrameCount)"
                    )
                    .font(.caption)
                    .foregroundStyle(.white)
                }
                .padding(14)
                .background(.black.opacity(0.7))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .padding()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

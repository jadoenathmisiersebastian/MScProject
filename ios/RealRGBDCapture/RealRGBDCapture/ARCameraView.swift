import ARKit
import SwiftUI

struct ARCameraView: UIViewRepresentable {
    @ObservedObject var captureManager: CaptureManager
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> ARSCNView {
        let sceneView = ARSCNView(frame: .zero)

        captureManager.attach(session: sceneView.session)
        
        sceneView.session.delegate = context.coordinator
        sceneView.automaticallyUpdatesLighting = true
        sceneView.rendersCameraGrain = false

        let configuration = ARWorldTrackingConfiguration()

        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            print("This device does not support ARKit scene depth.")
            return sceneView
        }

        configuration.frameSemantics.insert(.sceneDepth)

        if ARWorldTrackingConfiguration.supportsFrameSemantics(
            .smoothedSceneDepth
        ) {
            configuration.frameSemantics.insert(.smoothedSceneDepth)
        }

        sceneView.session.run(
            configuration,
            options: [.resetTracking, .removeExistingAnchors]
        )

        return sceneView
    }

    func updateUIView(_ sceneView: ARSCNView, context: Context) {}

    static func dismantleUIView(
        _ sceneView: ARSCNView,
        coordinator: Coordinator
    ) {
        sceneView.session.pause()
    }

    final class Coordinator: NSObject, ARSessionDelegate {
        private var hasConfirmedDepth = false

        func session(_ session: ARSession, didUpdate frame: ARFrame) {
            guard !hasConfirmedDepth, let depthData = frame.sceneDepth else {
                return
            }

            hasConfirmedDepth = true

            let depthMap = depthData.depthMap
            let width = CVPixelBufferGetWidth(depthMap)
            let height = CVPixelBufferGetHeight(depthMap)

            print("ARKit scene depth active: \(width) x \(height)")
        }

        func session(
            _ session: ARSession,
            cameraDidChangeTrackingState camera: ARCamera
        ) {
            print("ARKit tracking state: \(camera.trackingState)")
        }
    }
}

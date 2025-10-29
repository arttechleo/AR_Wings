import * as faceDetection from '@tensorflow-models/face-detection';

export class FaceGate {
  static async create(debug) {
    // Uses MediaPipe Face Detection via CDN (lightweight + fast)
    const detector = await faceDetection.createDetector(
      faceDetection.SupportedModels.MediaPipeFaceDetector,
      {
        runtime: 'mediapipe',
        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_detection',
        modelType: 'short',
      }
    );
    debug.log('success', 'Face detector ready');
    return new FaceGate(detector, debug);
  }

  constructor(detector, debug) {
    this.detector = detector;
    this.debug = debug;
    this.lastFaces = [];
  }

  async estimate(video, facingMode) {
    try {
      this.lastFaces = await this.detector.estimateFaces(video, { flipHorizontal: facingMode === 'user' });
    } catch (e) {
      this.lastFaces = [];
      this.debug.log('error', `Face error: ${e.message}`);
    }
  }

  isFacePresent(minScore = 0.6) {
    return this.lastFaces.some(f => (f?.score ?? 0) >= minScore);
  }
}
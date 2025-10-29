import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';

export class PoseTracker {
  static async create(debug) {
    await tf.setBackend('webgl');
    await tf.ready();
    const detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
    );
    const t = new PoseTracker(detector, debug);
    debug.log('success', 'MoveNet loaded');
    return t;
  }

  constructor(detector, debug) {
    this.detector = detector;
    this.debug = debug;
    this.lastGood = null; // { left:{x,y,score}, right:{x,y,score} }
  }

  async estimate(video, facingMode) {
    try {
      const poses = await this.detector.estimatePoses(video);
      if (!poses?.length) { this.lastGood = null; this.debug.updatePoseStatus('No person'); return; }
      const kps = poses[0].keypoints;
      const L = kps.find(k => k.name === 'left_shoulder');
      const R = kps.find(k => k.name === 'right_shoulder');
      if (L?.score > 0.4 && R?.score > 0.4) {
        this.lastGood = { left: { x: L.x, y: L.y, score: L.score }, right: { x: R.x, y: R.y, score: R.score } };
        this.debug.updatePoseStatus(`Detected (L:${L.score.toFixed(2)}, R:${R.score.toFixed(2)})`);
      } else {
        this.lastGood = null;
        this.debug.updatePoseStatus('Low confidence');
      }
    } catch (e) {
      this.lastGood = null;
      this.debug.log('error', `Pose error: ${e.message}`);
    }
  }

  getLastShoulders() { return this.lastGood; }
}
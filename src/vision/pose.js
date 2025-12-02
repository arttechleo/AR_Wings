import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';

export class PoseTracker {
  static async create(debug) {
    // Try WebGPU first (faster on modern devices), fallback to WebGL
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    try {
      // Check if WebGPU is available
      if (navigator.gpu && !isMobile) {
        // WebGPU is faster but may not be available on all devices
        await tf.setBackend('webgpu');
        await tf.ready();
        debug.log('info', 'Using WebGPU backend (fastest)');
      } else {
        // Use WebGL (more compatible)
        await tf.setBackend('webgl');
        await tf.ready();
        debug.log('info', 'Using WebGL backend');
      }
    } catch (e) {
      // Fallback to WebGL if WebGPU fails
      await tf.setBackend('webgl');
      await tf.ready();
      debug.log('warning', `WebGPU unavailable, using WebGL: ${e.message}`);
    }
    
    // Use the fastest model: MoveNet Lightning (already optimized)
    const detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { 
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        enableSmoothing: false, // Disable smoothing for speed
        minPoseScore: 0.25 // Lower threshold for faster detection
      }
    );
    const t = new PoseTracker(detector, debug);
    debug.log('success', `MoveNet loaded (backend: ${tf.getBackend()})`);
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
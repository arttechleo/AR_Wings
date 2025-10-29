import * as bodySegmentation from '@tensorflow-models/body-segmentation';

export class Segmentation {
  static async create(debug) {
    // Use MediaPipe Selfie Segmentation via CDN for speed
    const segmenter = await bodySegmentation.createSegmenter(
      bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation,
      {
        runtime: 'mediapipe',
        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation',
        modelType: 'general',
      }
    );
    debug.log('success', 'Segmentation ready');
    return new Segmentation(segmenter, debug);
  }

  constructor(segmenter, debug) {
    this.segmenter = segmenter;
    this.debug = debug;
    this.maskCanvas = document.createElement('canvas');
    this.maskCtx = this.maskCanvas.getContext('2d');
  }

  async segment(video, facingMode) {
    try {
      const people = await this.segmenter.segmentPeople(video, { multiSegmentation: false, segmentationThreshold: 0.7 });
      if (!people?.length) return;

      const fg = { r: 255, g: 255, b: 255, a: 255 };
      const bg = { r: 0, g: 0, b: 0, a: 0 };
      const img = await bodySegmentation.toBinaryMask(people, fg, bg, true, 3);

      // Resize once
      if (this.maskCanvas.width !== img.width || this.maskCanvas.height !== img.height) {
        this.maskCanvas.width = img.width;
        this.maskCanvas.height = img.height;
      }
      this.maskCtx.putImageData(img, 0, 0);

      // when facingMode === 'user', mirroring happens in occlusion shader via uFlipX
    } catch (e) {
      // Non-fatal
    }
  }

  getMaskCanvas() { return this.maskCanvas; }
}
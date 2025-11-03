import * as bodySegmentation from '@tensorflow-models/body-segmentation';

export class Segmentation {
  static async create(debug) {
    // Use BodyPix (TFJS runtime) to avoid Mediapipe WASM conflicts
    // Optimize for mobile performance
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    const segmenter = await bodySegmentation.createSegmenter(
      bodySegmentation.SupportedModels.BodyPix,
      {
        runtime: 'tfjs',
        modelType: isMobile ? 'lite' : 'general', // Use lighter model on mobile
        enableSmoothing: true,
      }
    );
    debug.log('success', `Segmentation ready (${isMobile ? 'lite' : 'general'} model)`);
    return new Segmentation(segmenter, debug, isMobile);
  }

  constructor(segmenter, debug, isMobile = false) {
    this.segmenter = segmenter;
    this.debug = debug;
    this.isMobile = isMobile;
    this.maskCanvas = document.createElement('canvas');
    this.maskCtx = this.maskCanvas.getContext('2d');
  }

  async segment(video, facingMode) {
    try {
      // Use lower resolution on mobile for performance
      const outputStride = this.isMobile ? 16 : 8; // Lower stride = higher res but slower
      const people = await this.segmenter.segmentPeople(video, { 
        multiSegmentation: false, 
        segmentationThreshold: 0.7,
        flipHorizontal: facingMode === 'user'
      });
      if (!people?.length) return;

      const fg = { r: 255, g: 255, b: 255, a: 255 };
      const bg = { r: 0, g: 0, b: 0, a: 0 };
      // Use lower resolution mask on mobile
      const maskBlurAmount = this.isMobile ? 5 : 3;
      const img = await bodySegmentation.toBinaryMask(people, fg, bg, true, maskBlurAmount);

      // Resize canvas to match output (might be downscaled for performance)
      if (this.maskCanvas.width !== img.width || this.maskCanvas.height !== img.height) {
        this.maskCanvas.width = img.width;
        this.maskCanvas.height = img.height;
      }
      this.maskCtx.putImageData(img, 0, 0);

      // when facingMode === 'user', mirroring happens in occlusion shader via uFlipX
    } catch (e) {
      // Non-fatal - silent on mobile to avoid spam
      if (!this.isMobile) {
        console.warn('Segmentation error:', e);
      }
    }
  }

  getMaskCanvas() { return this.maskCanvas; }
}
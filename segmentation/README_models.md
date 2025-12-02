# Human Segmentation Models Comparison

This document compares candidate models for real-time human body segmentation in the browser for the AR Wings project.

## Models Evaluated

### 1. TensorFlow.js BodyPix
**Status**: ✅ **SELECTED**

**Pros:**
- Part of TensorFlow.js ecosystem (already a dependency)
- Provides both person segmentation and 24 body part segmentation
- Good documentation and community support
- Can achieve 30-45 FPS on modern hardware with optimization
- Multiple model variants (ResNet50, MobileNetV1) for speed/quality tradeoff
- Works with WebGL backend (already using TF.js)

**Cons:**
- Larger model size (~12-25MB depending on variant)
- Performance can vary on lower-end devices
- Requires TensorFlow.js (already installed, so not a con for us)

**Performance Estimates:**
- MobileNetV1 (fast): ~30-40 FPS on modern laptop (1280x720 input)
- ResNet50 (quality): ~20-30 FPS on modern laptop

**Integration Ease:**
- Very easy - single npm install, simple API
- Works directly with HTMLVideoElement
- Can output masks as ImageData or WebGLTexture

---

### 2. MediaPipe Selfie Segmentation
**Status**: ⚠️ Considered but not selected

**Pros:**
- Extremely fast (~60+ FPS possible)
- Very lightweight (~1-2MB)
- Optimized specifically for selfies/portraits
- Good edge quality

**Cons:**
- Less accurate for full body segmentation
- No body part segmentation (only person/background)
- Orientation detection would require separate pose model
- Less flexible than BodyPix

**Performance Estimates:**
- ~45-60 FPS on modern laptop

**Integration Ease:**
- Requires MediaPipe JavaScript SDK
- Different API structure than TF.js

---

### 3. ONNX Runtime Web with Custom Models
**Status**: ❌ Not selected

**Pros:**
- Can use state-of-the-art models (e.g., YOLOv8-seg, SAM)
- High quality results
- Cross-framework compatibility

**Cons:**
- More complex setup and build configuration
- Larger bundle size
- Requires model conversion and optimization
- Steeper learning curve
- Performance depends heavily on model choice

**Performance Estimates:**
- Varies widely (10-40 FPS depending on model)

**Integration Ease:**
- More complex - requires ONNX Runtime setup
- Need to handle model loading and preprocessing manually

---

## Selection Rationale

**BodyPix was selected** because:
1. We're already using TensorFlow.js, so no new major dependencies
2. Provides both segmentation and body parts (useful for orientation detection)
3. Good balance of quality and performance
4. Easy integration with existing codebase
5. Active maintenance and documentation
6. Can meet 30-45 FPS target with MobileNetV1 variant

## Implementation Strategy

- Use `MobileNetV1` variant for speed (target: 30-45 FPS)
- Downscale input frames to 512x384 or 640x480 for inference
- Run segmentation every 2-3 frames (similar to pose detection throttling)
- Upscale mask to full resolution for rendering
- Use body part segmentation to improve orientation detection


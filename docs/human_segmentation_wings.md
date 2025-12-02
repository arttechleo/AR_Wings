# AR Wings with Human Segmentation

## Overview

This feature adds real-time human body segmentation and orientation-aware wing positioning to the AR Wings application. The wings now intelligently position themselves based on whether the person is facing the camera or has their back to it, creating a more realistic AR experience.

## Features

- **Real-time Human Segmentation**: Uses TensorFlow.js BodyPix model to segment the human body from the background in real-time
- **Orientation Detection**: Determines if a person is facing the camera or has their back to it
- **Orientation-Aware Wing Positioning**:
  - **Facing Camera**: Wings appear further back, positioned behind the person's shoulders
  - **Back to Camera**: Wings appear closer, positioned on the person's back
- **Smooth Transitions**: Temporal smoothing prevents jitter when orientation changes
- **Performance Optimized**: Maintains 30-45 FPS on modern hardware through throttling and optimization

## Technical Implementation

### Segmentation Model

**Selected Model**: TensorFlow.js BodyPix (MobileNetV1 variant)

**Why BodyPix?**
- Part of TensorFlow.js ecosystem (already a dependency)
- Provides both person segmentation and 24 body part segmentation
- Good balance of quality and performance
- Can achieve 30-45 FPS with optimization
- Easy integration with existing codebase

**Configuration:**
- Architecture: MobileNetV1 (fast variant)
- Output Stride: 16 (faster inference)
- Multiplier: 0.75 (speed/quality tradeoff)
- Internal Resolution: Medium (512x384 or similar)

See `segmentation/README_models.md` for detailed model comparison.

### Architecture

```
Camera Feed
    ↓
[Pose Detection] (MoveNet - every 3 frames)
    ↓
[Segmentation] (BodyPix - every 2-3 frames)
    ↓
[Orientation Estimation] (Combines pose + segmentation)
    ↓
[WingsController] (Positioning & rotation)
    ↓
[Three.js Rendering] (Gaussian Splatting wings)
```

### Key Components

1. **segmentation/humanSegmentation.js**
   - Initializes BodyPix model
   - Processes video frames to generate segmentation masks
   - Estimates orientation from body parts

2. **segmentation/orientation.js**
   - Estimates orientation from pose keypoints
   - Provides temporal smoothing to reduce jitter
   - Combines multiple orientation sources

3. **wingsController.js**
   - Manages wing positioning based on orientation
   - Handles smooth transitions between orientations
   - Adjusts depth and scale dynamically

## Setup Instructions

### Prerequisites

- Node.js and npm installed
- Modern browser with WebGL support
- Camera permissions

### Installation

1. Install dependencies:
```bash
npm install
```

The project already includes `@tensorflow-models/body-pix` in dependencies.

2. Run the development server:
```bash
npm run dev
```

3. Open the application in your browser (HTTPS recommended for camera access)

4. Click "START AR" and allow camera permissions

### Building for Production

```bash
npm run build
```

## Usage

1. **Start the Application**: Click "START AR" button
2. **Position Yourself**: Stand in front of the camera
3. **Face the Camera**: Wings will appear behind you
4. **Turn Around**: Wings will reposition to your back

### Keyboard Shortcuts

- `D` key (if implemented): Toggle debug panel
- `?debug=1` URL parameter: Enable debug mode

## Performance Tuning

The application is optimized to maintain 30-45 FPS. If performance drops:

1. **Reduce Segmentation Frequency**: Increase `SEGMENTATION_SKIP_FRAMES` in `script.js`
2. **Lower Model Resolution**: Change `internalResolution` to 'low' in `humanSegmentation.js`
3. **Use Faster Model**: Already using MobileNetV1 (fastest variant)

### Performance Monitoring

The debug panel shows:
- **FPS**: Current frame rate
- **Seg: Xms**: Segmentation inference time in milliseconds

## Known Limitations

1. **Orientation Detection Accuracy**:
   - Less reliable when person is partially occluded
   - May be uncertain at extreme angles (profile view)
   - Requires clear view of face or upper body

2. **Performance**:
   - Lower-end devices may struggle to maintain 30 FPS
   - First few frames may be slower while models initialize

3. **Occlusion**:
   - Currently relies on depth-based positioning
   - Full occlusion compositing (mask-based) is prepared but not fully implemented
   - Wings may occasionally appear in front of person in edge cases

4. **Lighting Conditions**:
   - Works best in well-lit environments
   - Very dark or very bright conditions may reduce accuracy

5. **Multiple People**:
   - Currently tracks one person (first detected)
   - Multiple people may cause confusion

## Future Improvements

- Full mask-based occlusion compositing in shaders
- Better orientation detection using pose landmarks
- Support for multiple people
- Adaptive quality based on device performance
- Web Worker support for off-main-thread segmentation

## Troubleshooting

### Segmentation Not Working

- Check browser console for errors
- Ensure camera permissions are granted
- Verify TensorFlow.js backend is initialized (should show "webgl" in debug panel)
- Try refreshing the page to reload models

### Low Performance

- Check if WebGL is enabled
- Reduce browser window size
- Close other tabs/applications
- Check debug panel for segmentation inference time
- Consider using a device with better GPU

### Wings Not Positioning Correctly

- Ensure good lighting
- Stand at appropriate distance (2-6 feet from camera)
- Ensure shoulders are visible
- Check debug panel for pose detection confidence scores

## Dependencies

- `@tensorflow/tfjs`: ^4.22.0
- `@tensorflow-models/body-pix`: ^2.2.1
- `@tensorflow-models/pose-detection`: ^2.1.3
- `three`: ^0.180.0
- `@sparkjsdev/spark`: ^0.1.9

## References

- [TensorFlow.js BodyPix Documentation](https://github.com/tensorflow/tfjs-models/tree/master/body-pix)
- [BodyPix Model Card](https://github.com/tensorflow/tfjs-models/blob/master/body-pix/src/body_pix_model.ts)


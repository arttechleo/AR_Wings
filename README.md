# AR Wings - Gaussian Splatting AR Experience

An augmented reality web application that overlays Gaussian Splatting angel wings on users in real-time using pose detection and human segmentation.

## Features

- Real-time pose detection using TensorFlow.js MoveNet
- Gaussian Splatting wing rendering using Spark.js
- **Human Segmentation**: Real-time body segmentation with orientation-aware wing positioning (see [AR Wings with Human Segmentation](docs/human_segmentation_wings.md))
- Smooth wing positioning and rotation
- Support for front and rear cameras

## Quick Start

1. Install dependencies:
```bash
npm install
```

2. Run development server:
```bash
npm run dev
```

3. Open in browser and click "START AR"

## Project Structure

- `script.js` - Main application logic
- `index.html` - HTML structure
- `style.css` - Styling
- `segmentation/` - Human segmentation modules
  - `humanSegmentation.js` - BodyPix integration
  - `orientation.js` - Orientation detection
  - `README_models.md` - Model comparison documentation
- `wingsController.js` - Orientation-aware wing positioning
- `assets/` - Gaussian Splatting wing models (.ksplat files)

## AR Wings with Human Segmentation

The application now includes advanced human segmentation and orientation-aware wing positioning. See [Human Segmentation Documentation](docs/human_segmentation_wings.md) for detailed information about:

- How segmentation works
- Orientation detection
- Performance optimization
- Troubleshooting

## Technologies

- **Three.js** - 3D rendering
- **TensorFlow.js** - Machine learning inference
- **BodyPix** - Human body segmentation
- **MoveNet** - Pose detection
- **Spark.js** - Gaussian Splatting rendering
- **Vite** - Build tool

## Browser Requirements

- Modern browser with WebGL support
- Camera access (HTTPS required)
- JavaScript enabled

## Development

```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## License

See LICENSE file (if applicable)


import * as THREE from 'three';
import { DebugLogger } from './utils/debug.js';
import { startCamera, stopCamera, switchCamera, getFacingMode } from './systems/camera.js';
import { createScene, updateVideoPlaneTexture, disposeRenderer } from './three/scene.js';
import { WingsRig } from './three/wings.js';
import { OcclusionMask } from './three/occlusion.js';
import { PoseTracker } from './vision/pose.js';
import { FaceGate } from './vision/face.js';
import { Segmentation } from './vision/segmentation.js';


// ---- Global-ish UI refs (assigned after DOMContentLoaded) ----
let video;
let threeContainer;
let canvas2D;
let ctx2D;


// ---- Debug ----
const debug = new DebugLogger();
window.__debug = debug; // optional for quick console access


// ---- App state ----
let isRunning = false;
let frameCount = 0;
let lastFpsUpdate = performance.now();


// Detection throttles - optimized for mobile (30fps target)
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
const POSE_SKIP = isMobile ? 10 : 3; let poseCounter = 0;
const FACE_SKIP = isMobile ? 20 : 5; let faceCounter = 0;
const SEGM_SKIP = isMobile ? 15 : 2; let segmCounter = 0;
const DISABLE_SEGM_ON_MOBILE = isMobile; // Disable segmentation entirely on mobile for FPS
const RENDER_SKIP = 0; // Disabled - prefer other optimizations
let renderCounter = 0;
let lastSegmUpdate = 0;


// Subsystems
let three; // { renderer, scene, camera, videoPlane }
let wings; // WingsRig
let occluder; // OcclusionMask
let pose; // PoseTracker
let face; // FaceGate
let segm; // Segmentation



// Layout constants
const VIDEO_PLANE_DEPTH = -10.0;
const GROUP_DEPTH = -9.8; // occluder at -9.7 (slightly in front of wings)


// ---------- Bootstrap ----------
window.addEventListener('DOMContentLoaded', async () => {
  debug.log('info', '=== AR Wings (refactor) ===');
  // Bind DOM refs now that the document is ready
  video = document.getElementById('video');
  threeContainer = document.getElementById('three-container');
  canvas2D = document.getElementById('output-canvas');
  ctx2D = canvas2D.getContext('2d');
  setupControls();
  debug.updateStatus('Loading models...');

  // Preload models in parallel
  [pose, face, segm] = await Promise.all([
    PoseTracker.create(debug),
    FaceGate.create(debug),
    Segmentation.create(debug)
  ]);
  debug.updateModelStatus('Pose/Face/Segm ready');
  debug.updateStatus('Ready - Tap Start');
});


function setupControls() {
  const startBtn = document.getElementById('start-btn');
  const instructions = document.getElementById('instructions');
  const toggleBtn = document.getElementById('camera-toggle-btn');

  document.getElementById('toggle-debug').addEventListener('click', () => {
    const panel = document.getElementById('debug-panel');
    panel.classList.toggle('minimized');
    document.getElementById('toggle-debug').textContent = panel.classList.contains('minimized') ? '+' : '−';
  });
  document.getElementById('clear-debug').addEventListener('click', () => {
    document.getElementById('debug-logs').innerHTML = '';
  });

  startBtn.addEventListener('click', async () => {
    instructions.classList.add('hidden');
    await start();
    toggleBtn.style.display = 'block';
  });

  toggleBtn.addEventListener('click', async () => {
    debug.log('info', `Switching camera...`);
    await switchCamera();
    await restart();
  });
}


async function start() {
  try {
    // 1) Camera
    debug.updateStatus('Requesting camera access...');
    await startCamera();
    debug.updateVideoStatus(`Camera stream active (${getFacingMode()})`);
    
    // Wait for video metadata (faster, with fallback)
    if (video.readyState < 2) {
      await new Promise((res, rej) => {
        const timeout = setTimeout(() => {
          // Fallback: use default dimensions if metadata not ready
          if (video.videoWidth && video.videoHeight) {
            res();
          } else {
            // Use default mobile dimensions as fallback
            res();
          }
        }, 2000); // Reduced from 5000ms
        video.onloadedmetadata = () => {
          clearTimeout(timeout);
          res();
        };
        video.onerror = () => {
          clearTimeout(timeout);
          res(); // Continue even on error
        };
      });
    }
    
    // Use actual or fallback dimensions (read-only properties, use variables instead)
    let videoWidth = video.videoWidth || 640;
    let videoHeight = video.videoHeight || 480;
    if (!video.videoWidth || !video.videoHeight) {
      debug.log('warning', `Using fallback video dimensions: ${videoWidth}x${videoHeight}`);
    }
    
    debug.log('info', `Video dimensions: ${videoWidth}x${videoHeight}`);

    // 2) Canvas sizes
    canvas2D.width = videoWidth;
    canvas2D.height = videoHeight;

    // 3) 3D setup
    debug.updateStatus('Initializing 3D scene...');
    three = createScene({ video, container: threeContainer, videoPlaneDepth: VIDEO_PLANE_DEPTH, debug });
    
    // Ensure container is visible and sized
    threeContainer.style.width = '100vw';
    threeContainer.style.height = '100vh';
    threeContainer.style.position = 'absolute';
    threeContainer.style.top = '0';
    threeContainer.style.left = '0';
    
    wings = new WingsRig({ scene: three.scene, debug });
    await wings.loadAssets(three.renderer);

    // Only create occluder if segmentation is enabled (not on mobile)
    if (!DISABLE_SEGM_ON_MOBILE) {
      occluder = new OcclusionMask({
        scene: three.scene,
        camera: three.camera,
        depthZ: GROUP_DEPTH + 0.1, // slightly in front of wings
        debug,
      });
    } else {
      occluder = null; // No occlusion on mobile for performance
      debug.log('info', 'Segmentation disabled on mobile for performance');
    }

    isRunning = true;
    debug.updateStatus('Running - Stand back!');
    
    // Keep renderer sized with viewport
    window.addEventListener('resize', handleResize, { passive: true });
    handleResize(); // Initial size
    
    // Force first render
    updateVideoPlaneTexture(three.videoPlane);
    three.renderer.render(three.scene, three.camera);
    
    requestAnimationFrame(loop);
  } catch (error) {
    debug.log('error', `Start failed: ${error.message}`);
    debug.updateStatus(`Error: ${error.message}`);
    throw error;
  }
}


async function restart() {
  isRunning = false;
  stopCamera();

  // Dispose ThreeJS renderer cleanly
  disposeRenderer(three?.renderer, three?.containerEl);
  three = null;

  // Reset counters
  poseCounter = faceCounter = segmCounter = 0;

  await start();
}

function loop(now) {
  if (!isRunning) return;
  requestAnimationFrame(loop);

  // Frame skipping for mobile performance (only skip rendering, not detection)
  renderCounter++;
  const shouldRender = RENDER_SKIP === 0 || renderCounter % (RENDER_SKIP + 1) === 0;

  // FPS
  frameCount++;
  if (now - lastFpsUpdate >= 1000) {
    debug.updateFPS(frameCount / ((now - lastFpsUpdate) / 1000));
    frameCount = 0;
    lastFpsUpdate = now;
  }

  // Update 2D canvas (only when rendering)
  if (shouldRender) {
    ctx2D.clearRect(0, 0, canvas2D.width, canvas2D.height);
  }

  // Throttled detections (run asynchronously, don't block render)
  let shoulders = pose.getLastShoulders();
  poseCounter++;
  if (poseCounter >= POSE_SKIP) {
    poseCounter = 0;
    // Don't await - let it run in background
    pose.estimate(video, getFacingMode()).catch(() => {});
  }

  faceCounter++;
  if (faceCounter >= FACE_SKIP) {
    faceCounter = 0;
    face.estimate(video, getFacingMode()).catch(() => {});
  }

  // Skip segmentation entirely on mobile for performance
  if (!DISABLE_SEGM_ON_MOBILE) {
    segmCounter++;
    if (segmCounter >= SEGM_SKIP) {
      segmCounter = 0;
      segm.segment(video, getFacingMode()).catch(() => {});
    }

    // Update occlusion mask texture (only when segmentation runs, and less frequently)
    if (segmCounter === 0 && (now - lastSegmUpdate > 300)) { // Only update every 300ms
      lastSegmUpdate = now;
      const maskCanvas = segm.getMaskCanvas();
      if (maskCanvas) occluder.updateMask(maskCanvas, getFacingMode());
    }
  } else {
    // On mobile, hide occluder to save performance
    if (occluder?.mesh) {
      occluder.mesh.visible = false;
    }
  }

  // Decide visibility: face-gate (relaxed for better UX)
  // On mobile, skip face detection entirely for performance
  const faceOK = isMobile ? true : face.isFacePresent(0.5); // Always true on mobile
  const hasShoulders = !!shoulders;
  // Show wings if we have shoulders OR last anchor (even without face initially)
  const wingsVisible = hasShoulders || (wings.hasLastAnchor() && faceOK);

  // Anchor + position
  if (hasShoulders) {
    const { left, right } = shoulders;
    wings.updateFromShoulders({
      left,
      right,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      facingMode: getFacingMode(),
    });
    // debug dots (only when rendering)
    if (shouldRender) {
      drawPoint(ctx2D, left.x, left.y, '#00ff88');
      drawPoint(ctx2D, right.x, right.y, '#00ff88');
    }
  }
  wings.setVisible(wingsVisible);

  // Render 3D - optimize texture updates (only when we actually render)
  if (shouldRender) {
    if (three?.videoPlane?.material?.map) {
      // Only update texture periodically on mobile (not every frame)
      if (isMobile) {
        // Update every 3 frames on mobile
        if (renderCounter % 3 === 0 && video.readyState >= video.HAVE_CURRENT_DATA) {
          three.videoPlane.material.map.needsUpdate = true;
        }
      } else {
        // Update every frame on desktop
        if (video.readyState >= video.HAVE_CURRENT_DATA) {
          three.videoPlane.material.map.needsUpdate = true;
        }
      }
    }
    if (three?.renderer) {
      three.renderer.render(three.scene, three.camera);
    }
  }
}


function drawPoint(ctx, x, y, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, Math.PI * 2);
  ctx.fill();
}

function handleResize() {
  if (!three?.renderer || !three?.containerEl) return;
  
  // Use viewport dimensions for mobile
  const width = window.innerWidth || three.containerEl.clientWidth || 640;
  const height = window.innerHeight || three.containerEl.clientHeight || 480;
  
  three.renderer.setSize(width, height);
  three.camera.aspect = width / height;
  three.camera.updateProjectionMatrix();
  
  // Update video plane size
  if (three.videoPlane) {
    const fovRad = THREE.MathUtils.degToRad(three.camera.fov);
    const planeH = Math.abs(2 * Math.tan(fovRad / 2) * Math.abs(VIDEO_PLANE_DEPTH));
    const planeW = planeH * three.camera.aspect;
    three.videoPlane.scale.set(planeW, planeH, 1);
  }
}



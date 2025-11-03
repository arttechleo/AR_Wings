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
const POSE_SKIP = isMobile ? 6 : 3; let poseCounter = 0;
const FACE_SKIP = isMobile ? 12 : 5; let faceCounter = 0;
const SEGM_SKIP = isMobile ? 5 : 2; let segmCounter = 0;
const RENDER_SKIP = 0; // Disabled - prefer other optimizations
let renderCounter = 0;


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
    
    // Wait for video metadata
    if (video.readyState < 2) {
      await new Promise((res, rej) => {
        const timeout = setTimeout(() => rej(new Error('Video metadata timeout')), 5000);
        video.onloadedmetadata = () => {
          clearTimeout(timeout);
          res();
        };
        video.onerror = () => {
          clearTimeout(timeout);
          rej(new Error('Video element error'));
        };
      });
    }
    
    // Ensure video dimensions are valid
    if (!video.videoWidth || !video.videoHeight) {
      throw new Error('Video dimensions not available');
    }
    
    debug.log('info', `Video dimensions: ${video.videoWidth}x${video.videoHeight}`);

    // 2) Canvas sizes
    canvas2D.width = video.videoWidth;
    canvas2D.height = video.videoHeight;

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

    occluder = new OcclusionMask({
      scene: three.scene,
      camera: three.camera,
      depthZ: GROUP_DEPTH + 0.1, // slightly in front of wings
      debug,
    });

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

  segmCounter++;
  if (segmCounter >= SEGM_SKIP) {
    segmCounter = 0;
    segm.segment(video, getFacingMode()).catch(() => {});
  }

  // Update occlusion mask texture (only when segmentation runs)
  if (segmCounter === 0) {
    const maskCanvas = segm.getMaskCanvas();
    if (maskCanvas) occluder.updateMask(maskCanvas, getFacingMode());
  }

  // Decide visibility: face-gate
  const faceOK = face.isFacePresent(0.7);
  const hasShoulders = !!shoulders;
  const wingsVisible = faceOK && (hasShoulders || wings.hasLastAnchor());

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
      // Only update texture when video has new frame
      if (video.readyState >= video.HAVE_CURRENT_DATA) {
        three.videoPlane.material.map.needsUpdate = true;
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



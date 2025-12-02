import * as THREE from 'three';
import { DebugLogger } from './utils/debug.js';
import { startCamera, stopCamera, switchCamera, getFacingMode } from './systems/camera.js';
import { createScene, disposeRenderer } from './three/scene.js';
import { WingsRig } from './three/wings.js';
import { PoseTracker } from './vision/pose.js';


// ---- Global-ish UI refs (assigned after DOMContentLoaded) ----
let video;
let threeContainer;


// ---- Debug ----
const debug = new DebugLogger();
window.__debug = debug; // optional for quick console access


// ---- App state ----
let isRunning = false;
let frameCount = 0;
let lastFpsUpdate = performance.now();


// MVP: Minimal detection - only pose, maximum speed
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
const isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
const POSE_SKIP = isMobile ? 8 : 2; // More frequent for responsiveness
let poseCounter = 0;
let videoFrameCallback = null;


// Subsystems - MVP: Only what's needed
let three; // { renderer, scene, camera, videoPlane }
let wings; // WingsRig
let pose; // PoseTracker



// Layout constants
const VIDEO_PLANE_DEPTH = -10.0;
const GROUP_DEPTH = -9.8; // occluder at -9.7 (slightly in front of wings)


// ---------- Bootstrap ----------
window.addEventListener('DOMContentLoaded', async () => {
  debug.log('info', '=== AR Wings (refactor) ===');
  // Bind DOM refs now that the document is ready
  video = document.getElementById('video');
  threeContainer = document.getElementById('three-container');
  setupControls();
  debug.updateStatus('Loading pose model...');

  // MVP: Only load pose detection - nothing else
  pose = await PoseTracker.create(debug);
  debug.updateModelStatus('Pose ready (MVP mode)');
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

    // 2) 3D setup
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

    isRunning = true;
    debug.updateStatus('Running - Stand back!');
    
    // Keep renderer sized with viewport
    window.addEventListener('resize', handleResize, { passive: true });
    handleResize(); // Initial size
    
    // Use requestVideoFrameCallback for efficient video updates (if available)
    if (video.requestVideoFrameCallback) {
      const updateVideoFrame = () => {
        if (!isRunning || !three?.videoPlane?.material?.map) return;
        three.videoPlane.material.map.needsUpdate = true;
        if (isRunning) {
          videoFrameCallback = video.requestVideoFrameCallback(updateVideoFrame);
        }
      };
      videoFrameCallback = video.requestVideoFrameCallback(updateVideoFrame);
    }
    
    // Force first render
    if (three?.videoPlane?.material?.map) {
      three.videoPlane.material.map.needsUpdate = true;
    }
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

  // Clean up video frame callback if it exists
  if (videoFrameCallback && video.cancelVideoFrameCallback) {
    video.cancelVideoFrameCallback(videoFrameCallback);
    videoFrameCallback = null;
  }

  // Dispose ThreeJS renderer cleanly
  disposeRenderer(three?.renderer, three?.containerEl);
  three = null;

  // Reset counters
  poseCounter = 0;

  await start();
}

function loop(now) {
  if (!isRunning) return;
  requestAnimationFrame(loop);

  // FPS
  frameCount++;
  if (now - lastFpsUpdate >= 1000) {
    debug.updateFPS(frameCount / ((now - lastFpsUpdate) / 1000));
    frameCount = 0;
    lastFpsUpdate = now;
  }

  // MVP: Only pose detection - run asynchronously, don't block render
  let shoulders = pose.getLastShoulders();
  poseCounter++;
  if (poseCounter >= POSE_SKIP) {
    poseCounter = 0;
    // Don't await - let it run in background for zero latency
    pose.estimate(video, getFacingMode()).catch(() => {});
  }

  // MVP: Immediate visibility - no delays, no face checks, no segmentation
  const hasShoulders = !!shoulders;
  const splatsReady = wings.isSplatDataReady();
  const wingsVisible = splatsReady && (hasShoulders || wings.hasLastAnchor());

  // Immediate position update - no smoothing delays
  if (hasShoulders) {
    const { left, right } = shoulders;
    wings.updateFromShoulders({
      left,
      right,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      facingMode: getFacingMode(),
    });
  }
  wings.setVisible(wingsVisible);

  // MVP: Always render - video texture updated via callback (set in start())
  if (three?.renderer) {
    three.renderer.render(three.scene, three.camera);
  }
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



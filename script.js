// --- MODULE IMPORTS ---
import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection'; 
import { SplatMesh, SparkRenderer } from "@sparkjsdev/spark";
import { initPersonSegmentation, runPersonSegmentationFrame, isPersonSegmentationReady, generateMaskFromPose } from './segmentation/personSegmentation.js';
import { estimateOrientationFromPose, smoothOrientation } from './segmentation/orientation.js';
import { WingsController } from './wingsController.js';
import { OcclusionCompositor } from './render/occlusionCompositor.js';
import {
    isDepthSupported,
    initDepthSensing,
    getCurrentDepthTexture,
    getDepthParams,
    isDepthAvailable,
    disposeDepthSensing,
} from './depth/depthOcclusion.js';

/**
 * === CURRENT AR PIPELINE SUMMARY ===
 * 
 * The AR pipeline follows this flow:
 * 
 * 1. CAMERA INITIALIZATION
 *    - Request camera stream via getUserMedia
 *    - Create video element and play stream
 *    - Wait for metadata to get video dimensions
 * 
 * 2. 3D SCENE SETUP (Three.js)
 *    - Create WebGL renderer with alpha channel
 *    - Set up perspective camera
 *    - Create video background plane at VIDEO_PLANE_DEPTH (-10.0)
 *    - Initialize SparkRenderer for Gaussian Splatting support
 * 
 * 3. GAUSSIAN SPLATTING WINGS LOADING
 *    - Load left and right wing .ksplat files as SplatMesh objects
 *    - Add wings to a THREE.Group for coordinated positioning
 *    - Fallback to box geometry if splat files fail to load
 * 
 * 4. POSE DETECTION (MoveNet - TensorFlow.js)
 *    - Load MoveNet Lightning model for lightweight pose estimation
 *    - Run pose detection every POSE_DETECTION_SKIP_FRAMES (3) frames for performance
 *    - Extract shoulder keypoints (left_shoulder, right_shoulder)
 *    - Store last good pose keypoints for smoothing
 * 
 * 5. WING POSITIONING & RENDERING
 *    - Calculate average shoulder position → map to 3D coordinates
 *    - Position wings group based on normalized shoulder coordinates
 *    - Calculate dynamic horizontal offset from shoulder width
 *    - Apply rotation based on shoulder height difference (tilt detection)
 *    - Render wings at TEST_DEPTH_Z (-5.0) with BACK_OFFSET_Z offset
 * 
 * 6. RENDER LOOP
 *    - Update video texture
 *    - Run pose detection (throttled)
 *    - Position wings based on pose keypoints
 *    - Render Three.js scene (video background + wings)
 *    - Draw debug points on 2D canvas overlay
 * 
 * Current limitations:
 * - No human segmentation mask (only pose keypoints)
 * - Wings always appear at fixed depth relative to shoulders
 * - No occlusion handling (wings may appear in front of person)
 * - No orientation detection (front vs back)
 */ 

// Global variables for the scene and pose detection
let scene, camera;
let threeRendererInstance; 
let wingsAssetLeft, wingsAssetRight; 
let wingsGroup; 
let video, canvas, ctx;
let poseModel;
let debugLogger;
let isRunning = false;
let frameCount = 0;
let lastFpsUpdate = Date.now();
let videoBackgroundPlane;
let videoTexture = null; // Store video texture for occlusion compositor

// --- SEGMENTATION VARIABLES ---
let segmentationReady = false;
let segmentationMaskTexture = null;
let segmentationFrameCounter = 0;
let segmentationSkipFrames = 3; // Run segmentation every 3 frames for better performance
let orientationHistory = []; // Store recent orientations for smoothing
const MAX_ORIENTATION_HISTORY = 5;

// --- OCCLUSION COMPOSITOR ---
let occlusionCompositor = null;

// --- DEPTH/LIDAR OCCLUSION ---
let depthEnabled = false;
let depthOcclusionMesh = null;

// --- WINGS CONTROLLER ---
let wingsController = null; 

// --- STATE FLAGS ---
let isSplatAttempted = false;
let isSplatDataReady = false; 
let lastGoodPoseKeypoints = null; // 👈 CRITICAL: Stores the last known reliable pose data

// *** PERFORMANCE OPTIMIZATION VARIABLES ***
let POSE_DETECTION_SKIP_FRAMES = 6; // Run AI only once every 6 frames (increased for performance)
const MAX_POSE_SKIP_FRAMES = 10; // Maximum skip frames for adaptive tuning
let poseDetectionFrameCounter = 0;

// Performance monitoring
let segmentationInferenceTime = 0;
let lastSegmentationInferenceStart = 0;

// Debug toggles
let DEBUG_MODE = false; // Toggle with 'D' key or ?debug=1 URL param
const DEBUG_DISABLE_OCCLUSION = false; // Set to true to disable occlusion plane for debugging
const DEBUG_FORCE_WINGS_VISIBLE = false; // Set to true to always show wings (ignore pose/segmentation)

// Adaptive performance
let lowFpsCounter = 0;
const LOW_FPS_THRESHOLD = 25;
const LOW_FPS_DURATION = 3000; // 3 seconds
// ******************************************

// Smoothing variable for stable Group positioning
let smoothedGroupPosition = { x: 0, y: 0, z: 0 }; 
const SMOOTHING_FACTOR = 0.6; 

// Gaussian Splatting configuration
const USE_GAUSSIAN_SPLAT = true; 

// *** ASSET PATHS (Ensure these files exist in your 'assets' folder) ***
const SPLAT_PATH_LEFT_WING = new URL('./assets/leftwing.ksplat', import.meta.url).href;
const SPLAT_PATH_RIGHT_WING = new URL('./assets/rightwing.ksplat', import.meta.url).href;

// --- CRITICAL WING CONSTANTS ---
const WING_SPLAT_SCALE_FACTOR_BASE = 1.8; 
let currentWingScale = WING_SPLAT_SCALE_FACTOR_BASE;
let currentHorizontalOffset = 0; 
const WING_VERTICAL_SHIFT = 0.5; 
const SHOULDER_PIVOT_MULTIPLIER = 0.55; 
const MIN_HORIZONTAL_OFFSET = 0.25; 
const MAX_X_ROTATION = Math.PI / 6; 
const Y_DIFFERENCE_SENSITIVITY = 150; 

let CAMERA_MODE = 'environment';

// Video resolution configuration (can be reduced for performance)
const VIDEO_RESOLUTION = {
    width: { ideal: 960 },  // Reduced from 1280 for better performance
    height: { ideal: 540 }, // Reduced from 720 for better performance
}; 

// --- AR SETTINGS (FIXED VALUES) ---
const TEST_DEPTH_Z = -5.0; 
const BACK_OFFSET_Z = -5.0; 
const VIDEO_PLANE_DEPTH = -10.0; 
const SPLAY_ANGLE = Math.PI / 12; 


// === DEBUG LOGGER CLASS (STANDARD - UNCHANGED) ===
class DebugLogger {
    constructor() {
        this.logsContainer = document.getElementById('debug-logs');
        this.statusText = document.getElementById('status-text');
        this.videoStatus = document.getElementById('video-status');
        this.modelStatus = document.getElementById('model-status');
        this.poseStatus = document.getElementById('pose-status'); 
        this.assetStatus = document.getElementById('asset-status');
        this.fpsCounter = document.getElementById('fps-counter');
        this.positionStatus = document.getElementById('position-status'); 
        this.maxLogs = 30;
        this.setupControls();
    }
    setupControls() {
        const panel = document.getElementById('debug-panel');
        if (panel) {
            document.getElementById('toggle-debug').addEventListener('click', () => {
                panel.classList.toggle('minimized');
                document.getElementById('toggle-debug').textContent = panel.classList.contains('minimized') ? '+' : '−';
            });
            document.getElementById('clear-debug').addEventListener('click', () => {
                this.logsContainer.innerHTML = '';
            });
        }
    }
    log(type, message) {
        const logEntry = document.createElement('div');
        logEntry.className = `debug-log ${type}`;
        logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        if (this.logsContainer && this.logsContainer.children.length >= this.maxLogs) {
            this.logsContainer.removeChild(this.logsContainer.lastChild);
        }
        if (this.logsContainer) {
            this.logsContainer.prepend(logEntry);
        }
    }
    updateStatus(status) { if(this.statusText) this.statusText.textContent = status; }
    updateVideoStatus(status) { if(this.videoStatus) this.videoStatus.textContent = status; }
    updateModelStatus(status) { if(this.modelStatus) this.modelStatus.textContent = status; }
    updatePoseStatus(status) { if(this.poseStatus) this.poseStatus.textContent = status; } 
    updateAssetStatus(status) { if(this.assetStatus) this.assetStatus.textContent = status; }
    updateFPS(fps, segTime = 0) { 
        if(this.fpsCounter) {
            if (segTime > 0) {
                this.fpsCounter.textContent = `${fps.toFixed(1)} FPS (Seg: ${segTime.toFixed(1)}ms)`;
            } else {
                this.fpsCounter.textContent = `${fps.toFixed(1)} FPS`;
            }
        }
    }
    updatePositionStatus(posL, rotL, posR, rotR) {
        if (this.positionStatus) {
            this.positionStatus.textContent = `L P: (${posL.x.toFixed(2)}, ${posL.y.toFixed(2)}) R P: (${posR.x.toFixed(2)}, ${posR.y.toFixed(2)}) Offset: ${currentHorizontalOffset.toFixed(2)}`;
        }
    }
}
// === END DEBUG LOGGER CLASS ===

// --- CAMERA SWITCHING LOGIC (UNCHANGED) ---

function setupCameraToggle() {
    const toggleBtn = document.getElementById('camera-toggle-btn');
    if (toggleBtn) {
        toggleBtn.textContent = `Switch to ${CAMERA_MODE === 'user' ? 'Rear' : 'Front'} Camera`;
        toggleBtn.addEventListener('click', switchCamera);
    }
}

async function switchCamera() {
    debugLogger.log('info', `Switching camera from ${CAMERA_MODE} to ${CAMERA_MODE === 'user' ? 'environment' : 'user'}...`);
    
    isRunning = false; 
    if (video && video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
    
    CAMERA_MODE = CAMERA_MODE === 'user' ? 'environment' : 'user';

    const toggleBtn = document.getElementById('camera-toggle-btn');
    if (toggleBtn) {
        toggleBtn.textContent = `Switch to ${CAMERA_MODE === 'user' ? 'Rear' : 'Front'} Camera`;
    }
    
    // Clear smoothed group position and pose data
    smoothedGroupPosition = { x: 0, y: 0, z: 0 };
    lastGoodPoseKeypoints = null;

    await startAR();
}

// === WING SCALE ADJUSTMENT (UNCHANGED) ===
function calculateResponsiveWingScale(videoWidth, videoHeight, baseScale) {
    const aspect = videoWidth / videoHeight;
    let scaleAdjustment = 1.0;
    
    if (aspect < 1.0) { 
        scaleAdjustment = 0.85; 
    } else if (aspect > 1.7) { 
        scaleAdjustment = 1.1; 
    }
    
    const screenHeightFactor = window.innerHeight / 800;
    
    return baseScale * scaleAdjustment * Math.min(1.0, screenHeightFactor);
}


// --- POSE MODEL LOADING (OPTIMIZED: DEDICATED FUNCTION) ---
async function loadPoseModel() {
    if (poseModel === undefined) { 
        debugLogger.updateStatus('Initializing TensorFlow...');
        tf.setBackend('webgl'); 
        await tf.ready(); 
        debugLogger.log('success', `TensorFlow backend ready (${tf.getBackend()}).`);

        debugLogger.updateStatus('Loading AI model (MoveNet)...');
        poseModel = await poseDetection.createDetector(
            poseDetection.SupportedModels.MoveNet,
            { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
        );
        debugLogger.log('success', 'AI model loaded!');
        debugLogger.updateModelStatus('MoveNet Loaded');
    }
}
// --- END POSE MODEL LOADING ---


// --- INITIALIZE & START AR (REFACTORED) ---

function init() {
    debugLogger = new DebugLogger();
    debugLogger.log('info', '=== AR Back Wings Starting ===');
    
    if (typeof THREE === 'undefined' || typeof tf === 'undefined' || typeof poseDetection === 'undefined' || typeof SplatMesh === 'undefined') {
        debugLogger.log('error', 'Module imports failed.');
        document.getElementById('instructions').innerHTML = `<h2>Initialization Failed!</h2><p>Error: Required libraries failed to load.</p>`;
        return;
    }
    debugLogger.log('success', 'Core libraries loaded (THREE, TF, Spark.js)');

    // Start loading the heavy AI models immediately
    loadPoseModel().catch(err => {
        debugLogger.log('error', `FATAL: Could not load Pose Model: ${err.message}`);
    });
    
    // Start loading segmentation model (MediaPipe)
    initPersonSegmentation().then(success => {
        if (success) {
            segmentationReady = true;
            debugLogger.log('success', 'MediaPipe person segmentation model loaded!');
        } else {
            debugLogger.log('warning', 'Person segmentation model failed to load. Wings will work without occlusion.');
        }
    }).catch(err => {
        debugLogger.log('error', `Could not load Segmentation Model: ${err.message}`);
    });

    // Check for debug mode in URL
    const urlParams = new URLSearchParams(window.location.search);
    DEBUG_MODE = urlParams.get('debug') === '1';

    // Keyboard shortcuts for debug toggles
    window.addEventListener('keydown', (e) => {
        if (e.key === 'd' || e.key === 'D') {
            DEBUG_MODE = !DEBUG_MODE;
            debugLogger.log('info', `Debug mode: ${DEBUG_MODE ? 'ON' : 'OFF'}`);
        } else if (e.key === 'o' || e.key === 'O') {
            // Toggle occlusion (if DEBUG_DISABLE_OCCLUSION is mutable)
            const currentOcclusionState = !DEBUG_DISABLE_OCCLUSION;
            debugLogger.log('info', `Occlusion: ${currentOcclusionState ? 'ENABLED' : 'DISABLED'} (toggle via DEBUG_DISABLE_OCCLUSION flag)`);
        } else if (e.key === 'w' || e.key === 'W') {
            // Toggle force wings visible (if DEBUG_FORCE_WINGS_VISIBLE is mutable)
            const currentForceState = DEBUG_FORCE_WINGS_VISIBLE;
            debugLogger.log('info', `Force wings visible: ${currentForceState ? 'ENABLED' : 'DISABLED'} (toggle via DEBUG_FORCE_WINGS_VISIBLE flag)`);
        }
    });

    const startBtn = document.getElementById('start-btn');
    const instructions = document.getElementById('instructions');

    if (startBtn && instructions) {
        startBtn.addEventListener('click', async () => {
            instructions.classList.add('hidden');
            await startAR();
            setupCameraToggle(); 
            const toggleBtn = document.getElementById('camera-toggle-btn');
            if (toggleBtn) toggleBtn.style.display = 'block'; 
        });
    }

    debugLogger.updateStatus('Ready - Tap Start');
}

async function startAR() {
    try {
        const threeContainer = document.getElementById('three-container');
        canvas = document.getElementById('output-canvas');
        ctx = canvas.getContext('2d');
        video = document.getElementById('video');

        // 1. Request Camera Stream (reduced resolution for performance)
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                facingMode: CAMERA_MODE, 
                width: VIDEO_RESOLUTION.width, 
                height: VIDEO_RESOLUTION.height 
            }
        });
        video.srcObject = stream;
        
        // 2. CRITICAL: Attempt play()
        video.play().catch(error => {
            debugLogger.log('warning', `Video play() failed: ${error.message}`);
        }); 
        debugLogger.updateVideoStatus(`Camera stream active (${CAMERA_MODE})`);

        // 3. CRITICAL: Wait for video metadata to load
        await new Promise((resolve) => { video.onloadedmetadata = () => { resolve(video); }; });

        const vw = video.videoWidth;
        const vh = video.videoHeight;
        
        canvas.width = vw;
        canvas.height = vh;
        threeContainer.style.width = '100vw';
        threeContainer.style.height = '100vh';

        // Clean up previous THREE.js instance
        if (threeRendererInstance) {
            threeContainer.removeChild(threeRendererInstance.domElement);
            threeRendererInstance.dispose();
            threeRendererInstance = null;
        }

        debugLogger.updateStatus('Setting up 3D renderer...');
        setupThreeJS(vw, vh); 
        debugLogger.log('success', '3D renderer ready');
        
        currentWingScale = calculateResponsiveWingScale(vw, vh, WING_SPLAT_SCALE_FACTOR_BASE);
        debugLogger.log('info', `Set initial wing scale to: ${currentWingScale.toFixed(2)}`);
        
        debugLogger.updateStatus('Running - Stand back!');
        
        isRunning = true;
        renderLoop();
    } catch (error) {
        debugLogger.log('error', `INIT ERROR: ${error.name}: ${error.message}`);
        debugLogger.updateStatus('FATAL ERROR');
        const instructions = document.getElementById('instructions');
        if (instructions) instructions.classList.add('hidden');
    }
}

// === SETUP THREE.JS (UNCHANGED) ===
function setupThreeJS(videoWidth, videoHeight) {
    const threeContainer = document.getElementById('three-container');
    const containerRect = threeContainer.getBoundingClientRect();

    const threeRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    // Cap pixel ratio for performance (1.5 max on low-end devices)
    const maxPixelRatio = 1.5;
    threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, maxPixelRatio));
    threeRenderer.setSize(containerRect.width, containerRect.height);
    threeRenderer.setClearColor(0x000000, 0); 
    threeContainer.appendChild(threeRenderer.domElement);

    threeRendererInstance = threeRenderer;
    
    new SparkRenderer(threeRenderer);

    // Initialize occlusion compositor (reuse existing containerRect)
    occlusionCompositor = new OcclusionCompositor(threeRenderer, containerRect.width, containerRect.height);

    if (scene) {
        if (videoBackgroundPlane) scene.remove(videoBackgroundPlane);
        if (wingsGroup) scene.remove(wingsGroup);
    } else {
        scene = new THREE.Scene();
    }
    
    wingsGroup = new THREE.Group();
    scene.add(wingsGroup); 
    
    const aspect = containerRect.width / containerRect.height;
    camera = new THREE.PerspectiveCamera(65, aspect, 0.1, 100); 
    camera.position.set(0, 0, 0); 
    scene.add(new THREE.AmbientLight(0xffffff, 1.0));

    // Video Background Plane setup
    videoTexture = new THREE.VideoTexture(video);
    
    // Fix video orientation - mobile cameras output video upside down
    // Use geometry flip instead of texture flip for more reliable results
    videoTexture.flipY = false; // Keep texture unflipped
    
    if (CAMERA_MODE === 'user') {
        videoTexture.wrapS = THREE.RepeatWrapping; 
        videoTexture.offset.x = 1; 
        videoTexture.repeat.x = -1; 
    } else {
        videoTexture.wrapS = THREE.ClampToEdgeWrapping; 
        videoTexture.offset.x = 0; 
        videoTexture.repeat.x = 1; 
    }
    
    const planeGeometry = new THREE.PlaneGeometry(1, 1);
    // Flip Y via geometry to fix mobile orientation (more reliable than texture flip)
    planeGeometry.scale(1, -1, 1); 
    const planeMaterial = new THREE.MeshBasicMaterial({ map: videoTexture, side: THREE.DoubleSide, depthTest: false });
    videoBackgroundPlane = new THREE.Mesh(planeGeometry, planeMaterial);
    const viewAspect = containerRect.width / containerRect.height;
    const fovRad = THREE.MathUtils.degToRad(camera.fov);
    const planeHeight = Math.abs(2 * Math.tan(fovRad / 2) * VIDEO_PLANE_DEPTH);
    const planeWidth = planeHeight * viewAspect;
    videoBackgroundPlane.scale.set(planeWidth, planeHeight, 1);
    videoBackgroundPlane.position.z = VIDEO_PLANE_DEPTH;
    videoBackgroundPlane.renderOrder = 0; 
    scene.add(videoBackgroundPlane);


    // *** DUAL ASSET LOADING LOGIC ***
    if (!isSplatAttempted) {
        if (USE_GAUSSIAN_SPLAT && typeof SplatMesh !== 'undefined') {
            debugLogger.updateAssetStatus(`Checking ${SPLAT_PATH_LEFT_WING} and ${SPLAT_PATH_RIGHT_WING}...`);
            
            Promise.all([
                fetch(SPLAT_PATH_LEFT_WING).then(r => { if (!r.ok) throw new Error(`Left asset failed: ${r.status}`); return r; }),
                fetch(SPLAT_PATH_RIGHT_WING).then(r => { if (!r.ok) throw new Error(`Right asset failed: ${r.status}`); return r; })
            ])
            .then(() => loadSplatModels())
            .catch(err => {
                debugLogger.log('error', `FATAL Asset Load Error: ${err.message}. Falling back to boxes.`);
                createBoxWings();
            });
            isSplatAttempted = true; 
        } else {
            createBoxWings();
        }
    } else {
        if (wingsGroup && !scene.children.includes(wingsGroup)) scene.add(wingsGroup);
    }

    // Initialize depth sensing after renderer is ready
    tryInitDepth();
}
// === END SETUP THREE.JS ===

// === DEPTH SENSING INITIALIZATION ===
async function tryInitDepth() {
    try {
        if (await isDepthSupported()) {
            const gl = threeRendererInstance.getContext();
            if (gl) {
                const success = await initDepthSensing(gl);
                if (success) {
                    depthEnabled = true;
                    debugLogger.log('success', '[Depth] Depth sensing initialized (LiDAR / WebXR Depth).');
                    // Create depth occlusion plane if depth is enabled
                    createDepthOcclusionPlane();
                } else {
                    debugLogger.log('warning', '[Depth] Depth sensing initialization failed.');
                    depthEnabled = false;
                }
            } else {
                debugLogger.log('warning', '[Depth] No WebGL context available for depth sensing.');
                depthEnabled = false;
            }
        } else {
            debugLogger.log('info', '[Depth] No depth support detected; falling back to segmentation only.');
            depthEnabled = false;
        }
    } catch (error) {
        // Catch any errors in depth initialization - don't break the app
        console.error('[Depth] Error during depth initialization:', error);
        debugLogger.log('warning', '[Depth] Depth initialization error - using segmentation fallback.');
        depthEnabled = false;
    }
}

// === DEPTH OCCLUSION PLANE CREATION ===
function createDepthOcclusionPlane() {
    // Only create if depth is actually enabled
    if (!depthEnabled) {
        console.warn('[Depth] Cannot create depth occlusion plane: depth not enabled');
        return;
    }
    
    if (!videoBackgroundPlane || !videoTexture) {
        console.warn('[Depth] Cannot create depth occlusion plane: video plane not ready');
        return;
    }

    // Use same geometry as the video plane
    const geometry = videoBackgroundPlane.geometry.clone();

    const uniforms = {
        uDepthTexture: { value: null },
        uVideoTexture: { value: videoTexture },
        uNear: { value: 0.0 },   // to be filled from depth params
        uFar: { value: 5.0 },    // example max distance in meters
        uWingsDepth: { value: -5.0 },  // approximate z of wings in camera space
        uDepthScale: { value: 1.0 },   // scale/normalize as needed
        uUseBodyMask: { value: false },  // optional: blending with BodyPix
        uDepthEnabled: { value: 0.0 },  // 1.0 if depth texture is valid, 0.0 otherwise
    };

    const material = new THREE.ShaderMaterial({
        uniforms,
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            varying vec2 vUv;
            uniform sampler2D uDepthTexture;
            uniform sampler2D uVideoTexture;
            uniform float uNear;
            uniform float uFar;
            uniform float uWingsDepth;
            uniform float uDepthScale;
            uniform bool uUseBodyMask;
            uniform float uDepthEnabled;

            // Helper: convert raw depth to linear distance (depends on API)
            float depthToMeters(float rawDepth) {
                // Adjust this function to your depth API:
                // e.g., if rawDepth is already [0,1] mapped to [near, far]
                float depth = rawDepth * uDepthScale;
                return mix(uNear, uFar, depth);
            }

            void main() {
                // If depth is not enabled or texture is not available, don't occlude
                if (uDepthEnabled < 0.5) discard;

                float rawDepth = texture2D(uDepthTexture, vUv).r;
                // Check for invalid depth values (0 or near-zero typically means no depth)
                if (rawDepth < 0.001) discard;
                
                float sceneDepth = depthToMeters(rawDepth);

                // Heuristic: if real-world depth is closer than the wings depth,
                // treat that pixel as "real world in front", so we occlude wings.
                // uWingsDepth should be in meters relative to camera; adjust as needed.
                // Note: uWingsDepth is negative (behind camera), so we compare absolute values
                bool isRealWorldInFront = (sceneDepth > 0.0 && sceneDepth < abs(uWingsDepth));

                if (!isRealWorldInFront) {
                    // Real-world is farther than wings or invalid → no occlusion
                    discard;
                }

                // Real-world in front: draw the video pixel to cover wings
                vec4 videoColor = texture2D(uVideoTexture, vUv);
                gl_FragColor = videoColor;
            }
        `,
        transparent: true,
        depthTest: true,
        depthWrite: false,
    });

    depthOcclusionMesh = new THREE.Mesh(geometry, material);
    depthOcclusionMesh.position.copy(videoBackgroundPlane.position);

    // Ensure it sits just in front of the wings in render order and depth
    videoBackgroundPlane.renderOrder = 0;
    if (wingsGroup) {
        wingsGroup.renderOrder = 1;
    }
    depthOcclusionMesh.renderOrder = 2;

    // Position it slightly closer to the camera than wings in Z, if using depthTest
    // Get wings depth from wingsController or use default
    const wingsZ = wingsGroup ? wingsGroup.position.z : TEST_DEPTH_Z;
    depthOcclusionMesh.position.z = wingsZ + 0.01;

    // Initially hidden - will be shown when depth is available
    depthOcclusionMesh.visible = false;
    
    // Add to scene but keep it hidden until depth data is available
    if (scene) {
        scene.add(depthOcclusionMesh);
        debugLogger.log('info', '[Depth] Depth occlusion plane created (will be activated when depth data is available)');
    }
}

// === UPDATE DEPTH OCCLUSION ===
function updateDepthOcclusion() {
    // Only update if depth is enabled and mesh exists
    if (!depthEnabled || !depthOcclusionMesh) {
        return;
    }

    const depthTexture = getCurrentDepthTexture();
    if (!depthTexture) {
        // Hide occlusion plane if no depth data and disable in shader
        if (depthOcclusionMesh.material) {
            depthOcclusionMesh.material.uniforms.uDepthEnabled.value = 0.0;
        }
        depthOcclusionMesh.visible = false;
        return;
    }

    const params = getDepthParams() || {};
    const mat = depthOcclusionMesh.material;

    mat.uniforms.uDepthTexture.value = depthTexture;
    mat.uniforms.uDepthEnabled.value = 1.0; // Enable depth in shader
    if (typeof params.near === 'number') mat.uniforms.uNear.value = params.near;
    if (typeof params.far === 'number') mat.uniforms.uFar.value = params.far;
    if (typeof params.depthScale === 'number') mat.uniforms.uDepthScale.value = params.depthScale;

    // Update uWingsDepth based on current wings position
    if (wingsGroup) {
        const wingsZ = wingsGroup.position.z;
        mat.uniforms.uWingsDepth.value = wingsZ;
        
        // Update occlusion plane position to match wings
        depthOcclusionMesh.position.z = wingsZ + 0.01;
    }

    // Add to scene and show occlusion plane when depth is actually available
    if (!scene.children.includes(depthOcclusionMesh)) {
        scene.add(depthOcclusionMesh);
    }
    depthOcclusionMesh.visible = true;
}
// === END DEPTH OCCLUSION ===

// --- ASSET LOADING AND FALLBACK (UNCHANGED) ---

function loadSplatModels() {
    if (wingsAssetLeft) wingsGroup.remove(wingsAssetLeft);
    if (wingsAssetRight) wingsGroup.remove(wingsAssetRight);
    
    debugLogger.updateAssetStatus('Loading Gaussian Splats...');

    try {
        wingsAssetLeft = new SplatMesh({ 
            url: SPLAT_PATH_LEFT_WING, 
            fileType: 'ksplat', 
            onLoad: (mesh) => {
                mesh.scale.set(1, 1, -1); 
                checkSplatDataReady();
            }
        });
        wingsAssetLeft.visible = false;
        wingsAssetLeft.renderOrder = 1; 
        wingsGroup.add(wingsAssetLeft);
        
        wingsAssetRight = new SplatMesh({ 
            url: SPLAT_PATH_RIGHT_WING, 
            fileType: 'ksplat', 
            onLoad: (mesh) => {
                mesh.scale.set(1, 1, -1); 
                checkSplatDataReady();
            }
        });
        wingsAssetRight.visible = false;
        wingsAssetRight.renderOrder = 1; 
        wingsGroup.add(wingsAssetRight);
        
    } catch (err) {
        debugLogger.log('error', `Splat instantiation error: ${err.message}. Falling back to boxes.`);
        createBoxWings();
    }
}

let loadedCount = 0;
function checkSplatDataReady() {
    loadedCount++;
    if (loadedCount === 2) {
        isSplatDataReady = true; 
        debugLogger.log('success', 'Gaussian Splat data loaded and ready!');
        debugLogger.updateAssetStatus('Gaussian Splats active');
        
        // Initialize WingsController once wings are loaded
        if (wingsGroup && wingsAssetLeft && wingsAssetRight && !wingsController) {
            wingsController = new WingsController(wingsGroup, wingsAssetLeft, wingsAssetRight, {
                baseScale: currentWingScale,
                offsetY: -WING_VERTICAL_SHIFT,
                depthWhenFacingCamera: -5.5,
                depthWhenBackToCamera: -4.5,
            });
            debugLogger.log('success', 'WingsController initialized');
        }
        
        loadedCount = 0; 
    }
}

function createBoxWings() {
    if (wingsAssetLeft) wingsGroup.remove(wingsAssetLeft);
    if (wingsAssetRight) wingsGroup.remove(wingsAssetRight);

    const wingGeometry = new THREE.BoxGeometry(0.5, 0.8, 0.08); 
    const wingMaterial = new THREE.MeshBasicMaterial({ color: 0x00ccff, transparent: true, opacity: 0.8 });

    wingsAssetLeft = new THREE.Mesh(wingGeometry, wingMaterial);
    wingsAssetRight = new THREE.Mesh(wingGeometry.clone(), wingMaterial.clone());

    wingsGroup.add(wingsAssetLeft);
    wingsGroup.add(wingsAssetRight);

    wingsAssetLeft.visible = false;
    wingsAssetRight.visible = false;
    isSplatAttempted = false;
    isSplatDataReady = true; 
    
    // Initialize WingsController for box wings too
    if (wingsGroup && wingsAssetLeft && wingsAssetRight && !wingsController) {
        wingsController = new WingsController(wingsGroup, wingsAssetLeft, wingsAssetRight, {
            baseScale: 1.2,
            offsetY: -WING_VERTICAL_SHIFT,
            depthWhenFacingCamera: -5.5,
            depthWhenBackToCamera: -4.5,
        });
    }
    
    debugLogger.updateAssetStatus('Box placeholder active (Fallback)');
}


// === MAIN RENDER LOOP (OPTIMIZED AND CORRECTED) ===
async function renderLoop() {
    if (!isRunning) return;

    requestAnimationFrame(renderLoop);

    // FPS Counter and Performance Monitoring
    frameCount++;
    const now = Date.now();
    if (now - lastFpsUpdate >= 1000) {
        const fps = frameCount / ((now - lastFpsUpdate) / 1000);
        debugLogger.updateFPS(fps, segmentationInferenceTime);
        frameCount = 0;
        lastFpsUpdate = now;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let currentPoseKeypoints = lastGoodPoseKeypoints; // Start with the last known data
    let currentSegmentationResult = null;
    let currentOrientation = { isFacingCamera: false, isBackToCamera: false, confidence: 0 };

    // --- 1. THROTTLED POSE DETECTION ---
    if (video.readyState >= video.HAVE_ENOUGH_DATA && poseModel) {
        poseDetectionFrameCounter++;

        // Run the expensive AI operation only every N frames
        if (poseDetectionFrameCounter >= POSE_DETECTION_SKIP_FRAMES) {
            poseDetectionFrameCounter = 0; // Reset counter
            try {
                const newPoses = await poseModel.estimatePoses(video);

                if (newPoses.length > 0) {
                    const potentialKeypoints = newPoses[0].keypoints;
                    const leftShoulder = potentialKeypoints.find(kp => kp.name === 'left_shoulder');
                    const rightShoulder = potentialKeypoints.find(kp => kp.name === 'right_shoulder');
                    
                    // Only update and store if detection is high confidence
                    if (leftShoulder && rightShoulder && leftShoulder.score > 0.4 && rightShoulder.score > 0.4) {
                        lastGoodPoseKeypoints = potentialKeypoints; // 👈 Store the good data
                        currentPoseKeypoints = potentialKeypoints; // Use the fresh data for this frame
                        debugLogger.updatePoseStatus(`Detected (L:${leftShoulder.score.toFixed(2)}, R:${rightShoulder.score.toFixed(2)})`);
                    } else {
                        // Confidence is low, clear the stored pose
                        lastGoodPoseKeypoints = null;
                        currentPoseKeypoints = null;
                        debugLogger.updatePoseStatus('Low confidence / Person too far');
                    }
                } else {
                    // No person detected in this AI run
                    lastGoodPoseKeypoints = null;
                    currentPoseKeypoints = null;
                    debugLogger.updatePoseStatus('No person detected');
                }
            } catch (err) {
                debugLogger.log('error', `Pose detection error: ${err.message}`);
                // On error, let the wings fade out
                lastGoodPoseKeypoints = null;
                currentPoseKeypoints = null;
            }
        }
    }
    // --- END THROTTLED POSE DETECTION ---

    // --- 2. THROTTLED SEGMENTATION (MediaPipe - runs every frame or every 2 frames) ---
    // currentSegmentationResult is already declared above
    // Reduce segmentation frequency when depth is enabled (depth handles occlusion)
    const effectiveSegmentationSkipFrames = depthEnabled && isDepthAvailable() 
        ? Math.max(segmentationSkipFrames * 2, 6) // Double skip frames when depth works
        : segmentationSkipFrames;
    
    if (video.readyState >= video.HAVE_ENOUGH_DATA && segmentationReady && isPersonSegmentationReady()) {
        segmentationFrameCounter++;
        
        if (segmentationFrameCounter >= effectiveSegmentationSkipFrames) {
            segmentationFrameCounter = 0;
            lastSegmentationInferenceStart = performance.now();
            
            try {
                const segResult = await runPersonSegmentationFrame(video);
                if (segResult && segResult.maskTexture) {
                    currentSegmentationResult = segResult;
                    segmentationMaskTexture = segResult.maskTexture;
                    segmentationInferenceTime = performance.now() - lastSegmentationInferenceStart;
                }
            } catch (err) {
                if (DEBUG_MODE) debugLogger.log('error', `Segmentation error: ${err.message}`);
            }
        } else {
            // Use last segmentation result (reuse mask texture)
            if (segmentationMaskTexture) {
                currentSegmentationResult = { maskTexture: segmentationMaskTexture };
            }
        }
    } else if (currentPoseKeypoints && currentPoseKeypoints.length > 0) {
        // Fallback: generate mask from pose keypoints
        try {
            const poseMask = generateMaskFromPose(currentPoseKeypoints, video.videoWidth, video.videoHeight);
            if (poseMask) {
                currentSegmentationResult = poseMask;
                segmentationMaskTexture = poseMask.maskTexture;
            }
        } catch (err) {
            // Ignore errors in fallback
        }
    }

    // --- 3. ORIENTATION ESTIMATION (improved with better face detection) ---
    // currentOrientation is already declared above, just reset it
    currentOrientation = { isFacingCamera: false, isBackToCamera: false, confidence: 0 };
    
    if (currentPoseKeypoints && currentPoseKeypoints.length > 0) {
        // Get orientation from pose keypoints (improved detection)
        const poseOrientation = estimateOrientationFromPose(currentPoseKeypoints);
        currentOrientation = poseOrientation;
        
        // Add to history for smoothing
        orientationHistory.push(currentOrientation);
        if (orientationHistory.length > MAX_ORIENTATION_HISTORY) {
            orientationHistory.shift();
        }
        
        // Smooth orientation to avoid jitter
        const smoothedOrientation = smoothOrientation(orientationHistory, 3);
        currentOrientation = smoothedOrientation;
    }

    // --- 4. WING POSITIONING USING WINGSCONTROLLER ---
    // Force wings visible for debugging if flag is set
    if (DEBUG_FORCE_WINGS_VISIBLE && isSplatDataReady && wingsGroup) {
        wingsGroup.visible = true;
        if (wingsAssetLeft) wingsAssetLeft.visible = true;
        if (wingsAssetRight) wingsAssetRight.visible = true;
        if (scene && !scene.children.includes(wingsGroup)) {
            scene.add(wingsGroup);
        }
    } else if (currentPoseKeypoints && isSplatDataReady && wingsController) {
        // Update wings controller with pose and orientation
        wingsController.update({
            keypoints: currentPoseKeypoints,
            orientation: currentOrientation,
            segmentationMask: currentSegmentationResult ? currentSegmentationResult.mask : null,
            videoWidth: video.videoWidth,
            videoHeight: video.videoHeight,
            cameraMode: CAMERA_MODE,
        });

        // Draw debug points only if debug mode is on
        if (DEBUG_MODE) {
            const leftShoulder = currentPoseKeypoints.find(kp => kp.name === 'left_shoulder');
            const rightShoulder = currentPoseKeypoints.find(kp => kp.name === 'right_shoulder');
            if (leftShoulder && rightShoulder) {
                drawDebugPoints(ctx, [leftShoulder, rightShoulder]);
            }
        }
        
        const wingOrientation = wingsController.getOrientation();
        if (DEBUG_MODE) {
            debugLogger.updatePositionStatus(
                wingsAssetLeft.position, 
                wingsAssetLeft.rotation, 
                wingsAssetRight.position, 
                wingsAssetRight.rotation
            );
        }
    } else if (wingsController && !DEBUG_FORCE_WINGS_VISIBLE) {
        // Hide wings if no pose detected (unless debug force is enabled)
        wingsController.setWingsVisible(false);
    }

    // --- 5. RENDER WITH OCCLUSION COMPOSITING ---
    if (threeRendererInstance) {
        if (videoBackgroundPlane && videoBackgroundPlane.material.map) {
            videoBackgroundPlane.material.map.needsUpdate = true;
        }

        // Ensure wings group is visible based on debug flags or pose detection
        const shouldShowWings = DEBUG_FORCE_WINGS_VISIBLE || 
                                (wingsGroup && currentPoseKeypoints && currentPoseKeypoints.length > 0);
        
        if (shouldShowWings && wingsGroup) {
            wingsGroup.visible = true;
            // Also ensure individual wings are visible
            if (wingsAssetLeft) {
                wingsAssetLeft.visible = true;
                // Ensure wing is in the scene
                if (wingsGroup && !wingsGroup.children.includes(wingsAssetLeft)) {
                    wingsGroup.add(wingsAssetLeft);
                }
            }
            if (wingsAssetRight) {
                wingsAssetRight.visible = true;
                // Ensure wing is in the scene
                if (wingsGroup && !wingsGroup.children.includes(wingsAssetRight)) {
                    wingsGroup.add(wingsAssetRight);
                }
            }
            // Ensure wingsGroup is in the scene
            if (scene && !scene.children.includes(wingsGroup)) {
                scene.add(wingsGroup);
            }
        }

        // --- Update depth occlusion before rendering ---
        updateDepthOcclusion();

        // --- Determine occlusion method: prefer depth, fall back to segmentation ---
        // Only use depth occlusion if depth is enabled, mesh exists, is visible, AND depth data is actually available
        let useDepthOcclusion = false;
        try {
            useDepthOcclusion = depthEnabled && depthOcclusionMesh && 
                              depthOcclusionMesh.visible && 
                              isDepthAvailable();
        } catch (error) {
            // If depth check fails, fall back to segmentation
            console.warn('[Depth] Error checking depth availability:', error);
            useDepthOcclusion = false;
        }
        
        // Ensure depth occlusion mesh doesn't interfere when not in use
        if (depthOcclusionMesh && !useDepthOcclusion) {
            depthOcclusionMesh.visible = false;
            if (depthOcclusionMesh.material) {
                depthOcclusionMesh.material.uniforms.uDepthEnabled.value = 0.0;
            }
        }
        
        // Render with appropriate occlusion method
        if (!DEBUG_DISABLE_OCCLUSION) {
            if (useDepthOcclusion) {
                // Use depth-based occlusion (depth occlusion mesh handles it in the scene)
                threeRendererInstance.render(scene, camera);
            } else if (occlusionCompositor && currentSegmentationResult && 
                      currentSegmentationResult.maskTexture && videoTexture && wingsGroup) {
                // Fall back to segmentation-based occlusion
                occlusionCompositor.render(scene, camera, currentSegmentationResult.maskTexture, 
                    videoTexture, wingsGroup, videoBackgroundPlane);
            } else {
                // No occlusion - render normally (wings should be visible)
                threeRendererInstance.render(scene, camera);
            }
        } else {
            // Occlusion disabled for debugging - render normally
            threeRendererInstance.render(scene, camera);
        }
    }

    // --- 6. ADAPTIVE PERFORMANCE TUNING ---
    const fps = frameCount / ((Date.now() - lastFpsUpdate) / 1000);
    if (fps < LOW_FPS_THRESHOLD && fps > 0) {
        lowFpsCounter += 16; // Approximate frame time (60fps = 16ms)
        if (lowFpsCounter > LOW_FPS_DURATION) {
            // Increase skip frames to reduce load
            if (POSE_DETECTION_SKIP_FRAMES < MAX_POSE_SKIP_FRAMES) {
                POSE_DETECTION_SKIP_FRAMES += 1;
                if (DEBUG_MODE) debugLogger.log('warning', `Performance: Increased pose skip to ${POSE_DETECTION_SKIP_FRAMES}`);
            }
            if (segmentationSkipFrames < 5) {
                segmentationSkipFrames += 1; // Reduce segmentation frequency
                if (DEBUG_MODE) debugLogger.log('warning', `Performance: Increased segmentation skip to ${segmentationSkipFrames}`);
            }
            lowFpsCounter = 0;
        }
    } else if (fps > 40 && (POSE_DETECTION_SKIP_FRAMES > 4 || segmentationSkipFrames > 2)) {
        // If FPS is good, gradually reduce skip frames for better quality
        lowFpsCounter = Math.max(0, lowFpsCounter - 8);
        if (lowFpsCounter === 0 && (POSE_DETECTION_SKIP_FRAMES > 4 || segmentationSkipFrames > 2)) {
            if (POSE_DETECTION_SKIP_FRAMES > 4) {
                POSE_DETECTION_SKIP_FRAMES -= 1;
                if (DEBUG_MODE) debugLogger.log('info', `Performance: Reduced pose skip to ${POSE_DETECTION_SKIP_FRAMES}`);
            }
            if (segmentationSkipFrames > 2) {
                segmentationSkipFrames -= 1;
                if (DEBUG_MODE) debugLogger.log('info', `Performance: Reduced segmentation skip to ${segmentationSkipFrames}`);
            }
        }
    } else {
        lowFpsCounter = Math.max(0, lowFpsCounter - 16); // Gradually reset
        if (fps > 40 && segmentationSkipFrames > 1) {
            segmentationSkipFrames = 1; // Can increase frequency if performance is good
        }
    }
}
// === END MAIN RENDER LOOP ===

// === GROUP POSITIONING FUNCTION (UNCHANGED) ===
function positionWingsGroup(group, avgKeypointX, avgKeypointY) {
    const depth = TEST_DEPTH_Z; 
    
    const normX = (coord, dim) => (coord / dim) * 2 - 1;
    const normY = (coord, dim) => -(coord / dim) * 2 + 1; 

    let targetX = normX(avgKeypointX, video.videoWidth);
    let targetY = normY(avgKeypointY, video.videoHeight);
    let targetZ = depth; 

    if (CAMERA_MODE === 'user') {
        targetX = -targetX; 
    }
    
    targetY -= (WING_VERTICAL_SHIFT * 1.0); 
    targetZ += BACK_OFFSET_Z; 

    // Apply Smoothing and set Position for the GROUP
    smoothedGroupPosition.x = smoothedGroupPosition.x + (targetX - smoothedGroupPosition.x) * SMOOTHING_FACTOR;
    smoothedGroupPosition.y = smoothedGroupPosition.y + (targetY - smoothedGroupPosition.y) * SMOOTHING_FACTOR;
    smoothedGroupPosition.z = smoothedGroupPosition.z + (targetZ - smoothedGroupPosition.z) * SMOOTHING_FACTOR;
    
    group.position.set(smoothedGroupPosition.x, smoothedGroupPosition.y, smoothedGroupPosition.z);
}

// === INDIVIDUAL WING POSITIONING FUNCTION (UNCHANGED) ===
function positionIndividualWing(wing, side) {
    
    const FIXED_SCALE = 1.0; 
    
    if (side === 'left') {
        wing.position.set(currentHorizontalOffset * FIXED_SCALE, 0, 0); 
    } else if (side === 'right') {
        wing.position.set(-currentHorizontalOffset * FIXED_SCALE, 0, 0); 
    }
    
    let finalScaleFactor = wing instanceof SplatMesh ? currentWingScale : 1.2; 
    wing.scale.set(finalScaleFactor, finalScaleFactor, finalScaleFactor * 1.5); 

    const baseRotX = -Math.PI * 0.2; 
    const baseRotY = Math.PI; 
    let targetRotZ = 0;
    
    if (side === 'left') {
        targetRotZ = Math.PI + SPLAY_ANGLE;
    } else if (side === 'right') {
        targetRotZ = -Math.PI - SPLAY_ANGLE;
    }
    
    wing.rotation.set(baseRotX, baseRotY, targetRotZ);
}

// Draw Debug Points (only in debug mode)
function drawDebugPoints(ctx, keypoints) {
    if (!DEBUG_MODE) return; // Skip if debug mode is off
    
    ctx.fillStyle = '#00ff88'; 
    keypoints.forEach(kp => {
        if (kp.score > 0.4) {
            let x = kp.x;
            const y = kp.y;
            
            if (CAMERA_MODE === 'user') {
                x = canvas.width - x;
            }
            
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();
        }
    });
}

// === START WHEN PAGE LOADS (UNCHANGED) ===
window.addEventListener('DOMContentLoaded', () => {
    init();
});
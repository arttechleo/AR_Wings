// --- MODULE IMPORTS ---
import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection'; 
// Ensure @sparkjsdev/spark is installed and accessible via import maps or bundling
import { SplatMesh, SparkRenderer } from "@sparkjsdev/spark"; 

// Global variables for the scene and pose detection
let scene, camera;
let threeRendererInstance; 
let wingsAssetLeft, wingsAssetRight; 
let wingsGroup; // Group for centering and rotating both wings together
let video, canvas, ctx;
let poseModel;
let debugLogger;
let isRunning = false;
let frameCount = 0;
let lastFpsUpdate = Date.now();
let videoBackgroundPlane; 

// --- STATE FLAGS ---
let isSplatAttempted = false;
let isSplatDataReady = false; 

// Smoothing variable for stable Group positioning
let smoothedGroupPosition = { x: 0, y: 0, z: 0 }; 
const SMOOTHING_FACTOR = 0.6; 

// Gaussian Splatting configuration
const USE_GAUSSIAN_SPLAT = true; 

// *** ASSET PATHS (Ensure these files exist in your 'assets' folder) ***
const SPLAT_PATH_LEFT_WING = new URL('./assets/leftwing.ksplat', import.meta.url).href;
const SPLAT_PATH_RIGHT_WING = new URL('./assets/rightwing.ksplat', import.meta.url).href;

// --- CRITICAL WING CONSTANTS ---
// BASE SCALE: Adjusted from 3.0 to a smaller value for mobile screens
const WING_SPLAT_SCALE_FACTOR_BASE = 1.8; 
// Dynamic scale applied during runtime
let currentWingScale = WING_SPLAT_SCALE_FACTOR_BASE;

// WING_VERTICAL_SHIFT determines how far DOWN (positive value) the wings pivot point is
// from the detected shoulder mid-point.
const WING_VERTICAL_SHIFT = 0.5; 
// Minimal shift outward from the shoulder dot (determines shoulder-to-wing-root distance)
const WING_HORIZONTAL_OFFSET = 3.75;

// ROTATION CONSTANTS
const MAX_X_ROTATION = Math.PI / 6; // Limit wing rotation to 30 degrees up/down
const Y_DIFFERENCE_SENSITIVITY = 150; // Pixel difference in shoulder height to achieve max rotation

let CAMERA_MODE = 'environment'; // Starts with rear camera

// --- AR SETTINGS (FIXED VALUES) ---
const TEST_DEPTH_Z = -5.0; // The Z-depth of the detection plane itself.
const BACK_OFFSET_Z = -5.0; // Pushed far back behind the user
const VIDEO_PLANE_DEPTH = -10.0; // Pushed behind all AR content
// Angle to slightly splay the wings (in radians)
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
    updateFPS(fps) { if(this.fpsCounter) this.fpsCounter.textContent = fps.toFixed(1); }
    updatePositionStatus(posL, rotL, posR, rotR) {
        if (this.positionStatus) {
            this.positionStatus.textContent = `L P: (${posL.x.toFixed(2)}, ${posL.y.toFixed(2)}) R P: (${posR.x.toFixed(2)}, ${posR.y.toFixed(2)}) Z: ${posL.z.toFixed(2)}`;
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
    
    isRunning = false; // Halt the render loop temporarily
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
    
    // Clear smoothed group position
    smoothedGroupPosition = { x: 0, y: 0, z: 0 };

    await startAR();
}

// === NEW: RESPONSIVE WING SCALE ADJUSTMENT ===
function calculateResponsiveWingScale(videoWidth, videoHeight, baseScale) {
    // Determine scale based on the visible size of the person (relative to screen).
    // A full-screen person (tall) should have a smaller scale relative to their height.
    const aspect = videoWidth / videoHeight;
    let scaleAdjustment = 1.0;
    
    // Adjust scale based on aspect ratio (e.g., taller screens need slightly smaller assets)
    if (aspect < 1.0) { // Portrait mode (mobile)
        scaleAdjustment = 0.85; 
    } else if (aspect > 1.7) { // Very wide screen (desktop/landscape)
        scaleAdjustment = 1.1; 
    }
    
    // Further scale down if the actual window height is very small compared to a reference (e.g. debugging)
    // This makes the scale generally appropriate for the display size.
    const screenHeightFactor = window.innerHeight / 800; // Use 800px as a desktop reference
    
    return baseScale * scaleAdjustment * Math.min(1.0, screenHeightFactor);
}

// --- INITIALIZE & START AR (MODIFIED) ---

function init() {
    debugLogger = new DebugLogger();
    debugLogger.log('info', '=== AR Back Wings Starting ===');
    
    if (typeof THREE === 'undefined' || typeof tf === 'undefined' || typeof poseDetection === 'undefined' || typeof SplatMesh === 'undefined') {
        debugLogger.log('error', 'Module imports failed. Check console for module errors.');
        document.getElementById('instructions').innerHTML = `
            <h2>Initialization Failed!</h2>
            <p>Error: Required libraries failed to load. Check console for module errors.</p>
        `;
        return;
    }
    debugLogger.log('success', 'Core libraries loaded (THREE, TF, Spark.js)');

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
        debugLogger.updateStatus('Initializing TensorFlow...');
        
        if (poseModel === undefined) { 
            tf.setBackend('webgl'); 
            await tf.ready(); 
            debugLogger.log('success', `TensorFlow backend ready (${tf.getBackend()}).`);
        }
        
        const threeContainer = document.getElementById('three-container');
        canvas = document.getElementById('output-canvas');
        ctx = canvas.getContext('2d');
        video = document.getElementById('video');

        // 1. Request Camera Stream using the current CAMERA_MODE
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: CAMERA_MODE, width: { ideal: 1280 }, height: { ideal: 720 } }
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

        // Remove old renderer and dispose of resources on camera switch
        if (threeRendererInstance) {
            threeContainer.removeChild(threeRendererInstance.domElement);
            threeRendererInstance.dispose();
            threeRendererInstance = null;
        }

        debugLogger.updateStatus('Setting up 3D renderer...');
        setupThreeJS(vw, vh); 
        debugLogger.log('success', '3D renderer ready');
        
        // --- MOBILE SCALING IMPLEMENTATION ---
        currentWingScale = calculateResponsiveWingScale(vw, vh, WING_SPLAT_SCALE_FACTOR_BASE);
        debugLogger.log('info', `Set initial wing scale to: ${currentWingScale.toFixed(2)}`);
        // --- END SCALING ---

        // Only load AI model on initial load
        if (poseModel === undefined) {
            debugLogger.updateStatus('Loading AI model...');
            poseModel = await poseDetection.createDetector(
                poseDetection.SupportedModels.MoveNet,
                { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
            );
            debugLogger.log('success', 'AI model loaded!');
        }
        
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

// === SETUP THREE.JS (VIDEO PLANE LOGIC INCLUDED) ===
function setupThreeJS(videoWidth, videoHeight) {
    const threeContainer = document.getElementById('three-container');
    const containerRect = threeContainer.getBoundingClientRect();

    const threeRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    threeRenderer.setPixelRatio(window.devicePixelRatio);
    threeRenderer.setSize(containerRect.width, containerRect.height);
    threeRenderer.setClearColor(0x000000, 0); 
    threeContainer.appendChild(threeRenderer.domElement);

    threeRendererInstance = threeRenderer;
    
    new SparkRenderer(threeRenderer);

    if (scene) {
        if (videoBackgroundPlane) scene.remove(videoBackgroundPlane);
        if (wingsGroup) scene.remove(wingsGroup);
    } else {
        scene = new THREE.Scene();
    }
    
    // CREATE NEW GROUP for the wings
    wingsGroup = new THREE.Group();
    scene.add(wingsGroup); 
    
    const aspect = containerRect.width / containerRect.height;
    camera = new THREE.PerspectiveCamera(65, aspect, 0.1, 100); 
    camera.position.set(0, 0, 0); 
    scene.add(new THREE.AmbientLight(0xffffff, 1.0));

    // Video Background Plane setup
    const videoTexture = new THREE.VideoTexture(video);
    videoTexture.flipY = false; 
    if (CAMERA_MODE === 'user') {
        // Apply mirroring for selfie camera
        videoTexture.wrapS = THREE.RepeatWrapping; videoTexture.offset.x = 1; videoTexture.repeat.x = -1; 
    } else {
        videoTexture.wrapS = THREE.ClampToEdgeWrapping; videoTexture.offset.x = 0; videoTexture.repeat.x = 1; 
    }
    const planeGeometry = new THREE.PlaneGeometry(1, 1);
    planeGeometry.scale(1, -1, 1); // Flip Y to match standard video conventions
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
            
            // Check both assets via fetch before loading
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
        // RE-ADD GROUP IF ALREADY LOADED
        if (wingsGroup && !scene.children.includes(wingsGroup)) scene.add(wingsGroup);
    }
}
// === END SETUP THREE.JS ===

// --- ASSET LOADING AND FALLBACK (MODIFIED) ---

function loadSplatModels() {
    // Clean up if re-loading
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
        loadedCount = 0; // Reset for potential re-load
    }
}

function createBoxWings() {
    // Clean up if re-loading
    if (wingsAssetLeft) wingsGroup.remove(wingsAssetLeft);
    if (wingsAssetRight) wingsGroup.remove(wingsAssetRight);

    // Fallback: Create two separate box placeholders
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
    
    debugLogger.updateAssetStatus('Box placeholder active (Fallback)');
}

// === MAIN RENDER LOOP ===
async function renderLoop() {
    if (!isRunning) return;

    requestAnimationFrame(renderLoop);

    // FPS Counter
    frameCount++;
    const now = Date.now();
    if (now - lastFpsUpdate >= 1000) {
        const fps = frameCount / ((now - lastFpsUpdate) / 1000);
        debugLogger.updateFPS(fps);
        frameCount = 0;
        lastFpsUpdate = now;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 2. Pose Detection Logic
    if (video.readyState >= video.HAVE_ENOUGH_DATA && poseModel) {
        try {
            const poses = await poseModel.estimatePoses(video);
            
            if (poses.length > 0) {
                const keypoints = poses[0].keypoints;
                const leftShoulder = keypoints.find(kp => kp.name === 'left_shoulder');
                const rightShoulder = keypoints.find(kp => kp.name === 'right_shoulder');

                if (leftShoulder && rightShoulder && leftShoulder.score > 0.4 && rightShoulder.score > 0.4) {
                    
                    debugLogger.updatePoseStatus(`Detected (L:${leftShoulder.score.toFixed(2)}, R:${rightShoulder.score.toFixed(2)})`);
                    
                    if (wingsAssetLeft && wingsAssetRight && isSplatDataReady) {
                        
                        // 1. CALCULATE GROUP POSITION (Average of shoulders)
                        const avgShoulderX = (leftShoulder.x + rightShoulder.x) / 2;
                        const avgShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
                        
                        positionWingsGroup(wingsGroup, avgShoulderX, avgShoulderY);
                        
                        // 2. CALCULATE GROUP ROTATION (Based on shoulder height difference)
                        const yDiff = leftShoulder.y - rightShoulder.y; 
                        let targetRotX = (yDiff / Y_DIFFERENCE_SENSITIVITY) * MAX_X_ROTATION;
                        targetRotX = THREE.MathUtils.clamp(targetRotX, -MAX_X_ROTATION, MAX_X_ROTATION);
                        
                        if (CAMERA_MODE === 'user') {
                            targetRotX = -targetRotX; // Reverse tilt for selfie camera
                        }
                        
                        // Apply smoothing to the rotation for stability
                        wingsGroup.rotation.x += (targetRotX - wingsGroup.rotation.x) * SMOOTHING_FACTOR;

                        // 3. POSITION INDIVIDUAL WINGS RELATIVE TO GROUP CENTER
                        positionIndividualWing(wingsAssetLeft, 'left');
                        positionIndividualWing(wingsAssetRight, 'right');

                        wingsAssetLeft.visible = true;
                        wingsAssetRight.visible = true;
                        debugLogger.updatePositionStatus(wingsAssetLeft.position, wingsAssetLeft.rotation, wingsAssetRight.position, wingsAssetRight.rotation);
                    } else {
                        if(wingsAssetLeft) wingsAssetLeft.visible = false;
                        if(wingsAssetRight) wingsAssetRight.visible = false;
                    }

                    drawDebugPoints(ctx, [leftShoulder, rightShoulder]); 

                } else {
                    if(wingsAssetLeft) wingsAssetLeft.visible = false;
                    if(wingsAssetRight) wingsAssetRight.visible = false;
                    debugLogger.updatePoseStatus('Low confidence');
                }
            } else {
                if(wingsAssetLeft) wingsAssetLeft.visible = false;
                if(wingsAssetRight) wingsAssetRight.visible = false;
                debugLogger.updatePoseStatus('No person detected');
            }
        } catch (err) {
            debugLogger.log('error', `Pose detection error: ${err.message}`);
        }
    }

    // 3. Render the scene
    if (threeRendererInstance) {
        if (videoBackgroundPlane && videoBackgroundPlane.material.map) {
            videoBackgroundPlane.material.map.needsUpdate = true;
        }
        threeRendererInstance.render(scene, camera);
    }
}
// === END MAIN RENDER LOOP ===

// === GROUP POSITIONING FUNCTION ===

function positionWingsGroup(group, avgKeypointX, avgKeypointY) {
    const depth = TEST_DEPTH_Z; 
    
    // Convert canvas coordinates (0 to width/height) to normalized device coordinates (-1 to 1)
    const normX = (coord, dim) => (coord / dim) * 2 - 1;
    const normY = (coord, dim) => -(coord / dim) * 2 + 1; 

    let targetX = normX(avgKeypointX, video.videoWidth);
    let targetY = normY(avgKeypointY, video.videoHeight);
    let targetZ = depth; 

    // 1. Compensate for video mirroring (front camera)
    if (CAMERA_MODE === 'user') {
        targetX = -targetX; 
    }
    
    // 2. Apply VERTICAL SHIFT to the group (wings pivot point)
    targetY -= (WING_VERTICAL_SHIFT * 1.0); 
    
    // 3. Apply Z Depth Offset (pushes it behind the user)
    targetZ += BACK_OFFSET_Z; 

    // Apply Smoothing and set Position for the GROUP
    smoothedGroupPosition.x = smoothedGroupPosition.x + (targetX - smoothedGroupPosition.x) * SMOOTHING_FACTOR;
    smoothedGroupPosition.y = smoothedGroupPosition.y + (targetY - smoothedGroupPosition.y) * SMOOTHING_FACTOR;
    smoothedGroupPosition.z = smoothedGroupPosition.z + (targetZ - smoothedGroupPosition.z) * SMOOTHING_FACTOR;
    
    group.position.set(smoothedGroupPosition.x, smoothedGroupPosition.y, smoothedGroupPosition.z);
}

// === INDIVIDUAL WING POSITIONING FUNCTION (USES DYNAMIC SCALE) ===
/**
 * Position and Scale a single wing asset relative to the wingsGroup center.
 */
function positionIndividualWing(wing, side) {
    
    // 1. Position RELATIVE to the Group Center (0,0,0 of the group is the average shoulder point)
    const FIXED_SCALE = 1.0; 
    
    // The relative X position is based on the WING_HORIZONTAL_OFFSET
    if (side === 'left') {
        // Left wing moves positive X (right of center)
        wing.position.set(WING_HORIZONTAL_OFFSET * FIXED_SCALE, 0, 0); 
    } else if (side === 'right') {
        // Right wing moves negative X (left of center)
        wing.position.set(-WING_HORIZONTAL_OFFSET * FIXED_SCALE, 0, 0); 
    }
    
    // 2. Apply Dynamic Scale
    let finalScaleFactor = wing instanceof SplatMesh ? currentWingScale : 1.2; 
    wing.scale.set(finalScaleFactor, finalScaleFactor, finalScaleFactor * 1.5); 

    // 3. Apply INDIVIDUAL WING ROTATION (Local Rotations)
    
    // X-axis: STAND THE WINGS UP 
    const baseRotX = -Math.PI * 0.2; 

    // Y-axis: FACE THE GOLDEN SIDE FORWARD (Flips them to face the camera from the back)
    const baseRotY = Math.PI; 
    
    // Z-axis: Pivot and Splay 
    let targetRotZ = 0;
    
    if (side === 'left') {
        targetRotZ = Math.PI + SPLAY_ANGLE;
    } else if (side === 'right') {
        targetRotZ = -Math.PI - SPLAY_ANGLE;
    }
    
    // The individual wing's X-rotation is zeroed out since the group handles the tilt
    wing.rotation.set(baseRotX, baseRotY, targetRotZ);
}

// Draw Debug Points (UNCHANGED)
function drawDebugPoints(ctx, keypoints) {
    
    ctx.fillStyle = '#00ff88';
    keypoints.forEach(kp => {
        if (kp.score > 0.4) {
            let x = kp.x;
            const y = kp.y;
            
            // X-mirroring logic for debug points
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
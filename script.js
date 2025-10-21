// --- MODULE IMPORTS ---
import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection'; 
import { SplatMesh, SparkRenderer } from "@sparkjsdev/spark"; 

// Global variables for the scene and pose detection
let scene, camera;
let threeRendererInstance; 
let wingsAssetLeft, wingsAssetRight; 
// 🔑 NEW GLOBAL GROUP FOR JOINT ROTATION
let wingsGroup; 
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

// Smoothing variables for stable positioning
let smoothedPosLeft = { x: 0, y: 0, z: 0 }; 
let smoothedPosRight = { x: 0, y: 0, z: 0 }; 
const SMOOTHING_FACTOR = 0.6; 

// Gaussian Splatting configuration
const USE_GAUSSIAN_SPLAT = true; 

// *** ASSET PATHS (Ensure these files exist in your 'assets' folder) ***
const SPLAT_PATH_LEFT_WING = new URL('./assets/leftwing.ksplat', import.meta.url).href;
const SPLAT_PATH_RIGHT_WING = new URL('./assets/rightwing.ksplat', import.meta.url).href;

// *** ADJUSTED CONSTANTS FOR SHOULDER BINDING ***
// CRITICAL: Increased negative value to push the wing's attachment point up onto the dot.
const WING_VERTICAL_SHIFT = 0.5; // ⬅️ ADJUSTED: Significantly pushes the wing's center DOWN
// Minimal shift outward from the shoulder dot
const WING_HORIZONTAL_OFFSET = 3.75;

// 🔑 NEW ROTATION CONSTANTS
const MAX_X_ROTATION = Math.PI / 6; // Limit wing rotation to 30 degrees up/down
const Y_DIFFERENCE_SENSITIVITY = 150; // Pixel difference in shoulder height to achieve max rotation

let CAMERA_MODE = 'user'; // Starts with front/selfie camera

// --- AR SETTINGS (FIXED VALUES) ---
// This is the Z-depth of the detection plane itself.
const TEST_DEPTH_Z = -5.0; 
// *** CRITICAL BACK DEPTH: Pushed far back ***
const BACK_OFFSET_Z = -5.0; 
// Scale factor for the Gaussian Splat 
const WING_SPLAT_SCALE_FACTOR = 3.0; 
// Angle to slightly splay the wings (in radians)
const SPLAY_ANGLE = Math.PI / 12; 
const VIDEO_PLANE_DEPTH = -10.0; 

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
    
    // Clear smoothed positions 
    smoothedPosLeft = { x: 0, y: 0, z: 0 };
    smoothedPosRight = { x: 0, y: 0, z: 0 };


    await startAR();
}

// --- INITIALIZE & START AR (UNCHANGED) ---

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
        // 🔑 REMOVE OLD GROUP ON CAMERA SWITCH/RE-INIT
        if (wingsGroup) scene.remove(wingsGroup);
    } else {
        scene = new THREE.Scene();
    }
    
    // 🔑 CREATE NEW GROUP
    wingsGroup = new THREE.Group();
    scene.add(wingsGroup); // Add the group to the scene
    
    const aspect = containerRect.width / containerRect.height;
    camera = new THREE.PerspectiveCamera(65, aspect, 0.1, 100); 
    camera.position.set(0, 0, 0); 
    scene.add(new THREE.AmbientLight(0xffffff, 1.0));

    // Video Background Plane setup
    const videoTexture = new THREE.VideoTexture(video);
    videoTexture.flipY = false; 
    if (CAMERA_MODE === 'user') {
        videoTexture.wrapS = THREE.RepeatWrapping; videoTexture.offset.x = 1; videoTexture.repeat.x = -1; 
    } else {
        videoTexture.wrapS = THREE.ClampToEdgeWrapping; videoTexture.offset.x = 0; videoTexture.repeat.x = 1; 
    }
    const planeGeometry = new THREE.PlaneGeometry(1, 1);
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
        // 🔑 RE-ADD GROUP IF ALREADY LOADED
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
        // 🔑 ADD TO GROUP
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
        // 🔑 ADD TO GROUP
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

    // 🔑 ADD TO GROUP
    wingsGroup.add(wingsAssetLeft);
    wingsGroup.add(wingsAssetRight);

    wingsAssetLeft.visible = false;
    wingsAssetRight.visible = false;
    isSplatAttempted = false;
    isSplatDataReady = true; 
    
    debugLogger.updateAssetStatus('Box placeholder active (Fallback)');
}

// === MAIN RENDER LOOP (MODIFIED) ===
async function renderLoop() {
    if (!isRunning) return;

    requestAnimationFrame(renderLoop);

    // FPS Counter (omitted for brevity)
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
                        
                        // 🔑 1. CALCULATE GROUP POSITION (Average of shoulders)
                        const avgShoulderX = (leftShoulder.x + rightShoulder.x) / 2;
                        const avgShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
                        
                        // Pass average position to a new group positioning function
                        positionWingsGroup(wingsGroup, avgShoulderX, avgShoulderY);
                        
                        // 🔑 2. CALCULATE GROUP ROTATION (Based on shoulder height difference)
                        // If one shoulder is higher than the other (head tilt/slouch)
                        // Rotation must be inverted when in selfie mode
                        const yDiff = leftShoulder.y - rightShoulder.y; 
                        let targetRotX = (yDiff / Y_DIFFERENCE_SENSITIVITY) * MAX_X_ROTATION;
                        targetRotX = THREE.MathUtils.clamp(targetRotX, -MAX_X_ROTATION, MAX_X_ROTATION);
                        
                        if (CAMERA_MODE === 'user') {
                            targetRotX = -targetRotX; // Reverse tilt for selfie camera
                        }
                        
                        // Apply smoothing to the rotation for stability
                        wingsGroup.rotation.x += (targetRotX - wingsGroup.rotation.x) * SMOOTHING_FACTOR;

                        // 🔑 3. POSITION INDIVIDUAL WINGS RELATIVE TO GROUP CENTER
                        // We only need to set the scale and the *local* rotations
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

// === NEW GROUP POSITIONING FUNCTION ===

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
    targetY -= (WING_VERTICAL_SHIFT * 1.0); // 1.0 is scale factor equivalent
    
    // 3. Apply Z Depth Offset (pushes it behind the user)
    targetZ += BACK_OFFSET_Z; 

    // Apply Smoothing and set Position for the GROUP
    // Use 'smoothedPosLeft' as the shared smoother for the group's position
    smoothedPosLeft.x = smoothedPosLeft.x + (targetX - smoothedPosLeft.x) * SMOOTHING_FACTOR;
    smoothedPosLeft.y = smoothedPosLeft.y + (targetY - smoothedPosLeft.y) * SMOOTHING_FACTOR;
    smoothedPosLeft.z = smoothedPosLeft.z + (targetZ - smoothedPosLeft.z) * SMOOTHING_FACTOR;
    
    group.position.set(smoothedPosLeft.x, smoothedPosLeft.y, smoothedPosLeft.z);
}

// === MODIFIED INDIVIDUAL WING POSITIONING FUNCTION ===

/**
 * Position and Scale a single wing asset based on a keypoint.
 * NOTE: The wing's position is now relative to the wingsGroup center (avg shoulder point)
 */
function positionIndividualWing(wing, side) {
    
    // 1. Position RELATIVE to the Group Center (0,0,0 of the group is the average shoulder point)
    const FIXED_SCALE = 1.0; 
    
    // The relative X position is based on the WING_HORIZONTAL_OFFSET
    if (side === 'left') {
        wing.position.set(WING_HORIZONTAL_OFFSET * FIXED_SCALE, 0, 0); 
    } else if (side === 'right') {
        wing.position.set(-WING_HORIZONTAL_OFFSET * FIXED_SCALE, 0, 0); 
    }
    
    // 2. Apply Fixed Scale (Same as before)
    let finalScaleFactor = wing instanceof SplatMesh ? WING_SPLAT_SCALE_FACTOR : 1.2; 
    wing.scale.set(finalScaleFactor, finalScaleFactor, finalScaleFactor * 1.5); 

    // 3. Apply INDIVIDUAL WING ROTATION (No more X rotation, that's done on the group)
    
    // X-axis: STAND THE WINGS UP (Makes them vertical relative to the floor)
    const baseRotX = -Math.PI * 0.2; 

    // Y-axis: FACE THE GOLDEN SIDE FORWARD (Flips them to face the camera from the back)
    const baseRotY = Math.PI; 
    
    // Z-axis: Pivot and Splay (Remains the same for the 180 degree pivot and splay outwards)
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
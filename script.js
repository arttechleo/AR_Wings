// --- MODULE IMPORTS ---
import * as THREE from 'three';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { SplatMesh, SparkRenderer } from "@sparkjsdev/spark"; 

// Global variables for the scene and pose detection
let scene, camera;
let threeRendererInstance; 
let wingsAsset; 
let video, canvas, ctx;
let poseModel;
let debugLogger;
let isRunning = false;
let frameCount = 0;
let lastFpsUpdate = Date.now();
let videoBackgroundPlane; // New: For 3D video texture

// --- STATE FLAGS ---
let isSplatAttempted = false;
let isSplatDataReady = false; 

// Smoothing variables for stable positioning
let smoothedPos = { x: 0, y: 0, z: 0 }; 
let smoothedRot = { x: 0, y: 0, z: 0 }; 
const SMOOTHING_FACTOR = 0.6; 

// Gaussian Splatting configuration
const USE_GAUSSIAN_SPLAT = true; 
const SPLAT_PATH_WINGS = './assets/wings.ksplat'; 
const WING_DOWNWARD_SHIFT = 0.2; 

// *** CONFIRMED MODE: For laptop camera (selfie) ***
const CAMERA_MODE = 'environment'; 

// --- AR SETTINGS (FIXED VALUES) ---
const BACK_OFFSET_Z = -0.7; // Pushes the wings further back from the detection plane
const WING_SPLAT_SCALE_FACTOR = 0.5; 
const TEST_DEPTH_Z = -1.0; 
const VIDEO_PLANE_DEPTH = -5.0; // Far back in 3D space to act as a background

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
    updatePositionStatus(pos, rot) {
        if (this.positionStatus) {
            this.positionStatus.textContent = `P: (${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)}) R: (${rot.x.toFixed(2)}, ${rot.y.toFixed(2)}, ${rot.z.toFixed(2)})`;
        }
    }
}
// === END DEBUG LOGGER CLASS ===

// --- INITIALIZE & START AR (STANDARD - UNCHANGED) ---
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
        });
    }

    debugLogger.updateStatus('Ready - Tap Start');
}

async function startAR() {
    try {
        debugLogger.updateStatus('Initializing TensorFlow...');
        tf.setBackend('webgl'); 
        await tf.ready(); 
        debugLogger.log('success', `TensorFlow backend ready (${tf.getBackend()}).`);

        const threeContainer = document.getElementById('three-container');
        canvas = document.getElementById('output-canvas');
        ctx = canvas.getContext('2d');
        video = document.getElementById('video');

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: CAMERA_MODE, width: { ideal: 1280 }, height: { ideal: 720 } }
        });
        video.srcObject = stream;
        video.play(); 
        debugLogger.updateVideoStatus('Camera stream active');

        await new Promise((resolve) => { video.onloadedmetadata = () => { resolve(video); }; });

        const vw = video.videoWidth;
        const vh = video.videoHeight;
        
        canvas.width = vw;
        canvas.height = vh;
        threeContainer.style.width = '100vw';
        threeContainer.style.height = '100vh';

        debugLogger.updateStatus('Setting up 3D renderer...');
        setupThreeJS(vw, vh); 
        debugLogger.log('success', '3D renderer ready');

        debugLogger.updateStatus('Loading AI model...');
        poseModel = await poseDetection.createDetector(
            poseDetection.SupportedModels.MoveNet,
            { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
        );
        debugLogger.log('success', 'AI model loaded!');
        debugLogger.updateStatus('Running - Stand back!');

        isRunning = true;
        renderLoop();
    } catch (error) {
        debugLogger.log('error', `INIT ERROR: ${error.name}: ${error.message}`);
        debugLogger.updateStatus('FATAL ERROR');
        document.getElementById('instructions').classList.remove('hidden');
        document.getElementById('instructions').innerHTML = `
            <h2>Initialization Failed!</h2>
            <p>Error: ${error.message}</p>
            <p>Check if HTTPS is active and camera permissions are granted.</p>
        `;
    }
}

// === SETUP THREE.JS (CRITICAL CHANGES FOR VIDEO BACKGROUND) ===
function setupThreeJS(videoWidth, videoHeight) {
    const threeContainer = document.getElementById('three-container');
    const containerRect = threeContainer.getBoundingClientRect();

    const threeRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    threeRenderer.setPixelRatio(window.devicePixelRatio);
    threeRenderer.setSize(containerRect.width, containerRect.height);
    // Ensure GL background is transparent
    threeRenderer.setClearColor(0x000000, 0); 
    threeContainer.appendChild(threeRenderer.domElement);

    threeRendererInstance = threeRenderer;
    
    new SparkRenderer(threeRenderer);

    scene = new THREE.Scene();
    const aspect = containerRect.width / containerRect.height;
    
    // Use an FOV to match a typical camera (e.g., 60-75 degrees)
    camera = new THREE.PerspectiveCamera(65, aspect, 0.1, 100); 
    camera.position.set(0, 0, 0); 
    scene.add(new THREE.AmbientLight(0xffffff, 1.0));

    // ----------------------------------------------------
    // *** NEW: CREATE VIDEO BACKGROUND PLANE ***
    // Renders the live video feed as a 3D texture.
    // ----------------------------------------------------
    const videoTexture = new THREE.VideoTexture(video);
    
    if (CAMERA_MODE === 'user') {
        // FIX: flipY = true corrects the upside-down orientation for selfie mode
        videoTexture.flipY = true; 
        // Mirror horizontally for selfie mode
        videoTexture.wrapS = THREE.RepeatWrapping;
        videoTexture.offset.x = 1;
        videoTexture.repeat.x = -1;
    } else {
        // Default flip for environment mode
        videoTexture.flipY = false;
    }

    const planeGeometry = new THREE.PlaneGeometry(1, 1);
    const planeMaterial = new THREE.MeshBasicMaterial({
        map: videoTexture,
        side: THREE.FrontSide,
        depthTest: false // Render this layer first
    });
    
    videoBackgroundPlane = new THREE.Mesh(planeGeometry, planeMaterial);
    
    // Calculate the size the plane needs to be to fill the screen at VIDEO_PLANE_DEPTH
    const viewAspect = containerRect.width / containerRect.height;
    const fovRad = THREE.MathUtils.degToRad(camera.fov);
    const planeHeight = Math.abs(2 * Math.tan(fovRad / 2) * VIDEO_PLANE_DEPTH);
    const planeWidth = planeHeight * viewAspect;

    videoBackgroundPlane.scale.set(planeWidth, planeHeight, 1);
    videoBackgroundPlane.position.z = VIDEO_PLANE_DEPTH;
    videoBackgroundPlane.renderOrder = 0; // Render first (background)

    scene.add(videoBackgroundPlane);

    // ----------------------------------------------------
    // *** ASSET LOADING LOGIC ***
    // ----------------------------------------------------
    if (USE_GAUSSIAN_SPLAT && typeof SplatMesh !== 'undefined') {
        debugLogger.updateAssetStatus(`Checking ${SPLAT_PATH_WINGS}...`);

        fetch(SPLAT_PATH_WINGS)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP Error ${response.status}: Failed to fetch asset.`);
                }
                debugLogger.log('info', 'Asset resource check passed. Starting SplatMesh load.');
                loadSplatModel();
            })
            .catch(err => {
                debugLogger.log('error', `FATAL Asset Load Error: ${err.message}. Falling back to boxes.`);
                createBoxWings();
            });

        isSplatAttempted = true; 
    } else {
        createBoxWings();
    }
}
// === END SETUP THREE.JS ===

// --- ASSET LOADING AND FALLBACK (STANDARD - UNCHANGED) ---

function loadSplatModel() {
    debugLogger.updateAssetStatus('Loading Gaussian Splat...');

    try {
        wingsAsset = new SplatMesh({ 
            url: SPLAT_PATH_WINGS, 
            fileType: 'ksplat', 
            onLoad: (mesh) => {
                isSplatDataReady = true; 
                debugLogger.log('success', 'Gaussian Splat data loaded and ready!');
                debugLogger.updateAssetStatus('Gaussian Splats active');
                
                // Z-inversion for Spark/Three.js camera alignment
                mesh.scale.set(1, 1, -1); 
            }
        });
        wingsAsset.visible = false;
        wingsAsset.renderOrder = 1; // Render second (in front of background)

        scene.add(wingsAsset);
        
    } catch (err) {
        debugLogger.log('error', `Splat instantiation error: ${err.message}. Falling back to boxes.`);
        createBoxWings();
    }
}

function createBoxWings() {
    if (wingsAsset && scene.children.includes(wingsAsset)) {
        scene.remove(wingsAsset);
    }
    
    const wingGeometry = new THREE.BoxGeometry(0.5, 0.8, 0.08); 
    const wingMaterial = new THREE.MeshBasicMaterial({
        color: 0x00ccff,
        transparent: true,
        opacity: 0.8
    });

    wingsAsset = new THREE.Mesh(wingGeometry, wingMaterial);
    scene.add(wingsAsset);

    wingsAsset.visible = false;
    isSplatAttempted = false;
    isSplatDataReady = true; 
    
    debugLogger.updateAssetStatus('Box placeholder active (Fallback)');
}

// === MAIN RENDER LOOP (UPDATED) ===
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

    // CRITICAL: Clear the 2D canvas. The video is now handled by the 3D plane.
    // We only use the 2D canvas for debug points (z:3).
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 2. Pose Detection Logic
    if (video.readyState >= video.HAVE_ENOUGH_DATA && poseModel) {
        try {
            const poses = await poseModel.estimatePoses(video);
            
            if (poses.length > 0) {
                const keypoints = poses[0].keypoints;
                const leftShoulder = keypoints.find(kp => kp.name === 'left_shoulder');
                const rightShoulder = keypoints.find(kp => kp.name === 'right_shoulder');
                const leftHip = keypoints.find(kp => kp.name === 'left_hip');
                const rightHip = keypoints.find(kp => kp.name === 'right_hip');

                if (leftShoulder && rightShoulder && leftShoulder.score > 0.4 && rightShoulder.score > 0.4) {
                    
                    debugLogger.updatePoseStatus(`Detected (L:${leftShoulder.score.toFixed(2)}, R:${rightShoulder.score.toFixed(2)})`);

                    const spineCenter = getSpineCenter(leftShoulder, rightShoulder, leftHip, rightHip);
                    const bodyAngle = Math.atan2(rightShoulder.y - leftShoulder.y, rightShoulder.x - leftShoulder.x);

                    if (wingsAsset && isSplatDataReady) {
                        positionSingleWingSet(wingsAsset, spineCenter, 0, bodyAngle);
                        wingsAsset.visible = true;
                        debugLogger.updatePositionStatus(wingsAsset.position, wingsAsset.rotation);
                    } else if (wingsAsset) {
                        wingsAsset.visible = false; 
                    }

                    // Draw debug points to the now-transparent #output-canvas (z:3)
                    drawDebugPoints(ctx, [leftShoulder, rightShoulder]); 

                } else {
                    wingsAsset.visible = false;
                    debugLogger.updatePoseStatus('Low confidence');
                }
            } else {
                wingsAsset.visible = false;
                debugLogger.updatePoseStatus('No person detected');
            }
        } catch (err) {
            debugLogger.log('error', `Pose detection error: ${err.message}`);
        }
    }

    // 3. Render the scene (This is where the depth sorting happens)
    if (threeRendererInstance) {
        // Must manually update video texture in the loop
        if (videoBackgroundPlane && videoBackgroundPlane.material.map) {
            videoBackgroundPlane.material.map.needsUpdate = true;
        }
        threeRendererInstance.render(scene, camera);
    }
}
// === END MAIN RENDER LOOP ===

// === HELPER FUNCTIONS (STANDARD - UNCHANGED) ===

function getSpineCenter(leftShoulder, rightShoulder, leftHip, rightHip) {
    let spineCenter = {
        x: (leftShoulder.x + rightShoulder.x) / 2,
        y: (leftShoulder.y + rightShoulder.y) / 2
    };

    if (leftHip && rightHip && leftHip.score > 0.3 && rightHip.score > 0.3) {
        const hipCenterY = (leftHip.y + rightHip.y) / 2;
        spineCenter.y = (spineCenter.y * 0.7 + hipCenterY * 0.3); 
    }
    return spineCenter;
}

/**
 * Position, Scale, and Rotate the single wings asset based on the spine center.
 */
function positionSingleWingSet(wing, spineCenter, shoulderDist, bodyAngle) {
    
    const depth = TEST_DEPTH_Z; 
    const FIXED_SCALE = 1.0; 
    
    const normX = (coord, dim) => (coord / dim) * 2 - 1;
    const normY = (coord, dim) => -(coord / dim) * 2 + 1; 

    let targetX = normX(spineCenter.x, video.videoWidth);
    let targetY = normY(spineCenter.y, video.videoHeight);
    let targetZ = depth; 

    // Apply Camera Corrections & Depth Offset
    if (CAMERA_MODE === 'user') {
        targetX = -targetX; 
    }
    targetY -= (WING_DOWNWARD_SHIFT * FIXED_SCALE); 
    targetZ += BACK_OFFSET_Z; 

    // Apply Smoothing to Position
    smoothedPos.x = smoothedPos.x + (targetX - smoothedPos.x) * SMOOTHING_FACTOR;
    smoothedPos.y = smoothedPos.y + (targetY - smoothedPos.y) * SMOOTHING_FACTOR;
    smoothedPos.z = smoothedPos.z + (targetZ - smoothedPos.z) * SMOOTHING_FACTOR;
    wing.position.set(smoothedPos.x, smoothedPos.y, smoothedPos.z);
    
    // Apply Fixed Scale
    let finalScaleFactor = wing instanceof SplatMesh ? WING_SPLAT_SCALE_FACTOR : 1.2; 
    wing.scale.set(finalScaleFactor, finalScaleFactor, finalScaleFactor * 1.5); 

    // Apply Rotation
    const targetRotX = 0; 
    const targetRotY = Math.PI; // FLIP 180 degrees
    const bodyRotationInfluence = CAMERA_MODE === 'user' ? bodyAngle : -bodyAngle; 
    const targetRotZ = bodyRotationInfluence * 1.0; 
    
    smoothedRot.x = smoothedRot.x + (targetRotX - smoothedRot.x) * SMOOTHING_FACTOR;
    smoothedRot.y = smoothedRot.y + (targetRotY - smoothedRot.y) * SMOOTHING_FACTOR;
    smoothedRot.z = smoothedRot.z + (targetRotZ - smoothedRot.z) * SMOOTHING_FACTOR;
    
    wing.rotation.set(smoothedRot.x, smoothedRot.y, smoothedRot.z);
}

// Draw Debug Points (Only draws on the topmost canvas)
function drawDebugPoints(ctx, keypoints) {
    
    ctx.fillStyle = '#00ff88';
    keypoints.forEach(kp => {
        if (kp.score > 0.4) {
            let x = kp.x;
            const y = kp.y;
            
            // X-mirroring logic for debug points on top of selfie video
            if (CAMERA_MODE === 'user') {
                x = canvas.width - x;
            }
            
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();
        }
    });
}

// === START WHEN PAGE LOADS (STANDARD - UNCHANGED) ===
window.addEventListener('DOMContentLoaded', () => {
    init();
});
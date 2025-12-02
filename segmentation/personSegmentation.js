/**
 * Person Segmentation Module (Lightweight - MediaPipe)
 * 
 * Provides real-time person segmentation using MediaPipe Selfie Segmentation.
 * Optimized for performance with downscaled processing.
 */

import { SelfieSegmentation } from '@mediapipe/selfie_segmentation';
import * as THREE from 'three';

let segmentationModel = null;
let isInitializing = false;
let lastMaskTexture = null;
let lastMaskCanvas = null;
let isReady = false;

// Configuration - optimized for performance
const SEGMENTATION_CONFIG = {
    inputWidth: 256,   // Downscaled for fast inference
    inputHeight: 144,
    modelSelection: 0, // 0 = general, 1 = landscape (faster)
};

// Processing canvas
let processingCanvas = null;
let processingCtx = null;

/**
 * Initialize MediaPipe Selfie Segmentation
 */
export async function initPersonSegmentation() {
    if (segmentationModel !== null) {
        return true;
    }

    if (isInitializing) {
        while (isInitializing) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        return isReady;
    }

    isInitializing = true;

    try {
        // Create processing canvas
        processingCanvas = document.createElement('canvas');
        processingCanvas.width = SEGMENTATION_CONFIG.inputWidth;
        processingCanvas.height = SEGMENTATION_CONFIG.inputHeight;
        processingCtx = processingCanvas.getContext('2d', { willReadFrequently: true });

        // Initialize MediaPipe Selfie Segmentation
        segmentationModel = new SelfieSegmentation({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`;
            }
        });

        segmentationModel.setOptions({
            modelSelection: SEGMENTATION_CONFIG.modelSelection, // 0 = general, 1 = landscape (faster)
        });

        // Set up callback to handle results
        segmentationModel.onResults((results) => {
            if (pendingPromiseResolver) {
                try {
                    if (!results || !results.segmentationMask) {
                        pendingPromiseResolver(lastMaskTexture ? { maskTexture: lastMaskTexture, rawMask: lastMaskCanvas } : null);
                        pendingPromiseResolver = null;
                        return;
                    }

                    // Use stored video dimensions
                    const videoWidth = currentVideoWidth || processingCanvas.width * 4;
                    const videoHeight = currentVideoHeight || processingCanvas.height * 4;

                    // Create mask canvas at full video resolution
                    const maskCanvas = document.createElement('canvas');
                    maskCanvas.width = videoWidth;
                    maskCanvas.height = videoHeight;
                    const maskCtx = maskCanvas.getContext('2d');

                    // Draw the segmentation mask (upscaled to full resolution)
                    maskCtx.drawImage(
                        results.segmentationMask,
                        0, 0,
                        maskCanvas.width,
                        maskCanvas.height
                    );

                    // Create or update Three.js texture
                    if (!lastMaskTexture) {
                        lastMaskTexture = new THREE.CanvasTexture(maskCanvas);
                        lastMaskTexture.flipY = false;
                        lastMaskTexture.minFilter = THREE.LinearFilter;
                        lastMaskTexture.magFilter = THREE.LinearFilter;
                        lastMaskTexture.format = THREE.AlphaFormat;
                    } else {
                        lastMaskTexture.image = maskCanvas;
                        lastMaskTexture.needsUpdate = true;
                    }

                    lastMaskCanvas = maskCanvas;
                    pendingPromiseResolver({ maskTexture: lastMaskTexture, rawMask: maskCanvas });
                    pendingPromiseResolver = null;
                } catch (err) {
                    console.error('Error processing segmentation results:', err);
                    pendingPromiseResolver(lastMaskTexture ? { maskTexture: lastMaskTexture, rawMask: lastMaskCanvas } : null);
                    pendingPromiseResolver = null;
                }
            }
        });

        await segmentationModel.initialize();
        
        isReady = true;
        isInitializing = false;
        console.log('MediaPipe Selfie Segmentation initialized');
        return true;
    } catch (error) {
        console.error('Failed to initialize MediaPipe segmentation:', error);
        isInitializing = false;
        isReady = false;
        return false;
    }
}

// Store pending promise resolver and video dimensions
let pendingPromiseResolver = null;
let currentVideoWidth = 0;
let currentVideoHeight = 0;

/**
 * Generate a segmentation mask from video frame
 * @param {HTMLVideoElement} videoEl - Video element
 * @returns {Promise<{ maskTexture: THREE.Texture, rawMask: HTMLCanvasElement } | null>}
 */
export async function runPersonSegmentationFrame(videoEl) {
    if (!videoEl || videoEl.readyState < videoEl.HAVE_ENOUGH_DATA || !segmentationModel || !isReady) {
        return lastMaskTexture ? { maskTexture: lastMaskTexture, rawMask: lastMaskCanvas } : null;
    }

    // Store video dimensions for callback
    currentVideoWidth = videoEl.videoWidth || videoEl.clientWidth;
    currentVideoHeight = videoEl.videoHeight || videoEl.clientHeight;

    return new Promise((resolve) => {
        try {
            // Downscale video frame for processing (faster inference)
            processingCtx.drawImage(
                videoEl,
                0, 0,
                SEGMENTATION_CONFIG.inputWidth,
                SEGMENTATION_CONFIG.inputHeight
            );

            // Store resolver for callback
            pendingPromiseResolver = resolve;

            // Send frame to MediaPipe (callback will be called via onResults)
            segmentationModel.send({ image: processingCanvas });
        } catch (error) {
            console.error('Segmentation error:', error);
            resolve(lastMaskTexture ? { maskTexture: lastMaskTexture, rawMask: lastMaskCanvas } : null);
        }
    });
}

/**
 * Generate mask from pose keypoints (fallback method)
 * Creates a simple silhouette around detected body keypoints
 */
export function generateMaskFromPose(keypoints, videoWidth, videoHeight) {
    if (!keypoints || keypoints.length === 0) {
        return null;
    }

    const canvas = document.createElement('canvas');
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    const ctx = canvas.getContext('2d');

    // Find key body points
    const head = keypoints.find(kp => kp.name === 'nose') || 
                 keypoints.find(kp => kp.name === 'left_eye') ||
                 keypoints.find(kp => kp.name === 'right_eye');
    const leftShoulder = keypoints.find(kp => kp.name === 'left_shoulder');
    const rightShoulder = keypoints.find(kp => kp.name === 'right_shoulder');
    const leftHip = keypoints.find(kp => kp.name === 'left_hip');
    const rightHip = keypoints.find(kp => kp.name === 'right_hip');

    if (!leftShoulder || !rightShoulder) {
        return null;
    }

    ctx.fillStyle = 'white';
    ctx.strokeStyle = 'white';

    // Draw approximate body silhouette
    const shoulderWidth = Math.abs(rightShoulder.x - leftShoulder.x);
    const avgShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
    const avgShoulderX = (leftShoulder.x + rightShoulder.x) / 2;

    // Draw head if available
    if (head && head.score > 0.3) {
        ctx.beginPath();
        ctx.arc(head.x, head.y, 40, 0, Math.PI * 2);
        ctx.fill();
    }

    // Draw torso
    if (leftHip && rightHip && leftHip.score > 0.3 && rightHip.score > 0.3) {
        const hipWidth = Math.abs(rightHip.x - leftHip.x);
        const avgHipY = (leftHip.y + rightHip.y) / 2;
        const avgHipX = (leftHip.x + rightHip.x) / 2;
        const torsoHeight = avgHipY - avgShoulderY;

        ctx.beginPath();
        ctx.ellipse(
            avgShoulderX, avgShoulderY + torsoHeight / 2,
            Math.max(shoulderWidth / 2, 60), torsoHeight / 2,
            0, 0, Math.PI * 2
        );
        ctx.fill();
    } else {
        // Fallback: just upper body
        ctx.beginPath();
        ctx.ellipse(
            avgShoulderX, avgShoulderY + 100,
            Math.max(shoulderWidth / 2, 60), 150,
            0, 0, Math.PI * 2
        );
        ctx.fill();
    }

    // Convert to texture
    const texture = new THREE.CanvasTexture(canvas);
    texture.flipY = false;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.format = THREE.AlphaFormat;

    return { maskTexture: texture, rawMask: canvas };
}

/**
 * Check if segmentation is ready
 */
export function isPersonSegmentationReady() {
    return isReady;
}

/**
 * Dispose resources
 */
export function disposePersonSegmentation() {
    if (segmentationModel) {
        segmentationModel.close();
        segmentationModel = null;
    }
    lastMaskTexture = null;
    lastMaskCanvas = null;
    processingCanvas = null;
    processingCtx = null;
    isReady = false;
}

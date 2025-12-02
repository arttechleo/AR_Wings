/**
 * Human Segmentation Module
 * 
 * Provides real-time human body segmentation using TensorFlow.js BodyPix.
 * Returns segmentation masks and body part information for AR compositing.
 */

import * as bodyPix from '@tensorflow-models/body-pix';
import * as tf from '@tensorflow/tfjs';

let bodyPixNet = null;
let isInitializing = false;
let lastSegmentationResult = null;
let offscreenCanvas = null;
let offscreenCtx = null;

// Configuration for segmentation performance
const SEGMENTATION_CONFIG = {
    architecture: 'MobileNetV1',  // Fast variant for 30-45 FPS target
    outputStride: 16,              // 16 = faster, 8 = more accurate
    multiplier: 0.75,              // 0.75 = faster, 1.0 = more accurate
    quantBytes: 2,                 // 2 = faster, 4 = more accurate
    enableBodyPartSegmentation: true, // Enable for orientation detection
};

// Input resolution for segmentation (downscale for performance)
const SEGMENTATION_WIDTH = 512;
const SEGMENTATION_HEIGHT = 384;

/**
 * Initialize the BodyPix model
 * @returns {Promise<boolean>} True if initialization successful
 */
export async function initHumanSegmentation() {
    if (bodyPixNet !== null) {
        return true; // Already initialized
    }

    if (isInitializing) {
        // Wait for ongoing initialization
        while (isInitializing) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        return bodyPixNet !== null;
    }

    isInitializing = true;

    try {
        // Use MobileNetV1 for speed (target: 30-45 FPS)
        bodyPixNet = await bodyPix.load({
            architecture: SEGMENTATION_CONFIG.architecture,
            outputStride: SEGMENTATION_CONFIG.outputStride,
            multiplier: SEGMENTATION_CONFIG.multiplier,
            quantBytes: SEGMENTATION_CONFIG.quantBytes,
        });

        // Create offscreen canvas for mask processing (fallback to regular canvas if OffscreenCanvas not supported)
        try {
            offscreenCanvas = new OffscreenCanvas(SEGMENTATION_WIDTH, SEGMENTATION_HEIGHT);
            offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
        } catch (e) {
            // Fallback: Use regular canvas if OffscreenCanvas not supported
            offscreenCanvas = document.createElement('canvas');
            offscreenCanvas.width = SEGMENTATION_WIDTH;
            offscreenCanvas.height = SEGMENTATION_HEIGHT;
            offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });
        }

        isInitializing = false;
        return true;
    } catch (error) {
        console.error('Failed to initialize BodyPix:', error);
        isInitializing = false;
        return false;
    }
}

/**
 * Run segmentation on a video frame
 * @param {HTMLVideoElement} videoEl - Video element to segment
 * @returns {Promise<{mask: HTMLCanvasElement | null, isFacingCamera: boolean, isBackToCamera: boolean} | null>}
 */
export async function runSegmentationFrame(videoEl) {
    if (!bodyPixNet || !videoEl || videoEl.readyState < videoEl.HAVE_ENOUGH_DATA) {
        return lastSegmentationResult; // Return last result if model not ready
    }

    try {
        // Run segmentation with body parts for orientation detection
        const segmentation = await bodyPixNet.segmentPersonParts(videoEl, {
            flipHorizontal: false,
            internalResolution: 'medium', // 'low', 'medium', 'high', 'full'
            segmentationThreshold: 0.7,
        });

        // Create mask canvas
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = videoEl.videoWidth;
        maskCanvas.height = videoEl.videoHeight;
        const maskCtx = maskCanvas.getContext('2d');

        // Convert segmentation to mask
        // BodyPix toMask returns a canvas with the mask
        const maskImageCanvas = bodyPix.toMask(segmentation, 
            { r: 255, g: 255, b: 255, a: 255 }, // Foreground color (white)
            { r: 0, g: 0, b: 0, a: 0 }         // Background color (transparent)
        );

        // Scale mask to video resolution
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        maskCtx.drawImage(maskImageCanvas, 0, 0, maskCanvas.width, maskCanvas.height);

        // Estimate orientation from body parts
        const orientation = estimateOrientationFromBodyParts(segmentation);

        lastSegmentationResult = {
            mask: maskCanvas,
            isFacingCamera: orientation.isFacingCamera,
            isBackToCamera: orientation.isBackToCamera,
            bodyParts: segmentation, // Store full segmentation for additional processing
        };

        return lastSegmentationResult;
    } catch (error) {
        console.error('Segmentation error:', error);
        return lastSegmentationResult; // Return last known good result
    }
}

/**
 * Estimate person orientation from body part segmentation
 * BodyPix body part IDs:
 * 0: background, 1: torso, 2: left shoulder, 3: left elbow, 4: left wrist,
 * 5: right shoulder, 6: right elbow, 7: right wrist, 8: left hip, 9: left knee,
 * 10: left ankle, 11: right hip, 12: right knee, 13: right ankle,
 * 14: left face, 15: right face, 16: torso front, 17: torso back
 * 
 * @param {Object} segmentation - BodyPix segmentation result
 * @returns {{isFacingCamera: boolean, isBackToCamera: boolean}}
 */
function estimateOrientationFromBodyParts(segmentation) {
    if (!segmentation || !segmentation.data) {
        return { isFacingCamera: false, isBackToCamera: false };
    }

    const data = segmentation.data;
    const width = segmentation.width;
    const height = segmentation.height;

    let facePixels = 0;        // Parts 14, 15 (left_face, right_face)
    let torsoFrontPixels = 0;  // Part 16 (torso_front)
    let torsoBackPixels = 0;   // Part 17 (torso_back)
    let totalBodyPixels = 0;

    // Count pixels for each body part
    for (let i = 0; i < data.length; i++) {
        const partId = data[i];
        
        if (partId === 0) continue; // Skip background
        
        totalBodyPixels++;
        
        if (partId === 14 || partId === 15) {
            facePixels++;
        } else if (partId === 16) {
            torsoFrontPixels++;
        } else if (partId === 17) {
            torsoBackPixels++;
        }
    }

    if (totalBodyPixels === 0) {
        return { isFacingCamera: false, isBackToCamera: false };
    }

    const faceRatio = facePixels / totalBodyPixels;
    const torsoFrontRatio = torsoFrontPixels / totalBodyPixels;
    const torsoBackRatio = torsoBackPixels / totalBodyPixels;

    // Heuristics for orientation detection:
    // - Clear face detection (parts 14, 15) → facing camera
    // - More torso_front than torso_back → facing camera
    // - More torso_back than torso_front → back to camera
    // - No face and mostly torso_back → back to camera

    const FACE_THRESHOLD = 0.02;  // At least 2% of body pixels are face
    const TORSO_FRONT_THRESHOLD = 0.15;  // At least 15% are front torso
    const TORSO_BACK_THRESHOLD = 0.15;   // At least 15% are back torso

    const isFacingCamera = faceRatio > FACE_THRESHOLD || 
                          (torsoFrontRatio > TORSO_FRONT_THRESHOLD && torsoFrontRatio > torsoBackRatio);
    
    const isBackToCamera = torsoBackRatio > TORSO_BACK_THRESHOLD && 
                          torsoBackRatio > torsoFrontRatio && 
                          faceRatio < FACE_THRESHOLD;

    return {
        isFacingCamera,
        isBackToCamera,
    };
}

/**
 * Get the last segmentation result (useful for throttling)
 * @returns {Object | null}
 */
export function getLastSegmentationResult() {
    return lastSegmentationResult;
}

/**
 * Check if segmentation is initialized
 * @returns {boolean}
 */
export function isSegmentationReady() {
    return bodyPixNet !== null;
}

/**
 * Dispose of the model and clean up resources
 */
export function disposeSegmentation() {
    if (bodyPixNet) {
        bodyPixNet.dispose();
        bodyPixNet = null;
    }
    lastSegmentationResult = null;
    offscreenCanvas = null;
    offscreenCtx = null;
}


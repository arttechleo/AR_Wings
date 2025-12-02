/**
 * Orientation Detection Helper
 * 
 * Provides utilities for estimating person orientation (front/back) 
 * based on pose keypoints and segmentation data.
 */

/**
 * Estimate orientation from pose keypoints
 * Uses face/head visibility and shoulder/hip positions to determine orientation
 * 
 * @param {Array} keypoints - Array of pose keypoints from MoveNet
 * @returns {{isFacingCamera: boolean, isBackToCamera: boolean, confidence: number}}
 */
export function estimateOrientationFromPose(keypoints) {
    if (!keypoints || keypoints.length === 0) {
        return { isFacingCamera: false, isBackToCamera: false, confidence: 0 };
    }

    // Find relevant keypoints
    const nose = keypoints.find(kp => kp.name === 'nose');
    const leftEye = keypoints.find(kp => kp.name === 'left_eye');
    const rightEye = keypoints.find(kp => kp.name === 'right_eye');
    const leftShoulder = keypoints.find(kp => kp.name === 'left_shoulder');
    const rightShoulder = keypoints.find(kp => kp.name === 'right_shoulder');
    const leftHip = keypoints.find(kp => kp.name === 'left_hip');
    const rightHip = keypoints.find(kp => kp.name === 'right_hip');

    // Calculate confidence based on keypoint visibility
    let confidence = 0;
    let faceVisible = false;
    let upperBodyVisible = false;

    // Check face visibility
    if (nose && nose.score > 0.5) {
        faceVisible = true;
        confidence += 0.3;
    }
    if ((leftEye && leftEye.score > 0.5) || (rightEye && rightEye.score > 0.5)) {
        faceVisible = true;
        confidence += 0.2;
    }

    // Check upper body visibility
    if (leftShoulder && rightShoulder && 
        leftShoulder.score > 0.4 && rightShoulder.score > 0.4) {
        upperBodyVisible = true;
        confidence += 0.3;
    }
    if (leftHip && rightHip && 
        leftHip.score > 0.4 && rightHip.score > 0.4) {
        confidence += 0.2;
    }

    // Improved orientation heuristics:
    // - Face visible (nose/eyes) → likely facing camera (high confidence)
    // - No face but shoulders wider than hips → likely facing camera
    // - No face, shoulders similar to hips → likely back to camera
    // - Ear keypoints can also help (MoveNet provides left_ear, right_ear)

    let isFacingCamera = false;
    let isBackToCamera = false;

    const leftEar = keypoints.find(kp => kp.name === 'left_ear');
    const rightEar = keypoints.find(kp => kp.name === 'right_ear');
    const earVisible = (leftEar && leftEar.score > 0.4) || (rightEar && rightEar.score > 0.4);

    if (faceVisible) {
        // Clear face detection → facing camera (high confidence)
        isFacingCamera = true;
        confidence = Math.min(1.0, confidence + 0.2);
    } else if (earVisible && upperBodyVisible) {
        // Ears visible but no face → likely side profile or slightly angled
        // Check shoulder/hip ratio for better guess
        if (leftShoulder && rightShoulder && leftHip && rightHip) {
            const shoulderWidth = Math.abs(rightShoulder.x - leftShoulder.x);
            const hipWidth = Math.abs(rightHip.x - leftHip.x);
            
            if (shoulderWidth > hipWidth * 1.15) {
                isFacingCamera = true;
                confidence = Math.min(0.7, confidence);
            } else {
                isBackToCamera = true;
                confidence = Math.min(0.6, confidence);
            }
        }
    } else if (upperBodyVisible) {
        // No face, no ears, but upper body visible → check geometry
        if (leftShoulder && rightShoulder && leftHip && rightHip) {
            const shoulderWidth = Math.abs(rightShoulder.x - leftShoulder.x);
            const hipWidth = Math.abs(rightHip.x - leftHip.x);
            
            // When facing camera, shoulders are typically wider than hips
            // When back to camera, shoulders and hips are similar width
            if (shoulderWidth > hipWidth * 1.1) {
                isFacingCamera = true;
                confidence = Math.min(0.75, confidence);
            } else {
                isBackToCamera = true;
                confidence = Math.min(0.65, confidence);
            }
        } else if (leftShoulder && rightShoulder) {
            // Only shoulders visible - default to back (wings on back)
            isBackToCamera = true;
            confidence = Math.min(0.6, confidence);
        }
    }

    return {
        isFacingCamera,
        isBackToCamera,
        confidence,
    };
}

/**
 * Combine orientation estimates from multiple sources with temporal smoothing
 * 
 * @param {Array<Object>} orientationHistory - Array of recent orientation estimates
 * @param {number} requiredConsistency - Number of consistent frames needed (default: 3)
 * @returns {{isFacingCamera: boolean, isBackToCamera: boolean, confidence: number}}
 */
export function smoothOrientation(orientationHistory, requiredConsistency = 3) {
    if (!orientationHistory || orientationHistory.length === 0) {
        return { isFacingCamera: false, isBackToCamera: false, confidence: 0 };
    }

    // Count recent orientations
    let facingCount = 0;
    let backCount = 0;
    let totalConfidence = 0;

    const recentHistory = orientationHistory.slice(-requiredConsistency);

    for (const orientation of recentHistory) {
        if (orientation.isFacingCamera) facingCount++;
        if (orientation.isBackToCamera) backCount++;
        totalConfidence += orientation.confidence || 0.5;
    }

    const avgConfidence = totalConfidence / recentHistory.length;

    // Require consistent detection across multiple frames to avoid jitter
    const isFacingCamera = facingCount >= requiredConsistency - 1;
    const isBackToCamera = backCount >= requiredConsistency - 1 && !isFacingCamera;

    return {
        isFacingCamera,
        isBackToCamera,
        confidence: avgConfidence,
    };
}


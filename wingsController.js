/**
 * WingsController
 * 
 * Manages orientation-aware positioning and behavior of Gaussian Splatting wings
 * based on person segmentation and orientation detection.
 */

import * as THREE from 'three';

/**
 * WingsController class
 * Handles positioning, rotation, and scaling of wings based on orientation
 */
export class WingsController {
    constructor(wingsGroup, leftWing, rightWing, config = {}) {
        this.wingsGroup = wingsGroup;
        this.leftWing = leftWing;
        this.rightWing = rightWing;

        // Configuration
        this.config = {
            // Base offsets (in 3D space)
            offsetX: config.offsetX || 0,
            offsetY: config.offsetY || -0.5,
            offsetZ: config.offsetZ || -5.0,
            
            // Depth adjustments based on orientation
            depthWhenFacingCamera: config.depthWhenFacingCamera || -5.5, // Further back
            depthWhenBackToCamera: config.depthWhenBackToCamera || -4.5, // Closer to camera
            
            // Scale adjustments
            baseScale: config.baseScale || 1.8,
            scaleMultiplier: config.scaleMultiplier || 1.0,
            
            // Rotation adjustments
            rotationSmoothing: config.rotationSmoothing || 0.6,
            positionSmoothing: config.positionSmoothing || 0.6,
            
            // Transition smoothing
            orientationTransitionFrames: config.orientationTransitionFrames || 10,
        };

        // State
        this.currentOrientation = {
            isFacingCamera: false,
            isBackToCamera: false,
        };
        this.orientationTransition = 0.0; // 0.0 = back to camera, 1.0 = facing camera
        this.smoothedPosition = { x: 0, y: 0, z: 0 };
        this.lastShoulderPosition = null;
    }

    /**
     * Update wings position and orientation based on pose keypoints and segmentation
     * @param {Object} params - Update parameters
     * @param {Array} params.keypoints - Pose keypoints
     * @param {Object} params.orientation - Orientation info {isFacingCamera, isBackToCamera, confidence}
     * @param {HTMLCanvasElement} params.segmentationMask - Segmentation mask (optional, for future occlusion)
     * @param {number} params.videoWidth - Video width for coordinate normalization
     * @param {number} params.videoHeight - Video height for coordinate normalization
     * @param {boolean} params.cameraMode - Camera mode ('user' or 'environment')
     */
    update({
        keypoints,
        orientation,
        segmentationMask,
        videoWidth,
        videoHeight,
        cameraMode = 'environment',
    }) {
        if (!keypoints || keypoints.length === 0) {
            // Hide wings if no person detected
            this.setWingsVisible(false);
            return;
        }

        // Update orientation state with smoothing
        this.updateOrientation(orientation);

        // Find shoulder keypoints
        const leftShoulder = keypoints.find(kp => kp.name === 'left_shoulder');
        const rightShoulder = keypoints.find(kp => kp.name === 'right_shoulder');

        if (!leftShoulder || !rightShoulder || 
            leftShoulder.score < 0.4 || rightShoulder.score < 0.4) {
            this.setWingsVisible(false);
            return;
        }

        // Calculate average shoulder position
        const avgShoulderX = (leftShoulder.x + rightShoulder.x) / 2;
        const avgShoulderY = (leftShoulder.y + rightShoulder.y) / 2;

        // Update position based on shoulders and orientation
        this.updateWingsPosition(avgShoulderX, avgShoulderY, videoWidth, videoHeight, cameraMode);

        // Update individual wing positions
        this.updateIndividualWings(leftShoulder, rightShoulder, videoWidth, videoHeight, cameraMode);

        // Show wings
        this.setWingsVisible(true);
    }

    /**
     * Update orientation with temporal smoothing to avoid jitter
     */
    updateOrientation(orientation) {
        if (!orientation) return;

        // Gradually transition between orientations
        const targetTransition = orientation.isFacingCamera ? 1.0 : 
                                orientation.isBackToCamera ? 0.0 : 
                                this.orientationTransition; // Maintain current if unclear

        const transitionDelta = (targetTransition - this.orientationTransition) / 
                               this.config.orientationTransitionFrames;
        
        this.orientationTransition = THREE.MathUtils.clamp(
            this.orientationTransition + transitionDelta,
            0.0,
            1.0
        );

        // Update current orientation based on transition
        this.currentOrientation = {
            isFacingCamera: this.orientationTransition > 0.5,
            isBackToCamera: this.orientationTransition < 0.5,
        };
    }

    /**
     * Update wings group position based on shoulder position and orientation
     */
    updateWingsPosition(avgShoulderX, avgShoulderY, videoWidth, videoHeight, cameraMode) {
        // Normalize coordinates to [-1, 1] range
        const normX = (coord, dim) => (coord / dim) * 2 - 1;
        const normY = (coord, dim) => -(coord / dim) * 2 + 1;

        let targetX = normX(avgShoulderX, videoWidth);
        let targetY = normY(avgShoulderY, videoHeight);

        // Flip X for front camera
        if (cameraMode === 'user') {
            targetX = -targetX;
        }

        // Adjust depth based on orientation
        // When facing camera: wings appear further back (behind person)
        // When back to camera: wings appear closer (on person's back)
        const depthBlend = this.orientationTransition;
        const targetZ = this.config.depthWhenBackToCamera * (1 - depthBlend) + 
                       this.config.depthWhenFacingCamera * depthBlend;

        // Apply vertical offset
        targetY += this.config.offsetY;

        // Smooth position
        this.smoothedPosition.x += (targetX - this.smoothedPosition.x) * this.config.positionSmoothing;
        this.smoothedPosition.y += (targetY - this.smoothedPosition.y) * this.config.positionSmoothing;
        this.smoothedPosition.z += (targetZ - this.smoothedPosition.z) * this.config.positionSmoothing;

        this.wingsGroup.position.set(
            this.smoothedPosition.x,
            this.smoothedPosition.y,
            this.smoothedPosition.z
        );
    }

    /**
     * Update individual wing positions and rotations
     */
    updateIndividualWings(leftShoulder, rightShoulder, videoWidth, videoHeight, cameraMode) {
        const normX = (coord, dim) => (coord / dim) * 2 - 1;

        const normXLeft = normX(leftShoulder.x, videoWidth);
        const normXRight = normX(rightShoulder.x, videoWidth);

        let sxL = normXLeft;
        let sxR = normXRight;
        if (cameraMode === 'user') {
            sxL = -sxL;
            sxR = -sxR;
        }

        // Calculate horizontal offset based on shoulder width
        const normalizedShoulderDistance = Math.abs(sxR - sxL);
        const wingRootOffset = Math.max(
            (normalizedShoulderDistance / 2.0) * 0.55,
            0.25
        );

        // Position individual wings
        this.leftWing.position.set(wingRootOffset, 0, 0);
        this.rightWing.position.set(-wingRootOffset, 0, 0);

        // Apply scale
        const finalScale = this.config.baseScale * this.config.scaleMultiplier;
        this.leftWing.scale.set(finalScale, finalScale, finalScale * 1.5);
        this.rightWing.scale.set(finalScale, finalScale, finalScale * 1.5);

        // Calculate rotation based on shoulder tilt
        const yDiff = leftShoulder.y - rightShoulder.y;
        const MAX_X_ROTATION = Math.PI / 6;
        const Y_DIFFERENCE_SENSITIVITY = 150;
        let targetRotX = (yDiff / Y_DIFFERENCE_SENSITIVITY) * MAX_X_ROTATION;
        targetRotX = THREE.MathUtils.clamp(targetRotX, -MAX_X_ROTATION, MAX_X_ROTATION);

        if (cameraMode === 'user') {
            targetRotX = -targetRotX;
        }

        // Smooth rotation
        this.wingsGroup.rotation.x += (targetRotX - this.wingsGroup.rotation.x) * this.config.rotationSmoothing;

        // Base wing rotations
        const baseRotX = -Math.PI * 0.2;
        const baseRotY = Math.PI;
        const SPLAY_ANGLE = Math.PI / 12;

        this.leftWing.rotation.set(baseRotX, baseRotY, Math.PI + SPLAY_ANGLE);
        this.rightWing.rotation.set(baseRotX, baseRotY, -Math.PI - SPLAY_ANGLE);
    }

    /**
     * Set wings visibility
     */
    setWingsVisible(visible) {
        if (this.wingsGroup) this.wingsGroup.visible = visible;
        if (this.leftWing) this.leftWing.visible = visible;
        if (this.rightWing) this.rightWing.visible = visible;
    }

    /**
     * Get current orientation state
     */
    getOrientation() {
        return { ...this.currentOrientation, transition: this.orientationTransition };
    }

    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
    }
}


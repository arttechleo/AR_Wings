/**
 * Occlusion Compositor
 * 
 * Handles shader-based occlusion using segmentation masks.
 * Uses a post-processing approach to composite wings with person segmentation.
 */

import * as THREE from 'three';

/**
 * Create an occlusion compositing render target and shader
 */
export class OcclusionCompositor {
    constructor(renderer, width, height) {
        this.renderer = renderer;
        this.width = width;
        this.height = height;

        // Create render targets
        // Wings-only render target (without video background)
        this.wingsRenderTarget = new THREE.WebGLRenderTarget(width, height, {
            format: THREE.RGBAFormat,
            type: THREE.UnsignedByteType,
        });
        
        // Full scene render target (for fallback)
        this.sceneRenderTarget = new THREE.WebGLRenderTarget(width, height, {
            format: THREE.RGBAFormat,
            type: THREE.UnsignedByteType,
        });

        // Create compositing material with shader
        this.compositeMaterial = this.createCompositeMaterial();

        // Full-screen quad for compositing
        const quadGeometry = new THREE.PlaneGeometry(2, 2);
        this.compositeMesh = new THREE.Mesh(quadGeometry, this.compositeMaterial);
        this.compositeScene = new THREE.Scene();
        this.compositeScene.add(this.compositeMesh);
        this.compositeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    }

    /**
     * Create shader material for occlusion compositing
     * Properly handles: Video background + Person (from mask) + Wings (occluded by person)
     */
    createCompositeMaterial() {
        return new THREE.ShaderMaterial({
            uniforms: {
                wingsTexture: { value: null },
                videoTexture: { value: null },
                maskTexture: { value: null },
                maskThreshold: { value: 0.3 },
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D wingsTexture;
                uniform sampler2D videoTexture;
                uniform sampler2D maskTexture;
                uniform float maskThreshold;
                varying vec2 vUv;

                void main() {
                    vec4 wingsColor = texture2D(wingsTexture, vUv);
                    vec4 videoColor = texture2D(videoTexture, vec2(vUv.x, 1.0 - vUv.y)); // Fix video Y coordinate
                    vec4 maskColor = texture2D(maskTexture, vUv);
                    
                    // Extract mask value (white = person, black = background)
                    float maskValue = max(maskColor.r, max(maskColor.g, maskColor.a));
                    float isPerson = step(maskThreshold, maskValue);
                    
                    // Goal: Wings visible but occluded by person
                    // Where person is (isPerson = 1.0): show video (person occludes wings)
                    // Where no person (isPerson = 0.0): show wings composited over video
                    vec3 background = videoColor.rgb;
                    vec3 wingsOverBg = mix(background, wingsColor.rgb, wingsColor.a);
                    vec3 result = mix(wingsOverBg, videoColor.rgb, isPerson);
                    
                    gl_FragColor = vec4(result, 1.0);
                }
            `,
            transparent: false, // No transparency needed, we're doing proper compositing
        });
    }

    /**
     * Update mask texture
     */
    updateMaskTexture(maskTexture) {
        if (this.compositeMaterial) {
            this.compositeMaterial.uniforms.maskTexture.value = maskTexture;
        }
    }

    /**
     * Render scene with occlusion compositing
     * @param {THREE.Scene} scene - The 3D scene (contains video plane and wings)
     * @param {THREE.Camera} camera - The camera
     * @param {THREE.Texture} maskTexture - Segmentation mask texture
     * @param {THREE.Texture} videoTexture - Video background texture
     * @param {THREE.Object3D} wingsGroup - Wings group to render separately
     * @param {THREE.Object3D} videoPlane - Video background plane (to exclude from wings render)
     */
    render(scene, camera, maskTexture, videoTexture = null, wingsGroup = null, videoPlane = null) {
        if (!maskTexture || !videoTexture) {
            // No mask or video - render normally
            this.renderer.render(scene, camera);
            return;
        }

        // Update mask and video textures
        this.updateMaskTexture(maskTexture);
        if (this.compositeMaterial) {
            this.compositeMaterial.uniforms.videoTexture.value = videoTexture;
        }

        const oldRenderTarget = this.renderer.getRenderTarget();
        const oldAutoClear = this.renderer.autoClear;

        // Ensure wings are visible before rendering
        let wingsGroupWasVisible = true;
        if (wingsGroup) {
            wingsGroupWasVisible = wingsGroup.visible;
            wingsGroup.visible = true; // Ensure wings are visible for rendering
        }

        // Render wings only (hide video plane temporarily)
        let videoPlaneWasVisible = true;
        if (videoPlane) {
            videoPlaneWasVisible = videoPlane.visible;
            videoPlane.visible = false; // Hide video for wings-only render
        }

        // Render wings to texture
        this.renderer.autoClear = true;
        this.renderer.setRenderTarget(this.wingsRenderTarget);
        this.renderer.clear();
        this.renderer.render(scene, camera);
        
        // Restore visibility states
        if (videoPlane) {
            videoPlane.visible = videoPlaneWasVisible;
        }
        if (wingsGroup) {
            wingsGroup.visible = wingsGroupWasVisible; // Restore original state
        }
        
        // Update shader with wings-only texture
        if (this.compositeMaterial) {
            this.compositeMaterial.uniforms.wingsTexture.value = this.wingsRenderTarget.texture;
        }

        // Render composite to screen
        this.renderer.setRenderTarget(oldRenderTarget);
        this.renderer.autoClear = oldAutoClear;
        this.renderer.clear();
        this.renderer.render(this.compositeScene, this.compositeCamera);
    }

    /**
     * Resize render targets
     */
    resize(width, height) {
        this.width = width;
        this.height = height;
        this.sceneRenderTarget.setSize(width, height);
        this.wingsRenderTarget.setSize(width, height);
    }

    /**
     * Dispose resources
     */
    dispose() {
        if (this.sceneRenderTarget) {
            this.sceneRenderTarget.dispose();
        }
        if (this.wingsRenderTarget) {
            this.wingsRenderTarget.dispose();
        }
        if (this.compositeMaterial) {
            this.compositeMaterial.dispose();
        }
    }
}


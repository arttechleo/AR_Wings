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

        // Create render target for scene rendering
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
     */
    createCompositeMaterial() {
        return new THREE.ShaderMaterial({
            uniforms: {
                sceneTexture: { value: null },
                maskTexture: { value: null },
                maskThreshold: { value: 0.5 },
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D sceneTexture;
                uniform sampler2D maskTexture;
                uniform float maskThreshold;
                varying vec2 vUv;

                void main() {
                    vec4 sceneColor = texture2D(sceneTexture, vUv);
                    vec4 maskColor = texture2D(maskTexture, vUv);
                    
                    // Extract alpha from mask (white = person, black = background)
                    float maskValue = maskColor.r; // Assuming grayscale mask
                    
                    // If mask indicates person (above threshold), blend appropriately
                    // For occlusion: if person is present, reduce wing opacity in that area
                    float alpha = sceneColor.a;
                    
                    if (maskValue > maskThreshold) {
                        // Person is here - reduce wing visibility (occlude)
                        // Keep video background visible
                        alpha = mix(sceneColor.a, 0.0, maskValue);
                    }
                    
                    gl_FragColor = vec4(sceneColor.rgb, alpha);
                }
            `,
            transparent: true,
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
     */
    render(scene, camera, maskTexture) {
        if (!maskTexture) {
            // No mask - render normally
            this.renderer.render(scene, camera);
            return;
        }

        // Update mask texture
        this.updateMaskTexture(maskTexture);

        // Render scene to render target
        const oldRenderTarget = this.renderer.getRenderTarget();
        this.renderer.setRenderTarget(this.sceneRenderTarget);
        this.renderer.render(scene, camera);
        this.renderer.setRenderTarget(oldRenderTarget);

        // Update composite shader with scene texture
        this.compositeMaterial.uniforms.sceneTexture.value = this.sceneRenderTarget.texture;

        // Render composite to screen
        this.renderer.render(this.compositeScene, this.compositeCamera);
    }

    /**
     * Resize render targets
     */
    resize(width, height) {
        this.width = width;
        this.height = height;
        this.sceneRenderTarget.setSize(width, height);
    }

    /**
     * Dispose resources
     */
    dispose() {
        if (this.sceneRenderTarget) {
            this.sceneRenderTarget.dispose();
        }
        if (this.compositeMaterial) {
            this.compositeMaterial.dispose();
        }
    }
}


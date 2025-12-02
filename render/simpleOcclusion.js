/**
 * Simple Occlusion Renderer
 * 
 * Renders wings with depth-based occlusion using segmentation mask.
 * Wings are visible but occluded where the person's body is detected.
 */

import * as THREE from 'three';

/**
 * Simple occlusion renderer that uses depth and mask for proper compositing
 */
export class SimpleOcclusionRenderer {
    constructor(renderer, width, height) {
        this.renderer = renderer;
        this.width = width;
        this.height = height;

        // Create render target for wings (without video background)
        this.wingsRenderTarget = new THREE.WebGLRenderTarget(width, height, {
            format: THREE.RGBAFormat,
            type: THREE.UnsignedByteType,
        });

        // Create compositing material
        this.compositeMaterial = this.createCompositeMaterial();

        // Full-screen quad
        const quadGeometry = new THREE.PlaneGeometry(2, 2);
        this.compositeMesh = new THREE.Mesh(quadGeometry, this.compositeMaterial);
        this.compositeScene = new THREE.Scene();
        this.compositeScene.add(this.compositeMesh);
        this.compositeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    }

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
                    vec4 videoColor = texture2D(videoTexture, vUv);
                    vec4 maskColor = texture2D(maskTexture, vUv);
                    
                    // Extract mask value (white = person)
                    float maskValue = max(maskColor.r, max(maskColor.g, maskColor.a));
                    float isPerson = step(maskThreshold, maskValue);
                    
                    // Where person is: show video (person occludes wings)
                    // Where no person: show wings composited over video
                    vec3 result = mix(
                        mix(videoColor.rgb, wingsColor.rgb, wingsColor.a), // Background: wings over video
                        videoColor.rgb, // Person: just video
                        isPerson
                    );
                    
                    gl_FragColor = vec4(result, 1.0);
                }
            `,
            transparent: false,
        });
    }

    /**
     * Render scene with occlusion
     * @param {THREE.Scene} scene - Full scene
     * @param {THREE.Camera} camera - Camera
     * @param {THREE.Object3D} videoPlane - Video background plane (to hide for wings-only render)
     * @param {THREE.Object3D} wingsGroup - Wings group
     * @param {THREE.Texture} maskTexture - Segmentation mask
     * @param {THREE.Texture} videoTexture - Video texture
     */
    render(scene, camera, videoPlane, wingsGroup, maskTexture, videoTexture) {
        if (!maskTexture || !videoTexture || !wingsGroup) {
            // Fallback: render normally
            this.renderer.render(scene, camera);
            return;
        }

        // Update textures
        if (this.compositeMaterial) {
            this.compositeMaterial.uniforms.maskTexture.value = maskTexture;
            this.compositeMaterial.uniforms.videoTexture.value = videoTexture;
        }

        // Render wings only (hide video plane)
        const oldRenderTarget = this.renderer.getRenderTarget();
        const videoVisible = videoPlane ? videoPlane.visible : true;
        
        if (videoPlane) {
            videoPlane.visible = false;
        }

        this.renderer.setRenderTarget(this.wingsRenderTarget);
        this.renderer.clear();
        this.renderer.render(scene, camera);

        if (videoPlane) {
            videoPlane.visible = videoVisible;
        }

        // Update shader
        if (this.compositeMaterial) {
            this.compositeMaterial.uniforms.wingsTexture.value = this.wingsRenderTarget.texture;
        }

        // Render composite to screen
        this.renderer.setRenderTarget(oldRenderTarget);
        this.renderer.clear();
        this.renderer.render(this.compositeScene, this.compositeCamera);
    }

    resize(width, height) {
        this.width = width;
        this.height = height;
        this.wingsRenderTarget.setSize(width, height);
    }

    dispose() {
        if (this.wingsRenderTarget) {
            this.wingsRenderTarget.dispose();
        }
        if (this.compositeMaterial) {
            this.compositeMaterial.dispose();
        }
    }
}


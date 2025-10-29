import * as THREE from 'three';

// Creates a depth-only writer using the body mask.
// Pixels where mask >= threshold will write depth and thus occlude wings behind.
export class OcclusionMask {
  constructor({ scene, camera, depthZ = -9.8, threshold = 0.5, debug }) {
    this.scene = scene;
    this.camera = camera;
    this.threshold = threshold;
    this.debug = debug;

    this.maskCanvas = document.createElement('canvas');
    this.maskTex = new THREE.Texture(this.maskCanvas);
    this.maskTex.minFilter = THREE.LinearFilter;
    this.maskTex.magFilter = THREE.LinearFilter;
    this.maskTex.format = THREE.RGBAFormat;

    const geo = new THREE.PlaneGeometry(1, 1);
    geo.scale(1, -1, 1);

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uMask: { value: this.maskTex },
        uThreshold: { value: threshold },
        uFlipX: { value: 0 }, // for selfie mirroring
      },
      vertexShader: /* glsl */`
        varying vec2 vUv;
        uniform float uFlipX;
        void main() {
          vUv = uv;
          if (uFlipX > 0.5) vUv = vec2(1.0 - vUv.x, vUv.y);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */`
        varying vec2 vUv;
        uniform sampler2D uMask;
        uniform float uThreshold;
        void main(){
          float a = texture2D(uMask, vUv).a; // use alpha of binary mask
          if (a < uThreshold) discard; // only keep person pixels (write depth)
          gl_FragColor = vec4(0.0, 0.0, 0.0, a);
        }
             `,
      depthWrite: true,
      depthTest: true,
      transparent: true,
      colorWrite: false, // depth-only
    });

    // Match video plane size
    const fovRad = THREE.MathUtils.degToRad(camera.fov);
    const aspect = this._getAspect();
    const planeH = Math.abs(2 * Math.tan(fovRad / 2) * depthZ);
    const planeW = planeH * aspect;
     this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.position.z = depthZ;
    this.mesh.scale.set(planeW, planeH, 1);
    this.mesh.renderOrder = 1; // after video plane

    scene.add(this.mesh);
  }

  _getAspect() {
    const el = document.getElementById('three-container');
    const rect = el.getBoundingClientRect();
    return rect.width / rect.height;
  }

  updateMask(canvas, facingMode) {
    // Resize internal canvas on demand
    if (this.maskCanvas.width !== canvas.width || this.maskCanvas.height !== canvas.height) {
      this.maskCanvas.width = canvas.width;
      this.maskCanvas.height = canvas.height;
    }
    const ctx = this.maskCanvas.getContext('2d');
    // draw directly (already binary mask with alpha)
    ctx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
    ctx.drawImage(canvas, 0, 0);

    this.maskTex.needsUpdate = true;
    this.material.uniforms.uFlipX.value = facingMode === 'user' ? 1.0 : 0.0;
  }
}
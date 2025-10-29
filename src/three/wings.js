import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
// Optional ksplat fallback
import { SplatMesh, SparkRenderer } from '@sparkjsdev/spark';

const SPLAT_LEFT = new URL('../../assets/leftwing.ksplat', import.meta.url).href;
const SPLAT_RIGHT = new URL('../../assets/rightwing.ksplat', import.meta.url).href;
const PLY_LEFT = new URL('../../assets/leftwing.ply', import.meta.url).href;
const PLY_RIGHT = new URL('../../assets/rightwing.ply', import.meta.url).href;

// Tuning
const WING_VERTICAL_SHIFT = 0.5;
const SHOULDER_PIVOT_MULTIPLIER = 0.55;
const MIN_HORIZONTAL_OFFSET = 0.25;
const MAX_X_ROTATION = Math.PI / 6;
const Y_DIFF_SENS = 150;
const BASE_SCALE = 1.8;
const SPLAY_ANGLE = Math.PI / 12;

export class WingsRig {
  constructor({ scene, debug }) {
    this.scene = scene;
    this.debug = debug;
    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.left = null;
    this.right = null;
    this.lastAnchor = null; // cache of last shoulders
    this.currentScale = BASE_SCALE;
    this.currentOffset = 0;

    // Depth placement (same region as video plane but closer to camera)
    this.group.position.z = -9.9;
  }

  async loadAssets() {
    // Try PLY first
    try {
      await this._loadPLY();
      this.debug.updateAssetStatus('.PLY wings loaded');
      return;
    } catch (e) {
      this.debug.log('warning', `PLY load failed (${e?.message}). Falling back to ksplat/boxes.`);
    }
 // Try ksplat
    try {
      await this._loadSplat();
      this.debug.updateAssetStatus('ksplat wings loaded');
      return;
    } catch (e) {}

    // Fallback cube wings
    this._loadBoxes();
    this.debug.updateAssetStatus('Box wings (fallback)');
  }

  async _loadPLY() {
    const loader = new PLYLoader();
    const [geoL, geoR] = await Promise.all([
      new Promise((res, rej) => loader.load(PLY_LEFT, res, undefined, rej)),
      new Promise((res, rej) => loader.load(PLY_RIGHT, res, undefined, rej)),
    ]);

    geoL.computeVertexNormals();
    geoR.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9, metalness: 0.05 });

    this.left = new THREE.Mesh(geoL, mat);
    this.right = new THREE.Mesh(geoR, mat.clone());
    this.left.castShadow = this.right.castShadow = false;

    this.left.visible = this.right.visible = false;
    this.left.renderOrder = this.right.renderOrder = 2;

    this.group.add(this.left);
    this.group.add(this.right);
  }
 async _loadSplat() {
    // Ensure Spark renderer is attached (no-op if already)
    const fakeRenderer = this._findRenderer();
    if (fakeRenderer) new SparkRenderer(fakeRenderer);

    this.left = new SplatMesh({ url: SPLAT_LEFT, fileType: 'ksplat' });
    this.right = new SplatMesh({ url: SPLAT_RIGHT, fileType: 'ksplat' });
    this.left.visible = this.right.visible = false;
    this.left.renderOrder = this.right.renderOrder = 2;
    this.group.add(this.left);
    this.group.add(this.right);
  }

  _findRenderer() {
    // crude way to find a WebGLRenderer from the scene graph owner document
    const canvases = document.querySelectorAll('canvas');
    for (const c of canvases) {
      if (c?.getContext && c.getContext('webgl')) return { domElement: c, setSize() {}, setPixelRatio() {}, dispose() {} };
    }
    return null;
  }

  _loadBoxes() {
    const geo = new THREE.BoxGeometry(0.5, 0.8, 0.08);
    const mat = new THREE.MeshBasicMaterial({ color: 0x00ccff, transparent: true, opacity: 0.8 });
    this.left = new THREE.Mesh(geo, mat);
    this.right = new THREE.Mesh(geo.clone(), mat.clone());
    this.left.visible = this.right.visible = false;
    this.left.renderOrder = this.right.renderOrder = 2;
    this.group.add(this.left);
    this.group.add(this.right);
  }

  setVisible(v) {
    if (!this.left || !this.right) return;
    this.left.visible = v;
    this.right.visible = v;
  }
   hasLastAnchor() { return !!this.lastAnchor; }

  updateFromShoulders({ left, right, videoWidth, videoHeight, facingMode }) {
    this.lastAnchor = { left, right };

    // Center point of shoulders, normalized to [-1, 1]
    const avgX = (left.x + right.x) / 2;
    const avgY = (left.y + right.y) / 2;
    const normX = (avgX / videoWidth) * 2 - 1;
    const normY = -(avgY / videoHeight) * 2 + 1;

    // Position the group (depth fixed)
    let x = normX;
    let y = normY - WING_VERTICAL_SHIFT;
    if (facingMode === 'user') x = -x; // mirror correction
    this.group.position.x += (x - this.group.position.x) * 0.6;
    this.group.position.y += (y - this.group.position.y) * 0.6;

    // Horizontal offset between wings from shoulder span
    const nL = (left.x / videoWidth) * 2 - 1;
    const nR = (right.x / videoWidth) * 2 - 1;
    let sxL = facingMode === 'user' ? -nL : nL;
    let sxR = facingMode === 'user' ? -nR : nR;
    const span = Math.abs(sxR - sxL);
    const offset = Math.max((span / 2) * SHOULDER_PIVOT_MULTIPLIER, MIN_HORIZONTAL_OFFSET);
    this.currentOffset = offset;

    // Pitch from shoulder slope
    const yDiff = left.y - right.y;
    let rotX = (yDiff / Y_DIFF_SENS) * MAX_X_ROTATION;
    rotX = THREE.MathUtils.clamp(rotX, -MAX_X_ROTATION, MAX_X_ROTATION);
    if (facingMode === 'user') rotX = -rotX;
    this.group.rotation.x += (rotX - this.group.rotation.x) * 0.6;

    // Scale vs display size
    const aspect = videoWidth / videoHeight;
    let scaleAdj = aspect < 1 ? 0.85 : aspect > 1.7 ? 1.1 : 1.0;
    const screenHFactor = window.innerHeight / 800;
    this.currentScale = BASE_SCALE * scaleAdj * Math.min(1.0, screenHFactor);

    // Apply to each wing
    this._positionSingle(this.left, 'left');
    this._positionSingle(this.right, 'right');
  }
    _positionSingle(wing, side) {
    if (!wing) return;
    const FIXED_SCALE = 1.0;
    const x = (side === 'left' ? this.currentOffset : -this.currentOffset) * FIXED_SCALE;
    wing.position.set(x, 0, 0);

    const scaleZFactor = 1.5; // a bit thicker for PLY normals look
    wing.scale.set(this.currentScale, this.currentScale, this.currentScale * scaleZFactor);

    const baseRotX = -Math.PI * 0.2;
    const baseRotY = Math.PI; // face camera
    const rotZ = side === 'left' ? Math.PI + SPLAY_ANGLE : -Math.PI - SPLAY_ANGLE;
    wing.rotation.set(baseRotX, baseRotY, rotZ);
  }
}
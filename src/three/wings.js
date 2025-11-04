import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { SplatMesh, SparkRenderer } from '@sparkjsdev/spark';

// Path resolution: from src/three/wings.js, go up two levels to project root, then into assets
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
    this.isSplatDataReady = false; // Flag to track when splats are loaded
    this.loadedCount = 0; // Count loaded splats

    // Depth placement (same region as video plane but closer to camera)
    // Wings should be visible in front of video plane
    this.group.position.z = -9.9;
    this.group.visible = true; // Ensure group is visible
  }

  async loadAssets(renderer) {
    // Try ksplat first (priority) - now using direct loader
    try {
      this.debug.updateAssetStatus('Loading ksplat wings...');
      await this._loadSplat(renderer);
      this.debug.updateAssetStatus('✓ ksplat wings loaded');
      this.debug.log('success', 'Ksplat wings successfully loaded');
      return;
    } catch (e) {
      this.debug.log('error', `ksplat load failed: ${e?.message}. Stack: ${e?.stack}`);
      this.debug.updateAssetStatus(`ksplat failed: ${e?.message}`);
      // Don't return here - fall through to PLY fallback
    }
    
    // Try PLY as fallback
    try {
      this.debug.updateAssetStatus('Loading PLY wings...');
      await this._loadPLY();
      this.debug.updateAssetStatus('✓ PLY wings loaded');
      return;
    } catch (e) {
      this.debug.log('warning', `PLY load failed (${e?.message}). Falling back to boxes.`);
    }

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
  async _loadSplat(renderer) {
    if (!renderer) {
      throw new Error('Renderer required for ksplat loading');
    }
    
    // Initialize SparkRenderer with actual Three.js renderer (only once)
    // This is critical - must be done before creating SplatMesh
    try {
      if (!renderer._sparkRendererInitialized) {
        new SparkRenderer(renderer);
        renderer._sparkRendererInitialized = true;
        this.debug.log('info', 'SparkRenderer initialized');
      }
    } catch (e) {
      this.debug.log('warning', `SparkRenderer init warning: ${e?.message}`);
    }

    // Verify files exist first
    // Log the resolved paths for debugging
    this.debug.log('info', `Resolved paths - Left: ${SPLAT_LEFT}, Right: ${SPLAT_RIGHT}`);
    
    try {
      const [leftRes, rightRes] = await Promise.all([
        fetch(SPLAT_LEFT, { method: 'HEAD' }).catch((e) => {
          this.debug.log('error', `Left wing HEAD request failed: ${e.message}`);
          return null;
        }),
        fetch(SPLAT_RIGHT, { method: 'HEAD' }).catch((e) => {
          this.debug.log('error', `Right wing HEAD request failed: ${e.message}`);
          return null;
        })
      ]);
      
      if (!leftRes || !leftRes.ok) {
        throw new Error(`Left wing file not found at: ${SPLAT_LEFT} (status: ${leftRes?.status || 'network error'})`);
      }
      if (!rightRes || !rightRes.ok) {
        throw new Error(`Right wing file not found at: ${SPLAT_RIGHT} (status: ${rightRes?.status || 'network error'})`);
      }
      
      this.debug.log('info', 'Ksplat files verified, starting load...');
    } catch (e) {
      throw new Error(`File verification failed: ${e.message}`);
    }

    // Create SplatMesh instances using the same pattern that worked before
    // The onLoad callback pattern is critical for @sparkjsdev/spark
    this.debug.log('info', 'Creating SplatMesh instances...');
    
    this.left = new SplatMesh({ 
      url: SPLAT_LEFT, 
      fileType: 'ksplat',
      onLoad: (mesh) => {
        if (mesh) {
          mesh.scale.set(1, 1, -1); // Same scale as previous working version
        }
        this._checkSplatDataReady();
        this.debug.log('success', 'Left wing ksplat loaded');
      }
    });
    
    this.right = new SplatMesh({ 
      url: SPLAT_RIGHT, 
      fileType: 'ksplat',
      onLoad: (mesh) => {
        if (mesh) {
          mesh.scale.set(1, 1, -1); // Same scale as previous working version
        }
        this._checkSplatDataReady();
        this.debug.log('success', 'Right wing ksplat loaded');
      }
    });
    
    this.left.visible = this.right.visible = false;
    this.left.renderOrder = 1; // Same as previous working version
    this.right.renderOrder = 1; // Same as previous working version
    
    // Ensure wings are added to group
    this.group.add(this.left);
    this.group.add(this.right);
    
    // Debug: log when wings are added
    this.debug.log('info', `Wings added to group. Left: ${!!this.left}, Right: ${!!this.right}, Group in scene: ${this.scene.children.includes(this.group)}`);
    
    // Wait for both to load (with timeout)
    // The onLoad callbacks will set isSplatDataReady
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        if (this.isSplatDataReady) {
          resolve();
        } else {
          reject(new Error(`Ksplat load timeout - Left: ${!!this.left}, Right: ${!!this.right}, Ready: ${this.isSplatDataReady}`));
        }
      }, 20000);
      
      // Check if already loaded
      if (this.isSplatDataReady) {
        clearTimeout(timeout);
        resolve();
        return;
      }
      
      // Poll for ready status
      const pollInterval = setInterval(() => {
        if (this.isSplatDataReady) {
          clearInterval(pollInterval);
          clearTimeout(timeout);
          resolve();
        }
      }, 200);
    });
  }

  _checkSplatDataReady() {
    this.loadedCount++;
    if (this.loadedCount === 2) {
      this.isSplatDataReady = true;
      this.debug.log('success', 'Gaussian Splat data loaded and ready!');
      this.debug.updateAssetStatus('Gaussian Splats active');
      this.loadedCount = 0; // Reset for potential reload
    }
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
    if (!this.left || !this.right) {
      this.debug?.log('warning', `Cannot set visibility - wings not loaded (left: ${!!this.left}, right: ${!!this.right})`);
      return;
    }
    
    // Only log visibility changes to avoid spam
    if (this.left.visible !== v || this.right.visible !== v) {
      this.left.visible = v;
      this.right.visible = v;
      this.group.visible = v; // Ensure group is also visible
      
      if (this.debug) {
        // Only log when visibility changes (not every frame)
        this.debug.log('info', `Wings visibility changed: ${v} | Splats ready: ${this.isSplatDataReady}`);
        // Debug position when making visible
        if (v) {
          this.debug.log('info', `Wings visible - Group pos: (${this.group.position.x.toFixed(2)}, ${this.group.position.y.toFixed(2)}, ${this.group.position.z.toFixed(2)}) | Scale: ${this.currentScale.toFixed(2)}`);
        }
      }
    }
  }
   hasLastAnchor() { return !!this.lastAnchor; }
   isSplatDataReady() { return this.isSplatDataReady; }

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

    // Apply scale - check if it's a SplatMesh (same as previous working version)
    const scaleZFactor = 1.5;
    let finalScaleFactor = (wing instanceof SplatMesh || wing.constructor.name === 'SplatMesh') 
      ? this.currentScale 
      : 1.2;
    wing.scale.set(finalScaleFactor, finalScaleFactor, finalScaleFactor * scaleZFactor);

    const baseRotX = -Math.PI * 0.2;
    const baseRotY = Math.PI; // face camera
    const rotZ = side === 'left' ? Math.PI + SPLAY_ANGLE : -Math.PI - SPLAY_ANGLE;
    wing.rotation.set(baseRotX, baseRotY, rotZ);
  }
}
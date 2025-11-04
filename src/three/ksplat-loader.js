import * as THREE from 'three';

/**
 * Direct Ksplat Loader - parses .ksplat files and creates Three.js meshes
 * This replaces the unreliable @sparkjsdev/spark library
 */
export class KsplatLoader {
  constructor() {
    this.cache = new Map();
  }

  async load(url) {
    if (this.cache.has(url)) {
      return this.cache.get(url);
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to load ${url}: ${response.status}`);
      }

      const arrayBuffer = await response.arrayBuffer();
      const data = new Uint8Array(arrayBuffer);
      
      // Parse ksplat format
      const splat = this.parseKsplat(data);
      
      // Create Three.js mesh from splat data
      const mesh = this.createSplatMesh(splat);
      
      this.cache.set(url, mesh);
      return mesh;
    } catch (error) {
      console.error(`KsplatLoader error loading ${url}:`, error);
      throw error;
    }
  }

  parseKsplat(data) {
    // Try to detect format - check for common headers
    const header = String.fromCharCode(...data.slice(0, Math.min(10, data.length)));
    
    // Check if it's a standard ksplat format
    // Format can vary: might start with "KSP" or have no header
    let offset = 0;
    let splatCount = 0;

    // Try to detect format by checking first bytes
    if (data.length < 4) {
      throw new Error('File too small to be a valid ksplat');
    }

    // Check for magic number or header
    if (String.fromCharCode(data[0], data[1], data[2]) === 'KSP') {
      // Has KSP header
      const version = data[3];
      offset = 4;
      // Read splat count after header
      splatCount = new DataView(data.buffer, offset, 4).getUint32(0, true);
      offset += 4;
    } else {
      // No header - assume raw format, count is first 4 bytes
      splatCount = new DataView(data.buffer, 0, 4).getUint32(0, true);
      offset = 4;
    }

    // Validate splat count (reasonable range)
    if (splatCount === 0 || splatCount > 10000000) {
      // Try alternative: assume it's a PLY-like format or check file size
      // Each splat is typically 44 bytes (position 12 + rotation 16 + scale 12 + color 4)
      const estimatedCount = Math.floor((data.length - offset) / 44);
      if (estimatedCount > 0 && estimatedCount < 10000000) {
        splatCount = estimatedCount;
        console.warn(`Using estimated splat count: ${splatCount}`);
      } else {
        throw new Error(`Invalid splat count: ${splatCount}. File might be corrupted or wrong format.`);
      }
    }

    // Each splat: position (3x float32), rotation (4x float32 quaternion), scale (3x float32), color (4x uint8)
    // Total: 12 + 16 + 12 + 4 = 44 bytes per splat
    const splatSize = 44;
    const expectedSize = offset + splatCount * splatSize;

    if (data.length < expectedSize) {
      throw new Error(`Invalid ksplat file: expected ${expectedSize} bytes, got ${data.length}. Splat count: ${splatCount}`);
    }

    const positions = new Float32Array(splatCount * 3);
    const colors = new Uint8Array(splatCount * 4);
    const scales = new Float32Array(splatCount * 3);
    const rotations = new Float32Array(splatCount * 4);

    let currentOffset = offset;
    for (let i = 0; i < splatCount; i++) {
      const base = i * 3;
      const colorBase = i * 4;
      const rotBase = i * 4;
      const scaleBase = i * 3;

      // Position (3 floats, little-endian)
      positions[base] = new DataView(data.buffer, currentOffset, 4).getFloat32(0, true);
      positions[base + 1] = new DataView(data.buffer, currentOffset + 4, 4).getFloat32(0, true);
      positions[base + 2] = new DataView(data.buffer, currentOffset + 8, 4).getFloat32(0, true);
      currentOffset += 12;

      // Rotation quaternion (4 floats, little-endian)
      rotations[rotBase] = new DataView(data.buffer, currentOffset, 4).getFloat32(0, true);
      rotations[rotBase + 1] = new DataView(data.buffer, currentOffset + 4, 4).getFloat32(0, true);
      rotations[rotBase + 2] = new DataView(data.buffer, currentOffset + 8, 4).getFloat32(0, true);
      rotations[rotBase + 3] = new DataView(data.buffer, currentOffset + 12, 4).getFloat32(0, true);
      currentOffset += 16;

      // Scale (3 floats, little-endian)
      scales[scaleBase] = new DataView(data.buffer, currentOffset, 4).getFloat32(0, true);
      scales[scaleBase + 1] = new DataView(data.buffer, currentOffset + 4, 4).getFloat32(0, true);
      scales[scaleBase + 2] = new DataView(data.buffer, currentOffset + 8, 4).getFloat32(0, true);
      currentOffset += 12;

      // Color (4 uint8, RGBA)
      colors[colorBase] = data[currentOffset];
      colors[colorBase + 1] = data[currentOffset + 1];
      colors[colorBase + 2] = data[currentOffset + 2];
      colors[colorBase + 3] = data[currentOffset + 3] || 255; // Default alpha if missing
      currentOffset += 4;
    }

    return {
      positions,
      colors,
      scales,
      rotations,
      count: splatCount
    };
  }

  createSplatMesh(splat) {
    // Use BufferGeometry with points for Gaussian splatting
    const geometry = new THREE.BufferGeometry();
    
    // Set attributes
    geometry.setAttribute('position', new THREE.BufferAttribute(splat.positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(splat.colors, 4, true));
    geometry.setAttribute('scale', new THREE.BufferAttribute(splat.scales, 3));
    geometry.setAttribute('rotation', new THREE.BufferAttribute(splat.rotations, 4));

    // Custom shader material for Gaussian splatting
    const material = new THREE.ShaderMaterial({
      vertexShader: `
        attribute vec3 position;
        attribute vec4 color;
        attribute vec3 scale;
        attribute vec4 rotation;

        varying vec4 vColor;
        varying vec3 vScale;
        varying vec4 vRotation;

        void main() {
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mvPosition;
          
          // Calculate point size based on distance and scale
          float dist = length(mvPosition.xyz);
          float avgScale = (scale.x + scale.y + scale.z) / 3.0;
          gl_PointSize = 1000.0 * avgScale / dist;
          
          vColor = color / 255.0;
          vScale = scale;
          vRotation = rotation;
        }
      `,
      fragmentShader: `
        varying vec4 vColor;
        varying vec3 vScale;
        varying vec4 vRotation;
        
        vec3 rotateQuaternion(vec3 v, vec4 q) {
          return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
        }
        
        void main() {
          vec2 coord = gl_PointCoord - vec2(0.5);
          
          // Apply rotation to coordinate
          vec3 coord3d = vec3(coord * 2.0, 0.0);
          coord3d = rotateQuaternion(coord3d, vRotation);
          coord = coord3d.xy;
          
          // Apply scale
          coord /= vScale.xy;
          
          // Gaussian falloff
          float dist = length(coord);
          float alpha = exp(-dot(coord, coord) * 2.0);
          
          // Ensure alpha doesn't go to zero too quickly
          alpha = clamp(alpha, 0.0, 1.0);
          
          // Discard if alpha is too low (before color assignment)
          if (alpha < 0.01) discard;
          
          gl_FragColor = vec4(vColor.rgb, vColor.a * alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.NormalBlending,
      vertexColors: true
    });

    const points = new THREE.Points(geometry, material);
    points.frustumCulled = false;
    
    return points;
  }
}


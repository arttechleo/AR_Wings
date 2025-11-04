import * as THREE from 'three';

export function createScene({ video, container, videoPlaneDepth = -10, debug }) {
  // Use viewport dimensions for mobile compatibility
  const width = window.innerWidth || container.clientWidth || 640;
  const height = window.innerHeight || container.clientHeight || 480;

  // Optimize renderer settings for mobile
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  const renderer = new THREE.WebGLRenderer({ 
    alpha: true, 
    antialias: !isMobile, // Disable antialiasing on mobile for performance
    powerPreference: 'high-performance'
  });
  // Lower pixel ratio on mobile for better performance
  const pixelRatio = isMobile ? Math.min(window.devicePixelRatio || 1, 1.2) : Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(pixelRatio);
  renderer.setSize(width, height);
  renderer.setClearColor(0x000000, 0);
  
  // Ensure canvas is properly styled
  renderer.domElement.style.position = 'absolute';
  renderer.domElement.style.top = '0';
  renderer.domElement.style.left = '0';
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  renderer.domElement.style.display = 'block';
  
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const aspect = width / height;
  const camera = new THREE.PerspectiveCamera(65, aspect, 0.01, 100);
  camera.position.set(0, 0, 0);
  scene.add(new THREE.AmbientLight(0xffffff, 1.0));

  // Video plane - optimize for performance
  const tex = new THREE.VideoTexture(video);
  tex.flipY = false;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  // Don't auto-update every frame - we'll control it manually for performance
  tex.generateMipmaps = false; // Disable mipmaps for video (not needed)
  
  const planeGeo = new THREE.PlaneGeometry(1, 1);
  planeGeo.scale(1, -1, 1); // unflip selfie-type frames consistently
  const mat = new THREE.MeshBasicMaterial({ 
    map: tex, 
    side: THREE.DoubleSide, 
    depthTest: false,
    depthWrite: false
  });
  const plane = new THREE.Mesh(planeGeo, mat);

  const fovRad = THREE.MathUtils.degToRad(camera.fov);
  const planeH = Math.abs(2 * Math.tan(fovRad / 2) * Math.abs(videoPlaneDepth));
  const planeW = planeH * aspect;
  plane.scale.set(planeW, planeH, 1);
  plane.position.z = videoPlaneDepth;
  plane.renderOrder = 0;
  scene.add(plane);

  // Force texture update
  tex.needsUpdate = true;

  return { renderer, scene, camera, videoPlane: plane, containerEl: container };
}

export function updateVideoPlaneTexture(plane) {
  if (!plane?.material?.map) return;
  plane.material.map.needsUpdate = true;
}

export function disposeRenderer(renderer, containerEl) {
  if (!renderer) return;
  renderer.dispose();
  if (renderer.domElement && containerEl?.contains(renderer.domElement)) {
    containerEl.removeChild(renderer.domElement);
  }
}
import * as THREE from 'three';

export function createScene({ video, container, videoPlaneDepth = -10, debug }) {
  const containerRect = container.getBoundingClientRect();

  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(containerRect.width, containerRect.height);
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const aspect = containerRect.width / containerRect.height;
  const camera = new THREE.PerspectiveCamera(65, aspect, 0.01, 100);
  camera.position.set(0, 0, 0);
  scene.add(new THREE.AmbientLight(0xffffff, 1.0));

  // Video plane
  const tex = new THREE.VideoTexture(video);
  tex.flipY = false;
  const planeGeo = new THREE.PlaneGeometry(1, 1);
  planeGeo.scale(1, -1, 1); // unflip selfie-type frames consistently
  const mat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, depthTest: false });
  const plane = new THREE.Mesh(planeGeo, mat);

  const fovRad = THREE.MathUtils.degToRad(camera.fov);
  const planeH = Math.abs(2 * Math.tan(fovRad / 2) * videoPlaneDepth);
  const planeW = planeH * aspect;
  plane.scale.set(planeW, planeH, 1);
  plane.position.z = videoPlaneDepth;
  plane.renderOrder = 0;
  scene.add(plane);

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
let facingMode = 'environment';
const video = document.getElementById('video');

export async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play().catch(() => {});
}

export function stopCamera() {
  const s = video.srcObject;
  if (!s) return;
  s.getTracks().forEach(t => t.stop());
  video.srcObject = null;
}

export async function switchCamera() {
  stopCamera();
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  await startCamera();
}

export function getFacingMode() { return facingMode; }
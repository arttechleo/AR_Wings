let facingMode = 'environment';
function getVideoEl() {
  return document.getElementById('video');
}

export async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  const video = getVideoEl();
  if (!video) throw new Error('Video element not found');
  video.srcObject = stream;
  await video.play().catch(() => {});
}

export function stopCamera() {
  const video = getVideoEl();
  if (!video) return;
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
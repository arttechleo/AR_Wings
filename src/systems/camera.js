let facingMode = 'environment';
function getVideoEl() {
  return document.getElementById('video');
}

export async function startCamera() {
  let stream;
  const tryConstraints = async (fm) => navigator.mediaDevices.getUserMedia({
    video: { facingMode: fm, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  try {
    stream = await tryConstraints(facingMode);
  } catch (e) {
    // Fallback to alternate camera if the requested one isn't available
    const alt = facingMode === 'user' ? 'environment' : 'user';
    try { stream = await tryConstraints(alt); facingMode = alt; } catch (e2) { throw e; }
  }
  const video = getVideoEl();
  if (!video) throw new Error('Video element not found');
  // Ensure attributes for autoplay across browsers
  video.setAttribute('muted', 'true');
  video.setAttribute('playsinline', 'true');
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
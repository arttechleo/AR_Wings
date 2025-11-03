let facingMode = 'environment';
function getVideoEl() {
  return document.getElementById('video');
}

export async function startCamera() {
  let stream;
  // Mobile-friendly constraints with fallbacks
  const tryConstraints = async (fm) => {
    // Try with ideal resolution first
    try {
      return await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: fm, 
          width: { ideal: 1280, min: 640 }, 
          height: { ideal: 720, min: 480 } 
        },
        audio: false,
      });
    } catch (e) {
      // Fallback to basic constraints (mobile-friendly)
      return await navigator.mediaDevices.getUserMedia({
        video: { facingMode: fm },
        audio: false,
      });
    }
  };
  
  try {
    stream = await tryConstraints(facingMode);
  } catch (e) {
    // Fallback to alternate camera if the requested one isn't available
    const alt = facingMode === 'user' ? 'environment' : 'user';
    try { 
      stream = await tryConstraints(alt); 
      facingMode = alt; 
    } catch (e2) { 
      throw new Error(`Camera access failed: ${e.message}. Tried both cameras.`); 
    }
  }
  
  const video = getVideoEl();
  if (!video) throw new Error('Video element not found');
  
  // Ensure attributes for autoplay across browsers (critical for mobile)
  video.setAttribute('muted', 'true');
  video.setAttribute('playsinline', 'true');
  video.setAttribute('autoplay', 'true');
  video.muted = true;
  video.playsInline = true;
  
  video.srcObject = stream;
  
  // Wait for video to be ready and play
  return new Promise((resolve, reject) => {
    video.onloadedmetadata = () => {
      video.play()
        .then(() => resolve())
        .catch(err => {
          console.warn('Video play() failed, continuing anyway:', err);
          resolve(); // Continue even if play() fails - stream is active
        });
    };
    video.onerror = () => reject(new Error('Video element error'));
    // Timeout fallback
    setTimeout(() => resolve(), 1000);
  });
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
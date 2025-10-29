export class DebugLogger {
  constructor() {
    this.logsContainer = document.getElementById('debug-logs');
    this.statusText = document.getElementById('status-text');
    this.videoStatus = document.getElementById('video-status');
    this.modelStatus = document.getElementById('model-status');
    this.poseStatus = document.getElementById('pose-status');
    this.assetStatus = document.getElementById('asset-status');
    this.fpsCounter = document.getElementById('fps-counter');
    this.positionStatus = document.getElementById('position-status');
    this.maxLogs = 30;
  }
  log(type, message) {
    const logEntry = document.createElement('div');
    logEntry.className = `debug-log ${type}`;
    logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    if (this.logsContainer && this.logsContainer.children.length >= this.maxLogs) {
      this.logsContainer.removeChild(this.logsContainer.lastChild);
    }
    if (this.logsContainer) this.logsContainer.prepend(logEntry);
  }
  updateStatus(s) { this.statusText && (this.statusText.textContent = s); }
  updateVideoStatus(s) { this.videoStatus && (this.videoStatus.textContent = s); }
  updateModelStatus(s) { this.modelStatus && (this.modelStatus.textContent = s); }
  updatePoseStatus(s) { this.poseStatus && (this.poseStatus.textContent = s); }
  updateAssetStatus(s) { this.assetStatus && (this.assetStatus.textContent = s); }
  updateFPS(fps) { this.fpsCounter && (this.fpsCounter.textContent = fps.toFixed(1)); }
  updatePositionStatus(posL, rotL, posR, rotR, offset) {
    if (!this.positionStatus) return;
    this.positionStatus.textContent = `L P: (${posL.x.toFixed(2)}, ${posL.y.toFixed(2)}) R P: (${posR.x.toFixed(2)}, ${posR.y.toFixed(2)}) Offset: ${offset.toFixed(2)}`;
  }
}
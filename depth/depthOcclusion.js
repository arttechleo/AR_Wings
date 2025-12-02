/**
 * Depth Occlusion Module
 * 
 * Provides depth/LiDAR-based occlusion for AR applications.
 * Supports WebXR Depth Sensing API and can be extended for native bridges (ARKit/ARCore).
 * 
 * This module is designed to be non-breaking: if depth is not available,
 * it returns null/false and the application falls back to segmentation-based occlusion.
 */

import * as THREE from 'three';

// Internal state
let xrSession = null;
let xrDepthData = null;
let depthTexture = null;
let depthParams = null;
let depthSupported = false;
let depthInitialized = false;
let glContext = null;

// Depth texture cache
let depthTextureCache = null;

/**
 * Check if depth sensing is supported
 * @returns {Promise<boolean>} True if depth sensing is available
 */
export async function isDepthSupported() {
    try {
        // Check for WebXR Depth Sensing support
        if (typeof navigator !== 'undefined' && navigator.xr) {
            try {
                const supported = await navigator.xr.isSessionSupported('immersive-ar');
                if (supported) {
                    // WebXR Depth Sensing API support (feature detection)
                    // Note: Actual depth data will only be available during an active XR session
                    depthSupported = true;
                    return true;
                }
            } catch (err) {
                // WebXR not supported or error - continue checking other options
            }
        }
        
        // Check for iOS ARKit bridge (if available via window.webkit)
        if (typeof window !== 'undefined' && window.webkit?.messageHandlers?.depthData) {
            depthSupported = true;
            return true;
        }
    } catch (error) {
        // Any error means depth is not supported
        console.warn('[Depth] Error checking depth support:', error);
    }
    
    depthSupported = false;
    return false;
}

/**
 * Initialize depth sensing for the current rendering context
 * @param {WebGLRenderingContext} gl - WebGL context from Three.js renderer
 * @returns {Promise<boolean>} True if initialization successful
 */
export async function initDepthSensing(gl) {
    if (depthInitialized) {
        return true;
    }
    
    if (!gl) {
        console.warn('[Depth] No WebGL context provided');
        return false;
    }
    
    glContext = gl;
    
    // Check if depth is supported first
    const supported = await isDepthSupported();
    if (!supported) {
        console.log('[Depth] Depth sensing not supported on this device');
        return false;
    }
    
    try {
        // Try WebXR Depth Sensing API first
        if (navigator.xr && navigator.xr.isSessionSupported) {
            const arSupported = await navigator.xr.isSessionSupported('immersive-ar');
            if (arSupported) {
                // Note: Full WebXR session initialization would typically happen
                // in a user gesture handler. For now, we prepare the infrastructure.
                console.log('[Depth] WebXR Depth Sensing API available (will initialize on session start)');
                depthInitialized = true;
                return true;
            }
        }
        
        // Check for native bridge (ARKit via WKWebView)
        if (typeof window !== 'undefined' && window.webkit?.messageHandlers?.depthData) {
            console.log('[Depth] Native ARKit depth bridge detected');
            setupNativeDepthBridge();
            depthInitialized = true;
            return true;
        }
        
        depthInitialized = false;
        return false;
    } catch (error) {
        console.error('[Depth] Initialization error:', error);
        depthInitialized = false;
        return false;
    }
}

/**
 * Setup native ARKit depth bridge (iOS/WKWebView)
 * This sets up message handlers for receiving depth data from native code
 */
function setupNativeDepthBridge() {
    if (typeof window === 'undefined') return;
    
    // Listen for depth data from native bridge
    window.addEventListener('message', (event) => {
        if (event.data && event.data.type === 'depthData') {
            handleNativeDepthData(event.data);
        }
    });
    
    // Request depth data from native
    if (window.webkit?.messageHandlers?.requestDepthData) {
        window.webkit.messageHandlers.requestDepthData.postMessage({ enabled: true });
    }
}

/**
 * Handle depth data from native bridge
 * @param {Object} data - Depth data from native code
 */
function handleNativeDepthData(data) {
    if (!data.depthBuffer || !glContext) return;
    
    try {
        // Convert native depth data to texture
        // Format depends on native implementation (typically Uint16Array or Float32Array)
        const depthArray = data.depthBuffer;
        const width = data.width || 256;
        const height = data.height || 192;
        
        // Create or update depth texture
        if (!depthTextureCache || depthTextureCache.width !== width || depthTextureCache.height !== height) {
            depthTextureCache = new THREE.DataTexture(
                depthArray,
                width,
                height,
                THREE.RedFormat,
                THREE.UnsignedShortType
            );
            depthTextureCache.minFilter = THREE.NearestFilter;
            depthTextureCache.magFilter = THREE.NearestFilter;
        } else {
            depthTextureCache.image.data = depthArray;
            depthTextureCache.needsUpdate = true;
        }
        
        depthTexture = depthTextureCache;
        
        // Store depth parameters
        depthParams = {
            width,
            height,
            minDepth: data.minDepth || 0.0,
            maxDepth: data.maxDepth || 5.0,
            depthScale: data.depthScale || 1000.0, // Typical: mm to meters
        };
    } catch (error) {
        console.error('[Depth] Error processing native depth data:', error);
    }
}

/**
 * Initialize WebXR session with depth sensing
 * This should be called after user interaction (e.g., button click)
 * @param {WebGLRenderingContext} gl - WebGL context
 * @returns {Promise<boolean>} True if session started successfully
 */
export async function startXRSessionWithDepth(gl) {
    if (!navigator.xr) {
        return false;
    }
    
    try {
        // Request immersive AR session with depth sensing
        xrSession = await navigator.xr.requestSession('immersive-ar', {
            requiredFeatures: ['depth-sensing'],
            optionalFeatures: ['local-floor', 'bounded-floor']
        });
        
        if (!xrSession) {
            return false;
        }
        
        // Get WebGL layer
        const glLayer = new XRWebGLLayer(xrSession, gl, {
            depth: true,
            stencil: false,
            antialias: true,
            ignoreDepthValues: false
        });
        
        xrSession.updateRenderState({ baseLayer: glLayer });
        
        // Setup depth data callback
        setupXRDepthData();
        
        console.log('[Depth] WebXR session with depth sensing started');
        return true;
    } catch (error) {
        console.error('[Depth] Failed to start WebXR session:', error);
        return false;
    }
}

/**
 * Setup WebXR depth data handling
 */
function setupXRDepthData() {
    if (!xrSession) return;
    
    // WebXR Depth Sensing API (when available)
    // This will be called in the render loop to get depth data per frame
}

/**
 * Get current depth texture from WebXR or native bridge
 * This should be called in the render loop
 * @returns {THREE.Texture | null} Depth texture or null if not available
 */
export function getCurrentDepthTexture() {
    // If we have a cached depth texture, return it
    if (depthTexture && depthTexture.isTexture) {
        return depthTexture;
    }
    
    // Try to get depth from WebXR frame (in render loop)
    if (xrSession && xrDepthData) {
        return convertXRDepthToTexture(xrDepthData);
    }
    
    return null;
}

/**
 * Convert WebXR depth data to Three.js texture
 * @param {XRDepthInformation} depthInfo - WebXR depth information
 * @returns {THREE.Texture | null}
 */
function convertXRDepthToTexture(depthInfo) {
    if (!depthInfo || !glContext) return null;
    
    try {
        // Get depth data from XR frame
        // Note: WebXR Depth Sensing API structure may vary
        // This is a placeholder implementation
        
        // Access depth data (format depends on WebXR implementation)
        const depthBuffer = depthInfo.getDepthInMeters ? depthInfo.getDepthInMeters() : null;
        
        if (!depthBuffer || !depthBuffer.data) {
            return null;
        }
        
        const width = depthBuffer.width || 256;
        const height = depthBuffer.height || 192;
        
        // Create or update texture
        if (!depthTextureCache || depthTextureCache.width !== width || depthTextureCache.height !== height) {
            depthTextureCache = new THREE.DataTexture(
                depthBuffer.data,
                width,
                height,
                THREE.RedFormat,
                THREE.FloatType
            );
            depthTextureCache.minFilter = THREE.NearestFilter;
            depthTextureCache.magFilter = THREE.NearestFilter;
        } else {
            depthTextureCache.image.data = depthBuffer.data;
            depthTextureCache.needsUpdate = true;
        }
        
        // Update depth params
        depthParams = {
            width,
            height,
            minDepth: depthInfo.nearDepth || 0.0,
            maxDepth: depthInfo.farDepth || 5.0,
            depthScale: 1.0, // Already in meters
        };
        
        return depthTextureCache;
    } catch (error) {
        console.error('[Depth] Error converting XR depth data:', error);
        return null;
    }
}

/**
 * Update depth data from WebXR frame
 * Call this in the render loop when rendering XR frames
 * @param {XRFrame} frame - WebXR frame
 */
export function updateDepthFromXRFrame(frame) {
    if (!frame || !xrSession) return;
    
    try {
        // Get depth information from XR frame
        // WebXR Depth Sensing API (when available)
        const depthInfo = frame.getDepthInformation && frame.getDepthInformation();
        
        if (depthInfo) {
            xrDepthData = depthInfo;
            depthTexture = convertXRDepthToTexture(depthInfo);
        }
    } catch (error) {
        // Depth may not be available this frame
        xrDepthData = null;
    }
}

/**
 * Get depth parameters (near, far, scale, dimensions)
 * @returns {Object | null} Depth parameters or null
 */
export function getDepthParams() {
    return depthParams ? { ...depthParams } : null;
}

/**
 * Check if depth is currently available
 * @returns {boolean}
 */
export function isDepthAvailable() {
    return depthTexture !== null || (xrSession && xrDepthData !== null);
}

/**
 * Clean up depth sensing resources
 */
export function disposeDepthSensing() {
    if (xrSession) {
        xrSession.end();
        xrSession = null;
    }
    
    if (depthTextureCache) {
        depthTextureCache.dispose();
        depthTextureCache = null;
    }
    
    depthTexture = null;
    depthParams = null;
    xrDepthData = null;
    depthInitialized = false;
    glContext = null;
}


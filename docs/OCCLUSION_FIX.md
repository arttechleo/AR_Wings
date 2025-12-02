# Occlusion and Visibility Issues - Fix Plan

## Current Issues:
1. Video is upside down on mobile phones
2. Gaussian splatting wings are not visible when human is detected
3. No occlusion segmentation working (wings should appear behind person)

## Root Causes:

1. **Video Orientation**: Mobile cameras output video with different orientation than desktop
2. **Wings Not Visible**: Occlusion compositor may be interfering or wings are being hidden incorrectly
3. **No Occlusion**: Current occlusion system isn't working properly

## Solution Approach:

### Phase 1: Fix Visibility and Orientation
- Fix video texture orientation for mobile devices
- Ensure wings render normally without occlusion compositor interference
- Verify wings are visible when person detected

### Phase 2: Implement Proper Occlusion
- Use depth-based rendering with segmentation mask
- Render wings at correct depth behind person
- Use mask to hide wings where person body overlaps

## Implementation Notes:

The occlusion compositor needs to:
1. Render wings separately from video background
2. Use segmentation mask to determine where person is
3. Composite: person areas show video (occluding wings), background shows wings + video


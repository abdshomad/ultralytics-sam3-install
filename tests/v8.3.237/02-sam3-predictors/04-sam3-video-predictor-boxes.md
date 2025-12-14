# Test 02-04: SAM3 Video Predictor Boxes

## Test ID
02-04

## Test Name
SAM3 Video Predictor Boxes

## Objective
Validate that `SAM3VideoPredictor` correctly performs video tracking with box prompts, including frame-by-frame tracking, memory management, and consistent object tracking across video frames.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- Test video file available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import SAM
   from ultralytics.models.sam.predict import SAM3VideoPredictor
   import numpy as np
   ```

2. **Test Predictor Initialization**
   - Initialize `SAM3VideoPredictor` with `sam3.pt`
   - Verify video tracker is initialized
   - Check memory management setup

3. **Test Single Object Tracking**
   - Load test video
   - Provide initial box coordinates for first frame
   - Run tracking across video
   - Verify object is tracked consistently
   - Check tracking accuracy

4. **Test Multiple Object Tracking**
   - Provide multiple box coordinates
   - Run multi-object tracking
   - Verify all objects are tracked
   - Check object ID consistency

5. **Test Frame-by-Frame Tracking**
   - Process video frame by frame
   - Verify tracking continues across frames
   - Check memory is maintained between frames
   - Test tracking recovery after occlusion

6. **Test Box Prompt Updates**
   - Update box prompts mid-tracking
   - Verify tracking adapts to new boxes
   - Check smooth transition

7. **Test Tracking Memory**
   - Verify memory is maintained across frames
   - Check memory efficiency
   - Test memory clearing when needed

8. **Test Tracking Accuracy**
   - Compare tracked boxes with ground truth
   - Calculate tracking metrics (IoU, accuracy)
   - Verify tracking quality

## Expected Results

- `SAM3VideoPredictor` initializes successfully
- Single object tracking works across video frames
- Multiple objects are tracked simultaneously
- Frame-by-frame tracking is consistent
- Box prompt updates are handled correctly
- Memory management is efficient
- Tracking accuracy is high (IoU > 0.7)

## Validation Criteria

- Predictor initializes with video tracker
- Single object tracking maintains consistent ID
- Multiple objects are tracked with separate IDs
- Tracking continues across all frames
- Box updates are handled smoothly
- Memory is maintained efficiently
- Tracking accuracy meets quality thresholds
- No tracking failures for visible objects

## Dependencies

- SAM3 weights file (`sam3.pt`)
- Test video with moving objects
- Ground truth annotations (optional, for accuracy testing)
- Related tests: 02-05, 03-07

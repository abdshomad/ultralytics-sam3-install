# Test 00-04: Video Concept Tracking

## Test ID
00-04

## Test Name
Video Concept Tracking

## Objective
Validate that `SAM3VideoPredictor` correctly performs video tracking with box prompts, including frame-by-frame tracking, memory management, and consistent object tracking across video frames. This test implements the README usage example (lines 223-244).

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`models/sam3.pt`) available
- Test video file available (optional - test will skip gracefully if not found)
- GPU available (recommended)

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics.models.sam.predict import SAM3VideoPredictor
   import numpy as np
   import matplotlib.pyplot as plt
   ```

2. **Check Requirements**
   - Verify model file exists at `models/sam3.pt`
   - Locate test video from submodules (optional)
   - If video not found, skip test gracefully with informative message

3. **Initialize Predictor**
   - Initialize `SAM3VideoPredictor` with model path
   - Configure with appropriate overrides (conf=0.25, task="segment", mode="predict", half=True)
   - Verify predictor is created successfully

4. **Define Bounding Box Prompts**
   - Provide initial box coordinates for objects to track
   - Use example coordinates from README or appropriate coordinates for test video

5. **Test Video Tracking**
   - Process video with bounding box prompts
   - Use stream=True for frame-by-frame processing
   - Verify object tracking across video frames
   - Check tracking consistency

6. **Collect Frames for Visualization**
   - Process first few frames (limit to 5 for visualization)
   - Extract annotated frames with segmentation masks
   - Store frames for side-by-side display

7. **Create Side-by-Side Visualizations**
   - Display multiple frames side-by-side showing tracking consistency
   - Save visualization outputs to `tests/v8.3.237/00-basic/outputs/`
   - Verify visualizations are created correctly

## Expected Results

- `SAM3VideoPredictor` initializes successfully
- Video tracking works across video frames (if video available)
- Frame-by-frame tracking is consistent
- Memory management is efficient
- Tracking accuracy is maintained across frames
- Side-by-side visualizations are created and saved successfully (if video available)
- Test skips gracefully if video not available

## Validation Criteria

- Predictor initializes with video tracker
- Video tracking maintains consistent object IDs (if video available)
- Tracking continues across all processed frames
- Memory is maintained efficiently
- Visualizations display correctly with multiple frames side-by-side
- Test handles missing video gracefully

## Dependencies

- SAM3 weights file (`models/sam3.pt`)
- Test video file (optional) - may be in `submodules/sam3/assets/videos/` or `submodules/inference/assets/`
- Related tests: 00-02, 00-05

## Output Files

- `tests/v8.3.237/00-basic/outputs/04-video-tracking.png` - Visualization showing multiple frames with tracking results (if video available)

## Notes

- This test is designed to skip gracefully if no test video is available
- Video processing may take significant time depending on video length
- Only first 5 frames are visualized for efficiency

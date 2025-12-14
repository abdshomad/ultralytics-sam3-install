# Test 10-02: Cross-Feature Compatibility

## Test ID
10-02

## Test Name
Cross-Feature Compatibility

## Objective
Validate that SAM3 features work correctly with other Ultralytics features, ensuring compatibility across the framework.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- YOLO model weights files available
- Test images and videos available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import SAM, YOLO
   import numpy as np
   ```

2. **Test SAM3 with YOLO Detection**
   - Run YOLO detection on image
   - Use YOLO boxes as prompts for SAM3
   - Verify SAM3 segmentation works with YOLO boxes
   - Check compatibility

3. **Test SAM3 with YOLO Tracking**
   - Run YOLO tracking on video
   - Use tracked boxes as prompts for SAM3
   - Verify SAM3 works with YOLO tracking
   - Check integration

4. **Test Export Compatibility**
   - Export SAM3 model to ONNX
   - Verify export works correctly
   - Test exported model functionality
   - Check compatibility

5. **Test Training Integration**
   - Use SAM3 predictions in training pipeline
   - Verify training works with SAM3
   - Check integration points

6. **Test Validation Integration**
   - Use SAM3 in validation pipeline
   - Verify validation works correctly
   - Check metrics calculation

7. **Test Multi-Model Workflow**
   - Use YOLO for detection
   - Use SAM3 for segmentation
   - Verify both work together
   - Check workflow compatibility

## Expected Results

- SAM3 works with YOLO detection
- SAM3 works with YOLO tracking
- Export compatibility is maintained
- Training integration works
- Validation integration works
- Multi-model workflows function correctly

## Validation Criteria

- All integration points work correctly
- No compatibility errors occur
- Features work together seamlessly
- Export functionality is preserved
- Training/validation integration works
- Multi-model workflows are functional
- No conflicts between features

## Dependencies

- SAM3 weights file (`sam3.pt`)
- YOLO model weights files
- Test images and videos
- Related tests: All previous categories, 10-01, 10-03

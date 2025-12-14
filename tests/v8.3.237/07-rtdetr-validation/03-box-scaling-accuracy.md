# Test 07-03: Box Scaling Accuracy

## Test ID
07-03

## Test Name
Box Scaling Accuracy

## Objective
Validate that box scaling in `predictions.json` for RT-DETR is correct, ensuring predictions are properly scaled and match ground truth annotations.

## Prerequisites
- Ultralytics v8.3.237 installed
- RT-DETR model weights file available
- Validation dataset with ground truth annotations available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import RTDETR
   import json
   import numpy as np
   ```

2. **Test Predictions JSON Generation**
   - Run validation on dataset
   - Generate `predictions.json`
   - Verify file is created
   - Check file format is valid JSON

3. **Test Box Scaling**
   - Load predictions from JSON
   - Extract box coordinates
   - Verify boxes are properly scaled
   - Check box coordinates are in correct format

4. **Test Scaling Accuracy**
   - Compare predictions with ground truth
   - Calculate IoU for boxes
   - Verify scaling doesn't introduce errors
   - Check box accuracy

5. **Test Different Image Sizes**
   - Test with various image sizes
   - Verify scaling works for all sizes
   - Check consistency across sizes

6. **Test Box Format**
   - Verify boxes are in correct format (xyxy, xywh, etc.)
   - Check format matches expected standard
   - Test format consistency

7. **Test Prediction Consistency**
   - Run validation multiple times
   - Verify predictions are consistent
   - Check no scaling variations

## Expected Results

- `predictions.json` is generated correctly
- Boxes are properly scaled
- Scaling accuracy is high
- Different image sizes work correctly
- Box format is correct
- Predictions are consistent

## Validation Criteria

- JSON file is created and valid
- Boxes are scaled correctly (verified against ground truth)
- IoU with ground truth is high (> 0.7)
- All image sizes work correctly
- Box format matches expected standard
- Predictions are consistent across runs
- No scaling errors occur

## Dependencies

- RT-DETR model weights file
- Validation dataset with ground truth
- Related tests: 07-01, 07-02

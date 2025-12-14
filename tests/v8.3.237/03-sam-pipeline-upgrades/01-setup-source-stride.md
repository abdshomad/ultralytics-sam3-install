# Test 03-01: Setup Source Stride

## Test ID
03-01

## Test Name
Setup Source Stride

## Objective
Validate that `Predictor.setup_source` now accepts an explicit `stride` parameter and that SAM/SAM2/SAM3 predictors use it to enforce square image sizes and consistent feature shapes.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM/SAM2/SAM3 weights files available
- Test images with various aspect ratios available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import SAM
   from ultralytics.models.sam.predict import Predictor
   import numpy as np
   from PIL import Image
   ```

2. **Test setup_source with Stride Parameter**
   - Initialize SAM predictor
   - Call `setup_source` with explicit `stride` parameter
   - Verify stride parameter is accepted
   - Check stride is used in processing

3. **Test Square Image Enforcement**
   - Load non-square test image
   - Process with `setup_source` using stride
   - Verify image is resized to square
   - Check aspect ratio handling

4. **Test Consistent Feature Shapes**
   - Process multiple images with same stride
   - Verify feature shapes are consistent
   - Check feature dimensions match stride requirements

5. **Test Different Stride Values**
   - Test with stride=16 (default)
   - Test with stride=8
   - Test with stride=32
   - Verify all stride values work correctly

6. **Test SAM/SAM2/SAM3 Consistency**
   - Test stride handling in SAM predictor
   - Test stride handling in SAM2 predictor
   - Test stride handling in SAM3 predictor
   - Verify consistent behavior across models

7. **Test Feature Shape Validation**
   - Extract features after setup_source
   - Verify feature shapes match expected dimensions
   - Check feature consistency across images

## Expected Results

- `setup_source` accepts `stride` parameter
- Images are resized to square when stride is specified
- Feature shapes are consistent across images
- Different stride values work correctly
- SAM/SAM2/SAM3 all handle stride consistently
- Feature dimensions match stride requirements

## Validation Criteria

- `stride` parameter is accepted by `setup_source`
- Non-square images are converted to square
- Feature shapes are consistent (within tolerance)
- All stride values work without errors
- SAM/SAM2/SAM3 behavior is consistent
- Feature dimensions are correct for given stride
- No shape mismatches occur

## Dependencies

- SAM/SAM2/SAM3 weights files
- Test images with various aspect ratios
- Related tests: 03-02, 03-03

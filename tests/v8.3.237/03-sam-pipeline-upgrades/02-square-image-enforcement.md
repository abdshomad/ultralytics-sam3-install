# Test 03-02: Square Image Enforcement

## Test ID
03-02

## Test Name
Square Image Enforcement

## Objective
Validate that SAM/SAM2/SAM3 predictors enforce square image sizes through the shared `stride` handling, avoiding subtle spatial shape bugs and mismatches in encoders/decoders.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM/SAM2/SAM3 weights files available
- Test images with various aspect ratios (portrait, landscape, square)
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import SAM
   import numpy as np
   from PIL import Image
   ```

2. **Test Portrait Image (Height > Width)**
   - Load portrait-oriented test image
   - Process through SAM predictor
   - Verify image is resized to square
   - Check no shape mismatches occur

3. **Test Landscape Image (Width > Height)**
   - Load landscape-oriented test image
   - Process through SAM predictor
   - Verify image is resized to square
   - Check aspect ratio is preserved where possible

4. **Test Square Image**
   - Load square test image
   - Process through SAM predictor
   - Verify no unnecessary resizing occurs
   - Check processing is efficient

5. **Test Encoder/Decoder Shape Consistency**
   - Process images through encoder
   - Verify encoder output shapes are consistent
   - Process through decoder
   - Verify decoder input/output shapes match
   - Check no shape mismatches

6. **Test SAM/SAM2/SAM3 Consistency**
   - Test square enforcement in SAM
   - Test square enforcement in SAM2
   - Test square enforcement in SAM3
   - Verify all models enforce square images consistently

7. **Test Feature Shape Consistency**
   - Extract features from different aspect ratio images
   - Verify feature shapes are identical
   - Check spatial dimensions match

8. **Test Edge Cases**
   - Test with very wide images
   - Test with very tall images
   - Test with small images
   - Verify all edge cases handled correctly

## Expected Results

- Portrait images are resized to square
- Landscape images are resized to square
- Square images are processed efficiently
- Encoder/decoder shapes are consistent
- SAM/SAM2/SAM3 all enforce square images
- Feature shapes are consistent across aspect ratios
- Edge cases are handled correctly
- No shape mismatch errors occur

## Validation Criteria

- All non-square images are converted to square
- Square images are not unnecessarily resized
- Encoder output shapes are consistent
- Decoder input/output shapes match
- No shape mismatch errors occur
- Feature shapes are identical for same-size inputs
- Edge cases are handled gracefully
- Processing is efficient

## Dependencies

- SAM/SAM2/SAM3 weights files
- Test images with various aspect ratios
- Related tests: 03-01, 03-03, 03-04

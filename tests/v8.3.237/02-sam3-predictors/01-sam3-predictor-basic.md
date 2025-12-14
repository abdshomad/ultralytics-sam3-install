# Test 02-01: SAM3 Predictor Basic

## Test ID
02-01

## Test Name
SAM3 Predictor Basic

## Objective
Validate that `SAM3Predictor` correctly implements SAM3-style interactive segmentation with point and box prompts, including proper initialization, prompt handling, and mask generation.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- Test images available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import SAM
   from ultralytics.models.sam.predict import SAM3Predictor
   import numpy as np
   from PIL import Image
   ```

2. **Test Predictor Initialization**
   - Initialize `SAM3Predictor` with `sam3.pt`
   - Verify predictor is created successfully
   - Check predictor configuration

3. **Test Point Prompt Segmentation**
   - Load test image
   - Provide point coordinates (positive point)
   - Run segmentation
   - Verify mask is generated
   - Test with negative points
   - Verify mask refinement

4. **Test Box Prompt Segmentation**
   - Provide box coordinates [x1, y1, x2, y2]
   - Run segmentation with box prompt
   - Verify mask matches box region
   - Check mask quality

5. **Test Combined Prompts**
   - Combine point and box prompts
   - Run segmentation
   - Verify combined prompt handling
   - Check results

6. **Test Multiple Objects**
   - Segment multiple objects in same image
   - Verify all objects are segmented
   - Check mask separation

7. **Test Image Feature Extraction**
   - Extract image features
   - Verify feature shape and type
   - Test feature reuse for multiple queries

## Expected Results

- `SAM3Predictor` initializes successfully
- Point prompts produce accurate masks
- Box prompts produce masks matching box regions
- Combined prompts work correctly
- Multiple objects can be segmented
- Image features can be extracted and reused
- Segmentation results are accurate and consistent

## Validation Criteria

- Predictor initializes without errors
- Point prompts generate masks with IoU > 0.7 for known objects
- Box prompts generate masks covering box region
- Combined prompts produce correct results
- Multiple objects are properly segmented
- Feature extraction produces valid feature tensors
- Feature reuse improves efficiency
- Results are consistent across runs

## Dependencies

- SAM3 weights file (`sam3.pt`) - must be manually downloaded
- Test images with known objects
- Related tests: 01-05, 02-02, 02-06

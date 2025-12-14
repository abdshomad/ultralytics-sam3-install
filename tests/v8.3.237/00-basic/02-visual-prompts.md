# Test 00-02: Visual Prompts (Points and Boxes)

## Test ID
00-02

## Test Name
Visual Prompts (Points and Boxes)

## Objective
Validate that `SAM3Predictor` correctly implements SAM3-style interactive segmentation with point and box prompts, including proper initialization, prompt handling, and mask generation. This test implements the README usage example (lines 176-203).

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`models/sam3.pt`) available
- Test images available from submodules
- GPU available (recommended)

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics.models.sam.predict import SAM3Predictor
   import numpy as np
   from PIL import Image
   import matplotlib.pyplot as plt
   ```

2. **Check Requirements**
   - Verify model file exists at `models/sam3.pt`
   - Locate test image from submodules
   - Get image dimensions for coordinate calculation

3. **Initialize Predictor**
   - Initialize `SAM3Predictor` with model path
   - Configure with appropriate overrides (conf=0.25, task="segment", mode="predict", half=True)
   - Verify predictor is created successfully

4. **Set Image**
   - Load test image using `predictor.set_image()`
   - Store original image for visualization

5. **Test Point Prompt Segmentation**
   - Provide point coordinates (center of image)
   - Provide point labels (positive point: 1)
   - Run segmentation with point prompt
   - Verify mask is generated

6. **Test Box Prompt Segmentation**
   - Provide box coordinates [x1, y1, x2, y2] (center region)
   - Run segmentation with box prompt
   - Verify mask matches box region

7. **Test Combined Prompts**
   - Combine point and box prompts
   - Run segmentation with combined prompts
   - Verify combined prompt handling

8. **Create Side-by-Side Visualizations**
   - Display original image and all three result types side-by-side
   - Save visualization outputs to `tests/v8.3.237/00-basic/outputs/`
   - Verify visualizations are created correctly

## Expected Results

- `SAM3Predictor` initializes successfully
- Point prompts produce accurate masks
- Box prompts produce masks matching box regions
- Combined prompts work correctly
- Segmentation results are accurate and consistent
- Side-by-side visualizations are created and saved successfully

## Validation Criteria

- Predictor initializes without errors
- Point prompts generate masks with IoU > 0.7 for known objects
- Box prompts generate masks covering box region
- Combined prompts produce correct results
- Visualizations display correctly with original and all results side-by-side
- Results are consistent across runs

## Dependencies

- SAM3 weights file (`models/sam3.pt`) - must be manually downloaded
- Test images from `submodules/sam3/assets/images/` or `submodules/inference/assets/`
- Related tests: 00-01, 00-03

## Output Files

- `tests/v8.3.237/00-basic/outputs/02-visual-prompts-all.png` - Visualization of all prompt types (points, boxes, combined)

# Test 00-03: Image Exemplar-Based Segmentation

## Test ID
00-03

## Test Name
Image Exemplar-Based Segmentation

## Objective
Validate that `SAM3SemanticPredictor` correctly performs exemplar-based concept segmentation using box prompts to identify and segment similar instances across an image. This test implements the README usage example (lines 205-221).

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`models/sam3.pt`) available
- BPE vocabulary file (`models/bpe_simple_vocab_16e6.txt.gz`) available
- Test images with multiple similar objects available from submodules
- GPU available (recommended)

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics.models.sam.predict import SAM3SemanticPredictor
   import numpy as np
   from PIL import Image, ImageDraw
   import matplotlib.pyplot as plt
   ```

2. **Check Requirements**
   - Verify model file exists at `models/sam3.pt`
   - Verify BPE vocabulary exists at `models/bpe_simple_vocab_16e6.txt.gz`
   - Locate test image from submodules (preferably with multiple similar objects)

3. **Initialize Predictor**
   - Initialize `SAM3SemanticPredictor` with model path and BPE vocabulary
   - Configure with appropriate overrides (conf=0.25, task="segment", mode="predict", half=True)
   - Verify predictor is created successfully

4. **Set Image**
   - Load test image using `predictor.set_image()`
   - Store original image for visualization
   - Get image dimensions for exemplar box calculation

5. **Define Exemplar Box**
   - Provide exemplar box coordinates [x1, y1, x2, y2]
   - Use example coordinates from README or calculate center region
   - Draw exemplar box on image for visualization

6. **Test Exemplar-Based Segmentation**
   - Provide exemplar box as bounding box prompt
   - Run exemplar-based segmentation
   - Verify all similar instances are segmented
   - Check mask quality

7. **Create Side-by-Side Visualizations**
   - Display original image, exemplar box, and all detected similar instances side-by-side
   - Save visualization outputs to `tests/v8.3.237/00-basic/outputs/`
   - Verify visualizations are created correctly

## Expected Results

- `SAM3SemanticPredictor` supports exemplar-based prompts
- Single exemplar box segments all similar instances
- Concept propagation finds all similar instances
- Exemplar-based segmentation is accurate
- Side-by-side visualizations are created and saved successfully

## Validation Criteria

- Predictor handles exemplar prompts correctly
- Single exemplar finds all similar instances (recall > 0.8)
- Concept propagation is effective
- Segmentation accuracy is high (IoU > 0.7 for known instances)
- Visualizations display correctly with original, exemplar box, and results side-by-side
- Results are consistent

## Dependencies

- SAM3 weights file (`models/sam3.pt`)
- BPE vocabulary file (`models/bpe_simple_vocab_16e6.txt.gz`)
- Test images with multiple similar objects from `submodules/sam3/assets/images/` or `submodules/inference/assets/`
- Related tests: 00-01, 00-02

## Output Files

- `tests/v8.3.237/00-basic/outputs/03-exemplar-segmentation.png` - Visualization showing original, exemplar box, and all detected similar instances

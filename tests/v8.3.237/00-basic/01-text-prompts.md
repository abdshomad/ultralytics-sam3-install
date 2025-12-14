# Test 00-01: Text Prompts

## Test ID
00-01

## Test Name
Text Prompts

## Objective
Validate that `SAM3SemanticPredictor` correctly performs text-based concept segmentation on images, including proper text encoding, concept matching, and mask generation for both simple concepts and descriptive phrases. This test implements the README usage example (lines 150-174).

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`models/sam3.pt`) available
- BPE vocabulary file (`models/bpe_simple_vocab_16e6.txt.gz`) available
- Test images available from submodules
- GPU available (recommended)

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics.models.sam.predict import SAM3SemanticPredictor
   import numpy as np
   from PIL import Image
   import matplotlib.pyplot as plt
   ```

2. **Check Requirements**
   - Verify model file exists at `models/sam3.pt`
   - Verify BPE vocabulary exists at `models/bpe_simple_vocab_16e6.txt.gz`
   - Locate test image from submodules

3. **Initialize Predictor**
   - Initialize `SAM3SemanticPredictor` with model path and BPE vocabulary
   - Configure with appropriate overrides (conf=0.25, task="segment", mode="predict", half=True)
   - Verify predictor is created successfully

4. **Set Image**
   - Load test image using `predictor.set_image()`
   - Store original image for visualization

5. **Test Simple Concept Text Prompts**
   - Provide simple text prompts: ["person", "bus", "glasses"]
   - Run semantic segmentation
   - Verify masks are generated for matching concepts
   - Check mask accuracy

6. **Test Descriptive Phrase Text Prompts**
   - Provide descriptive text prompts: ["person with red cloth", "person with blue cloth"]
   - Run semantic segmentation
   - Verify complex prompt handling
   - Check segmentation results

7. **Create Side-by-Side Visualizations**
   - Display original image and segmentation results side-by-side
   - Save visualization outputs to `tests/v8.3.237/00-basic/outputs/`
   - Verify visualizations are created correctly

## Expected Results

- `SAM3SemanticPredictor` initializes with text encoder when BPE path provided
- Simple concept text prompts produce accurate masks
- Descriptive phrase prompts are handled correctly
- Text encoding produces valid feature vectors
- Segmentation results match text descriptions
- Side-by-side visualizations are created and saved successfully

## Validation Criteria

- Predictor initializes with text encoder
- Text prompts generate masks for matching concepts
- Simple concepts are properly segmented
- Descriptive phrases are processed correctly
- Text encoding produces valid features
- Mask quality is acceptable (IoU > 0.6 for known objects)
- Visualizations display correctly with original and results side-by-side
- Results are consistent across runs

## Dependencies

- BPE vocabulary file (`models/bpe_simple_vocab_16e6.txt.gz`) - required for text encoding
- SAM3 weights file (`models/sam3.pt`)
- Test images from `submodules/sam3/assets/images/` or `submodules/inference/assets/`
- Related tests: 00-02, 00-03, 00-05

## Output Files

- `tests/v8.3.237/00-basic/outputs/01-text-prompts-simple.png` - Visualization of simple concept results
- `tests/v8.3.237/00-basic/outputs/01-text-prompts-phrases.png` - Visualization of descriptive phrase results

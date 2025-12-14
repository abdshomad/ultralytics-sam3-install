# Test 02-02: SAM3 Semantic Predictor Text

## Test ID
02-02

## Test Name
SAM3 Semantic Predictor Text

## Objective
Validate that `SAM3SemanticPredictor` correctly performs text-based concept segmentation on images, including proper text encoding, concept matching, and mask generation for text prompts.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`) available
- Test images with known objects available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import SAM
   from ultralytics.models.sam.predict import SAM3SemanticPredictor
   import numpy as np
   from PIL import Image
   ```

2. **Test Predictor Initialization with BPE**
   - Initialize `SAM3SemanticPredictor` with `sam3.pt` and BPE path
   - Verify text encoder is loaded
   - Check predictor supports text prompts

3. **Test Single Concept Text Prompt**
   - Load test image with known objects
   - Provide text prompt (e.g., "person")
   - Run semantic segmentation
   - Verify masks are generated for matching concepts
   - Check mask accuracy

4. **Test Multiple Concept Text Prompts**
   - Provide multiple text prompts (e.g., "person", "car", "dog")
   - Run segmentation
   - Verify all concepts are detected
   - Check mask separation for different concepts

5. **Test Complex Text Prompts**
   - Test with descriptive prompts (e.g., "person with red hat")
   - Test with compound concepts
   - Verify complex prompt handling
   - Check segmentation results

6. **Test Text Encoding**
   - Verify text is encoded correctly
   - Check encoded features have correct shape
   - Test with various text inputs

7. **Test Presence Scores**
   - Check presence scores are returned
   - Verify scores indicate concept presence
   - Test score thresholds

## Expected Results

- `SAM3SemanticPredictor` initializes with text encoder when BPE path provided
- Single concept text prompts produce accurate masks
- Multiple concept prompts detect all concepts
- Complex text prompts are handled correctly
- Text encoding produces valid feature vectors
- Presence scores indicate concept presence accurately
- Segmentation results match text descriptions

## Validation Criteria

- Predictor initializes with text encoder
- Text prompts generate masks for matching concepts
- Multiple concepts are properly segmented
- Complex prompts are processed correctly
- Text encoding produces valid features
- Presence scores are meaningful (higher for present concepts)
- Mask quality is acceptable (IoU > 0.6 for known objects)
- Results are consistent across runs

## Dependencies

- BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`) - required for text encoding
- SAM3 weights file (`sam3.pt`)
- Test images with labeled objects
- Related tests: 01-03, 02-03, 02-06

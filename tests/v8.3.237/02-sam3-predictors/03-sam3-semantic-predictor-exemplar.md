# Test 02-03: SAM3 Semantic Predictor Exemplar

## Test ID
02-03

## Test Name
SAM3 Semantic Predictor Exemplar

## Objective
Validate that `SAM3SemanticPredictor` correctly performs exemplar-based concept segmentation using box prompts to identify and segment similar instances across an image.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`) available (optional)
- Test images with multiple similar objects available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import SAM
   from ultralytics.models.sam.predict import SAM3SemanticPredictor
   import numpy as np
   from PIL import Image
   ```

2. **Test Predictor Initialization**
   - Initialize `SAM3SemanticPredictor` with `sam3.pt`
   - Verify predictor supports exemplar prompts
   - Check exemplar handling capability

3. **Test Single Exemplar Box**
   - Load test image with multiple similar objects
   - Provide exemplar box coordinates for one object
   - Run exemplar-based segmentation
   - Verify all similar instances are segmented
   - Check mask quality

4. **Test Multiple Exemplar Boxes**
   - Provide multiple exemplar boxes
   - Run segmentation
   - Verify all exemplar instances are segmented
   - Check concept consistency

5. **Test Exemplar with Text**
   - Combine exemplar box with text prompt
   - Run segmentation
   - Verify combined prompt handling
   - Check results match both exemplar and text

6. **Test Exemplar Concept Propagation**
   - Use exemplar to identify concept
   - Verify concept is propagated to similar instances
   - Check all instances of concept are found

7. **Test Exemplar Accuracy**
   - Test with known object locations
   - Verify exemplar-based segmentation finds all instances
   - Check false positive rate is low

## Expected Results

- `SAM3SemanticPredictor` supports exemplar-based prompts
- Single exemplar box segments all similar instances
- Multiple exemplar boxes work correctly
- Exemplar combined with text works properly
- Concept propagation finds all similar instances
- Exemplar-based segmentation is accurate
- False positive rate is acceptable

## Validation Criteria

- Predictor handles exemplar prompts correctly
- Single exemplar finds all similar instances (recall > 0.8)
- Multiple exemplars work correctly
- Combined prompts produce correct results
- Concept propagation is effective
- Segmentation accuracy is high (IoU > 0.7 for known instances)
- False positive rate is low (< 0.1)
- Results are consistent

## Dependencies

- SAM3 weights file (`sam3.pt`)
- BPE vocabulary file (optional, for combined prompts)
- Test images with multiple similar objects
- Related tests: 02-02, 02-06

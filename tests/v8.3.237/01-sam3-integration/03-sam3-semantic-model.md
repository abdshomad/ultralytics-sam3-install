# Test 01-03: SAM3 Semantic Model

## Test ID
01-03

## Test Name
SAM3 Semantic Model

## Objective
Validate that `SAM3SemanticModel` correctly implements semantic segmentation capabilities with text and exemplar prompts, including proper text encoding and concept segmentation functionality.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`) available
- Test images available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics.models.sam.modules.sam import SAM3SemanticModel
   from ultralytics import SAM
   import numpy as np
   from PIL import Image
   ```

2. **Test Semantic Model Initialization**
   - Initialize `SAM3SemanticModel` with BPE path
   - Verify text encoder is loaded
   - Check model has semantic segmentation capabilities

3. **Test Text Prompt Encoding**
   - Provide text prompt (e.g., "person", "car", "dog")
   - Verify text is encoded correctly
   - Check encoded features have correct shape

4. **Test Text-Based Segmentation**
   - Load test image
   - Run segmentation with text prompt
   - Verify masks are generated for matching concepts
   - Check mask quality and accuracy

5. **Test Exemplar-Based Segmentation**
   - Provide exemplar box coordinates
   - Run segmentation with exemplar prompt
   - Verify masks match exemplar region
   - Check concept consistency

6. **Test Combined Prompts**
   - Use both text and exemplar prompts
   - Verify combined prompt processing
   - Check segmentation results

7. **Test Multiple Concepts**
   - Segment multiple concepts in single image
   - Verify all concepts are detected
   - Check mask separation

## Expected Results

- `SAM3SemanticModel` initializes with text encoder
- Text prompts are correctly encoded
- Text-based segmentation produces accurate masks
- Exemplar-based segmentation works correctly
- Combined prompts are processed properly
- Multiple concepts can be segmented simultaneously
- Masks are of high quality and match concepts

## Validation Criteria

- Model initializes with text encoder when BPE path provided
- Text encoding produces valid feature vectors
- Segmentation masks are generated for text prompts
- Exemplar prompts produce masks matching exemplar regions
- Combined prompts work correctly
- Multiple concepts are properly segmented
- Mask quality is acceptable (IoU > 0.7 for known objects)

## Dependencies

- BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`) - required for text encoding
- SAM3 weights file (`sam3.pt`)
- Test images with known objects
- Related tests: 01-02, 02-02, 02-03

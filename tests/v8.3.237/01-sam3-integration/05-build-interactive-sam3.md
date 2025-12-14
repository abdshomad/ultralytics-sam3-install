# Test 01-05: Build Interactive SAM3

## Test ID
01-05

## Test Name
Build Interactive SAM3

## Objective
Validate that `build_interactive_sam3` function correctly builds SAM3 tracker for interactive segmentation, including proper initialization of all components and support for interactive workflows.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`) available (optional)
- Test images available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics.models.sam.build_sam3 import build_interactive_sam3
   from ultralytics import SAM
   import numpy as np
   ```

2. **Test Basic Interactive SAM3 Building**
   - Call `build_interactive_sam3` with checkpoint path
   - Verify tracker is created successfully
   - Check all required components are initialized

3. **Test Interactive Segmentation Setup**
   - Initialize SAM with `sam3.pt`
   - Verify interactive predictor is created
   - Check predictor supports point/box prompts

4. **Test Interactive Workflow**
   - Load test image
   - Provide point prompts
   - Run interactive segmentation
   - Verify masks are generated
   - Test refinement with additional points

5. **Test Box Prompts**
   - Provide box coordinates
   - Run segmentation with box prompt
   - Verify mask matches box region
   - Check mask quality

6. **Test Multiple Prompts**
   - Combine point and box prompts
   - Verify combined prompt handling
   - Check segmentation results

7. **Test Feature Reuse**
   - Extract image features once
   - Reuse features for multiple queries
   - Verify efficiency improvement
   - Check results consistency

## Expected Results

- `build_interactive_sam3` successfully creates interactive tracker
- Interactive predictor supports point and box prompts
- Segmentation works with point prompts
- Segmentation works with box prompts
- Combined prompts are handled correctly
- Image features can be reused across queries
- Interactive workflow is smooth and responsive

## Validation Criteria

- Tracker is created without errors
- Interactive predictor is properly initialized
- Point prompts produce accurate masks
- Box prompts produce masks matching box regions
- Combined prompts work correctly
- Feature reuse improves efficiency (faster subsequent queries)
- Results are consistent across multiple queries
- Interactive workflow is functional

## Dependencies

- SAM3 weights file (`sam3.pt`) - must be manually downloaded
- BPE vocabulary file (optional, for semantic features)
- Test images for interactive segmentation
- Related tests: 01-04, 02-01, 02-06

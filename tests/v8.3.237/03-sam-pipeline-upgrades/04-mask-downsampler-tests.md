# Test 03-04: Mask Downsampler Tests

## Test ID
03-04

## Test Name
Mask Downsampler Tests

## Objective
Validate that `MaskDownSampler` has been updated to be more flexible, supporting interpolation to fixed sizes and higher-resolution mask handling.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- Test masks of various resolutions available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics.models.sam.modules.sam import MaskDownSampler
   import torch
   import numpy as np
   ```

2. **Test MaskDownSampler Initialization**
   - Initialize `MaskDownSampler` with various configurations
   - Verify downsampler is created successfully
   - Check configuration options

3. **Test Interpolation to Fixed Sizes**
   - Create masks of various sizes
   - Process through MaskDownSampler with fixed target size
   - Verify masks are interpolated correctly
   - Check interpolation quality

4. **Test Higher-Resolution Mask Handling**
   - Process high-resolution masks (e.g., 4K, 8K)
   - Verify downsampler handles high-res masks
   - Check memory efficiency
   - Test with various high resolutions

5. **Test Downsampling Quality**
   - Compare original and downsampled masks
   - Verify important mask features are preserved
   - Check edge preservation
   - Test IoU between original and downsampled

6. **Test Various Downsampling Ratios**
   - Test 2x downsampling
   - Test 4x downsampling
   - Test 8x downsampling
   - Verify all ratios work correctly

7. **Test Batch Processing**
   - Process batch of masks
   - Verify batch processing works
   - Check output shapes are correct

8. **Test Memory Efficiency**
   - Process large masks
   - Monitor memory usage
   - Verify memory is managed efficiently

## Expected Results

- `MaskDownSampler` initializes successfully
- Masks are interpolated to fixed sizes correctly
- High-resolution masks are handled properly
- Downsampling preserves important features
- Various downsampling ratios work correctly
- Batch processing works efficiently
- Memory usage is reasonable

## Validation Criteria

- MaskDownSampler initializes without errors
- Interpolation produces masks of correct size
- High-resolution masks are processed correctly
- Downsampling preserves mask quality (IoU > 0.85)
- All downsampling ratios work
- Batch processing produces correct output shapes
- Memory usage is acceptable
- No memory leaks occur

## Dependencies

- SAM3 weights file (`sam3.pt`)
- Test masks of various resolutions
- Related tests: 03-03, 03-06

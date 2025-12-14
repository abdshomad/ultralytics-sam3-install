# Test 03-03: Memory Encoder Flexibility

## Test ID
03-03

## Test Name
Memory Encoder Flexibility

## Objective
Validate that `MemoryEncoder` has been updated to be more flexible, supporting interpolation to fixed sizes and higher-resolution mask handling, especially for SAM3 video tracking scenarios.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- Test video file available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics.models.sam.modules.sam import MemoryEncoder
   from ultralytics import SAM
   import torch
   import numpy as np
   ```

2. **Test MemoryEncoder Initialization**
   - Initialize `MemoryEncoder` with various configurations
   - Verify encoder is created successfully
   - Check configuration options are respected

3. **Test Interpolation to Fixed Sizes**
   - Create masks of various sizes
   - Process through MemoryEncoder with fixed size target
   - Verify masks are interpolated to target size
   - Check interpolation quality

4. **Test Higher-Resolution Mask Handling**
   - Process high-resolution masks
   - Verify encoder handles high-res masks correctly
   - Check memory efficiency
   - Test with various resolutions

5. **Test Video Frame Processing**
   - Process video frames through MemoryEncoder
   - Verify memory is maintained across frames
   - Check frame-to-frame consistency
   - Test memory updates

6. **Test Custom Attention Modules**
   - Test MemoryEncoder with custom attention
   - Verify custom attention is accepted
   - Check attention mechanism works correctly

7. **Test Memory Efficiency**
   - Process multiple frames
   - Monitor memory usage
   - Verify memory is managed efficiently
   - Check no memory leaks

8. **Test Mask Quality Preservation**
   - Process masks through encoder
   - Verify mask quality is preserved
   - Check interpolation doesn't degrade quality significantly

## Expected Results

- `MemoryEncoder` initializes with various configurations
- Masks are interpolated to fixed sizes correctly
- High-resolution masks are handled properly
- Video frame processing works correctly
- Custom attention modules are supported
- Memory is managed efficiently
- Mask quality is preserved

## Validation Criteria

- MemoryEncoder initializes without errors
- Interpolation produces masks of correct size
- High-resolution masks are processed correctly
- Video frames are handled consistently
- Custom attention works when provided
- Memory usage is reasonable
- Mask quality is acceptable (IoU > 0.9 after interpolation)
- No memory leaks occur

## Dependencies

- SAM3 weights file (`sam3.pt`)
- Test video file
- Related tests: 03-04, 03-05, 02-04

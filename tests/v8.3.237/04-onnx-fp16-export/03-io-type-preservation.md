# Test 04-03: I/O Type Preservation

## Test ID
04-03

## Test Name
I/O Type Preservation

## Objective
Validate that `keep_io_types=True` is correctly implemented, ensuring input and output types remain FP32 even when model weights are converted to FP16.

## Prerequisites
- Ultralytics v8.3.237 installed
- YOLO model weights file available
- ONNX runtime installed
- CPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import YOLO
   import onnxruntime as ort
   import numpy as np
   ```

2. **Test Input Type Preservation**
   - Export model to ONNX FP16
   - Load exported model
   - Check input type is FP32
   - Verify FP32 inputs are accepted

3. **Test Output Type Preservation**
   - Run inference with FP32 inputs
   - Check output type is FP32
   - Verify outputs are in FP32 format

4. **Test keep_io_types Parameter**
   - Verify `keep_io_types=True` is used in conversion
   - Check parameter is correctly passed
   - Test parameter effect

5. **Test Inference with FP32 I/O**
   - Create FP32 input array
   - Run inference
   - Verify inference works correctly
   - Check output is FP32

6. **Test Type Consistency**
   - Verify input/output types are consistent
   - Check no type mismatches occur
   - Test multiple inference calls

7. **Test Compatibility**
   - Test with different ONNX runtime versions
   - Verify compatibility is maintained
   - Check no type errors occur

## Expected Results

- Input types remain FP32 after conversion
- Output types remain FP32 after conversion
- `keep_io_types=True` is correctly implemented
- FP32 inputs/outputs work correctly
- Type consistency is maintained
- Compatibility is preserved

## Validation Criteria

- Input type is FP32 (verified programmatically)
- Output type is FP32 (verified programmatically)
- `keep_io_types=True` is used in conversion
- FP32 inference works without errors
- Type consistency is maintained across calls
- No type mismatch errors occur
- Compatibility is preserved

## Dependencies

- YOLO model weights file
- ONNX runtime
- Related tests: 04-01, 04-02

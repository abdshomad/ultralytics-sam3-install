# Test 04-02: Weight Conversion

## Test ID
04-02

## Test Name
Weight Conversion

## Objective
Validate that model weights are correctly converted from FP32 to FP16 during ONNX export, maintaining model accuracy while reducing file size.

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

2. **Test Weight Conversion Process**
   - Load YOLO model
   - Export to ONNX FP32 (baseline)
   - Export to ONNX FP16
   - Compare weight formats

3. **Test Weight Value Accuracy**
   - Extract weights from FP32 model
   - Extract weights from FP16 model
   - Compare weight values
   - Verify conversion accuracy

4. **Test Weight Precision**
   - Check FP16 weights have correct precision
   - Verify no significant precision loss
   - Test critical weight layers

5. **Test Model Size Reduction**
   - Compare FP32 and FP16 model file sizes
   - Verify size reduction is approximately 50%
   - Check all weights are converted

6. **Test Weight Loading**
   - Load FP16 ONNX model
   - Verify weights load correctly
   - Check weight shapes are correct

7. **Test Conversion Completeness**
   - Verify all model weights are converted
   - Check no FP32 weights remain
   - Test conversion coverage

## Expected Results

- Weights are converted from FP32 to FP16
- Weight values are accurately converted
- Precision is maintained within acceptable limits
- Model size is reduced significantly
- Weights load correctly in FP16 format
- All weights are converted completely

## Validation Criteria

- Weight conversion completes successfully
- Weight values match within FP16 precision limits
- Precision loss is minimal (< 0.1% relative error)
- Model size reduction is > 40%
- All weights are in FP16 format
- Weight loading works correctly
- Conversion coverage is 100%
- No conversion errors occur

## Dependencies

- YOLO model weights file
- ONNX runtime
- Related tests: 04-01, 04-03

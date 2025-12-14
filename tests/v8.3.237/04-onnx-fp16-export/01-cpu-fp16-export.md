# Test 04-01: CPU FP16 Export

## Test ID
04-01

## Test Name
CPU FP16 Export

## Objective
Validate that ONNX export now supports `half=True` on CPU, converting model weights to FP16 using `onnxruntime.transformers.float16.convert_float_to_float16(keep_io_types=True)`.

## Prerequisites
- Ultralytics v8.3.237 installed
- YOLO model weights file available (e.g., `yolo11n.pt`)
- ONNX runtime installed
- CPU available (GPU not required for this test)

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import YOLO
   import onnxruntime as ort
   ```

2. **Test ONNX FP16 Export on CPU**
   - Load YOLO model
   - Export to ONNX with `half=True` on CPU
   - Verify export completes successfully
   - Check no errors occur

3. **Test Export File Generation**
   - Verify ONNX file is created
   - Check file size is reduced (FP16 should be ~50% of FP32)
   - Verify file is valid ONNX format

4. **Test Model Weight Conversion**
   - Load exported ONNX model
   - Verify weights are in FP16 format
   - Check weight values are correct
   - Verify conversion was successful

5. **Test I/O Type Preservation**
   - Verify input/output types remain FP32
   - Check `keep_io_types=True` is respected
   - Test model inference with FP32 inputs

6. **Test Model Functionality**
   - Load exported ONNX model
   - Run inference on test image
   - Verify predictions are correct
   - Compare with original model predictions

7. **Test Error Handling**
   - Test export with invalid model
   - Verify graceful error handling
   - Check error messages are clear

## Expected Results

- ONNX export with `half=True` works on CPU
- Export file is created successfully
- File size is reduced (approximately 50% of FP32)
- Weights are converted to FP16
- I/O types remain FP32
- Model functionality is preserved
- Error handling is graceful

## Validation Criteria

- Export completes without errors
- ONNX file is created and valid
- File size reduction is significant (> 40%)
- Weights are in FP16 format
- Input/output types are FP32
- Inference produces correct results
- Prediction accuracy matches original (mAP difference < 0.01)
- Errors are handled gracefully

## Dependencies

- YOLO model weights file
- ONNX runtime
- Test image for inference validation
- Related tests: 04-02, 04-03, 04-04

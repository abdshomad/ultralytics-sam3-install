# Test 01-04: SAM Entrypoint Detection

## Test ID
01-04

## Test Name
SAM Entrypoint Detection

## Objective
Verify that the SAM entrypoint correctly detects `sam3.pt` model files and routes to the appropriate SAM3 builder (`build_interactive_sam3`) instead of SAM/SAM2 builders.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- SAM/SAM2 weights files available for comparison (optional)
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import SAM
   from ultralytics.models.sam.model import SAM as SAMModel
   ```

2. **Test SAM3 Detection**
   - Initialize SAM with `sam3.pt` path
   - Verify model type is detected as SAM3
   - Check that `build_interactive_sam3` is called

3. **Test Model Routing**
   - Verify SAM3 model is created (not SAM or SAM2)
   - Check predictor type is SAM3Predictor or appropriate SAM3 variant
   - Verify model attributes match SAM3 structure

4. **Test Entrypoint with Different Paths**
   - Test with absolute path to `sam3.pt`
   - Test with relative path to `sam3.pt`
   - Test with path containing spaces
   - Verify all paths are handled correctly

5. **Test Error Handling**
   - Test with non-existent `sam3.pt` file
   - Verify appropriate error message
   - Test with invalid model file
   - Check error handling is graceful

6. **Test Model Type Detection**
   - Verify `sam3.pt` is distinguished from `sam.pt` and `sam2.pt`
   - Check correct builder is selected based on file name
   - Verify no conflicts between model types

## Expected Results

- SAM entrypoint correctly detects `sam3.pt` files
- `build_interactive_sam3` is called for SAM3 models
- SAM3 model is created (not SAM or SAM2)
- Correct predictor type is assigned
- Different path formats are handled correctly
- Appropriate errors are raised for invalid files
- Model type detection works correctly

## Validation Criteria

- SAM initialization with `sam3.pt` creates SAM3 model
- `build_interactive_sam3` function is invoked
- Model object is instance of SAM3Model or uses SAM3 components
- Predictor is SAM3Predictor or appropriate SAM3 variant
- Path handling works for absolute, relative, and paths with spaces
- Error messages are clear and helpful
- No conflicts with SAM/SAM2 model detection

## Dependencies

- SAM3 weights file (`sam3.pt`) - must be manually downloaded
- Related tests: 01-01, 01-05, 02-06

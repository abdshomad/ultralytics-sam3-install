# Test 01-02: SAM3 Model Initialization

## Test ID
01-02

## Test Name
SAM3 Model Initialization

## Objective
Verify that SAM3 models (`SAM3Model` and `SAM3SemanticModel`) can be properly initialized with various configurations and that all initialization parameters work correctly.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`) available (for semantic model)
- GPU available
- Test image available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics.models.sam.modules.sam import SAM3Model, SAM3SemanticModel
   from ultralytics import SAM
   ```

2. **Test Basic SAM3Model Initialization**
   - Initialize `SAM3Model` with default parameters
   - Verify model loads successfully
   - Check model attributes are set correctly

3. **Test SAM3SemanticModel Initialization**
   - Initialize `SAM3SemanticModel` with BPE path
   - Verify text encoder is initialized
   - Check model can process text prompts

4. **Test Model Loading from Checkpoint**
   - Load model from `sam3.pt` checkpoint
   - Verify weights are loaded correctly
   - Check model is in evaluation mode

5. **Test Model Configuration Options**
   - Test initialization with different image sizes
   - Test with/without compile option
   - Test with/without BPE path for semantic model

6. **Test Model Device Placement**
   - Verify model can be moved to CPU
   - Verify model can be moved to GPU if available
   - Check device placement is correct

## Expected Results

- `SAM3Model` initializes successfully with default parameters
- `SAM3SemanticModel` initializes successfully with BPE path
- Model weights load correctly from checkpoint
- Model is in evaluation mode after initialization
- Configuration options are respected
- Model can be placed on appropriate device

## Validation Criteria

- No errors during model initialization
- Model object is properly instantiated
- All required attributes exist and are set correctly
- Weights are loaded from checkpoint file
- Model is in evaluation mode (`model.training == False`)
- Device placement works correctly
- Text encoder exists in semantic model when BPE path provided

## Dependencies

- SAM3 weights file (`sam3.pt`) - must be manually downloaded
- BPE vocabulary file for semantic model tests
- Related tests: 01-01, 01-03, 02-01

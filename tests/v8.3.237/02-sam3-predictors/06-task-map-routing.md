# Test 02-06: Task Map Routing

## Test ID
02-06

## Test Name
Task Map Routing

## Objective
Validate that SAM3 predictors are correctly wired into `ultralytics.models.sam.__all__` and SAM's `task_map`, ensuring that `SAM("sam3.pt")` routes to the appropriate SAM3 predictor based on the task and model type.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`) available (for semantic predictors)
- Test images/videos available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import SAM
   from ultralytics.models.sam import __all__
   from ultralytics.models.sam.model import SAM as SAMModel
   ```

2. **Test SAM3 in __all__ Export**
   - Check that `SAM3Predictor`, `SAM3SemanticPredictor`, `SAM3VideoPredictor`, `SAM3VideoSemanticPredictor` are in `__all__`
   - Verify all SAM3 predictors are exported

3. **Test Task Map Registration**
   - Check SAM's `task_map` includes SAM3 entries
   - Verify `sam3.pt` is mapped to correct predictor
   - Check task routing logic

4. **Test Basic SAM3 Routing**
   - Initialize `SAM("sam3.pt")`
   - Verify correct predictor is created
   - Check predictor type is SAM3Predictor or appropriate variant

5. **Test Semantic Routing**
   - Initialize SAM3 with semantic capabilities
   - Verify SAM3SemanticPredictor is used when appropriate
   - Check text encoder is available

6. **Test Video Routing**
   - Initialize SAM3 for video tasks
   - Verify SAM3VideoPredictor or SAM3VideoSemanticPredictor is used
   - Check video tracker is initialized

7. **Test Predictor Selection Logic**
   - Test with different model configurations
   - Verify correct predictor is selected
   - Check predictor matches task requirements

8. **Test Predictor Exports**
   - Verify predictors are accessible via imports
   - Check direct instantiation works
   - Test predictor API consistency

## Expected Results

- All SAM3 predictors are in `__all__` export list
- SAM's `task_map` includes SAM3 entries
- `SAM("sam3.pt")` routes to correct predictor
- Semantic predictor is used when BPE path provided
- Video predictor is used for video tasks
- Predictor selection logic works correctly
- Predictors are accessible via imports
- API is consistent across predictors

## Validation Criteria

- `SAM3Predictor`, `SAM3SemanticPredictor`, `SAM3VideoPredictor`, `SAM3VideoSemanticPredictor` are in `__all__`
- `task_map` contains SAM3 entries
- `SAM("sam3.pt")` creates correct predictor type
- Predictor selection matches task requirements
- Direct imports work correctly
- Predictor APIs are consistent
- No routing errors occur

## Dependencies

- SAM3 weights file (`sam3.pt`)
- BPE vocabulary file (for semantic predictor tests)
- Related tests: 01-04, 02-01, 02-07

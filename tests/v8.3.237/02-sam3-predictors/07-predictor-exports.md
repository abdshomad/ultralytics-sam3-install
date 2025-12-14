# Test 02-07: Predictor Exports

## Test ID
02-07

## Test Name
Predictor Exports

## Objective
Validate that all SAM3 predictors are properly exported and accessible through the public API, including direct imports and usage through the SAM model interface.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`) available (for semantic predictors)
- GPU available

## Test Steps

1. **Test Direct Imports**
   ```python
   from ultralytics.models.sam.predict import (
       SAM3Predictor,
       SAM3SemanticPredictor,
       SAM3VideoPredictor,
       SAM3VideoSemanticPredictor
   )
   ```
   - Verify all imports succeed
   - Check no import errors

2. **Test __all__ Export**
   - Import from `ultralytics.models.sam`
   - Verify all SAM3 predictors are in `__all__`
   - Check export completeness

3. **Test Direct Instantiation**
   - Create `SAM3Predictor` directly
   - Create `SAM3SemanticPredictor` directly
   - Create `SAM3VideoPredictor` directly
   - Create `SAM3VideoSemanticPredictor` directly
   - Verify all instantiate correctly

4. **Test Through SAM Interface**
   - Initialize via `SAM("sam3.pt")`
   - Verify predictor is accessible
   - Check predictor type

5. **Test API Consistency**
   - Verify all predictors have consistent API
   - Check common methods exist
   - Test method signatures

6. **Test Predictor Attributes**
   - Check predictor has required attributes
   - Verify model attribute exists
   - Test configuration access

7. **Test Predictor Methods**
   - Test common methods (predict, __call__, etc.)
   - Verify methods work correctly
   - Check return types

## Expected Results

- All SAM3 predictors can be imported directly
- All predictors are in `__all__` export list
- Direct instantiation works for all predictors
- SAM interface provides access to predictors
- API is consistent across predictors
- Required attributes and methods exist
- Methods work correctly

## Validation Criteria

- Direct imports succeed without errors
- All predictors are exported in `__all__`
- Direct instantiation works correctly
- SAM interface routes to correct predictor
- API consistency is maintained
- Required attributes are present
- Methods execute successfully
- Return types are correct

## Dependencies

- SAM3 weights file (`sam3.pt`)
- BPE vocabulary file (for semantic predictors)
- Related tests: 02-01, 02-06

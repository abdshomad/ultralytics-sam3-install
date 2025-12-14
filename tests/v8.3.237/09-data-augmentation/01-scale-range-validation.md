# Test 09-01: Scale Range Validation

## Test ID
09-01

## Test Name
Scale Range Validation

## Objective
Validate that the `scale` hyperparameter range is correctly documented and enforced as `0.0 - 1.0` (instead of "≥ 0.0"), preventing unstable configurations.

## Prerequisites
- Ultralytics v8.3.237 installed
- YOLO model weights file available
- Training dataset available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import YOLO
   from ultralytics.cfg import get_cfg
   ```

2. **Test Documentation Update**
   - Check documentation for `scale` parameter
   - Verify range is documented as `0.0 - 1.0`
   - Check guide and macro tables are updated

3. **Test Scale Range Enforcement**
   - Try to set `scale` to value < 0.0
   - Verify value is rejected or clamped
   - Test with `scale=-0.1`

4. **Test Upper Bound**
   - Try to set `scale` to value > 1.0
   - Verify value is rejected or clamped
   - Test with `scale=1.5`, `scale=2.0`

5. **Test Valid Range Values**
   - Test with `scale=0.0` (minimum)
   - Test with `scale=0.5` (middle)
   - Test with `scale=1.0` (maximum)
   - Verify all valid values work

6. **Test Configuration Validation**
   - Load configuration with invalid scale
   - Verify validation catches invalid values
   - Check error messages are clear

7. **Test Training Stability**
   - Train with scale values in valid range
   - Verify training is stable
   - Check no instability issues occur

## Expected Results

- Documentation shows `scale` range as `0.0 - 1.0`
- Values < 0.0 are rejected or clamped
- Values > 1.0 are rejected or clamped
- Valid range values work correctly
- Configuration validation works
- Training is stable with valid values

## Validation Criteria

- Documentation is updated correctly
- Invalid values are rejected/clamped
- Valid values work correctly
- Configuration validation works
- Training stability is maintained
- No errors occur with valid values
- Clear error messages for invalid values

## Dependencies

- YOLO model weights file
- Training dataset
- Related tests: 09-02, 09-03

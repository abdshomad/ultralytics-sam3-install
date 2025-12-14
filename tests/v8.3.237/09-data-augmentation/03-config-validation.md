# Test 09-03: Config Validation

## Test ID
09-03

## Test Name
Config Validation

## Objective
Validate that configuration validation correctly enforces the `scale` parameter range and prevents unstable configurations from being used.

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

2. **Test Config Loading**
   - Load configuration with valid scale value
   - Verify config loads successfully
   - Check scale value is accepted

3. **Test Invalid Config Rejection**
   - Try to load config with `scale < 0.0`
   - Verify config is rejected or value is clamped
   - Check error/warning is issued

4. **Test Invalid Config Rejection (Upper)**
   - Try to load config with `scale > 1.0`
   - Verify config is rejected or value is clamped
   - Check error/warning is issued

5. **Test Config Validation Messages**
   - Trigger validation error
   - Verify error messages are clear
   - Check messages indicate correct range

6. **Test Config Override**
   - Load config with invalid scale
   - Override with valid value at runtime
   - Verify override works correctly

7. **Test Multiple Invalid Values**
   - Test various invalid scale values
   - Verify all are caught by validation
   - Check consistent validation behavior

## Expected Results

- Valid configs load successfully
- Invalid configs are rejected or clamped
- Error messages are clear and informative
- Config overrides work correctly
- Validation catches all invalid values
- Consistent validation behavior

## Validation Criteria

- Config loading works for valid values
- Invalid values are rejected/clamped
- Error messages are clear
- Config overrides work correctly
- All invalid values are caught
- Validation is consistent
- No validation errors for valid configs

## Dependencies

- YOLO model weights file
- Training dataset
- Related tests: 09-01, 09-02

# Test 06-02: Workers Override

## Test ID
06-02

## Test Name
Workers Override

## Objective
Validate that `workers` parameter can be overridden when resuming training, allowing modification of data loading workers without restarting the training process.

## Prerequisites
- Ultralytics v8.3.237 installed
- YOLO model weights file available
- Training dataset available
- Partially trained model checkpoint available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import YOLO
   ```

2. **Test Initial Training Setup**
   - Start training with initial `workers` value
   - Create checkpoint after some epochs
   - Verify training proceeds

3. **Test Resume with Workers Override**
   - Resume training from checkpoint
   - Override `workers` with different value
   - Verify override is accepted
   - Check training continues

4. **Test Workers Effect**
   - Monitor data loading performance
   - Verify new `workers` value is used
   - Check data loading speed changes

5. **Test Different Workers Values**
   - Test with `workers=0` (single-threaded)
   - Test with `workers=4`
   - Test with `workers=8`
   - Verify all values work correctly

6. **Test Override Persistence**
   - Verify override persists during training
   - Check workers count doesn't revert
   - Test multiple epochs

7. **Test Training Continuity**
   - Verify training continues smoothly after override
   - Check no data loading errors occur
   - Verify model state is preserved

## Expected Results

- `workers` can be overridden when resuming
- Override is accepted without errors
- New workers value is used for data loading
- Different workers values work correctly
- Override persists throughout training
- Training continues smoothly after override

## Validation Criteria

- Override parameter is accepted
- No errors occur during override
- Data loading uses new workers count
- All tested workers values work
- Override persists (verified across epochs)
- Training continues without interruption
- Model state is preserved

## Dependencies

- YOLO model weights file
- Training dataset
- Partially trained checkpoint
- Related tests: 06-01, 06-03 through 06-08

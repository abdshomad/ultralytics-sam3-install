# Test 06-05: Time Override

## Test ID
06-05

## Test Name
Time Override

## Objective
Validate that `time` parameter can be overridden when resuming training, allowing modification of training time limit without restarting the training process.

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
   - Start training with initial `time` value
   - Create checkpoint after some epochs
   - Verify training proceeds

3. **Test Resume with Time Override**
   - Resume training from checkpoint
   - Override `time` with different value
   - Verify override is accepted
   - Check training continues

4. **Test Time Effect**
   - Monitor training duration
   - Verify new `time` limit is used
   - Check training stops at new time limit

5. **Test Different Time Values**
   - Test with `time=3600` (1 hour)
   - Test with `time=7200` (2 hours)
   - Test with `time=None` (no limit)
   - Verify all values work correctly

6. **Test Override Persistence**
   - Verify override persists during training
   - Check time limit doesn't revert
   - Test training duration

7. **Test Training Continuity**
   - Verify training continues smoothly after override
   - Check time limit is respected
   - Verify model state is preserved

## Expected Results

- `time` can be overridden when resuming
- Override is accepted without errors
- New time limit is used
- Different time values work correctly
- Override persists throughout training
- Training continues smoothly after override

## Validation Criteria

- Override parameter is accepted
- No errors occur during override
- Training respects new time limit
- All tested time values work
- Override persists (verified during training)
- Training continues without interruption
- Model state is preserved

## Dependencies

- YOLO model weights file
- Training dataset
- Partially trained checkpoint
- Related tests: 06-01 through 06-04, 06-06 through 06-08

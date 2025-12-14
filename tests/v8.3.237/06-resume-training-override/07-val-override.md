# Test 06-07: Val Override

## Test ID
06-07

## Test Name
Val Override

## Objective
Validate that `val` parameter can be overridden when resuming training, allowing modification of validation frequency without restarting the training process.

## Prerequisites
- Ultralytics v8.3.237 installed
- YOLO model weights file available
- Training and validation datasets available
- Partially trained model checkpoint available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import YOLO
   ```

2. **Test Initial Training Setup**
   - Start training with initial `val` value
   - Create checkpoint after some epochs
   - Verify training proceeds

3. **Test Resume with Val Override**
   - Resume training from checkpoint
   - Override `val` with different value
   - Verify override is accepted
   - Check training continues

4. **Test Val Effect**
   - Monitor validation execution
   - Verify new `val` frequency is used
   - Check validation runs at new interval

5. **Test Different Val Values**
   - Test with `val=True` (validate every epoch)
   - Test with `val=False` (no validation)
   - Test with `val=5` (validate every 5 epochs)
   - Verify all values work correctly

6. **Test Override Persistence**
   - Verify override persists during training
   - Check validation frequency doesn't revert
   - Test multiple epochs

7. **Test Training Continuity**
   - Verify training continues smoothly after override
   - Check validation uses new frequency
   - Verify model state is preserved

## Expected Results

- `val` can be overridden when resuming
- Override is accepted without errors
- New validation frequency is used
- Different val values work correctly
- Override persists throughout training
- Training continues smoothly after override

## Validation Criteria

- Override parameter is accepted
- No errors occur during override
- Validation runs at new frequency
- All tested val values work
- Override persists (verified across epochs)
- Training continues without interruption
- Model state is preserved

## Dependencies

- YOLO model weights file
- Training and validation datasets
- Partially trained checkpoint
- Related tests: 06-01 through 06-06, 06-08

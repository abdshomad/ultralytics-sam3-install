# Test 06-01: Save Period Override

## Test ID
06-01

## Test Name
Save Period Override

## Objective
Validate that `save_period` parameter can be overridden when resuming training, allowing modification of checkpoint saving frequency without restarting the training process.

## Prerequisites
- Ultralytics v8.3.237 installed
- YOLO model weights file available
- Training dataset available
- Partially trained model checkpoint available (for resume)

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import YOLO
   ```

2. **Test Initial Training Setup**
   - Start training with initial `save_period` value
   - Create checkpoint after some epochs
   - Verify checkpoint is saved

3. **Test Resume with Save Period Override**
   - Resume training from checkpoint
   - Override `save_period` with different value
   - Verify override is accepted
   - Check training continues

4. **Test Save Period Effect**
   - Monitor checkpoint saving frequency
   - Verify new `save_period` is used
   - Check checkpoints are saved at new interval

5. **Test Different Save Period Values**
   - Test with `save_period=1` (every epoch)
   - Test with `save_period=5` (every 5 epochs)
   - Test with `save_period=10` (every 10 epochs)
   - Verify all values work correctly

6. **Test Override Persistence**
   - Verify override persists during training
   - Check save period doesn't revert to original
   - Test multiple epochs

7. **Test Training Continuity**
   - Verify training continues smoothly after override
   - Check no training interruption occurs
   - Verify model state is preserved

## Expected Results

- `save_period` can be overridden when resuming
- Override is accepted without errors
- New save period is used for subsequent checkpoints
- Different save period values work correctly
- Override persists throughout training
- Training continues smoothly after override

## Validation Criteria

- Override parameter is accepted
- No errors occur during override
- Checkpoints are saved at new interval
- All tested save period values work
- Override persists (verified across multiple epochs)
- Training continues without interruption
- Model state is preserved

## Dependencies

- YOLO model weights file
- Training dataset
- Partially trained checkpoint
- Related tests: 06-02 through 06-08

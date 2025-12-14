# Test 06-06: Freeze Override

## Test ID
06-06

## Test Name
Freeze Override

## Objective
Validate that `freeze` parameter can be overridden when resuming training, allowing modification of layer freezing without restarting the training process.

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
   - Start training with initial `freeze` value
   - Create checkpoint after some epochs
   - Verify training proceeds

3. **Test Resume with Freeze Override**
   - Resume training from checkpoint
   - Override `freeze` with different value
   - Verify override is accepted
   - Check training continues

4. **Test Freeze Effect**
   - Monitor layer freezing status
   - Verify new `freeze` configuration is used
   - Check frozen layers match new configuration

5. **Test Different Freeze Values**
   - Test with `freeze=None` (no freezing)
   - Test with `freeze=10` (freeze first 10 layers)
   - Test with `freeze=[0, 1, 2]` (freeze specific layers)
   - Verify all values work correctly

6. **Test Override Persistence**
   - Verify override persists during training
   - Check freeze configuration doesn't revert
   - Test multiple epochs

7. **Test Training Continuity**
   - Verify training continues smoothly after override
   - Check layer freezing is applied correctly
   - Verify model state is preserved

## Expected Results

- `freeze` can be overridden when resuming
- Override is accepted without errors
- New freeze configuration is used
- Different freeze values work correctly
- Override persists throughout training
- Training continues smoothly after override

## Validation Criteria

- Override parameter is accepted
- No errors occur during override
- Layer freezing matches new configuration
- All tested freeze values work
- Override persists (verified across epochs)
- Training continues without interruption
- Model state is preserved

## Dependencies

- YOLO model weights file
- Training dataset
- Partially trained checkpoint
- Related tests: 06-01 through 06-05, 06-07, 06-08

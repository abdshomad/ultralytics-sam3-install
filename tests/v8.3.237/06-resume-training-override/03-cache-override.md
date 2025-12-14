# Test 06-03: Cache Override

## Test ID
06-03

## Test Name
Cache Override

## Objective
Validate that `cache` parameter can be overridden when resuming training, allowing modification of dataset caching strategy without restarting the training process.

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
   - Start training with initial `cache` value
   - Create checkpoint after some epochs
   - Verify training proceeds

3. **Test Resume with Cache Override**
   - Resume training from checkpoint
   - Override `cache` with different value
   - Verify override is accepted
   - Check training continues

4. **Test Cache Effect**
   - Monitor dataset loading behavior
   - Verify new `cache` strategy is used
   - Check caching behavior changes

5. **Test Different Cache Values**
   - Test with `cache=False` (no caching)
   - Test with `cache="ram"` (RAM caching)
   - Test with `cache="disk"` (disk caching)
   - Verify all values work correctly

6. **Test Override Persistence**
   - Verify override persists during training
   - Check cache strategy doesn't revert
   - Test multiple epochs

7. **Test Training Continuity**
   - Verify training continues smoothly after override
   - Check no caching errors occur
   - Verify model state is preserved

## Expected Results

- `cache` can be overridden when resuming
- Override is accepted without errors
- New cache strategy is used
- Different cache values work correctly
- Override persists throughout training
- Training continues smoothly after override

## Validation Criteria

- Override parameter is accepted
- No errors occur during override
- Dataset uses new cache strategy
- All tested cache values work
- Override persists (verified across epochs)
- Training continues without interruption
- Model state is preserved

## Dependencies

- YOLO model weights file
- Training dataset
- Partially trained checkpoint
- Related tests: 06-01, 06-02, 06-04 through 06-08

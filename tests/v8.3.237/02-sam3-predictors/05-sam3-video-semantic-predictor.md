# Test 02-05: SAM3 Video Semantic Predictor

## Test ID
02-05

## Test Name
SAM3 Video Semantic Predictor

## Objective
Validate that `SAM3VideoSemanticPredictor` correctly performs video concept tracking using text prompts, boxes, and masklets, including semantic consistency across frames and concept-level tracking.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available
- BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`) available
- Test video file available
- GPU available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import SAM
   from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor
   import numpy as np
   ```

2. **Test Predictor Initialization**
   - Initialize `SAM3VideoSemanticPredictor` with `sam3.pt` and BPE path
   - Verify text encoder is loaded
   - Check video semantic tracker is initialized

3. **Test Text-Based Video Tracking**
   - Load test video
   - Provide text prompt (e.g., "person", "car")
   - Run semantic tracking across video
   - Verify concept is tracked consistently
   - Check all instances of concept are found

4. **Test Box + Text Tracking**
   - Provide initial box coordinates and text prompt
   - Run tracking with combined prompts
   - Verify tracking uses both box and text
   - Check tracking accuracy

5. **Test Masklet-Based Tracking**
   - Provide masklets from previous frames
   - Run tracking with masklet prompts
   - Verify masklet information is used
   - Check tracking consistency

6. **Test Combined Prompts (Text + Boxes + Masklets)**
   - Combine text, boxes, and masklets
   - Run tracking with all prompt types
   - Verify all prompts are utilized
   - Check tracking quality

7. **Test Concept Consistency**
   - Track concept across video
   - Verify concept identity is maintained
   - Check semantic consistency

8. **Test Multiple Concepts**
   - Track multiple concepts simultaneously
   - Verify all concepts are tracked
   - Check concept separation

9. **Test Tracking Memory**
   - Verify semantic memory is maintained
   - Check memory efficiency
   - Test memory updates across frames

## Expected Results

- `SAM3VideoSemanticPredictor` initializes with text encoder
- Text-based tracking works across video frames
- Box + text tracking combines both prompts effectively
- Masklet-based tracking uses previous frame information
- Combined prompts work correctly
- Concept consistency is maintained across frames
- Multiple concepts can be tracked simultaneously
- Memory management is efficient

## Validation Criteria

- Predictor initializes with text encoder and video tracker
- Text prompts track concepts across frames
- Combined prompts improve tracking accuracy
- Masklets enhance tracking consistency
- Concept identity is maintained (semantic consistency > 0.8)
- Multiple concepts are tracked separately
- Memory is managed efficiently
- Tracking accuracy is high (IoU > 0.7)

## Dependencies

- BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`) - required for text encoding
- SAM3 weights file (`sam3.pt`)
- Test video with multiple concepts
- Related tests: 02-02, 02-04, 03-07

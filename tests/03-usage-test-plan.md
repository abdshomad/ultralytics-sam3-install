# Test Plan: Usage Examples

## Overview

This test plan covers all SAM3 usage examples from README.md. These tests verify that the SAM3 integration works correctly for various segmentation tasks including text-based, visual prompts, exemplar-based, and video tracking.

## Test Organization

Tests are organized with 2-digit test numbers:
- **Test file naming**: `{test-number}-{test-name}.png` (screenshots)
- **Screenshot location**: `./screenshots/03-usage/{test-number}-{test-name}.png`

## Prerequisites

All usage tests require:
- Virtual environment activated
- SAM3 model file (`models/sam3.pt`)
- BPE vocabulary file (`models/bpe_simple_vocab_16e6.txt.gz`) - for semantic tests
- Test images (use available images from `submodules/sam3/assets/images/` or `submodules/inference/assets/`)
- GPU available (recommended but optional)

## Test Cases

### Test 01: Text-Based Concept Segmentation

**Location in README**: Lines 150-174

**Script**:
```python
from ultralytics.models.sam.predict import SAM3SemanticPredictor

# Initialize predictor with configuration
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="models/sam3.pt",  # Path to downloaded model
    half=True,  # Use FP16 for faster inference
)
predictor = SAM3SemanticPredictor(
    overrides=overrides,
    bpe_path="models/bpe_simple_vocab_16e6.txt.gz",  # Path to BPE vocabulary
)

# Set image once for multiple queries
predictor.set_image("path/to/image.jpg")

# Query with text prompts to find all instances
results = predictor(text=["person", "bus", "glasses"], save=True)

# Works with descriptive phrases
results = predictor(text=["person with red cloth", "person with blue cloth"], save=True)
```

**Prerequisites**: 
- Model file (`models/sam3.pt`)
- BPE vocabulary (`models/bpe_simple_vocab_16e6.txt.gz`)
- Test image (use `submodules/sam3/assets/images/test_image.jpg` or similar)

**Test Steps**:
1. Check if model and BPE files exist
2. Locate test image
3. Create Python script with example code (update image path)
4. Execute script
5. Capture output showing:
   - Predictor initialization
   - Image loading
   - Segmentation results
6. Save screenshot

**Expected Result**: Script executes successfully and performs text-based segmentation

**Screenshot**: `./screenshots/03-usage/01-text-based-segmentation.png`

---

### Test 02: Visual Prompts (Points and Boxes)

**Location in README**: Lines 180-203

**Script**:
```python
from ultralytics.models.sam.predict import SAM3Predictor

# Initialize predictor
overrides = dict(conf=0.25, task="segment", mode="predict", model="models/sam3.pt", half=True)
predictor = SAM3Predictor(overrides=overrides)

# Set image
predictor.set_image("path/to/image.jpg")

# Segment with point prompts
results = predictor(points=[[500, 375]], point_labels=[1], save=True)

# Segment with bounding box prompts
results = predictor(bboxes=[[100, 100, 200, 200]], save=True)

# Combine points and boxes
results = predictor(
    points=[[500, 375]],
    point_labels=[1],
    bboxes=[[100, 100, 200, 200]],
    save=True
)
```

**Prerequisites**: 
- Model file (`models/sam3.pt`)
- Test image (use `submodules/sam3/assets/images/test_image.jpg` or similar)

**Test Steps**:
1. Check if model file exists
2. Locate test image
3. Create Python script with example code (update image path and coordinates if needed)
4. Execute script
5. Capture output showing:
   - Predictor initialization
   - Point-based segmentation
   - Box-based segmentation
   - Combined prompts
6. Save screenshot

**Expected Result**: Script executes successfully and performs segmentation with visual prompts

**Screenshot**: `./screenshots/03-usage/02-visual-prompts-points-boxes.png`

---

### Test 03: Image Exemplar-Based Segmentation

**Location in README**: Lines 209-221

**Script**:
```python
from ultralytics.models.sam.predict import SAM3SemanticPredictor

# Initialize predictor
overrides = dict(conf=0.25, task="segment", mode="predict", model="models/sam3.pt", half=True)
predictor = SAM3SemanticPredictor(overrides=overrides, bpe_path="models/bpe_simple_vocab_16e6.txt.gz")

# Set image
predictor.set_image("path/to/image.jpg")

# Provide bounding box examples to segment similar objects
results = predictor(bboxes=[[480.0, 290.0, 590.0, 650.0]], save=True)
```

**Prerequisites**: 
- Model file (`models/sam3.pt`)
- BPE vocabulary (`models/bpe_simple_vocab_16e6.txt.gz`)
- Test image (use `submodules/sam3/assets/images/test_image.jpg` or similar)

**Test Steps**:
1. Check if model and BPE files exist
2. Locate test image
3. Create Python script with example code (update image path and bbox coordinates if needed)
4. Execute script
5. Capture output showing:
   - Predictor initialization
   - Exemplar-based segmentation
6. Save screenshot

**Expected Result**: Script executes successfully and performs exemplar-based segmentation

**Screenshot**: `./screenshots/03-usage/03-exemplar-based-segmentation.png`

---

### Test 04: Video Concept Tracking

**Location in README**: Lines 227-244

**Script**:
```python
from ultralytics.models.sam.predict import SAM3VideoPredictor

# Create video predictor
overrides = dict(conf=0.25, task="segment", mode="predict", model="models/sam3.pt", half=True)
predictor = SAM3VideoPredictor(overrides=overrides)

# Track objects using bounding box prompts
results = predictor(
    source="path/to/video.mp4",
    bboxes=[[706.5, 442.5, 905.25, 555], [598, 635, 725, 750]],
    stream=True
)

# Process and display results
for r in results:
    r.show()  # Display frame with segmentation masks
```

**Prerequisites**: 
- Model file (`models/sam3.pt`)
- Test video file (may not be available)

**Test Steps**:
1. Check if model file exists
2. Check if test video is available
3. If video not available, skip test and document in screenshot
4. If available, create Python script with example code (update video path and bbox coordinates)
5. Execute script (may process only first few frames for testing)
6. Capture output showing:
   - Predictor initialization
   - Video processing
   - Frame results
7. Save screenshot

**Expected Result**: Script executes successfully and processes video frames (or shows skip message if video unavailable)

**Note**: This test may be skipped if no test video is available

**Screenshot**: `./screenshots/03-usage/04-video-concept-tracking.png`

---

### Test 05: GPU Usage Example

**Location in README**: Lines 250-270

**Script**:
```python
from ultralytics.models.sam.predict import SAM3SemanticPredictor

# Initialize predictor with device specification
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="models/sam3.pt",
    device=0,  # Use GPU 0
    half=True,
)
predictor = SAM3SemanticPredictor(
    overrides=overrides,
    bpe_path="models/bpe_simple_vocab_16e6.txt.gz",
)

# Run inference on GPU
predictor.set_image("path/to/image.jpg")
results = predictor(text=["person"], save=True)
```

**Prerequisites**: 
- Model file (`models/sam3.pt`)
- BPE vocabulary (`models/bpe_simple_vocab_16e6.txt.gz`)
- Test image (use `submodules/sam3/assets/images/test_image.jpg` or similar)
- GPU available

**Test Steps**:
1. Check if model and BPE files exist
2. Check if GPU is available
3. Locate test image
4. Create Python script with example code (update image path)
5. Execute script
6. Capture output showing:
   - Predictor initialization with device=0
   - GPU usage confirmation
   - Segmentation results
7. Save screenshot

**Expected Result**: Script executes successfully and uses GPU for inference

**Screenshot**: `./screenshots/03-usage/05-gpu-usage-example.png`

## Execution Guidelines

1. **Prerequisites Check**: Before each test, verify required files and resources
2. **Test Images**: Use available images from submodules if test images not specified
3. **Model Files**: Check for model files before running tests
4. **Error Handling**: If prerequisites are missing, document in screenshot
5. **Output Capture**: Capture both stdout and stderr for all scripts
6. **Performance**: Tests may take time due to model loading and inference

## Dependencies

- All tests require model file (`models/sam3.pt`) from installation
- Tests 01, 03, 05 require BPE vocabulary file
- Tests should be run after installation tests complete
- GPU is recommended but not required (will fall back to CPU)

## Notes

- Model loading takes time on first use
- Inference speed depends on GPU availability
- Test images should be appropriate for segmentation tasks
- Video test may be skipped if no test video is available
- All tests should save results (save=True) for verification
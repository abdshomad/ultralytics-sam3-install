# Test 00-05: GPU Usage

## Test ID
00-05

## Test Name
GPU Usage

## Objective
Validate that `SAM3SemanticPredictor` correctly uses GPU when explicitly specified, including proper device selection, GPU memory management, and verification of GPU utilization. This test implements the README usage example (lines 246-270).

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`models/sam3.pt`) available
- BPE vocabulary file (`models/bpe_simple_vocab_16e6.txt.gz`) available
- Test images available from submodules
- GPU available (recommended, but test will work with CPU)

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics.models.sam.predict import SAM3SemanticPredictor
   import numpy as np
   from PIL import Image
   import matplotlib.pyplot as plt
   import torch
   ```

2. **Check GPU Availability**
   - Check if CUDA is available using `torch.cuda.is_available()`
   - Get GPU device information (name, count)
   - Determine device to use (GPU 0 or CPU)

3. **Check Requirements**
   - Verify model file exists at `models/sam3.pt`
   - Verify BPE vocabulary exists at `models/bpe_simple_vocab_16e6.txt.gz`
   - Locate test image from submodules

4. **Initialize Predictor with Device Specification**
   - Initialize `SAM3SemanticPredictor` with model path and BPE vocabulary
   - Configure with explicit device specification (device=0 for GPU 0)
   - Configure with appropriate overrides (conf=0.25, task="segment", mode="predict", half=True)
   - Verify predictor is created successfully with correct device

5. **Set Image**
   - Load test image using `predictor.set_image()`
   - Store original image for visualization

6. **Run Inference on GPU**
   - Provide text prompt (e.g., "person")
   - Run semantic segmentation
   - Verify inference completes successfully
   - Check GPU memory usage if GPU available

7. **Create Side-by-Side Visualizations**
   - Display original image, segmentation results, and GPU information side-by-side
   - Include GPU device name, memory usage, and device status
   - Save visualization outputs to `tests/v8.3.237/00-basic/outputs/`
   - Verify visualizations are created correctly

## Expected Results

- GPU availability is correctly detected
- `SAM3SemanticPredictor` initializes with specified device
- Inference runs successfully on GPU (or CPU if GPU not available)
- GPU memory is utilized when GPU is available
- Segmentation results are accurate
- Side-by-side visualizations are created and saved successfully
- GPU information is displayed correctly

## Validation Criteria

- GPU availability check works correctly
- Predictor initializes with correct device specification
- Inference completes without errors
- GPU memory is allocated when GPU is available
- Segmentation results match expected output
- Visualizations display correctly with original, results, and GPU info
- GPU information is accurate and informative

## Dependencies

- BPE vocabulary file (`models/bpe_simple_vocab_16e6.txt.gz`) - required for text encoding
- SAM3 weights file (`models/sam3.pt`)
- Test images from `submodules/sam3/assets/images/` or `submodules/inference/assets/`
- PyTorch with CUDA support (for GPU usage)
- Related tests: 00-01, 00-02, 00-03

## Output Files

- `tests/v8.3.237/00-basic/outputs/05-gpu-usage.png` - Visualization showing original image, results, and GPU information

## Notes

- This test will work with or without GPU availability
- If GPU is not available, the test will use CPU and display appropriate information
- GPU memory information is only available when CUDA is available

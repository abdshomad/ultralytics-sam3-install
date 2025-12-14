# Test 01-01: SAM3 Model Builder Stack

## Test ID
01-01

## Test Name
SAM3 Model Builder Stack

## Objective
Validate that the SAM 3 model builder stack (`build_sam3.py`) correctly constructs models with all required components: ViT backbone, transformer encoder/decoder, text encoder, geometry encoders, and video tracker.

## Prerequisites
- Ultralytics v8.3.237 installed
- SAM3 weights file (`sam3.pt`) available (manually downloaded)
- BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`) available
- GPU available for model initialization
- Python environment with required dependencies

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics.models.sam.build_sam3 import build_sam3_image_model, build_sam3_video_model
   import torch
   ```

2. **Test Image Model Builder**
   - Call `build_sam3_image_model()` with valid checkpoint path
   - Verify model structure includes:
     - ViT backbone
     - Transformer encoder/decoder
     - Text encoder (if BPE path provided)
     - Geometry encoders

3. **Test Video Model Builder**
   - Call `build_sam3_video_model()` with valid checkpoint path
   - Verify model structure includes:
     - All image model components
     - Video tracker component

4. **Test Model Components**
   - Verify backbone is instance of ViT-based architecture
   - Verify encoder/decoder are transformer-based
   - Verify text encoder exists when BPE path provided
   - Verify geometry encoders are present

5. **Test Model Forward Pass**
   - Create dummy input tensors
   - Run forward pass through model
   - Verify output shapes are correct

## Expected Results

- `build_sam3_image_model()` successfully creates a model with all required components
- `build_sam3_video_model()` successfully creates a model with video tracking capability
- All model components are properly initialized
- Forward pass completes without errors
- Output shapes match expected dimensions

## Validation Criteria

- Model builder functions execute without errors
- Model object is instance of `SAM3Model` or `SAM3SemanticModel`
- Model has `backbone`, `encoder`, `decoder` attributes
- Text encoder exists when BPE path is provided
- Video model has tracker component
- Forward pass produces valid outputs with correct shapes
- Model can be moved to GPU if available

## Dependencies

- SAM3 weights file must be manually downloaded (not auto-downloaded)
- BPE vocabulary file required for text encoder functionality
- Related tests: 01-02, 01-03

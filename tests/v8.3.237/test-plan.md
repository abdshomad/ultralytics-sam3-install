# Test Plan for Ultralytics v8.3.237

## Overview

This test plan covers all features introduced in Ultralytics v8.3.237 release, including SAM 3 integration, export improvements, training enhancements, and bug fixes.

**Release Reference**: [v8.3.237 Release Notes](https://github.com/ultralytics/ultralytics/releases/tag/v8.3.237)

## Test Plan Structure

Tests are organized using a numbered structure with descriptive category folder names:
- **2-digit category folders with names** (01-sam3-integration/, 02-sam3-predictors/, etc.) representing major feature areas
- **2-digit test numbers** (01, 02, 03...) within each category
- **Test file naming**: `{test-number}-{test-name}.md`

### Directory Structure

```
tests/v8.3.237/
├── 01-sam3-integration/          # SAM 3 Integration Tests (5 tests)
├── 02-sam3-predictors/           # SAM 3 Predictors & APIs Tests (7 tests)
├── 03-sam-pipeline-upgrades/     # SAM Pipeline Upgrades Tests (7 tests)
├── 04-onnx-fp16-export/          # ONNX FP16 Export Tests (4 tests)
├── 05-edgetpu-imx-export/        # Edge TPU & IMX Export Tests (4 tests)
├── 06-resume-training-override/  # Resume Training Override Tests (8 tests)
├── 07-rtdetr-validation/         # RT-DETR Validation Tests (3 tests)
├── 08-obb-plotting/              # OBB Plotting Tests (4 tests)
├── 09-data-augmentation/         # Data Augmentation Tests (3 tests)
└── 10-integration-tests/          # Integration Tests (3 tests)
```

## Test Categories

### Category 01: SAM 3 Integration (01-sam3-integration/)
Tests for SAM 3 model builder stack, model initialization, and entrypoint detection.

**Key Features:**
- New SAM 3 model builder stack (`build_sam3.py`) with ViT backbone, transformer encoder/decoder, text encoder, geometry encoders, and video tracker
- `SAM3Model`, `SAM3SemanticModel` and SAM3-specific modules
- SAM entrypoint detection for `sam3.pt` and routing via `build_interactive_sam3`

### Category 02: SAM 3 Predictors (02-sam3-predictors/)
Tests for all four SAM3 predictor classes and their routing through task_map.

**Key Features:**
- `SAM3Predictor` – SAM3-style interactive segmentation
- `SAM3SemanticPredictor` – text & exemplar based concept segmentation on images
- `SAM3VideoPredictor` – video tracking with box prompts
- `SAM3VideoSemanticPredictor` – video concept tracking (text + boxes + masklets)
- Integration into `ultralytics.models.sam.__all__` and SAM's `task_map`

### Category 03: SAM Pipeline Upgrades (03-sam-pipeline-upgrades/)
Tests for stride handling, square image enforcement, memory encoding, and mask processing improvements.

**Key Features:**
- `Predictor.setup_source` now accepts explicit `stride` parameter
- Square image size enforcement and consistent feature shapes
- More flexible `MemoryEncoder` and `MaskDownSampler`
- Memory attention with custom attention modules (RoPE-based attention for SAM3)
- `SAM3Model` mask post-processing and non-overlap suppression

### Category 04: ONNX FP16 Export (04-onnx-fp16-export/)
Tests for CPU-based FP16 export functionality and error handling.

**Key Features:**
- ONNX export supports `half=True` on CPU
- Converts model weights to FP16 using `onnxruntime.transformers.float16.convert_float_to_float16(keep_io_types=True)`
- Graceful failure handling with warnings instead of aborting export

### Category 05: Edge TPU & IMX Export (05-edgetpu-imx-export/)
Tests for dependency management and apt requirement checking.

**Key Features:**
- `export_edgetpu`: shell `apt-get` calls replaced with centralized `check_apt_requirements(["edgetpu-compiler"])`
- `export_imx`: Java installs use `check_apt_requirements()` for `openjdk-21-jre` (Ubuntu/Debian Trixie) and `openjdk-17-jre` (Raspberry Pi/Debian Bookworm)
- `check_apt_requirements()` runs `apt update` with `check=True`, surfacing update failures

### Category 06: Resume Training Override (06-resume-training-override/)
Tests for overriding various training parameters during resume.

**Key Features:**
- Overrideable parameters: `save_period`, `workers`, `cache`, `patience`, `time`, `freeze`, `val`, `plots`
- Runtime/logging parameter modification without restarting training

### Category 07: RT-DETR Validation (07-rtdetr-validation/)
Tests for validation transform simplification and scaling fixes.

**Key Features:**
- Simplified RT-DETR validation transforms
- Removed custom `ratio_pad` injection, replaced with clean `Compose([])`
- Added no-op `scale_preds()` override for explicit and safe scaling behavior

### Category 08: OBB Plotting (08-obb-plotting/)
Tests for improved OBB plotting robustness and format handling.

**Key Features:**
- `OBBValidator.plot_predictions()` accepts prediction dicts instead of raw tensors
- Early-return for empty predictions
- Direct use of `plot_images()`, avoiding redundant `xywh2xyxy` conversions

### Category 09: Data Augmentation (09-data-augmentation/)
Tests for scale range validation and configuration checks.

**Key Features:**
- `scale` hyperparameter range updated from "≥ 0.0" to **`0.0 - 1.0`**
- Documentation updated in guide and macro tables

### Category 10: Integration Tests (10-integration-tests/)
End-to-end tests and cross-feature compatibility validation.

**Key Features:**
- Complete SAM3 workflows from model loading to prediction
- Cross-feature compatibility testing
- Performance benchmarks

## Test File Format

Each test file contains:
- **Test ID**: Category-TestNumber (e.g., 01-01)
- **Test Name**: Descriptive name
- **Objective**: What the test validates
- **Prerequisites**: Required setup, models, data, dependencies
- **Test Steps**: Detailed step-by-step instructions
- **Expected Results**: What should happen
- **Validation Criteria**: How to verify success
- **Dependencies**: Related tests or features

## Execution Guidelines

1. **Prerequisites**: Ensure all dependencies are installed, including SAM3 weights (`sam3.pt`) and BPE vocabulary (`bpe_simple_vocab_16e6.txt.gz`)
2. **Test Order**: Tests within categories can generally be run independently, but integration tests (Category 10) should be run after other categories
3. **Environment**: Tests should be run in the project's virtual environment with GPU support available
4. **Data**: Use test images/videos from Ultralytics assets or provide appropriate test data

## Test Coverage Summary

- **Total Test Categories**: 10
- **Total Test Files**: 48
- **Coverage Areas**: SAM3 integration, predictors, pipeline upgrades, exports, training, validation, plotting, augmentation, integration

## References

- [Ultralytics v8.3.237 Release Notes](https://github.com/ultralytics/ultralytics/releases/tag/v8.3.237)
- [Ultralytics Documentation](https://docs.ultralytics.com)
- [SAM3 Repository](https://github.com/facebookresearch/sam3)

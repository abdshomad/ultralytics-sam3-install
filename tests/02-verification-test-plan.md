# Test Plan: Verification Scripts

## Overview

This test plan covers all GPU and system verification scripts from README.md. These tests verify that the system is properly configured with GPU support and CUDA availability.

## Test Organization

Tests are organized with 2-digit test numbers:
- **Test file naming**: `{test-number}-{test-name}.png` (screenshots)
- **Screenshot location**: `./screenshots/02-verification/{test-number}-{test-name}.png`

## Test Cases

### Test 01: GPU Verification Python Script

**Location in README**: Lines 116-126

**Script**:
```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
```

**Prerequisites**: 
- Virtual environment activated
- PyTorch installed

**Test Steps**:
1. Activate virtual environment
2. Create temporary Python script with the verification code
3. Execute script
4. Capture output showing:
   - CUDA availability status
   - CUDA version
   - Number of GPUs
   - GPU names and memory
5. Save screenshot

**Expected Result**: Script executes successfully and displays GPU information

**Screenshot**: `./screenshots/02-verification/01-gpu-verification-python.png`

---

### Test 02: NVIDIA SMI

**Location in README**: Line 313

**Command**:
```bash
nvidia-smi
```

**Prerequisites**: NVIDIA drivers installed

**Test Steps**:
1. Check if `nvidia-smi` command is available
2. Execute `nvidia-smi` command
3. Capture output showing:
   - Driver version
   - CUDA version
   - GPU information
   - Memory usage
4. Save screenshot

**Expected Result**: Command executes and displays NVIDIA GPU information

**Note**: If command not available, screenshot should show error message

**Screenshot**: `./screenshots/02-verification/02-nvidia-smi.png`

---

### Test 03: NVCC Version

**Location in README**: Line 318

**Command**:
```bash
nvcc --version
```

**Prerequisites**: CUDA toolkit installed

**Test Steps**:
1. Check if `nvcc` command is available
2. Execute `nvcc --version` command
3. Capture output showing CUDA compiler version
4. Save screenshot

**Expected Result**: Command executes and displays CUDA compiler version

**Note**: If command not available, screenshot should show error message

**Screenshot**: `./screenshots/02-verification/03-nvcc-version.png`

---

### Test 04: PyTorch CUDA Check

**Location in README**: Lines 322-325

**Script**:
```python
import torch
print(torch.cuda.is_available())
```

**Prerequisites**: 
- Virtual environment activated
- PyTorch installed

**Test Steps**:
1. Activate virtual environment
2. Create temporary Python script with the check code
3. Execute script
4. Capture output (True/False)
5. Save screenshot

**Expected Result**: Script executes and prints `True` or `False` for CUDA availability

**Screenshot**: `./screenshots/02-verification/04-pytorch-cuda-check.png`

## Execution Guidelines

1. **Prerequisites Check**: Before each test, verify required prerequisites
2. **Virtual Environment**: Tests 01 and 04 require activated virtual environment
3. **Error Handling**: If prerequisites are missing, capture error message in screenshot
4. **Output Capture**: Capture both stdout and stderr for all commands
5. **Non-Critical**: Missing GPU/CUDA should not cause test failures, just document in screenshot

## Dependencies

- Tests 01 and 04 require PyTorch to be installed (from installation tests)
- Tests 02 and 03 are system-level checks that don't depend on Python environment
- All tests can be run independently

## Notes

- GPU availability is optional but recommended for SAM3 usage
- If GPU is not available, tests should still complete but show appropriate messages
- These verification tests are useful for troubleshooting installation issues
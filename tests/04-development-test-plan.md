# Test Plan: Development Scripts

## Overview

This test plan covers all development and maintenance scripts from README.md. These tests verify that development workflows and maintenance commands work correctly.

## Test Organization

Tests are organized with 2-digit test numbers:
- **Test file naming**: `{test-number}-{test-name}.png` (screenshots)
- **Screenshot location**: `./screenshots/04-development/{test-number}-{test-name}.png`

## Test Cases

### Test 01: Update Submodules

**Location in README**: Line 286

**Command**:
```bash
git submodule update --remote --recursive
```

**Prerequisites**: Git repository with submodules

**Test Steps**:
1. Verify we're in a git repository
2. Check if submodules exist
3. Execute submodule update command
4. Capture output showing:
   - Submodule update progress
   - Any changes pulled
5. Save screenshot

**Expected Result**: Command executes successfully and updates all submodules to latest commits

**Screenshot**: `./screenshots/04-development/01-update-submodules.png`

---

### Test 02: UV Sync Dependencies

**Location in README**: Lines 299-300

**Command**:
```bash
source .venv/bin/activate
uv sync --no-install-project
```

**Prerequisites**: 
- Virtual environment exists (`.venv/`)
- `pyproject.toml` file exists
- `uv` installed

**Test Steps**:
1. Check if `.venv/` directory exists
2. Check if `pyproject.toml` exists
3. Activate virtual environment
4. Execute `uv sync --no-install-project`
5. Capture output showing:
   - Virtual environment activation
   - Dependency synchronization
   - Any packages installed/updated
6. Save screenshot

**Expected Result**: Command executes successfully and syncs all dependencies from `pyproject.toml`

**Note**: The `--no-install-project` flag prevents trying to install this project itself

**Screenshot**: `./screenshots/04-development/02-uv-sync-dependencies.png`

---

### Test 03: Manual Model Download

**Location in README**: Lines 339-341

**Command**:
```bash
source .venv/bin/activate
python3 -c "from huggingface_hub import hf_hub_download; import os; hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt', token=os.environ.get('HF_TOKEN'), local_dir='models')"
```

**Prerequisites**: 
- Virtual environment activated
- `huggingface_hub` installed
- `HF_TOKEN` environment variable set (or in `.env` file)
- Access to `facebook/sam3` repository on Hugging Face

**Test Steps**:
1. Check if virtual environment exists
2. Check if `HF_TOKEN` is available (from `.env` or environment)
3. If token not available, skip test and document in screenshot
4. If token available:
   - Activate virtual environment
   - Load HF_TOKEN from environment or `.env`
   - Execute model download command
   - Capture output showing:
     - Download progress
     - File location
5. Save screenshot

**Expected Result**: Command executes successfully and downloads `sam3.pt` to `models/` directory

**Note**: This test may be skipped if HF_TOKEN is not available

**Screenshot**: `./screenshots/04-development/03-manual-model-download.png`

## Execution Guidelines

1. **Prerequisites Check**: Before each test, verify required prerequisites
2. **Virtual Environment**: Tests 02 and 03 require activated virtual environment
3. **Error Handling**: If prerequisites are missing, document in screenshot
4. **Output Capture**: Capture both stdout and stderr for all commands
5. **Non-Destructive**: Tests should not break existing setup
6. **Token Security**: HF_TOKEN should be loaded from `.env` file, not hardcoded

## Dependencies

- Test 02 requires `pyproject.toml` from installation
- Test 03 requires Hugging Face access and token
- All tests can be run independently
- Tests should be run after installation tests complete

## Notes

- Submodule update may take time depending on network speed
- UV sync is fast if dependencies are already installed
- Model download requires Hugging Face access and can take several minutes
- Manual model download is only needed if automatic download during installation failed
- HF_TOKEN should be kept secure and not exposed in screenshots
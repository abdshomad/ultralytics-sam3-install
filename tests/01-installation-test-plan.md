# Test Plan: Installation Scripts

## Overview

This test plan covers all installation and setup scripts from README.md. These tests verify that the installation process works correctly and all setup commands execute successfully.

## Test Organization

Tests are organized with 2-digit test numbers:
- **Test file naming**: `{test-number}-{test-name}.png` (screenshots)
- **Screenshot location**: `./screenshots/01-installation/{test-number}-{test-name}.png`

## Test Cases

### Test 01: Git Clone with Submodules

**Location in README**: Lines 45-47

**Command**:
```bash
git clone --recurse-submodules https://github.com/your-username/ultralytics-sam3-install.git
cd ultralytics-sam3-install
```

**Prerequisites**: None (will check if already cloned)

**Test Steps**:
1. Check if repository is already cloned
2. If not, execute git clone command (or show dry-run)
3. Verify submodules are initialized
4. Capture screenshot of output

**Expected Result**: Repository cloned with all submodules initialized

**Screenshot**: `./screenshots/01-installation/01-git-clone-submodules.png`

---

### Test 02: Git Submodule Update

**Location in README**: Line 53

**Command**:
```bash
git submodule update --init --recursive
```

**Prerequisites**: Git repository with submodules

**Test Steps**:
1. Verify we're in a git repository
2. Check if submodules exist
3. Execute submodule update command
4. Capture screenshot of output

**Expected Result**: All submodules initialized and updated

**Screenshot**: `./screenshots/01-installation/02-git-submodule-update.png`

---

### Test 03: Copy Environment Example

**Location in README**: Line 72

**Command**:
```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

**Prerequisites**: `.env.example` file exists

**Test Steps**:
1. Check if `.env.example` exists
2. Check if `.env` already exists (skip if it does)
3. Execute copy command
4. Verify `.env` file created
5. Capture screenshot of output

**Expected Result**: `.env` file created from `.env.example`

**Screenshot**: `./screenshots/01-installation/03-copy-env-example.png`

---

### Test 04: Installation Script

**Location in README**: Line 92

**Command**:
```bash
./install.sh
```

**Prerequisites**: 
- `uv` installed
- Git submodules initialized
- CUDA available (optional, for GPU support)

**Test Steps**:
1. Verify `uv` is installed
2. Check if virtual environment already exists (may skip some steps)
3. Execute installation script
4. Monitor output for key steps:
   - uv check
   - Submodule initialization
   - CUDA detection
   - Python 3.12 installation
   - Virtual environment creation
   - PyTorch installation
   - Dependency installation
   - Model download (if HF_TOKEN set)
   - GPU verification
5. Capture screenshot of full output

**Expected Result**: Installation completes successfully with all components installed

**Note**: If already installed, script should skip redundant steps

**Screenshot**: `./screenshots/01-installation/04-install-script.png`

---

### Test 05: Activate Virtual Environment

**Location in README**: Line 109

**Command**:
```bash
source .venv/bin/activate
```

**Prerequisites**: `.venv/` directory exists (created by install.sh)

**Test Steps**:
1. Check if `.venv/` directory exists
2. Execute activation command
3. Verify activation by checking:
   - `which python` points to `.venv/bin/python`
   - `VIRTUAL_ENV` environment variable is set
4. Capture screenshot showing activated environment

**Expected Result**: Virtual environment successfully activated

**Screenshot**: `./screenshots/01-installation/05-activate-venv.png`

## Execution Guidelines

1. **Prerequisites Check**: Before each test, verify required prerequisites
2. **Non-Destructive**: Tests should not break existing setup
3. **Skip Logic**: If prerequisites are missing, document in screenshot
4. **Output Capture**: Capture both stdout and stderr for all commands
5. **Error Handling**: If a test fails, capture error message in screenshot

## Dependencies

- Test 04 (install.sh) should be run before Test 05 (activate venv)
- Tests 01-03 can be run in any order
- Test 04 depends on Tests 01-02 completing successfully

## Notes

- Installation script may take 10-15 minutes on first run
- Subsequent runs should be faster (2-5 minutes) as PyTorch installation is skipped
- Model download requires HF_TOKEN in `.env` file
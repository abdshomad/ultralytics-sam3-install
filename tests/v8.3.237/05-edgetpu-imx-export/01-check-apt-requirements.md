# Test 05-01: Check APT Requirements

## Test ID
05-01

## Test Name
Check APT Requirements

## Objective
Validate that `check_apt_requirements()` function correctly checks and installs apt packages, runs `apt update` with `check=True`, and surfaces update failures instead of silently ignoring them.

## Prerequisites
- Ultralytics v8.3.237 installed
- Linux system (Ubuntu/Debian) with apt package manager
- Sudo/root access for apt operations (or test in environment that allows it)
- YOLO model weights file available

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics.utils.checks import check_apt_requirements
   ```

2. **Test Function Availability**
   - Verify `check_apt_requirements` function exists
   - Check function is callable
   - Verify function signature

3. **Test Single Package Check**
   - Call `check_apt_requirements(["package-name"])`
   - Verify function executes
   - Check return value/status

4. **Test Multiple Package Check**
   - Call with multiple packages
   - Verify all packages are checked
   - Check function handles multiple packages

5. **Test APT Update Execution**
   - Call function and monitor apt update
   - Verify `apt update` is run with `check=True`
   - Check update failures are surfaced

6. **Test Error Handling**
   - Test with non-existent package
   - Verify error is raised/surfaced
   - Check error message is clear

7. **Test Update Failure Surfacing**
   - Simulate apt update failure (if possible)
   - Verify failure is not silently ignored
   - Check error is properly reported

8. **Test Package Installation**
   - Test with package that needs installation
   - Verify package is installed if missing
   - Check installation success

## Expected Results

- `check_apt_requirements` function exists and is callable
- Single package checks work correctly
- Multiple package checks work correctly
- `apt update` is executed with `check=True`
- Update failures are surfaced (not ignored)
- Errors are handled gracefully
- Package installation works when needed

## Validation Criteria

- Function exists and is accessible
- Function accepts list of package names
- `apt update` is called with `check=True` (verified in code/logs)
- Update failures raise exceptions or return error status
- Error messages are clear and informative
- Package installation succeeds when packages are missing
- No silent failures occur

## Dependencies

- Linux system with apt package manager
- Sudo/root access (for actual installation tests)
- Related tests: 05-02, 05-03, 05-04

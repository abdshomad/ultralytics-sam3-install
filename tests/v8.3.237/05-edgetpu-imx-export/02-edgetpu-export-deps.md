# Test 05-02: Edge TPU Export Dependencies

## Test ID
05-02

## Test Name
Edge TPU Export Dependencies

## Objective
Validate that `export_edgetpu` uses centralized `check_apt_requirements(["edgetpu-compiler"])` instead of shell `apt-get` calls, ensuring consistent dependency management.

## Prerequisites
- Ultralytics v8.3.237 installed
- Linux system (Ubuntu/Debian) with apt package manager
- YOLO model weights file available
- Edge TPU compiler available (or test dependency check only)

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import YOLO
   from ultralytics.utils.checks import check_apt_requirements
   ```

2. **Test Export Function Availability**
   - Verify `export_edgetpu` method exists
   - Check method is callable
   - Verify method signature

3. **Test Dependency Check Integration**
   - Call `export_edgetpu` on model
   - Verify `check_apt_requirements(["edgetpu-compiler"])` is called
   - Check dependency check happens before export

4. **Test Centralized Function Usage**
   - Verify no direct `apt-get` shell calls in export_edgetpu
   - Check `check_apt_requirements` is used instead
   - Verify centralized approach

5. **Test Dependency Installation**
   - Test with missing edgetpu-compiler
   - Verify package is installed via check_apt_requirements
   - Check installation succeeds

6. **Test Export After Dependency Check**
   - Verify export proceeds after dependency check
   - Check export completes successfully
   - Verify Edge TPU model is generated

7. **Test Error Handling**
   - Test with dependency installation failure
   - Verify error is handled gracefully
   - Check error messages are clear

## Expected Results

- `export_edgetpu` method exists and works
- `check_apt_requirements(["edgetpu-compiler"])` is called
- No direct `apt-get` shell calls are used
- Dependency is installed if missing
- Export proceeds after dependency check
- Edge TPU model is generated successfully
- Errors are handled gracefully

## Validation Criteria

- Export method exists and is callable
- `check_apt_requirements` is called with correct package
- No shell `apt-get` calls in export_edgetpu code
- Dependency installation works
- Export completes successfully
- Edge TPU model file is generated
- Error handling is appropriate

## Dependencies

- Linux system with apt package manager
- Edge TPU compiler (or test dependency check only)
- YOLO model weights file
- Related tests: 05-01, 05-03

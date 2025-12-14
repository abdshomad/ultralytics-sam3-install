# Test 05-03: IMX Export Java Dependencies

## Test ID
05-03

## Test Name
IMX Export Java Dependencies

## Objective
Validate that `export_imx` uses `check_apt_requirements()` for Java installations: `openjdk-21-jre` on Ubuntu/Debian Trixie and `openjdk-17-jre` on Raspberry Pi/Debian Bookworm.

## Prerequisites
- Ultralytics v8.3.237 installed
- Linux system (Ubuntu/Debian) with apt package manager
- YOLO model weights file available
- Test on different system types if possible (Ubuntu Trixie, Debian Bookworm, Raspberry Pi)

## Test Steps

1. **Import Required Modules**
   ```python
   from ultralytics import YOLO
   from ultralytics.utils.checks import check_apt_requirements
   import platform
   ```

2. **Test Export Function Availability**
   - Verify `export_imx` method exists
   - Check method is callable
   - Verify method signature

3. **Test System Detection**
   - Detect system type (Ubuntu Trixie, Debian Bookworm, Raspberry Pi)
   - Verify correct Java version is selected
   - Check system detection logic

4. **Test Ubuntu/Debian Trixie Java Installation**
   - Test on Ubuntu/Debian Trixie system
   - Verify `check_apt_requirements(["openjdk-21-jre"])` is called
   - Check Java 21 is installed/checked

5. **Test Raspberry Pi/Debian Bookworm Java Installation**
   - Test on Raspberry Pi or Debian Bookworm
   - Verify `check_apt_requirements(["openjdk-17-jre"])` is called
   - Check Java 17 is installed/checked

6. **Test Centralized Function Usage**
   - Verify no direct Java installation commands
   - Check `check_apt_requirements` is used
   - Verify centralized approach

7. **Test Dependency Installation**
   - Test with missing Java
   - Verify correct Java version is installed
   - Check installation succeeds

8. **Test Export After Java Check**
   - Verify export proceeds after Java check
   - Check export completes successfully
   - Verify IMX model is generated

## Expected Results

- `export_imx` method exists and works
- System type is detected correctly
- Correct Java version is selected based on system
- `check_apt_requirements` is called with correct Java package
- Java is installed if missing
- Export proceeds after Java check
- IMX model is generated successfully

## Validation Criteria

- Export method exists and is callable
- System detection works correctly
- Correct Java version is selected (21 for Trixie, 17 for Bookworm/RPi)
- `check_apt_requirements` is called with correct package
- Java installation works
- Export completes successfully
- IMX model file is generated
- No direct Java installation commands are used

## Dependencies

- Linux system with apt package manager
- Different system types for comprehensive testing (optional)
- YOLO model weights file
- Related tests: 05-01, 05-02, 05-04

# Testing Configuration for Local AI Robot Assistant
# Comprehensive testing strategy with reorganized structure

## Test Categories & Organization

### 1. Unit Tests (src/*/test/)
**Purpose**: Test individual nodes/classes in isolation with mocked dependencies
**Discovery**: `colcon test --packages-select <package>`
**Location**: Within each ROS2 package directory
**Characteristics**:
- Single node/class focus
- Mocked external dependencies (hardware, other nodes)
- Fast execution (< 1 second per test)
- No hardware required
- Run in CI on every commit

**Examples**:
```bash
# Run all unit tests
colcon test --packages-select perception_nodes actuation_nodes localization_nodes

# Run specific package unit tests
colcon test --packages-select perception_nodes

# Run with verbose output
colcon test --packages-select perception_nodes --pytest-args "-v"
```

### 2. Integration Tests (integration_tests/)
**Purpose**: Test multi-node interactions and system-level behavior without hardware
**Discovery**: `pytest integration_tests/`
**Location**: Top-level integration_tests/ directory
**Characteristics**:
- Multi-node interactions
- ROS2 message passing between nodes
- System-level behavior validation
- Moderate execution time (1-30 seconds per test)
- No hardware required (mocked/simulated)
- Run in CI on PR merge

**Examples**:
```bash
# Run all integration tests (no hardware)
pytest integration_tests/ -m "not hardware"

# Run specific integration test
pytest integration_tests/test_camera_pipeline_integration.py -v

# Run with coverage
pytest integration_tests/ --cov=src --cov-report=html
```

### 3. Hardware Tests (hardware_tests/)
**Purpose**: Validate hardware connectivity and basic functionality
**Discovery**: `pytest hardware_tests/`
**Location**: Top-level hardware_tests/ directory
**Characteristics**:
- Requires physical hardware
- Device connectivity validation
- Basic functionality testing
- Longer execution time (10-60 seconds per test)
- Run manually or in hardware CI pipeline

**Examples**:
```bash
# Run all hardware tests
pytest hardware_tests/ -v

# Run specific hardware category
pytest hardware_tests/ -m "camera"
pytest hardware_tests/ -m "uart"

# Run with specific device
pytest hardware_tests/test_camera_capture.py --device /dev/video0
```

### 4. Manual Tests (manual_tests/)
**Purpose**: Interactive testing requiring human verification
**Discovery**: Direct execution
**Location**: Top-level manual_tests/ directory
**Characteristics**:
- Human interaction required
- Visual/audio verification
- Complex setup procedures
- Run during development and validation phases

## Test Execution Commands

### Development Workflow
```bash
# 1. Quick unit tests during development
colcon test --packages-select <current_package>

# 2. Integration tests before PR
pytest integration_tests/ -m "not hardware" -v

# 3. Full validation before release
./scripts/test_reorganization_validate.sh --include-hardware
```

### CI/CD Pipeline
```bash
# Continuous Integration (every commit)
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
colcon test --packages-select perception_nodes actuation_nodes localization_nodes
pytest integration_tests/ -m "not hardware" --junitxml=results.xml

# Hardware Integration (nightly/weekly)
pytest integration_tests/ -m "hardware" --junitxml=hardware_results.xml
pytest hardware_tests/ --junitxml=hardware_validation.xml
```

## Configuration Files

### pytest.ini
Main pytest configuration with:
- Test discovery paths
- Marker definitions
- Output formatting
- Timeout configuration
- Warning filters

### pyproject.toml (testing section)
```toml
[tool.pytest.ini_options]
# (configuration moved to pytest.ini for better compatibility)

[tool.coverage.run]
source = ["src"]
omit = [
    "*/test/*",
    "*/tests/*",
    "*/__pycache__/*",
    "*/build/*",
    "*/install/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "class.*\\bProtocol\\):",
    "@abstractmethod",
]
```

## Markers Usage

### Test Marking Examples
```python
import pytest

@pytest.mark.unit
def test_camera_driver_initialization():
    """Unit test for camera driver."""
    pass

@pytest.mark.integration
def test_camera_pipeline_end_to_end():
    """Integration test for complete camera pipeline."""
    pass

@pytest.mark.hardware
@pytest.mark.camera
def test_physical_camera_capture():
    """Hardware test requiring physical camera."""
    pass

@pytest.mark.slow
def test_long_running_operation():
    """Test that takes significant time."""
    pass
```

### Running by Markers
```bash
# Run only unit tests
pytest -m "unit"

# Run integration tests excluding hardware
pytest -m "integration and not hardware"

# Run camera-related tests
pytest -m "camera"

# Run fast tests only
pytest -m "not slow"
```

## Import Updates Required

After reorganization, update imports in moved files:

### integration_tests/test_camera_pipeline_integration.py
```python
# Update relative imports to absolute imports
from perception_nodes.camera_driver import CameraDriver
from perception_nodes.image_undistort_node import ImageUndistortNode
```

### integration_tests/test_nvdewarper_integration.py
```python
# Update relative imports to absolute imports
from perception_nodes.camera_driver import CameraDriver
from perception_nodes.image_undistort_node import ImageUndistortNode
```

## Validation Steps

1. **Pre-reorganization**: Run current tests to establish baseline
2. **Post-reorganization**: Run reorganization validation script
3. **Import fixes**: Update any broken imports in moved files
4. **CI updates**: Update GitHub Actions or CI configuration
5. **Documentation**: Update README and contribution guidelines

## Benefits of Reorganization

1. **Clear Separation**: Unit vs Integration vs Hardware tests
2. **Faster CI**: Unit tests run quickly on every commit
3. **Better Discovery**: colcon finds unit tests, pytest finds integration tests
4. **Scalability**: Easy to add new test categories
5. **Maintenance**: Clear ownership and responsibility for test types
6. **Debugging**: Easier to identify which layer a test failure belongs to

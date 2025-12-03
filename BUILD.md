# Building SampleMaker

This document explains how to build SampleMaker with its C++ extension module (boopy) for different platforms.

## Overview

SampleMaker includes a C++ component called "boopy" that provides high-performance Boolean polygon operations using Boost.Polygon and is exposed to Python via pybind11. The project uses:

- **scikit-build-core**: Modern build system that uses CMake and pure pyproject.toml configuration
- **CMake**: For configuring the C++ extension build
- **cibuildwheel**: For cross-platform wheel building
- **astral uv**: For faster dependency installation during builds

## Prerequisites

### All Platforms
- Python 3.8 or later
- CMake 3.15 or later
- A C++14-compatible compiler
- pybind11 (installed automatically during build)
- Boost (header-only, for polygon operations)

### Linux
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake libboost-dev

# RHEL/CentOS/Fedora
sudo yum install gcc-c++ make cmake boost-devel
```

### macOS
```bash
brew install boost cmake
```

### Windows
```powershell
# Using vcpkg
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg.exe install boost-polygon:x64-windows
```

## Local Development Build

### Installing in Development Mode

```bash
# Clone the repository
git clone https://github.com/lmidolo/samplemaker.git
cd samplemaker

# Install in editable mode with build dependencies
pip install -e .
```

### Testing the Build

```python
# Test that boopy imports correctly
python -c "from samplemaker.resources import boopy; print('Success!')"
```

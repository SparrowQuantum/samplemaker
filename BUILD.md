# Building SampleMaker

This document explains how to build SampleMaker with its C++ extension module (boopy) for different platforms.

## Overview

SampleMaker includes a C++ component called "boopy" that provides high-performance Boolean polygon operations using 
Boost.Polygon and is exposed to Python via pybind11. The project uses:

- **[scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/)**: 
  Modern build system that uses CMake and pure `pyproject.toml` configuration.
- **[CMake](https://cmake.org/)**: For configuring the C++ extension build.
- **[Astral uv](https://docs.astral.sh/uv/)**: For faster dependency installation during builds.
- **[cibuildwheel](https://cibuildwheel.pypa.io/en/stable/)**: 
  For cross-platform wheel building.


## Local Development Build

### Prerequisites
- Python 3.10 or later
- CMake 3.15 or later
- A C++14-compatible compiler
- pybind11 (installed automatically during build)
- Boost (header-only, for polygon operations)

### Linux

Install the required packages using your distribution's package manager:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake libboost-dev

# RHEL/CentOS/Fedora
sudo yum install gcc-c++ make cmake boost-devel
```

Navigate to the project directory and build:
```bash
# Using Astral uv (recommended)
uv sync

# Using pip
pip install -e .
```

### macOS
Work in progress...

### Windows
To build on Windows, ensure you have Visual Studio with C++ build tools installed.

Boost can be installed via vcpkg:
```powershell
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg.exe install boost-polygon
C:\vcpkg\vcpkg.exe integrate install
```

Open `x64 Native Tools Command Prompt for VS` and navigate to the project directory:
```cmd
cd path\to\samplemaker
set CMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

# Astral uv (recommended)
uv sync

# Using pip
pip install -e .
```

If boost is not found, try setting the CMAKE_TOOLCHAIN_FILE directly in the build command:
```cmd
cd path\to\samplemaker

# Astral uv (recommended)
uv sync -C cmake.args="-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake"

# Using pip
pip install -e . -C cmake.args="-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
```
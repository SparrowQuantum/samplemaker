# Quick Start Guide

## Building SampleMaker with C++ Extensions

### Prerequisites

Make sure you have the required system dependencies installed:

**Linux:**
```bash
sudo apt-get install build-essential cmake libboost-dev
```

**macOS:**
```bash
brew install boost cmake
```

**Windows:**
```powershell
# Install vcpkg and boost
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg.exe install boost-polygon:x64-windows
```

### Local Development

```bash
# Clone and install in editable mode
git clone https://github.com/lmidolo/samplemaker.git
cd samplemaker
pip install -e .

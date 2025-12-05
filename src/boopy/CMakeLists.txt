cmake_minimum_required(VERSION 3.15...3.29)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Find Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Find Boost (header-only for polygon)
if(POLICY CMP0167)
  # Keep old behavior for Boost finding (incompatible with cmake >= 3.30)
  cmake_policy(SET CMP0167 OLD)
endif()
find_package(Boost REQUIRED)

# Create the Python extension module
pybind11_add_module(boopy boopy.cpp)

# Link Boost headers
target_include_directories(boopy PRIVATE ${Boost_INCLUDE_DIRS})

# Set C++ standard
target_compile_features(boopy PRIVATE cxx_std_14)

# Set visibility
set_target_properties(boopy PROPERTIES 
CXX_VISIBILITY_PRESET hidden
)

# Install the module to the correct location within the package
install(TARGETS boopy LIBRARY DESTINATION resources)

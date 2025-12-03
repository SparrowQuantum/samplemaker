# -*- coding: utf-8 -*-
"""
Resources module for samplemaker.

This module provides access to resource files and compiled extensions,
including the boopy C++ extension for Boolean polygon operations.
"""

# Import the boopy C++ extension
try:
    from . import boopy

    __all__ = ["boopy"]
except ImportError as e:
    msg = "Failed to import boopy module. "
    "Make sure the C++ extension is compiled correctly."
    raise ImportError(msg) from e

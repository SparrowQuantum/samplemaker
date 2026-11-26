"""The Python version of Sample Maker.

Sample Maker is a scripting tool for designing lithographic masks in the GDSII format.
The package `samplemaker` comes with different tools and submodules for the creation
and manipulation of basic shapes, periodic shapes, sequences (e.g. waveguides),
circuits, and complex devices.

The code has been developed primarily for nanophotonics, but it can be easily extended
to different applications in micro and nano device fabrication.

Sample Maker is developed and maintained by Leonardo Midolo (Niels Bohr Institute,
University of Copenhagen). It is based on the MATLAB(R) code developed by Leonardo
Midolo between 2013 and 2019. The first version of the rewritten Python code has been
released in October 2021.

This software has been realized with the financial support from the European Research
Council (ERC) under the European Union's Horizon 2020 research and innovation programme
(Grant agreement No. 949043, NANOMEQ).

.. include:: ./documentation.md
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from samplemaker.devices import DevicePort
    from samplemaker.shapes import Box, GeomGroup


__pdoc__: dict[str, bool | str] = {
    "samplemaker.Tutorials": False,
    "samplemaker.tests": False,
    "samplemaker.resources": False,
    "samplemaker.gdsreader": False,
    "samplemaker.devices.DevicePort": False,
}

LayoutPool: dict[str, "GeomGroup"] = {}
"""Contains the current layout pool, mapping SREF names to GeomGroup instances.

This class should generally not be used directly, but only through the Mask class.
"""

# Additional cache pool:

_DevicePool: dict[int, str] = {}
"""Connects a device hash to a SREF to be instantiated."""

_DeviceLocalParamPool: dict[int, dict[str, object]] = {}
"""Connects a device hash to local parameters created by the call to geom()."""

_DeviceLocalPortPool: dict[int, dict[str, "DevicePort"]] = {}
"""Connects a device hash to its local ports."""

_DeviceCountPool: dict[str, int] = {}
"""Connects a device name to a device count."""

_BoundingBoxPool: dict[str, "Box"] = {}
"""Connects a SREF name to its bounding box."""

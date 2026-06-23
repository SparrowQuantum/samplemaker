"""Base classes for generating re-usable device classes.

The Device class
----------------
The `Device` class is an interface to generate specific devices.
In `samplemaker` a device is the combination of drawing commands that generate
a specific pattern which is typically re-used in different layouts.
The devices are parametric, in the sense that the pattern can be a function of
external parameters.
For example, a cross mark device can be a function of the mark size, the width of
cross arms, etc.

The `Device` class provides a common interface to define parameters and drawing
functions. To create a new device, simply derive the `Device` class and provide
an implementation for parameters and geometry:

    def MyFancyDevice(Device):
        def initialize(self):
            # Initialization stuff goes here
            pass
        def parameters(self):
            # Parameters are defined here
            pass
        def geom(self):
            # Drawing goes here
            pass
        def ports(self):
            # Ports go here
            pass

    # To use:
    dev = MyFancyDevice.build()
    geom = dev.run()

As bare minimum, the initialize, parameters and geom methods should be re-implemented in
a device class.

### Device name
When initializing the device it is very important to give it a unique name as a string
(and optionally a description).
This is done in the `Device.initialize` method:

    def initialize(self):
        self.set_name("MYDEVICE")
        self.set_decription("First version of MYDEVICE")

The name is used to call devices later on (see Device registraiton below) and
to instantiate them in circuits.

### Parameters
Device paramters must be defined via the function `Device.add_parameter` as follows:

    def parameters(self):
        self.add_parameter("my_param", default_value, "Description", type, (min, max))

The default value is usually hard-coded into the device itself and should be the
reference value for creating the geometry. Optionally, the parameter type can be
specified as `bool` or `int` or `float`. The use of string is not recommended but it is
not forbidden. Also optionally, a range can be specified for integer and float values as
a tuple. This helps the user of the device in figuring out what makes sense.
For example:

    def parameters(self):
        self.add_parameter("length", 15, "Length of the marker", float, 2, 30)

To use one of the parameter in the drawing, use the `Device.get_params` function to get
a dictionary of the parameters with values:

    def geom(self):
        p = self.get_params()
        L = p["length"] # Marker length

### Drawing the geometry
The `Device.geom` method should be implemented so that it returns a
`samplemaker.shapes.GeomGroup` object with the geometry.

The user should never run the `Device.geom` method, but use instead `Device.run`
(see example above).

### Building, running, what is all that?
A device object is never instantiated via its constructor __init__ but using the class
method `Device.build`. Building a device is like initializing it, setting the default
parameters and preparing the device to be drawn.
To actually draw the geometry you use the `Device.run` method, which ensures that the
exact sequence of operation is carried out.

### Device registration
To use the re-use the devices later on, it is common practice to build a library of
devices (containing all the classes) and register the devices to a shared dictionary
that other functions can use to build/run named devices.
This is achieved via the `registerDevicesInModule` which can be called at the end
of each python script and will update a hidden device database.
Building a device is then simply done as

    dev = Device.build_registered("MYDEVICE")
    geom = dev.run()

See example


Device Ports
------------
An important part of device creation is to define ports that can connect a device to
another one. To define ports, `samplemaker` provides the base class `DevicePort`.
The class defines a named port, with position and orientation.

The device ports are specified by re-implementing the `Device.ports` function and
calling the `Device.addport` method:

    def ports(self):
        p1 = DevicePort(20,40,True,True)
        self.addport("port1", p1)

The above code generates a `DevicePort` placed at (20,40) facing east. The two boolean
define whether the port is oriented horizontally or vertically and if it faces forward
or backward. Check the documentation of `DevicePort` to learn more about port
properties.

Quite often, when creating ports, it is necessary to use some variable which is only
defined locally in the geom() function. To define ports directly from geom(), one can
use the method `Device.addlocalport` instead and leave the ports() methods not
implemented:

    def geom(self):
        p1 = DevicePort(20,40,True,True)
        self.addlocalport("port1", p1)

Do not use `Device.addport` from the geom() command as it will not work.

### Implementing custom device ports
Users can define different port types by inheriting the `DevicePort` class. For example
one might be interested in defining optical ports (for waveguide devices) and electrical
ports (for electronic circuits). The default DevicePort in fact cannot be connected to
anything until the user supplies a connector function. For example:

    def OpticalPortConnector(port1: "DevicePort",port2: "DevicePort") -> "GeomGroup":
        # functions that calculate and draw the connector
        return geom

    class OpticalPort(DevicePort):
        def __init__(self,x0,y0,horizontal,forward,width,name):
            super().__init__(x0,y0,horizontal,forward)
            self.width = width
            self.name=name
            self.connector_function=OpticalPortConnector

The newly created OpticalPort can now be connected to other OpticalPort ports.

Circuits
---------
Once ports are specified, it is possible to create circuits that connect various devices
with each others. A circuit is itself a `Device` with parameters and ports, except the
drawing routine is controlled by a netlist that defines what devices should be
instantiated, where, and how connectivity is defined.

### Defining a netlist
The `NetList` class speficies a circuit layout. To specify a Netlist, you need to
provide a list of entries via the class `NetListEntry`. A single entry of the netlist
correspond to a device name (which should be registered) position, and connectivity:

    portmap = {"port1":"inA","port2":"inB"}
    params = {"length":16}
    entry1 = NetListEntry("MYDEVICE", 0, 0, "E", portmap, params)

In the above example MYDEVICE will be placed in 0,0 facing East ("E") and his parameter
"length" will be set to 16. Additionally the named DevicePort "port1" has been assigned
to wire "inA" and "port2" to wire "inB". The circuit builder will look for any other
entry where a port has been assigned to wire "inA" and run the connector (provided by
user) between the two ports. If a matching port cannot be found, the wire will become
the name of an external port of the entire circuit.

The netlist is then built specifying the list of entries and a circuit can be build
exactly as a standard device:

    netlist = NetList("my_circuit", [entry1,entry2,entry3])
    cir_dev = Circuit.build() # Note that we build first
    cir_dev.set_param("NETLIST") = netlist # Set the NETLIST parameter
    g = cir_dev.run() # and finally run the device

More details on specifying circuits are given in the tutorials, where it is also
explained how to nest circuits together (i.e. creating netlists of netslists)
"""

import inspect
import math
import sys
import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path as _Path
from typing import Any, Self, TypeAlias
from warnings import deprecated

import numpy as np

from samplemaker import (
    LayoutPool,
    _BoundingBoxPool,
    _DeviceCountPool,
    _DeviceLocalParamPool,
    _DevicePool,
)
from samplemaker.gdswriter import GDSWriter
from samplemaker.makers import make_sref, make_text
from samplemaker.shapes import GeomGroup, Poly

ConnectorFunctionType: TypeAlias = Callable[["DevicePort", "DevicePort"], GeomGroup]


class IncompatiblePortError(RuntimeError):
    """Exception raised when trying to link incompatible ports."""


def _empty_connector_function(port1: "DevicePort", port2: "DevicePort") -> GeomGroup:
    """Default connector function that raises an error.

    Parameters
    ----------
    port1 : DevicePort
        The first port.
    port2 : DevicePort
        The second port.

    Returns
    -------
    GeomGroup
        An empty geometry group.

    Raises
    ------
    IncompatiblePortError
        Always raised to indicate that the ports are incompatible.

    """
    msg = (
        f"Cannot connect ports {port1.name} and {port2.name} "
        "as no connector function is defined."
    )
    raise NotImplementedError(msg)


class DevicePort:
    """Class that defines a device port."""

    def __init__(self, x0: float, y0: float, horizontal: bool, forward: bool) -> None:
        """Initialize a device port.

        Parameters
        ----------
        x0 : float
            The x-coordinate of the port.
        y0 : float
            The y-coordinate of the port.
        horizontal : bool
            Whether the port is horizontal or vertical. If True, the port points in the
            east/west direction. If False, the port points in the north/south direction.
        forward : bool
            Whether the port points in the positive coordinate direction.

        """
        self.x0 = x0
        self.y0 = y0
        self.__px = x0
        self.__py = y0
        self.hv = horizontal
        self.bf = forward
        self.__hv = horizontal
        self.__bf = forward
        self.name = ""
        # _geometry can carry a full geom to which it is connected:
        self._geometry = GeomGroup()
        # any other port shared with this port in the same device:
        self._parentports = {}
        self.connector_function: ConnectorFunctionType = _empty_connector_function

    def set_name(self, name: str) -> None:
        """Set the name of the device port.

        Parameters
        ----------
        name : str
            The name of the port.

        """
        self.name = name

    def angle(self) -> float:
        """Get the angle of the port in radians.

        Returns
        -------
        float
            The orientation of the port in radians (east = zero).

        """
        return math.pi * (3 - (self.hv + self.bf * 2)) / 2

    def set_angle(self, angle: float) -> None:
        """Set the port orientation by specifying the angle in radians.

        Parameters
        ----------
        angle : float
            The orientation of the port in radians.

        Returns
        -------
        None

        """
        i = round(3 - angle * 2 / math.pi) % 4
        self.hv = i % 2 == 1
        self.bf = math.floor(i / 2) == 1

    def printangle(self) -> None:
        """Print the angle of the port as a string.

        Returns
        -------
        None

        """
        print(self.angle_to_text())

    def angle_to_text(self) -> str:
        """Get the angle as a string.

        Returns
        -------
        str
            The orientation of the port as a string ("N", "S", "W", or "E").

        """
        if self.hv and self.bf:
            return "E"
        if self.hv and not self.bf:
            return "W"
        if not self.hv and self.bf:
            return "N"
        if not self.hv and not self.bf:
            return "S"
        # Should never happen. We catch it anyway
        msg = "Invalid port orientation."
        raise RuntimeError(msg)

    def dx(self) -> float:
        """Get the x component of the port orientation.

        Returns
        -------
        float
            The x component of the port orientation.

        """
        return self.hv * (2 * self.bf - 1)

    def dy(self) -> float:
        """Get the y component of the port orientation.

        Returns
        -------
        float
            The y component of the port orientation.

        """
        return (not self.hv) * (2 * self.bf - 1)

    def rotate(self, x0: float, y0: float, angle: float) -> None:
        """Rotate the port around a point by a given angle.

        Parameters
        ----------
        x0 : float
            The x-coordinate of the point to rotate around.
        y0 : float
            The y-coordinate of the point to rotate around.
        angle : float
            The angle to rotate by, in degrees.

        Returns
        -------
        None

        """
        xc = self.x0 - x0
        yc = self.y0 - y0
        cost = math.cos(math.radians(angle))
        sint = math.sin(math.radians(angle))
        self.x0 = cost * xc - sint * yc + x0
        self.y0 = sint * xc + cost * yc + y0
        self.set_angle(self.angle() + math.radians(angle))

    def move_straight(self, amount: float) -> None:
        """Move the port straight by the given amount.

        Parameters
        ----------
        amount : float
            The distance to move the port.

        Returns
        -------
        None

        """
        self.x0 += self.dx() * amount
        self.y0 += self.dy() * amount

    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use DevicePort.move_straight() instead."
    )
    def S(self, amount: float) -> None:  # noqa: N802
        """Move the port straight by the given amount.

        Parameters
        ----------
        amount : float
            The distance to move the port.

        Returns
        -------
        None

        """
        self.move_straight(amount)

    def bend_left(self, radius: float) -> None:
        """Make a 90 degree left bend with the given radius.

        Parameters
        ----------
        radius : float
            The radius of the bend.

        Returns
        -------
        None

        """
        xc = self.x0 - self.dy() * radius
        yc = self.y0 + self.dx() * radius
        phi = self.angle() - math.pi / 2
        self.x0 = radius * math.cos(phi + math.pi / 2) + xc
        self.y0 = radius * math.sin(phi + math.pi / 2) + yc
        self.set_angle(self.angle() + math.pi / 2)

    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use DevicePort.bend_left() instead."
    )
    def BL(self, radius: float) -> None:  # noqa: N802
        """Make a 90 degree left bend with the given radius.

        DEPRECATED: Use bend_left() instead.

        Parameters
        ----------
        radius : float
            The radius of the bend.

        Returns
        -------
        None

        """
        self.bend_left(radius)

    def bend_right(self, radius: float) -> None:
        """Make a 90 degree right bend with the given radius.

        Parameters
        ----------
        radius : float
            The radius of the bend.

        Returns
        -------
        None

        """
        xc = self.x0 + self.dy() * radius
        yc = self.y0 - self.dx() * radius
        phi = self.angle() + math.pi / 2
        self.x0 = radius * math.cos(phi - math.pi / 2) + xc
        self.y0 = radius * math.sin(phi - math.pi / 2) + yc
        self.set_angle(self.angle() - math.pi / 2)

    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use DevicePort.bend_right() instead."
    )
    def BR(self, radius: float) -> None:  # noqa: N802
        """Make a 90 degree right bend with the given radius.

        DEPRECATED: Use bend_right() instead.

        Parameters
        ----------
        radius : float
            The radius of the bend.

        Returns
        -------
        None

        """
        self.bend_right(radius)

    def reset(self) -> None:
        """Reset the port position and orientation to a fixed position.

        If the `fix()` method was called before, the port will be reset to the position
        and orientation at the time of the `fix()` call. Otherwise it will be reset to
        the initial position and orientation when the port was created.

        Returns
        -------
        None

        """
        self.x0 = self.__px
        self.y0 = self.__py
        self.hv = self.__hv
        self.bf = self.__bf

    def fix(self) -> None:
        """Fix the port position and orientation to the current values.

        The port can then later be returned to this position and orientation by calling
        the `reset()` method.

        Returns
        -------
        None

        """
        self.__px = self.x0
        self.__py = self.y0
        self.__hv = self.hv
        self.__bf = self.bf

    def dist(self, other: "DevicePort") -> float:
        """Calculate the distance between two ports.

        Parameters
        ----------
        other : DevicePort
            The other port to calculate the distance to.

        Returns
        -------
        float
            The distance between the two ports.

        """
        dx = other.x0 - self.x0
        dy = other.y0 - self.y0
        return math.sqrt(dx * dx + dy * dy)


class Device:
    """Base class for devices.

    To create a new device, inherit this class and re-implement the `initialize()`,
    `parameters()`, and `geom()` geom methods. Optionally the `ports()` method can be
    re-implemented if ports are defined outside the `geom()` method.

    To instantiate a device, use the class method `Device.build()` which will return an
    instance of the device ready to be rendered via the `run()` method.
    """

    def __init__(self) -> None:
        """Initialize a Device. Should never be called by the user."""
        self._p = {}
        self._pdescr = {}
        self._ptype = {}  # stores the type of the parameter
        self._prange = {}  # stores the min-max range of the parameter in a tuple
        self._localp = {}
        self.addlocalparameter("_ports_", {}, "Ports calculated by geom")
        self._x0 = 0
        self._y0 = 0
        self._hv = True
        self._bf = True
        self._ports = {}
        self._name = ""
        self._description = "No description yet"
        self.use_references = True

    def __flatdict(self, d: dict[str, Any], parent_str: str) -> dict[str, Any]:
        flatdict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                newdict = self.__flatdict(value, parent_str + key + "::")
                flatdict.update(newdict)
            elif not isinstance(value, list):
                flatdict[parent_str + key] = value
        return flatdict

    def __hash__(self) -> int:
        """Hash function for the device.

        Based on the device name, parameters and if a sequencer is used, the sequencer
        options.

        Returns
        -------
        int
            The hash value of the device.

        """
        if hasattr(self, "_seq"):
            fldict = self.__flatdict(self._seq.options, "")
            return hash(
                (frozenset(self._p.items()), self._name, frozenset(fldict.items()))
            )

        return hash((frozenset(self._p.items()), self._name))

    def angle(self) -> float:
        """Return the orientation of the device in radians.

        Returns
        -------
        float
            The orientation in radians (east = zero).

        """
        return math.pi * (3 - (self._hv + self._bf * 2)) / 2

    def set_angle(self, angle: float) -> None:
        """Change the orientation of the device.

        Parameters
        ----------
        angle : float
            The new angle in radians.

        Returns
        -------
        None

        """
        i = round(3 - angle * 2 / math.pi) % 4
        self._hv = i % 2 == 1
        self._bf = math.floor(i / 2) == 1

    def set_position(self, x0: float, y0: float) -> None:
        """Change the position of the device.

        Parameters
        ----------
        x0 : float
            X offset.
        y0 : float
            Y offset.

        Returns
        -------
        None

        """
        self._x0 = x0
        self._y0 = y0

    def addport(self, port: DevicePort) -> None:
        """Call this from the ports() method to add a port to the device.

        Parameters
        ----------
        port : DevicePort
            The device port.

        Returns
        -------
        None

        """
        self._ports[port.name] = port

    def addparameter(
        self,
        param_name: str,
        default_value: Any,  # noqa: ANN401
        param_description: str,
        param_type: type = float,
        param_range: tuple[float, float] = (0, np.inf),
    ) -> None:
        """Call this from the parameters() method to add a parameter to the device.

        Parameters
        ----------
        param_name : str
            The name of the parameter.
        default_value : TYPE
            The default value.
        param_description : str
            A text describing the parameters.
        param_type : TYPE, optional
            The type of the parameter, by default float.
        param_range : tuple, optional
            A tuple specifying the min and max value of the parameter , by default
            (0,np.inf).

        Returns
        -------
        None

        Raises
        ------
        ValueError
             If the parameter name contains a ":" character. This is a special character
             used by the `Circuit` class.

        """
        if param_name.find(":") != -1:
            msg = f"Cannot define variable names containing ':'. Got '{param_name}'."
            raise ValueError(msg)
        self._p[param_name] = default_value
        self._pdescr[param_name] = param_description
        self._ptype[param_name] = param_type
        self._prange[param_name] = param_range

    def addlocalparameter(
        self,
        param_name: str,
        default_value: Any,  # noqa: ANN401
        param_description: str,
        param_type: type = float,
        param_range: tuple[float, float] = (0, np.inf),
    ) -> None:
        """Define a local parameter that is only used within the class.

        Parameters
        ----------
        param_name : str
            The parameter name.
        default_value : Any
            The value of the paramter.
        param_description : str
            Description of the parameter.
        param_type : type, optional
            The type of the paramter, by default float.
        param_range : tuple, optional
            A tuple specifying the min and max value of the parameter , by default
            (0,np.inf).

        Returns
        -------
        None

        Raises
        ------
        ValueError
             If the parameter name contains a ":" character. This is a special character
             used by the `Circuit` class.

        """
        if param_name.find(":") != -1:
            msg = f"Cannot define variable names containing ':'. Got '{param_name}'."
            raise ValueError(msg)
        self._localp[param_name] = default_value
        self._pdescr[param_name] = param_description
        self._ptype[param_name] = param_type
        self._prange[param_name] = param_range

    def addlocalport(self, port: DevicePort) -> None:
        """Add a local port.

        Local ports are ports defined from the `geom()` method as opposed to an
        overridden `ports()` method.

        If you need some info from the `geom()` function to create ports just add local
        ports and they will be automatically added to ports by the `ports()` function.

        Parameters
        ----------
        port : DevicePort
            The port to be added.

        Returns
        -------
        None

        """
        self._localp["_ports_"][port.name] = port

    def get_localport(self, portname: str) -> DevicePort:
        """Get the local port (i.e. within the geom() function.).

        Parameters
        ----------
        portname : str
            The port name.

        Returns
        -------
        DevicePort
            The port.

        Raises
        ------
        ValueError
            If the port is not defined by the device.

        """
        lports = self._localp["_ports_"]
        if portname not in lports:
            msg = (
                f"Could not find port named {portname} in {self._name} as it was "
                f"not defined by device."
            )
            raise ValueError(msg)
        return lports[portname]

    def remove_localport(self, portname: str) -> None:
        """Remove a local port.

        Parameters
        ----------
        portname : str
            The name of the port to be removed.

        Returns
        -------
        None

        """
        lports = self._localp["_ports_"]
        if portname in lports:
            self._localp["_ports_"].pop(portname)

    def set_param(self, param_name: str, value: Any) -> None:  # noqa: ANN401
        """Change a paramter. To be called after build().

        Parameters
        ----------
        param_name : str
            The parameter to be changed.
        value : Any
            The new value of the parameter.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the parameter is not defined by the device.

        """
        param_hier = param_name.split("::")
        p = self._p
        for i, cur_p in enumerate(param_hier):
            if i == (len(param_hier) - 1) and cur_p in p:
                p[cur_p] = value
            elif cur_p in p:
                p = p[cur_p]
            else:
                msg = (
                    f"Could not set parameter '{param_name}', "
                    "as it was not defined by device."
                )
                raise ValueError(msg)

    def get_params(
        self, cast_types: bool = True, clip_in_range: bool = True
    ) -> dict[str, Any]:
        """Return the dictionary with all parameters.

        Should be called by geom() functions.

        Parameters
        ----------
        cast_types : bool, optional
            Attempts to do a type-cast on the parameter, by default True.
        clip_in_range : bool, optional
            Clips the value in the range specified, by default True.

        Returns
        -------
        dict[str, Any]
            A dictionary with the parameter value map.

        """
        if cast_types:
            for p, val in self._p.items():
                self._p[p] = self._ptype[p](val)
        if clip_in_range:
            for p, val in self._p.items():
                v = max(val, self._prange[p][0])
                v = min(v, self._prange[p][1])
                self._p[p] = v
        return self._p

    def get_port(self, port_name: str) -> DevicePort:
        """Get a named port.

        Parameters
        ----------
        port_name : str
            Name of the port.

        Returns
        -------
        DevicePort
            The DevicePort object associated to the port.

        Raises
        ------
        ValueError
            If the port is not defined by the device.

        """
        if port_name not in self._ports:
            msg = (
                f"Could not find port named {port_name} in {self._name} as it was "
                f"not defined by device."
            )
            raise ValueError(msg)
        return self._ports[port_name]

    def set_name(self, name: str) -> None:
        """Set the device name (should be called from initialize).

        Parameters
        ----------
        name : str
            The device name.

        Returns
        -------
        None

        """
        self._name = name

    def set_description(self, descr: str) -> None:
        """Set the device description (should be called from initialize).

        Parameters
        ----------
        descr : str
            The device description.

        Returns
        -------
        None

        """
        self._description = descr

    def initialize(self) -> None:
        """Override this function in your device to initialize and set the device name.

        Should call `set_name()` and `set_description()`.
        If a sequencer is needed in the `geom()` method it should also be defined here
        as `self._seq`.

        Returns
        -------
        None

        """

    def parameters(self) -> None:
        """Override this function in your device to define parameters of the device.

        Should call the `add_parameter()` method to define parameters.

        Returns
        -------
        None

        """

    def geom(self) -> GeomGroup:
        """Override this function in your device to generate the geometry of the device.

        Returns
        -------
        GeomGroup
            The geometry of the device.

        """
        return GeomGroup()

    def run(self) -> GeomGroup:
        """Run the device and generate the geometry.

        Returns
        -------
        GeomGroup
            The geometry of the device.

        """
        if self.use_references:
            # Check if it is in the device pool
            hsh = self.__hash__()
            srefname = self._p["NETLIST"].name if "NETLIST" in self._p else self._name
            if srefname not in _DeviceCountPool:
                _DeviceCountPool[srefname] = 0

            if hsh not in _DevicePool:
                _DeviceCountPool[srefname] += 1
                srefname += f"_{_DeviceCountPool[srefname]:04d}"
                LayoutPool[srefname] = self.geom()
                _BoundingBoxPool[srefname] = LayoutPool[srefname].bounding_box()
                _DevicePool[hsh] = srefname
                _DeviceLocalParamPool[hsh] = deepcopy(self._localp)
            else:
                srefname += f"_{_DeviceCountPool[srefname]:04d}"
                self._localp = _DeviceLocalParamPool[hsh]
            # now create a ref
            g = make_sref(
                self._x0,
                self._y0,
                _DevicePool[hsh],
                LayoutPool[_DevicePool[hsh]],
                angle=math.degrees(self.angle()),
            )
        else:
            g = self.geom()
            g.rotate_translate(self._x0, self._y0, math.degrees(self.angle()))

        # this will get the proper local parameters as if self.geom() ran properly:
        self.ports()

        # Now rotate/translate all ports
        for port in self._ports.values():
            port.rotate(0, 0, math.degrees(self.angle()))
            port.x0 += self._x0
            port.y0 += self._y0
        return g

    def ports(self) -> None:
        """Add ports to the device.

        Called automatically when the device is built/run.

        Override this to define ports. Do not override if localports are used via the
        geom() function.

        Returns
        -------
        None

        """
        if "_ports_" in self._localp:
            for p in self._localp["_ports_"].values():
                self.addport(deepcopy(p))

    @staticmethod
    def build_registered(name: str) -> "Device":
        """Build a device from the pool of registered device names.

        Parameters
        ----------
        name : str
            The device name to be built.

        Returns
        -------
        Device
            The device to be built.

        """
        if name not in _DeviceList:
            msg = f"No device named {name} found."
            raise ValueError(msg)
        return _DeviceList[name].build()

    @classmethod
    def build(cls) -> Self:
        """Class method to build a device.

        Returns
        -------
        Self
            Instance of the Device ready to be rendered via the run() method.

        """
        device = cls()
        device.initialize()
        device.parameters()
        device.ports()
        return device


class NetListEntry:
    """Class that defines a single entry in a netlist."""

    def __init__(
        self,
        devname: str,
        x0: float,
        y0: float,
        rot: str,
        portmap: dict[str, str],
        params: dict[str, Any],
    ) -> None:
        """Initialize a netlist entry.

        Parameters
        ----------
        devname : str
            The registered device name.
        x0 : float
            x coordinate of the device.
        y0 : float
            y coordinate of the device.
        rot : str
            String that defines the orientation of the device (can only be "N", "S",
            "W" or "E").
        portmap : dict[str, str]
            A dictionary that associates a port in the device to a wire. The keys of
            the dictionary are the port names defined in the device and the values are
            the wire names. Wires with the same name will be connected together.
        params : dict[str, Any]
            A dictionary of parameters to be used when creating the device.

        """
        self.devname = devname
        self.x0 = x0
        self.y0 = y0
        self.rot = 0
        if rot == "N":
            self.rot = 90
        if rot == "W":
            self.rot = 180
        if rot == "S":
            self.rot = 270
        self.portmap = portmap
        self.params = params

    def __hash__(self) -> int:
        """Hash function for the netlist entry.

        Based on the device name, position, orientation, portmap and parameters.

        Returns
        -------
        int
            The hash value of the netlist entry.

        """
        return hash(
            (
                self.devname,
                self.x0,
                self.y0,
                self.rot,
                frozenset(self.portmap.items()),
                frozenset(self.params.items()),
            )
        )


class NetList:
    """Defines a netlist for drawing a circuit.

    This is a helper class used with the `Circuit` that defines the devices present in
    the circuit as well as how they are connected, both internally and externally.
    """

    def __init__(self, name: str, entry_list: list[NetListEntry]) -> None:
        """Initialize a NetList for circuit generation.

        Parameters
        ----------
        name : str
            The netlist name.
        entry_list : list[NetListEntry]
            list of `NetListEntry` objects.

        """
        self.name: str = name
        self.entry_list: list[NetListEntry] = entry_list
        self.external_ports: Sequence[str] = []
        self.aligned_ports: Sequence[str] = []
        self.paths: dict[str, list[float]] = {}

    def __hash__(self) -> int:
        """Hash function for the netlist.

        Based on the netlist name, entry list, external ports and aligned ports.

        Returns
        -------
        int
            The hash value of the netlist.

        """
        return hash(
            (
                self.name,
                tuple(self.entry_list),
                tuple(self.external_ports),
                tuple(self.aligned_ports),
            )
        )

    def set_external_ports(self, ext_ports: Sequence[str]) -> None:
        """Define a list of wires that should be connected outside the circuit.

        Parameters
        ----------
        ext_ports : Sequence[str]
            A sequence of strings with the wires assigned to ports in the netlist entry.

        Returns
        -------
        None

        """
        self.external_ports = ext_ports

    def set_aligned_ports(self, aligned_ports: Sequence[str]) -> None:
        """Define a list of wires that should be aligned with each other.

        Parameters
        ----------
        aligned_ports : Sequence[str]
             A sequence of strings with the wires assigned to ports in the netlist
             entry.

        Returns
        -------
        None

        """
        self.aligned_ports = aligned_ports

    def set_path(self, port_name: str, coords: list[float]) -> None:
        """Define a specific path that a wire should follow.

        Parameters
        ----------
        port_name : str
            The name of the wire.
        coords : list[float]
            list of coordinates, x1, y1, x2, y2... that the wire should follow.

        Returns
        -------
        None

        """
        self.paths[port_name] = coords

    @classmethod
    def import_circuit(
        cls, file_name: str, circuit_name: str = ""
    ) -> Self | dict[str, Self]:
        """Generate a NetList object from a circuit file.

        The input is a text file with circuit description similar to the
        SPICE netlist format (yet with some important differences).
        Check the tutorials for examples.

        Parameters
        ----------
        file_name : str
            The circuit filename.
        circuit_name : str, optional
            The subcircuit to load inside the circuit file, by default "", which reads
            the entire circuit structure.

        Returns
        -------
        NetList | dict[str, NetList]
            The NetList with the imported circuit.

        """
        # Import all netlists
        with _Path(file_name).open() as f:
            current_netlist = ""
            current_entrylist = []
            current_align = []
            current_path = {}
            all_lists = {}  # stores all the imported netlists
            for line in f:
                tokens = line.split()
                if len(tokens) == 0:  # empty line
                    continue
                if tokens[0][0] == "#":  # Comment
                    continue
                if tokens[0] == ".CIRCUIT" and current_netlist == "":
                    current_netlist = tokens[1:]
                    print("Reading ", current_netlist[0])
                    continue
                if tokens[0] == ".ALIGN" and current_netlist != "":
                    current_align = tokens[1:]
                    continue
                if tokens[0] == ".PATH" and current_netlist != "":
                    # expect 4 tokens minimum
                    wirename = tokens[1]
                    pathlist = []
                    if (len(tokens[2:]) % 3) > 0:
                        msg = (
                            "Wrong number of values specified for .PATH command. "
                            f"Got {len(tokens[2:])} values, expected a multiple of 3."
                        )
                        raise ValueError(msg)

                    for i in range(0, len(tokens[2:]), 3):
                        pathlist += [float(tokens[2 + i])]  # X
                        pathlist += [float(tokens[3 + i])]  # Y
                        angle = tokens[4 + i]  # angle
                        if angle == "N":
                            pathlist += [90]
                        if angle == "E":
                            pathlist += [0]
                        if angle == "W":
                            pathlist += [180]
                        if angle == "S":
                            pathlist += [270]

                    current_path[wirename] = pathlist
                    continue
                if tokens[0] == ".END" and current_netlist != "":
                    clist = cls(current_netlist[0], deepcopy(current_entrylist))
                    clist.aligned_ports = current_align
                    clist.external_ports = current_netlist[1:]
                    clist.paths = deepcopy(current_path)
                    current_entrylist = []
                    current_netlist = ""
                    current_align = []
                    current_path.clear()
                    all_lists[clist.name] = clist
                    continue
                # Now we should have only entries
                if current_netlist != "":
                    # parse entries
                    devname = tokens[0]
                    params = {}
                    cin = 1
                    if devname == "X":
                        params["NETLIST"] = all_lists[tokens[1]]  # must exist
                        cin += 1
                    x = float(tokens[cin])
                    cin += 1
                    y = float(tokens[cin])
                    cin += 1
                    rot = tokens[cin]
                    cin += 1
                    portdict = {}
                    while tokens[cin] != ".":  # Should always end the entry with a dot
                        portdict[tokens[cin]] = tokens[cin + 1]
                        cin += 2
                    cin += 1
                    while cin < len(tokens):
                        params[tokens[cin]] = float(tokens[cin + 1])
                        cin += 2
                    current_entrylist.append(
                        NetListEntry(devname, x, y, rot, portdict, params)
                    )

        if circuit_name == "":
            return all_lists
        return all_lists[circuit_name]

    @classmethod
    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use Netlist.import_circuit() instead."
    )
    def ImportCircuit(  # noqa: N802
        cls, file_name: str, circuit_name: str = ""
    ) -> Self | dict[str, Self]:
        """Generate a NetList object from a circuit file.

        The input is a text file with circuit description similar to the
        SPICE netlist format (yet with some important differences).
        Check the tutorials for examples.

        Parameters
        ----------
        file_name : str
            The circuit filename.
        circuit_name : str, optional
            The subcircuit to load inside the circuit file, by default "", which reads
            the entire circuit structure.

        Returns
        -------
        NetList | dict[str, NetList]
            The NetList with the imported circuit.

        """
        return cls.import_circuit(file_name, circuit_name)


class Circuit(Device):
    """A Circuit is a Device that generates its geometry from a NetList.

    The netlist is a parameter of the circuit and can be changed after building the
    device by calling

        cir_dev.set_param("NETLIST", new_netlist)

    which will automatically add the parameters associated with the devices in the
    netlist to the circuit parameters. They can then be set by calling

        cir_dev.set_param("dev_MYDEVICE_1::length", 12)

    In this example we set parameter "length" of the first instance of MYDEVICE to 12.
    The format is "dev_%devicename_%number", where %devicename is the registered device
    name and %number is the device entrylist number (starting from 1).

    Like the `Device` class the final geometry can be generated by calling the `run()`
    method.
    """

    def __flatdict(self, d: dict[str, Any], parent_str: str) -> dict[str, Any]:
        flatdict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                newdict = self.__flatdict(value, parent_str + key + "::")
                flatdict.update(newdict)
            else:
                flatdict[parent_str + key] = value
        return flatdict

    def __hash__(self) -> int:
        """Hash function for the circuit.

        Based on the NETLIST parameter and the device name.

        Returns
        -------
        int
            The hash value of the circuit.

        """
        flatdict = self.__flatdict(self._p, "")
        return hash((frozenset(flatdict.items()), self._name))

    def initialize(self) -> None:
        """Name the Circuit as 'X' to be referred in other circuits.

        Returns
        -------
        None

        """
        self._name = "X"

    def parameters(self) -> None:
        """Define the parameter NETLIST taking a `NetList` object as input.

        Returns
        -------
        None

        """
        self.addparameter(
            param_name="NETLIST",
            default_value=[],
            param_description="A list of NetListEntry specifying a circuit",
        )
        self.addlocalparameter(
            param_name="external_ports",
            default_value={},
            param_description="Locally store the ports that connect to the circuit",
        )

    def _update_parameters(self) -> None:
        netlist = self._p["NETLIST"].entry_list
        for i, nle in enumerate(netlist):
            if nle.devname not in _DeviceList:
                msg = f"No device named {nle.devname} found."
                raise ValueError(msg)

            dev = _DeviceList[nle.devname].build()
            for key, value in nle.params.items():
                dev.set_param(key, value)
            self.addparameter(
                param_name=f"dev_{nle.devname}_{i + 1:d}",
                default_value=dev._p,
                param_description=f"Device parameters for {nle.devname}",
            )

    def set_param(self, param_name: str, value: Any) -> None:  # noqa: ANN401
        """Set the value of a parameter.

        In a Circuit, you can refer to individual device parameters by using the
        following convention:

            dev.set_param("dev_MYDEVICE_1::length", 12)

        which sets parameter "length" of the first instance of MYDEVICE to 12.
        The format is "dev_%devicename_%number", where %devicename is the registered
        device name and %number is the device entrylist number (starting from 1).

        Parameters
        ----------
        param_name : str
            The name of the parameter to be set.
        value : Any
            The new value of the parameter.

        Returns
        -------
        None

        """
        super().set_param(param_name, value)
        if param_name == "NETLIST":
            self._update_parameters()

    def _build_device_geometry(
        self,
        nle: NetListEntry,
        instance_number: int,
    ) -> tuple[GeomGroup, dict[str, DevicePort]]:
        if nle.devname not in _DeviceList:
            msg = f"No device named {nle.devname} found."
            raise ValueError(msg)

        dev = _DeviceList[nle.devname].build()
        dev.use_references = self.use_references

        # Force sequencer reset if has _seq subfield.
        # This keeps device behavior independent from prior sequencer state.
        if hasattr(dev, "_seq"):
            dev._seq.reset()

        dev._p = self._p[f"dev_{nle.devname}_{instance_number:d}"]
        dev._x0 = nle.x0
        dev._y0 = nle.y0
        dev.set_angle(math.radians(nle.rot))
        return dev.run(), dev._ports

    def _collect_mapped_ports(
        self,
        nle: NetListEntry,
        geom: GeomGroup,
        device_ports: dict[str, DevicePort],
        input_ports: dict[str, DevicePort],
        output_ports: dict[str, DevicePort],
    ) -> None:
        for devport, conn_name in nle.portmap.items():
            if devport not in device_ports:
                msg = (
                    f"Could not find port named {devport} in {nle.devname} as it was "
                    f"not defined by device."
                )
                raise ValueError(msg)

            port = device_ports[devport]
            port._geometry = geom
            port._parentports = device_ports
            if conn_name in input_ports:
                output_ports[conn_name] = port
            else:
                input_ports[conn_name] = port

    def _align_ports(
        self,
        aligned_ports: Sequence[str],
        input_ports: dict[str, DevicePort],
        output_ports: dict[str, DevicePort],
    ) -> None:
        for portname in aligned_ports:
            if not (portname in input_ports and portname in output_ports):
                continue

            # port2 is always slave to port1
            port1 = input_ports[portname]
            port2 = output_ports[portname]
            if port1.dx() != 0 and port2.dx() != 0:
                ydiff = port2.y0 - port1.y0
                port2._geometry.translate(0, -ydiff)
                for parent_port in port2._parentports.values():
                    parent_port.y0 -= ydiff
            if port1.dy() != 0 and port2.dy() != 0:
                xdiff = port2.x0 - port1.x0
                port2._geometry.translate(-xdiff, 0)
                for parent_port in port2._parentports.values():
                    parent_port.x0 -= xdiff

    def _connect_via_path(
        self,
        start_port: DevicePort,
        end_port: DevicePort,
        path_points: list[float],
        portname: str,
    ) -> GeomGroup:
        if len(path_points) % 3 > 0:
            msg = (
                f"Specified path for wire {portname} "
                f"should include 3 values for each point (x,y,angle)"
            )
            raise ValueError(msg)

        geometry = GeomGroup()
        current_port = deepcopy(start_port)
        for idx in range(0, len(path_points), 3):
            next_port = deepcopy(current_port)
            next_port.x0 = path_points[idx]
            next_port.y0 = path_points[idx + 1]
            next_port.set_angle(math.radians(path_points[idx + 2]))
            next_port.bf = not next_port.bf
            geometry += start_port.connector_function(current_port, next_port)
            next_port.bf = not next_port.bf
            current_port = deepcopy(next_port)

        geometry += start_port.connector_function(current_port, end_port)
        return geometry

    def _connect_ports(
        self,
        g: GeomGroup,
        input_ports: dict[str, DevicePort],
        output_ports: dict[str, DevicePort],
        external_ports: Sequence[str],
        paths: dict[str, list[float]],
    ) -> GeomGroup:
        for portname, input_port in input_ports.items():
            if portname in output_ports:
                output_port = output_ports[portname]
                if input_port.connector_function != output_port.connector_function:
                    msg = f"Incompatible ports for connection named {portname}"
                    raise IncompatiblePortError(msg)

                if portname in paths:
                    g += self._connect_via_path(
                        start_port=input_port,
                        end_port=output_port,
                        path_points=paths[portname],
                        portname=portname,
                    )
                else:
                    g += input_port.connector_function(input_port, output_port)
            elif portname in external_ports:
                input_port._geometry = GeomGroup()
                input_port._parentports = {}
                port = deepcopy(input_port)
                port.name = portname
                self._localp["external_ports"][portname] = port
            else:
                # Stacklevel=3 to point at code calling self.run() and not this method.
                msg = f"Port {portname} is unconnected."
                warnings.warn(msg, UserWarning, stacklevel=3)
        return g

    def geom(self) -> GeomGroup:
        """Draws the entire circuit.

        Returns
        -------
        GeomGroup
            The entire circuit geometry including connectors.

        """
        netlist = self._p["NETLIST"].entry_list
        external_ports = self._p["NETLIST"].external_ports
        aligned_ports = self._p["NETLIST"].aligned_ports
        paths = self._p["NETLIST"].paths

        g = GeomGroup()
        input_ports: dict[str, DevicePort] = {}
        output_ports: dict[str, DevicePort] = {}
        for instance_number, nle in enumerate(netlist, start=1):
            geom, device_ports = self._build_device_geometry(nle, instance_number)
            g += geom
            self._collect_mapped_ports(
                nle=nle,
                geom=geom,
                device_ports=device_ports,
                input_ports=input_ports,
                output_ports=output_ports,
            )

        self._align_ports(aligned_ports, input_ports, output_ports)
        return self._connect_ports(g, input_ports, output_ports, external_ports, paths)

    def ports(self) -> None:
        """Add external ports that are not connected in the netlist.

        Returns
        -------
        None

        """
        ext_ports = self._localp["external_ports"]
        for p in ext_ports.values():
            self.addport(deepcopy(p))


_DeviceList: dict[str, type[Device]] = {"X": Circuit}


def register_devices_in_module(module_name: str) -> None:
    """Register the device names in a global variable.

    To be called at the end of a python module containing device classes that inherit
    the `Device` class.

    Parameters
    ----------
    module_name : str
        The python module name, if used in the same file, just use `__name__`.

    Returns
    -------
    None

    """
    for _, obj in inspect.getmembers(sys.modules[module_name]):
        if inspect.isclass(obj):
            # Recursively check bases for Device
            baseobj = obj.__bases__[0]
            while baseobj != Device:
                if len(baseobj.__bases__) != 0:
                    baseobj = baseobj.__bases__[0]
                else:
                    break

            if baseobj == Device:
                oj = obj()
                oj.initialize()
                _DeviceList[oj._name] = obj
                print(f"Loaded {oj._name}: {oj._description}")


@deprecated(
    "This function is deprecated and will be removed "
    "in a future version. Use register_devices_in_module() instead."
)
def registerDevicesInModule(module_name: str) -> None:  # noqa: N802
    """Register the device names in a global variable.

    To be called at the end of a python module containing device classes that inherit
    the `Device` class.

    Parameters
    ----------
    module_name : str
        The python module name, if used in the same file, just use `__name__`.

    Returns
    -------
    None

    """
    register_devices_in_module(module_name)


def create_device_library(devname: str, params: dict, filename: str) -> None:
    """Generate a GDS file with a re-usable GDS-format device.

    Also exports ports as text element in GDS.
    Flattens everything.

    Parameters
    ----------
    devname : str
        The registered name of the device.
    params : dict
        The parameters to be used when saving. Modifies the default.
    filename: str
        The output library filename

    Returns
    -------
    None

    """
    dev = Device.build_registered(devname)
    for key, value in params.items():
        dev.set_param(key, value)
    g = dev.run()
    for p, val in dev._ports.items():
        idtxt = (
            f"__PORT__ {p} "
            f"{val.angle_to_text()} "
            f"{val.__class__.__module__} "
            f"{val.__class__.__name__}"
        )
        g += make_text(val.x0, val.y0, idtxt, 0, 0)

    g = g.flatten()

    gdsw = GDSWriter()
    gdsw.open_library(filename)
    gdsw.write_structure(devname, g)
    gdsw.close_library()


@deprecated(
    "This function is deprecated and will be removed "
    "in a future version. Use create_device_library() instead."
)
def CreateDeviceLibrary(devname: str, params: dict, filename: str) -> None:  # noqa: N802
    """Generate a GDS file with a re-usable GDS-format device.

    Also exports ports as text element in GDS.
    Flattens everything.

    DEPRECATED: Use create_device_library() instead.

    Parameters
    ----------
    devname : str
        The registered name of the device.
    params : dict
        The parameters to be used when saving. Modifies the default.
    filename: str
        The output library filename

    Returns
    -------
    None

    """
    create_device_library(devname, params, filename)


def export_device_schematics(filename: str = "SampleMakerLibrary.lel") -> None:
    """Generate a Layout Editor library file (LEL).

    This file contains the Devices currently loaded on the Device List. The library file
    can be used in combination with Layout Editor Schematic to produce spice netlists
    for circuit design.

    Parameters
    ----------
    filename : str, optional
        The library filename with .lel extension, by default "SampleMakerLibrary.lel".

    Returns
    -------
    None

    """
    with _Path(filename).open("w") as f:
        for devobj in _DeviceList.values():
            oj = devobj()
            oj.initialize()
            oj.parameters()
            oj.ports()
            if oj._name in {"X", "TABLE"}:
                continue

            f.write(f"<Component {devobj.__name__}>\n")
            f.write("<Description>\n {oj._description} \n</Description>\n")
            f.write("<Parameter>\n")
            for p, val in oj._p.items():
                valstring = " ".join(map(str, [val]))
                valstring = valstring.replace("<", "-").replace(">", "-")
                f.write("<string " + p + " " + valstring + ">\n")
            f.write("</Parameter>\n")
            f.write("<Prefix " + oj._name + ">\n")
            # f.write("<Label>\n$devicename\n</Label>\n")

            f.write("<Symbol>\n")
            dev = oj.build()
            g = dev.run()
            g = g.flatten()
            # bb = g.bounding_box()
            scale = 1
            # scale = bb.width
            # if bb.height>bb.width:
            #    scale = bb.height
            g.scale(0, 0, 100 / scale, 100 / scale)

            for g_member in g.group:
                if isinstance(g_member, Poly):
                    bb = g_member.bounding_box()
                    if bb.width < 2 and bb.height < 0.2:
                        continue
                    x = g_member.data[0::2]
                    y = g_member.data[1::2]
                    for i in range(len(x) - 1):
                        if (x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2 > 0.2:
                            f.write(
                                f"<Line {int(x[i])} {int(y[i])} "
                                f"{int(x[i + 1])} {int(y[i + 1])} wire>\n"
                            )

            dev.ports()
            port_info = (
                f"<Port {int(port.x0 * 100 / scale)} "
                f"{int(port.y0 * 100 / scale)} {pname}>\n"
                for pname, port in dev._ports.items()
            )
            f.writelines(port_info)

            f.write("</Symbol>\n")
            # f.write("<Offsetlabel 0 -50 -50>\n")
            f.write("<Netlist spice>\n")
            f.write("$devicename ")
            f.writelines(f"{pname} $node({pname}) " for pname in dev._ports)
            f.write(". ")
            for p in oj._p:
                f.write(f"{p} ${p} ")

            f.write("\n</Netlist>\n")
            f.write("</Component>\n")


@deprecated(
    "This function is deprecated and will be removed "
    "in a future version. Use export_device_schematics() instead."
)
def ExportDeviceSchematics(filename: str = "SampleMakerLibrary.lel") -> None:  # noqa: N802
    """Generate a Layout Editor library file (LEL).

    This file contains the Devices currently loaded on the Device List. The library file
    can be used in combination with Layout Editor Schematic to produce spice netlists
    for circuit design.

    DEPRECATED: Use export_device_schematics() instead.

    Parameters
    ----------
    filename : str, optional
        The library filename with .lel extension, by default "SampleMakerLibrary.lel".

    Returns
    -------
    None

    """
    export_device_schematics(filename)

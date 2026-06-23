"""Base waveguide library.

Implements a simple waveguide sequencer and optical ports.
This module can be used as template to develop different waveguide libraries.

"""

import math
from copy import deepcopy
from typing import Any
from warnings import deprecated

import numpy as np

import samplemaker.makers as sm
import samplemaker.sequencer as smseq
from samplemaker.devices import DevicePort
from samplemaker.routers import WaveguideConnect
from samplemaker.shapes import GeomGroup

# First step in defining a waveguide library is to define a sequencer
# and its dictionary.


# Let's define some options for the BaseWaveguide sequencer
def create_base_waveguide_options() -> smseq.OPTIONS_TYPE:
    """Create a dictionary with the default options for the BaseWaveguide sequencer.

    Returns
    -------
    smseq.OPTIONS_TYPE
        The default options for the BaseWaveguide sequencer.

    """
    options = smseq.default_options()
    # Let's define the default waveguide layer
    options["wgLayer"] = 1
    # For waveguide bends, let's use a fixed resolution
    options["bendResolution"] = 30
    # Let's define the default waveguide width
    options["defaultWidth"] = 0.3
    return options


@deprecated(
    "BaseWaveguideOptions is deprecated and will be removed"
    "in a future version, use create_base_waveguide_options instead"
)
def BaseWaveguideOptions() -> smseq.OPTIONS_TYPE:  # noqa: N802
    """Create a dictionary with the default options for the BaseWaveguide sequencer.

    DEPRECATED. Use create_base_waveguide_options instead.

    Returns
    -------
    smseq.OPTIONS_TYPE
        The default options for the BaseWaveguide sequencer.

    """
    return create_base_waveguide_options()


# Let's define the sequencer state class
# We could use the default, but we would like to store
# the current waveguide width as well using the parameter 'w'
class BaseWaveguideState(smseq.SequencerState):
    """The sequencer state for BaseWaveguide library."""

    def __init__(self) -> None:
        """Initialize the sequencer state.

        Defines 'w' as current waveguide width.
        """
        super().__init__()
        self.state["w"] = 0  # The value will be set by the INIT command


# Let's define the INIT command, which is always the first to execute
def base_waveguide_init(
    args: smseq.ARGS_TYPE,  # noqa: ARG001
    state: smseq.STATE_TYPE,
    options: smseq.OPTIONS_TYPE,
) -> GeomGroup:
    """Initialize the sequencer state.

    Parameters
    ----------
    args: smseq.ARGS_TYPE
        The arguments for the INIT command, which can be used to initialize the state.
    state : smseq.STATE_TYPE
        The sequencer state to be initialized.
    options : smseq.OPTIONS_TYPE
        The sequencer options, which can be used to initialize the state.

    Returns
    -------
    GeomGroup
        Empty geometry group.

    """
    smseq._init_state([], state, options)
    if not options["__no_init__"]:
        state["w"] = options["defaultWidth"]
    return GeomGroup()


@deprecated(
    "BaseWaveguideINIT is deprecated and will be removed"
    "in a future version, use base_waveguide_init instead"
)
def BaseWaveguideINIT(  # noqa: N802
    args: smseq.ARGS_TYPE,
    state: smseq.STATE_TYPE,
    options: smseq.OPTIONS_TYPE,
) -> GeomGroup:
    """Initialize the sequencer state.

    Parameters
    ----------
    args: smseq.ARGS_TYPE
        The arguments for the INIT command, which can be used to initialize the state.
    state : smseq.STATE_TYPE
        The sequencer state to be initialized.
    options : smseq.OPTIONS_TYPE
        The sequencer options, which can be used to initialize the state.

    Returns
    -------
    GeomGroup
        Empty geometry group.

    """
    return base_waveguide_init(args, state, options)


def base_waveguide_straight(
    args: smseq.ARGS_TYPE, state: smseq.STATE_TYPE, options: smseq.OPTIONS_TYPE
) -> GeomGroup:
    """Draw straight waveguide.

    Parameters
    ----------
    args : smseq.ARGS_TYPE
        1 argument: waveguide length.
    state : smseq.STATE_TYPE
        Current state.
    options : smseq.OPTIONS_TYPE
        The sequencer options.

    Returns
    -------
    GeomGroup
        The waveguide geometry.

    """
    dist = args[0]
    if dist == 0:
        return GeomGroup()

    # Let's draw a simple rectangle
    wg = sm.make_rect(0, 0, dist, state["w"], numkey=4, layer=options["wgLayer"])
    # Now rotate and translate according to pointer orientation
    wg.rotate_translate(state["x"], state["y"], state["a"])
    # Finally, update the state
    state["x"] += dist * math.cos(math.radians(state["a"]))
    state["y"] += dist * math.sin(math.radians(state["a"]))
    state["__OL__"] += dist
    return wg


@deprecated(
    "BaseWaveguideS is deprecated and will be removed"
    "in a future version, use base_waveguide_straight instead"
)
def BaseWaveguideS(  # noqa: N802
    args: smseq.ARGS_TYPE, state: smseq.STATE_TYPE, options: smseq.OPTIONS_TYPE
) -> GeomGroup:
    """Draw straight waveguide.

    DEPRECATED. Use base_waveguide_straight instead.

    Parameters
    ----------
    args : smseq.ARGS_TYPE
        1 argument: waveguide length.
    state : smseq.STATE_TYPE
        Current state.
    options : smseq.OPTIONS_TYPE
        The sequencer options.

    Returns
    -------
    GeomGroup
        The waveguide geometry.

    """
    return base_waveguide_straight(args, state, options)


def base_waveguide_bend(
    args: smseq.ARGS_TYPE, state: smseq.STATE_TYPE, options: smseq.OPTIONS_TYPE
) -> GeomGroup:
    """Draw circular bend waveguide.

    Parameters
    ----------
    args : smseq.ARGS_TYPE
        2 arguments: angle of bend (in degrees), radius of bend.
    state : smseq.STATE_TYPE
        Current state.
    options : smseq.OPTIONS_TYPE
        The sequencer options.

    Returns
    -------
    GeomGroup
        The waveguide geometry.

    """
    angle = args[0]
    radius = args[1]
    if angle == 0:
        return GeomGroup()

    wg = sm.make_arc(
        0,
        radius,
        radius,
        radius,
        -90,
        state["w"],
        0,
        abs(angle),
        vertices=options["bendResolution"],
        to_poly=True,
        layer=options["wgLayer"],
    )
    xf = radius * math.sin(math.radians(abs(angle)))
    yf = radius * (1 - math.cos(math.radians(abs(angle))))
    if angle < 0:
        wg.mirror_y(0)
        yf = -yf
    ept = sm.make_dot(xf, yf)  # helps calculating the end point
    # Now rotate and translate according to pointer orientation
    wg.rotate_translate(state["x"], state["y"], state["a"])
    ept.rotate_translate(state["x"], state["y"], state["a"])
    # Finally, update the state
    state["x"] = ept.x
    state["y"] = ept.y
    state["a"] += angle
    state["__OL__"] += radius * 2 * math.pi / 360 * abs(angle)

    return wg


@deprecated(
    "BaseWaveguideB is deprecated and will be removed"
    "in a future version, use base_waveguide_bend instead"
)
def BaseWaveguideB(  # noqa: N802
    args: smseq.ARGS_TYPE, state: smseq.STATE_TYPE, options: smseq.OPTIONS_TYPE
) -> GeomGroup:
    """Draw circular bend waveguide.

    DEPRECATED. Use base_waveguide_bend instead.

    Parameters
    ----------
    args : smseq.ARGS_TYPE
        2 arguments: angle of bend (in degrees), radius of bend.
    state : smseq.STATE_TYPE
        Current state.
    options : smseq.OPTIONS_TYPE
        The sequencer options.

    Returns
    -------
    GeomGroup
        The waveguide geometry.

    """
    return base_waveguide_bend(args, state, options)


def base_waveguide_cosine_bend(
    args: smseq.ARGS_TYPE, state: smseq.STATE_TYPE, options: smseq.OPTIONS_TYPE
) -> GeomGroup:
    """Draw cosine bend waveguide.

    Keeping the same the same direction, the function bends the waveguide using a cosine
    function.

    Parameters
    ----------
    args : smseq.ARGS_TYPE
        2 arguments: offset (in um), radius of bend.
    state : smseq.STATE_TYPE
        Current state.
    options : smseq.OPTIONS_TYPE
        The sequencer options.

    Returns
    -------
    GeomGroup
        The waveguide geometry.

    """
    off = args[0]
    radius = args[1]
    delta = 0.01  # at the very beginning and at the end go straight by delta
    radius -= delta
    if radius == 0:
        return GeomGroup()
    npts = options["bendResolution"]
    amp = math.pi * off / 4 / radius
    t = np.linspace(0, 2, npts)
    s = [math.asin(math.tan(math.atan(amp) * x) / amp) for x in t if x < 1]
    s += [
        math.asin(math.tan(math.atan(amp) * (x - 2)) / amp) + math.pi
        for x in t
        if x >= 1
    ]
    s = np.array(s)
    xpts = s / math.pi * 2 * radius + state["x"]
    ypts = off * (np.cos(s + math.pi) + 1) / 2 + state["y"]
    xpts = np.append(xpts[0], xpts + delta)
    xpts = list(np.append(xpts, xpts[-1] + delta))
    ypts = np.append(ypts[0], ypts)
    ypts = list(np.append(ypts, ypts[-1]))
    ol = np.sum(np.sqrt(np.power(np.ediff1d(xpts), 2) + np.power(np.ediff1d(ypts), 2)))
    wg = sm.make_path(xpts, ypts, state["w"], to_poly=True, layer=options["wgLayer"])
    outdot = sm.make_dot(xpts[-1], ypts[-1])
    wg.rotate(state["x"], state["y"], state["a"])
    outdot.rotate(state["x"], state["y"], state["a"])
    state["x"] = outdot.x
    state["y"] = outdot.y
    state["__OL__"] += ol
    return wg


@deprecated(
    "BaseWaveguideC is deprecated and will be removed"
    "in a future version, use base_waveguide_cosine_bend instead"
)
def BaseWaveguideC(  # noqa: N802
    args: smseq.ARGS_TYPE, state: smseq.STATE_TYPE, options: smseq.OPTIONS_TYPE
) -> GeomGroup:
    """Draw cosine bend waveguide.

    Keeping the same the same direction, the function bends the waveguide using a cosine
    function.

    DEPRECATED. Use base_waveguide_cosine_bend instead.

    Parameters
    ----------
    args : smseq.ARGS_TYPE
        2 arguments: offset (in um), radius of bend.
    state : smseq.STATE_TYPE
        Current state.
    options : smseq.OPTIONS_TYPE
        The sequencer options.

    Returns
    -------
    GeomGroup
        The waveguide geometry.

    """
    return base_waveguide_cosine_bend(args, state, options)


def base_waveguide_taper(
    args: smseq.ARGS_TYPE, state: smseq.STATE_TYPE, options: smseq.OPTIONS_TYPE
) -> GeomGroup:
    """Draw linear taper.

    Parameters
    ----------
    args : smseq.ARGS_TYPE
        2 arguments: length of taper (in um), final width (if <0, the defaultWidth value
        is used).
    state : smseq.STATE_TYPE
        Current state.
    options : smseq.OPTIONS_TYPE
        The sequencer options.

    Returns
    -------
    GeomGroup
        The waveguide geometry.

    """
    dist = args[0]
    wf = args[1]
    if dist == 0:
        return GeomGroup()
    if wf < 0:
        wf = options["defaultWidth"]
    a = math.radians(state["a"])
    xf = state["x"] + dist * math.cos(a)
    yf = state["y"] + dist * math.sin(a)
    wg = sm.make_tapered_path(
        [state["x"], xf], [state["y"], yf], [state["w"], wf], layer=options["wgLayer"]
    )
    state["x"] = xf
    state["y"] = yf
    state["w"] = wf
    state["__OL__"] += dist
    return wg


@deprecated(
    "BaseWaveguideT is deprecated and will be removed"
    "in a future version, use base_waveguide_taper instead"
)
def BaseWaveguideT(  # noqa: N802
    args: smseq.ARGS_TYPE, state: smseq.STATE_TYPE, options: smseq.OPTIONS_TYPE
) -> GeomGroup:
    """Draw linear taper.

    DEPRECATED. Use base_waveguide_taper instead.

    Parameters
    ----------
    args : smseq.ARGS_TYPE
        2 arguments: length of taper (in um), final width (if <0, the defaultWidth value
        is used).
    state : smseq.STATE_TYPE
        Current state.
    options : smseq.OPTIONS_TYPE
        The sequencer options.

    Returns
    -------
    GeomGroup
        The waveguide geometry.

    """
    return base_waveguide_taper(args, state, options)


def base_waveguide_offset(
    args: smseq.ARGS_TYPE,
    state: smseq.STATE_TYPE,
    options: smseq.OPTIONS_TYPE,  # noqa: ARG001
) -> GeomGroup:
    """Offset the waveguide (jumps left or right of waveguide).

    Parameters
    ----------
    args : smseq.ARGS_TYPE
        1 argument: offset (in um), positive means on left of waveguide direction.
    state : smseq.STATE_TYPE
        Current state.
    options : smseq.OPTIONS_TYPE
        The sequencer options.

    Returns
    -------
    GeomGroup
        The waveguide geometry.

    """
    off = args[0]
    a = math.radians(state["a"] + 90)
    state["x"] += off * math.cos(a)
    state["y"] += off * math.sin(a)
    return GeomGroup()


@deprecated(
    "BaseWaveguideOFF is deprecated and will be removed"
    "in a future version, use base_waveguide_offset instead"
)
def BaseWaveguideOFF(  # noqa: N802
    args: smseq.ARGS_TYPE,
    state: smseq.STATE_TYPE,
    options: smseq.OPTIONS_TYPE,
) -> GeomGroup:
    """Offset the waveguide (jumps left or right of waveguide).

    DEPRECATED. Use base_waveguide_offset instead.

    Parameters
    ----------
    args : smseq.ARGS_TYPE
        1 argument: offset (in um), positive means on left of waveguide direction.
    state : smseq.STATE_TYPE
        Current state.
    options : smseq.OPTIONS_TYPE
        The sequencer options.

    Returns
    -------
    GeomGroup
        The waveguide geometry.

    """
    return base_waveguide_offset(args, state, options)


def create_base_waveguide_commands() -> smseq.COMMANDS_DICT_TYPE:
    """Create a dictionary with the command list and corresponding functions.

    Returns
    -------
    smseq.COMMANDS_DICT_TYPE
        The command list to be used by the sequencer.

    """
    command_list = smseq.default_command_list()
    command_list["INIT"] = (0, base_waveguide_init)
    command_list["S"] = (1, base_waveguide_straight)
    command_list["B"] = (2, base_waveguide_bend)
    command_list["C"] = (2, base_waveguide_cosine_bend)
    command_list["T"] = (2, base_waveguide_taper)
    command_list["OFF"] = (1, base_waveguide_offset)
    return command_list


@deprecated(
    "BaseWaveguideCommands is deprecated and will be removed"
    "in a future version, use init_waveguide_commands instead"
)
def BaseWaveguideCommands() -> smseq.COMMANDS_DICT_TYPE:  # noqa: N802
    """Create a dictionary with the command list and corresponding functions.

    DEPRECATED. Use init_waveguide_commands instead.

    Returns
    -------
    smseq.COMMANDS_DICT_TYPE
        The command list to be used by the sequencer.

    """
    return create_base_waveguide_commands()


# Finally, create a custom sequencer
class BaseWaveguideSequencer(smseq.Sequencer):
    """Simple waveguide sequencer."""

    def __init__(self, seq: list[list[Any]]) -> None:
        """Create a custom sequencer for simple waveguides.

        Parameters
        ----------
        seq : list[list[Any]]
            The sequence to be executed.

        """
        opts = create_base_waveguide_options()
        state = BaseWaveguideState()
        cmds = create_base_waveguide_commands()
        super().__init__(seq, opts, state, cmds)


# some global connector options
BaseWaveguideConnectorOptions: dict[str, float | smseq.OPTIONS_TYPE] = {
    "bending_radius": 3.0,
    "sequencer_options": create_base_waveguide_options(),
}


def connect_ports(port1: DevicePort, port2: DevicePort) -> GeomGroup:
    """Connect two waveguide ports using the BaseWaveguideSequencer.

    Parameters
    ----------
    port1 : DevicePort
        The first port.
    port2 : DevicePort
        The second port.

    Returns
    -------
    GeomGroup
        The waveguide geometry.

    """
    radius = BaseWaveguideConnectorOptions["bending_radius"]
    if not isinstance(radius, float):
        msg = "BaseWaveguideConnectorOptions['bending_radius'] must be a float"
        raise TypeError(msg)

    res = WaveguideConnect(port1, port2, radius)
    if res[0]:
        so = BaseWaveguideSequencer(res[1])
        seq_options = BaseWaveguideConnectorOptions["sequencer_options"]
        if not isinstance(seq_options, dict):
            msg = "BaseWaveguideConnectorOptions['sequencer_options'] must be a dict"
            raise TypeError(msg)

        so.options = deepcopy(seq_options)
        g = so.run()
        g.rotate_translate(port1.x0, port1.y0, math.degrees(port1.angle()))
        return g
    return GeomGroup()


@deprecated(
    "BaseWaveguideConnector is deprecated and will be removed"
    "in a future version, use connect_ports instead"
)
def BaseWaveguideConnector(port1: DevicePort, port2: DevicePort) -> GeomGroup:  # noqa: N802
    """Connect two waveguide ports using the BaseWaveguideSequencer.

    DEPRECATED. Use connect_ports instead.

    Parameters
    ----------
    port1 : DevicePort
        The first port.
    port2 : DevicePort
        The second port.

    Returns
    -------
    GeomGroup
        The waveguide geometry.

    """
    return connect_ports(port1, port2)


# Now let's create a new DevicePort with a connector function
class BaseWaveguidePort(DevicePort):
    """A simple waveguide port, with a connector function to connect to other ports."""

    def __init__(
        self,
        x0: float,
        y0: float,
        orient: str = "East",
        width: float = 0,
        name: str = "",
    ) -> None:
        """Initialize a waveguide port.

        Parameters
        ----------
        x0 : float
            x coordinate of the port.
        y0 : float
            y coordinate of the port.
        orient : str, optional
            Orientation of the port, by default "East". Can be "North", "South", "East",
            "West" or their first letters.
        width : float, optional
            Width of the waveguide port.
        name : str, optional
            Name of the waveguide port.

        """
        orient = orient.lower()
        horizontal = True
        forward = True
        if orient in {"west", "w"}:
            forward = False
        if orient in {"north", "n"}:
            horizontal = False
        if orient in {"south", "s"}:
            horizontal = False
            forward = False

        super().__init__(x0, y0, horizontal, forward)
        self.width = width
        self.name = name
        self.connector_function = connect_ports

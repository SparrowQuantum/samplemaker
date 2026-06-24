"""Classes to handle custom sequences of shapes (e.g. waveguides).

The concept of sequence
-----------------------

A sequence in `samplemaker` is a list of instructions to be executed in sequence
that act on a drawing state machine.
The machine is initialized with some internal setting and each instruction can
modify the internal settings of the machine. Moreover, it is expected that
some of the instruction actually return a geometry that is eventually drawn
on screen, but that is not necessary.

While the above description might sound abstract, the sequencer is nothing more
than a compiler of a short code with user-defined instructions.
It becomes very handy when designing waveguides.

Each instruction is built as a list contaning a text command followed by an arbitrary
number of parameters (or function arguments).
For example, one simple sequence of commands is

    seq = [["S",3], ["B", 90, 3]]

It contains two instructions. The first is the command "S" with just one argument
and the second is the command "B" with two arguments.

The sequencer requires the user to provide a set of functions to be called when
the instruction is recognized. For example "S" can be associated to the 'go straight'
command and a function

    def S_command(args,state,options):
        # Draw something based on the machine state, options and args.
        seq.state["x"] += args[0] # Modify the state

    # tell the sequencer to call S_command whenever the "S" instruction is encountered:
    seq_dictionary = {"S": (1, S_command)}

The sequencer offers a template-based programming of arbitrary sequences.
In this way, the same sequence can be used with different dictionaries and result
in different geometries that use the same conceptual instructions.

When is this useful? Mostly when using routines that automatically perform
routing between parts of the design (e.g. waveguide routing). If the function
returns a sequence instead of an actual geometry, the same routing function
can be used for different circuit design platforms.

Additionally, the level of automation in circuit design can be highly improved, as
some functions can be 'smart' and perform specific actions depending on the current
machine state.

The best way to learn how to master sequencers is to look at the tutorials distributed
with `samplemaker`.

"""

import math
import warnings
from collections.abc import Callable, MutableMapping, Sequence
from copy import deepcopy
from inspect import signature
from typing import Any, TypeAlias

from samplemaker.devices import _DeviceList
from samplemaker.shapes import GeomGroup

ARGS_TYPE: TypeAlias = Sequence[Any]
STATE_TYPE: TypeAlias = MutableMapping[str, Any]
OPTIONS_TYPE: TypeAlias = MutableMapping[str, Any]
SEQ_TYPE: TypeAlias = Sequence[Sequence[Any]]

COMMAND_CALLABLE_TYPE: TypeAlias = Callable[
    [ARGS_TYPE, STATE_TYPE, OPTIONS_TYPE], GeomGroup
]
_CMD_DICT_VAL_TYPE: TypeAlias = tuple[int, COMMAND_CALLABLE_TYPE]
COMMANDS_DICT_TYPE: TypeAlias = dict[str, _CMD_DICT_VAL_TYPE]


def _change_state(
    args: ARGS_TYPE,
    state: STATE_TYPE,
    options: OPTIONS_TYPE,  # noqa: ARG001
) -> GeomGroup:
    state[args[0]] = args[1]
    return GeomGroup()


def _center_state(
    args: ARGS_TYPE,
    state: STATE_TYPE,
    options: OPTIONS_TYPE,  # noqa: ARG001
) -> GeomGroup:
    state["__XC__"] = -state["x"] + args[0]
    state["__YC__"] = -state["y"] + args[1]
    return GeomGroup()


def _store_state(
    args: ARGS_TYPE,  # noqa: ARG001
    state: STATE_TYPE,
    options: OPTIONS_TYPE,  # noqa: ARG001
) -> GeomGroup:
    state["STORED"] += [[state["x"], state["y"]]]
    return GeomGroup()


def _init_state(
    args: ARGS_TYPE,  # noqa: ARG001
    state: STATE_TYPE,
    options: OPTIONS_TYPE,
) -> GeomGroup:
    if not options["__no_init__"]:
        state["x"] = 0
        state["y"] = 0
        state["a"] = 0
        state["__OL__"] = 0  # Optical length
        state["__XC__"] = 0
        state["__YC__"] = 0
        state["STORED"] = []
    return GeomGroup()


def _insert_device(
    args: ARGS_TYPE, state: STATE_TYPE, options: OPTIONS_TYPE
) -> GeomGroup:
    devname = args[0]
    inport = args[1]
    outport = args[2]
    if devname not in _DeviceList:
        msg = f"No device found with name {devname}."
        raise ValueError(msg)

    dev = _DeviceList[devname].build()
    dev._p = options["dev_" + devname]
    if hasattr(dev, "_seq"):
        dev._seq.state = deepcopy(state)
        dev._seq.options = deepcopy(options)

    g = dev.run()
    if inport not in dev._ports or outport not in dev._ports:
        msg = (
            f"Device {devname} has no port called {inport} or {outport}. "
            f"Available ports are {list(dev._ports.keys())}"
        )
        raise ValueError(msg)

    p1 = dev._ports[inport]
    xd = p1.x0
    yd = p1.y0
    ad = math.degrees(p1.angle()) + 180
    p2 = dev._ports[outport]
    xdo = p2.x0
    ydo = p2.y0
    ado = math.degrees(p2.angle())
    g.rotate(xd, yd, -ad + state["a"])
    g.translate(state["x"] - xd, state["y"] - yd)
    state["x"] += (xdo - xd) * math.cos(math.radians(state["a"] - ad)) - (
        ydo - yd
    ) * math.sin(math.radians(state["a"] - ad))
    state["y"] += (xdo - xd) * math.sin(math.radians(state["a"] - ad)) + (
        ydo - yd
    ) * math.cos(math.radians(state["a"] - ad))
    state["a"] += ado

    return g


def default_command_list() -> COMMANDS_DICT_TYPE:
    """Create a basic dictionary with basic commands required by the sequencer.

    These include

    * STATE: change the state variable to something else
    * CENTER: forces the current position state to change
    * STORE: stores the current position state
    * DEV: Inserts a device at the current postion

    Returns
    -------
    COMMANDS_DICT_TYPE
        The default command dictionary.

    """
    return {
        "INIT": (0, _init_state),
        "STATE": (2, _change_state),
        "CENTER": (2, _center_state),
        "STORE": (0, _store_state),
        "DEV": (3, _insert_device),
    }


def default_options() -> OPTIONS_TYPE:
    """Create default options for the sequencer.

    This returns the essential base options.

    Returns
    -------
    OPTIONS_TYPE
        Returns the default options for the sequencer.

    """
    defopts = {}
    for dname in _DeviceList:
        dev = _DeviceList[dname]()
        dev.parameters()
        defopts["dev_" + dname] = dev._p

    defopts["__no_init__"] = False  # Disable INIT command
    return defopts


class SequencerState:
    """Class to handle the state of the sequencer.

    The state is a dictionary that can be modified by the instructions in the sequence.
    The state is initialized to default values when a SequencerState object is created.

    """

    def __init__(self) -> None:
        """Initialize a sequencer state to default values.

        The default sequencer state contains the following variables:

        * 'x': The current x-coordinate of the sequencer. Initially 0
        * 'y': The current y-coordinate of the sequencer. Initially 0
        * 'a': The current angle (or orientation) of the sequencer. Initially 0 (east)
        * '__OL__': Stores the accumulated distance.
        * '__XC__': Stores the current x-coordinate when calling 'CENTER'
        * '__YC__': Stores the current y-coordinate when calling 'CENTER'
        * 'STORED': Stores the current position when calling 'STORE'

        """
        self.state: STATE_TYPE = {}
        self.state["x"] = 0
        self.state["y"] = 0
        self.state["a"] = 0
        self.state["__OL__"] = 0  # Optical length
        self.state["__XC__"] = 0
        self.state["__YC__"] = 0
        self.state["STORED"] = []


class Sequencer:
    """Class to handle the execution of a sequence of instructions."""

    def __init__(
        self,
        seq: SEQ_TYPE,
        seq_options: OPTIONS_TYPE,
        seq_state: SequencerState,
        seq_dictionary: COMMANDS_DICT_TYPE,
    ) -> None:
        """Initialize a new sequencer object.

        Requires a sequence, an option dictionary, a state object, and a dictionary to
        interpret commands.

        Parameters
        ----------
        seq : SEQ_TYPE
            The sequence to be executed (list of instructions).
        seq_options : OPTIONS_TYPE
            Dictionary with all the options to be passed to instructions.
        seq_state : SequencerState
            SequencerState object with the initial state of the sequencer.
        seq_dictionary : COMMANDS_DICT_TYPE
            The dictionary with instructions.

        """
        self.seq = seq
        self.options = seq_options
        self.dic = seq_dictionary
        self.state = seq_state.state
        self.debug_state = False

    def set_debug_state(self, value: bool) -> None:
        """Set debug mode.

        In debug mode the state is printed at all steps.

        Parameters
        ----------
        value : bool
            True to set debug mode on. False to set debug mode off.

        Returns
        -------
        None

        """
        self.debug_state = value

    def get_state(self) -> STATE_TYPE:
        """Get the current state of the sequencer.

        Returns
        -------
        STATE_TYPE
            Returns the current state of the sequencer.

        """
        return deepcopy(self.state)

    def reset(self) -> None:
        """Reset the sequencer position state to (0,0) and direction state to zero.

        Returns
        -------
        None

        """
        self.state["x"] = 0
        self.state["y"] = 0
        self.state["a"] = 0

    def run(self) -> GeomGroup:
        """Execute the sequence and get the final geometry object.

        Returns
        -------
        GeomGroup
            The resulting geometry.

        """
        g = GeomGroup()
        init_fun = self.dic["INIT"][1]
        init_fun_sig = signature(init_fun)
        nargs = len(init_fun_sig.parameters)
        if nargs == 3:
            init_fun([], self.state, self.options)
        elif nargs == 2:
            # Legacy init function signature, only state and options are passed
            warnings.warn(
                "The supplied INIT command function signature is deprecated. "
                "Use the new signature with three parameters: args, state, and "
                "options.",
                DeprecationWarning,
                stacklevel=2,
            )
            init_fun(self.state, self.options)  # type: ignore[arg-type]
        else:
            msg = (
                f"The INIT command function must have either 2 or 3 parameters, "
                f"but it has {nargs} parameters."
            )
            raise ValueError(msg)

        for instr in self.seq:
            if not len(instr):
                continue

            cmd = instr[0]
            args = instr[1:]
            if cmd not in self.dic:
                msg = (
                    f"Command {cmd} does not exist. "
                    f"Available commands are {list(self.dic.keys())}"
                )
                raise ValueError(msg)

            action = self.dic[cmd]
            if action[0] != len(args):
                msg = (
                    f"Wrong number of arguments for command {cmd}."
                    f" Expected {action[0]}, got {len(args)}"
                )
                raise ValueError(msg)

            g += action[1](args, self.state, self.options)
            if self.debug_state:
                print(f"self state {self.state}")

        g.translate(self.state["__XC__"], self.state["__YC__"])
        self.state["x"] += self.state["__XC__"]
        self.state["y"] += self.state["__YC__"]
        for coords in self.state["STORED"]:
            coords[0] += self.state["__XC__"]
            coords[1] += self.state["__YC__"]
        if self.debug_state:
            print(f"final state {self.state}")

        return g

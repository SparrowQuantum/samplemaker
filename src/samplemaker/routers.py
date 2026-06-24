"""Automatic port-to-port routing functions."""

import math
import warnings
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray

import samplemaker.makers as sm
from samplemaker.devices import DevicePort


# The following are routines for the connector
def _connectable_facing(
    port1: DevicePort, port2: DevicePort, rad: float = 3
) -> tuple[bool, list[list[Any]]]:
    """Calculate if two ports are directly connectable and facing each other.

    This function returns true and a sequence if two ports are directly connectable and
    facing each other. The sequence is either a straight line or a cosine bend.

    Parameters
    ----------
    port1 : DevicePort
        Start port for the connection.
    port2 : DevicePort
        End port for the connection.
    rad : float, optional
        The maximum bend radius in um, by default 3.

    Returns
    -------
    bool
        True if connection succeeded, False otherwise.
    list[list[Any]]
        A sequence to perform the connection.

    """
    # Get the vector from port 1 to port 2
    dx = port2.x0 - port1.x0
    dy = port2.y0 - port1.y0
    if port1.dx() != 0:
        # Case1: port 1 is horizontal
        if abs(dy) < 2 * rad:
            # the y offset is small enough to use a C bend
            dxsign = 1
            if abs(dx) != 0:  # Note: sometimes this can be zero
                dxsign = dx / abs(dx)
            if port1.dx() + port2.dx() == 0 and dxsign == port1.dx():
                # facing each other checks
                if abs(dy) < 1e-3:
                    # will use straight line
                    return True, [["S", abs(dx)]]
                # will create a C bend
                slen = (abs(dx) - 2 * rad) / 2
                if slen < 0:
                    return True, [["C", port1.dx() * dy, abs(dx) / 2]]
                return True, [
                    ["S", slen],
                    ["C", port1.dx() * dy, rad],
                    ["S", slen],
                ]
        return False, []
    # Case2 : port 1 is vertical
    if abs(dx) < 2 * rad:
        # the y offset is small enough to use a C bend
        dysign = 1
        if abs(dy) != 0:
            dysign = dy / abs(dy)
        if port1.dy() + port2.dy() == 0 and dysign == port1.dy():
            # facing each other checks
            if abs(dx) < 1e-3:
                # will use straight line
                return True, [["S", abs(dy)]]
            # will create a C bend
            slen = (abs(dy) - 2 * rad) / 2
            if slen < 0:
                return True, [["C", -port1.dy() * dx, abs(dy) / 2]]
            return True, [
                ["S", slen],
                ["C", -port1.dy() * dx, rad],
                ["S", slen],
            ]
    return False, []


def _connectable_bend(
    port1: DevicePort, port2: DevicePort, rad: float = 3
) -> tuple[bool, list[list[Any]]]:
    """Calculate if two ports can be connected with a single bend.

    The function calculates the projected intersection of two straight paths and returns
    a  sequence that connects the ports. It might sometimes fail if ports are too close.

    Parameters
    ----------
    port1 : DevicePort
        Start port for the connection.
    port2 : DevicePort
        End port for the connection.
    rad : float, optional
        The maximum bend radius in um, by default 3.

    Returns
    -------
    bool
        True if connection succeeded, False otherwise.
    list[list[Any]]
        A sequence to perform the connection.

    """
    dx1 = port1.dx()
    dx2 = port2.dx()
    dy1 = port1.dy()
    dy2 = port2.dy()
    det = -dx1 * dy2 + dx2 * dy1
    if det == 0:
        return False, []
    dx = port2.x0 - port1.x0
    dy = port2.y0 - port1.y0
    t = (-dx * dy2 + dy * dx2) / det
    s = (-dx * dy1 + dy * dx1) / det
    if t > 0 and s > 0:
        xstp = (t - rad) * port1.dx()
        ystp = (t - rad) * port1.dy()
        s1 = math.sqrt(xstp * xstp + ystp * ystp)
        # xstp = (s-rad)*port2.dx()
        # ystp = (s-rad)*port2.dy()
        # s2 = math.sqrt(xstp*xstp+ystp*ystp)
        p1 = deepcopy(port1)
        p1.move_straight(s1)
        if det > 0:
            p1.bend_left(rad)
        else:
            p1.bend_right(rad)
        res = _connectable_facing(p1, port2, rad)
        seq = [["S", s1], ["B", det * 90, rad]] + res[1]
        return True, seq
    return False, []


def _connect_step(
    port1: DevicePort, port2: DevicePort, rad: float = 3
) -> tuple[bool, list[list[Any]]]:
    """Perform a single connection step.

    Attempts at getting port1 closer to `port2` by bending left or right or going
    straight. This connector works well for optical waveguides.

    Parameters
    ----------
    port1 : DevicePort
        Start port for the connection.
    port2 : DevicePort
        End port for the connection.
    rad : float, optional
        The maximum bend radius in um, by default 3.

    Returns
    -------
    bool
        True if connection succeeded, False otherwise.
    list[list[Any]]
        A sequence to perform the connection.

    """
    seq = []
    if port1.dx() != 0:
        if abs(port2.y0 - port1.y0) < 2 * rad:  # It's better to bend if too close
            s_len = -1
        else:
            s_len = port1.dx() * (port2.x0 + port2.dx() * rad - port1.x0) - rad
        # print("s_len in x",s_len)
        if port2.dx() == 0:
            if abs(port2.x0 - port1.x0) < 4 * rad:
                s_len += 2 * rad
            else:
                s_len -= 2 * rad
    else:
        if abs(port2.x0 - port1.x0) < 2 * rad:  # It's better to bend if too close
            s_len = -1
        else:
            s_len = port1.dy() * (port2.y0 + port2.dy() * rad - port1.y0) - rad
        # print("s_len in y",s_len)
        if port2.dy() == 0:
            if abs(port2.y0 - port1.y0) < 4 * rad:
                s_len += 2 * rad
            else:
                s_len -= 2 * rad

    if s_len > 0:
        # print("Guessing I should move S by ", s_len)
        port1.move_straight(s_len)
        seq = [["S", s_len]]
    # Now see if we get closer by going left or right
    p1 = deepcopy(port1)
    p1.fix()
    p1.bend_left(rad)
    dl = p1.dist(port2)
    res = _connectable_bend(p1, port2, rad)
    if res[0]:
        seq += [["B", 90, rad]] + res[1]
        return True, seq

    p1.reset()
    p1.bend_right(rad)
    dr = p1.dist(port2)
    res = _connectable_bend(p1, port2, rad)
    if res[0]:
        seq += [["B", -90, rad]] + res[1]
        return True, seq

    # print("L distance is ", dl)
    # print("R distance is ", dr)
    # Should I go left or right?
    if dl < dr:
        port1.bend_left(rad)
        port1.fix()
        return False, [*seq, ["B", 90, rad]]
    port1.bend_right(rad)
    port1.fix()
    return False, [*seq, ["B", -90, rad]]


def connect_waveguide_ports(
    port1: DevicePort, port2: DevicePort, rad: float = 3
) -> tuple[bool, list[list[Any]]]:
    """Calculate a sequence of commands to connect two ports.

    Given a start port and an end port, the function attempts to connect the ports using
    a sequence of straight lines (sequencer command S), 90 degrees bends (sequencer
    command B) and cosine bends (sequencer command C). The bending radius is also given.
    If the ports are too close to be connected via Manhattan-style connectors the
    function returns False. The sequence can be used in combination with any
    `samplemaker.sequencer.Sequencer` class that implements the commands S, C, and B.

    Parameters
    ----------
    port1 : DevicePort
        Start port for the connection.
    port2 : DevicePort
        End port for the connection.
    rad : float, optional
        The maximum bend radius in um, by default 3.

    Returns
    -------
    bool
        True if connection succeeded, False otherwise.
    list[list[Any]]
        A sequence that realizes the connection.

    """
    # Trivial cases first
    res = _connectable_facing(port1, port2, rad)
    if res[0]:
        # print("connectable facing")
        return True, res[1]
    res = _connectable_bend(port1, port2, rad)
    if res[0]:
        # print("connectable")
        return True, res[1]
    p1 = deepcopy(port1)
    seq = []
    for _ in range(4):
        res = _connect_step(p1, port2, rad)
        seq += res[1]
        if res[0]:
            return True, seq

    return False, []


def WaveguideConnect(  # noqa: N802
    port1: DevicePort, port2: DevicePort, rad: float = 3
) -> tuple[bool, list[list[Any]]]:
    """Calculate a sequence of commands to connect two ports.

    Given a start port and an end port, the function attempts to connect the ports using
    a sequence of straight lines (sequencer command S), 90 degrees bends (sequencer
    command B) and cosine bends (sequencer command C). The bending radius is also given.
    If the ports are too close to be connected via Manhattan-style connectors the
    function returns False. The sequence can be used in combination with any
    `samplemaker.sequencer.Sequencer` class that implements the commands S, C, and B.

    DEPRECATED: Use connect_waveguide_ports() instead.

    Parameters
    ----------
    port1 : DevicePort
        Start port for the connection.
    port2 : DevicePort
        End port for the connection.
    rad : float, optional
        The maximum bend radius in um, by default 3.

    Returns
    -------
    bool
        True if connection succeeded, False otherwise.
    list[list[Any]]
        A sequence that realizes the connection.

    """
    warnings.warn(
        "This function is deprecated and will be removed "
        "in a future version. Use connect_waveguide_ports() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return connect_waveguide_ports(port1, port2, rad)


def calculate_elbow_path(
    port1: DevicePort, port2: DevicePort, offset: float = 5
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate the connector path between two ports using an elbow style connection.

    Typically used for electrical interconnects. Does not check collisions.
    The offset parameter controls how far should the connector go straight out
    of the ports before attempting a connection (using cubic Bezier).

    Parameters
    ----------
    port1 : DevicePort
        Start port for the connection.
    port2 : DevicePort
        End port for the connection.
    offset : float, optional
        How far should the connector stick away from ports, by default 5.

    Returns
    -------
    xpts : np.ndarray
        1D array of X coordinates of the connector path.
    ypts : np.ndarray
        1D array of Y coordinates of the connector path.

    """
    x0 = port1.x0
    y0 = port1.y0
    r0 = port1.angle()
    # Rotate all in the reference of port1
    p2dot = sm.make_dot(port2.x0, port2.y0)
    p2dot.rotate(x0, y0, -math.degrees(r0))
    x1 = p2dot.x - x0
    y1 = p2dot.y - y0
    if abs(y1) < 0.005:
        xpts = np.array([0, x1])
        ypts = np.array([0, y1])
    else:
        aout = port2.angle() - r0 % (2 * math.pi)
        # offset
        xs = offset
        xs1 = xs + 3 * offset
        xe = x1 + offset * math.cos(aout)
        ye = y1 + offset * math.sin(aout)
        xe1 = xe + 3 * offset * math.cos(aout)
        ye1 = ye + 3 * offset * math.sin(aout)
        t = np.array([0, 0.25, 0.5, 0.75, 1])
        xpts = (
            np.power(1 - t, 3) * xs
            + 3 * np.power(1 - t, 2) * t * xs1
            + 3 * (1 - t) * np.power(t, 2) * xe1
            + np.power(t, 3) * xe
        )
        ypts = 3 * (1 - t) * np.power(t, 2) * ye1 + np.power(t, 3) * ye
        xpts = np.append([0], xpts)
        xpts = np.append(xpts, [x1])
        ypts = np.append([0], ypts)
        ypts = np.append(ypts, [y1])

    cost = math.cos(r0)
    sint = math.sin(r0)
    x = xpts.copy()
    y = ypts.copy()
    xpts = cost * x - sint * y + x0
    ypts = sint * x + cost * y + y0

    return xpts, ypts


def ElbowRouter(  # noqa: N802
    port1: DevicePort, port2: DevicePort, offset: float = 5
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate the connector path between two ports using an elbow style connection.

    Typically used for electrical interconnects. Does not check collisions.
    The offset parameter controls how far should the connector go straight out

    of the ports before attempting a connection (using cubic Bezier).

    DEPRECATED: Use calculate_elbow_path() instead.

    Parameters
    ----------
    port1 : DevicePort
        Start port for the connection.
    port2 : DevicePort
        End port for the connection.
    offset : float, optional
        How far should the connector stick away from ports, by default 5.

    Returns
    -------
    xpts : np.ndarray
        1D array of X coordinates of the connector path.
    ypts : np.ndarray
        1D array of Y coordinates of the connector path.

    """
    warnings.warn(
        "This function is deprecated and will be removed "
        "in a future version. Use calculate_elbow_path() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return calculate_elbow_path(port1, port2, offset)

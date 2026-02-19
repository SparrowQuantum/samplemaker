"""
Basic functions to plot and inspect geometries.

These are very basic plotting functions to speed up the development of masks
and circuits. They can be used instead of writing and opening GDS files external
viewers.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon, Ellipse, Arrow, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.widgets import Slider

import samplemaker.shapes as smsh
from samplemaker.devices import Device, DevicePort
from samplemaker.shapes import GeomGroup

_ViewerCurrentSliders: list[Slider] = []
_ViewerCurrentDevice: Device | None = None
_ViewerCurrentAxes: plt.Axes | None = None


def __get_geom_patches(grp: GeomGroup):
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    patches = []
    for geom in grp.group:
        if geom.layer < 0:
            continue
        lcolor = colors[np.mod(geom.layer, 10)]
        if isinstance(geom, smsh.Poly):
            n = int(len(geom.data) / 2)
            xy = np.reshape(geom.data, (n, 2))
            tmpp = Polygon(xy, closed=True)
            tmpp.set_facecolor(lcolor)
            patches.append(tmpp)
        elif isinstance(geom, smsh.Circle):
            tmpc = Circle((geom.x0, geom.y0), geom.r)
            tmpc.set_facecolor(lcolor)
            patches.append(tmpc)
        elif isinstance(geom, smsh.Path):
            xy = np.transpose([geom.xpts, geom.ypts])
            tmpp = Polygon(xy, closed=False)
            tmpp.set_edgecolor(lcolor)
            tmpp.set_fill(False)
            patches.append(tmpp)
        elif isinstance(geom, smsh.Text):
            print("text display is not supported, please convert to polygon first.")
        elif isinstance(geom, smsh.Ellipse):
            tmpe = Ellipse((geom.x0, geom.y0), geom.r * 2, geom.r1 * 2, angle=geom.rot)
            tmpe.set_facecolor(lcolor)
            patches.append(tmpe)
        elif isinstance(geom, smsh.Ring):
            gpl = geom.to_polygon()
            geom = gpl.group[0]
            n = int(len(geom.data) / 2)
            xy = np.reshape(geom.data, (n, 2))
            tmpp = Polygon(xy, closed=True)
            tmpp.set_facecolor(lcolor)
            patches.append(tmpp)
        elif isinstance(geom, smsh.Arc):
            gpl = geom.to_polygon()
            geom = gpl.group[0]
            n = int(len(geom.data) / 2)
            xy = np.reshape(geom.data, (n, 2))
            tmpp = Polygon(xy, closed=True)
            tmpp.set_facecolor(lcolor)
            patches.append(tmpp)
        elif isinstance(geom, smsh.SRef):
            print("SRef and ARef display is not supported, please flatten first.")
    return patches


def _get_port_patches(port: DevicePort):
    if port.name == "":
        return []
    tpath = TextPath((port.x0, port.y0), port.name, size=1)
    return [Arrow(port.x0, port.y0, port.dx(), port.dy()), PathPatch(tpath)]


def __get_device_ports_patches(dev: Device):
    patches = []
    for port in dev._ports.values():
        patches += _get_port_patches(port)

    return patches


def GeomView(grp: GeomGroup):
    """
    Plot a geometry in a matplotlib window.

    Only polygons and circles are displayed. Most elements are either ignored or
    converted to polygon.

    No flattening is performed, thus structure references are not displayed.

    Parameters
    ----------
    grp : GeomGroup
        The geometry to be displayed.

    Returns
    -------
    None.

    """
    plt.close("all")
    fig, ax = plt.subplots()
    patches = __get_geom_patches(grp)
    p = PatchCollection(patches, match_original=True)
    ax.add_collection(p)
    plt.grid()
    plt.axis("equal")
    plt.show()


def __update_scrollbar(val):
    global _ViewerCurrentDevice
    global _ViewerCurrentSliders
    global _ViewerCurrentAxes
    dev = _ViewerCurrentDevice
    if dev is None or _ViewerCurrentAxes is None:
        return

    pn = 0
    for param in dev._p.keys():
        # print(_ViewerCurrentSliders[pn].val)
        dev.set_param(param, _ViewerCurrentSliders[pn].val)
        pn = pn + 1

    # xlim = _ViewerCurrentAxes.get_xlim()
    # ylim = _ViewerCurrentAxes.get_ylim()
    dev.use_references = False
    dev.initialize()
    g = dev.run()
    bb = g.bounding_box()
    # g=g.flatten()
    patches = __get_geom_patches(g)
    patches += __get_device_ports_patches(dev)
    p = PatchCollection(patches, match_original=True)
    _ViewerCurrentAxes.clear()
    _ViewerCurrentAxes.add_collection(p)
    _ViewerCurrentAxes.grid(True)
    _ViewerCurrentAxes.set_xlim((bb.llx, bb.urx()))
    _ViewerCurrentAxes.set_ylim((bb.lly, bb.ury()))
    _ViewerCurrentAxes.aspect = "equal"
    _ViewerCurrentAxes.set_title(dev._name)


def DeviceInspect(devcl: Device):
    """
    Interactive display of devices defined from `samplemaker.devices`.
    The device is rendered according to the default parameters.
    Additionally a set of scrollbars is created to interactively modify
    the parameters and observe the changes in real time.
    If the device includes ports, they are displayed as blue arrows.

    Parameters
    ----------
    devcl : samplemaker.devices.Device
        A device object to be displayed.

    Returns
    -------
    None.

    """
    global _ViewerCurrentDevice
    global _ViewerCurrentSliders
    global _ViewerCurrentAxes
    dev = devcl.build()
    _ViewerCurrentDevice = dev
    g = dev.run()
    g = g.flatten()
    plt.close("all")
    fig, ax = plt.subplots()
    patches = __get_geom_patches(g)
    patches += __get_device_ports_patches(dev)
    p = PatchCollection(patches, match_original=True)
    ax.add_collection(p)
    _ViewerCurrentAxes = ax
    plt.grid()
    plt.axis("equal")
    plt.title(dev._name)
    plt.subplots_adjust(bottom=0.5)
    _ViewerCurrentSliders = []
    n_params = len(dev._p.keys())
    for pn, param in enumerate(dev._p.keys()):
        ax_amp = plt.axes((0.25, 0.05 + pn * 1.0 / n_params * 0.35, 0.65, 0.03))
        minv = 0
        maxv = dev._p[param] * 10
        prange = dev._prange[param]
        if prange[1] != np.inf:
            maxv = prange[1]
        if prange[0] != 0:
            minv = prange[0]

        valstep = dev._p[param] / 10
        if valstep == 0:
            valstep = 0.1
        if isinstance(dev._ptype[param], int):
            maxv = int(maxv)
            valstep = 1
        if isinstance(dev._ptype[param], bool):
            maxv = 1
            valstep = 1
        if maxv == 0:
            maxv = 1
        samp = Slider(
            ax_amp,
            param,
            minv,
            maxv,
            valinit=dev._p[param],
            color="green",
            valstep=valstep,
        )
        samp.on_changed(__update_scrollbar)
        # cb = lambda x: onclick(x,param)
        # samp.connect_event('button_press_event', cb)
        _ViewerCurrentSliders.append(samp)

    plt.show()
    # return sliders

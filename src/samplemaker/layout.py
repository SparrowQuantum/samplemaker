"""Classes to configure the mask layout.

Mask layout
-----------
The `Mask` class handles the layout of the structure. It should be seen as the
main interface to the final GDS file. In essence, a Mask contains the database
of GDS cells to be exported.

It is recommended to instantiate only a single `Mask` object in each script.
An empty GDS file can be created as follows:

    mask = Mask("test_mask")
    mask.exportGDS()

By default, the GDS file contains a single structure called 'CELL00'. To modify
the symbol name, change the `Mask.mainsymbol` variable.
By default, new geometry elements should be added to the main cell with the
`Mask.addToMainCell` method. To add more cells manually, use `Mask.addCell` instead.
At export time, all cell references that are not referenced by the main cell are
automatically discarded.

### Cache system
To speed up script execution, it is possible to activate a cache system by
the `Mask.set_cache` method.

    mask.set_cache(True)

The cache uses the python `pickle` package to save the current geometry to file
when exporting to GDS. When the script starts, if the cache is turned on and the
cache file exist, the data is loaded in memory and updated only where necessary.

Additionally, if a structure is not changed and a GDS file already exists, the
GDS data from the previous file is loaded and copied to the output file.

By default, the cache is disabled as for small masks with few polygons there is
no significant advantage in run time. Using the cache is highly recommended for large
masks.

### Electron beam lithography and write-fields
A write-field is a square area of the design where electron-beam lithography
tools write without moving the stage. Within this area, the patterns are usually
accurately reproduced.
When masks contain multiple write-fields, it is recommended to avoid placing
polygons that overlap the fields, as the coarse stage motion will likely result
in so-called stitching errors.

To help the mask design process, it is possible to define and display write-fields
in the `Mask` class. These can either be added individually or as a grid. To add
a 10x10, 500x500 um2 large write-field grid:

    mask.addWriteFieldGrid(500,0,0,10,10)

Write-fields are only used as a visual aid in `samplemaker` to assist the placement
of geoemetries in the mask.

Aligment marks
--------------

When running multiple exposures in UV or e-beam lithography, it is usually required
to place aligment marks in the layout.
Marks are defined separately using the `Marker` and `MarkerSet` classes.
The common approach is to define a named `MarkerSet` and add it to the list of
marker sets in the `Mask` class:

    markerset = MarkerSet("Ebeam1", markdev,
                x0=0,y0=0,mset=4,xdist=2000,ydist=2000)
    mask.addMarkers(markerset)

The above example creates a 2x2 mark set (mset=4) 2 mm apart called "Ebeam1".
The actual shape used to draw the marker is provided by the Device object "markdev"
(see `samplemaker.devices` submodule).


Devices and Device tables
-------------------------
Samplemaker features a Device creation system in the `samplemaker.devices` submodule.
It allows generating parametric drawings which can be reused and instantiated with
different parameters. Device tables are one- or two-dimensional arrays of devices
created by sweeping some of the parameters.
The `DeviceTable` class helps in generating these tables without using "for" loops.
It places the devices consistently according to their bounding boxes.
Device tables are great for making parametric sweeps for e.g. lithographic tuning
of devices or testing different device property.

Additionally, it is possible to define table headers (text to be placed on the
side of each table) to print the parameter being sweeped and its values.
This is done via the `DeviceTableAnnotations` class.

"""

import math
import pickle  # for caching
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path as _Path
from typing import Any, TypeAlias
from warnings import deprecated

from samplemaker import (
    LayoutPool,
    _BoundingBoxPool,
    _DeviceCountPool,
    _DeviceLocalParamPool,
    _DevicePool,
    _legacy,
)
from samplemaker.devices import Device, DevicePort, IncompatiblePortError
from samplemaker.gdsreader import GDSReader
from samplemaker.gdswriter import GDSWriter
from samplemaker.makers import make_aref, make_circle, make_path, make_text
from samplemaker.shapes import Box, GeomGroup, SRef

TAB_POS_TYPE: TypeAlias = list[list[tuple[float, float]]]
TAB_POS_INPUT_TYPE: TypeAlias = Sequence[Sequence[Sequence[float]]]


class Marker:
    """Class that defines a single Marker.

    Use this class with custom devices to place a single marker in the layout.
    """

    def __init__(self, name: str, dev: Device, x0: float = 0, y0: float = 0) -> None:
        """Initialize the Marker class.

        Parameters
        ----------
        name : str
            Provide a string to define the mark group.
        dev : samplemaker.devices.Device
            A device object that produces a marker.
        x0 : float, optional
            Position of the marker, x-coordinate, by default 0.
        y0 : float, optional
            Position of the marker, y-coordinate, by default 0.

        """
        self.name = name
        self.dev = dev
        self.x0 = x0
        self.y0 = y0

    def get_geom(self) -> GeomGroup:
        """Create the marker geometry.

        Returns
        -------
        GeomGroup
            A geometry containing the marker.

        """
        self.dev.use_references = True
        g = self.dev.run()
        g.translate(self.x0, self.y0)
        return g


class MarkerSet(Marker):
    """Class that defines a set of markers."""

    def __init__(
        self,
        name: str,
        dev: Device,
        x0: float = 0,
        y0: float = 0,
        mset: int = 4,
        xdist: float = 1000,
        ydist: float = 1000,
    ) -> None:
        """Initialize the MarkerSet class.

        Parameters
        ----------
        name : str
            The name of the marker set.
        dev : samplemaker.devices.Device
            A sample maker device to use for drawing the marker.
        x0 : float, optional
            Position of the marker, x-coordinate, by default 0.
        y0 : float, optional
            Position of the marker, y-coordinate, by default 0.
        mset : int, optional
            Number of markers (can be 1, 2 or 4), by default 4.
        xdist : float, optional
            X-distance between two markers, by default 1000.
        ydist : float, optional
            Y-distance between two markers, by default 1000.

        """
        super().__init__(name, dev, x0, y0)
        self.mset = mset
        self.xdist = xdist
        self.ydist = ydist

    def get_geom(self) -> GeomGroup:
        """Create the marker geometry and places copies of them in the mask.

        Returns
        -------
        GeomGroup
            A geometry containing the marker.

        """
        self.dev.use_references = True
        g = self.dev.run()
        sref = g.group[0]
        if self.mset == 2:
            return make_aref(
                self.x0,
                self.y0,
                sref.cellname,
                sref.group,
                2,
                1,
                self.xdist,
                0,
                0,
                self.ydist,
            )
        if self.mset == 4:
            return make_aref(
                self.x0,
                self.y0,
                sref.cellname,
                sref.group,
                2,
                2,
                self.xdist,
                0,
                0,
                self.ydist,
            )
        return g


class DeviceTableAnnotations:
    """Class to control the annotations of a DeviceTable.

    You can define headers on the four edges of a table. An instance of this object
    should be passed to `DeviceTable.set_annotations` method to add headers.
    """

    def __init__(
        self,
        rowfmt: str,
        colfmt: str,
        xoff: float,
        yoff: float,
        rowvars: Sequence[str],
        colvars: Sequence[str],
        text_width: float = 1,
        text_height: float = 10,
        left: bool = True,
        right: bool = True,
        above: bool = True,
        below: bool = True,
    ) -> None:
        """Initialize the DeviceTableAnnotations class.

        Parameters
        ----------
        rowfmt : str
            A template string for formatting the rows text. %I and %J will be replaced
            with the row and column number, respectively. %Cn and %Rn will be replaced
            by the n-th column and row variable value, defined in rowvars and colvars.
            For example if the colvars is ("var0","var1",), the format %C0 will be
            replaced by the value of var0 on each column and %C1 will be replaced by the
            value of var1.
        colfmt : str
            A template string for formatting the column text. Same as rowfmt.
        xoff : float
            Distance of header text from the edge of the table in the x direction.
        yoff : float
            As xoff but in the y direction.
        rowvars : Sequence[str]
            A sequence containing the names of variables that will change along rows.
        colvars : Sequence[str]
            A sequence containing the names of variables that will change along columns.
        text_width : float, optional
            Width of text to be rendered, by default 1.
        text_height : float, optional
            Size of text to be rendered, by default 10.
        left : bool, optional
            Render header on the left side of the table, by default True.
        right : bool, optional
            Render header on the right side of the table, by default True.
        above : bool, optional
            Render header on top of the table, by default True.
        below : bool, optional
            Render header on the bottom of the table, by default True.

        """
        self.colfmt = colfmt
        self.rowfmt = rowfmt
        self.colvars = colvars
        self.rowvars = rowvars
        self.left = left
        self.right = right
        self.above = above
        self.below = below
        self.xoff = xoff
        self.yoff = yoff
        self.text_width = text_width
        self.text_height = text_height
        self.to_poly = True

    def set_poly_text(self, to_poly: bool) -> None:
        """Set how the table annotations should be rendered.

        if to_poly is True, the text will be rendered as polygons. Otherwise the text
        is rendered as a text object.

        Parameters
        ----------
        to_poly : bool
            Set this to True to render polygon text.

        Returns
        -------
        None

        """
        self.to_poly = to_poly

    def render(
        self,
        i: int,
        j: int,
        rows: int,
        cols: int,
        x0: float,
        y0: float,
        rowdict: Mapping[str, Sequence[Any]],
        coldict: Mapping[str, Sequence[Any]],
    ) -> GeomGroup:
        """Render the text for a given element in a table.

        This function should not be called by the user. It is intended to be run by the
        DeviceTable class.

        Parameters
        ----------
        i : int
            Row index of the table.
        j : int
            Column index of the table.
        rows : int
            Number of rows.
        cols : int
            Number of columns.
        x0 : float
            X-Position of the item on the table.
        y0 : float
            Y-Position of the item on the table.
        rowdict : Mapping[str, Sequence[Any]]
            The dictionary associating the variables and values that change along rows.
        coldict : Mapping[str, Sequence[Any]]
            The dictionary associating the variables and values that change along
            columns.

        Returns
        -------
        samplemaker.shapes.GeomGroup
            A geometry with the annotation associated to the table element i,j.

        """
        coltxt = deepcopy(self.colfmt)
        rowtxt = deepcopy(self.rowfmt)
        coltxt = coltxt.replace("%I", str(i))
        rowtxt = rowtxt.replace("%I", str(i))
        coltxt = coltxt.replace("%J", str(j))
        rowtxt = rowtxt.replace("%J", str(j))
        for v in range(len(self.colvars)):
            pstr = "%C" + str(v)
            rval = coldict[self.colvars[v]][j]
            rval = round(rval * 1000) / 1000
            coltxt = coltxt.replace(pstr, str(rval))
            rowtxt = rowtxt.replace(pstr, str(rval))
        for v in range(len(self.rowvars)):
            pstr = "%R" + str(v)
            rval = rowdict[self.rowvars[v]][i]
            rval = round(rval * 1000) / 1000
            coltxt = coltxt.replace(pstr, str(rval))
            rowtxt = rowtxt.replace(pstr, str(rval))
        g = GeomGroup()
        if self.left and j == 0:
            x = x0 - self.xoff
            y = y0
            g += make_text(x, y, rowtxt, self.text_height, self.text_width)
        if self.right and j == (cols - 1):
            x = x0 + self.xoff
            y = y0
            g += make_text(x, y, rowtxt, self.text_height, self.text_width)
        if self.above and i == (rows - 1):
            x = x0
            y = y0 + self.yoff
            g += make_text(x, y, coltxt, self.text_height, self.text_width)
        if self.below and i == 0:
            x = x0
            y = y0 - self.yoff
            g += make_text(x, y, coltxt, self.text_height, self.text_width)
        if self.to_poly:
            g.all_to_poly()
        return g


class DeviceTable:
    """A table of `samplemaker.devices.Device` objects.

    Will generate the device geometries in an array. The array can have 1 or more rows
    and 1 or more columns.

    On each row and column the device will be instantiated according to the values
    provided by the rowvars and colvars parameters.

    The positions of the rendered devices can be controlled by using the
    `set_table_positions` method to control. Alternatively, the `auto_align` method
    should be used to create a regularly-spaced table.
    """

    def __init__(
        self,
        dev: Device,
        nrow: int,
        ncol: int,
        rowvars: Mapping[str, Sequence[Any]],
        colvars: Mapping[str, Sequence[Any]],
    ) -> None:
        """Initialize the DeviceTable class.

        Parameters
        ----------
        dev : samplemaker.device.Device
            The device to be initialized in the table. The device should be already
            built via the build() command.
        nrow : int
            Number of rows (typically in y direction).
        ncol : int
            Number of columns (typically in x direction).
        rowvars : dict[str, Sequence]
            A dictionary that associates a device parameter to a sequence of values.
            The sequence of values should have the same size as the number of rows.
            On each row the device parameter will be changed according to the values
            listed. Multiple parameters can be swept simultaneously.
        colvars : dict[str, Sequence]
            Same as rowvars but controls the parameters being changed along columns.

        """
        self.dev = deepcopy(dev)  # A prebuilt device with preset parameters
        self.nrow = nrow
        self.ncol = ncol
        self.rowvars = rowvars
        self.colvars = colvars
        self.col_linkports: Sequence[tuple[str, str]] = ()
        self.row_linkports: Sequence[tuple[str, str]] = ()
        self.col_alignports = False
        self.row_alignports = False
        self.device_rotation = 0
        self.annotations = None
        self.use_references = True
        self.pos_xy = [[(0, 0) for _ in range(ncol)] for _ in range(nrow)]
        self._external_ports = {}
        self._geometries = []
        self._portmap = []
        self._backup_dev = deepcopy(dev)  # Keep it to reset the whole thing
        self._getgeom_ran = False

    def set_table_positions(self, positions: TAB_POS_INPUT_TYPE) -> None:
        """Define the position of each element in the table.

        Uses a 3-dimensional Sequence of the kind pos[i][j][k] where i,j control the row
        and column element and k=0,1 are the x and y coordinate.

        Parameters
        ----------
        positions : TAB_POS_INPUT_TYPE
            The sequence describing the position of each element in the table.

        Returns
        -------
        None

        """
        self.pos_xy = [
            [(positions[i][j][0], positions[i][j][1]) for j in range(self.ncol)]
            for i in range(self.nrow)
        ]
        self._geometries = []
        self._portmap = []

    def shift_table_origin(self, dx: float, dy: float) -> None:
        """Shift the position of all elements in the table by a given amount.

        Parameters
        ----------
        dx : float
            The amount to shift in the x direction.
        dy : float
            The amount to shift in the y direction.

        Returns
        -------
        None

        """
        newpos = [
            [
                (dx + self.pos_xy[i][j][0], dy + self.pos_xy[i][j][1])
                for j in range(self.ncol)
            ]
            for i in range(self.nrow)
        ]
        self.set_table_positions(newpos)

    def set_linked_ports(
        self,
        row_linkports: Sequence[tuple[str, str]] | None = None,
        col_linkports: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        """Automatically route ports between devices across columns and rows.

        Parameters
        ----------
        row_linkports : Sequence[tuple[str, str]] | None, optional
            Sequence of tuples containing port names that should be linked along rows,
            by default ().
        col_linkports : Sequence[tuple[str, str]] | None, optional
            Sequence of tuples containing port names that should be linked along
            columns, by default ().

        Returns
        -------
        None

        """
        self.col_linkports = col_linkports if col_linkports is not None else ()
        self.row_linkports = row_linkports if row_linkports is not None else ()

    def set_aligned_ports(
        self, align_rows: bool = False, align_columns: bool = False
    ) -> None:
        """Align ports along columns and rows.

        Parameters
        ----------
        align_rows : bool, optional
            If true, the first pair specified in row_linkports will be aligned, by
            default False.
        align_columns : bool, optional
            If true, the first pair specified in col_linkports will be aligned, by
            default False.

        Returns
        -------
        None

        """
        self.col_alignports = align_columns
        self.row_alignports = align_rows

    def set_device_rotation(self, device_rotation: float) -> None:
        """Rotate each device in the table.

        Parameters
        ----------
        device_rotation : float
            Angle in degrees.

        Returns
        -------
        None

        """
        self.device_rotation = device_rotation
        self._geometries = []
        self._portmap = []

    def set_annotations(self, annotations: DeviceTableAnnotations) -> None:
        """Set the table headers.

        Parameters
        ----------
        annotations : DeviceTableAnnotations
            The annotations to use.

        Returns
        -------
        None

        """
        self.annotations = annotations

    def get_external_ports(self) -> dict[str, DevicePort]:
        """Get all instantiated ports in the table.

        This allows tables to be connected to external devices or ports.

        Returns
        -------
        dict[str, DevicePort]
            A dictionary of all external ports.

        """
        return deepcopy(self._external_ports)

    def _build_geomarray(self) -> None:
        dev = self.dev
        self._portmap = [[{} for _ in range(self.ncol)] for _ in range(self.nrow)]
        self._geometries = [
            [GeomGroup() for _ in range(self.ncol)] for _ in range(self.nrow)
        ]
        for i in range(self.ncol):
            for var, valuelist in self.colvars.items():
                if len(valuelist) != self.ncol:
                    dev.set_param(var, valuelist[0])
                else:
                    dev.set_param(var, valuelist[i])
            for j in range(self.nrow):
                for var, valuelist in self.rowvars.items():
                    if len(valuelist) != self.nrow:
                        dev.set_param(var, valuelist[0])
                    else:
                        dev.set_param(var, valuelist[j])
                dev.set_angle(math.radians(self.device_rotation))
                dev.use_references = self.use_references
                self._geometries[j][i] = dev.run()
                self._portmap[j][i] = deepcopy(dev._ports)

    def __place_portmap(self) -> None:
        # Adjusts the portmap according to the current positions
        if self._geometries == []:
            self._build_geomarray()

        for i in range(self.ncol):
            for j in range(self.nrow):
                geom = self._geometries[j][i]
                geom.translate(self.pos_xy[j][i][0], self.pos_xy[j][i][1])

                for pp in self._portmap[j][i].values():
                    pp.x0 += self.pos_xy[j][i][0]
                    pp.y0 += self.pos_xy[j][i][1]

    def auto_align(self, min_dist_x: float, min_dist_y: float, numkey: int = 5) -> None:
        """Align devices in the table automatically according to their bounding boxes.

        Parameters
        ----------
        min_dist_x : float
            The distance between devices along columns.
        min_dist_y : float
            The distance between devices along rows.
        numkey : int, optional
            Selects which point of the bounding box should be aligned. Specify the
            box corner by visually matching it to the numerical keypad of standard
            keyboards (e.g. 1 is lower left corner, 3, lower-right, etc), by default 5
            (center).

        Returns
        -------
        None

        """
        if self._geometries == []:
            self._build_geomarray()

        # Get all BB (NOTE: this is slow for large devices with lots of features)
        bboxes = [
            [self._geometries[j][i].bounding_box() for i in range(self.ncol)]
            for j in range(self.nrow)
        ]
        pos_xy = [[[0.0, 0.0] for _ in range(self.ncol)] for _ in range(self.nrow)]
        # Place them according to the numkey point
        x_extr_r = [-1e23] * self.ncol
        x_extr_l = [1e23] * self.ncol
        y_extr_t = [-1e23] * self.nrow
        y_extr_b = [1e23] * self.nrow
        for i in range(self.ncol):
            for j in range(self.nrow):
                (bx, by) = bboxes[j][i].get_numkey_point(numkey)
                pos_xy[j][i][0] = -bx
                pos_xy[j][i][1] = -by
                bboxes[j][i].llx -= bx
                bboxes[j][i].lly -= by
                x_extr_r[i] = max(x_extr_r[i], bboxes[j][i].urx())
                y_extr_t[j] = max(y_extr_t[j], bboxes[j][i].ury())
                x_extr_l[i] = min(x_extr_l[i], bboxes[j][i].llx)
                y_extr_b[j] = min(y_extr_b[j], bboxes[j][i].lly)

        sx = [(x_extr_r[i - 1] - x_extr_l[i] + min_dist_x) for i in range(1, self.ncol)]
        sy = [(y_extr_t[j - 1] - y_extr_b[j] + min_dist_y) for j in range(1, self.nrow)]

        for i in range(self.ncol):
            for j in range(self.nrow):
                if i != 0:
                    pos_xy[j][i][0] += sum(sx[0:i])
                if j != 0:
                    pos_xy[j][i][1] += sum(sy[0:j])

        self.set_table_positions(pos_xy)

    def get_geometries(self) -> GeomGroup:
        """Build the table and returns all the geometries.

        Returns
        -------
        GeomGroup
            The rendered table geometry

        Raises
        ------
        IncompatiblePortError
            Raised if ports that are linked together have different connector functions.

        """
        if self._getgeom_ran:
            self._geometries = []
            self._portmap = []
            self.dev = deepcopy(self._backup_dev)

        if not self._geometries:
            self._build_geomarray()

        self.__place_portmap()
        g = GeomGroup()
        portmap = self._portmap
        for i in range(self.ncol):
            for j in range(self.nrow):
                geom = self._geometries[j][i]
                # The position is already set during __place_portmap()
                g += geom
                # annotations
                if self.annotations:
                    g += self.annotations.render(
                        j,
                        i,
                        self.nrow,
                        self.ncol,
                        self.pos_xy[j][i][0],
                        self.pos_xy[j][i][1],
                        self.rowvars,
                        self.colvars,
                    )

                # Column linking
                clports = self.col_linkports
                clalign = self.col_alignports
                rlports = self.row_linkports
                rlalign = self.row_alignports
                if i > 0:
                    for links in clports:
                        if links[0] in portmap[j][i - 1] and links[1] in portmap[j][i]:
                            p1 = portmap[j][i - 1][links[0]]
                            p2 = portmap[j][i][links[1]]

                            if p1.connector_function != p2.connector_function:
                                msg = (
                                    f"Incompatible ports for connection between "
                                    f"{p1.name} and {p2.name}"
                                )
                                raise IncompatiblePortError(msg)

                            if clalign and p1.dx() != 0 and p2.dx() != 0:
                                ydiff = p2.y0 - p1.y0
                                geom.translate(0, -ydiff)
                                for pp in portmap[j][i].values():
                                    pp.y0 -= ydiff

                            g += p1.connector_function(p1, p2)

                if j > 0:
                    for links in rlports:
                        if links[0] in portmap[j - 1][i] and links[1] in portmap[j][i]:
                            p1 = portmap[j - 1][i][links[0]]
                            p2 = portmap[j][i][links[1]]

                            if p1.connector_function != p2.connector_function:
                                msg = (
                                    f"Incompatible ports for connection between "
                                    f"{p1.name} and {p2.name}"
                                )
                                raise IncompatiblePortError(msg)

                            if rlalign and p1.dy() != 0 and p2.dy() != 0:
                                xdiff = p2.x0 - p1.x0
                                geom.translate(-xdiff, 0)
                                for pp in portmap[j][i].values():
                                    pp.x0 -= xdiff
                            g += p1.connector_function(p1, p2)

                # Store external ports to expose them
                for pp in portmap[j][i].values():
                    p1 = deepcopy(pp)
                    p1.name += f"_{j:d}_{i:d}"
                    self._external_ports[p1.name] = p1
        self._getgeom_ran = True
        return g

    @staticmethod
    def create_regular_grid(
        rows: int,
        cols: int,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        x0: float = 0,
        y0: float = 0,
    ) -> TAB_POS_TYPE:
        """Create coordinates for a regular table array.

        Returns a tuple that can be passed to `DeviceTable.set_table_positions`.

        Parameters
        ----------
        rows : int
            Number of rows.
        cols : int
            Number of columns.
        ax : float
            x-step along rows.
        ay : float
            y-step along rows.
        bx : float
            x-step along columns.
        by : float
            y-step along columns.
        x0 : float, optional
            x-coordinate of the origin, by default 0.
        y0 : float, optional
            y-coordinate of the origin, by default 0.

        Returns
        -------
        TAB_POS_TYPE
            3-dimensional list of positions.

        """
        return [
            [(x0 + i * ax + j * bx, y0 + i * ay + j * by) for i in range(cols)]
            for j in range(rows)
        ]

    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use DeviceTable.create_regular_grid() instead."
    )
    @staticmethod
    def Regular(  # noqa: N802
        rows: int,
        cols: int,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        x0: float = 0,
        y0: float = 0,
    ) -> TAB_POS_TYPE:
        """Create coordinates for a regular table array.

        Returns a tuple that can be passed to `DeviceTable.set_table_positions`.

        Parameters
        ----------
        rows : int
            Number of rows.
        cols : int
            Number of columns.
        ax : float
            x-step along rows.
        ay : float
            y-step along rows.
        bx : float
            x-step along columns.
        by : float
            y-step along columns.
        x0 : float, optional
            x-coordinate of the origin, by default 0.
        y0 : float, optional
            y-coordinate of the origin, by default 0.

        Returns
        -------
        TAB_POS_TYPE
            3-dimensional list of positions.

        """
        return DeviceTable.create_regular_grid(rows, cols, ax, ay, bx, by, x0, y0)


class Mask:
    """Main class for managing mask layouts and exporting them to GDS files."""

    def __init__(self, name: str = "layout001") -> None:
        """Initialize a Mask class. The name given is used as base name for file export.

        Parameters
        ----------
        name : str
            Name of the mask.

        """
        self.name = name
        self.mainsymbol = "CELL00"
        self.writefields = []
        self.cache = False
        self.clear()  # A new mask clears the pool

    def clear(self) -> None:
        """Clear the mask and all its content.

        Returns
        -------
        None

        """
        LayoutPool.clear()
        _DeviceCountPool.clear()
        _DeviceLocalParamPool.clear()
        _DevicePool.clear()
        _BoundingBoxPool.clear()
        self.writefields.clear()
        self.__add_basic_elements()

    def set_cache(self, cache: bool) -> None:
        """Turn on or off the cache system.

        When cache is turned on, the layout is stored on disk (with .cache extension)
        and reloaded when the mask is created again (for example when running the same
        script multiple times). Additionally, the cache system re-uses the GDS bitstream
        from a previously generated GDS file.

        Any changes made to the devices or instances are automatically detected
        and updated even if the cache is on.

        This option saves a lot of time when executing scripts while making minor
        changes to the device parameters.

        Parameters
        ----------
        cache : bool
            Set to True to turn cache on.

        Returns
        -------
        None

        """
        self.cache = cache
        if cache:
            self._import_cache()

    @staticmethod
    def __add_basic_elements() -> None:
        # Adding a circle to the layout pool
        if "_CIRCLE" not in LayoutPool:
            c = make_circle(0, 0, 1, layer=0, to_poly=True, vertices=12)
            LayoutPool["_CIRCLE"] = c
            _BoundingBoxPool["_CIRCLE"] = Box(-1, -1, 2, 2)

    def add_to_main_cell(self, geom_group: GeomGroup) -> None:
        """Add a geometry to the main cell.

        Parameters
        ----------
        geom_group : GeomGroup
            The geometry to be added.

        Returns
        -------
        None

        """
        if self.mainsymbol not in LayoutPool:
            LayoutPool[self.mainsymbol] = geom_group
        else:
            LayoutPool[self.mainsymbol] += geom_group

    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use Mask.add_to_main_cell() instead."
    )
    def addToMainCell(self, geom_group: GeomGroup) -> None:  # noqa: N802
        """Add a geometry to the main cell.

        DEPRECATED: Use Mask.add_to_main_cell() instead.

        Parameters
        ----------
        geom_group : GeomGroup
            The geometry to be added.

        Returns
        -------
        None

        """
        self.add_to_main_cell(geom_group)

    def add_cell(self, cellname: str, geom_group: GeomGroup) -> None:
        """Add a new cell to the GDS structure and assigns a geometry to it.

        Parameters
        ----------
        cellname : str
            The name of the cell.
        geom_group : GeomGroup
            The geometry to be added.

        Returns
        -------
        None

        """
        LayoutPool[cellname] = geom_group

    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use Mask.add_cell() instead."
    )
    def addCell(self, cellname: str, geom_group: GeomGroup) -> None:  # noqa: N802
        """Add a new cell to the GDS structure and assigns a geometry to it.

        DEPRECATED: Use Mask.add_cell() instead.

        Parameters
        ----------
        cellname : str
            The name of the cell.
        geom_group : GeomGroup
            The geometry to be added.

        Returns
        -------
        None

        """
        self.add_cell(cellname, geom_group)

    def get_cell(self, cellname: str) -> GeomGroup:
        """Get a reference to the GeomGroup corresponding to the cellname.

        Note: if you modify the cell geometry, it will also be modified in the mask.

        Parameters
        ----------
        cellname : str
            The name of the cell.

        Returns
        -------
        GeomGroup
            Reference to the geometry group.

        """
        if cellname not in LayoutPool:
            msg = f"Cell named {cellname} does not exist."
            raise ValueError(msg)

        return LayoutPool[cellname]

    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use Mask.get_cell() instead."
    )
    def getCell(self, cellname: str) -> GeomGroup:  # noqa: N802
        """Get a reference to the GeomGroup corresponding to the cellname.

        Note: if you modify the cell geometry, it will also be modified in the mask.

        DEPRECATED: Use Mask.get_cell() instead.

        Parameters
        ----------
        cellname : str
            The name of the cell.

        Returns
        -------
        GeomGroup
            Reference to the geometry group.

        """
        return self.get_cell(cellname)

    def _export_cache(self) -> None:
        print("Storing objects in cache file...")
        # Note that we do not need the full geometry, as we will just reload
        # it from the GDS file. So we keep the references only.
        # We might, however, need to re-compute the bounding boxes
        # for example in table autoalignment. Thus we replace the reference
        # groups with theyr bboxes
        # for key,val in LayoutPool.items():
        #    val.keep_refs_only()

        data = (
            LayoutPool,
            _DeviceCountPool,
            _DeviceLocalParamPool,
            _DevicePool,
            _BoundingBoxPool,
        )
        with _Path(self.name + ".cache").open("wb") as cachefile:
            pickle.dump(data, cachefile)
        print("Done.")

    def _import_cache(self) -> None:
        try:
            with _Path(self.name + ".cache").open("rb") as cachefile:
                print("Loading cache data...")
                data = pickle.load(cachefile)  # noqa: S301, to be replaced
                print("Done.")
                for key in data[0]:
                    LayoutPool[key] = data[0][key]
                LayoutPool.pop(self.mainsymbol, None)
                for key in data[1]:
                    _DeviceCountPool[key] = data[1][key]
                for key in data[2]:
                    _DeviceLocalParamPool[key] = data[2][key]
                for key in data[3]:
                    _DevicePool[key] = data[3][key]
                for key in data[4]:
                    _BoundingBoxPool[key] = data[4][key]
        except OSError:
            pass

    def __cleanup_cellref(self) -> None:
        # Remove useless references
        reflist = LayoutPool[self.mainsymbol].get_sref_list()
        reflist.add(self.mainsymbol)

        unref = []
        unref_hsh = []
        for ref in LayoutPool:
            if ref not in reflist:
                unref += [ref]

        for ref in unref:
            LayoutPool.pop(ref, None)
        for key, value in _DevicePool.items():
            if value in unref:
                unref_hsh += [key]

        for hsh in unref_hsh:
            # _DeviceCountPool.pop(hsh,None)
            _DeviceLocalParamPool.pop(hsh, None)
            _DevicePool.pop(hsh, None)

    def export_gds(self) -> None:
        """Finalize the mask, perform cache operations, if any, and write to GDS.

        Returns
        -------
        None

        """
        self.__cleanup_cellref()
        if self.cache:
            try:
                gdsr = GDSReader()
                gdsr.quick_read(self.name + ".gds")
                gdsr.celldata.pop(self.mainsymbol, None)
            except OSError:
                pass

        gdsw = GDSWriter()
        gdsw.open_library(self.name + ".gds")
        if self.cache:
            gdsw.write_pool_use_cache(LayoutPool, gdsr.celldata)
        else:
            gdsw.write_pool(LayoutPool)
        gdsw.close_library()
        if self.cache:
            self._export_cache()

    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use Mask.export_gds() instead."
    )
    def exportGDS(self) -> None:  # noqa: N802
        """Finalize the mask, perform cache operations, if any, and write to GDS.

        DEPRECATED: Use Mask.export_gds() instead.

        Returns
        -------
        None

        """
        self.export_gds()

    def import_gds(self, filename: str) -> None:
        """Import the full mask from GDS file.

        Parameters
        ----------
        filename : str
            name of the GDS file to read from.

        Returns
        -------
        None

        """
        self.clear()

        reflist = set()
        mainsymbolcandidates = set()

        gdsr = GDSReader()
        gdsr.quick_read(filename)
        for cname in gdsr.celldata:
            gg = gdsr.get_cell(cname)
            self.add_cell(cname, gg)
            reflist.update(gg.get_sref_list())
        for cname in gdsr.celldata:
            if cname not in reflist:
                mainsymbolcandidates.add(cname)
        if len(mainsymbolcandidates) == 1:
            self.mainsymbol = next(iter(mainsymbolcandidates))
        else:
            nsubref = 0
            for cname in mainsymbolcandidates:
                nrefs = len(LayoutPool[cname].get_sref_list())
                if nrefs > nsubref:
                    nsubref = nrefs
                    self.mainsymbol = cname
        # Update references after reading
        for cname in gdsr.celldata:
            for e in LayoutPool[cname].group:
                if isinstance(e, SRef):
                    e.group = LayoutPool[e.cellname]

    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use Mask.import_gds() instead."
    )
    def importGDS(self, filename: str) -> None:  # noqa: N802
        """Import the full mask from GDS file.

        DEPRECATED: Use Mask.import_gds() instead.

        Parameters
        ----------
        filename : str
            name of the GDS file to read from.

        Returns
        -------
        None

        """
        self.import_gds(filename)

    def add_markers(self, markerset: "MarkerSet") -> None:
        """Add a marker set to the mask.

        Parameters
        ----------
        markerset : MarkerSet
            The MarkerSet class to be added.

        Returns
        -------
        None

        """
        g = markerset.get_geom()
        if self.mainsymbol not in LayoutPool:
            LayoutPool[self.mainsymbol] = g
        else:
            LayoutPool[self.mainsymbol] += g

    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use Mask.add_markers() instead."
    )
    def addMarkers(self, markerset: "MarkerSet") -> None:  # noqa: N802
        """Add a marker set to the mask.

        DEPRECATED: Use Mask.add_markers() instead.

        Parameters
        ----------
        markerset : MarkerSet
            The MarkerSet class to be added.

        Returns
        -------
        None

        """
        self.add_markers(markerset)

    def add_writefield(
        self, wf_size: float, x0: float, y0: float, passes: int = 1, shift: float = 0
    ) -> None:
        """Add a square writefield centered in x0,y0.

        Parameters
        ----------
        wf_size : float
            Size in um of the writefield.
        x0 : float
            X-coordinate of the writefield center in um.
        y0 : float
            Y-coordinate of the writefield center in um.
        passes : int, optional
            Number of write-field passes, not shown in the mask, by default 1.
        shift : float, optional
            Shift of each multi-pass writefield, by default 0.

        Returns
        -------
        None

        """
        self.writefields += [(wf_size, x0, y0, passes, shift)]

    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use Mask.add_writefield() instead."
    )
    def addWriteField(  # noqa: N802
        self, wf_size: float, x0: float, y0: float, passes: int = 1, shift: float = 0
    ) -> None:
        """Add a square writefield centered in x0,y0.

        DEPRECATED: Use Mask.add_writefield() instead.

        Parameters
        ----------
        wf_size : float
            Size in um of the writefield.
        x0 : float
            X-coordinate of the writefield center in um.
        y0 : float
            Y-coordinate of the writefield center in um.
        passes : int, optional
            Number of write-field passes, not shown in the mask, by default 1.
        shift : float, optional
            Shift of each multi-pass writefield, by default 0.

        Returns
        -------
        None

        """
        self.add_writefield(wf_size, x0, y0, passes, shift)

    def add_writefield_grid(
        self,
        wf_size: float,
        x0: float,
        y0: float,
        nx: int | _legacy.MissingType = _legacy.MISSING,
        ny: int | _legacy.MissingType = _legacy.MISSING,
        passes: int = 1,
        shift: float = 0,
        **kwargs: int,
    ) -> None:
        """Create a grid nx x ny of writefields with given size and position.

        Parameters
        ----------
        wf_size : float
            Size in um of the writefield.
        x0 : float
            X-coordinate of the writefield center in um.
        y0 : float
            Y-coordinate of the writefield center in um.
        nx : int
            Number of write fields in x direction.
        ny : int
            Number of write fields in y direction.
        passes : int, optional
            Number of write-field passes, not shown in the mask, by default 1.
        shift : float, optional
            Shift of each multi-pass writefield, by default 0.
        kwargs : int
            Additional keyword arguments. Supports 'Nx' and 'Ny' for backward
            compatibility.

        Returns
        -------
        None

        """
        nx = _legacy.get_kwarg("nx", nx, "Nx", kwargs)
        ny = _legacy.get_kwarg("ny", ny, "Ny", kwargs)
        _legacy.ensure_empty_kwargs("Mask.add_writefield_grid", kwargs)
        _legacy.check_missing_args("Mask.add_writefield_grid", nx=nx, ny=ny)

        nx = _legacy.ensure_arg_type("nx", nx)
        ny = _legacy.ensure_arg_type("ny", ny)

        for i in range(nx):
            for j in range(ny):
                self.add_writefield(
                    wf_size, i * wf_size + x0, j * wf_size + y0, passes, shift
                )

        # Adding writefields
        if self.writefields:
            wfs = GeomGroup()
            for wf in self.writefields:
                s = wf[0]
                x = wf[1]
                y = wf[2]
                wfpath = make_path(
                    [-s / 2, s / 2, s / 2, -s / 2, -s / 2],
                    [-s / 2, -s / 2, s / 2, s / 2, -s / 2],
                    0.1,
                    layer=10,
                )
                wfpath.translate(x, y)
                wfs += wfpath
            self.add_to_main_cell(wfs)

    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use Mask.add_writefield_grid() instead."
    )
    def addWriteFieldGrid(  # noqa: N802
        self,
        wf_size: float,
        x0: float,
        y0: float,
        nx: int | _legacy.MissingType = _legacy.MISSING,
        ny: int | _legacy.MissingType = _legacy.MISSING,
        passes: int = 1,
        shift: float = 0,
        **kwargs: int,
    ) -> None:
        """Create a grid nx x ny of writefields with given size and position.

        DEPRECATED: Use Mask.add_writefield_grid() instead.

        Parameters
        ----------
        wf_size : float
            Size in um of the writefield.
        x0 : float
            X-coordinate of the writefield center in um.
        y0 : float
            Y-coordinate of the writefield center in um.
        nx : int
            Number of write fields in x direction.
        ny : int
            Number of write fields in y direction.
        passes : int, optional
            Number of write-field passes, not shown in the mask, by default 1.
        shift : float, optional
            Shift of each multi-pass writefield, by default 0.
        kwargs : int
            Additional keyword arguments. Supports 'Nx' and 'Ny' for backward
            compatibility.

        Returns
        -------
        None

        """
        self.add_writefield_grid(
            wf_size, x0, y0, nx=nx, ny=ny, passes=passes, shift=shift, **kwargs
        )

    def add_device_table(
        self, device_table: DeviceTable, x0: float, y0: float, cell: str = ""
    ) -> None:
        """Add a `DeviceTable` to the layout.

        Parameters
        ----------
        device_table : DeviceTable
            A DeviceTable object to be placed in the layout.
        x0 : float
            Controls the x position of the table center.
        y0 : float
            Controls the y position of the table center.
        cell : str, optional
            Adds the table to a named cell, by default "" (main cell).

        Returns
        -------
        None

        """
        geoms = device_table.get_geometries()
        bb = geoms.bounding_box()
        geoms.translate(-bb.cx() + x0, -bb.cy() + y0)
        if not cell:
            self.add_to_main_cell(geoms)
        else:
            self.add_cell(cell, geoms)

    @deprecated(
        "This method is deprecated and will be removed "
        "in a future version. Use Mask.add_device_table() instead."
    )
    def addDeviceTable(  # noqa: N802
        self, device_table: DeviceTable, x0: float, y0: float, cell: str = ""
    ) -> None:
        """Add a `DeviceTable` to the layout.

        DEPRECATED: Use Mask.add_device_table() instead.

        Parameters
        ----------
        device_table : DeviceTable
            A DeviceTable object to be placed in the layout.
        x0 : float
            Controls the x position of the table center.
        y0 : float
            Controls the y position of the table center.
        cell : str, optional
            Adds the table to a named cell, by default "" (main cell).

        Returns
        -------
        None

        """
        self.add_device_table(device_table, x0, y0, cell)

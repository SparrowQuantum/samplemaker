"""Shape classes.

Supported by the GDS format and most lithography systems / pattern generators.

Basic shapes in `samplemaker`
-----------------------------

The following GDS shapes are provided:

* `Poly`: closed curve polygon (GDS BOUNDARY element).
* `Path`: open curve polyline with given width (GDS PATH element).
* `Text`: text object for annotations (GDS TEXT element).
* `SRef`: reference to a single cell (GDS SREF element).
* `ARef`: array of reference to cells (GDS AREF element).

Additionally, the following non-GDS shapes are available:

* `Circle`: defined by center and radius.
* `Ellipse`: ellipses with rotation.
* `Ring`: elliptical or circular rings.
* `Arc`: like rings but covering a sector angle only.

Two more objects `Dot` and `Box` are available but not drawable.
They are sometimes useful for calculations
(e.g. bounding boxes or point transformations).

All the above classes (except for `SRef` and `ARef`) implement a `to_polygon` method
to convert all shapes back to GDS-exportable polygons and contain a `layer` information.
In practice, the above classes are never needed. The shapes are created using
the functions provided in the `samplemaker.makers` submodule and manipulated with the
`GeomGroup` class methods.

The `GeomGroup` object
----------------------

The most important class defined in the `samplemaker.shapes` module is the `GeomGroup`
class. It is an object the represents a drawing as a collection of the basic shapes
listed above. `GeomGroup` objects can be combined together, moved, scaled, and
manipulated. Additionally, several boolean operations are provided.

### Operations on `GeomGroup`
Some methods perform operations directly on the object from which they are called.
Boolean functions are an example of methods that modify the current object.
Other methods return a copy of the object without modifying it, for example the
`GeomGroup.flatten` method. It is recommended to check the reference of each command to
understand the function behavior.

When assigning an object to another one, a shallow copy is made, so that the copied
object and the source object still refer to the same geometry. To perform a deep copy,
use the method `GeomGroup.copy` instead:

    g1 = GeomGroup()
    g2 = g1 # Now both g1 and g2 refer to the same object
    g2.set_layer(3) # Both g1 and g2 have changed layer to 3
    g3 = g1.copy() # g3 is now a separate (deep) copy of g1.
    g3.set_layer(4) # Only g3 is set to layer 4.

In combining multiple geometries, it is often convenient to perform shallow copies
to save memory and computation time. For example

    # Shallow copy of geom2 into geomA. Any change to geom2 will affect geom_a:
    geom_a += geom2

    # Deep copy, any change to geom2 will not affect geom_b:
    geom_b += geom2.copy()


"""

import math
import pathlib
from collections.abc import Sequence
from copy import deepcopy
from typing import Collection, Self

import numpy as np

import samplemaker.resources.boopy as boopy
from samplemaker import _BoundingBoxPool

_glyphs = dict()

_STENCIL_FONT_FILENAME = "sm_stencil_font.txt"
_STENCIL_FONT_ENCODING = "ISO-8859-1"
_STENCIL_FONT_PATH = (
    pathlib.Path(__file__).parent / "resources" / _STENCIL_FONT_FILENAME
)


class GeomGroup:
    """A group of geometry elements.

    The group can contain primitive shapes (for example :class:`Poly`, :class:`Path`,
    :class:`Circle`) and hierarchical references (:class:`SRef`, :class:`ARef`).
    """

    def __init__(self) -> None:
        """Create an empty GeomGroup with no elements."""
        self.group = list()

    def __add__(self, other: "GeomGroup") -> "GeomGroup":
        """Combine two geometry groups.

        Parameters
        ----------
        other : GeomGroup
            The GeomGroup you want to add.

        Returns
        -------
        GeomGroup
            The resulting GeomGroup.

        """
        gg = GeomGroup()
        gg.group = self.group + other.group
        return gg

    def add(self, geom: "Poly | SRef | Path | Text") -> None:
        """Add a shape to the group.

        Parameters
        ----------
        geom : Poly | SRef | Path | Text
            The geometry to be added.

        Returns
        -------
        None

        """
        self.group.append(geom)

    def copy(self) -> Self:
        """Make a deep copy of the object.

        Returns
        -------
        Self
            A copy of the group.

        """
        return deepcopy(self)

    def flatten(self, layer_list: Collection[int] | None = None) -> "GeomGroup":
        """Flatten the entire group.

        Turns all SREF and AREF objects in flattened objects. All references to cell
        are removed. A new flattened group is returned and no changes are made to the
        calling object.

        Parameters
        ----------
        layer_list : Collection[int], optional
            A set of layers that should be used when flattening. All layers are
            flattened by default.

        Returns
        -------
        GeomGroup
            A detached copy of the flattened geometry.

        """
        if not layer_list:
            layer_list = self.get_layer_list()

        g = GeomGroup()
        for geom in self.group:
            if isinstance(geom, SRef):
                flatg = geom.group.flatten(layer_list)
                g += geom.place_group(flatg)
            elif geom.layer in layer_list:
                g.add(deepcopy(geom))
        return g

    def get_sref_list(self) -> set[str]:
        """Get a set of cell names referenced in the group.

        This method is recursive.

        Returns
        -------
        sref_list : set[str]
            The complete reference set.

        """
        sref_list = set()
        for geom in self.group:
            if isinstance(geom, SRef):
                sref_list.add(geom.cellname)
                sref_list.update(geom.group.get_sref_list())
        return sref_list

    def get_layer_list(self) -> set[int]:
        """Return a set containing the layers in the group.

        Returns
        -------
        layer_list: set[int]
            The complete layer list.

        """
        layer_list = set()
        for geom in self.group:
            if isinstance(geom, SRef):
                layer_list.update(geom.group.get_layer_list())
            else:
                layer_list.add(geom.layer)

        return layer_list

    def translate(self, dx: float, dy: float) -> Self:
        """Shift the entire geometry by dx and dy.

        Parameters
        ----------
        dx : float
            Shift in x direction.
        dy : float
            Shift in y direction.

        Returns
        -------
        Self
            Reference to the object.

        """
        for geom in self.group:
            geom.translate(dx, dy)
        return self

    def rotate_translate(self, dx: float, dy: float, rot: float) -> Self:
        """Rotate the group around 0,0 and then translate by dx,dy.

        Typically faster than using rotate() followed by translate().

        Parameters
        ----------
        dx : float
            Shift in x direction.
        dy : float
            Shift in y direction.
        rot : float
            Rotation angle in degrees.

        Returns
        -------
        GeomGroup
            Reference to the object.

        """
        for geom in self.group:
            geom.rotate_translate(dx, dy, rot)
        return self

    def rotate(self, x0: float, y0: float, rot: float) -> Self:
        """Rotate the geometry around x0,y0 by a given angle.

        Parameters
        ----------
        x0 : float
            x-coordinate of center of rotation.
        y0 : float
            y-coordinate of center of rotation.
        rot : float
            rotation angle in degrees.

        Returns
        -------
        Self
            Reference to the object.

        """
        for geom in self.group:
            geom.rotate(x0, y0, rot)
        return self

    def scale(self, x0: float, y0: float, scale_x: float, scale_y: float) -> Self:
        """Scale the geometry using x0,y0 as center.

        Parameters
        ----------
        x0 : float
            x-coordinate of center of scaling.
        y0 : float
            y-coordinate of center of scaling.
        scale_x : float
            scaling factor in x direction.
        scale_y : float
            scaling factor in y direction.

        Returns
        -------
        Self
            Reference to the object.

        """
        for geom in self.group:
            geom.scale(x0, y0, scale_x, scale_y)
        return self

    def mirrorX(self, x0: float) -> Self:
        """Mirror the geometry around x-axis.

        Parameters
        ----------
        x0 : float
            x-coordinate of the mirroring axis.

        Returns
        -------
        Self
            Reference to the object.

        """
        for geom in self.group:
            geom.mirrorX(x0)
        return self

    def mirrorY(self, y0: float) -> Self:
        """Mirror the geometry around y-axis.

        Parameters
        ----------
        y0 : float
            y-coordinate of the mirroring axis.

        Returns
        -------
        Self
            Reference to the object.

        """
        for geom in self.group:
            geom.mirrorY(y0)
        return self

    def __entity_count(
        self, recursive: bool = True, layer_wise: bool = False, layer: int = 0
    ) -> dict:
        """Count entities in the group, optionally recursively and per layer.

        Parameters
        ----------
        recursive : bool, optional
            If `True`, include elements inside references recursively.
        layer_wise : bool, optional
            If `True`, count only entities in `layer`.
        layer : int, optional
            Layer number used when `layer_wise` is `True`.

        Returns
        -------
        dict
            Dictionary containing per-type entity counts.

        """
        cnt = dict()
        lfgroup = self.group
        if layer_wise:
            lfgroup = [g for g in self.group if g.layer == layer]
        cnt["NPoly"] = len([g for g in lfgroup if type(g) is Poly])
        cnt["NPath"] = len([g for g in lfgroup if type(g) is Path])
        cnt["NText"] = len([g for g in lfgroup if type(g) is Text])
        cnt["NCircle"] = len([g for g in lfgroup if type(g) is Circle])
        cnt["NEllipse"] = len([g for g in lfgroup if type(g) is Ellipse])
        cnt["NRing"] = len([g for g in lfgroup if type(g) is Ring])
        cnt["NArc"] = len([g for g in lfgroup if type(g) is Arc])
        if not recursive:
            cnt["NSRef"] = len([g for g in self.group if type(g) is SRef])
            cnt["NARef"] = len([g for g in self.group if type(g) is ARef])
        else:
            for g in self.group:
                if isinstance(g, SRef):
                    subcnt = g.group.__entity_count(recursive, layer_wise, layer)
                    if type(g) is ARef:
                        for e in subcnt.keys():
                            subcnt[e] *= g.ncols * g.nrows
                    for name in cnt:
                        cnt[name] += subcnt[name]
        return cnt

    def __str__(self) -> str:
        """Display basic geometry information (size and layers).

        Returns
        -------
        msg : str
            A string with the basic information about the geometry.

        """
        bb = self.bounding_box()
        w = float(round(bb.width * 1e3)) / 1e3
        h = float(round(bb.height * 1e3)) / 1e3
        msg = "GeomGroup (" + str(w) + " x " + str(h) + ")\n"
        msg += "Layers: " + str(self.get_layer_list()) + "\n"
        return msg

    def info(self) -> dict:
        """Generate a dict of useful statistics on the group (element count, size).

        Returns
        -------
        dict
            Contains information on the group geometry.

        """
        stat = dict()
        bb = self.bounding_box()
        layer_list = self.get_layer_list()
        stat["BoundingBox"] = {
            "x": bb.cx(),
            "y": bb.cy(),
            "width": bb.width,
            "height": bb.height,
        }
        stat["LayerList"] = list(layer_list)
        for layer in layer_list:
            cnt = self.__entity_count(True, True, layer)
            cnt = {k: v for k, v in cnt.items() if v != 0}
            stat["Layer" + str(layer)] = cnt
        cnt = self.__entity_count(True)
        cnt = {k: v for k, v in cnt.items() if v != 0}
        stat["TotalCount"] = cnt
        return stat

    def bounding_box(self) -> "Box":
        """Calculate the group bounding box.

        Returns
        -------
        bb : Box
            The box representing the bounding box of the geometry.

        """
        if len(self.group) != 0:
            bb = self.group[0].bounding_box()

        for geom in self.group:
            bb.combine(geom.bounding_box())
        return bb

    def to_boxes(self, layer: int) -> "GeomGroup":
        """Generate a GeomGroup with the bounding boxes of each individual geometry.

        Can be used to create a coarse overlay mask. Acts on a single layer.

        Parameters
        ----------
        layer : int
            The layer to use for bounding boxes.

        Returns
        -------
        GeomGroup
            A geometry group with the bounding boxes of each element in the original
            group.

        """
        bb = GeomGroup()
        for geom in self.group:
            if geom.layer == layer:
                bb += geom.bounding_box().toRect()
        bb.set_layer(layer)
        return bb

    def set_layer(self, layer: int) -> Self:
        """Assign a new layer to all the shapes in the geometry.

        Parameters
        ----------
        layer : int
            The new layer to be assigned.

        Returns
        -------
        Self
            Reference to the object.

        """
        for geom in self.group:
            geom.layer = layer
        return self

    def select_layer(self, layer: int) -> "GeomGroup":
        """Create a new GeomGroup containing only shapes in a given layer.

        Parameters
        ----------
        layer : int
            The selected layer.

        Returns
        -------
        GeomGroup
            A new GeomGroup object with elements of the selected layer.

        """
        g = GeomGroup()
        for geom in self.group:
            if geom.layer == layer:
                g.add(geom)
        return g

    def select_layers(self, layers: list[int]) -> "GeomGroup":
        """Create a new GeomGroup containing only shapes in a list of layers.

        Parameters
        ----------
        layers : list[int]
            The selected layer list.

        Returns
        -------
        GeomGroup
            A new GeomGroup object with elements of the selected layer list.

        """
        g = GeomGroup()
        for geom in self.group:
            if geom.layer in layers:
                g.add(geom)
        return g

    def deselect_layers(self, layers: list[int]) -> "GeomGroup":
        """Create a new GeomGroup containing only shapes that are not in layer list.

        Parameters
        ----------
        layers : list[int]
            A list of layer to deselect.

        Returns
        -------
        GeomGroup
            A new GeomGroup object without elements of the selected layer.

        """
        g = GeomGroup()
        for geom in self.group:
            if geom.layer not in layers:
                g.add(geom)
        return g

    def select(self, query_str: str) -> "GeomGroup":
        """Select shapes based on geometrical properties.

        The output GeomGroup contains only the elements that satisfy the conditions
        expressed in the query string. For example, to select only polygons with area
        smaller than 1, do `sel = geom.select("A<=1")`.

        The geometry is flattened prior to selection if sub-references are encountered.
        The following properties can be used

            * "A": Area
            * "P": Perimeter
            * "W": Bounding box width
            * "H": Bounding box height
            * "L": Layer
            * "T": type (can be Poly, Path, Text, Circle, Ellipse, Ring, Arc)
            * "x": centroid x coordinate
            * "y": centroid y coordinate
            * "llx": lower-left x coordinate of the bounding box
            * "lly": lower-left y coordinate of the bounding box
            * "urx": upper-right x coordinate of the bounding box
            * "ury": upper-right y coordinate of the bounding box

        Any Python logical expression can be used including mathematical operations
        between variables, for example the shape index lower than 0.2 can be queried as
        follows `sel = geom.select("12.566*A/P**2<=0.2")`.

        Boolean operation should be expressed using parenthesis and bitwise operator
        &,^,|,! for example `sel = geom.select("(x<1) & (y>5)")`.


        Parameters
        ----------
        query_str : str
            A string defining the selection condition.

        Raises
        ------
        NameError
            If variable names are not recognized.

        Returns
        -------
        GeomGroup
            A geometry group containing the elments that satisfy the criteria.

        """
        allowed_names = {
            "A": "Polygon area",
            "P": "Polygon perimeter",
            "W": "Bounding box width",
            "H": "Bounding box height",
            "L": "Layer",
            "T": "Type",
            "x": "X position, center or reference pos",
            "y": "Y position, center or reference pos",
            "llx": "lower left x position of the bb",
            "lly": "lower left y position of the bb",
            "urx": "upper right x position of the bb",
            "ury": "upper right y position of the bb",
        }
        code = compile(query_str, "<string>", "eval")
        sflat = self
        if self.get_sref_list():
            sflat = self.flatten()
        # Pre-allocate Bounding boxes
        bbs = [g.bounding_box() for g in sflat.group]
        for name in code.co_names:
            if name not in allowed_names:
                msg = f"Use of expression {name} not allowed"
                raise NameError(msg)

            # Prepare the local variable dictionary

            if name == "A":  # Prepare area array
                allowed_names[name] = np.array([g.area() for g in sflat.group])
            if name == "P":  # Prepare area array
                allowed_names[name] = np.array([g.perimeter() for g in sflat.group])
            if name == "L":  # Prepare layer array
                allowed_names[name] = np.array([g.layer for g in sflat.group])
            if name == "W":  # Prepare width array
                allowed_names[name] = np.array([b.width for b in bbs])
            if name == "H":  # Prepare height array
                allowed_names[name] = np.array([b.height for b in bbs])
            if name == "x" or name == "y":  # Prepare centroid array
                allowed_names["x"] = np.array([g.centroid()[0] for g in sflat.group])
                allowed_names["y"] = np.array([g.centroid()[1] for g in sflat.group])
            if name == "llx":  # Prepare LL array
                allowed_names["llx"] = np.array([b.llx for b in bbs])
            if name == "lly":  # Prepare LL array
                allowed_names["lly"] = np.array([b.lly for b in bbs])
            if name == "urx":  # Prepare UR array
                allowed_names["urx"] = np.array([b.urx() for b in bbs])
            if name == "ury":  # Prepare UR array
                allowed_names["ury"] = np.array([b.ury() for b in bbs])
            if name == "T":  # Prepare type array
                allowed_names[name] = np.array(
                    [str(g.__class__.__name__) for g in sflat.group]
                )

        # Now execute
        g = GeomGroup()
        sel = eval(code, {"__builtins__": {}}, allowed_names)
        g.group[:] = [sflat.group[i] for i, val in enumerate(sel) if val]
        return g

    def find_matching_patterns(
        self, pattern: "GeomGroup", layer: int
    ) -> list[list[float]]:
        """Find the position of a given repeating pattern in the geometry.

        User provides a pattern as a geom group. The pattern should not be
        disjoint (i.e. after boolean union, it should contain one element only)
        The code search for the identical pattern in the whole group for a given
        layer. Once found, an array of x,y coordinates is returned, corresponding
        to coordinates of the pattern.
        Note, the code performs boolean union on the input pattern as well as
        on the entire geometry.

        Parameters
        ----------
        pattern : GeomGroup
            The pattern to be searched. Must be a single connected polygon.
        layer : int
            The layer to perform the search on.

        Returns
        -------
        list[list[float]]
            A list of coordinate pairs, corresponding to the location of the pattern.

        """
        psearch = pattern.copy()
        psearch.all_to_poly()
        psearch.boolean_union(layer)
        if len(psearch.group) != 1:
            msg = "It is only possible to search for a single polygon shape."
            raise ValueError(msg)
        plook = self.copy().boolean_union(layer)
        b1 = psearch.bounding_box()
        psearch.translate(-b1.cx(), -b1.cy())
        g2 = psearch.group[0]
        res = []
        for g in plook.group:
            if isinstance(g, Poly):
                bb = g.bounding_box()
                g.translate(-bb.cx(), -bb.cy())
                if g.identical_to(g2):
                    res += [[bb.cx(), bb.cy()]]
        return res

    def get_area(self) -> float:
        """Calculate the total area of the group.

        Returns
        -------
        float
            The total area of the group.

        """
        area = 0
        for i in range(len(self.group)):
            if isinstance(self.group[i], SRef):
                area += self.group[i].group.get_area()
            else:
                area += self.group[i].area()
        return float(round(area * 1e6)) / 1e6

    def path_to_poly(self) -> None:
        """Convert all path objects in the current group to polygons.

        Returns
        -------
        None

        """
        paths = GeomGroup()
        for i in range(len(self.group)):
            if isinstance(self.group[i], Path):
                paths += self.group[i].to_polygon()

        self.group[:] = [g for g in self.group if not isinstance(g, Path)]
        self.group = self.group + paths.group

    def text_to_poly(self) -> None:
        """Convert all text objects in the current group to polygons.

        Returns
        -------
        None

        """
        polys = GeomGroup()
        for i in range(len(self.group)):
            if isinstance(self.group[i], Text):
                polys += self.group[i].to_polygon()

        self.group[:] = [g for g in self.group if not isinstance(g, Text)]
        self.group = self.group + polys.group

    def all_to_poly(
        self, Npts_circ: int = 12, Npts_arc: int = 32, split_arc: bool = False
    ) -> None:
        """Convert all elements except for SRef and Aref to polygons.

        Returns
        -------
        None

        """
        polys = GeomGroup()
        for i in range(len(self.group)):
            g = self.group[i]
            if isinstance(g, Poly):
                polys += self.group[i].to_polygon()
            elif isinstance(g, Text):
                polys += self.group[i].to_polygon()
            elif isinstance(g, Path):
                polys += self.group[i].to_polygon()
            elif isinstance(g, Circle):
                polys += self.group[i].to_polygon(Npts_circ)
            elif isinstance(g, Ellipse):
                polys += self.group[i].to_polygon(Npts_arc)
            elif isinstance(g, Ring):
                polys += self.group[i].to_polygon(Npts_arc)
            elif isinstance(g, Arc):
                polys += self.group[i].to_polygon(Npts_arc, split_arc)

        self.group[:] = [g for g in self.group if isinstance(g, SRef)]
        self.group = self.group + polys.group

    def poly_to_circle(
        self, thresh: float = 0.95, vcount: int = 10, include_refs: bool = True
    ) -> None:
        """Conditionally convert all polygons to circle.

        The polygons need to meet a circularity threshold and a vertex count larger than
        the vcount parameter to be converted.

        Parameters
        ----------
        thresh : float, optional
            Circularity threshold (1=perfect circle), by default 0.95.
        vcount : int, optional
            Minimum number of vertices to perform the conversion, by default 10.
        include_refs : bool, optional
            Perform recursive conversion to SRefs and ARefs, by default True.

        Returns
        -------
        None

        """
        polys = GeomGroup()
        for i in range(len(self.group)):
            if isinstance(self.group[i], Poly):
                convp = self.group[i].to_circle(thresh, vcount)
                if not convp.group:
                    polys.group += [self.group[i]]
                else:
                    polys += convp
                continue
            elif isinstance(self.group[i], SRef):
                if include_refs:
                    self.group[i].group.poly_to_circle(thresh, vcount)
                continue
            # None of the above, just keep
            polys.group += [self.group[i]]

        self.group[:] = [g for g in self.group if isinstance(g, SRef)]
        self.group = self.group + polys.group

    def in_polygons(self, x: float, y: float) -> bool:
        """Check if a given coordinate is inside the GeomGroup polygons.

        Parameters
        ----------
        x : float
            x coordinate.
        y : float
            y coordinate.

        Returns
        -------
        bool
            True if coordinate is inside the polygon.

        """
        for i in range(len(self.group)):
            if isinstance(self.group[i], SRef):
                if self.group[i].point_inside(x, y):
                    return True
        return False

    def keep_refs_only(self) -> None:
        """Keep only Sref and Aref objects in the group.

        Returns
        -------
        None

        """
        self.group[:] = [g for g in self.group if isinstance(g, SRef)]

    def __get_boopy__(self, layer: int) -> "boopy.PolyGroup":
        """Build a boopy polygon group from polygons on a given layer.

        Parameters
        ----------
        layer : int
            Layer to extract.

        Returns
        -------
        boopy.PolyGroup
            Polygon group containing integer polygon data for `layer`.

        """
        pg0 = boopy.PolyGroup()
        for i in range(len(self.group)):
            if isinstance(self.group[i], Poly) and self.group[i].layer == layer:
                pdata = self.group[i].int_data()
                pg0.addPolyData(pdata)
        return pg0

    def __set_boopy__(self, pg0: boopy.PolyGroup, layer: int) -> None:
        """Append polygons from a `boopy.PolyGroup` object back into the geometry.

        Parameters
        ----------
        pg0 : boopy.PolyGroup
            Source polygon group holding integer coordinates.
        layer : int
            Layer assigned to imported polygons.

        Returns
        -------
        None

        """
        npoly = pg0.getPolyCount()
        polys = GeomGroup()
        for i in range(npoly):
            poly = Poly([], [], layer)
            pdata = np.array(pg0.getPoly(i))
            pdata = pdata / 1000.0
            poly.set_data(pdata)
            polys.add(poly)
        self.group = self.group + polys.group

    def boolean_union(self, layer: int) -> Self:
        """Perform a boolean union (OR) operation of all polygons matching a layer.

        This operation is performed in-place.

        All other elements (circles, paths, texts) are ignored unless they have been
        already converted to polygons.

        Parameters
        ----------
        layer : int
            The layer in which the union should be performed.

        Returns
        -------
        Self
            Reference to the object.

        """
        # Get the boost python data
        pg0 = self.__get_boopy__(layer)
        # Remove the old polygons
        self.group[:] = [
            g for g in self.group if not (isinstance(g, Poly) and g.layer == layer)
        ]
        pg0.assign()
        # Put back the boost python data
        self.__set_boopy__(pg0, layer)
        return self

    def boolean_difference(
        self, targetB: "GeomGroup", layerA: int, layerB: int
    ) -> Self:
        """Perform a boolean difference operation between polygons.

        The operation is done in-place in the calling group matching `layerA` and the
        polygons in group `targetB`, matching `layerB`.

        All other elements (circles, paths, texts) are ignored unless they have been
        already converted to polygons.

        Parameters
        ----------
        targetB: GeomGroup
            The geometry to be subtracted.
        layerA : int
            The layer from which subtraction should be performed.
        layerB: int
            The layer to be subtracted.

        Returns
        -------
        Self
            Reference to the object.

        """
        # Get the boost python data
        polygroup_a = self.__get_boopy__(layerA)
        polygroup_b = targetB.__get_boopy__(layerB)
        # Difference
        polygroup_a.difference(polygroup_b)
        # Remove the old polygons
        self.group[:] = [
            g for g in self.group if not (isinstance(g, Poly) and g.layer == layerA)
        ]
        # Put back the boost python data (merge is automatically done)
        self.__set_boopy__(polygroup_a, layerA)
        return self

    def boolean_xor(self, targetB: "GeomGroup", layerA: int, layerB: int) -> Self:
        """Perform a boolean exclusive-OR (XOR) operation between polygons.

        The operation is done in-place in the calling group matching `layerA` and the
        polygons in group `targetB`, matching `layerB`.

        All other elements (circles, paths, texts) are ignored unless they have been
        already converted to polygons.

        Parameters
        ----------
        targetB: GeomGroup
            The geometry to be XORed.
        layerA : int
            The layer from which XOR operation should be performed.
        layerB: int
            The layer to be XORed.

        Returns
        -------
        GeomGroup
            Reference to the object.

        """
        # Get the boost python data
        polygroup_a = self.__get_boopy__(layerA)
        polygroup_b = targetB.__get_boopy__(layerB)
        # Difference
        polygroup_a.exor(polygroup_b)
        # Remove the old polygons
        self.group[:] = [
            g for g in self.group if not (isinstance(g, Poly) and g.layer == layerA)
        ]
        # Put back the boost python data (merge is automatically done)
        self.__set_boopy__(polygroup_a, layerA)
        return self

    def boolean_intersection(
        self, targetB: "GeomGroup", layerA: int, layerB: int
    ) -> Self:
        """Perform a boolean intersection (AND) operation between polygons.

        The operation is done in-place in the calling group matching `layerA` and the
        polygons in group `targetB`, matching `layerB`.

        All other elements (circles, paths, texts) are ignored unless they have been
        already converted to polygons.

        Parameters
        ----------
        targetB: GeomGroup
            The geometry to be intersected.
        layerA : int
            The layer from which subtraction should be performed.
        layerB: int
            The layer to be subtracted.

        Returns
        -------
        Self
            Reference to the object.

        """
        # Get the boost python data
        polygroup_a = self.__get_boopy__(layerA)
        polygroup_b = targetB.__get_boopy__(layerB)
        # Difference
        polygroup_a.intersection(polygroup_b)
        # Remove the old polygons
        self.group[:] = [
            g for g in self.group if not (isinstance(g, Poly) and g.layer == layerA)
        ]
        # Put back the boost python data (merge is automatically done)
        self.__set_boopy__(polygroup_a, layerA)
        return self

    def poly_resize(
        self,
        offset: float,
        layer: int,
        corner_fill_arc: bool = False,
        num_circle_segments: int = 0,
    ) -> Self:
        """Offset the polygon by a certain distance.

        This operation is performed in-place.

        Acts only on polygons and on a single layer.

        Parameters
        ----------
        offset : float
            Positive or negative offset (resizing) amount.
        layer : int
            The layer to be resized.
        corner_fill_arc : bool, optional
            Rounds the convex corners, by default False.
        num_circle_segments : int, optional
            If corner_fill_arc is True, the number of segments to be used for arc
            filling, by default 0.

        Returns
        -------
        Self
            Reference to the object.

        """
        polygroup = self.__get_boopy__(layer)
        polygroup.resize(round(offset * 1000), corner_fill_arc, num_circle_segments)
        self.group[:] = [
            g for g in self.group if not (isinstance(g, Poly) and g.layer == layer)
        ]
        self.__set_boopy__(polygroup, layer)
        return self

    def poly_anisotropic_resize(self, angles: list, deltas: list, layer: int) -> Self:
        """Perform an anisotropic offset of the polygons in a given layer.

        Requires an offset array in deltas matching the angle of expansion. Angles
        should cover -90 to 90 degrees.

        Parameters
        ----------
        angles : list
            list of angles in degrees.
        deltas : list
            offset at a given angle.
        layer : int
            the layer to be resized.

        Returns
        -------
        Self
            Reference to the object.

        """
        for i in range(len(self.group)):
            if isinstance(self.group[i], Poly) and self.group[i].layer == layer:
                self.group[i].anisotropic_resize(angles, deltas)
        return self

    def poly_outlining(
        self,
        offset: float,
        layer: int,
        distance: float = 0,
        corner_fill_arc: bool = False,
        num_circle_segments: int = 0,
    ) -> Self:
        """Calculate the polygon outline of the polygons in a given layer.

        This operation is performed in-place.

        Also works on circles (ignores the other elements, which must be converted to
        poly first)

        Parameters
        ----------
        offset : float
            Positive or negative offset (resizing) amount.
        layer : int
            The layer to be resized.
        distance: float, optional
            How far should the outline be displaced from the polygon edge.
            Negative values mean inward distance, by default 0.
        corner_fill_arc : bool, optional
            Rounds the convex corners, by default False.
        num_circle_segments : int, optional
            If corner_fill_arc is True, the number of segments to be used for arc
            filling, by default 0.

        Returns
        -------
        Self
            Reference to the object.

        """
        polygroup = self.__get_boopy__(layer)
        original_polygroup = self.__get_boopy__(layer)
        if distance != 0:
            polygroup.resize(
                round((offset + distance) * 1000), corner_fill_arc, num_circle_segments
            )
            original_polygroup.resize(
                round(distance * 1000), corner_fill_arc, num_circle_segments
            )
        else:
            polygroup.resize(round(offset * 1000), corner_fill_arc, num_circle_segments)
        self.group[:] = [
            g for g in self.group if not (isinstance(g, Poly) and g.layer == layer)
        ]
        if offset > 0:
            polygroup.difference(original_polygroup)
            self.__set_boopy__(polygroup, layer)
        else:
            original_polygroup.difference(polygroup)
            self.__set_boopy__(original_polygroup, layer)

        # Circles
        for i in range(len(self.group)):
            if isinstance(self.group[i], Circle):
                g = self.group[i]
                self.group[i] = Arc(
                    g.x0,
                    g.y0,
                    g.r + offset / 2 + distance,
                    g.r + offset / 2 + distance,
                    g.layer,
                    0,
                    offset,
                    0,
                    360,
                )

        return self

    def invert(self, layer: int, offset: float = 0) -> Self:
        """Perform a boolean inverse operation (NOT) on the polygons in a layer.

        This operation is performed in-place.

        The result is the negative of the mask on the bounding box polygon.
        An offset can be specified to bloat/shrink the bounding box before inversion.

        Parameters
        ----------
        layer : int
            The layer to be inverted.
        offset : float, optional
            Resizing amount (positive or negative) of the bounding box before inversion,
            by default 0.

        Returns
        -------
        Self
            Reference to the inverted object.

        """
        polygroup = self.__get_boopy__(layer)
        sel = self.select_layer(layer)
        if len(sel.group) == 0:
            return self
        bb = sel.bounding_box().toRect()
        bb.set_layer(layer)
        if offset != 0:
            bb.poly_resize(offset, layer)
        bb_polygroup = bb.__get_boopy__(layer)
        bb_polygroup.difference(polygroup)
        self.group[:] = [
            g for g in self.group if not (isinstance(g, Poly) and g.layer == layer)
        ]
        self.__set_boopy__(bb_polygroup, layer)

        return self

    def trapezoids(self, layer: int) -> Self:
        """Convert and fractures all polygons in a set of trapezoids.

        This operation is performed in-place.

        Parameters
        ----------
        layer : int
            The layer to be fractured.

        Returns
        -------
        Self
            Reference to the object.

        """
        polygroup = self.__get_boopy__(layer)
        polygroup.trapezoids()
        self.group[:] = [
            g for g in self.group if not (isinstance(g, Poly) and g.layer == layer)
        ]
        self.__set_boopy__(polygroup, layer)
        return self

    def poly_filter(self, keep_str: str) -> int:
        """Perform filtering of vertices based on a condition string.

        This operation is performed in-place.

        Possible conditions are expressed based on the following variables:

            * 'A': "Three-point polygon unsigned area",
            * 'As': "Three-point polygon signed area",
            * 'P': "three point perimeter",
            * 'S': "Shape index of the triangle (4*pi*A/(P**2))",
            * 'x': "x-coordinate of query point",
            * 'y': "y-coordinate of query point",
            * 'xm': "x-coordinate of previous point",
            * 'ym': "y-coordinate of previous point",
            * 'xp': "x-coordinate of next point",
            * 'yp': "y-coordinate of next point",
            * 'dm': "distance between query point and previous point",
            * 'dp': "distance between query point and next point",
            * 'd0': "distance between previous and next point"

        These variables are defined over 3 consecutive points.
        If the condition is met, the middle point is kept, otherwise
        it is discarded and the next 3 points are tested. All layers are used

        Parameters
        ----------
        keep_str : str
            String that contains the condition to keep a vertex or not.

        Raises
        ------
        NameError
            If a name that is not allowed is used.

        Returns
        -------
        int
            The number of vertices discarded.

        """
        ndisc = 0
        for g in self.group:
            if isinstance(g, Poly):
                ndisc += g.three_point_filter(keep_str)
        return ndisc


class Dot:
    """Point helper class used for geometric transformations."""

    def __init__(self, x: float, y: float) -> None:
        """Create a point.

        Parameters
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.

        """
        self.x = x
        self.y = y

    def translate(self, dx: float, dy: float) -> None:
        """Translate the point.

        Parameters
        ----------
        dx : float
            Translation along x.
        dy : float
            Translation along y.

        Returns
        -------
        None

        """
        self.x += dx
        self.y += dy

    def rotate(self, x0: float, y0: float, rot: float) -> None:
        """Rotate the point around a center.

        Parameters
        ----------
        x0 : float
            Rotation center x coordinate.
        y0 : float
            Rotation center y coordinate.
        rot : float
            Rotation angle in degrees.

        Returns
        -------
        None

        """
        xc = self.x - x0
        yc = self.y - y0
        cost = math.cos(rot / 180 * math.pi)
        sint = math.sin(rot / 180 * math.pi)
        self.x = cost * xc - sint * yc + x0
        self.y = sint * xc + cost * yc + y0

    def rotate_translate(self, x0: float, y0: float, rot: float) -> None:
        """Rotate the point around origin and then translate.

        Parameters
        ----------
        x0 : float
            Translation along x after rotation.
        y0 : float
            Translation along y after rotation.
        rot : float
            Rotation angle in degrees around origin.

        Returns
        -------
        None

        """
        cost = math.cos(rot / 180 * math.pi)
        sint = math.sin(rot / 180 * math.pi)
        x = self.x
        y = self.y
        self.x = cost * x - sint * y + x0
        self.y = sint * x + cost * y + y0

    def scale(self, x0: float, y0: float, scale_x: float, scale_y: float) -> None:
        """Scale the point around a center.

        This operation translates the point as if it was part of a scaled geometry.

        Parameters
        ----------
        x0 : float
            Scaling center x coordinate.
        y0 : float
            Scaling center y coordinate.
        scale_x : float
            Scale factor along x.
        scale_y : float
            Scale factor along y.

        Returns
        -------
        None

        """
        self.x = (self.x - x0) * scale_x + x0
        self.y = (self.y - y0) * scale_y + y0

    def mirrorX(self, x0: float) -> None:
        """Mirror the point with respect to a vertical axis.

        Parameters
        ----------
        x0 : float
            X coordinate of the mirror axis.

        Returns
        -------
        None

        """
        self.x = 2 * x0 - self.x

    def mirrorY(self, y0: float) -> None:
        """Mirror the point with respect to a horizontal axis.

        Parameters
        ----------
        y0 : float
            Y coordinate of the mirror axis.

        Returns
        -------
        None

        """
        self.y = 2 * y0 - self.y


class Box:
    """Axis-aligned bounding box utility class.

    This class is used for bounding box calculations and transformations. It is not a
    geometry element.
    """

    def __init__(self, llx: float, lly: float, width: float, height: float) -> None:
        """Initialize a box object.

        Parameters
        ----------
        llx : float
            lower-left x-coordinate.
        lly : float
            lower-left y-coordinate.
        width : float
            width of the box.
        height : float
            height of the box.

        """
        self.llx = llx
        self.lly = lly
        self.width = width
        self.height = height

    def cx(self) -> float:
        """Get the x-coordinate of the box center.

        Returns
        -------
        float
            x-coordinate of the box center.

        """
        return self.llx + self.width / 2

    def cy(self) -> float:
        """Get the y-coordinate of the box center.

        Returns
        -------
        float
            y-coordinate of the box center.

        """
        return self.lly + self.height / 2

    def urx(self) -> float:
        """Get the x-coordinate of the upper-right corner.

        Returns
        -------
        float
            x-coordinate of the upper-right corner.

        """
        return self.llx + self.width

    def ury(self) -> float:
        """Get the y-coordinate of the upper-right corner.

        Returns
        -------
        float
            y-coordinate of the upper-right corner.

        """
        return self.lly + self.height

    def combine(self, other: "Box") -> None:
        """Extend the box to fit another box.

        Parameters
        ----------
        other : Box
            The other box that should be combined.

        Returns
        -------
        None

        """
        tmp_urx = self.urx()
        tmp_ury = self.ury()
        if other.llx < self.llx:
            self.llx = other.llx
        if other.lly < self.lly:
            self.lly = other.lly
        if other.urx() > tmp_urx:
            tmp_urx = other.urx()
        if other.ury() > tmp_ury:
            tmp_ury = other.ury()

        self.width = tmp_urx - self.llx
        self.height = tmp_ury - self.lly

    def toPoly(self) -> "Poly":
        """Convert the box to a `Poly` object that can be added to geometry groups.

        The resulting polygon will be initialized in layer 0.

        Returns
        -------
        Poly
            The poly representing the box.

        """
        return Poly(
            [self.llx, self.urx(), self.urx(), self.llx],
            [self.lly, self.lly, self.ury(), self.ury()],
            0,
        )

    def toRect(self) -> "GeomGroup":
        """Create a group with a rectangle for drawing.

        Returns
        -------
        GeomGroup
            The group containing the bounding box rectangle.

        """
        g = GeomGroup()
        g.add(self.toPoly())
        return g

    def get_numkey_point(self, numkey: int) -> tuple[float, float]:
        """Get a tuple with coordinates of the point matching a numerical keypad.

        E.g. 5 is the center, 1 is the lower left corner, etc...

        Parameters
        ----------
        numkey : int
            A number between 1 and 9 corresponding to the box point.

        Returns
        -------
        tuple[float, float]
            The coordinates corresponding to the keypad.

        """
        if numkey < 1 or numkey > 9:
            msg = f"numkey should be between 1 and 9. Provided value is {numkey}"
            raise ValueError(msg)
        numkey = int(numkey)
        xoff = -((numkey - 1) % 3 - 1)
        yoff = math.floor((9 - numkey) / 3) - 1
        return self.cx() - xoff * self.width / 2, self.cy() - yoff * self.height / 2


class Poly:
    """Closed polygon represented by interleaved coordinate data."""

    def __init__(
        self, xpts: Sequence[float], ypts: Sequence[float], layer: int
    ) -> None:
        """Initialize a polygon from x/y coordinates.

        Parameters
        ----------
        xpts : Sequence[float]
            Polygon x coordinates.
        ypts : Sequence[float]
            Polygon y coordinates.
        layer : int
            Layer number.

        """
        self.layer = layer
        self.set_points(xpts, ypts)

    def set_points(self, xpts: Sequence[float], ypts: Sequence[float]) -> None:
        """Set polygon points from x and y coordinate arrays.

        Parameters
        ----------
        xpts : Sequence[float]
            Polygon x coordinates.
        ypts : Sequence[float]
            Polygon y coordinates.

        Returns
        -------
        None

        """
        # Note: only for polygon class, we store the points in GDS format,
        # already scaled to nanometers and as X0,Y0,X1,Y1,X2,Y2...
        # rdata = np.round_((np.array([xpts,ypts])*1000)).astype(int)
        rdata = np.array([xpts, ypts], dtype="float64")
        self.data = np.transpose(rdata).reshape(-1)
        self.data = np.append(self.data, self.data[0:2])
        self.Npts = math.floor(self.data.size / 2)

    def set_data(self, data: np.ndarray) -> None:
        """Set polygon coordinates from interleaved coordinate data.

        Parameters
        ----------
        data : np.ndarray
            Flat array `[x0, y0, x1, y1, ...]`.

        Returns
        -------
        None

        """
        self.data = data
        self.Npts = math.floor(self.data.size / 2)

    def int_data(self) -> np.ndarray:
        """Get polygon data scaled to integer nanometer units.

        Returns
        -------
        np.ndarray
            Integer coordinate array.

        """
        return np.round(self.data * 1000).astype(int)

    def set_int_data(self, idata: np.ndarray) -> None:
        """Set polygon data from integer nanometer units.

        Parameters
        ----------
        idata : np.ndarray
            Integer coordinate array.

        Returns
        -------
        None

        """
        self.data = idata.astype("float64") / 1000
        self.Npts = self.data.size / 2

    def translate(self, dx: float, dy: float) -> None:
        """Translate the polygon.

        Parameters
        ----------
        dx : float
            Translation along x.
        dy : float
            Translation along y.

        Returns
        -------
        None

        """
        self.data[0::2] += dx
        self.data[1::2] += dy

    def rotate_translate(self, x0: float, y0: float, rot: float) -> None:
        """Rotate the polygon around origin and then translate.

        Parameters
        ----------
        x0 : float
            Translation along x after rotation.
        y0 : float
            Translation along y after rotation.
        rot : float
            Rotation angle in degrees around origin.

        Returns
        -------
        None

        """
        cost = math.cos(rot / 180 * math.pi)
        sint = math.sin(rot / 180 * math.pi)
        x = np.copy(self.data[0::2])
        y = np.copy(self.data[1::2])
        self.data[0::2] = cost * (x) - sint * (y) + x0
        self.data[1::2] = sint * (x) + cost * (y) + y0

    def rotate(self, x0: float, y0: float, rot: float) -> None:
        """Rotate the polygon around a center.

        Parameters
        ----------
        x0 : float
            Rotation center x coordinate.
        y0 : float
            Rotation center y coordinate.
        rot : float
            Rotation angle in degrees.

        Returns
        -------
        None

        """
        cost = math.cos(rot / 180 * math.pi)
        sint = math.sin(rot / 180 * math.pi)
        x = np.copy(self.data[0::2])
        y = np.copy(self.data[1::2])
        self.data[0::2] = cost * (x - x0) - sint * (y - y0) + x0
        self.data[1::2] = sint * (x - x0) + cost * (y - y0) + y0

    def scale(self, x0: float, y0: float, scale_x: float, scale_y: float) -> None:
        """Scale polygon coordinates around a center.

        Parameters
        ----------
        x0 : float
            Scaling center x coordinate.
        y0 : float
            Scaling center y coordinate.
        scale_x : float
            Scale factor along x.
        scale_y : float
            Scale factor along y.

        Returns
        -------
        None

        """
        x = self.data[0::2]
        y = self.data[1::2]
        self.data[0::2] = scale_x * (x - x0) + x0
        self.data[1::2] = scale_y * (y - y0) + y0

    def mirrorX(self, x0: float) -> None:
        """Mirror polygon vertices with respect to a vertical axis.

        Parameters
        ----------
        x0 : float
            X coordinate of mirror axis.

        Returns
        -------
        None

        """
        self.data[0::2] = 2 * x0 - self.data[0::2]

    def mirrorY(self, y0: float) -> None:
        """Mirror polygon vertices with respect to a horizontal axis.

        Parameters
        ----------
        y0 : float
            Y coordinate of mirror axis.

        Returns
        -------
        None

        """
        self.data[1::2] = 2 * y0 - self.data[1::2]

    def bounding_box(self) -> Box:
        """Compute the polygon bounding box.

        Returns
        -------
        Box
            Axis-aligned bounding box.

        """
        llx = min(self.data[0::2])
        urx = max(self.data[0::2])
        lly = min(self.data[1::2])
        ury = max(self.data[1::2])
        return Box(llx, lly, urx - llx, ury - lly)

    def area(self) -> float:
        """Compute polygon area.

        Returns
        -------
        float
            Absolute polygon area.

        """
        area = 0.0
        x = self.data[0::2]
        y = self.data[1::2]
        n = int(len(x))
        j = n - 1
        for i in range(n):
            area += x[j] * y[i] - x[i] * y[j]
            j = i
        return float(round(1e6 * abs(area / 2.0))) / 1.0e6

    def centroid(self) -> tuple[float, float]:
        """Compute polygon centroid.

        Returns
        -------
        tuple[float, float]
            Centroid coordinates `(x, y)`.

        """
        cx = 0
        cy = 0
        area = 0
        x = self.data[0::2]
        y = self.data[1::2]
        n = int(len(x))
        j = n - 1
        for i in range(n):
            shl = x[j] * y[i] - x[i] * y[j]
            area += shl
            cx += (x[j] + x[i]) * shl
            cy += (y[j] + y[i]) * shl
            j = i
        cx /= 3 * area
        cy /= 3 * area
        return cx, cy

    def perimeter(self) -> float:
        """Compute polygon perimeter.

        Returns
        -------
        float
            Polygon perimeter length.

        """
        p = 0
        x = self.data[0::2]
        y = self.data[1::2]
        n = int(len(x))
        j = n - 1
        for i in range(n):
            p += np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            j = i
        return p

    def three_point_filter(self, keep_str: str) -> int:
        """Perform filtering of vertices based on a condition string.

        Possible conditions are expressed based on the following variables:

            * 'A': "Three-point polygon unsigned area",
            * 'As': "Three-point polygon signed area",
            * 'P': "three point perimeter",
            * 'S': "Shape index of the triangle (4*pi*A/(P**2))",
            * 'x': "x-coordinate of query point",
            * 'y': "y-coordinate of query point",
            * 'xm': "x-coordinate of previous point",
            * 'ym': "y-coordinate of previous point",
            * 'xp': "x-coordinate of next point",
            * 'yp': "y-coordinate of next point",
            * 'dm': "distance between query point and previous point",
            * 'dp': "distance between query point and next point",
            * 'd0': "distance between previous and next point"

        These variables are defined over 3 consecutive points.
        If the condition is met, the middle point is kept, otherwise
        it is discarded and the next 3 points are tested.

        Parameters
        ----------
        keep_str : str
            String that contains the condition to keep a vertex or not.

        Raises
        ------
        NameError
            If a name that is not allowed is used.

        Returns
        -------
        int
            The number of vertices discarded.

        """
        allowed_names = {
            "A": "Three-point polygon unsigned area",
            "As": "Three-point polygon signed area",
            "P": "three point perimeter",
            "S": "Shape index of the triangle (4*pi*A/(P**2))",
            "x": "x-coordinate of query point",
            "y": "y-coordinate of query point",
            "xm": "x-coordinate of previous point",
            "ym": "y-coordinate of previous point",
            "xp": "x-coordinate of next point",
            "yp": "y-coordinate of next point",
            "dm": "distance between query point and previous point",
            "dp": "distance between query point and next point",
            "d0": "distance between previous and next point",
        }
        code = compile(keep_str, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of expression {name} not allowed")

        #        g.group[:] = [sflat.group[i] for i,val in enumerate(sel) if val]
        #       return g
        x = self.data[0::2]
        y = self.data[1::2]
        xf = []
        yf = []
        n = int(self.Npts)
        ndisc = 0
        j = n - 1
        k = n - 2
        for i in range(n):
            attr = x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j])
            d1 = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            d2 = np.sqrt((x[j] - x[k]) ** 2 + (y[j] - y[k]) ** 2)
            d3 = np.sqrt((x[i] - x[k]) ** 2 + (y[i] - y[k]) ** 2)
            allowed_names["A"] = abs(attr) / 2
            allowed_names["As"] = attr / 2
            allowed_names["P"] = d1 + d2 + d3
            allowed_names["S"] = abs(attr) * 2 * np.pi / allowed_names["P"] ** 2
            allowed_names["x"] = x[j]
            allowed_names["y"] = y[j]
            allowed_names["xm"] = x[k]
            allowed_names["ym"] = y[k]
            allowed_names["xp"] = x[i]
            allowed_names["yp"] = y[i]
            allowed_names["dm"] = d2
            allowed_names["dp"] = d1
            allowed_names["d0"] = d3
            sel = eval(code, {"__builtins__": {}}, allowed_names)
            if sel:
                xf += [x[j]]
                yf += [y[j]]
                k = j
                j = i
            else:
                ndisc += 1
                j = i
        self.set_points(xf, yf)
        return ndisc

    def to_polygon(self) -> GeomGroup:
        """Return the polygon as a single-element geometry group.

        Returns
        -------
        GeomGroup
            Group containing this polygon.

        """
        g = GeomGroup()
        g.add(self)
        return g

    def to_circle(self, thresh: float = 0.95, vcount: int = 10) -> GeomGroup:
        """Approximate polygon as a circle when sufficiently circular.

        Parameters
        ----------
        thresh : float, optional
            Minimum circularity threshold. Default is 0.95.
        vcount : int, optional
            Minimum number of vertices required to test conversion. Default is 10.

        Returns
        -------
        GeomGroup
            Empty group if conversion fails, otherwise a group with one circle.

        """
        g = GeomGroup()
        if self.Npts < vcount:
            return g
        # Attempt to perform a conversion, return empty group if failed
        # Check circularity
        xpts = self.data[0::2]
        ypts = self.data[1::2]
        cx, cy = self.centroid()
        rpts = np.sqrt((xpts - cx) ** 2 + (ypts - cy) ** 2)
        r_avg = rpts.mean()
        rmin = np.min(rpts)
        rmax = np.max(rpts)
        circularity = 1 - (rmax - rmin) / r_avg
        if circularity >= thresh:
            g.add(Circle(cx, cy, r_avg, self.layer))
        return g

    def identical_to(self, p2: "Poly") -> bool:
        """Check whether two polygons are identical.

        Parameters
        ----------
        p2 : Poly
            Polygon to compare with.

        Returns
        -------
        bool
            `True` if the polygons contain the same ordered vertices modulo starting
            index.

        """
        x = self.data[0::2]
        y = self.data[1::2]
        x2 = p2.data[0::2]
        y2 = p2.data[1::2]
        x3 = np.append(x, x)
        y3 = np.append(y, y)
        for e in range(0, len(x)):
            k = 0
            for w in range(e, e + len(x)):
                if x2[k] == x3[w] and y2[k] == y3[w]:
                    k += 1
                else:
                    break
            # if all n elements are same circularly
            if k == len(x):
                return True
        return False

    def point_inside(self, x: float, y: float) -> bool:
        """Test whether a point lies inside the polygon.

        Parameters
        ----------
        x : float
            Query x coordinate.
        y : float
            Query y coordinate.

        Returns
        -------
        bool
            `True` when the point is inside.

        """
        c = False
        n = self.Npts
        xpts = self.data[0::2]
        ypts = self.data[1::2]
        bpx = xpts[0]
        bpy = ypts[0]
        for i in range(n - 1):
            fpx = xpts[i + 1]
            fpy = ypts[i + 1]
            a = (fpy > y) != (bpy > y)
            if bpy - fpy == 0:
                b = True
            else:
                b = x < ((bpx - fpx) * (y - fpy) / (bpy - fpy) + fpx)
            if a and b:
                c = not c
            bpx = fpx
            bpy = fpy
        return c

    def anisotropic_resize(
        self, angle: Sequence[float], deltas: Sequence[float]
    ) -> None:
        """Perform an anisotropic offset of the polygon.

        Requires an offset array in deltas matching the angle of expansion. Angles
        should cover -90 to 90 degrees.

        Parameters
        ----------
        angle : Sequence[float]
            Sequence of angles in degrees.
        deltas : Sequence[float]
            Sequence of offsets at the corresponding angles.

        Returns
        -------
        None

        """
        xpts = self.data[0::2]
        ypts = self.data[1::2]
        normals = []
        for i in range(len(xpts) - 1):
            x1 = xpts[i]
            y1 = ypts[i]
            x2 = xpts[i + 1]
            y2 = ypts[i + 1]
            # calculate normal
            b = x1 - x2
            a = y2 - y1
            c = x2 * y1 - x1 * y2

            nf = math.sqrt(a * a + b * b)
            nx = b / nf
            ny = a / nf
            alpha = math.degrees(math.atan2(ny, nx))

            d = np.interp(alpha, angle, deltas)
            c += nf * d
            normals.append([a, b, c])

        xpts = []
        ypts = []
        normals.append(normals[0])
        for i in range(len(normals) - 1):
            n1 = normals[i]
            n2 = normals[i + 1]
            D = n2[1] * n1[0] - n2[0] * n1[1]
            x = -n1[2] * n2[1] + n2[2] * n1[1]
            y = +n1[2] * n2[0] - n2[2] * n1[0]
            xpts.append(x / D)
            ypts.append(y / D)
        self.set_points(xpts, ypts)


class Path:
    """Open polyline with finite width."""

    def __init__(
        self, xpts: Sequence[float], ypts: Sequence[float], width: float, layer: int
    ) -> None:
        """Initialize a path.

        Parameters
        ----------
        xpts : Sequence[float]
            X coordinates of path vertices.
        ypts : Sequence[float]
            Y coordinates of path vertices.
        width : float
            Path width.
        layer : int
            Layer number.

        """
        self.xpts = xpts
        self.ypts = ypts
        self.width = width
        self.layer = layer
        self.Npts = len(xpts)

    def translate(self, dx: float, dy: float) -> None:
        """Translate the path.

        Parameters
        ----------
        dx : float
            Translation along x.
        dy : float
            Translation along y.

        Returns
        -------
        None

        """
        for i in range(self.Npts):
            self.xpts[i] = self.xpts[i] + dx
            self.ypts[i] = self.ypts[i] + dy

    def rotate_translate(self, x0: float, y0: float, rot: float) -> None:
        """Rotate around origin and then translate path vertices.

        Parameters
        ----------
        x0 : float
            Translation along x after rotation.
        y0 : float
            Translation along y after rotation.
        rot : float
            Rotation angle in degrees around origin.

        Returns
        -------
        None

        """
        cost = math.cos(rot / 180 * math.pi)
        sint = math.sin(rot / 180 * math.pi)
        for i in range(self.Npts):
            x = self.xpts[i]
            y = self.ypts[i]
            self.xpts[i] = cost * (x) - sint * (y) + x0
            self.ypts[i] = sint * (x) + cost * (y) + y0

    def rotate(self, x0: float, y0: float, rot: float) -> None:
        """Rotate the path around a center.

        Parameters
        ----------
        x0 : float
            Rotation center x coordinate.
        y0 : float
            Rotation center y coordinate.
        rot : float
            Rotation angle in degrees.

        Returns
        -------
        None

        """
        cost = math.cos(rot / 180 * math.pi)
        sint = math.sin(rot / 180 * math.pi)
        for i in range(self.Npts):
            x = self.xpts[i]
            y = self.ypts[i]
            self.xpts[i] = cost * (x - x0) - sint * (y - y0) + x0
            self.ypts[i] = sint * (x - x0) + cost * (y - y0) + y0

    def scale(self, x0: float, y0: float, scale_x: float, scale_y: float) -> None:
        """Scale path vertices around a center.

        Parameters
        ----------
        x0 : float
            Scaling center x coordinate.
        y0 : float
            Scaling center y coordinate.
        scale_x : float
            Scale factor along x.
        scale_y : float
            Scale factor along y.

        Returns
        -------
        None

        """
        for i in range(self.Npts):
            x = self.xpts[i]
            y = self.ypts[i]
            self.xpts[i] = scale_x * (x - x0) + x0
            self.ypts[i] = scale_y * (y - y0) + y0
            self.width *= scale_x

    def mirrorX(self, x0: float) -> None:
        """Mirror path vertices with respect to a vertical axis.

        Parameters
        ----------
        x0 : float
            X coordinate of mirror axis.

        Returns
        -------
        None

        """
        for i in range(self.Npts):
            self.xpts[i] = 2 * x0 - self.xpts[i]

    def mirrorY(self, y0: float) -> None:
        """Mirror path vertices with respect to a horizontal axis.

        Parameters
        ----------
        y0 : float
            Y coordinate of mirror axis.

        Returns
        -------
        None

        """
        for i in range(self.Npts):
            self.ypts[i] = 2 * y0 - self.ypts[i]

    def bounding_box(self) -> Box:
        """Compute the path bounding box.

        Returns
        -------
        Box
            Axis-aligned bounding box of path vertices.

        """
        llx = min(self.xpts)
        urx = max(self.xpts)
        lly = min(self.ypts)
        ury = max(self.ypts)
        return Box(llx, lly, urx - llx, ury - lly)

    def path_length(self) -> float:
        """Compute centerline path length.

        Returns
        -------
        float
            Total length of the path centerline.

        """
        x = self.xpts
        y = self.ypts
        plen = 0.0
        for i in range(1, self.Npts):
            plen += np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
        return plen

    def area(self) -> float:
        """Estimate path area as `length * width`.

        Returns
        -------
        float
            Approximate area.

        """
        # Approximately the path length * width
        return self.path_length() * self.width

    def centroid(self) -> tuple[float, float]:
        """Estimate path centroid from average vertex position.

        Returns
        -------
        tuple[float, float]
            Approximate centroid `(x, y)`.

        """
        # Give the average x,y
        cx = np.array(self.xpts).mean()
        cy = np.array(self.ypts).mean()
        return cx, cy

    def perimeter(self) -> float:
        """Estimate path perimeter.

        Returns
        -------
        float
            Approximate perimeter based on centerline length and width.

        """
        # Approximately twice the length and twice width
        return self.path_length() * 2 + self.width * 2

    def to_polygon(self) -> GeomGroup:
        """Convert the path to polygon geometry.

        Returns
        -------
        GeomGroup
            Group containing polygon approximation of the path.

        """
        x = self.xpts
        y = self.ypts
        w = self.width
        p1 = Poly([0], [0], self.layer)
        if self.Npts == 1:
            p1.set_points(
                [-w / 2, w / 2, w / 2, -w / 2], [-w / 2, -w / 2, w / 2, w / 2]
            )
            p1.translate(x, y)

        if self.Npts == 2:
            ang1 = math.atan2(y[1] - y[0], x[1] - x[0])
            c1 = w / 2 * math.cos(ang1 - math.pi / 2)
            c2 = w / 2 * math.cos(ang1 + math.pi / 2)
            s1 = w / 2 * math.sin(ang1 - math.pi / 2)
            s2 = w / 2 * math.sin(ang1 + math.pi / 2)
            p1.set_points(
                [x[0] + c1, x[1] + c1, x[1] + c2, x[0] + c2],
                [y[0] + s1, y[1] + s1, y[1] + s2, y[0] + s2],
            )

        if self.Npts > 2:
            xp1 = []
            yp1 = []
            xp2 = []
            yp2 = []
            for j in range(1, self.Npts - 1):
                ang1 = math.atan2(y[j] - y[j - 1], x[j] - x[j - 1])
                ang2 = math.atan2(y[j + 1] - y[j], x[j + 1] - x[j])
                d = (x[j + 1] - x[j - 1]) * (y[j] - y[j - 1]) - (
                    y[j + 1] - y[j - 1]
                ) * (x[j] - x[j - 1])
                if j == 1:
                    xp1.append(x[j - 1] + w / 2 * math.cos(ang1 - math.pi / 2))
                    yp1.append(y[j - 1] + w / 2 * math.sin(ang1 - math.pi / 2))
                    xp2.append(x[j - 1] + w / 2 * math.cos(ang1 + math.pi / 2))
                    yp2.append(y[j - 1] + w / 2 * math.sin(ang1 + math.pi / 2))

                if d < 0:
                    xp1.append(x[j] + w / 2 * math.cos(ang1 - math.pi / 2))
                    yp1.append(y[j] + w / 2 * math.sin(ang1 - math.pi / 2))
                    xp1.append(x[j] + w / 2 * math.cos(ang2 - math.pi / 2))
                    yp1.append(y[j] + w / 2 * math.sin(ang2 - math.pi / 2))
                    wx = w / 2 / math.cos((ang2 - ang1) / 2)
                    a0 = math.pi / 2 - (ang1 + ang2) / 2
                    xp2.append(x[j] - wx * math.cos(a0))
                    yp2.append(y[j] + wx * math.sin(a0))
                else:
                    xp2.append(x[j] + w / 2 * math.cos(ang1 + math.pi / 2))
                    yp2.append(y[j] + w / 2 * math.sin(ang1 + math.pi / 2))
                    xp2.append(x[j] + w / 2 * math.cos(ang2 + math.pi / 2))
                    yp2.append(y[j] + w / 2 * math.sin(ang2 + math.pi / 2))
                    wx = w / 2 / math.cos((ang2 - ang1) / 2)
                    a0 = math.pi / 2 - (ang1 + ang2) / 2
                    xp1.append(x[j] + wx * math.cos(a0))
                    yp1.append(y[j] - wx * math.sin(a0))
                if j == self.Npts - 2:
                    xp1.append(x[j + 1] + w / 2 * math.cos(ang2 - math.pi / 2))
                    yp1.append(y[j + 1] + w / 2 * math.sin(ang2 - math.pi / 2))
                    xp2.append(x[j + 1] + w / 2 * math.cos(ang2 + math.pi / 2))
                    yp2.append(y[j + 1] + w / 2 * math.sin(ang2 + math.pi / 2))

            xp2.reverse()
            yp2.reverse()
            p1.set_points(xp1 + xp2, yp1 + yp2)
        g = GeomGroup()
        g.add(p1)
        return g


class Text:
    """Text annotation object with stroke-based polygon conversion."""

    def __init__(
        self,
        x0: float,
        y0: float,
        text: str,
        posu: int,
        posv: int,
        height: float,
        width: float,
        angle: float,
        layer: int,
    ) -> None:
        """Create a text object.

        Parameters
        ----------
        x0 : float
            Text anchor x coordinate.
        y0 : float
            Text anchor y coordinate.
        text : str
            Text string.
        posu : int
            Horizontal alignment index.
        posv : int
            Vertical alignment index.
        height : float
            Character height.
        width : float
            Stroke width.
        angle : float
            Rotation angle in degrees.
        layer : int
            Layer number.

        """
        self.x0 = x0
        self.y0 = y0
        self.text = text
        self.posu = posu
        self.posv = posv
        self.height = height
        self.width = width
        self.angle = angle
        self.layer = layer

    def translate(self, dx: float, dy: float) -> None:
        """Translate text anchor position.

        Parameters
        ----------
        dx : float
            Translation along x.
        dy : float
            Translation along y.

        Returns
        -------
        None

        """
        self.x0 += dx
        self.y0 += dy

    def rotate_translate(self, dx: float, dy: float, rot: float) -> None:
        """Rotate around origin then translate text anchor.

        Parameters
        ----------
        dx : float
            Translation along x after rotation.
        dy : float
            Translation along y after rotation.
        rot : float
            Rotation angle in degrees around origin.

        Returns
        -------
        None

        """
        cost = math.cos(rot / 180 * math.pi)
        sint = math.sin(rot / 180 * math.pi)
        xv = self.x0
        yv = self.y0
        self.x0 = cost * xv - sint * yv + dx
        self.y0 = sint * xv + cost * yv + dy
        self.angle += rot

    def rotate(self, xc: float, yc: float, rot: float) -> None:
        """Rotate text around a center.

        Parameters
        ----------
        xc : float
            Rotation center x coordinate.
        yc : float
            Rotation center y coordinate.
        rot : float
            Rotation angle in degrees.

        Returns
        -------
        None

        """
        cost = math.cos(rot / 180 * math.pi)
        sint = math.sin(rot / 180 * math.pi)
        xv = self.x0 - xc
        yv = self.y0 - yc
        self.x0 = cost * xv - sint * yv + xc
        self.y0 = sint * xv + cost * yv + yc
        self.angle += rot

    def scale(self, xc: float, yc: float, scale_x: float, scale_y: float) -> None:
        """Scale text anchor and glyph dimensions.

        Parameters
        ----------
        xc : float
            Scaling center x coordinate.
        yc : float
            Scaling center y coordinate.
        scale_x : float
            Scale factor along x.
        scale_y : float
            Scale factor along y.

        Returns
        -------
        None

        """
        self.x0 = scale_x * (self.x0 - xc) + xc
        self.y0 = scale_y * (self.y0 - yc) + yc
        self.height *= scale_y
        self.width *= scale_x

    def mirrorX(self, xc: float) -> None:
        """Mirror text with respect to a vertical axis.

        Parameters
        ----------
        xc : float
            X coordinate of mirror axis.

        Returns
        -------
        None

        """
        self.x0 = 2 * xc - self.x0
        self.angle = 180 - self.angle

    def mirrorY(self, yc: float) -> None:
        """Mirror text with respect to a horizontal axis.

        Parameters
        ----------
        yc : float
            Y coordinate of mirror axis.

        Returns
        -------
        None

        """
        self.y0 = 2 * yc - self.y0
        self.angle = -self.angle

    def bounding_box(self) -> Box:
        """Return a degenerate bounding box at text anchor.

        Returns
        -------
        Box
            Bounding box estimate with zero width and height.

        """
        # Note this cannot be properly estimated
        return Box(self.x0, self.y0, 0, 0)

    def area(self) -> float:
        """Return area proxy for text.

        Returns
        -------
        float
            Always `0`.

        """
        return 0.0

    def centroid(self) -> tuple[float, float]:
        """Return text anchor as centroid.

        Returns
        -------
        tuple[float, float]
            Anchor coordinates `(x, y)`.

        """
        return self.x0, self.y0

    def perimeter(self) -> float:
        """Return perimeter proxy for text.

        Returns
        -------
        float
            Always `0`.

        """
        return 0.0

    def __to_path(self) -> GeomGroup:
        """Convert text glyphs into stroked path geometry.

        Returns
        -------
        GeomGroup
            Group containing path strokes for all supported glyphs.

        """
        offset = 0
        g = GeomGroup()
        for c in self.text:
            if c == " ":
                offset += self.height
            if c in _glyphs:
                letter = deepcopy(_glyphs[c][0])
                letter.set_layer(self.layer)
                letter.scale(0, 0, self.height, self.height)
                for p in letter.group:
                    # Note there are only paths in the group
                    p.width = self.width
                letter.translate(offset, 0)
                g += letter
                offset += _glyphs[c][1] * self.height
        # Now shift depending on posu/posv
        # offset contains the length of the text
        g.translate(-self.posu * offset / 2, (self.posv - 2) * self.height / 2)
        g.rotate(0, 0, self.angle)
        g.translate(self.x0, self.y0)
        return g

    def to_polygon(self) -> GeomGroup:
        """Convert text into polygon geometry.

        Returns
        -------
        GeomGroup
            Group containing polygon representation of the text.

        """
        g = self.__to_path()
        g.path_to_poly()
        return g


class RefBase:
    """Base transformation class for cell references."""

    def __init__(
        self, x0: float, y0: float, mag: float, angle: float, mirror: bool
    ) -> None:
        """Initialize common reference transform parameters.

        Parameters
        ----------
        x0 : float
            Reference x coordinate.
        y0 : float
            Reference y coordinate.
        mag : float
            Magnification factor.
        angle : float
            Rotation angle in degrees.
        mirror : bool
            Whether reflection is enabled.

        """
        self.x0 = x0
        self.y0 = y0
        self.mag = mag
        self.angle = angle
        self.mirror = mirror
        self.layer = 0  # Unused

    def translate(self, dx: float, dy: float) -> None:
        """Translate reference position.

        Parameters
        ----------
        dx : float
            Translation along x.
        dy : float
            Translation along y.

        Returns
        -------
        None

        """
        self.x0 += dx
        self.y0 += dy

    def rotate_translate(self, dx: float, dy: float, rot: float) -> None:
        """Rotate around origin and then translate reference position.

        Parameters
        ----------
        dx : float
            Translation along x after rotation.
        dy : float
            Translation along y after rotation.
        rot : float
            Rotation angle in degrees.

        Returns
        -------
        None

        """
        cost = math.cos(rot / 180 * math.pi)
        sint = math.sin(rot / 180 * math.pi)
        xv = self.x0
        yv = self.y0
        self.x0 = cost * xv - sint * yv + dx
        self.y0 = sint * xv + cost * yv + dy
        self.angle += rot
        self.angle = self.angle % 360

    def rotate(self, xc: float, yc: float, rot: float) -> None:
        """Rotate reference position around a center.

        Parameters
        ----------
        xc : float
            Rotation center x coordinate.
        yc : float
            Rotation center y coordinate.
        rot : float
            Rotation angle in degrees.

        Returns
        -------
        None

        """
        cost = math.cos(rot / 180 * math.pi)
        sint = math.sin(rot / 180 * math.pi)
        xv = self.x0 - xc
        yv = self.y0 - yc
        self.x0 = cost * xv - sint * yv + xc
        self.y0 = sint * xv + cost * yv + yc
        self.angle += rot
        self.angle = self.angle % 360

    def scale(self, xc: float, yc: float, scale_x: float, scale_y: float) -> None:
        """Scale reference position and magnification.

        Parameters
        ----------
        xc : float
            Scaling center x coordinate.
        yc : float
            Scaling center y coordinate.
        scale_x : float
            Scale factor along x.
        scale_y : float
            Scale factor along y.

        Returns
        -------
        None

        """
        self.x0 = scale_x * (self.x0 - xc) + xc
        self.y0 = scale_y * (self.y0 - yc) + yc
        self.mag *= scale_x

    def mirrorX(self, xc: float) -> None:
        """Mirror reference with respect to a vertical axis.

        Parameters
        ----------
        xc : float
            X coordinate of mirror axis.

        Returns
        -------
        None

        """
        self.x0 = 2 * xc - self.x0
        self.mirror = not self.mirror
        self.angle = 180 - self.angle
        self.angle = self.angle % 360

    def mirrorY(self, yc: float) -> None:
        """Mirror reference with respect to a horizontal axis.

        Parameters
        ----------
        yc : float
            Y coordinate of mirror axis.

        Returns
        -------
        None

        """
        self.y0 = 2 * yc - self.y0
        self.mirror = not self.mirror
        self.angle = -self.angle

    def centroid(self) -> tuple[float, float]:
        """Return reference anchor point.

        Returns
        -------
        tuple[float, float]
            Anchor coordinates `(x, y)`.

        """
        return self.x0, self.y0


class SRef(RefBase):
    """Single-cell reference geometry."""

    def __init__(
        self,
        x0: float,
        y0: float,
        cellname: str,
        group: GeomGroup,
        mag: float,
        angle: float,
        mirror: bool,
    ) -> None:
        """Create a single cell reference.

        Parameters
        ----------
        x0 : float
            Reference x coordinate.
        y0 : float
            Reference y coordinate.
        cellname : str
            Name of the referenced cell.
        group : GeomGroup
            Geometry of the referenced cell.
        mag : float
            Magnification factor.
        angle : float
            Rotation angle in degrees.
        mirror : bool
            Whether reflection is enabled.

        """
        RefBase.__init__(self, x0, y0, mag, angle, mirror)
        self.cellname = cellname
        self.group = group

    def bounding_box(self) -> Box:
        """Compute transformed bounding box of referenced geometry.

        Returns
        -------
        Box
            Bounding box after magnification, rotation, and mirror.

        """
        if self.cellname in _BoundingBoxPool:
            bb = _BoundingBoxPool[self.cellname]
        else:
            bb = self.group.bounding_box()
        p = bb.toPoly()
        p.scale(0, 0, self.mag, self.mag)
        p.rotate_translate(self.x0, self.y0, self.angle)
        if self.mirror:
            p.mirrorY(self.y0)
        return p.bounding_box()

    def place_group(self, flat_group: GeomGroup) -> GeomGroup:
        """Apply reference transform to a flattened group.

        Parameters
        ----------
        flat_group : GeomGroup
            Group to transform and place.

        Returns
        -------
        GeomGroup
            Transformed geometry group.

        """
        # scale first
        if self.mag != 1:
            flat_group.scale(0, 0, self.mag, self.mag)
        # roto-translate
        if self.mirror:
            flat_group.mirrorY(0)
        if self.angle != 0:
            flat_group.rotate_translate(self.x0, self.y0, self.angle)
        else:
            flat_group.translate(self.x0, self.y0)
        return flat_group


class ARef(SRef):
    """Array reference to repeated cell placements."""

    def __init__(
        self,
        x0: float,
        y0: float,
        cellname: str,
        group: GeomGroup,
        ncols: int,
        nrows: int,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        mag: float,
        angle: float,
        mirror: bool,
    ) -> None:
        """Create an array reference.

        Parameters
        ----------
        x0 : float
            Reference origin x coordinate.
        y0 : float
            Reference origin y coordinate.
        cellname : str
            Name of the referenced cell.
        group : GeomGroup
            Geometry of the referenced cell.
        ncols : int
            Number of columns.
        nrows : int
            Number of rows.
        ax : float
            X increment per column.
        ay : float
            Y increment per column.
        bx : float
            X increment per row.
        by : float
            Y increment per row.
        mag : float
            Magnification factor.
        angle : float
            Rotation angle in degrees.
        mirror : bool
            Whether reflection is enabled.

        """
        SRef.__init__(self, x0, y0, cellname, group, mag, angle, mirror)
        self.ncols = ncols
        self.nrows = nrows
        self.ax = ax
        self.ay = ay
        self.bx = bx
        self.by = by

    def bounding_box(self) -> Box:
        """Compute bounding box of all array instances.

        Returns
        -------
        Box
            Bounding box covering all placements.

        """
        bb = SRef.bounding_box(self)
        bbn = deepcopy(bb)
        for i in range(self.ncols):
            for j in range(self.nrows):
                dx = i * self.ax + j * self.bx
                dy = i * self.ay + j * self.by
                bbn.combine(Box(bb.llx + dx, bb.lly + dy, bb.width, bb.height))
        return bbn

    def place_group(self, flat_group: GeomGroup) -> GeomGroup:
        """Instantiate flattened group across the array lattice.

        Parameters
        ----------
        flat_group : GeomGroup
            Flattened geometry for one array element.

        Returns
        -------
        GeomGroup
            Group containing all array instances.

        """
        SRef.place_group(self, flat_group)
        base_group = flat_group.copy()
        for i in range(self.ncols):
            for j in range(self.nrows):
                if i == 0 and j == 0:
                    continue
                dx = i * self.ax + j * self.bx
                dy = i * self.ay + j * self.by
                ng = base_group.copy()
                ng.translate(dx, dy)
                flat_group += ng
        return flat_group


class Circle:
    """Circular geometry primitive."""

    def __init__(self, x0: float, y0: float, r: float, layer: int) -> None:
        """Create a circle.

        Parameters
        ----------
        x0 : float
            Center x coordinate.
        y0 : float
            Center y coordinate.
        r : float
            Radius.
        layer : int
            Layer number.

        """
        self.x0 = x0
        self.y0 = y0
        self.r = r
        self.layer = layer

    def translate(self, dx: float, dy: float) -> None:
        """Translate circle center.

        Parameters
        ----------
        dx : float
            Translation along x.
        dy : float
            Translation along y.

        Returns
        -------
        None

        """
        self.x0 += dx
        self.y0 += dy

    def rotate_translate(self, xc: float, yc: float, rot: float) -> None:
        """Rotate center around origin and then translate.

        Parameters
        ----------
        xc : float
            Translation along x after rotation.
        yc : float
            Translation along y after rotation.
        rot : float
            Rotation angle in degrees around origin.

        Returns
        -------
        None

        """
        cost = math.cos(rot / 180 * math.pi)
        sint = math.sin(rot / 180 * math.pi)
        x = self.x0
        y = self.y0
        self.x0 = cost * x - sint * y + xc
        self.y0 = sint * x + cost * y + yc

    def rotate(self, xc: float, yc: float, rot: float) -> None:
        """Rotate circle center around a point.

        Parameters
        ----------
        xc : float
            Rotation center x coordinate.
        yc : float
            Rotation center y coordinate.
        rot : float
            Rotation angle in degrees.

        Returns
        -------
        None

        """
        cost = math.cos(rot / 180 * math.pi)
        sint = math.sin(rot / 180 * math.pi)
        x = self.x0
        y = self.y0
        self.x0 = cost * (x - xc) - sint * (y - yc) + xc
        self.y0 = sint * (x - xc) + cost * (y - yc) + yc

    def scale(self, xc: float, yc: float, scale_x: float, scale_y: float) -> None:
        """Scale circle center and radius.

        Parameters
        ----------
        xc : float
            Scaling center x coordinate.
        yc : float
            Scaling center y coordinate.
        scale_x : float
            Scale factor along x.
        scale_y : float
            Scale factor along y.

        Returns
        -------
        None

        """
        self.x0 = scale_x * (self.x0 - xc) + xc
        self.y0 = scale_y * (self.y0 - yc) + yc
        self.r = scale_x * self.r

    def mirrorX(self, xc: float) -> None:
        """Mirror circle center with respect to a vertical axis.

        Parameters
        ----------
        xc : float
            X coordinate of mirror axis.

        Returns
        -------
        None

        """
        self.x0 = 2 * xc - self.x0

    def mirrorY(self, yc: float) -> None:
        """Mirror circle center with respect to a horizontal axis.

        Parameters
        ----------
        yc : float
            Y coordinate of mirror axis.

        Returns
        -------
        None

        """
        self.y0 = 2 * yc - self.y0

    def bounding_box(self) -> Box:
        """Compute circle bounding box.

        Returns
        -------
        Box
            Axis-aligned bounding box.

        """
        return Box(self.x0 - self.r, self.y0 - self.r, 2 * self.r, 2 * self.r)

    def area(self) -> float:
        """Compute circle area.

        Returns
        -------
        float
            Circle area.

        """
        return np.pi * self.r * self.r

    def centroid(self) -> float:
        """Return circle centroid.

        Returns
        -------
        tuple[float, float]
            Center coordinates `(x, y)`.

        """
        return self.x0, self.y0

    def perimeter(self) -> float:
        """Compute circle perimeter.

        Returns
        -------
        float
            Circumference.

        """
        return 2 * np.pi * self.r

    def to_polygon(self, Npts: int = 12) -> GeomGroup:
        """Approximate the circle with a polygon.

        Parameters
        ----------
        Npts : int, optional
            Number of polygon vertices. Default is 12.

        Returns
        -------
        GeomGroup
            Group containing one polygon approximation.

        """
        xc = np.array([0.0] * Npts)
        yc = np.array([0.0] * Npts)
        for i in range(Npts):
            xc[i] = math.cos(i * 2 * math.pi / Npts)
            yc[i] = math.sin(i * 2 * math.pi / Npts)
        g = GeomGroup()
        g.add(Poly(self.r * xc + self.x0, self.r * yc + self.y0, self.layer))
        return g


class Ellipse(Circle):
    """Ellipse primitive with independent x/y radii and rotation."""

    def __init__(
        self, x0: float, y0: float, rX: float, rY: float, layer: int, rot: float
    ) -> None:
        """Create an ellipse.

        Parameters
        ----------
        x0 : float
            Center x coordinate.
        y0 : float
            Center y coordinate.
        rX : float
            Radius along x before rotation.
        rY : float
            Radius along y before rotation.
        layer : int
            Layer number.
        rot : float
            Rotation angle in degrees.

        """
        Circle.__init__(self, x0, y0, rX, layer)
        self.r1 = rY
        self.rot = rot

    def rotate_translate(self, xc: float, yc: float, rot: float) -> None:
        """Rotate around origin then translate center and orientation.

        Parameters
        ----------
        xc : float
            Translation along x after rotation.
        yc : float
            Translation along y after rotation.
        rot : float
            Rotation angle in degrees around origin.

        Returns
        -------
        None

        """
        Circle.rotate_translate(self, xc, yc, rot)
        self.rot += rot

    def rotate(self, xc: float, yc: float, rot: float) -> None:
        """Rotate ellipse around a center.

        Parameters
        ----------
        xc : float
            Rotation center x coordinate.
        yc : float
            Rotation center y coordinate.
        rot : float
            Rotation angle in degrees.

        Returns
        -------
        None

        """
        Circle.rotate(self, xc, yc, rot)
        self.rot += rot

    def scale(self, xc: float, yc: float, scale_x: float, scale_y: float) -> None:
        """Scale ellipse center and radii.

        Parameters
        ----------
        xc : float
            Scaling center x coordinate.
        yc : float
            Scaling center y coordinate.
        scale_x : float
            Scale factor along x.
        scale_y : float
            Scale factor along y.

        Returns
        -------
        None

        """
        Circle.scale(self, xc, yc, scale_x, scale_y)
        self.r1 *= scale_y

    def mirrorX(self, xc: float) -> None:
        """Mirror ellipse with respect to a vertical axis.

        Parameters
        ----------
        xc : float
            X coordinate of mirror axis.

        Returns
        -------
        None

        """
        Circle.mirrorX(self, xc)
        self.rot = 180 - self.rot

    def mirrorY(self, yc: float) -> None:
        """Mirror ellipse with respect to a horizontal axis.

        Parameters
        ----------
        yc : float
            Y coordinate of mirror axis.

        Returns
        -------
        None

        """
        Circle.mirrorY(self, yc)
        self.rot = -self.rot

    def bounding_box(self) -> Box:
        """Compute ellipse bounding box from polygon approximation.

        Returns
        -------
        Box
            Axis-aligned bounding box.

        """
        g = self.to_polygon(12)
        return g.bounding_box()

    def area(self) -> float:
        """Compute ellipse area.

        Returns
        -------
        float
            Ellipse area.

        """
        return np.pi * self.r * self.r1

    def perimeter(self) -> float:
        """Estimate ellipse perimeter.

        Returns
        -------
        float
            Ramanujan approximation of perimeter.

        """
        a = self.r
        b = self.r1
        return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

    def to_polygon(self, Npts: int = 32) -> GeomGroup:
        """Approximate the ellipse with a polygon.

        Parameters
        ----------
        Npts : int, optional
            Number of polygon vertices. Default is 32.

        Returns
        -------
        GeomGroup
            Group containing one polygon approximation.

        """
        xc = np.array([0.0] * Npts)
        yc = np.array([0.0] * Npts)
        for i in range(Npts):
            xc[i] = math.cos(i * 2 * math.pi / Npts)
            yc[i] = math.sin(i * 2 * math.pi / Npts)
        g = GeomGroup()
        g.add(Poly(self.r * xc + self.x0, self.r1 * yc + self.y0, self.layer))
        g.rotate(self.x0, self.y0, self.rot)
        return g


class Ring(Ellipse):
    """Elliptical annulus primitive."""

    def __init__(
        self,
        x0: float,
        y0: float,
        rX: float,
        rY: float,
        layer: int,
        rot: float,
        w: float,
    ) -> None:
        """Create a ring.

        Parameters
        ----------
        x0 : float
            Center x coordinate.
        y0 : float
            Center y coordinate.
        rX : float
            Outer-center radius along x.
        rY : float
            Outer-center radius along y.
        layer : int
            Layer number.
        rot : float
            Rotation angle in degrees.
        w : float
            Ring width.

        """
        Ellipse.__init__(self, x0, y0, rX, rY, layer, rot)
        self.w = w

    def scale(self, xc: float, yc: float, scale_x: float, scale_y: float) -> None:
        """Scale ring center, radii, and width.

        Parameters
        ----------
        xc : float
            Scaling center x coordinate.
        yc : float
            Scaling center y coordinate.
        scale_x : float
            Scale factor along x.
        scale_y : float
            Scale factor along y.

        Returns
        -------
        None

        """
        Ellipse.scale(self, xc, yc, scale_x, scale_y)
        self.w *= scale_x

    def bounding_box(self) -> Box:
        """Compute ring bounding box from polygon approximation.

        Returns
        -------
        Box
            Axis-aligned bounding box.

        """
        g = self.to_polygon(12)
        return g.bounding_box()

    def area(self) -> float:
        """Compute ring area.

        Returns
        -------
        float
            Area enclosed by outer contour minus inner contour.

        """
        a1 = np.pi * (self.r + self.w / 2) * (self.r1 + self.w / 2)
        a2 = np.pi * (self.r - self.w / 2) * (self.r1 - self.w / 2)
        return a1 - a2

    def perimeter(self) -> float:
        """Estimate ring perimeter from polygon approximation.

        Returns
        -------
        float
            Approximate total contour length.

        """
        g = self.to_polygon(12)
        return g.group[0].perimeter()

    def to_polygon(self, Npts: int = 32) -> GeomGroup:
        """Approximate ring contours with a polygon.

        Parameters
        ----------
        Npts : int, optional
            Number of segments used per contour. Default is 32.

        Returns
        -------
        GeomGroup
            Group containing one polygon for the ring.

        """
        xpts = np.array([0.0] * (2 + Npts * 2))
        ypts = np.array([0.0] * (2 + Npts * 2))
        for i in range(1 + Npts):
            xpts[i] = math.cos(i * 2 * math.pi / Npts) * (self.r + self.w / 2) + self.x0
            ypts[i] = (
                math.sin(i * 2 * math.pi / Npts) * (self.r1 + self.w / 2) + self.y0
            )
        for i in range(1 + Npts):
            j = Npts - i
            xpts[i + 1 + Npts] = (
                math.cos(j * 2 * math.pi / Npts) * (self.r - self.w / 2) + self.x0
            )
            ypts[i + 1 + Npts] = (
                math.sin(j * 2 * math.pi / Npts) * (self.r1 - self.w / 2) + self.y0
            )
        p1 = Poly(xpts, ypts, self.layer)
        p1.rotate(self.x0, self.y0, self.rot)
        g = GeomGroup()
        g.add(p1)
        return g


class Arc(Ring):
    """Arc segment of an elliptical ring."""

    def __init__(
        self,
        x0: float,
        y0: float,
        rX: float,
        rY: float,
        layer: int,
        rot: float,
        w: float,
        a1: float,
        a2: float,
    ) -> None:
        """Create an arc.

        Parameters
        ----------
        x0 : float
            Center x coordinate.
        y0 : float
            Center y coordinate.
        rX : float
            Radius along x.
        rY : float
            Radius along y.
        layer : int
            Layer number.
        rot : float
            Rotation angle in degrees.
        w : float
            Arc width.
        a1 : float
            Start angle in degrees.
        a2 : float
            End angle in degrees.

        """
        Ring.__init__(self, x0, y0, rX, rY, layer, rot, w)
        self.a1 = a1
        self.a2 = a2

    def bounding_box(self) -> Box:
        """Compute arc bounding box from polygon approximation.

        Returns
        -------
        Box
            Axis-aligned bounding box.

        """
        g = self.to_polygon(12)
        return g.bounding_box()

    def area(self) -> float:
        """Compute arc area from parent ring area and angular span.

        Returns
        -------
        float
            Arc area.

        """
        ra = Ring.area(self)
        return ra * (math.radians(self.a2) - math.radians(self.a1)) / 2 / np.pi

    def centroid(self) -> tuple[float, float]:
        """Get the centroid of the arc.

        Returns
        -------
        tuple[float, float]
            Coordinates of the centroid `(x, y)`

        """
        g = self.to_polygon(12)
        return g.group[0].centroid()

    def to_polygon(self, Npts: int = 32, autosplit: bool = False) -> GeomGroup:
        """Convert arc to polygon.

        Parameters
        ----------
        Npts : int, optional
            Number of points to approximate the arc. Default is 32.
        autosplit : bool, optional
            Whether to split the arc into multiple polygons. Default is False.

        Returns
        -------
        GeomGroup
            GeomGroup containing the polygon representation of the arc.

        """
        Npts += 1
        th = np.linspace(math.radians(self.a1), math.radians(self.a2), Npts)
        xpts1 = np.cos(th) * (self.r + self.w / 2) + self.x0
        ypts1 = np.sin(th) * (self.r1 + self.w / 2) + self.y0
        xpts2 = np.cos(th) * (self.r - self.w / 2) + self.x0
        ypts2 = np.sin(th) * (self.r1 - self.w / 2) + self.y0
        g = GeomGroup()
        if autosplit:
            for i in range(Npts - 1):
                p1 = Poly(
                    ypts=np.append(
                        xpts1[i : (i + 2)],
                        xpts2[(-Npts + 1 + i) : (-Npts - 1 + i) : -1],
                    ),
                    xpts=np.append(
                        ypts1[i : (i + 2)],
                        ypts2[(-Npts + 1 + i) : (-Npts - 1 + i) : -1],
                    ),
                    layer=self.layer,
                )
                p1.rotate(self.x0, self.y0, self.rot)
                g.add(p1)
        else:
            p1 = Poly(
                xpts=np.append(xpts1, xpts2[::-1]),
                ypts=np.append(ypts1, ypts2[::-1]),
                layer=self.layer,
            )
            p1.rotate(self.x0, self.y0, self.rot)
            g.add(p1)
        return g


# Load fonts and store the glyphs
# Maybe we should place this somewhere else
caps = dict()
with open(_STENCIL_FONT_PATH, encoding=_STENCIL_FONT_ENCODING) as f:
    c = "a"
    for line in f:
        test = line.rstrip("\n").split(" ")
        if len(test) == 1:
            c = test[0]
            caps[c] = []
        else:
            nums = list(map(float, test))
            caps[c] += nums

for i in caps:
    data = caps[i]
    gl = GeomGroup()
    xpts = []
    ypts = []
    for j in range(0, len(data), 3):
        x = data[j] / 3.6
        y = (data[j + 1]) / 3.6
        flag = data[j + 2]
        if flag == 0:
            if j > 0:
                gl.add(Path(xpts, ypts, 1, 0))
            xpts = [x]
            ypts = [y]
        elif flag > 0:
            xpts.append(x)
            ypts.append(y)
        else:
            gl.add(Path(xpts, ypts, 1, 0))
    _glyphs[i] = (gl, x)

del caps

"""Classes for drawing photonic crystals and periodic structures.

Crystals
--------
Crystals are periodic structures arranged in two dimensions. They are defined by
a unit cell and a set of lattice sites (usually with some periodicity)

In this module, a unit cell is given by a user-provided function that takes parameters
as input (e.g. the radius of a circle) and produces a geometry.

For example the default unit cell function is a circle defined as:

    def __circ_cellfun__(x,y,params):
    if params=="test":
        return 1
    else:
        return sm.make_circle(x, y, params[0], 0)

The `Crystal` class provides a template for periodic structures consisting of an
array of lattice sites (coordinates) in normalized units and a list of parameters
to be passed to the unit cell function.
Thus a crystal is created by multiple calls to the unit cell function.
Note that the unit cell function can also return references to another cell, for example
a cell that contains a single circle.

Two make_ functions are provided to create a samplemaker.shapes.GeomGroup object
with the designed parameters.

"""

import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from typing import Any, Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

import samplemaker.makers as sm
from samplemaker import _legacy
from samplemaker.layout import LayoutPool
from samplemaker.shapes import GeomGroup, Poly

CELLFUN_TYPE: TypeAlias = Callable[[float, float, Sequence[float]], GeomGroup]


class Crystal:
    """Class containing the lattice sites and parameters of a crystal.

    Comes with several templates for common photonic crystals.
    """

    def __init__(
        self,
        xpts: Iterable[float] | None = None,
        ypts: Iterable[float] | None = None,
        params: Iterable[Iterable[float]] | None = None,
    ) -> None:
        """Initialize a Crystal template.

        Parameters
        ----------
        xpts : Iterable[float], optional
            List of x-coordinates (normalized) of the lattice sites, by default [].
        ypts : Iterable[float], optional
            List of y-coordinates (normalized) of the lattice sites, by default [].
        params : Iterable[Iterable[float]], optional
            2D list of parameter values of the lattice sites. Should be of the form
            params[pindex,site_index], by default [].

        """
        xpts = [] if xpts is None else xpts
        ypts = [] if ypts is None else ypts
        params = [] if params is None else params

        self.xpts = np.array(xpts, dtype=np.float64)
        self.ypts = np.array(ypts, dtype=np.float64)
        self.params = np.array(params, dtype=np.float64)

    def remove_at_index(self, index: Sequence[int]) -> None:
        """Remove lattice sites from a list of indices.

        Parameters
        ----------
        index : Sequence[int]
            indexes to be removed from the list used to initialize the crystal.

        Returns
        -------
        None

        """
        if len(index) > 0:
            self.xpts = np.delete(self.xpts, index)
            self.ypts = np.delete(self.ypts, index)
            self.params = np.delete(self.params, index, axis=1)

    def shift_at_index(
        self,
        index: list[int],
        shift_x: float,
        shift_y: float,
        relative: bool = False,
        orig_x: float = 0,
        orig_y: float = 0,
    ) -> None:
        """Shifts the lattice sites specified in the list.

        Parameters
        ----------
        index : list[int]
            A list of indexes to be shifted.
        shift_x : float
            x-amount of shift (in normalized units).
        shift_y : float
            y-amount of shift (in normalized units).
        relative : bool, optional
            Perform a relative shift from the origin, by default False.
        orig_x : float, optional
            x-coordinate of the origin of shift (if relative), by default 0.
        orig_y : float, optional
            y-coordinate of the origin of shift (if relative), by default 0.

        Returns
        -------
        None

        """
        if len(index) > 0:
            if relative:
                x_offset = (2.0 * (self.xpts[index] > orig_x) - 1) * shift_x
                self.xpts[index] = self.xpts[index] + x_offset
                y_offset = (2.0 * (self.ypts[index] > orig_y) - 1) * shift_y
                self.ypts[index] = self.ypts[index] + y_offset
            else:
                self.xpts[index] = self.xpts[index] + shift_x
                self.ypts[index] = self.ypts[index] + shift_y

    def param_at_index(self, index: int, pindex: int, pvalues: float) -> None:
        """Set a parameter at the lattice index.

        Parameters
        ----------
        index : int
            The lattice index.
        pindex : int
            The parameter index.
        pvalues : float
            The new value of the parameter to be set.

        Returns
        -------
        None

        """
        self.params[pindex, index] = pvalues

    def coord_to_index(self, xc: float | ArrayLike, yc: float | ArrayLike) -> list[int]:
        """Convert a coordinate to an index (if matches).

        Parameters
        ----------
        xc : float | ArrayLike
            x-coordinate(s) in normalized units.
        yc : float | ArrayLike
            y-coordinate(s) in normalized units.

        Returns
        -------
        sel : list[int]
            A list of coordinate indices.

        """
        xc = np.asarray(xc)
        yc = np.asarray(yc)
        sel = []
        for i in range(xc.size):
            sx = abs(self.xpts - xc[i]) < 1e-6
            sy = abs(self.ypts - yc[i]) < 1e-6
            res = np.where(sx & sy)
            if res[0].size == 0:
                msg = f"No coordinate match for ({xc[i]}, {yc[i]})"
                warnings.warn(msg, UserWarning, stacklevel=2)
            else:
                sel.append(res[0][0])
        return sel

    def remove_crystal(self, crystal: "Crystal") -> None:
        """Subtracts a crystal from another crystal.

        Parameters
        ----------
        crystal : Crystal
            Another crystal whose lattice sites will be removed.

        Returns
        -------
        None

        """
        self.remove_at_index(self.coord_to_index(crystal.xpts, crystal.ypts))

    def add_crystal(self, crystal: "Crystal") -> None:
        """Add another crystal to the current crystal.

        Parameters
        ----------
        crystal : Crystal
            Another crystal to be added to the existing one.

        Returns
        -------
        None

        """
        self.xpts = np.append(self.xpts, crystal.xpts)
        self.ypts = np.append(self.ypts, crystal.ypts)
        self.params = np.append(self.params, crystal.params, axis=1)

    def copy(self) -> Self:
        """Create a deep copy of the crystal.

        Returns
        -------
        Self
            A deepcopy of crystal.

        """
        return deepcopy(self)

    @classmethod
    def triangular_hexagonal(
        cls,
        n: int | _legacy.MissingType = _legacy.MISSING,
        filled: bool | _legacy.MissingType = _legacy.MISSING,
        nparams: int = 1,
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Create a triangular photonic crystal in the shape of a hexagon.

        Often useful for point-defect cavities.

        Parameters
        ----------
        n : int
            Number of lattice sites extending in the radial direction (0 means one hole
            in the center).
        filled : bool
            If True, creates a filled hexagonal crystal, otherwise a ring of radius n.
        nparams : int, optional
            Number of parameters to be controlled for each lattice site, by default 1.
        kwargs : dict
            Additional keyword arguments. Supports 'N' and 'Nparams' for backward
            compatibility.

        Returns
        -------
        Self
            A crystal object with the pre-compiled lattice sites.

        """
        n = _legacy.get_kwarg("n", n, "N", kwargs)
        nparams = _legacy.get_optional_kwarg("nparams", nparams, 1, "Nparams", kwargs)
        _legacy.ensure_empty_kwargs("Crystal.triangular_hexagonal", kwargs)
        _legacy.check_missing_args("Crystal.triangular_hexagonal", n=n, filled=filled)

        n = _legacy.ensure_arg_type("n", n)
        filled = _legacy.ensure_arg_type("filled", filled)

        if n == 0:
            return cls(np.array([0]), np.array([0]), np.ones((nparams, 1)))
        xpts = np.array([])
        ypts = np.array([])

        if filled:
            for i in range(n):
                tmpc = cls.triangular_hexagonal(i, False)
                xpts = np.append(xpts, tmpc.xpts)
                ypts = np.append(ypts, tmpc.ypts)
        else:
            th = np.array(list(range(0, 361, 60)))
            cx = n * np.cos(np.radians(th))
            cy = n * np.sin(np.radians(th))
            for i in range(6):
                xint = np.linspace(cx[i], cx[i + 1], n + 1)
                m = (cy[i + 1] - cy[i]) / (cx[i + 1] - cx[i])
                yint = m * (xint[0:-1:] - cx[i]) + cy[i]
                xpts = np.append(xpts, xint[0:-1:])
                ypts = np.append(ypts, yint)

        params = np.ones((nparams, xpts.size))
        return cls(xpts, ypts, params)

    @classmethod
    def triangular_box(
        cls,
        nx: int | _legacy.MissingType = _legacy.MISSING,
        ny: int | _legacy.MissingType = _legacy.MISSING,
        nparams: int = 1,
        **kwargs: int,
    ) -> Self:
        """Create a triangular photonic crystal in the shape of a box.

        Parameters
        ----------
        nx : int
            Number of holes in the x direction, the crystal will span from
            -nx to nx (double size).
        ny : int
            Number of holes in the y direction, note that we consider ny=1 the
            row where y=sqrt(3). The crystal will span from -ny to ny.
        nparams : int, optional
            Number of parameters to be controlled for each lattice site, by default 1.
        kwargs : dict
            Additional keyword arguments. Supports 'Nx', 'Ny' and 'Nparams' for backward
            compatibility.

        Returns
        -------
        Self
            A crystal object with the pre-compiled lattice sites.

        """
        nx = _legacy.get_kwarg("nx", nx, "Nx", kwargs)
        ny = _legacy.get_kwarg("ny", ny, "Ny", kwargs)
        nparams = _legacy.get_optional_kwarg("nparams", nparams, 1, "Nparams", kwargs)
        _legacy.ensure_empty_kwargs("Crystal.triangular_box", kwargs)
        _legacy.check_missing_args("Crystal.triangular_box", nx=nx, ny=ny)

        nx = _legacy.ensure_arg_type("nx", nx)
        ny = _legacy.ensure_arg_type("ny", ny)

        if nx == 0 and ny == 0:
            return cls(np.array([0]), np.array([0]), np.ones((nparams, 1)))

        x1 = np.array(list(range(-nx, nx + 1)))
        y1 = np.array([e * math.sqrt(3) for e in range(-ny, ny + 1)])
        x2 = np.array([e + 0.5 for e in range(-nx, nx)])
        y2 = np.array([math.sqrt(3) / 2 + math.sqrt(3) * e for e in range(-ny, ny)])
        x1_mesh, y1_mesh = np.meshgrid(x1, y1)
        x2_mesh, y2_mesh = np.meshgrid(x2, y2)
        xpts = np.append(x1_mesh.reshape(-1), x2_mesh.reshape(-1))
        ypts = np.append(y1_mesh.reshape(-1), y2_mesh.reshape(-1))
        params = np.ones((nparams, xpts.size))
        return cls(xpts, ypts, params)

    @classmethod
    def triangular_heterophc(
        cls,
        nx: float | _legacy.MissingType = _legacy.MISSING,
        ny: int | _legacy.MissingType = _legacy.MISSING,
        spacing: list[float] | _legacy.MissingType = _legacy.MISSING,
        periods: list[int] | _legacy.MissingType = _legacy.MISSING,
        nparams: int = 1,
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Create a triangular photonic crystal.

        The resulting crystal is in the shape of a rectangular box using a
        heterostructure.

        Parameters
        ----------
        nx : float
            Number of holes in the x direction, the crystal will span from
            -nx to nx (double size). Can be a fraction to end the crystal with a partial
            period.
        ny : int
            Number of holes in the y direction, note that we consider ny=1 the
            row where y=sqrt(3). The crystal will span from -ny to ny.
        spacing : list[float]
            Array of lattice constants to be used for the various sections of the hetero
            phc.
        periods : list[int]
            How many times should each spacing be repeated (always end with 1 for the
            remaining).
        nparams : int, optional
            Number of parameters to be controlled for each lattice site, by default 1.
        kwargs : dict
            Additional keyword arguments. Supports 'Nx', 'Ny' and 'Nparams' for backward
            compatibility.

        Returns
        -------
        Self
            A crystal object with the pre-compiled lattice sites.

        """
        nx = _legacy.get_kwarg("nx", nx, "Nx", kwargs)
        ny = _legacy.get_kwarg("ny", ny, "Ny", kwargs)
        nparams = _legacy.get_optional_kwarg("nparams", nparams, 1, "Nparams", kwargs)
        _legacy.ensure_empty_kwargs("Crystal.triangular_heterophc", kwargs)
        _legacy.check_missing_args(
            "Crystal.triangular_heterophc",
            nx=nx,
            ny=ny,
            spacing=spacing,
            periods=periods,
        )

        nx = _legacy.ensure_arg_type("nx", nx)
        ny = _legacy.ensure_arg_type("ny", ny)
        spacing = _legacy.ensure_arg_type("spacing", spacing)
        periods = _legacy.ensure_arg_type("periods", periods)

        startx = 0
        x1 = []
        x2 = []
        a = spacing
        totalp = np.sum(periods)
        nx_original = nx
        nx = math.ceil(nx)

        for i in range(len(a)):
            xchunk1 = startx + np.array(list(range(periods[i] + 1))) * a[i]
            xchunk2 = startx + (0.5 + np.array(list(range(periods[i])))) * a[i]
            startx = xchunk1[-1]
            x1 = np.append(x1, xchunk1)
            x2 = np.append(x2, xchunk2)

        x1 = np.append(x1, startx + np.array(list(range(int(nx - totalp + 1)))))
        x2 = np.append(x2, startx + (0.5 + np.array(list(range(int(nx - totalp))))))
        x1 = np.append(x1, -x1[::-1])
        x2 = np.append(x2, -x2[::-1])
        x1 = np.sort(np.unique(x1))
        x2 = np.sort(np.unique(x2))
        y1 = np.array([e * math.sqrt(3) for e in range(-ny, ny + 1)])
        y2 = np.array([math.sqrt(3) / 2 + math.sqrt(3) * e for e in range(-ny, ny)])
        x1_mesh, y1_mesh = np.meshgrid(x1, y1)
        x2_mesh, y2_mesh = np.meshgrid(x2, y2)
        xpts = np.append(x1_mesh.reshape(-1), x2_mesh.reshape(-1))
        ypts = np.append(y1_mesh.reshape(-1), y2_mesh.reshape(-1))
        params = np.ones((nparams, xpts.size))
        heterophc = cls(xpts, ypts, params)
        if ny != 0:
            heterophc.remove_crystal(cls.triangular_heterophc(nx, 0, a, periods))

        # Get rid of extra final holes if fraction Nx
        if nx_original - nx < 0:
            maxx = np.max(heterophc.xpts)
            minx = np.min(heterophc.xpts)
            sx1 = heterophc.xpts > (maxx - 0.1)
            sx2 = heterophc.xpts < (minx + 0.1)
            sel = np.where(sx1 | sx2)[0]
            heterophc.remove_at_index(sel.tolist())

        return heterophc


def make_phc_circle(x: float, y: float, params: Sequence[float]) -> GeomGroup:
    """Create a circular unit cell for a photonic crystal.

    Parameters
    ----------
    x : float
        x-coordinate of the center of the circle.
    y : float
        y-coordinate of the center of the circle.
    params : Sequence[float]
        Parameters for the unit cell. It should be a sequence containing the radius of
        the circle.

    Returns
    -------
    GeomGroup
        A geometry containing the circular unit cell.

    """
    return sm.make_circle(x, y, params[0], 0)


def make_phc_circle_ref(x: float, y: float, params: Sequence[float]) -> GeomGroup:
    """Create a circular unit cell for a photonic crystal using a circle reference.

    Parameters
    ----------
    x : float
        x-coordinate of the center of the circle.
    y : float
        y-coordinate of the center of the circle.
    params : Sequence[float]
        Parameters for the unit cell. It should be a sequence containing the radius of
        the circle.

    Returns
    -------
    GeomGroup
        A geometry containing the circular unit cell.

    """
    return sm.make_sref(x, y, "_CIRCLE", LayoutPool["_CIRCLE"], mag=params[0])


def _validate_crystal(crystal: Crystal) -> None:
    if len(crystal.xpts) != len(crystal.ypts):
        msg = "The number of x-coordinates must match the number of y-coordinates."
        raise ValueError(msg)
    if len(crystal.xpts) == 0:
        # We allow empty crystals
        return
    if crystal.params.ndim != 2:
        msg = "The params array must be 2-dimensional."
        raise ValueError(msg)
    if crystal.params.shape[1] != len(crystal.xpts):
        msg = "The number of parameter sets must match the number of lattice sites."
        raise ValueError(msg)


def make_phc(
    crystal: Crystal,
    scaling: float,
    cellparams: list[float],
    x0: float,
    y0: float,
    cellfun: CELLFUN_TYPE = make_phc_circle,
    name: str = "",
) -> GeomGroup:
    """Create a photonic crystal geometry.

    Parameters
    ----------
    crystal : Crystal
        The crystal template.
    scaling : float
        An overall scaling factor in um.
    cellparams : list[float]
        A list with the scaling parameters to be passed to the cell function.
    x0 : float
        Position x-coordinate in um.
    y0 : float
        Position y-coordinate in um.
    cellfun : Callable[[float, float, list[float] | str], GeomGroup], optional
        A function of the type fun(x,y,params) that returns the geometry of the unit
        cell. It should also return the number of parameters required to draw the unit
        cell if "test" is passed as params, by default make_phc_circle.
    name : str, optional
        DEPRECATED. Name of the crystal, by default "".

    Returns
    -------
    GeomGroup
        A geometry containing the full crystal.

    Raises
    ------
    TypeError
        If the cellfun function does not return a GeomGroup when called with
        valid parameters.
    ValueError
        If the passed crystal has inconsistent dimensions or parameters or if the number
        of cell parameters does not match the number of parameter sets in the crystal.

    """
    if name:
        msg = (
            "The 'name' parameter is deprecated and will be removed in future versions."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    _validate_crystal(crystal)

    phc = GeomGroup()
    if len(crystal.xpts) == 0:
        return phc

    if len(cellparams) != crystal.params.shape[0]:
        msg = (
            "The number of cell parameters must match the number of parameter sets "
            "in the crystal."
        )
        raise ValueError(msg)

    xpts_scaled = np.asarray(crystal.xpts) * scaling
    ypts_scaled = np.asarray(crystal.ypts) * scaling
    params = np.asarray(crystal.params) * np.asarray(cellparams)[:, np.newaxis]
    for i, (x, y) in enumerate(zip(xpts_scaled, ypts_scaled, strict=True)):
        g = cellfun(x, y, params[:, i].tolist())
        if not isinstance(g, GeomGroup):
            msg = (
                "The cellfun function must return a GeomGroup when called with "
                "valid parameters."
            )
            raise TypeError(msg)
        phc.group.extend(g.group)

    phc.translate(x0, y0)
    return phc


def make_phc_inpoly(
    crystal: Crystal,
    poly: Poly,
    scaling: float,
    cellparams: list[float],
    x0: float,
    y0: float,
    cellfun: CELLFUN_TYPE = make_phc_circle,
    name: str = "",
) -> GeomGroup:
    """Create a photonic crystal geometry clipped inside a polygon area.

    Parameters
    ----------
    crystal : Crystal
        The crystal template.
    poly: Poly
        The polygon used to clip. Should be created with samplemaker.shapes.Poly
    scaling : float
        An overall scaling factor in um.
    cellparams : list[float]
        A list with the scaling parameters to be passed to the cell function.
    x0 : float
        Position x-coordinate in um.
    y0 : float
        Position y-coordinate in um.
    cellfun : Callable[[float, float, list[float] | str], GeomGroup], optional
        A function of the type fun(x,y,params) that returns the geometry of the unit
        cell. It should also return the number of parameters required to draw the unit
        cell if "test" is passed as params, by default make_phc_circle.
    name : str, optional
        DEPRECATED. Name of the crystal, by default "".

    Returns
    -------
    GeomGroup
        A geometry containing the full crystal.

    Raises
    ------
    TypeError
        If the cellfun function does not return a GeomGroup when called with
        valid parameters.
    ValueError
        If the passed crystal has inconsistent dimensions or parameters or if the number
        of cell parameters does not match the number of parameter sets in the crystal.

    """
    if name:
        msg = (
            "The 'name' parameter is deprecated and will be removed in future versions."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    _validate_crystal(crystal)

    phc = GeomGroup()
    if len(crystal.xpts) == 0:
        return phc

    if len(cellparams) != crystal.params.shape[0]:
        msg = (
            "The number of cell parameters must match the number of parameter sets "
            "in the crystal."
        )
        raise ValueError(msg)

    xpts_scaled = np.asarray(crystal.xpts) * scaling
    ypts_scaled = np.asarray(crystal.ypts) * scaling
    params = np.asarray(crystal.params) * np.asarray(cellparams)[:, np.newaxis]
    for i, (x, y) in enumerate(zip(xpts_scaled, ypts_scaled, strict=True)):
        if not poly.point_inside(x, y):
            continue
        g = cellfun(x, y, params[:, i].tolist())
        if not isinstance(g, GeomGroup):
            msg = (
                "The cellfun function must return a GeomGroup when called with "
                "valid parameters."
            )
            raise TypeError(msg)
        phc.group.extend(g.group)

    phc.translate(x0, y0)
    return phc

"""Unit tests for shapes module."""

from dataclasses import dataclass

import numpy as np
import pytest

# Used automatically by pytest to reset state before each test:
from fixtures import reset_samplemaker  # noqa: F401

from samplemaker import shapes as sp

_COORD = tuple[float, float]
_TF = tuple[float, ...]


@pytest.fixture
def box_llxy() -> _COORD:
    """Box lower-left x and y coordinates."""
    return 0.0, 1.0


@pytest.fixture
def box_wh() -> _COORD:
    """Box width and height."""
    return 2.0, 3.0


@pytest.fixture
def box_obj(box_llxy: _COORD, box_wh: _COORD) -> sp.Box:
    llx, lly = box_llxy
    width, height = box_wh
    return sp.Box(llx, lly, width, height)


@pytest.fixture
def poly_pts() -> tuple[_TF, _TF]:
    """Points in test polygon."""
    xpts = (0.0, 3.0, 3.0, 0.0)
    ypts = (0.0, 0.0, 2.0, 2.0)
    return xpts, ypts


@pytest.fixture
def poly_obj(poly_pts: tuple[_TF, _TF]) -> sp.Poly:
    xpts, ypts = poly_pts
    return sp.Poly(xpts, ypts, layer=2)


@pytest.fixture
def path_width() -> float:
    """Width of test path."""
    return 0.5


@pytest.fixture
def path_obj(poly_pts: tuple[_TF, _TF], path_width: float) -> sp.Path:
    xpts, ypts = poly_pts
    # xpts and ypts need to be mutable for the transformations,
    # so we convert them to lists here.
    return sp.Path(list(xpts), list(ypts), path_width, layer=2)


@pytest.fixture
def text_obj() -> sp.Text:
    return sp.Text(
        x0=1.0,
        y0=2.0,
        text="Hello, world!",
        posu=0,  # 0, 1 or 2 for left, center or right justified text
        posv=0,  # 0, 1 or 2 for bottom, middle or top justified text
        height=10.0,
        width=1.0,
        angle=0.0,
        layer=3,
    )


@pytest.fixture
def geomgroup_obj(box_obj: sp.Box) -> sp.GeomGroup:
    return box_obj.toRect()


@dataclass(frozen=True)
class SRefKwargs:
    x0: float
    y0: float
    cellname: str
    mag: float
    angle: float
    mirror: bool


@pytest.fixture
def sref_kwargs() -> SRefKwargs:
    return SRefKwargs(
        x0=1.0,
        y0=2.0,
        cellname="MyCell",
        mag=1.0,
        angle=0.0,
        mirror=False,
    )


@pytest.fixture
def sref_obj(geomgroup_obj: sp.GeomGroup, sref_kwargs: SRefKwargs) -> sp.SRef:
    return sp.SRef(
        x0=sref_kwargs.x0,
        y0=sref_kwargs.y0,
        cellname=sref_kwargs.cellname,
        group=geomgroup_obj,
        mag=sref_kwargs.mag,
        angle=sref_kwargs.angle,
        mirror=sref_kwargs.mirror,
    )


@dataclass(frozen=True)
class ARefKwargs:
    x0: float
    y0: float
    cellname: str
    ncols: int
    nrows: int
    ax: float
    ay: float
    bx: float
    by: float
    mag: float
    angle: float
    mirror: bool


@pytest.fixture
def aref_kwargs() -> ARefKwargs:
    return ARefKwargs(
        x0=1.0,
        y0=2.0,
        cellname="MyArrayCell",
        ncols=2,
        nrows=3,
        ax=4.0,
        ay=0.0,
        bx=0.0,
        by=5.0,
        mag=1.0,
        angle=0.0,
        mirror=False,
    )


@pytest.fixture
def aref_obj(geomgroup_obj: sp.GeomGroup, aref_kwargs: ARefKwargs) -> sp.ARef:
    return sp.ARef(
        x0=aref_kwargs.x0,
        y0=aref_kwargs.y0,
        cellname=aref_kwargs.cellname,
        group=geomgroup_obj,
        ncols=aref_kwargs.ncols,
        nrows=aref_kwargs.nrows,
        ax=aref_kwargs.ax,
        ay=aref_kwargs.ay,
        bx=aref_kwargs.bx,
        by=aref_kwargs.by,
        mag=aref_kwargs.mag,
        angle=aref_kwargs.angle,
        mirror=aref_kwargs.mirror,
    )


@pytest.fixture
def circle_obj() -> sp.Circle:
    return sp.Circle(x0=1.0, y0=2.0, r=3.0, layer=4)


@pytest.fixture
def ellipse_obj() -> sp.Ellipse:
    return sp.Ellipse(x0=1.0, y0=2.0, rX=3.0, rY=2.0, layer=4, rot=0.0)


@pytest.fixture
def ring_obj() -> sp.Ring:
    return sp.Ring(x0=1.0, y0=2.0, rX=3.0, rY=2.0, layer=4, rot=0.0, w=1.0)


@pytest.fixture
def arc_obj() -> sp.Arc:
    return sp.Arc(
        x0=1.0, y0=2.0, rX=3.0, rY=2.0, layer=4, rot=0.0, w=1.0, a1=0.0, a2=180.0
    )


def test_dot_transformations() -> None:
    d = sp.Dot(1.0, 2.0)
    assert (d.x, d.y) == pytest.approx((1.0, 2.0))

    d.translate(2.0, -1.0)
    assert (d.x, d.y) == pytest.approx((3.0, 1.0))

    d.rotate(0.0, 0.0, 90.0)
    assert (d.x, d.y) == pytest.approx((-1.0, 3.0))

    d.mirrorX(0.0)
    assert (d.x, d.y) == pytest.approx((1.0, 3.0))

    d.mirrorY(0.0)
    assert (d.x, d.y) == pytest.approx((1.0, -3.0))

    d.scale(0, 0, 2.0, 2.0)
    assert (d.x, d.y) == pytest.approx((2.0, -6.0))


class TestBox:
    def test_basic_ops(self, box_obj: sp.Box) -> None:
        expected_cx = box_obj.llx + box_obj.width / 2
        expected_cy = box_obj.lly + box_obj.height / 2
        expected_urx = box_obj.llx + box_obj.width
        expected_ury = box_obj.lly + box_obj.height

        assert box_obj.cx() == pytest.approx(expected_cx)
        assert box_obj.cy() == pytest.approx(expected_cy)
        assert box_obj.urx() == pytest.approx(expected_urx)
        assert box_obj.ury() == pytest.approx(expected_ury)

        other = sp.Box(-1.0, 0.0, 1.0, 1.0)
        box_obj.combine(other)

        expected_llx = min(box_obj.llx, other.llx)
        expected_lly = min(box_obj.lly, other.lly)
        expected_urx = max(box_obj.urx(), other.urx())
        expected_ury = max(box_obj.ury(), other.ury())
        expected_width = expected_urx - expected_llx
        expected_height = expected_ury - expected_lly

        assert box_obj.llx == pytest.approx(expected_llx)
        assert box_obj.lly == pytest.approx(expected_lly)
        assert box_obj.urx() == pytest.approx(expected_urx)
        assert box_obj.ury() == pytest.approx(expected_ury)
        assert box_obj.width == pytest.approx(expected_width)
        assert box_obj.height == pytest.approx(expected_height)

    def test_to_poly(self, box_obj: sp.Box) -> None:
        poly = box_obj.toPoly()
        assert isinstance(poly, sp.Poly)
        # poly.data should be of the format [x0, y0, x1, y1, x2, y2, x3, y3, ..., x0, y0]
        assert isinstance(poly.data, np.ndarray)
        assert len(poly.data) == 10

        reshaped_data = poly.data.reshape((-1, 2))
        xpts, ypts = reshaped_data[:, 0], reshaped_data[:, 1]

        llx, lly = box_obj.llx, box_obj.lly
        urx, ury = box_obj.urx(), box_obj.ury()
        expected_xpts = [llx, urx, urx, llx, llx]
        expected_ypts = [lly, lly, ury, ury, lly]

        assert list(xpts) == pytest.approx(expected_xpts)
        assert list(ypts) == pytest.approx(expected_ypts)
        assert poly.layer == 0

    def test_to_rect(self, box_obj: sp.Box) -> None:
        g = box_obj.toRect()
        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        assert isinstance(g.group[0], sp.Poly)

    def test_numkey_points(self, box_obj: sp.Box) -> None:
        llx, lly = box_obj.llx, box_obj.lly
        urx, ury = box_obj.urx(), box_obj.ury()
        cx = (llx + urx) / 2
        cy = (lly + ury) / 2
        expected_points = {
            1: (llx, lly),  # lower-left
            2: (cx, lly),  # lower-middle
            3: (urx, lly),  # lower-right
            4: (llx, cy),  # middle-left
            5: (cx, cy),  # center
            6: (urx, cy),  # middle-right
            7: (llx, ury),  # upper-left
            8: (cx, ury),  # upper-middle
            9: (urx, ury),  # upper-right
        }
        for numkey, (exp_x, exp_y) in expected_points.items():
            x, y = box_obj.get_numkey_point(numkey)
            assert (x, y) == pytest.approx((exp_x, exp_y))


class TestPoly:
    """
    Tests for the Poly class.

    Methods without coverage:

    * `three_point_filter`
    * `identical_to`
    * `anisotropic_resize`

    """

    def test_init_poly(self, poly_obj: sp.Poly, poly_pts: tuple[_TF, _TF]) -> None:
        # poly.data should be of the format
        # [x0, y0, x1, y1, x2, y2, x3, y3, ..., x0, y0]
        assert isinstance(poly_obj.data, np.ndarray)
        assert len(poly_obj.data) == 2 * (len(poly_pts[0]) + 1)
        assert poly_obj.Npts == len(poly_pts[0]) + 1
        assert poly_obj.layer == 2

        reshaped_data = poly_obj.data.reshape((-1, 2))
        resh_xpts, resh_ypts = reshaped_data[:, 0], reshaped_data[:, 1]
        # Close the polygon by repeating the first point at the end:
        expected_xpts = poly_pts[0] + (poly_pts[0][0],)
        expected_ypts = poly_pts[1] + (poly_pts[1][0],)
        assert list(resh_xpts) == pytest.approx(expected_xpts)
        assert list(resh_ypts) == pytest.approx(expected_ypts)

    def test_bounding_box(self, poly_obj: sp.Poly, poly_pts: tuple[_TF, _TF]) -> None:
        bb = poly_obj.bounding_box()
        xpts, ypts = poly_pts
        expected_llx = min(xpts)
        expected_lly = min(ypts)
        expected_urx = max(xpts)
        expected_ury = max(ypts)
        expected_width = expected_urx - expected_llx
        expected_height = expected_ury - expected_lly

        assert isinstance(bb, sp.Box)
        assert bb.llx == pytest.approx(expected_llx)
        assert bb.lly == pytest.approx(expected_lly)
        assert bb.urx() == pytest.approx(expected_urx)
        assert bb.ury() == pytest.approx(expected_ury)
        assert bb.width == pytest.approx(expected_width)
        assert bb.height == pytest.approx(expected_height)

    def test_area(self, poly_obj: sp.Poly) -> None:
        assert poly_obj.area() == pytest.approx(6.0)

    def test_perimeter(self, poly_obj: sp.Poly) -> None:
        assert poly_obj.perimeter() == pytest.approx(10.0)

    def test_centroid(self, poly_obj: sp.Poly) -> None:
        assert poly_obj.centroid() == pytest.approx((1.5, 1.0))

    def test_translate(self, poly_obj: sp.Poly) -> None:
        poly_obj.translate(1.0, 2.0)
        assert poly_obj.centroid() == pytest.approx((2.5, 3.0))
        assert poly_obj.area() == pytest.approx(6.0)
        assert poly_obj.perimeter() == pytest.approx(10.0)

    def test_rotate(self, poly_obj: sp.Poly) -> None:
        poly_obj.rotate(0.0, 0.0, 90.0)
        bb = poly_obj.bounding_box()
        assert bb.width == pytest.approx(2.0)
        assert bb.height == pytest.approx(3.0)
        assert poly_obj.centroid() == pytest.approx((-1.0, 1.5))
        assert poly_obj.area() == pytest.approx(6.0)
        assert poly_obj.perimeter() == pytest.approx(10.0)

        poly_obj.rotate(-1.0, 1.5, -45.0)
        bb = poly_obj.bounding_box()
        assert bb.width == pytest.approx(3.5355339059327378)
        assert bb.height == pytest.approx(3.5355339059327378)
        assert poly_obj.centroid() == pytest.approx((-1.0, 1.5))
        assert poly_obj.area() == pytest.approx(6.0)
        assert poly_obj.perimeter() == pytest.approx(10.0)

    def test_rotate_translate(self, poly_obj: sp.Poly) -> None:
        poly_obj.rotate_translate(2.0, 3.0, 180.0)
        bb = poly_obj.bounding_box()
        assert bb.width == pytest.approx(3.0)
        assert bb.height == pytest.approx(2.0)
        assert poly_obj.centroid() == pytest.approx((0.5, 2.0))
        assert poly_obj.area() == pytest.approx(6.0)
        assert poly_obj.perimeter() == pytest.approx(10.0)

    def test_scale(self, poly_obj: sp.Poly) -> None:
        poly_obj.scale(0.0, 0.0, 2.0, 3.0)
        bb = poly_obj.bounding_box()
        assert bb.width == pytest.approx(6.0)
        assert bb.height == pytest.approx(6.0)
        assert poly_obj.centroid() == pytest.approx((3.0, 3.0))
        assert poly_obj.area() == pytest.approx(36.0)
        assert poly_obj.perimeter() == pytest.approx(24.0)

    def test_mirror_x(self, poly_obj: sp.Poly) -> None:
        poly_obj.mirrorX(1.5)
        bb = poly_obj.bounding_box()
        assert bb.width == pytest.approx(3.0)
        assert bb.height == pytest.approx(2.0)
        assert poly_obj.centroid() == pytest.approx((1.5, 1.0))
        assert poly_obj.area() == pytest.approx(6.0)
        assert poly_obj.perimeter() == pytest.approx(10.0)

    def test_mirror_y(self, poly_obj: sp.Poly) -> None:
        poly_obj.mirrorY(1.0)
        bb = poly_obj.bounding_box()
        assert bb.width == pytest.approx(3.0)
        assert bb.height == pytest.approx(2.0)
        assert poly_obj.centroid() == pytest.approx((1.5, 1.0))
        assert poly_obj.area() == pytest.approx(6.0)
        assert poly_obj.perimeter() == pytest.approx(10.0)

    def test_int_data(self, poly_pts: tuple[_TF, _TF]) -> None:
        expected_xpts = poly_pts[0] + (poly_pts[0][0],)
        expected_ypts = poly_pts[1] + (poly_pts[1][0],)
        scaled_xpts = [1000 * x for x in expected_xpts]
        scaled_ypts = [1000 * y for y in expected_ypts]
        int_data = np.array([scaled_xpts, scaled_ypts], dtype=int).T.reshape(-1)
        poly = sp.Poly([], [], layer=2)
        poly.set_int_data(int_data)

        assert poly.int_data() == pytest.approx(int_data)
        assert poly.Npts == len(expected_xpts)

        reshaped_data = poly.data.reshape((-1, 2))
        resh_xpts, resh_ypts = reshaped_data[:, 0], reshaped_data[:, 1]
        assert list(resh_xpts) == pytest.approx(expected_xpts)
        assert list(resh_ypts) == pytest.approx(expected_ypts)

        bb = poly.bounding_box()
        assert bb.width == pytest.approx(3.0)
        assert bb.height == pytest.approx(2.0)
        assert poly.area() == pytest.approx(6.0)
        assert poly.perimeter() == pytest.approx(10.0)
        assert poly.centroid() == pytest.approx((1.5, 1.0))

    def test_to_polygon(self, poly_obj: sp.Poly) -> None:
        g = poly_obj.to_polygon()
        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        assert g.group[0] is poly_obj

    def test_to_circle(self, poly_obj: sp.Poly) -> None:
        # Rectangle will have a perfect circle fit
        # vcount too high -> return empty geometry group
        gg = poly_obj.to_circle(thresh=0.0, vcount=10)
        assert isinstance(gg, sp.GeomGroup)
        assert len(gg.group) == 0

        # threshold too high -> return empty geometry group
        gg = poly_obj.to_circle(thresh=1.1, vcount=4)
        assert isinstance(gg, sp.GeomGroup)
        assert len(gg.group) == 0

        # Perfect fit -> return one circle
        gg = poly_obj.to_circle(thresh=0.98, vcount=4)
        assert isinstance(gg, sp.GeomGroup)
        assert len(gg.group) == 1

        circle = gg.group[0]
        assert isinstance(circle, sp.Circle)
        assert circle.r == pytest.approx(1.802775637732)
        assert circle.x0 == pytest.approx(1.5)
        assert circle.y0 == pytest.approx(1.0)

    def test_point_inside(self, poly_obj: sp.Poly) -> None:
        # Points inside the rectangle
        assert poly_obj.point_inside(1.0, 1.0) is True
        assert poly_obj.point_inside(2.5, 0.5) is True

        # Points outside the rectangle
        assert poly_obj.point_inside(-1.0, 1.0) is False
        assert poly_obj.point_inside(4.0, 1.0) is False
        assert poly_obj.point_inside(1.5, -1.0) is False
        assert poly_obj.point_inside(1.5, 3.0) is False


class TestPath:
    def test_init_path(
        self, path_obj: sp.Path, poly_pts: tuple[_TF, _TF], path_width: float
    ) -> None:
        xpts, ypts = poly_pts
        assert path_obj.xpts == list(xpts)
        assert path_obj.ypts == list(ypts)
        assert path_obj.width == path_width
        assert path_obj.layer == 2
        assert path_obj.Npts == len(xpts)

    def test_bounding_box(self, path_obj: sp.Path, poly_pts: tuple[_TF, _TF]) -> None:
        bb = path_obj.bounding_box()
        xpts, ypts = poly_pts
        expected_llx = min(xpts)
        expected_lly = min(ypts)
        expected_urx = max(xpts)
        expected_ury = max(ypts)
        expected_width = expected_urx - expected_llx
        expected_height = expected_ury - expected_lly

        assert isinstance(bb, sp.Box)
        assert bb.llx == pytest.approx(expected_llx)
        assert bb.lly == pytest.approx(expected_lly)
        assert bb.urx() == pytest.approx(expected_urx)
        assert bb.ury() == pytest.approx(expected_ury)
        assert bb.width == pytest.approx(expected_width)
        assert bb.height == pytest.approx(expected_height)

    def test_translate(self, path_obj: sp.Path, poly_pts: tuple[_TF, _TF]) -> None:
        xpts, ypts = poly_pts
        path_obj.translate(1.0, 2.0)
        expected_xpts = [x + 1.0 for x in xpts]
        expected_ypts = [y + 2.0 for y in ypts]
        assert path_obj.xpts == pytest.approx(expected_xpts)
        assert path_obj.ypts == pytest.approx(expected_ypts)

    def test_rotate(self, path_obj: sp.Path, poly_pts: tuple[_TF, _TF]) -> None:
        xpts, ypts = poly_pts
        path_obj.rotate(0.0, 0.0, 90.0)
        expected_xpts = [-y for y in ypts]
        expected_ypts = [x for x in xpts]
        assert path_obj.xpts == pytest.approx(expected_xpts)
        assert path_obj.ypts == pytest.approx(expected_ypts)

    def test_rotate_translate(
        self, path_obj: sp.Path, poly_pts: tuple[_TF, _TF]
    ) -> None:
        xpts, ypts = poly_pts
        path_obj.rotate_translate(2.0, 3.0, 180.0)
        expected_xpts = [2.0 - x for x in xpts]
        expected_ypts = [3.0 - y for y in ypts]
        assert path_obj.xpts == pytest.approx(expected_xpts)
        assert path_obj.ypts == pytest.approx(expected_ypts)

    def test_scale(self, path_obj: sp.Path, poly_pts: tuple[_TF, _TF]) -> None:
        xpts, ypts = poly_pts
        path_obj.scale(0.0, 0.0, 2.0, 3.0)
        expected_xpts = [2.0 * x for x in xpts]
        expected_ypts = [3.0 * y for y in ypts]
        assert path_obj.xpts == pytest.approx(expected_xpts)
        assert path_obj.ypts == pytest.approx(expected_ypts)

    def test_mirror_x(self, path_obj: sp.Path, poly_pts: tuple[_TF, _TF]) -> None:
        xpts, ypts = poly_pts
        path_obj.mirrorX(1.5)
        expected_xpts = [2 * 1.5 - x for x in xpts]
        expected_ypts = list(ypts)
        assert path_obj.xpts == pytest.approx(expected_xpts)
        assert path_obj.ypts == pytest.approx(expected_ypts)

    def test_mirror_y(self, path_obj: sp.Path, poly_pts: tuple[_TF, _TF]) -> None:
        xpts, ypts = poly_pts
        path_obj.mirrorY(1.0)
        expected_xpts = list(xpts)
        expected_ypts = [2 * 1.0 - y for y in ypts]
        assert path_obj.xpts == pytest.approx(expected_xpts)
        assert path_obj.ypts == pytest.approx(expected_ypts)

    def test_path_length(self, path_obj: sp.Path) -> None:
        # Two sides of length 3 and one side of length 2 -> total length 8
        assert path_obj.path_length() == pytest.approx(8.0)

    def test_area(self, path_obj: sp.Path) -> None:
        # The area of a path is defined as the area of the polygon formed by
        # the centerline of the path and the width. For a rectangle with width
        # 0.5 and length 8, the area should be 4.
        assert path_obj.area() == pytest.approx(4.0)

    def test_perimeter(self, path_obj: sp.Path) -> None:
        # Outside perimeter of the path is approximately twice the path length
        # plus twice the path width. For a path length of 8 and width of 0.5,
        # the perimeter should be approximately 17.0.
        assert path_obj.perimeter() == pytest.approx(17.0)

    def test_to_polygon_single_points(self) -> None:
        # A path with only one point converts to a square cap polygon with side length
        # equal to the path width.
        path = sp.Path([0.0], [0.0], width=1.0, layer=2)
        g = path.to_polygon()

        # Square cap + closing point -> 5 points in total
        expected_xpts = [-0.5, 0.5, 0.5, -0.5, -0.5]
        expected_ypts = [-0.5, -0.5, 0.5, 0.5, -0.5]

        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        poly = g.group[0]
        assert isinstance(poly, sp.Poly)
        xpts = poly.data[0::2]
        ypts = poly.data[1::2]

        assert poly.layer == 2
        assert xpts == pytest.approx(expected_xpts)
        assert ypts == pytest.approx(expected_ypts)

    def test_to_polygon_line_segment(self) -> None:
        # A path with two points converts to a rectangle with length equal to the
        # distance between the two points and width equal to the path width.
        path = sp.Path([0.0, 3.0], [0.0, 0.0], width=1.0, layer=2)
        g = path.to_polygon()

        expected_xpts = [0.0, 3.0, 3.0, 0.0, 0.0]
        expected_ypts = [-0.5, -0.5, 0.5, 0.5, -0.5]

        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        poly = g.group[0]
        assert isinstance(poly, sp.Poly)
        xpts = poly.data[0::2]
        ypts = poly.data[1::2]

        assert poly.layer == 2
        assert xpts == pytest.approx(expected_xpts)
        assert ypts == pytest.approx(expected_ypts)

    def test_to_polygon_polyline(self, path_obj: sp.Path) -> None:
        # A multi-segment path converts to a single polygon outlining the stroke.
        g = path_obj.to_polygon()

        half = path_obj.width / 2
        expected_xpts = [
            0.0,
            3.0,
            3.0 + half,
            3.0 + half,
            3.0,
            0.0,
            0.0,
            3.0 - half,
            3.0 - half,
            0.0,
            0.0,
        ]
        expected_ypts = [
            -half,
            -half,
            0.0,
            2.0,
            2.0 + half,
            2.0 + half,
            2.0 - half,
            2.0 - half,
            half,
            half,
            -half,
        ]

        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        poly = g.group[0]
        assert isinstance(poly, sp.Poly)
        xpts = poly.data[0::2]
        ypts = poly.data[1::2]

        assert poly.layer == path_obj.layer
        assert xpts == pytest.approx(expected_xpts)
        assert ypts == pytest.approx(expected_ypts)


class TestText:
    def test_init_text(self, text_obj: sp.Text) -> None:
        assert text_obj.x0 == 1.0
        assert text_obj.y0 == 2.0
        assert text_obj.text == "Hello, world!"
        assert text_obj.posu == 0
        assert text_obj.posv == 0
        assert text_obj.height == 10.0
        assert text_obj.width == 1.0
        assert text_obj.angle == 0.0
        assert text_obj.layer == 3

    def test_translate(self, text_obj: sp.Text) -> None:
        text_obj.translate(1.0, 2.0)
        assert text_obj.x0 == pytest.approx(2.0)
        assert text_obj.y0 == pytest.approx(4.0)

    def test_rotate(self, text_obj: sp.Text) -> None:
        text_obj.rotate(0.0, 0.0, 90.0)
        assert text_obj.x0 == pytest.approx(-2.0)
        assert text_obj.y0 == pytest.approx(1.0)
        assert text_obj.angle == pytest.approx(90.0)

    def test_rotate_translate(self, text_obj: sp.Text) -> None:
        text_obj.rotate_translate(2.0, 3.0, 180.0)
        assert text_obj.x0 == pytest.approx(1.0)
        assert text_obj.y0 == pytest.approx(1.0)
        assert text_obj.angle == pytest.approx(180.0)

    def test_scale(self, text_obj: sp.Text) -> None:
        text_obj.scale(0.0, 0.0, 2.0, 3.0)
        assert text_obj.x0 == pytest.approx(2.0)
        assert text_obj.y0 == pytest.approx(6.0)
        assert text_obj.height == pytest.approx(30.0)
        assert text_obj.width == pytest.approx(2.0)

    def test_mirror_x(self, text_obj: sp.Text) -> None:
        text_obj.mirrorX(1.0)
        assert text_obj.x0 == pytest.approx(1.0)
        assert text_obj.y0 == pytest.approx(2.0)
        assert text_obj.angle == pytest.approx(180.0)

    def test_mirror_y(self, text_obj: sp.Text) -> None:
        text_obj.mirrorY(1.0)
        assert text_obj.x0 == pytest.approx(1.0)
        assert text_obj.y0 == pytest.approx(0.0)
        assert text_obj.angle == pytest.approx(0.0)

    def test_bounding_box(self, text_obj: sp.Text) -> None:
        bb = text_obj.bounding_box()
        assert isinstance(bb, sp.Box)
        # Since text isn't a geometric shape, we define its bounding box to be a single
        # point at the text origin (x0, y0).
        assert bb.cx() == pytest.approx(text_obj.x0)
        assert bb.cy() == pytest.approx(text_obj.y0)
        assert bb.width == 0
        assert bb.height == 0

    def test_area(self, text_obj: sp.Text) -> None:
        # Text doesn't have a well-defined area, so we return 0.
        assert text_obj.area() == 0

    def test_centroid(self, text_obj: sp.Text) -> None:
        # The centroid of the text is defined to be its origin (x0, y0).
        assert text_obj.centroid() == (text_obj.x0, text_obj.y0)

    def test_perimeter(self, text_obj: sp.Text) -> None:
        # Text doesn't have a well-defined perimeter, so we return 0.
        assert text_obj.perimeter() == 0

    def test_to_polygon(self, text_obj: sp.Text) -> None:
        # Text cannot be converted to a polygon, so we return an empty geometry group.
        g = text_obj.to_polygon()
        assert isinstance(g, sp.GeomGroup)
        assert all(isinstance(p, sp.Poly) for p in g.group)


class TestSRef:
    def test_init_sref(
        self,
        sref_obj: sp.SRef,
        geomgroup_obj: sp.GeomGroup,
        sref_kwargs: SRefKwargs,
    ) -> None:
        assert sref_obj.x0 == sref_kwargs.x0
        assert sref_obj.y0 == sref_kwargs.y0
        assert sref_obj.cellname == sref_kwargs.cellname
        assert sref_obj.mag == sref_kwargs.mag
        assert sref_obj.angle == sref_kwargs.angle
        assert sref_obj.mirror == sref_kwargs.mirror

        g = sref_obj.group
        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        assert g is geomgroup_obj

    @pytest.mark.parametrize("xoff, yoff", [(1.0, 2.0), (-1.0, -2.0), (0.5, -0.5)])
    def test_translate(
        self, sref_obj: sp.SRef, sref_kwargs: SRefKwargs, xoff: float, yoff: float
    ) -> None:
        sref_obj.translate(xoff, yoff)
        expected_x0 = sref_kwargs.x0 + xoff
        expected_y0 = sref_kwargs.y0 + yoff
        assert sref_obj.x0 == pytest.approx(expected_x0)
        assert sref_obj.y0 == pytest.approx(expected_y0)

    def test_rotate(self, sref_obj: sp.SRef, sref_kwargs: SRefKwargs) -> None:
        sref_obj.rotate(0.0, 0.0, 90.0)
        expected_x0 = -sref_kwargs.y0
        expected_y0 = sref_kwargs.x0
        expected_angle = (sref_kwargs.angle + 90.0) % 360
        assert sref_obj.x0 == pytest.approx(expected_x0)
        assert sref_obj.y0 == pytest.approx(expected_y0)
        assert sref_obj.angle == pytest.approx(expected_angle)

    def test_rotate_translate(self, sref_obj: sp.SRef, sref_kwargs: SRefKwargs) -> None:
        sref_obj.rotate_translate(2.0, 3.0, 180.0)
        expected_x0 = -sref_kwargs.x0 + 2.0
        expected_y0 = -sref_kwargs.y0 + 3.0
        expected_angle = (sref_kwargs.angle + 180.0) % 360
        assert sref_obj.x0 == pytest.approx(expected_x0)
        assert sref_obj.y0 == pytest.approx(expected_y0)
        assert sref_obj.angle == pytest.approx(expected_angle)

    @pytest.mark.parametrize("sx, sy", [(2.0, 3.0), (0.5, 0.5), (1.0, 1.0)])
    def test_scale(
        self, sref_obj: sp.SRef, sref_kwargs: SRefKwargs, sx: float, sy: float
    ) -> None:
        sref_obj.scale(0.0, 0.0, sx, sy)
        expected_x0 = sref_kwargs.x0 * sx
        expected_y0 = sref_kwargs.y0 * sy
        assert sref_obj.x0 == pytest.approx(expected_x0)
        assert sref_obj.y0 == pytest.approx(expected_y0)
        assert sref_obj.mag == pytest.approx(sref_kwargs.mag * sx)

    @pytest.mark.parametrize("xc", [0.0, 1.0, 2.0])
    def test_mirror_x(
        self, sref_obj: sp.SRef, sref_kwargs: SRefKwargs, xc: float
    ) -> None:
        sref_obj.mirrorX(xc)
        expected_x0 = 2 * xc - sref_kwargs.x0
        expected_y0 = sref_kwargs.y0
        expected_angle = (180.0 - sref_kwargs.angle) % 360
        assert sref_obj.x0 == pytest.approx(expected_x0)
        assert sref_obj.y0 == pytest.approx(expected_y0)
        assert sref_obj.angle == pytest.approx(expected_angle)

    def test_mirror_x_toggles_mirror_flag(self, sref_obj: sp.SRef) -> None:
        assert sref_obj.mirror is False
        sref_obj.mirrorX(0.0)
        assert sref_obj.mirror is True
        sref_obj.mirrorX(0.0)
        assert sref_obj.mirror is False

    @pytest.mark.parametrize("yc", [0.0, 1.0, 2.0])
    def test_mirror_y(
        self, sref_obj: sp.SRef, sref_kwargs: SRefKwargs, yc: float
    ) -> None:
        sref_obj.mirrorY(yc)
        expected_x0 = sref_kwargs.x0
        expected_y0 = 2 * yc - sref_kwargs.y0
        expected_angle = (-sref_kwargs.angle) % 360
        assert sref_obj.x0 == pytest.approx(expected_x0)
        assert sref_obj.y0 == pytest.approx(expected_y0)
        assert sref_obj.angle == pytest.approx(expected_angle)

    def test_mirror_y_toggles_mirror_flag(self, sref_obj: sp.SRef) -> None:
        assert sref_obj.mirror is False
        sref_obj.mirrorY(0.0)
        assert sref_obj.mirror is True
        sref_obj.mirrorY(0.0)
        assert sref_obj.mirror is False

    @pytest.mark.parametrize("xoff, yoff", [(1.0, 2.0), (-1.0, -2.0), (0.5, -0.5)])
    def test_centroid(
        self, sref_obj: sp.SRef, geomgroup_obj: sp.GeomGroup, xoff: float, yoff: float
    ) -> None:
        expected_x0 = sref_obj.x0 + xoff
        expected_y0 = sref_obj.y0 + yoff
        sref_obj.translate(xoff, yoff)
        assert sref_obj.centroid() == pytest.approx((expected_x0, expected_y0))

    @pytest.mark.parametrize("xoff, yoff", [(1.0, 2.0), (-1.0, -2.0), (0.5, -0.5)])
    def test_bounding_box(
        self, sref_obj: sp.SRef, geomgroup_obj: sp.GeomGroup, xoff: float, yoff: float
    ) -> None:
        sref_obj.translate(xoff, yoff)
        bb = sref_obj.bounding_box()
        assert isinstance(bb, sp.Box)
        ref_bb = geomgroup_obj.bounding_box()
        expected_cx = sref_obj.x0 + ref_bb.cx()
        expected_cy = sref_obj.y0 + ref_bb.cy()
        expected_width = ref_bb.width * sref_obj.mag
        expected_height = ref_bb.height * sref_obj.mag

        assert bb.cx() == pytest.approx(expected_cx)
        assert bb.cy() == pytest.approx(expected_cy)
        assert bb.width == pytest.approx(expected_width)
        assert bb.height == pytest.approx(expected_height)

    @pytest.mark.parametrize(
        "mag, angle, mirror",
        [(1.0, 0.0, False), (2.0, 90.0, False)],
    )
    def test_bounding_box_matches_placed_group(
        self,
        sref_obj: sp.SRef,
        geomgroup_obj: sp.GeomGroup,
        mag: float,
        angle: float,
        mirror: bool,
    ) -> None:
        sref_obj.mag = mag
        sref_obj.angle = angle
        sref_obj.mirror = mirror

        gflat = geomgroup_obj.flatten()
        placed = sref_obj.place_group(gflat.copy())

        sref_bb = sref_obj.bounding_box()
        placed_bb = placed.bounding_box()

        assert sref_bb.cx() == pytest.approx(placed_bb.cx())
        assert sref_bb.cy() == pytest.approx(placed_bb.cy())
        assert sref_bb.width == pytest.approx(placed_bb.width)
        assert sref_bb.height == pytest.approx(placed_bb.height)

    def test_bounding_box_uses_cached_pool_box(
        self, geomgroup_obj: sp.GeomGroup, sref_kwargs: SRefKwargs
    ) -> None:
        cellname = sref_kwargs.cellname
        pool_box = sp.Box(10.0, 20.0, 4.0, 6.0)
        sp._BoundingBoxPool[cellname] = pool_box
        sref = sp.SRef(
            x0=sref_kwargs.x0,
            y0=sref_kwargs.y0,
            cellname=cellname,
            group=geomgroup_obj,
            mag=6.0,
            angle=0.0,
            mirror=False,
        )

        bb = sref.bounding_box()
        assert bb.width == pytest.approx(pool_box.width * sref.mag)
        assert bb.height == pytest.approx(pool_box.height * sref.mag)
        assert bb.cx() == pytest.approx(sref.x0 + pool_box.cx() * sref.mag)
        assert bb.cy() == pytest.approx(sref.y0 + pool_box.cy() * sref.mag)

    @pytest.mark.xfail(
        strict=True,
        reason="Known mismatch: SRef.bounding_box does not match placed "
        "mirrored/rotated geometry.",
    )
    def test_bounding_box_mismatch_for_mirrored_rotated_sref_is_documented(
        self, sref_obj: sp.SRef, geomgroup_obj: sp.GeomGroup
    ) -> None:
        sref_obj.mag = 2.0
        sref_obj.angle = 45.0
        sref_obj.mirror = True

        placed = sref_obj.place_group(geomgroup_obj.flatten())
        sref_bb = sref_obj.bounding_box()
        placed_bb = placed.bounding_box()

        assert sref_bb.cx() == pytest.approx(placed_bb.cx())
        assert sref_bb.cy() == pytest.approx(placed_bb.cy())
        assert sref_bb.width == pytest.approx(placed_bb.width)
        assert sref_bb.height == pytest.approx(placed_bb.height)


class TestAref:
    def test_init_aref(
        self,
        aref_obj: sp.ARef,
        geomgroup_obj: sp.GeomGroup,
        aref_kwargs: ARefKwargs,
    ) -> None:
        assert aref_obj.x0 == aref_kwargs.x0
        assert aref_obj.y0 == aref_kwargs.y0
        assert aref_obj.cellname == aref_kwargs.cellname
        assert aref_obj.ncols == aref_kwargs.ncols
        assert aref_obj.nrows == aref_kwargs.nrows
        assert aref_obj.ax == aref_kwargs.ax
        assert aref_obj.ay == aref_kwargs.ay
        assert aref_obj.bx == aref_kwargs.bx
        assert aref_obj.by == aref_kwargs.by
        assert aref_obj.mag == aref_kwargs.mag
        assert aref_obj.angle == aref_kwargs.angle
        assert aref_obj.mirror == aref_kwargs.mirror

        g = aref_obj.group
        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        assert g is geomgroup_obj

    @pytest.mark.parametrize("xoff, yoff", [(1.0, 2.0), (-1.0, -2.0), (0.5, -0.5)])
    def test_translate(
        self, aref_obj: sp.ARef, aref_kwargs: ARefKwargs, xoff: float, yoff: float
    ) -> None:
        aref_obj.translate(xoff, yoff)
        assert aref_obj.x0 == pytest.approx(aref_kwargs.x0 + xoff)
        assert aref_obj.y0 == pytest.approx(aref_kwargs.y0 + yoff)

    def test_rotate(self, aref_obj: sp.ARef, aref_kwargs: ARefKwargs) -> None:
        aref_obj.rotate(0.0, 0.0, 90.0)
        assert aref_obj.x0 == pytest.approx(-aref_kwargs.y0)
        assert aref_obj.y0 == pytest.approx(aref_kwargs.x0)
        assert aref_obj.angle == pytest.approx((aref_kwargs.angle + 90.0) % 360)

    def test_rotate_translate(self, aref_obj: sp.ARef, aref_kwargs: ARefKwargs) -> None:
        aref_obj.rotate_translate(2.0, 3.0, 180.0)
        assert aref_obj.x0 == pytest.approx(-aref_kwargs.x0 + 2.0)
        assert aref_obj.y0 == pytest.approx(-aref_kwargs.y0 + 3.0)
        assert aref_obj.angle == pytest.approx((aref_kwargs.angle + 180.0) % 360)

    @pytest.mark.parametrize("sx, sy", [(2.0, 3.0), (0.5, 0.5), (1.0, 1.0)])
    def test_scale(
        self, aref_obj: sp.ARef, aref_kwargs: ARefKwargs, sx: float, sy: float
    ) -> None:
        aref_obj.scale(0.0, 0.0, sx, sy)
        assert aref_obj.x0 == pytest.approx(aref_kwargs.x0 * sx)
        assert aref_obj.y0 == pytest.approx(aref_kwargs.y0 * sy)
        assert aref_obj.mag == pytest.approx(aref_kwargs.mag * sx)

    @pytest.mark.parametrize("xc", [0.0, 1.0, 2.0])
    def test_mirror_x(
        self, aref_obj: sp.ARef, aref_kwargs: ARefKwargs, xc: float
    ) -> None:
        aref_obj.mirrorX(xc)
        assert aref_obj.x0 == pytest.approx(2 * xc - aref_kwargs.x0)
        assert aref_obj.y0 == pytest.approx(aref_kwargs.y0)
        assert aref_obj.angle == pytest.approx((180.0 - aref_kwargs.angle) % 360)

    def test_mirror_x_toggles_mirror_flag(self, aref_obj: sp.ARef) -> None:
        assert aref_obj.mirror is False
        aref_obj.mirrorX(0.0)
        assert aref_obj.mirror is True
        aref_obj.mirrorX(0.0)
        assert aref_obj.mirror is False

    @pytest.mark.parametrize("yc", [0.0, 1.0, 2.0])
    def test_mirror_y(
        self, aref_obj: sp.ARef, aref_kwargs: ARefKwargs, yc: float
    ) -> None:
        aref_obj.mirrorY(yc)
        assert aref_obj.x0 == pytest.approx(aref_kwargs.x0)
        assert aref_obj.y0 == pytest.approx(2 * yc - aref_kwargs.y0)
        assert aref_obj.angle == pytest.approx((-aref_kwargs.angle) % 360)

    def test_mirror_y_toggles_mirror_flag(self, aref_obj: sp.ARef) -> None:
        assert aref_obj.mirror is False
        aref_obj.mirrorY(0.0)
        assert aref_obj.mirror is True
        aref_obj.mirrorY(0.0)
        assert aref_obj.mirror is False

    def test_centroid(self, aref_obj: sp.ARef, aref_kwargs: ARefKwargs) -> None:
        expected_centroid = (aref_kwargs.x0, aref_kwargs.y0)
        assert aref_obj.centroid() == pytest.approx(expected_centroid)

    def test_bounding_box(
        self,
        aref_obj: sp.ARef,
        geomgroup_obj: sp.GeomGroup,
        aref_kwargs: ARefKwargs,
    ) -> None:
        bb = aref_obj.bounding_box()
        ref_bb = geomgroup_obj.bounding_box()
        llx = aref_kwargs.x0 + ref_bb.llx
        lly = aref_kwargs.y0 + ref_bb.lly
        urx = (
            aref_kwargs.x0
            + ref_bb.urx()
            + (aref_kwargs.ncols - 1) * aref_kwargs.ax
            + (aref_kwargs.nrows - 1) * aref_kwargs.bx
        )
        ury = (
            aref_kwargs.y0
            + ref_bb.ury()
            + (aref_kwargs.ncols - 1) * aref_kwargs.ay
            + (aref_kwargs.nrows - 1) * aref_kwargs.by
        )

        assert isinstance(bb, sp.Box)
        assert bb.llx == pytest.approx(llx)
        assert bb.lly == pytest.approx(lly)
        assert bb.urx() == pytest.approx(urx)
        assert bb.ury() == pytest.approx(ury)

    @pytest.mark.parametrize(
        "mag, angle, mirror",
        [(1.0, 0.0, False), (2.0, 90.0, False)],
    )
    def test_bounding_box_matches_placed_group(
        self,
        aref_obj: sp.ARef,
        geomgroup_obj: sp.GeomGroup,
        mag: float,
        angle: float,
        mirror: bool,
    ) -> None:
        aref_obj.mag = mag
        aref_obj.angle = angle
        aref_obj.mirror = mirror

        gflat = geomgroup_obj.flatten()
        placed = aref_obj.place_group(gflat.copy())

        aref_bb = aref_obj.bounding_box()
        placed_bb = placed.bounding_box()

        assert aref_bb.cx() == pytest.approx(placed_bb.cx())
        assert aref_bb.cy() == pytest.approx(placed_bb.cy())
        assert aref_bb.width == pytest.approx(placed_bb.width)
        assert aref_bb.height == pytest.approx(placed_bb.height)

    def test_place_group_makes_expected_number_of_copies(
        self,
        aref_obj: sp.ARef,
        geomgroup_obj: sp.GeomGroup,
        aref_kwargs: ARefKwargs,
    ) -> None:
        placed = aref_obj.place_group(geomgroup_obj.flatten())
        assert len(placed.group) == aref_kwargs.ncols * aref_kwargs.nrows

    @pytest.mark.xfail(
        strict=True,
        reason="Known mismatch: ARef.bounding_box does not match placed "
        "mirrored/rotated array geometry.",
    )
    def test_bounding_box_mismatch_for_mirrored_rotated_array_is_documented(
        self, aref_obj: sp.ARef, geomgroup_obj: sp.GeomGroup
    ) -> None:
        aref_obj.mag = 2.0
        aref_obj.angle = 45.0
        aref_obj.mirror = True

        placed = aref_obj.place_group(geomgroup_obj.flatten())
        aref_bb = aref_obj.bounding_box()
        placed_bb = placed.bounding_box()

        assert aref_bb.cx() == pytest.approx(placed_bb.cx())
        assert aref_bb.cy() == pytest.approx(placed_bb.cy())
        assert aref_bb.width == pytest.approx(placed_bb.width)
        assert aref_bb.height == pytest.approx(placed_bb.height)


class TestCircle:
    def test_init_circle(self, circle_obj: sp.Circle) -> None:
        assert circle_obj.x0 == 1.0
        assert circle_obj.y0 == 2.0
        assert circle_obj.r == 3.0
        assert circle_obj.layer == 4

    def test_translate(self, circle_obj: sp.Circle) -> None:
        circle_obj.translate(2.0, -1.0)

        assert circle_obj.x0 == pytest.approx(3.0)
        assert circle_obj.y0 == pytest.approx(1.0)

    def test_rotate(self, circle_obj: sp.Circle) -> None:
        circle_obj.rotate(0.0, 0.0, 90.0)

        assert circle_obj.x0 == pytest.approx(-2.0)
        assert circle_obj.y0 == pytest.approx(1.0)

    def test_rotate_translate(self, circle_obj: sp.Circle) -> None:
        circle_obj.rotate_translate(2.0, 3.0, 180.0)

        assert circle_obj.x0 == pytest.approx(1.0)
        assert circle_obj.y0 == pytest.approx(1.0)

    def test_scale(self, circle_obj: sp.Circle) -> None:
        circle_obj.scale(0.0, 0.0, 2.0, 3.0)

        assert circle_obj.x0 == pytest.approx(2.0)
        assert circle_obj.y0 == pytest.approx(6.0)
        assert circle_obj.r == pytest.approx(6.0)

    def test_mirror_x(self, circle_obj: sp.Circle) -> None:
        circle_obj.mirrorX(0.0)

        assert circle_obj.x0 == pytest.approx(-1.0)
        assert circle_obj.y0 == pytest.approx(2.0)

    def test_mirror_y(self, circle_obj: sp.Circle) -> None:
        circle_obj.mirrorY(0.0)

        assert circle_obj.x0 == pytest.approx(1.0)
        assert circle_obj.y0 == pytest.approx(-2.0)

    def test_bounding_box(self, circle_obj: sp.Circle) -> None:
        bb = circle_obj.bounding_box()

        assert isinstance(bb, sp.Box)
        assert bb.llx == pytest.approx(-2.0)
        assert bb.lly == pytest.approx(-1.0)
        assert bb.width == pytest.approx(6.0)
        assert bb.height == pytest.approx(6.0)

    def test_area(self, circle_obj: sp.Circle) -> None:
        assert circle_obj.area() == pytest.approx(np.pi * 9.0)

    def test_centroid(self, circle_obj: sp.Circle) -> None:
        assert circle_obj.centroid() == pytest.approx((1.0, 2.0))

    def test_perimeter(self, circle_obj: sp.Circle) -> None:
        assert circle_obj.perimeter() == pytest.approx(2.0 * np.pi * 3.0)

    def test_to_polygon(self, circle_obj: sp.Circle) -> None:
        g = circle_obj.to_polygon(Npts=12)

        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1

        poly = g.group[0]
        assert isinstance(poly, sp.Poly)
        assert poly.layer == circle_obj.layer
        assert poly.Npts == 13


class TestEllipse:
    def test_init_ellipse(self, ellipse_obj: sp.Ellipse) -> None:
        assert ellipse_obj.x0 == 1.0
        assert ellipse_obj.y0 == 2.0
        assert ellipse_obj.r == 3.0
        assert ellipse_obj.r1 == 2.0
        assert ellipse_obj.rot == 0.0
        assert ellipse_obj.layer == 4

    def test_rotate_translate(self, ellipse_obj: sp.Ellipse) -> None:
        ellipse_obj.rotate_translate(2.0, 3.0, 180.0)

        assert ellipse_obj.x0 == pytest.approx(1.0)
        assert ellipse_obj.y0 == pytest.approx(1.0)
        assert ellipse_obj.rot == pytest.approx(180.0)

    def test_rotate(self, ellipse_obj: sp.Ellipse) -> None:
        ellipse_obj.rotate(0.0, 0.0, 90.0)

        assert ellipse_obj.x0 == pytest.approx(-2.0)
        assert ellipse_obj.y0 == pytest.approx(1.0)
        assert ellipse_obj.rot == pytest.approx(90.0)

    def test_scale(self, ellipse_obj: sp.Ellipse) -> None:
        ellipse_obj.scale(0.0, 0.0, 2.0, 3.0)

        assert ellipse_obj.x0 == pytest.approx(2.0)
        assert ellipse_obj.y0 == pytest.approx(6.0)
        assert ellipse_obj.r == pytest.approx(6.0)
        assert ellipse_obj.r1 == pytest.approx(6.0)

    def test_mirror_x(self, ellipse_obj: sp.Ellipse) -> None:
        ellipse_obj.rot = 30.0
        ellipse_obj.mirrorX(0.0)

        assert ellipse_obj.x0 == pytest.approx(-1.0)
        assert ellipse_obj.y0 == pytest.approx(2.0)
        assert ellipse_obj.rot == pytest.approx(150.0)

    def test_mirror_y(self, ellipse_obj: sp.Ellipse) -> None:
        ellipse_obj.rot = 30.0
        ellipse_obj.mirrorY(0.0)

        assert ellipse_obj.x0 == pytest.approx(1.0)
        assert ellipse_obj.y0 == pytest.approx(-2.0)
        assert ellipse_obj.rot == pytest.approx(-30.0)

    def test_bounding_box(self, ellipse_obj: sp.Ellipse) -> None:
        bb = ellipse_obj.bounding_box()

        assert isinstance(bb, sp.Box)
        assert bb.llx == pytest.approx(-2.0)
        assert bb.lly == pytest.approx(0.0)
        assert bb.width == pytest.approx(6.0)
        assert bb.height == pytest.approx(4.0)

    def test_area(self, ellipse_obj: sp.Ellipse) -> None:
        assert ellipse_obj.area() == pytest.approx(np.pi * 3.0 * 2.0)

    def test_perimeter(self, ellipse_obj: sp.Ellipse) -> None:
        a = 3.0
        b = 2.0
        expected = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

        assert ellipse_obj.perimeter() == pytest.approx(expected)

    def test_to_polygon(self, ellipse_obj: sp.Ellipse) -> None:
        g = ellipse_obj.to_polygon(Npts=16)

        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        assert isinstance(g.group[0], sp.Poly)
        assert g.group[0].layer == ellipse_obj.layer


class TestRing:
    def test_init_ring(self, ring_obj: sp.Ring) -> None:
        assert ring_obj.x0 == 1.0
        assert ring_obj.y0 == 2.0
        assert ring_obj.r == 3.0
        assert ring_obj.r1 == 2.0
        assert ring_obj.w == 1.0
        assert ring_obj.rot == 0.0
        assert ring_obj.layer == 4

    def test_scale(self, ring_obj: sp.Ring) -> None:
        ring_obj.scale(0.0, 0.0, 2.0, 3.0)

        assert ring_obj.x0 == pytest.approx(2.0)
        assert ring_obj.y0 == pytest.approx(6.0)
        assert ring_obj.r == pytest.approx(6.0)
        assert ring_obj.r1 == pytest.approx(6.0)
        assert ring_obj.w == pytest.approx(2.0)

    def test_bounding_box(self, ring_obj: sp.Ring) -> None:
        bb = ring_obj.bounding_box()

        assert isinstance(bb, sp.Box)
        assert bb.llx == pytest.approx(-2.5)
        assert bb.lly == pytest.approx(-0.5)
        assert bb.width == pytest.approx(7.0)
        assert bb.height == pytest.approx(5.0)

    def test_area(self, ring_obj: sp.Ring) -> None:
        outer = np.pi * (ring_obj.r + ring_obj.w / 2) * (ring_obj.r1 + ring_obj.w / 2)
        inner = np.pi * (ring_obj.r - ring_obj.w / 2) * (ring_obj.r1 - ring_obj.w / 2)

        assert ring_obj.area() == pytest.approx(outer - inner)

    def test_perimeter(self, ring_obj: sp.Ring) -> None:
        poly = ring_obj.to_polygon(12).group[0]

        assert ring_obj.perimeter() == pytest.approx(poly.perimeter())

    def test_to_polygon(self, ring_obj: sp.Ring) -> None:
        g = ring_obj.to_polygon(Npts=12)

        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1

        poly = g.group[0]

        assert isinstance(poly, sp.Poly)
        assert poly.layer == ring_obj.layer
        assert poly.Npts == 27


class TestArc:
    def test_init_arc(self, arc_obj: sp.Arc) -> None:
        assert arc_obj.x0 == 1.0
        assert arc_obj.y0 == 2.0
        assert arc_obj.r == 3.0
        assert arc_obj.r1 == 2.0
        assert arc_obj.w == 1.0
        assert arc_obj.rot == 0.0
        assert arc_obj.a1 == 0.0
        assert arc_obj.a2 == 180.0
        assert arc_obj.layer == 4

    def test_bounding_box(self, arc_obj: sp.Arc) -> None:
        bb = arc_obj.bounding_box()

        assert isinstance(bb, sp.Box)
        assert bb.width > 0
        assert bb.height > 0

    def test_area(self, arc_obj: sp.Arc) -> None:
        ring_area = sp.Ring.area(arc_obj)
        expected = (
            ring_area * (np.radians(arc_obj.a2) - np.radians(arc_obj.a1)) / (2 * np.pi)
        )
        assert arc_obj.area() == pytest.approx(expected)

    def test_centroid(self, arc_obj: sp.Arc) -> None:
        poly = arc_obj.to_polygon(12).group[0]

        assert arc_obj.centroid() == pytest.approx(poly.centroid())

    def test_to_polygon(self, arc_obj: sp.Arc) -> None:
        g = arc_obj.to_polygon(Npts=16, autosplit=False)

        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        assert isinstance(g.group[0], sp.Poly)
        assert g.group[0].layer == arc_obj.layer

    def test_to_polygon_autosplit(self, arc_obj: sp.Arc) -> None:
        n_segments = 8
        g = arc_obj.to_polygon(Npts=n_segments, autosplit=True)

        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == n_segments
        assert all(isinstance(poly, sp.Poly) for poly in g.group)
        assert all(poly.layer == arc_obj.layer for poly in g.group)


class TestGeomGroup:
    def test_init_geomgroup(self) -> None:
        g = sp.GeomGroup()
        assert len(g.group) == 0

    def test_add_to_geomgroup(self, circle_obj: sp.Circle) -> None:
        g = sp.GeomGroup()
        g.add(circle_obj)

        assert len(g.group) == 1
        assert g.group[0] is circle_obj

    def test_add_two_geomgroups(
        self, circle_obj: sp.Circle, ellipse_obj: sp.Ellipse
    ) -> None:
        g1 = sp.GeomGroup()
        g1.add(circle_obj)

        g2 = sp.GeomGroup()
        g2.add(ellipse_obj)

        g = g1 + g2
        assert len(g.group) == 2
        assert g.group[0] is circle_obj
        assert g.group[1] is ellipse_obj

    def test_copy_geomgroup(self, circle_obj: sp.Circle) -> None:
        # Geometry copy creates a deep copy of the geometry, so that modifying the copy
        # does not affect the original.
        g = sp.GeomGroup()
        g.add(circle_obj)

        gcopy = g.copy()
        assert gcopy is not g
        assert len(gcopy.group) == 1
        assert isinstance(gcopy.group[0], sp.Circle)

        circle_copy = gcopy.group[0]
        assert circle_copy is not circle_obj
        assert circle_copy.x0 == circle_obj.x0
        assert circle_copy.y0 == circle_obj.y0
        assert circle_copy.r == circle_obj.r
        assert circle_copy.layer == circle_obj.layer

    def test_flatten_geomgroup(self, circle_obj: sp.Circle, sref_obj: sp.SRef) -> None:
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(sref_obj)

        gflat = g.flatten()

        assert len(gflat.group) == 2
        assert isinstance(gflat.group[0], sp.Circle)
        assert isinstance(gflat.group[1], sp.Poly)

        circle_copy = gflat.group[0]
        assert circle_copy is not circle_obj
        assert circle_copy.x0 == circle_obj.x0
        assert circle_copy.y0 == circle_obj.y0
        assert circle_copy.r == circle_obj.r
        assert circle_copy.layer == circle_obj.layer

        poly_obj = sref_obj.group.group[0]
        poly_copy = gflat.group[1]
        placed_poly = sref_obj.place_group(sref_obj.group.copy()).group[0]

        assert poly_copy is not poly_obj
        assert poly_copy.layer == placed_poly.layer
        assert poly_copy.Npts == placed_poly.Npts
        assert poly_copy.data == pytest.approx(placed_poly.data)

    def test_get_sref_list(self, circle_obj: sp.Circle, sref_obj: sp.SRef) -> None:
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(sref_obj)

        srefs = g.get_sref_list()

        assert isinstance(srefs, set)
        assert all(isinstance(s, str) for s in srefs)

        assert len(srefs) == 1
        assert sref_obj.cellname in srefs

    def test_get_layer_list(
        self, circle_obj: sp.Circle, ellipse_obj: sp.Ellipse
    ) -> None:
        circle_obj.layer = 11
        ellipse_obj.layer = 12
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(ellipse_obj)

        layers = g.get_layer_list()

        assert isinstance(layers, set)
        assert all(isinstance(layer, int) for layer in layers)

        assert len(layers) == 2
        assert 11 in layers
        assert 12 in layers

    def test_translate_rotate_scale_and_mirror_operations(
        self, circle_obj: sp.Circle, ellipse_obj: sp.Ellipse
    ) -> None:
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(ellipse_obj)

        g.translate(1.0, -1.0)
        assert circle_obj.x0 == pytest.approx(2.0)
        assert circle_obj.y0 == pytest.approx(1.0)
        assert ellipse_obj.x0 == pytest.approx(2.0)
        assert ellipse_obj.y0 == pytest.approx(1.0)

        g.rotate(0.0, 0.0, 90.0)
        assert circle_obj.x0 == pytest.approx(-1.0)
        assert circle_obj.y0 == pytest.approx(2.0)

        g.scale(0.0, 0.0, 2.0, 2.0)
        assert circle_obj.x0 == pytest.approx(-2.0)
        assert circle_obj.y0 == pytest.approx(4.0)
        assert circle_obj.r == pytest.approx(6.0)

        g.mirrorX(0.0)
        assert circle_obj.x0 == pytest.approx(2.0)
        assert circle_obj.y0 == pytest.approx(4.0)

        g.mirrorY(0.0)
        assert circle_obj.x0 == pytest.approx(2.0)
        assert circle_obj.y0 == pytest.approx(-4.0)

    def test_rotate_translate_operation(
        self, circle_obj: sp.Circle, ellipse_obj: sp.Ellipse
    ) -> None:
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(ellipse_obj)

        g.rotate_translate(2.0, 3.0, 180.0)

        assert circle_obj.x0 == pytest.approx(1.0)
        assert circle_obj.y0 == pytest.approx(1.0)
        assert ellipse_obj.x0 == pytest.approx(1.0)
        assert ellipse_obj.y0 == pytest.approx(1.0)

    def test_str_contains_group_size_and_layers(
        self, circle_obj: sp.Circle, ellipse_obj: sp.Ellipse
    ) -> None:
        circle_obj.layer = 71
        ellipse_obj.layer = 72
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(ellipse_obj)

        msg = str(g)

        assert "GeomGroup" in msg
        assert "Layers:" in msg

    def test_info_contains_bounding_box_and_counts(
        self, circle_obj: sp.Circle, ellipse_obj: sp.Ellipse
    ) -> None:
        circle_obj.layer = 51
        ellipse_obj.layer = 52
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(ellipse_obj)

        info = g.info()

        assert "BoundingBox" in info
        assert "LayerList" in info
        assert "TotalCount" in info
        assert set(info["LayerList"]) == {51, 52}
        assert info["TotalCount"]["NCircle"] == 1
        assert info["TotalCount"]["NEllipse"] == 1

    def test_bounding_box_combines_shapes(
        self, circle_obj: sp.Circle, ellipse_obj: sp.Ellipse
    ) -> None:
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(ellipse_obj)

        bb = g.bounding_box()
        cbb = circle_obj.bounding_box()
        ebb = ellipse_obj.bounding_box()

        assert bb.llx == pytest.approx(min(cbb.llx, ebb.llx))
        assert bb.lly == pytest.approx(min(cbb.lly, ebb.lly))
        assert bb.urx() == pytest.approx(max(cbb.urx(), ebb.urx()))
        assert bb.ury() == pytest.approx(max(cbb.ury(), ebb.ury()))

    def test_to_boxes_and_set_layer(
        self, circle_obj: sp.Circle, ellipse_obj: sp.Ellipse
    ) -> None:
        circle_obj.layer = 4
        ellipse_obj.layer = 5
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(ellipse_obj)

        boxes = g.to_boxes(layer=6)

        assert isinstance(boxes, sp.GeomGroup)
        assert len(boxes.group) == 0

        g.set_layer(3)
        assert all(geom.layer == 3 for geom in g.group)

        boxes = g.to_boxes(layer=3)
        assert len(boxes.group) == 2
        assert all(isinstance(geom, sp.Poly) for geom in boxes.group)
        assert all(geom.layer == 3 for geom in boxes.group)

    def test_select_layers(
        self, circle_obj: sp.Circle, ellipse_obj: sp.Ellipse
    ) -> None:
        circle_obj.layer = 4
        ellipse_obj.layer = 5
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(ellipse_obj)

        selected = g.select_layers([4, 8])

        assert len(selected.group) == 1
        assert selected.group[0].layer == 4

    def test_layer_select_and_deselect(
        self, circle_obj: sp.Circle, ellipse_obj: sp.Ellipse
    ) -> None:
        circle_obj.layer = 11
        ellipse_obj.layer = 12
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(ellipse_obj)

        sel = g.select_layer(11)
        desel = g.deselect_layers([11])

        assert len(sel.group) == 1
        assert sel.group[0].layer == 11
        assert len(desel.group) == 1
        assert desel.group[0].layer == 12

    def test_get_area_sums_non_ref_shapes(
        self, circle_obj: sp.Circle, ellipse_obj: sp.Ellipse
    ) -> None:
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(ellipse_obj)

        expected_area = circle_obj.area() + ellipse_obj.area()

        assert g.get_area() == pytest.approx(expected_area)

    def test_path_to_poly_and_text_to_poly(self, text_obj: sp.Text) -> None:
        path = sp.Path([0.0, 1.0], [0.0, 0.0], width=0.2, layer=60)
        g = sp.GeomGroup()
        g.add(path)
        g.add(text_obj)

        g.path_to_poly()
        assert not any(isinstance(geom, sp.Path) for geom in g.group)
        assert any(isinstance(geom, sp.Poly) for geom in g.group)

        g.text_to_poly()
        assert not any(isinstance(geom, sp.Text) for geom in g.group)
        assert any(isinstance(geom, sp.Poly) for geom in g.group)

    def test_all_to_poly_converts_circle_ellipse_ring_arc(
        self,
        circle_obj: sp.Circle,
        ellipse_obj: sp.Ellipse,
        ring_obj: sp.Ring,
        arc_obj: sp.Arc,
    ) -> None:
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(ellipse_obj)
        g.add(ring_obj)
        g.add(arc_obj)

        g.all_to_poly(Npts_circ=10, Npts_arc=12, split_arc=False)

        assert len(g.group) == 4
        assert all(isinstance(geom, sp.Poly) for geom in g.group)

    def test_all_to_poly_keeps_sref_and_converts_others(
        self, geomgroup_obj: sp.GeomGroup, circle_obj: sp.Circle
    ) -> None:
        ref = sp.SRef(
            x0=0.0,
            y0=0.0,
            cellname="CellForAllToPoly",
            group=geomgroup_obj,
            mag=1.0,
            angle=0.0,
            mirror=False,
        )
        g = sp.GeomGroup()
        g.add(ref)
        g.add(circle_obj)

        g.all_to_poly(Npts_circ=8)

        assert any(isinstance(geom, sp.SRef) for geom in g.group)
        assert any(isinstance(geom, sp.Poly) for geom in g.group)

    def test_poly_to_circle_converts_round_poly_and_keeps_non_round(self) -> None:
        circular_poly = sp.Circle(0.0, 0.0, 2.0, layer=5).to_polygon(Npts=64).group[0]
        box_poly = sp.Box(0.0, 0.0, 2.0, 2.0).toPoly()
        box_poly.layer = 5

        g = sp.GeomGroup()
        g.add(circular_poly)
        g.add(box_poly)

        g.poly_to_circle(thresh=0.98, vcount=10, include_refs=False)

        assert any(isinstance(geom, sp.Circle) for geom in g.group)
        assert any(isinstance(geom, sp.Poly) for geom in g.group)

    def test_poly_to_circle_include_refs_updates_referenced_group(
        self, geomgroup_obj: sp.GeomGroup
    ) -> None:
        ref_group = sp.GeomGroup()
        ref_group.add(sp.Circle(0.0, 0.0, 2.0, layer=7).to_polygon(Npts=64).group[0])
        ref = sp.SRef(
            x0=1.0,
            y0=2.0,
            cellname="PolyToCircleRefCell",
            group=ref_group,
            mag=1.0,
            angle=0.0,
            mirror=False,
        )
        g = geomgroup_obj.copy()
        g.add(ref)

        g.poly_to_circle(thresh=0.98, vcount=10, include_refs=True)

        assert any(isinstance(geom, sp.Circle) for geom in ref.group.group)

    def test_keep_refs_only_removes_non_refs(
        self, geomgroup_obj: sp.GeomGroup, circle_obj: sp.Circle
    ) -> None:
        ref = sp.SRef(
            x0=0.0,
            y0=0.0,
            cellname="KeepRefsCell",
            group=geomgroup_obj,
            mag=1.0,
            angle=0.0,
            mirror=False,
        )
        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(ref)

        g.keep_refs_only()

        assert len(g.group) == 1
        assert isinstance(g.group[0], sp.SRef)

    def test_select_filters_by_layer_and_type(
        self, circle_obj: sp.Circle, poly_obj: sp.Poly
    ) -> None:
        poly_obj.layer = 2

        g = sp.GeomGroup()
        g.add(circle_obj)
        g.add(poly_obj)

        selected = g.select("(L==2) & (T=='Poly')")

        assert len(selected.group) == 1
        assert isinstance(selected.group[0], sp.Poly)
        assert selected.group[0].layer == 2

    @pytest.mark.parametrize(
        "disallowed_query", ["__import__('os')", "open('file.txt')", "os.system('ls')"]
    )
    def test_select_raises_for_disallowed_names(
        self, disallowed_query: str, poly_obj: sp.Poly
    ) -> None:
        g = sp.GeomGroup()
        g.add(poly_obj)

        with pytest.raises(NameError, match=r"Use of expression .+ not allowed"):
            g.select(disallowed_query)

    def test_find_matching_patterns_returns_expected_centers(self) -> None:
        layer = 7
        pattern = sp.Box(0.0, 0.0, 1.0, 1.0).toRect()
        pattern.set_layer(layer)

        poly1 = sp.Box(0.0, 0.0, 1.0, 1.0).toRect()
        poly2 = sp.Box(3.0, 4.0, 1.0, 1.0).toRect()
        poly1.set_layer(layer)
        poly2.set_layer(layer)
        g = poly1 + poly2

        matches = g.find_matching_patterns(pattern, layer)

        assert len(matches) == 2
        assert any(np.allclose(m, [0.5, 0.5]) for m in matches)
        assert any(np.allclose(m, [3.5, 4.5]) for m in matches)

    def test_find_matching_patterns_raises_for_disjoint_pattern(self) -> None:
        layer = 8
        g1 = sp.Box(0.0, 0.0, 1.0, 1.0).toRect()
        g2 = sp.Box(3.0, 0.0, 1.0, 1.0).toRect()
        g1.set_layer(layer)
        g2.set_layer(layer)
        pattern = g1 + g2

        g = sp.Box(0.0, 0.0, 1.0, 1.0).toRect()
        g.set_layer(layer)

        with pytest.raises(ValueError):
            g.find_matching_patterns(pattern, layer)

    @pytest.mark.xfail(
        strict=True,
        reason="in_polygons erronously checks for SRefs "
        "in the geometry group instead of Poly objects.",
    )
    def test_in_polygons_returns_true_for_point_in_polygon(self) -> None:
        layer = 9
        g = sp.Box(0.0, 0.0, 2.0, 2.0).toRect()
        g.set_layer(layer)

        assert g.in_polygons(1.0, 1.0) is True

    @pytest.mark.xfail(
        strict=True,
        reason="in_polygons erronously checks for SRefs "
        "in the geometry group instead of Poly objects.",
    )
    def test_in_polygons_sref_raises_attribute_error(self, sref_obj: sp.SRef) -> None:
        g = sp.GeomGroup()
        g.add(sref_obj)

        assert g.in_polygons(0.0, 0.0) is False

    def test_in_polygons_returns_false_for_empty_group(self) -> None:
        g = sp.GeomGroup()
        assert g.in_polygons(0.0, 0.0) is False

    def test_boolean_union_merges_overlapping_polygons(self) -> None:
        layer = 6
        g1 = sp.Box(0.0, 0.0, 2.0, 2.0).toRect()
        g2 = sp.Box(1.0, 0.0, 2.0, 2.0).toRect()
        g1.set_layer(layer)
        g2.set_layer(layer)

        g = g1 + g2
        g.boolean_union(layer)

        assert g.get_area() == pytest.approx(6.0)
        assert len(g.group) == 1
        res_poly = g.group[0]
        assert isinstance(res_poly, sp.Poly)
        assert res_poly.Npts == 9
        assert res_poly.layer == layer
        assert res_poly.perimeter() == pytest.approx(10.0)
        assert res_poly.centroid() == pytest.approx((1.5, 1.0))

    def test_boolean_difference_subtracts_polygon_set(self) -> None:
        layer_a = 1
        layer_b = 2
        ga = sp.Box(0.0, 0.0, 2.0, 2.0).toRect()
        gb = sp.Box(1.0, 0.0, 2.0, 2.0).toRect()
        ga.set_layer(layer_a)
        gb.set_layer(layer_b)

        ga.boolean_difference(gb, layerA=layer_a, layerB=layer_b)
        out = ga.select_layer(layer_a)

        assert out.get_area() == pytest.approx(2.0)
        assert len(out.group) == 1
        res_poly = out.group[0]
        assert isinstance(res_poly, sp.Poly)
        assert res_poly.Npts == 5
        assert res_poly.layer == layer_a
        assert res_poly.perimeter() == pytest.approx(6.0)
        assert res_poly.centroid() == pytest.approx((0.5, 1.0))

    def test_boolean_xor_computes_exclusive_or(self) -> None:
        layer_a = 1
        layer_b = 2
        ga = sp.Box(0.0, 0.0, 2.0, 2.0).toRect()
        gb = sp.Box(1.0, 0.0, 2.0, 2.0).toRect()
        ga.set_layer(layer_a)
        gb.set_layer(layer_b)

        ga.boolean_xor(gb, layerA=layer_a, layerB=layer_b)
        out = ga.select_layer(layer_a)

        assert out.get_area() == pytest.approx(4.0)
        assert len(out.group) == 2
        for res_poly in out.group:
            assert isinstance(res_poly, sp.Poly)
            assert res_poly.Npts == 5
            assert res_poly.layer == layer_a
            assert res_poly.perimeter() == pytest.approx(6.0)

    def test_boolean_intersection_keeps_overlap_only(self) -> None:
        layer_a = 4
        layer_b = 5
        ga = sp.Box(0.0, 0.0, 2.0, 2.0).toRect()
        gb = sp.Box(1.0, 0.0, 2.0, 2.0).toRect()
        ga.set_layer(layer_a)
        gb.set_layer(layer_b)

        ga.boolean_intersection(gb, layerA=layer_a, layerB=layer_b)
        out = ga.select_layer(layer_a)

        assert out.get_area() == pytest.approx(2.0)
        assert len(out.group) == 1
        res_poly = out.group[0]
        assert isinstance(res_poly, sp.Poly)
        assert res_poly.Npts == 5
        assert res_poly.layer == layer_a
        assert res_poly.perimeter() == pytest.approx(6.0)
        assert res_poly.centroid() == pytest.approx((1.5, 1.0))

    @pytest.mark.parametrize("offset", [-0.5, 0.5])
    def test_poly_resize_changes_polygon_extent(self, offset: float) -> None:
        layer = 2
        g = sp.Box(0.0, 0.0, 2.0, 2.0).toRect()
        g.set_layer(layer)

        bb_before = g.bounding_box()
        g.poly_resize(offset, layer)
        bb_after = g.bounding_box()

        assert bb_after.width == pytest.approx((1 + offset) * bb_before.width)
        assert bb_after.height == pytest.approx((1 + offset) * bb_before.height)

    @pytest.mark.parametrize("delta", [-1, 0.5, 0.9, 1.0, 2.0])
    def test_poly_anisotropic_resize_updates_polygons_on_layer(
        self, delta: float
    ) -> None:
        target_layer = 4
        other_layer = 7
        g_target = sp.Box(0.0, 0.0, 2.0, 2.0).toRect()
        g_other = sp.Box(10.0, 10.0, 2.0, 2.0).toRect()
        g_target.set_layer(target_layer)
        g_other.set_layer(other_layer)

        g = g_target + g_other

        bb_target_before = g.select_layer(target_layer).bounding_box()
        bb_other_before = g.select_layer(other_layer).bounding_box()

        g.poly_anisotropic_resize([-90, 0, 90], [delta] * 3, target_layer)

        bb_target_after = g.select_layer(target_layer).bounding_box()
        bb_other_after = g.select_layer(other_layer).bounding_box()

        mult = delta - 1 if delta >= 1 else 1 - delta
        assert bb_target_after.width == pytest.approx(mult * bb_target_before.width)
        assert bb_target_after.height == pytest.approx(mult * bb_target_before.height)
        assert bb_other_after.width == pytest.approx(bb_other_before.width)
        assert bb_other_after.height == pytest.approx(bb_other_before.height)

    def test_poly_outlining_creates_nonzero_outline_polygon(self) -> None:
        layer = 8
        g = sp.Box(0.0, 0.0, 2.0, 2.0).toRect()
        g.set_layer(layer)

        g.poly_outlining(offset=0.5, layer=layer)

        assert len(g.group) == 1
        res_poly = g.group[0]
        assert isinstance(res_poly, sp.Poly)
        assert res_poly.layer == layer
        res_bb = res_poly.bounding_box()
        assert res_bb.width == pytest.approx(3.0)
        assert res_bb.height == pytest.approx(3.0)
        assert res_bb.cx() == pytest.approx(1.0)
        assert res_bb.cy() == pytest.approx(1.0)
        assert res_poly.point_inside(-0.1, -0.1)
        assert res_poly.point_inside(2.1, 2.1)
        assert not res_poly.point_inside(0.1, 0.1)

    def test_poly_outlining_converts_circles_to_arcs(
        self, circle_obj: sp.Circle
    ) -> None:
        g = sp.GeomGroup()
        g.add(circle_obj)

        g.poly_outlining(offset=0.5, layer=1)

        assert any(type(geom) is sp.Arc for geom in g.group)
        assert not any(type(geom) is sp.Circle for geom in g.group)

    def test_invert_empty_layer_returns_unchanged(self) -> None:
        layer = 3
        g = sp.Box(0.0, 0.0, 2.0, 2.0).toRect()
        g.set_layer(layer)

        before_area = g.get_area()
        g.invert(layer + 1)
        after_area = g.get_area()

        assert after_area == pytest.approx(before_area)

    def test_invert_on_single_box_in_layer_results_in_empty_layer(self) -> None:
        layer = 5
        g = sp.Box(0.0, 0.0, 2.0, 2.0).toRect()
        g.set_layer(layer)

        g.invert(layer)

        assert len(g.group) == 0

    def test_invert_on_circle_results_in_hole_in_polygon(
        self, circle_obj: sp.Circle
    ) -> None:
        layer = 4
        circle_obj.layer = layer
        g = circle_obj.to_polygon(Npts=16)

        bb_before = g.bounding_box()
        bb_before_area = bb_before.width * bb_before.height
        before_area = g.get_area()
        expected_after_area = bb_before_area - before_area

        g.invert(layer)
        after_area = g.get_area()
        assert after_area == pytest.approx(expected_after_area, rel=1e-2)

    def test_trapezoids_preserves_layer_area(self) -> None:
        layer = 5
        g = sp.Box(0.0, 0.0, 3.0, 2.0).toRect()
        g.set_layer(layer)

        before_area = g.get_area()
        g.trapezoids(layer)

        assert len(g.group) >= 1
        assert g.get_area() == pytest.approx(before_area)
        assert all(isinstance(geom, sp.Poly) for geom in g.group)

    @pytest.mark.parametrize("keep_str", ["x>1", "y<0.5", "x+y>2.5", "A>0"])
    def test_poly_filter(self, keep_str: str) -> None:
        poly = sp.Poly([0.0, 2.0, 2.0, 1.0, 0.0], [0.0, 0.0, 2.0, 2.0, 2.0], layer=2)
        g = sp.GeomGroup()
        g.add(poly)

        before_npts = poly.Npts
        ndisc = g.poly_filter(keep_str)

        assert ndisc >= 1
        assert g.group[0].Npts == before_npts - ndisc + 1

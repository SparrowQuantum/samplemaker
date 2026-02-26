"""Unit tests for shapes module."""

import numpy as np
import pytest

# Used automatically by pytest to reset state before each test:
from fixtures import reset_samplemaker  # noqa: F401

from samplemaker import shapes as sp

_COORD = tuple[float, float]
_TF = tuple[float, ...]
_SREF_KWARG_TYPE = dict[str, str | float | bool]
_AREF_KWARG_TYPE = dict[str, str | float | int | bool]


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


@pytest.fixture
def sref_kwargs() -> _SREF_KWARG_TYPE:
    return {
        "x0": 1.0,
        "y0": 2.0,
        "cellname": "MyCell",
        "mag": 1.0,
        "angle": 0.0,
        "mirror": False,
    }


@pytest.fixture
def sref_obj(geomgroup_obj: sp.GeomGroup, sref_kwargs: _SREF_KWARG_TYPE) -> sp.SRef:
    return sp.SRef(
        x0=sref_kwargs["x0"],
        y0=sref_kwargs["y0"],
        cellname=sref_kwargs["cellname"],
        group=geomgroup_obj,
        mag=sref_kwargs["mag"],
        angle=sref_kwargs["angle"],
        mirror=sref_kwargs["mirror"],
    )


@pytest.fixture
def aref_kwargs() -> _AREF_KWARG_TYPE:
    return {
        "x0": 1.0,
        "y0": 2.0,
        "cellname": "MyArrayCell",
        "ncols": 2,
        "nrows": 3,
        "ax": 4.0,
        "ay": 0.0,
        "bx": 0.0,
        "by": 5.0,
        "mag": 1.0,
        "angle": 0.0,
        "mirror": False,
    }


@pytest.fixture
def aref_obj(geomgroup_obj: sp.GeomGroup, aref_kwargs: _AREF_KWARG_TYPE) -> sp.ARef:
    return sp.ARef(
        x0=aref_kwargs["x0"],
        y0=aref_kwargs["y0"],
        cellname=aref_kwargs["cellname"],
        group=geomgroup_obj,
        ncols=aref_kwargs["ncols"],
        nrows=aref_kwargs["nrows"],
        ax=aref_kwargs["ax"],
        ay=aref_kwargs["ay"],
        bx=aref_kwargs["bx"],
        by=aref_kwargs["by"],
        mag=aref_kwargs["mag"],
        angle=aref_kwargs["angle"],
        mirror=aref_kwargs["mirror"],
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
        sref_kwargs: _SREF_KWARG_TYPE,
    ) -> None:
        assert sref_obj.x0 == sref_kwargs["x0"]
        assert sref_obj.y0 == sref_kwargs["y0"]
        assert sref_obj.cellname == sref_kwargs["cellname"]
        assert sref_obj.mag == sref_kwargs["mag"]
        assert sref_obj.angle == sref_kwargs["angle"]
        assert sref_obj.mirror == sref_kwargs["mirror"]

        g = sref_obj.group
        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        assert g is geomgroup_obj

    @pytest.mark.parametrize("xoff, yoff", [(1.0, 2.0), (-1.0, -2.0), (0.5, -0.5)])
    def test_translate(
        self, sref_obj: sp.SRef, sref_kwargs: _SREF_KWARG_TYPE, xoff: float, yoff: float
    ) -> None:
        sref_obj.translate(xoff, yoff)
        expected_x0 = sref_kwargs["x0"] + xoff
        expected_y0 = sref_kwargs["y0"] + yoff
        assert sref_obj.x0 == pytest.approx(expected_x0)
        assert sref_obj.y0 == pytest.approx(expected_y0)

    def test_rotate(self, sref_obj: sp.SRef, sref_kwargs: _SREF_KWARG_TYPE) -> None:
        sref_obj.rotate(0.0, 0.0, 90.0)
        expected_x0 = -sref_kwargs["y0"]
        expected_y0 = sref_kwargs["x0"]
        expected_angle = (sref_kwargs["angle"] + 90.0) % 360
        assert sref_obj.x0 == pytest.approx(expected_x0)
        assert sref_obj.y0 == pytest.approx(expected_y0)
        assert sref_obj.angle == pytest.approx(expected_angle)

    def test_rotate_translate(
        self, sref_obj: sp.SRef, sref_kwargs: _SREF_KWARG_TYPE
    ) -> None:
        sref_obj.rotate_translate(2.0, 3.0, 180.0)
        expected_x0 = -sref_kwargs["x0"] + 2.0
        expected_y0 = -sref_kwargs["y0"] + 3.0
        expected_angle = (sref_kwargs["angle"] + 180.0) % 360
        assert sref_obj.x0 == pytest.approx(expected_x0)
        assert sref_obj.y0 == pytest.approx(expected_y0)
        assert sref_obj.angle == pytest.approx(expected_angle)

    @pytest.mark.parametrize("sx, sy", [(2.0, 3.0), (0.5, 0.5), (1.0, 1.0)])
    def test_scale(
        self, sref_obj: sp.SRef, sref_kwargs: _SREF_KWARG_TYPE, sx: float, sy: float
    ) -> None:
        sref_obj.scale(0.0, 0.0, sx, sy)
        expected_x0 = sref_kwargs["x0"] * sx
        expected_y0 = sref_kwargs["y0"] * sy
        assert sref_obj.x0 == pytest.approx(expected_x0)
        assert sref_obj.y0 == pytest.approx(expected_y0)
        assert sref_obj.mag == pytest.approx(sref_kwargs["mag"] * sx)

    @pytest.mark.parametrize("xc", [0.0, 1.0, 2.0])
    def test_mirror_x(
        self, sref_obj: sp.SRef, sref_kwargs: _SREF_KWARG_TYPE, xc: float
    ) -> None:
        sref_obj.mirrorX(xc)
        expected_x0 = 2 * xc - sref_kwargs["x0"]
        expected_y0 = sref_kwargs["y0"]
        expected_angle = (180.0 - sref_kwargs["angle"]) % 360
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
        self, sref_obj: sp.SRef, sref_kwargs: _SREF_KWARG_TYPE, yc: float
    ) -> None:
        sref_obj.mirrorY(yc)
        expected_x0 = sref_kwargs["x0"]
        expected_y0 = 2 * yc - sref_kwargs["y0"]
        expected_angle = (-sref_kwargs["angle"]) % 360
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
        self, geomgroup_obj: sp.GeomGroup, sref_kwargs: _SREF_KWARG_TYPE
    ) -> None:
        cellname = sref_kwargs["cellname"]
        pool_box = sp.Box(10.0, 20.0, 4.0, 6.0)
        sp._BoundingBoxPool[cellname] = pool_box
        sref = sp.SRef(
            x0=sref_kwargs["x0"],
            y0=sref_kwargs["y0"],
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


class TestAref:
    def test_init_aref(
        self,
        aref_obj: sp.ARef,
        geomgroup_obj: sp.GeomGroup,
        aref_kwargs: _AREF_KWARG_TYPE,
    ) -> None:
        assert aref_obj.x0 == aref_kwargs["x0"]
        assert aref_obj.y0 == aref_kwargs["y0"]
        assert aref_obj.cellname == aref_kwargs["cellname"]
        assert aref_obj.ncols == aref_kwargs["ncols"]
        assert aref_obj.nrows == aref_kwargs["nrows"]
        assert aref_obj.ax == aref_kwargs["ax"]
        assert aref_obj.ay == aref_kwargs["ay"]
        assert aref_obj.bx == aref_kwargs["bx"]
        assert aref_obj.by == aref_kwargs["by"]
        assert aref_obj.mag == aref_kwargs["mag"]
        assert aref_obj.angle == aref_kwargs["angle"]
        assert aref_obj.mirror == aref_kwargs["mirror"]

        g = aref_obj.group
        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        assert g is geomgroup_obj

    @pytest.mark.parametrize("xoff, yoff", [(1.0, 2.0), (-1.0, -2.0), (0.5, -0.5)])
    def test_translate(
        self, aref_obj: sp.ARef, aref_kwargs: _AREF_KWARG_TYPE, xoff: float, yoff: float
    ) -> None:
        aref_obj.translate(xoff, yoff)
        assert aref_obj.x0 == pytest.approx(aref_kwargs["x0"] + xoff)
        assert aref_obj.y0 == pytest.approx(aref_kwargs["y0"] + yoff)

    def test_rotate(self, aref_obj: sp.ARef, aref_kwargs: _AREF_KWARG_TYPE) -> None:
        aref_obj.rotate(0.0, 0.0, 90.0)
        assert aref_obj.x0 == pytest.approx(-aref_kwargs["y0"])
        assert aref_obj.y0 == pytest.approx(aref_kwargs["x0"])
        assert aref_obj.angle == pytest.approx((aref_kwargs["angle"] + 90.0) % 360)

    def test_rotate_translate(
        self, aref_obj: sp.ARef, aref_kwargs: _AREF_KWARG_TYPE
    ) -> None:
        aref_obj.rotate_translate(2.0, 3.0, 180.0)
        assert aref_obj.x0 == pytest.approx(-aref_kwargs["x0"] + 2.0)
        assert aref_obj.y0 == pytest.approx(-aref_kwargs["y0"] + 3.0)
        assert aref_obj.angle == pytest.approx((aref_kwargs["angle"] + 180.0) % 360)

    @pytest.mark.parametrize("sx, sy", [(2.0, 3.0), (0.5, 0.5), (1.0, 1.0)])
    def test_scale(
        self, aref_obj: sp.ARef, aref_kwargs: _AREF_KWARG_TYPE, sx: float, sy: float
    ) -> None:
        aref_obj.scale(0.0, 0.0, sx, sy)
        assert aref_obj.x0 == pytest.approx(aref_kwargs["x0"] * sx)
        assert aref_obj.y0 == pytest.approx(aref_kwargs["y0"] * sy)
        assert aref_obj.mag == pytest.approx(aref_kwargs["mag"] * sx)

    @pytest.mark.parametrize("xc", [0.0, 1.0, 2.0])
    def test_mirror_x(
        self, aref_obj: sp.ARef, aref_kwargs: _AREF_KWARG_TYPE, xc: float
    ) -> None:
        aref_obj.mirrorX(xc)
        assert aref_obj.x0 == pytest.approx(2 * xc - aref_kwargs["x0"])
        assert aref_obj.y0 == pytest.approx(aref_kwargs["y0"])
        assert aref_obj.angle == pytest.approx((180.0 - aref_kwargs["angle"]) % 360)

    def test_mirror_x_toggles_mirror_flag(self, aref_obj: sp.ARef) -> None:
        assert aref_obj.mirror is False
        aref_obj.mirrorX(0.0)
        assert aref_obj.mirror is True
        aref_obj.mirrorX(0.0)
        assert aref_obj.mirror is False

    @pytest.mark.parametrize("yc", [0.0, 1.0, 2.0])
    def test_mirror_y(
        self, aref_obj: sp.ARef, aref_kwargs: _AREF_KWARG_TYPE, yc: float
    ) -> None:
        aref_obj.mirrorY(yc)
        assert aref_obj.x0 == pytest.approx(aref_kwargs["x0"])
        assert aref_obj.y0 == pytest.approx(2 * yc - aref_kwargs["y0"])
        assert aref_obj.angle == pytest.approx((-aref_kwargs["angle"]) % 360)

    def test_mirror_y_toggles_mirror_flag(self, aref_obj: sp.ARef) -> None:
        assert aref_obj.mirror is False
        aref_obj.mirrorY(0.0)
        assert aref_obj.mirror is True
        aref_obj.mirrorY(0.0)
        assert aref_obj.mirror is False

    def test_centroid(self, aref_obj: sp.ARef, aref_kwargs: _AREF_KWARG_TYPE) -> None:
        expected_centroid = (aref_kwargs["x0"], aref_kwargs["y0"])
        assert aref_obj.centroid() == pytest.approx(expected_centroid)

    def test_bounding_box(
        self,
        aref_obj: sp.ARef,
        geomgroup_obj: sp.GeomGroup,
        aref_kwargs: _AREF_KWARG_TYPE,
    ) -> None:
        bb = aref_obj.bounding_box()
        ref_bb = geomgroup_obj.bounding_box()
        llx = aref_kwargs["x0"] + ref_bb.llx
        lly = aref_kwargs["y0"] + ref_bb.lly
        urx = (
            aref_kwargs["x0"]
            + ref_bb.urx()
            + (aref_kwargs["ncols"] - 1) * aref_kwargs["ax"]
            + (aref_kwargs["nrows"] - 1) * aref_kwargs["bx"]
        )
        ury = (
            aref_kwargs["y0"]
            + ref_bb.ury()
            + (aref_kwargs["ncols"] - 1) * aref_kwargs["ay"]
            + (aref_kwargs["nrows"] - 1) * aref_kwargs["by"]
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
        aref_kwargs: _AREF_KWARG_TYPE,
    ) -> None:
        placed = aref_obj.place_group(geomgroup_obj.flatten())
        assert len(placed.group) == aref_kwargs["ncols"] * aref_kwargs["nrows"]

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

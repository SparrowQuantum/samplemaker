"""Unit tests for shapes module."""

import numpy as np
import pytest

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
def box(box_llxy: _COORD, box_wh: _COORD) -> sp.Box:
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
def poly(poly_pts: tuple[_TF, _TF]) -> sp.Poly:
    xpts, ypts = poly_pts
    return sp.Poly(xpts, ypts, layer=2)


@pytest.fixture
def path_width() -> float:
    """Width of test path."""
    return 0.5


@pytest.fixture
def path(poly_pts: tuple[_TF, _TF], path_width: float) -> sp.Path:
    xpts, ypts = poly_pts
    # xpts and ypts need to be mutable for the transformations,
    # so we convert them to lists here.
    return sp.Path(list(xpts), list(ypts), path_width, layer=2)


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
    def test_basic_ops(self, box: sp.Box) -> None:
        expected_cx = box.llx + box.width / 2
        expected_cy = box.lly + box.height / 2
        expected_urx = box.llx + box.width
        expected_ury = box.lly + box.height

        assert box.cx() == pytest.approx(expected_cx)
        assert box.cy() == pytest.approx(expected_cy)
        assert box.urx() == pytest.approx(expected_urx)
        assert box.ury() == pytest.approx(expected_ury)

        other = sp.Box(-1.0, 0.0, 1.0, 1.0)
        box.combine(other)

        expected_llx = min(box.llx, other.llx)
        expected_lly = min(box.lly, other.lly)
        expected_urx = max(box.urx(), other.urx())
        expected_ury = max(box.ury(), other.ury())
        expected_width = expected_urx - expected_llx
        expected_height = expected_ury - expected_lly

        assert box.llx == pytest.approx(expected_llx)
        assert box.lly == pytest.approx(expected_lly)
        assert box.urx() == pytest.approx(expected_urx)
        assert box.ury() == pytest.approx(expected_ury)
        assert box.width == pytest.approx(expected_width)
        assert box.height == pytest.approx(expected_height)

    def test_to_poly(self, box: sp.Box) -> None:
        poly = box.toPoly()
        assert isinstance(poly, sp.Poly)
        # poly.data should be of the format [x0, y0, x1, y1, x2, y2, x3, y3, ..., x0, y0]
        assert isinstance(poly.data, np.ndarray)
        assert len(poly.data) == 10

        reshaped_data = poly.data.reshape((-1, 2))
        xpts, ypts = reshaped_data[:, 0], reshaped_data[:, 1]

        llx, lly = box.llx, box.lly
        urx, ury = box.urx(), box.ury()
        expected_xpts = [llx, urx, urx, llx, llx]
        expected_ypts = [lly, lly, ury, ury, lly]

        assert list(xpts) == pytest.approx(expected_xpts)
        assert list(ypts) == pytest.approx(expected_ypts)
        assert poly.layer == 0

    def test_to_rect(self, box: sp.Box) -> None:
        g = box.toRect()
        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        assert isinstance(g.group[0], sp.Poly)

    def test_numkey_points(self, box: sp.Box) -> None:
        llx, lly = box.llx, box.lly
        urx, ury = box.urx(), box.ury()
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
            x, y = box.get_numkey_point(numkey)
            assert (x, y) == pytest.approx((exp_x, exp_y))


class TestPoly:
    """
    Tests for the Poly class.

    Methods without coverage:

    * `three_point_filter`
    * `identical_to`
    * `anisotropic_resize`

    """

    def test_init_poly(self, poly: sp.Poly, poly_pts: tuple[_TF, _TF]) -> None:
        # poly.data should be of the format
        # [x0, y0, x1, y1, x2, y2, x3, y3, ..., x0, y0]
        assert isinstance(poly.data, np.ndarray)
        assert len(poly.data) == 2 * (len(poly_pts[0]) + 1)
        assert poly.Npts == len(poly_pts[0]) + 1
        assert poly.layer == 2

        reshaped_data = poly.data.reshape((-1, 2))
        resh_xpts, resh_ypts = reshaped_data[:, 0], reshaped_data[:, 1]
        # Close the polygon by repeating the first point at the end:
        expected_xpts = poly_pts[0] + (poly_pts[0][0],)
        expected_ypts = poly_pts[1] + (poly_pts[1][0],)
        assert list(resh_xpts) == pytest.approx(expected_xpts)
        assert list(resh_ypts) == pytest.approx(expected_ypts)

    def test_bounding_box(self, poly: sp.Poly, poly_pts: tuple[_TF, _TF]) -> None:
        bb = poly.bounding_box()
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

    def test_area(self, poly: sp.Poly) -> None:
        assert poly.area() == pytest.approx(6.0)

    def test_perimeter(self, poly: sp.Poly) -> None:
        assert poly.perimeter() == pytest.approx(10.0)

    def test_centroid(self, poly: sp.Poly) -> None:
        assert poly.centroid() == pytest.approx((1.5, 1.0))

    def test_translate(self, poly: sp.Poly) -> None:
        poly.translate(1.0, 2.0)
        assert poly.centroid() == pytest.approx((2.5, 3.0))
        assert poly.area() == pytest.approx(6.0)
        assert poly.perimeter() == pytest.approx(10.0)

    def test_rotate(self, poly: sp.Poly) -> None:
        poly.rotate(0.0, 0.0, 90.0)
        bb = poly.bounding_box()
        assert bb.width == pytest.approx(2.0)
        assert bb.height == pytest.approx(3.0)
        assert poly.centroid() == pytest.approx((-1.0, 1.5))
        assert poly.area() == pytest.approx(6.0)
        assert poly.perimeter() == pytest.approx(10.0)

        poly.rotate(-1.0, 1.5, -45.0)
        bb = poly.bounding_box()
        assert bb.width == pytest.approx(3.5355339059327378)
        assert bb.height == pytest.approx(3.5355339059327378)
        assert poly.centroid() == pytest.approx((-1.0, 1.5))
        assert poly.area() == pytest.approx(6.0)
        assert poly.perimeter() == pytest.approx(10.0)

    def test_rotate_translate(self, poly: sp.Poly) -> None:
        poly.rotate_translate(2.0, 3.0, 180.0)
        bb = poly.bounding_box()
        assert bb.width == pytest.approx(3.0)
        assert bb.height == pytest.approx(2.0)
        assert poly.centroid() == pytest.approx((0.5, 2.0))
        assert poly.area() == pytest.approx(6.0)
        assert poly.perimeter() == pytest.approx(10.0)

    def test_scale(self, poly: sp.Poly) -> None:
        poly.scale(0.0, 0.0, 2.0, 3.0)
        bb = poly.bounding_box()
        assert bb.width == pytest.approx(6.0)
        assert bb.height == pytest.approx(6.0)
        assert poly.centroid() == pytest.approx((3.0, 3.0))
        assert poly.area() == pytest.approx(36.0)
        assert poly.perimeter() == pytest.approx(24.0)

    def test_mirror_x(self, poly: sp.Poly) -> None:
        poly.mirrorX(1.5)
        bb = poly.bounding_box()
        assert bb.width == pytest.approx(3.0)
        assert bb.height == pytest.approx(2.0)
        assert poly.centroid() == pytest.approx((1.5, 1.0))
        assert poly.area() == pytest.approx(6.0)
        assert poly.perimeter() == pytest.approx(10.0)

    def test_mirror_y(self, poly: sp.Poly) -> None:
        poly.mirrorY(1.0)
        bb = poly.bounding_box()
        assert bb.width == pytest.approx(3.0)
        assert bb.height == pytest.approx(2.0)
        assert poly.centroid() == pytest.approx((1.5, 1.0))
        assert poly.area() == pytest.approx(6.0)
        assert poly.perimeter() == pytest.approx(10.0)

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

    def test_to_polygon(self, poly: sp.Poly) -> None:
        g = poly.to_polygon()
        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        assert g.group[0] is poly

    def test_to_circle(self, poly: sp.Poly) -> None:
        # Rectangle will have a perfect circle fit
        # vcount too high -> return empty geometry group
        gg = poly.to_circle(thresh=0.0, vcount=10)
        assert isinstance(gg, sp.GeomGroup)
        assert len(gg.group) == 0

        # threshold too high -> return empty geometry group
        gg = poly.to_circle(thresh=1.1, vcount=4)
        assert isinstance(gg, sp.GeomGroup)
        assert len(gg.group) == 0

        # Perfect fit -> return one circle
        gg = poly.to_circle(thresh=0.98, vcount=4)
        assert isinstance(gg, sp.GeomGroup)
        assert len(gg.group) == 1

        circle = gg.group[0]
        assert isinstance(circle, sp.Circle)
        assert circle.r == pytest.approx(1.802775637732)
        assert circle.x0 == pytest.approx(1.5)
        assert circle.y0 == pytest.approx(1.0)

    def test_point_inside(self, poly: sp.Poly) -> None:
        # Points inside the rectangle
        assert poly.point_inside(1.0, 1.0) is True
        assert poly.point_inside(2.5, 0.5) is True

        # Points outside the rectangle
        assert poly.point_inside(-1.0, 1.0) is False
        assert poly.point_inside(4.0, 1.0) is False
        assert poly.point_inside(1.5, -1.0) is False
        assert poly.point_inside(1.5, 3.0) is False


class TestPath:
    def test_init_path(
        self, path: sp.Path, poly_pts: tuple[_TF, _TF], path_width: float
    ) -> None:
        xpts, ypts = poly_pts
        assert path.xpts == list(xpts)
        assert path.ypts == list(ypts)
        assert path.width == path_width
        assert path.layer == 2
        assert path.Npts == len(xpts)

    def test_bounding_box(self, path: sp.Path, poly_pts: tuple[_TF, _TF]) -> None:
        bb = path.bounding_box()
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

    def test_translate(self, path: sp.Path, poly_pts: tuple[_TF, _TF]) -> None:
        xpts, ypts = poly_pts
        path.translate(1.0, 2.0)
        expected_xpts = [x + 1.0 for x in xpts]
        expected_ypts = [y + 2.0 for y in ypts]
        assert path.xpts == pytest.approx(expected_xpts)
        assert path.ypts == pytest.approx(expected_ypts)

    def test_rotate(self, path: sp.Path, poly_pts: tuple[_TF, _TF]) -> None:
        xpts, ypts = poly_pts
        path.rotate(0.0, 0.0, 90.0)
        expected_xpts = [-y for y in ypts]
        expected_ypts = [x for x in xpts]
        assert path.xpts == pytest.approx(expected_xpts)
        assert path.ypts == pytest.approx(expected_ypts)

    def test_rotate_translate(self, path: sp.Path, poly_pts: tuple[_TF, _TF]) -> None:
        xpts, ypts = poly_pts
        path.rotate_translate(2.0, 3.0, 180.0)
        expected_xpts = [2.0 - x for x in xpts]
        expected_ypts = [3.0 - y for y in ypts]
        assert path.xpts == pytest.approx(expected_xpts)
        assert path.ypts == pytest.approx(expected_ypts)

    def test_scale(self, path: sp.Path, poly_pts: tuple[_TF, _TF]) -> None:
        xpts, ypts = poly_pts
        path.scale(0.0, 0.0, 2.0, 3.0)
        expected_xpts = [2.0 * x for x in xpts]
        expected_ypts = [3.0 * y for y in ypts]
        assert path.xpts == pytest.approx(expected_xpts)
        assert path.ypts == pytest.approx(expected_ypts)

    def test_mirror_x(self, path: sp.Path, poly_pts: tuple[_TF, _TF]) -> None:
        xpts, ypts = poly_pts
        path.mirrorX(1.5)
        expected_xpts = [2 * 1.5 - x for x in xpts]
        expected_ypts = list(ypts)
        assert path.xpts == pytest.approx(expected_xpts)
        assert path.ypts == pytest.approx(expected_ypts)

    def test_mirror_y(self, path: sp.Path, poly_pts: tuple[_TF, _TF]) -> None:
        xpts, ypts = poly_pts
        path.mirrorY(1.0)
        expected_xpts = list(xpts)
        expected_ypts = [2 * 1.0 - y for y in ypts]
        assert path.xpts == pytest.approx(expected_xpts)
        assert path.ypts == pytest.approx(expected_ypts)

    def test_path_length(self, path: sp.Path) -> None:
        # Two sides of length 3 and one side of length 2 -> total length 8
        assert path.path_length() == pytest.approx(8.0)

    def test_area(self, path: sp.Path) -> None:
        # The area of a path is defined as the area of the polygon formed by
        # the centerline of the path and the width. For a rectangle with width
        # 0.5 and length 8, the area should be 4.
        assert path.area() == pytest.approx(4.0)

    def test_perimeter(self, path: sp.Path) -> None:
        # Outside perimeter of the path is approximately twice the path length
        # plus twice the path width. For a path length of 8 and width of 0.5,
        # the perimeter should be approximately 17.0.
        assert path.perimeter() == pytest.approx(17.0)

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

    def test_to_polygon_polyline(self, path: sp.Path) -> None:
        # A multi-segment path converts to a single polygon outlining the stroke.
        g = path.to_polygon()

        half = path.width / 2
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

        assert poly.layer == path.layer
        assert xpts == pytest.approx(expected_xpts)
        assert ypts == pytest.approx(expected_ypts)


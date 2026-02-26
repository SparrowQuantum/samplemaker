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

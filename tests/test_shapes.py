"""Unit tests for shapes module."""

import numpy as np
import pytest

from samplemaker import shapes as sp


def test_dot_transformations():
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
    def test_basic_ops(self):
        box = sp.Box(0.0, 1.0, 2.0, 3.0)
        assert box.cx() == pytest.approx(1.0)
        assert box.cy() == pytest.approx(2.5)
        assert box.urx() == pytest.approx(2.0)
        assert box.ury() == pytest.approx(4.0)

        other = sp.Box(-1.0, 0.0, 1.0, 1.0)
        box.combine(other)
        assert box.llx == pytest.approx(-1.0)
        assert box.lly == pytest.approx(0.0)
        assert box.urx() == pytest.approx(2.0)
        assert box.ury() == pytest.approx(4.0)
        assert box.width == pytest.approx(3.0)
        assert box.height == pytest.approx(4.0)

    def test_to_poly(self):
        box = sp.Box(0.0, 1.0, 2.0, 3.0)
        p = box.toPoly()
        assert isinstance(p, sp.Poly)
        # p.data should be of the format [x0, y0, x1, y1, x2, y2, x3, y3, ..., x0, y0]
        assert isinstance(p.data, np.ndarray)
        assert len(p.data) == 10

        reshaped_data = p.data.reshape((-1, 2))
        xpts, ypts = reshaped_data[:, 0], reshaped_data[:, 1]
        expected_xpts = [0.0, 2.0, 2.0, 0.0, 0.0]
        expected_ypts = [1.0, 1.0, 4.0, 4.0, 1.0]
        assert list(xpts) == pytest.approx(expected_xpts)
        assert list(ypts) == pytest.approx(expected_ypts)
        assert p.layer == 0

    def test_to_rect(self):
        box = sp.Box(0.0, 1.0, 2.0, 3.0)
        g = box.toRect()
        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        assert isinstance(g.group[0], sp.Poly)

    def test_numkey_points(self):
        box = sp.Box(-1.0, 1.0, 4.0, 3.0)

        expected_points = {
            1: (-1.0, 1.0),  # ll
            2: (1.0, 1.0),  # lc
            3: (3.0, 1.0),  # lr
            4: (-1.0, 2.5),  # cl
            5: (1.0, 2.5),  # cc
            6: (3.0, 2.5),  # cr
            7: (-1.0, 4.0),  # ul
            8: (1.0, 4.0),  # uc
            9: (3.0, 4.0),  # ur
        }
        for numkey, (exp_x, exp_y) in expected_points.items():
            x, y = box.get_numkey_point(numkey)
            assert (x, y) == pytest.approx((exp_x, exp_y))


class TestPoly:
    XPTS = (0.0, 3.0, 3.0, 0.0)
    YPTS = (0.0, 0.0, 2.0, 2.0)
    LAYER = 2

    DATA_XPTS = (0.0, 3.0, 3.0, 0.0, 0.0)
    DATA_YPTS = (0.0, 0.0, 2.0, 2.0, 0.0)

    def test_init_poly(self):
        p = sp.Poly(self.XPTS, self.YPTS, layer=self.LAYER)

        # p.data should be of the format [x0, y0, x1, y1, x2, y2, x3, y3, ..., x0, y0]
        assert isinstance(p.data, np.ndarray)
        assert len(p.data) == 10
        assert p.Npts == 5
        assert p.layer == self.LAYER

        reshaped_data = p.data.reshape((-1, 2))
        resh_xpts, resh_ypts = reshaped_data[:, 0], reshaped_data[:, 1]
        expected_xpts = list(self.DATA_XPTS)
        expected_ypts = list(self.DATA_YPTS)
        assert list(resh_xpts) == pytest.approx(expected_xpts)
        assert list(resh_ypts) == pytest.approx(expected_ypts)

        bb = p.bounding_box()
        assert bb.width == pytest.approx(3.0)
        assert bb.height == pytest.approx(2.0)
        assert p.area() == pytest.approx(6.0)
        assert p.perimeter() == pytest.approx(10.0)
        assert p.centroid() == pytest.approx((1.5, 1.0))

    def test_translate(self):
        p = sp.Poly(self.XPTS, self.YPTS, layer=self.LAYER)

        p.translate(1.0, 2.0)
        assert p.centroid() == pytest.approx((2.5, 3.0))
        assert p.area() == pytest.approx(6.0)
        assert p.perimeter() == pytest.approx(10.0)

    def test_rotate(self):
        p = sp.Poly(self.XPTS, self.YPTS, layer=self.LAYER)

        p.rotate(0.0, 0.0, 90.0)
        bb = p.bounding_box()
        assert bb.width == pytest.approx(2.0)
        assert bb.height == pytest.approx(3.0)
        assert p.centroid() == pytest.approx((-1.0, 1.5))
        assert p.area() == pytest.approx(6.0)
        assert p.perimeter() == pytest.approx(10.0)

        p.rotate(-1.0, 1.5, -45.0)
        bb = p.bounding_box()
        assert bb.width == pytest.approx(3.5355339059327378)
        assert bb.height == pytest.approx(3.5355339059327378)
        assert p.centroid() == pytest.approx((-1.0, 1.5))
        assert p.area() == pytest.approx(6.0)
        assert p.perimeter() == pytest.approx(10.0)

    def test_rotate_translate(self):
        p = sp.Poly(self.XPTS, self.YPTS, layer=self.LAYER)

        p.rotate_translate(2.0, 3.0, 180.0)
        bb = p.bounding_box()
        assert bb.width == pytest.approx(3.0)
        assert bb.height == pytest.approx(2.0)
        assert p.centroid() == pytest.approx((0.5, 2.0))
        assert p.area() == pytest.approx(6.0)
        assert p.perimeter() == pytest.approx(10.0)

    def test_scale(self):
        p = sp.Poly(self.XPTS, self.YPTS, layer=self.LAYER)

        p.scale(0.0, 0.0, 2.0, 3.0)
        bb = p.bounding_box()
        assert bb.width == pytest.approx(6.0)
        assert bb.height == pytest.approx(6.0)
        assert p.centroid() == pytest.approx((3.0, 3.0))
        assert p.area() == pytest.approx(36.0)
        assert p.perimeter() == pytest.approx(24.0)

    def test_mirror_x(self):
        p = sp.Poly(self.XPTS, self.YPTS, layer=self.LAYER)

        p.mirrorX(1.5)
        bb = p.bounding_box()
        assert bb.width == pytest.approx(3.0)
        assert bb.height == pytest.approx(2.0)
        assert p.centroid() == pytest.approx((1.5, 1.0))
        assert p.area() == pytest.approx(6.0)
        assert p.perimeter() == pytest.approx(10.0)

    def test_mirror_y(self):
        p = sp.Poly(self.XPTS, self.YPTS, layer=self.LAYER)

        p.mirrorY(1.0)
        bb = p.bounding_box()
        assert bb.width == pytest.approx(3.0)
        assert bb.height == pytest.approx(2.0)
        assert p.centroid() == pytest.approx((1.5, 1.0))
        assert p.area() == pytest.approx(6.0)
        assert p.perimeter() == pytest.approx(10.0)

    def test_int_data(self):
        scaled_xpts = [1000 * x for x in self.DATA_XPTS]
        scaled_ypts = [1000 * y for y in self.DATA_YPTS]
        int_data = np.array([scaled_xpts, scaled_ypts], dtype=int).T.reshape(-1)
        p = sp.Poly([], [], layer=self.LAYER)
        p.set_int_data(int_data)

        assert p.int_data() == pytest.approx(int_data)
        assert p.Npts == 5

        reshaped_data = p.data.reshape((-1, 2))
        resh_xpts, resh_ypts = reshaped_data[:, 0], reshaped_data[:, 1]
        expected_xpts = list(self.DATA_XPTS)
        expected_ypts = list(self.DATA_YPTS)
        assert list(resh_xpts) == pytest.approx(expected_xpts)
        assert list(resh_ypts) == pytest.approx(expected_ypts)

        bb = p.bounding_box()
        assert bb.width == pytest.approx(3.0)
        assert bb.height == pytest.approx(2.0)
        assert p.area() == pytest.approx(6.0)
        assert p.perimeter() == pytest.approx(10.0)
        assert p.centroid() == pytest.approx((1.5, 1.0))

    def test_to_polygon(self):
        p = sp.Poly(self.XPTS, self.YPTS, layer=self.LAYER)
        g = p.to_polygon()
        assert isinstance(g, sp.GeomGroup)
        assert len(g.group) == 1
        assert g.group[0] is p

    def test_to_circle(self):
        # Rectangle will have a perfect circle fit
        p = sp.Poly(self.XPTS, self.YPTS, layer=self.LAYER)

        # vcount too high -> return empty geometry group
        gg = p.to_circle(thresh=0.0, vcount=10)
        assert isinstance(gg, sp.GeomGroup)
        assert len(gg.group) == 0

        # threshold too high -> return empty geometry group
        gg = p.to_circle(thresh=1.1, vcount=4)
        assert isinstance(gg, sp.GeomGroup)
        assert len(gg.group) == 0

        # Perfect fit -> return one circle
        gg = p.to_circle(thresh=0.98, vcount=4)
        assert isinstance(gg, sp.GeomGroup)
        assert len(gg.group) == 1

        circle = gg.group[0]
        assert isinstance(circle, sp.Circle)
        assert circle.r == pytest.approx(1.802775637732)
        assert circle.x0 == pytest.approx(1.5)
        assert circle.y0 == pytest.approx(1.0)

    def test_point_inside(self):
        p = sp.Poly(self.XPTS, self.YPTS, layer=self.LAYER)

        # Points inside the rectangle
        assert p.point_inside(1.0, 1.0) is True
        assert p.point_inside(2.5, 0.5) is True

        # Points outside the rectangle
        assert p.point_inside(-1.0, 1.0) is False
        assert p.point_inside(4.0, 1.0) is False
        assert p.point_inside(1.5, -1.0) is False
        assert p.point_inside(1.5, 3.0) is False
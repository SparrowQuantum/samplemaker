"""Unit tests for the makers module."""

import pytest

from samplemaker import makers as sm
from samplemaker import shapes as sp


@pytest.mark.parametrize("x0, y0", [(-1.0, 3.0), (0.0, 0.0), (2.0, -2.0)])
def test_make_dot(x0: float, y0: float) -> None:
    dot = sm.make_dot(x0, y0)
    assert isinstance(dot, sp.Dot)
    assert dot.x == x0
    assert dot.y == y0


def test_make_poly():
    xpts = [0.0, 1.0, 1.0, 0.0]
    ypts = [0.0, 0.0, 1.0, 1.0]
    poly_geom = sm.make_poly(xpts, ypts)

    assert isinstance(poly_geom, sp.GeomGroup)
    assert len(poly_geom.group) == 1

    poly = poly_geom.group[0]
    assert isinstance(poly, sp.Poly)
    assert poly.Npts == 5
    assert poly.layer == 1

    poly_xpts = poly.data[0::2]
    poly_ypts = poly.data[1::2]
    expected_xpts = xpts + [xpts[0]]
    expected_ypts = ypts + [ypts[0]]
    assert poly_xpts == pytest.approx(expected_xpts)
    assert poly_ypts == pytest.approx(expected_ypts)


def test_make_poly_nondefault_layer():
    xpts = [0.0, 1.0, 1.0, 0.0]
    ypts = [0.0, 0.0, 1.0, 1.0]
    layer = 2
    poly_geom = sm.make_poly(xpts, ypts, layer=layer)

    assert isinstance(poly_geom, sp.GeomGroup)
    assert len(poly_geom.group) == 1

    poly = poly_geom.group[0]
    assert isinstance(poly, sp.Poly)
    assert poly.layer == layer


def test_make_path():
    xpts = [0.0, 1.0, 1.0]
    ypts = [0.0, 0.0, 1.0]
    width = 0.5
    path_geom = sm.make_path(xpts, ypts, width, to_poly=False)

    assert isinstance(path_geom, sp.GeomGroup)
    assert len(path_geom.group) == 1

    path = path_geom.group[0]
    assert isinstance(path, sp.Path)
    assert path.xpts == pytest.approx(xpts)
    assert path.ypts == pytest.approx(ypts)
    assert path.width == width
    assert path.layer == 1
    assert path.Npts == 3


def test_make_path_nondefault_layer():
    xpts = [0.0, 1.0, 1.0]
    ypts = [0.0, 0.0, 1.0]
    width = 0.5
    layer = 2
    path_geom = sm.make_path(xpts, ypts, width, to_poly=False, layer=layer)

    assert isinstance(path_geom, sp.GeomGroup)
    assert len(path_geom.group) == 1

    path = path_geom.group[0]
    assert isinstance(path, sp.Path)
    assert path.layer == layer


def test_make_path_to_poly():
    xpts = [0.0, 1.0, 1.0]
    ypts = [0.0, 0.0, 1.0]
    width = 0.5
    path_geom = sm.make_path(xpts, ypts, width, to_poly=True)

    assert isinstance(path_geom, sp.GeomGroup)
    assert len(path_geom.group) == 1

    poly = path_geom.group[0]
    assert isinstance(poly, sp.Poly)
    assert poly.layer == 1


def test_make_path_to_poly_nondefault_layer():
    xpts = [0.0, 1.0, 1.0]
    ypts = [0.0, 0.0, 1.0]
    width = 0.5
    layer = 2
    path_geom = sm.make_path(xpts, ypts, width, to_poly=True, layer=layer)

    assert isinstance(path_geom, sp.GeomGroup)
    assert len(path_geom.group) == 1

    poly = path_geom.group[0]
    assert isinstance(poly, sp.Poly)
    assert poly.layer == layer


@pytest.mark.parametrize(
    "numkey,expected_posu,expected_posv", [(1, 0, 2), (4, 0, 1), (5, 1, 1), (9, 2, 0)]
)
def test_make_text(numkey: int, expected_posu: int, expected_posv: int) -> None:
    x0 = 0.0
    y0 = 0.0
    text_str = "Hello, World!"
    height = 10.0
    width = 1.0
    text_geom = sm.make_text(
        x0=x0,
        y0=y0,
        text=text_str,
        height=height,
        width=width,
        numkey=numkey,
    )

    assert isinstance(text_geom, sp.GeomGroup)
    assert len(text_geom.group) == 1

    text = text_geom.group[0]
    assert isinstance(text, sp.Text)
    assert text.text == text_str
    assert text.x0 == x0
    assert text.y0 == y0
    assert text.layer == 1
    assert text.angle == 0
    assert text.height == height
    assert text.width == width
    assert text.posu == expected_posu
    assert text.posv == expected_posv


def test_make_text_invalid_numkey_raises() -> None:
    with pytest.raises(ValueError, match="numkey should be between 1 and 9"):
        sm.make_text(
            x0=0.0,
            y0=0.0,
            text="bad",
            height=10.0,
            width=1.0,
            numkey=0,
        )


def test_make_text_to_poly():
    x0 = 0.0
    y0 = 0.0
    text_str = "Hello, World!"
    height = 10.0
    width = 1.0
    layer = 2
    text_geom = sm.make_text(
        x0=x0,
        y0=y0,
        text=text_str,
        height=height,
        width=width,
        numkey=1,
        to_poly=True,
        layer=layer,
    )

    assert isinstance(text_geom, sp.GeomGroup)
    assert text_geom.get_layer_list() == {layer}
    assert all(isinstance(p, sp.Poly) for p in text_geom.group)

    bb = text_geom.bounding_box()
    assert bb.llx == pytest.approx(x0 - width / 2)
    assert bb.lly == pytest.approx(y0 - width / 2)


def test_make_sref():
    x0 = 1.0
    y0 = -2.0
    cellname = "MY_CELL"
    base_geom = sm.make_rect(0, 0, 5, 2)
    sref_geom = sm.make_sref(x0, y0, cellname, base_geom)

    assert isinstance(sref_geom, sp.GeomGroup)
    assert len(sref_geom.group) == 1

    sref = sref_geom.group[0]
    assert isinstance(sref, sp.SRef)
    assert sref.x0 == x0
    assert sref.y0 == y0
    assert sref.mag == 1.0
    assert sref.angle == 0
    assert not sref.mirror
    assert sref.cellname == cellname
    assert sref.group == base_geom


def test_make_sref_transformed():
    x0 = 1.0
    y0 = -2.0
    cellname = "MY_CELL"
    base_geom = sm.make_rect(0, 0, 5, 2)
    mag = 2.3
    angle = 23.0
    mirror = True
    sref_geom = sm.make_sref(
        x0, y0, cellname, base_geom, mag=mag, angle=angle, mirror=mirror
    )

    assert isinstance(sref_geom, sp.GeomGroup)
    assert len(sref_geom.group) == 1

    sref = sref_geom.group[0]
    assert isinstance(sref, sp.SRef)
    assert sref.x0 == x0
    assert sref.y0 == y0
    assert sref.mag == mag
    assert sref.angle == angle
    assert sref.mirror == mirror
    assert sref.cellname == cellname
    assert sref.group == base_geom


def test_make_aref() -> None:
    base_geom = sm.make_rect(0.0, 0.0, 5.0, 2.0)
    aref_geom = sm.make_aref(
        x0=1.0,
        y0=-2.0,
        cellname="MY_ARRAY",
        group=base_geom,
        ncols=3,
        nrows=2,
        ax=10.0,
        ay=0.0,
        bx=0.0,
        by=20.0,
    )

    assert isinstance(aref_geom, sp.GeomGroup)
    assert len(aref_geom.group) == 1
    aref = aref_geom.group[0]
    assert isinstance(aref, sp.ARef)
    assert aref.x0 == 1.0
    assert aref.y0 == -2.0
    assert aref.cellname == "MY_ARRAY"
    assert aref.group == base_geom
    assert aref.ncols == 3
    assert aref.nrows == 2
    assert aref.ax == 10.0
    assert aref.ay == 0.0
    assert aref.bx == 0.0
    assert aref.by == 20.0
    assert aref.mag == 1.0
    assert aref.angle == 0
    assert not aref.mirror


def test_make_aref_transformed() -> None:
    base_geom = sm.make_rect(0.0, 0.0, 5.0, 2.0)
    aref_geom = sm.make_aref(
        x0=1.0,
        y0=-2.0,
        cellname="MY_ARRAY",
        group=base_geom,
        ncols=2,
        nrows=2,
        ax=7.0,
        ay=1.0,
        bx=-1.0,
        by=8.0,
        mag=1.5,
        angle=15.0,
        mirror=True,
    )
    aref = aref_geom.group[0]
    assert isinstance(aref, sp.ARef)
    assert aref.mag == pytest.approx(1.5)
    assert aref.angle == pytest.approx(15.0)
    assert aref.mirror


def test_make_circle() -> None:
    circle_geom = sm.make_circle(2.0, -3.0, 4.0, layer=3)
    assert isinstance(circle_geom, sp.GeomGroup)
    assert len(circle_geom.group) == 1
    circle = circle_geom.group[0]
    assert isinstance(circle, sp.Circle)
    assert circle.x0 == 2.0
    assert circle.y0 == -3.0
    assert circle.r == 4.0
    assert circle.layer == 3


def test_make_circle_to_poly() -> None:
    circle_geom = sm.make_circle(0.0, 0.0, 2.0, layer=4, to_poly=True, vertices=10)
    assert isinstance(circle_geom, sp.GeomGroup)
    assert len(circle_geom.group) == 1
    poly = circle_geom.group[0]
    assert isinstance(poly, sp.Poly)
    assert poly.layer == 4
    assert poly.Npts == 11


def test_make_ellipse() -> None:
    ellipse_geom = sm.make_ellipse(1.0, 2.0, 5.0, 3.0, 30.0, layer=6)
    assert isinstance(ellipse_geom, sp.GeomGroup)
    assert len(ellipse_geom.group) == 1
    ellipse = ellipse_geom.group[0]
    assert isinstance(ellipse, sp.Ellipse)
    assert ellipse.x0 == 1.0
    assert ellipse.y0 == 2.0
    assert ellipse.r == 5.0
    assert ellipse.r1 == 3.0
    assert ellipse.rot == 30.0
    assert ellipse.layer == 6


def test_make_ellipse_to_poly() -> None:
    ellipse_geom = sm.make_ellipse(
        x0=0.0,
        y0=0.0,
        rX=4.0,
        rY=2.0,
        rot=45.0,
        layer=7,
        to_poly=True,
        vertices=12,
    )
    assert isinstance(ellipse_geom, sp.GeomGroup)
    assert len(ellipse_geom.group) == 1
    poly = ellipse_geom.group[0]
    assert isinstance(poly, sp.Poly)
    assert poly.layer == 7
    assert poly.Npts == 13


def test_make_ring() -> None:
    ring_geom = sm.make_ring(1.0, -1.0, rX=6.0, rY=4.0, rot=10.0, w=0.8, layer=8)
    assert isinstance(ring_geom, sp.GeomGroup)
    assert len(ring_geom.group) == 1
    ring = ring_geom.group[0]
    assert isinstance(ring, sp.Ring)
    assert ring.x0 == 1.0
    assert ring.y0 == -1.0
    assert ring.r == 6.0
    assert ring.r1 == 4.0
    assert ring.rot == 10.0
    assert ring.w == 0.8
    assert ring.layer == 8


def test_make_ring_to_poly() -> None:
    ring_geom = sm.make_ring(
        x0=0.0,
        y0=0.0,
        rX=5.0,
        rY=3.0,
        rot=0.0,
        w=1.0,
        layer=9,
        to_poly=True,
        vertices=16,
    )
    assert isinstance(ring_geom, sp.GeomGroup)
    assert len(ring_geom.group) == 1
    poly = ring_geom.group[0]
    assert isinstance(poly, sp.Poly)
    assert poly.layer == 9
    assert poly.Npts == 35


def test_make_arc() -> None:
    arc_geom = sm.make_arc(2.0, 1.0, 7.0, 5.0, 20.0, 1.2, 15.0, 120.0, layer=10)
    assert isinstance(arc_geom, sp.GeomGroup)
    assert len(arc_geom.group) == 1
    arc = arc_geom.group[0]
    assert isinstance(arc, sp.Arc)
    assert arc.x0 == 2.0
    assert arc.y0 == 1.0
    assert arc.r == 7.0
    assert arc.r1 == 5.0
    assert arc.rot == 20.0
    assert arc.w == 1.2
    assert arc.a1 == 15.0
    assert arc.a2 == 120.0
    assert arc.layer == 10


def test_make_arc_to_poly() -> None:
    arc_geom = sm.make_arc(
        x0=0.0,
        y0=0.0,
        rX=6.0,
        rY=4.0,
        rot=0.0,
        w=1.0,
        a1=0.0,
        a2=90.0,
        layer=11,
        to_poly=True,
        vertices=10,
    )
    assert isinstance(arc_geom, sp.GeomGroup)
    assert len(arc_geom.group) == 1
    poly = arc_geom.group[0]
    assert isinstance(poly, sp.Poly)
    assert poly.layer == 11


def test_make_arc_to_poly_split() -> None:
    arc_geom = sm.make_arc(
        x0=0.0,
        y0=0.0,
        rX=6.0,
        rY=4.0,
        rot=0.0,
        w=1.0,
        a1=0.0,
        a2=90.0,
        layer=12,
        to_poly=True,
        vertices=8,
        split=True,
    )
    assert isinstance(arc_geom, sp.GeomGroup)
    assert len(arc_geom.group) == 8
    assert all(isinstance(poly, sp.Poly) for poly in arc_geom.group)
    assert all(poly.layer == 12 for poly in arc_geom.group)


def test_make_rect_center_numkey() -> None:
    rect_geom = sm.make_rect(2.0, 3.0, 8.0, 4.0, numkey=5, layer=13)
    assert isinstance(rect_geom, sp.GeomGroup)
    assert len(rect_geom.group) == 1
    poly = rect_geom.group[0]
    assert isinstance(poly, sp.Poly)
    assert poly.layer == 13
    bb = rect_geom.bounding_box()
    assert bb.llx == pytest.approx(-2.0)
    assert bb.lly == pytest.approx(1.0)
    assert bb.width == pytest.approx(8.0)
    assert bb.height == pytest.approx(4.0)


def test_make_rect_noncenter_numkey() -> None:
    rect_geom = sm.make_rect(2.0, 3.0, 8.0, 4.0, numkey=1, layer=14)
    bb = rect_geom.bounding_box()
    assert bb.llx == pytest.approx(2.0)
    assert bb.lly == pytest.approx(3.0)
    assert bb.width == pytest.approx(8.0)
    assert bb.height == pytest.approx(4.0)


def test_make_rounded_rect() -> None:
    rounded_rect_geom = sm.make_rounded_rect(
        x0=0.0,
        y0=0.0,
        width=10.0,
        height=6.0,
        corner_radius=1.0,
        resolution=8,
        numkey=5,
        layer=15,
    )
    assert isinstance(rounded_rect_geom, sp.GeomGroup)
    assert len(rounded_rect_geom.group) == 1
    assert all(isinstance(poly, sp.Poly) for poly in rounded_rect_geom.group)
    assert rounded_rect_geom.get_layer_list() == {15}
    bb = rounded_rect_geom.bounding_box()
    assert bb.width == pytest.approx(10.0)
    assert bb.height == pytest.approx(6.0)


def test_make_rounded_rect_noncenter_numkey() -> None:
    rounded_rect_geom = sm.make_rounded_rect(
        x0=2.0,
        y0=3.0,
        width=10.0,
        height=6.0,
        corner_radius=0.8,
        resolution=8,
        numkey=9,
        layer=16,
    )
    bb = rounded_rect_geom.bounding_box()
    assert bb.urx() == pytest.approx(2.0, abs=1e-3)
    assert bb.ury() == pytest.approx(3.0, abs=1e-3)
    assert bb.width == pytest.approx(10.0, abs=1e-3)
    assert bb.height == pytest.approx(6.0, abs=1e-3)


@pytest.mark.parametrize(
    "xpts,ypts,widths",
    [
        ([1.0], [2.0], [0.6]),
        ([0.0, 3.0], [0.0, 0.0], [0.4, 0.8]),
        ([0.0, 2.0, 4.0], [0.0, 1.0, 0.0], [0.5, 1.0, 0.5]),
    ],
)
def test_make_tapered_path(
    xpts: list[float], ypts: list[float], widths: list[float]
) -> None:
    tapered_geom = sm.make_tapered_path(xpts, ypts, widths, layer=17)
    assert isinstance(tapered_geom, sp.GeomGroup)
    assert len(tapered_geom.group) == 1
    poly = tapered_geom.group[0]
    assert isinstance(poly, sp.Poly)
    assert poly.layer == 17
    assert poly.Npts >= 5

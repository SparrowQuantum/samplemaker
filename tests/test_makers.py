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

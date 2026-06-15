"""Unit tests for phc module."""

import numpy as np
import pytest

import samplemaker.makers as sm
from samplemaker.phc import Crystal, make_phc, make_phc_inpoly
from samplemaker.shapes import Circle, GeomGroup, Poly


@pytest.fixture
def crystal_three_sites() -> Crystal:
    return Crystal(
        xpts=[-1.0, 0.5, 2.0],
        ypts=[0.0, 1.5, -2.0],
        params=[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
    )


def test_crystal_init_defaults() -> None:
    c = Crystal()
    assert c.xpts.dtype == np.float64
    assert c.ypts.dtype == np.float64
    assert c.params.dtype == np.float64
    assert c.xpts.size == 0
    assert c.ypts.size == 0
    assert c.params.size == 0


def test_crystal_init_with_data() -> None:
    c = Crystal(xpts=[0, 1], ypts=[2, 3], params=[[4, 5]])
    assert list(c.xpts) == pytest.approx([0.0, 1.0])
    assert list(c.ypts) == pytest.approx([2.0, 3.0])
    assert c.params.shape == (1, 2)
    assert list(c.params[0]) == pytest.approx([4.0, 5.0])


def test_remove_at_index_removes_sites(crystal_three_sites: Crystal) -> None:
    crystal_three_sites.remove_at_index([1])
    assert list(crystal_three_sites.xpts) == pytest.approx([-1.0, 2.0])
    assert list(crystal_three_sites.ypts) == pytest.approx([0.0, -2.0])
    assert crystal_three_sites.params.shape == (2, 2)
    assert list(crystal_three_sites.params[0]) == pytest.approx([1.0, 3.0])


def test_remove_at_index_empty_is_noop(crystal_three_sites: Crystal) -> None:
    original_x = crystal_three_sites.xpts.copy()
    original_y = crystal_three_sites.ypts.copy()
    original_p = crystal_three_sites.params.copy()

    crystal_three_sites.remove_at_index([])

    assert np.array_equal(crystal_three_sites.xpts, original_x)
    assert np.array_equal(crystal_three_sites.ypts, original_y)
    assert np.array_equal(crystal_three_sites.params, original_p)


def test_shift_at_index_absolute(crystal_three_sites: Crystal) -> None:
    crystal_three_sites.shift_at_index([0, 2], shift_x=0.25, shift_y=-0.5)
    assert list(crystal_three_sites.xpts) == pytest.approx([-0.75, 0.5, 2.25])
    assert list(crystal_three_sites.ypts) == pytest.approx([-0.5, 1.5, -2.5])


def test_shift_at_index_relative() -> None:
    c = Crystal(
        xpts=[-1.0, 0.5, 0.0],
        ypts=[2.0, -2.0, 0.0],
        params=[[1.0, 1.0, 1.0]],
    )
    c.shift_at_index([0, 1, 2], shift_x=0.3, shift_y=0.4, relative=True)

    assert list(c.xpts) == pytest.approx([-1.3, 0.8, -0.3])
    assert list(c.ypts) == pytest.approx([2.4, -2.4, -0.4])


def test_param_at_index_updates_value(crystal_three_sites: Crystal) -> None:
    crystal_three_sites.param_at_index(index=2, pindex=1, pvalues=123.0)
    assert crystal_three_sites.params[1, 2] == pytest.approx(123.0)


def test_coord_to_index_matches_with_tolerance(crystal_three_sites: Crystal) -> None:
    indices = crystal_three_sites.coord_to_index(xc=[0.5 + 5e-7], yc=[1.5 - 5e-7])
    assert indices == [1]


def test_coord_to_index_warns_on_missing(crystal_three_sites: Crystal) -> None:
    with pytest.warns(UserWarning, match="No coordinate match"):
        indices = crystal_three_sites.coord_to_index(xc=[99.0], yc=[99.0])
    assert indices == []


def test_add_and_remove_crystal() -> None:
    c1 = Crystal(xpts=[0.0], ypts=[0.0], params=[[1.0]])
    c2 = Crystal(xpts=[1.0], ypts=[2.0], params=[[3.0]])

    c1.add_crystal(c2)
    assert list(c1.xpts) == pytest.approx([0.0, 1.0])
    assert list(c1.ypts) == pytest.approx([0.0, 2.0])
    assert list(c1.params[0]) == pytest.approx([1.0, 3.0])

    c1.remove_crystal(c2)
    assert list(c1.xpts) == pytest.approx([0.0])
    assert list(c1.ypts) == pytest.approx([0.0])
    assert list(c1.params[0]) == pytest.approx([1.0])


def test_copy_is_deep(crystal_three_sites: Crystal) -> None:
    copied = crystal_three_sites.copy()
    copied.shift_at_index([0], shift_x=5.0, shift_y=6.0)
    copied.param_at_index(index=0, pindex=0, pvalues=999.0)

    assert crystal_three_sites.xpts[0] == pytest.approx(-1.0)
    assert crystal_three_sites.ypts[0] == pytest.approx(0.0)
    assert crystal_three_sites.params[0, 0] == pytest.approx(1.0)


def test_triangular_hexagonal_N0() -> None:
    c = Crystal.triangular_hexagonal(N=0, filled=False, Nparams=2)
    assert list(c.xpts) == pytest.approx([0.0])
    assert list(c.ypts) == pytest.approx([0.0])
    assert c.params.shape == (2, 1)
    assert list(c.params[:, 0]) == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("N", [1, 2, 4])
def test_triangular_hexagonal_ring_size(N: int) -> None:
    c = Crystal.triangular_hexagonal(N=N, filled=False)
    assert c.xpts.size == 6 * N
    assert c.ypts.size == 6 * N


@pytest.mark.parametrize("N", [1, 2, 3])
def test_triangular_hexagonal_filled_size(N: int) -> None:
    c = Crystal.triangular_hexagonal(N=N, filled=True)
    expected_points = 1 + 3 * N * (N - 1)
    assert c.xpts.size == expected_points
    assert c.ypts.size == expected_points


def test_triangular_box_basic_shape() -> None:
    c = Crystal.triangular_box(Nx=1, Ny=1, Nparams=2)
    assert c.xpts.size == 13
    assert c.ypts.size == 13
    assert c.params.shape == (2, 13)
    assert np.max(c.ypts) == pytest.approx(np.sqrt(3))
    assert np.min(c.ypts) == pytest.approx(-np.sqrt(3))


@pytest.mark.xfail(
    reason="Known bug in triangular_box: uses bitwise '&' in zero-dimension check.",
    strict=True,
)
def test_triangular_box_Nx0_Ny1_is_not_single_origin_site() -> None:
    c = Crystal.triangular_box(Nx=0, Ny=1, Nparams=1)
    assert c.xpts.size > 1
    assert not (c.xpts.size == 1 and c.ypts.size == 1)


def test_triangular_heterophc_is_symmetric_for_integer_Nx() -> None:
    c = Crystal.triangular_heterophc(
        Nx=3,
        Ny=1,
        spacing=[0.9, 1.0],
        periods=[1, 1],
    )
    assert np.max(c.xpts) == pytest.approx(-np.min(c.xpts))
    assert c.params.shape[1] == c.xpts.size


@pytest.mark.xfail(
    reason=(
        "Known bug in triangular_heterophc fractional trimming: min bound is taken "
        "from ypts instead of xpts."
    ),
    strict=True,
)
def test_triangular_heterophc_fractional_Nx_remains_x_symmetric() -> None:
    c = Crystal.triangular_heterophc(
        Nx=2.5,
        Ny=1,
        spacing=[1.0],
        periods=[1],
    )
    assert np.max(c.xpts) == pytest.approx(-np.min(c.xpts))


def test_make_phc_uses_scaled_coordinates_and_translates() -> None:
    c = Crystal(
        xpts=[0.0, 2.0],
        ypts=[1.0, -1.0],
        params=[[1.0, 2.0], [3.0, 4.0]],
    )
    calls: list[tuple[float, float, list[float]]] = []

    def custom_cellfun(x: float, y: float, params: list[float]) -> GeomGroup:
        calls.append((x, y, params))
        return sm.make_circle(x, y, params[0], layer=7)

    g = make_phc(
        crystal=c,
        scaling=2.5,
        cellparams=[10.0, 100.0],
        x0=1.0,
        y0=-2.0,
        cellfun=custom_cellfun,
    )

    assert isinstance(g, GeomGroup)
    assert len(calls) == 2
    assert calls[0][0] == pytest.approx(0.0)
    assert calls[0][1] == pytest.approx(2.5)
    assert calls[0][2] == pytest.approx([10.0, 300.0])
    assert calls[1][0] == pytest.approx(5.0)
    assert calls[1][1] == pytest.approx(-2.5)
    assert calls[1][2] == pytest.approx([20.0, 400.0])

    circles = [shape for shape in g.group if isinstance(shape, Circle)]
    assert len(circles) == 2
    assert circles[0].x0 == pytest.approx(1.0)
    assert circles[0].y0 == pytest.approx(0.5)
    assert circles[0].r == pytest.approx(10.0)
    assert circles[1].x0 == pytest.approx(6.0)
    assert circles[1].y0 == pytest.approx(-4.5)
    assert circles[1].r == pytest.approx(20.0)


def test_make_phc_inpoly_filters_sites() -> None:
    c = Crystal(
        xpts=[0.0, 2.0, -2.0],
        ypts=[0.0, 0.0, 0.0],
        params=[[1.0, 2.0, 3.0]],
    )
    poly = Poly(
        xpts=[-1.0, 1.0, 1.0, -1.0],
        ypts=[-1.0, -1.0, 1.0, 1.0],
        layer=1,
    )

    def custom_cellfun(x: float, y: float, params: list[float]) -> GeomGroup:
        return sm.make_circle(x, y, params[0], layer=3)

    g = make_phc_inpoly(
        crystal=c,
        poly=poly,
        scaling=1.0,
        cellparams=[5.0],
        x0=10.0,
        y0=20.0,
        cellfun=custom_cellfun,
    )

    circles = [shape for shape in g.group if isinstance(shape, Circle)]
    assert len(circles) == 1
    assert circles[0].x0 == pytest.approx(10.0)
    assert circles[0].y0 == pytest.approx(20.0)
    assert circles[0].r == pytest.approx(5.0)

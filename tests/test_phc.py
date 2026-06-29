"""Unit tests for phc module."""

from collections.abc import Sequence

import numpy as np
import pytest

from samplemaker import LayoutPool
import samplemaker.makers as sm
from samplemaker.phc import (
    Crystal,
    make_phc,
    make_phc_circle,
    make_phc_circle_ref,
    make_phc_inpoly,
)
from samplemaker.shapes import Circle, GeomGroup, Poly, SRef


def _site_set(
    xpts: np.ndarray, ypts: np.ndarray, digits: int = 9
) -> set[tuple[float, float]]:
    rounded_x = np.round(xpts.astype(float), digits)
    rounded_y = np.round(ypts.astype(float), digits)
    return set(zip(rounded_x.tolist(), rounded_y.tolist(), strict=True))


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


def test_crystal_copy_is_deep(crystal_three_sites: Crystal) -> None:
    copied = crystal_three_sites.copy()
    copied.shift_at_index([0], shift_x=5.0, shift_y=6.0)
    copied.param_at_index(index=0, pindex=0, pvalues=999.0)

    assert crystal_three_sites.xpts[0] == pytest.approx(-1.0)
    assert crystal_three_sites.ypts[0] == pytest.approx(0.0)
    assert crystal_three_sites.params[0, 0] == pytest.approx(1.0)


def test_triangular_hexagonal_n0() -> None:
    c = Crystal.triangular_hexagonal(n=0, filled=False, nparams=2)
    assert list(c.xpts) == pytest.approx([0.0])
    assert list(c.ypts) == pytest.approx([0.0])
    assert c.params.shape == (2, 1)
    assert list(c.params[:, 0]) == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("n", [1, 2, 4])
def test_triangular_hexagonal_ring_size(n: int) -> None:
    c = Crystal.triangular_hexagonal(n=n, filled=False)
    assert c.xpts.size == 6 * n
    assert c.ypts.size == 6 * n


def test_triangular_hexagonal_ring_n1_exact_coordinates() -> None:
    c = Crystal.triangular_hexagonal(n=1, filled=False)
    expected = {
        (1.0, 0.0),
        (0.5, np.sqrt(3) / 2),
        (-0.5, np.sqrt(3) / 2),
        (-1.0, 0.0),
        (-0.5, -np.sqrt(3) / 2),
        (0.5, -np.sqrt(3) / 2),
    }
    expected_site_set = _site_set(
        np.array([p[0] for p in expected]), np.array([p[1] for p in expected])
    )

    assert _site_set(c.xpts, c.ypts) == expected_site_set


@pytest.mark.parametrize("n", [1, 2, 3])
def test_triangular_hexagonal_filled_size(n: int) -> None:
    c = Crystal.triangular_hexagonal(n=n, filled=True)
    expected_points = 1 + 3 * n * (n - 1)
    assert c.xpts.size == expected_points
    assert c.ypts.size == expected_points


def test_triangular_hexagonal_filled_n2_exact_coordinates() -> None:
    c = Crystal.triangular_hexagonal(n=2, filled=True)
    expected = {
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, np.sqrt(3) / 2),
        (-0.5, np.sqrt(3) / 2),
        (-1.0, 0.0),
        (-0.5, -np.sqrt(3) / 2),
        (0.5, -np.sqrt(3) / 2),
    }
    expected_site_set = _site_set(
        np.array([p[0] for p in expected]), np.array([p[1] for p in expected])
    )
    assert _site_set(c.xpts, c.ypts) == expected_site_set


def test_triangular_box_basic_shape() -> None:
    c = Crystal.triangular_box(nx=1, ny=1, nparams=2)
    assert c.xpts.size == 13
    assert c.ypts.size == 13
    assert c.params.shape == (2, 13)
    assert np.max(c.ypts) == pytest.approx(np.sqrt(3))
    assert np.min(c.ypts) == pytest.approx(-np.sqrt(3))


def test_triangular_box_n1_n1_exact_coordinates() -> None:
    c = Crystal.triangular_box(nx=1, ny=1, nparams=1)
    s3 = np.sqrt(3)
    expected_x1 = [-1.0, 0.0, 1.0]
    expected_y1 = [-s3, 0.0, s3]
    expected_x2 = [-0.5, 0.5]
    expected_y2 = [-s3 / 2, s3 / 2]

    expected: set[tuple[float, float]] = set()
    for y in expected_y1:
        for x in expected_x1:
            expected.add((x, y))
    for y in expected_y2:
        for x in expected_x2:
            expected.add((x, y))

    expected_site_set = _site_set(
        np.array([p[0] for p in expected]), np.array([p[1] for p in expected])
    )
    assert _site_set(c.xpts, c.ypts) == expected_site_set


def test_triangular_box_nx0_ny1_exact_coordinates() -> None:
    c = Crystal.triangular_box(nx=0, ny=1, nparams=1)
    assert c.ypts.size == 3
    assert c.params.shape == (1, 3)
    assert c.xpts == pytest.approx([0.0, 0.0, 0.0])
    assert c.ypts == pytest.approx([-np.sqrt(3), 0.0, np.sqrt(3)])


def test_triangular_heterophc_is_symmetric_for_integer_nx() -> None:
    c = Crystal.triangular_heterophc(
        nx=3,
        ny=1,
        spacing=[0.9, 1.0],
        periods=[1, 1],
    )
    assert np.max(c.xpts) == pytest.approx(-np.min(c.xpts))
    assert c.params.shape[1] == c.xpts.size


def test_triangular_heterophc_ny0_exact_coordinates() -> None:
    c = Crystal.triangular_heterophc(
        nx=4,
        ny=0,
        spacing=[0.8, 1.1],
        periods=[1, 2],
        nparams=1,
    )
    expected_x = np.array([-4.0, -3.0, -1.9, -0.8, 0.0, 0.8, 1.9, 3.0, 4.0])
    expected_y = np.zeros_like(expected_x)
    assert _site_set(c.xpts, c.ypts) == _site_set(expected_x, expected_y)


def test_triangular_heterophc_uniform_spacing_row_structure() -> None:
    c = Crystal.triangular_heterophc(
        nx=2,
        ny=1,
        spacing=[1.0],
        periods=[1],
        nparams=1,
    )

    s3 = np.sqrt(3)
    expected: set[tuple[float, float]] = set()
    x1 = [-2.0, -1.0, 0.0, 1.0, 2.0]
    x2 = [-1.5, -0.5, 0.5, 1.5]

    for y in (-s3, s3):
        for x in x1:
            expected.add((x, y))
    for y in (-s3 / 2, s3 / 2):
        for x in x2:
            expected.add((x, y))

    expected_site_set = _site_set(
        np.array([p[0] for p in expected]), np.array([p[1] for p in expected])
    )
    assert _site_set(c.xpts, c.ypts) == expected_site_set


def test_triangular_heterophc_fractional_nx_remains_x_symmetric() -> None:
    c = Crystal.triangular_heterophc(
        nx=2.5,
        ny=1,
        spacing=[1.0],
        periods=[1],
    )
    assert np.max(c.xpts) == pytest.approx(-np.min(c.xpts))


def test_make_phc_circle_creates_expected_circle() -> None:
    circle_geom = make_phc_circle(x=1.0, y=2.0, params=[3.0])

    assert isinstance(circle_geom, GeomGroup)
    assert len(circle_geom.group) == 1
    circle = circle_geom.group[0]
    assert isinstance(circle, Circle)
    assert circle.x0 == pytest.approx(1.0)
    assert circle.y0 == pytest.approx(2.0)
    assert circle.r == pytest.approx(3.0)


def test_make_phc_circle_ref_creates_expected_circle() -> None:
    reference_circle = make_phc_circle(x=0.0, y=0.0, params=[1.0])
    LayoutPool["_CIRCLE"] = reference_circle

    circle_geom = make_phc_circle_ref(x=1.0, y=2.0, params=[3.0])

    assert isinstance(circle_geom, GeomGroup)
    assert len(circle_geom.group) == 1
    circle_ref = circle_geom.group[0]
    assert isinstance(circle_ref, SRef)
    assert circle_ref.x0 == pytest.approx(1.0)
    assert circle_ref.y0 == pytest.approx(2.0)
    assert circle_ref.mag == pytest.approx(3.0)
    assert circle_ref.angle == pytest.approx(0.0)
    assert circle_ref.mirror is False
    assert circle_ref.cellname == "_CIRCLE"
    assert circle_ref.group is reference_circle


def test_make_phc_uses_scaled_coordinates_and_translates() -> None:
    c = Crystal(
        xpts=[0.0, 2.0],
        ypts=[1.0, -1.0],
        params=[[1.0, 2.0]],
    )
    calls: list[tuple[float, float, list[float]]] = []

    def custom_cellfun(x: float, y: float, params: Sequence[float]) -> GeomGroup:
        calls.append((x, y, list(params)))
        return sm.make_circle(x, y, params[0], layer=7)

    g = make_phc(
        crystal=c,
        scaling=2.5,
        cellparams=[10.0],
        x0=1.0,
        y0=-2.0,
        cellfun=custom_cellfun,
    )

    assert isinstance(g, GeomGroup)
    assert len(calls) == 2
    assert calls[0][0] == pytest.approx(0.0)
    assert calls[0][1] == pytest.approx(2.5)
    assert calls[0][2] == pytest.approx([10.0])
    assert calls[1][0] == pytest.approx(5.0)
    assert calls[1][1] == pytest.approx(-2.5)
    assert calls[1][2] == pytest.approx([20.0])

    circles = [shape for shape in g.group if isinstance(shape, Circle)]
    assert len(circles) == 2
    assert circles[0].x0 == pytest.approx(1.0)
    assert circles[0].y0 == pytest.approx(0.5)
    assert circles[0].r == pytest.approx(10.0)
    assert circles[1].x0 == pytest.approx(6.0)
    assert circles[1].y0 == pytest.approx(-4.5)
    assert circles[1].r == pytest.approx(20.0)


def test_make_phc_returns_empty_group_for_empty_crystal() -> None:
    c = Crystal()
    g = make_phc(
        crystal=c,
        scaling=1.0,
        cellparams=[5.0],
        x0=0.0,
        y0=0.0,
    )
    assert isinstance(g, GeomGroup)
    assert len(g.group) == 0


def test_make_phc_raises_on_invalid_crystal() -> None:
    c_pts_mismatch = Crystal([1.0, 2.0], [3.0], [[1.0]])
    c_incorrect_params_ndims = Crystal([1.0], [2.0], [1.0])  # type: ignore[arg-type]
    c_incorrect_params_shape = Crystal([1.0, 2.0], [2.0, 2.0], [[1.0], [2.0]])

    with pytest.raises(ValueError, match="x-coordinates must match"):
        make_phc(
            crystal=c_pts_mismatch,
            scaling=1.0,
            cellparams=[5.0],
            x0=0.0,
            y0=0.0,
        )
    with pytest.raises(ValueError, match="params array must be 2-dimensional"):
        make_phc(
            crystal=c_incorrect_params_ndims,
            scaling=1.0,
            cellparams=[5.0],
            x0=0.0,
            y0=0.0,
        )
    with pytest.raises(
        ValueError, match="parameter sets must match the number of lattice sites"
    ):
        make_phc(
            crystal=c_incorrect_params_shape,
            scaling=1.0,
            cellparams=[5.0],
            x0=0.0,
            y0=0.0,
        )


def test_make_phc_raises_on_mismatched_cellparams() -> None:
    c = Crystal(
        xpts=[0.0, 1.0],
        ypts=[0.0, 1.0],
        params=[[1.0, 2.0], [3.0, 4.0]],
    )
    with pytest.raises(
        ValueError,
        match="The number of cell parameters must match the number of parameter sets",
    ):
        make_phc(
            crystal=c,
            scaling=1.0,
            cellparams=[5.0],  # Only one set of cell parameters
            x0=0.0,
            y0=0.0,
        )


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

    def custom_cellfun(x: float, y: float, params: Sequence[float]) -> GeomGroup:
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


def test_make_phc_inpoly_returns_empty_group_for_empty_crystal() -> None:
    c = Crystal()
    poly = Poly(xpts=[0.0, 1.0], ypts=[0.0, 1.0], layer=1)
    g = make_phc_inpoly(
        crystal=c,
        poly=poly,
        scaling=1.0,
        cellparams=[5.0],
        x0=0.0,
        y0=0.0,
    )
    assert isinstance(g, GeomGroup)
    assert len(g.group) == 0


def test_make_phc_inpoly_raises_on_invalid_crystal() -> None:
    c_pts_mismatch = Crystal([1.0, 2.0], [3.0], [[1.0]])
    c_incorrect_params_ndims = Crystal([1.0], [2.0], [1.0])  # type: ignore[arg-type]
    c_incorrect_params_shape = Crystal([1.0, 2.0], [2.0, 2.0], [[1.0], [2.0]])
    poly = Poly(xpts=[0.0, 1.0], ypts=[0.0, 1.0], layer=1)

    with pytest.raises(ValueError, match="x-coordinates must match"):
        make_phc_inpoly(
            crystal=c_pts_mismatch,
            poly=poly,
            scaling=1.0,
            cellparams=[5.0],
            x0=0.0,
            y0=0.0,
        )
    with pytest.raises(ValueError, match="params array must be 2-dimensional"):
        make_phc_inpoly(
            crystal=c_incorrect_params_ndims,
            poly=poly,
            scaling=1.0,
            cellparams=[5.0],
            x0=0.0,
            y0=0.0,
        )
    with pytest.raises(
        ValueError, match="parameter sets must match the number of lattice sites"
    ):
        make_phc_inpoly(
            crystal=c_incorrect_params_shape,
            poly=poly,
            scaling=1.0,
            cellparams=[5.0],
            x0=0.0,
            y0=0.0,
        )


def test_make_phc_inpoly_raises_on_mismatched_cellparams() -> None:
    c = Crystal(
        xpts=[0.0, 1.0],
        ypts=[0.0, 1.0],
        params=[[1.0, 2.0], [3.0, 4.0]],
    )
    poly = Poly(xpts=[-1.0, 1.0], ypts=[-1.0, 1.0], layer=1)
    with pytest.raises(
        ValueError,
        match="The number of cell parameters must match the number of parameter sets",
    ):
        make_phc_inpoly(
            crystal=c,
            poly=poly,
            scaling=1.0,
            cellparams=[5.0],  # Only one set of cell parameters
            x0=0.0,
            y0=0.0,
        )

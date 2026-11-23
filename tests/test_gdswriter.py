"""Unit tests for the samplemaker.gdswriter module."""

# Used automatically by pytest to reset state before each test:
import math
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

import samplemaker.makers as sm
import samplemaker.shapes as sp
from samplemaker import LayoutPool
from samplemaker.baselib.devices import CrossMark
from samplemaker.devices import Device
from samplemaker.gdsreader import GDSReader
from samplemaker.gdswriter import GDSWriter
from samplemaker.shapes import GeomGroup, Poly
from samplemaker.shapes import Path as GeomPath


@pytest.fixture
def gds_writer() -> GDSWriter:
    return GDSWriter()


@pytest.fixture
def cell_name() -> str:
    return "CELL00"


@pytest.fixture
def sample_device() -> Device:
    return CrossMark.build()


@pytest.fixture
def sample_geometry(sample_device: Device) -> GeomGroup:
    dev_geom: GeomGroup = sample_device.run()
    rect = sm.make_rect(0, 0, 10, 5, layer=1)
    circle = sm.make_circle(10, 10, 1.2, layer=2, to_poly=False)
    path = sm.make_path([0, 2, 4], [0, 1, 0], width=0.4, layer=3, to_poly=False)
    text = sm.make_text(1.0, 1.0, "GDS", height=3.0, width=0.3, numkey=1, layer=4)
    rounded_rect = sm.make_rounded_rect(-3, 8, 4, 4, 1, layer=9)
    referenced_group = sm.make_poly([0, 1, 1, 0], [0, 0, 1, 1], layer=5)
    LayoutPool["SUBCELL"] = referenced_group
    sref = sm.make_sref(
        x0=20.0,
        y0=20.0,
        cellname="SUBCELL",
        group=LayoutPool["SUBCELL"],
        mag=1.0,
        angle=30.0,
        mirror=False,
    )
    aref = sm.make_aref(
        x0=30.0,
        y0=30.0,
        cellname="SUBCELL",
        group=LayoutPool["SUBCELL"],
        ncols=2,
        nrows=2,
        ax=4.0,
        ay=0.0,
        bx=0.0,
        by=4.0,
        mag=1.1,
        angle=0.0,
        mirror=False,
    )
    ellipse = sm.make_ellipse(
        40.0,
        10.0,
        rx=2.0,
        ry=1.0,
        rot=15.0,
        layer=6,
        to_poly=False,
    )
    ring = sm.make_ring(
        50.0,
        10.0,
        rx=2.5,
        ry=1.5,
        rot=0.0,
        w=0.4,
        layer=7,
        to_poly=False,
    )
    arc = sm.make_arc(
        60.0,
        10.0,
        rx=2.5,
        ry=1.5,
        rot=0.0,
        w=0.4,
        a1=0,
        a2=120,
        layer=8,
        to_poly=False,
    )
    return (
        rect
        + circle
        + path
        + text
        + rounded_rect
        + sref
        + aref
        + ellipse
        + ring
        + arc
        + dev_geom
    )


@pytest.fixture
def layout_pool(
    cell_name: str,
    sample_geometry: GeomGroup,
) -> Generator[dict[str, GeomGroup], None, None]:
    original_layout_pool = LayoutPool.copy()
    LayoutPool[cell_name] = sample_geometry
    yield LayoutPool

    LayoutPool.clear()
    LayoutPool.update(original_layout_pool)


def test_layoutpool(layout_pool: dict[str, GeomGroup]) -> None:
    assert "CELL00" in layout_pool
    assert "BASELIB_CMARK_0001" in layout_pool
    assert "SUBCELL" in layout_pool
    assert all(isinstance(group, GeomGroup) for group in layout_pool.values())
    srefs = layout_pool["CELL00"].get_sref_list()
    assert srefs == {"BASELIB_CMARK_0001", "SUBCELL"}


def test_init_populates_standard_attributes(gds_writer: GDSWriter) -> None:
    assert gds_writer._circleres == 12
    assert gds_writer._arcres == 32
    assert gds_writer.fid is None
    assert gds_writer.xc.shape == (gds_writer.circleres,)
    assert gds_writer.yc.shape == (gds_writer.circleres,)


def test_circleres_setter_samples_circles(gds_writer: GDSWriter) -> None:
    # Default case
    expected_xc = np.array([math.cos(i * 2 * math.pi / 12) for i in range(12)])
    expected_yc = np.array([math.sin(i * 2 * math.pi / 12) for i in range(12)])
    np.testing.assert_allclose(gds_writer.xc, expected_xc)
    np.testing.assert_allclose(gds_writer.yc, expected_yc)

    gds_writer.circleres = 16
    expected_xc = np.array([math.cos(i * 2 * math.pi / 16) for i in range(16)])
    expected_yc = np.array([math.sin(i * 2 * math.pi / 16) for i in range(16)])
    np.testing.assert_allclose(gds_writer.xc, expected_xc)
    np.testing.assert_allclose(gds_writer.yc, expected_yc)


def test_circleres_arcres_setters_reject_invalid_values(gds_writer: GDSWriter) -> None:
    with pytest.raises(ValueError, match=r"circleres must be at least 3. Got 2."):
        gds_writer.circleres = 2
    with pytest.raises(ValueError, match=r"arcres must be at least 3. Got 2."):
        gds_writer.arcres = 2


def test_open_library_and_close_library_write_padded_gds(
    gds_writer: GDSWriter, tmp_path: Path
) -> None:
    output_file = tmp_path / "library.gds"
    gds_writer.open_library(str(output_file))
    gds_writer.close_library()

    data = output_file.read_bytes()
    assert len(data) > 0
    assert len(data) % 2048 == 0


def test_open_structure_and_close_structure_emit_structure_name(
    gds_writer: GDSWriter, tmp_path: Path
) -> None:
    output_file = tmp_path / "structure.gds"
    structure_name = "TOPCELL"

    gds_writer.open_library(str(output_file))
    gds_writer.open_structure(structure_name)
    gds_writer.close_structure()
    gds_writer.close_library()

    data = output_file.read_bytes()
    assert structure_name.encode() in data


def test_write_structure_supports_all_written_shape_types(
    gds_writer: GDSWriter, layout_pool: dict[str, GeomGroup], tmp_path: Path
) -> None:
    output_file = tmp_path / "mixed.gds"

    gds_writer.open_library(str(output_file))
    for cellname, group in layout_pool.items():
        gds_writer.write_structure(cellname, group)
    gds_writer.close_library()

    data = output_file.read_bytes()
    for cellname in layout_pool:
        assert cellname.encode() in data


@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize(("sx", "sy"), [(-2.0, 2.0), (2.0, -2.0), (-2.0, -2.0)])
def test_write_negatively_scaled_reference(
    gds_writer: GDSWriter, tmp_path: Path, array: bool, sx: float, sy: float
) -> None:
    source = sp.Box(0, 1, 2, 3).to_rect()
    if array:
        ref: sp.SRef = sp.ARef(1, 2, "SUB", source, 2, 2, 4, 1, -2, 5, 1, 30, True)
    else:
        ref = sp.SRef(1, 2, "SUB", source, 1, 30, True)
    ref.scale(-3, 4, sx, sy)
    expected = ref.to_polygon()
    top = GeomGroup()
    top.add(ref)

    output_file = tmp_path / "scaled_ref.gds"
    gds_writer.open_library(str(output_file))
    gds_writer.write_structure("TOP", top)
    gds_writer.close_library()

    reader = GDSReader()
    reader.quick_read(str(output_file))
    decoded = reader.get_cell("TOP").group[0]
    assert isinstance(decoded, sp.ARef if array else sp.SRef)
    decoded.group = source
    actual = decoded.to_polygon()
    assert len(actual.group) == len(expected.group)
    for actual_poly, expected_poly in zip(actual.group, expected.group, strict=True):
        assert isinstance(actual_poly, Poly)
        assert isinstance(expected_poly, Poly)
        np.testing.assert_allclose(actual_poly.data, expected_poly.data, atol=1e-10)


@pytest.mark.parametrize("npts", [7999, 8000, 8001, 9000])
def test_write_structure_preserves_large_polygon_geometry(
    gds_writer: GDSWriter, tmp_path: Path, npts: int
) -> None:
    bottom_x = np.linspace(0.0, 16.0, npts - 5)
    polygon = Poly(
        np.concatenate((bottom_x, [16.0, 6.0, 6.0, 0.0])) + 0.0004,
        np.concatenate((np.zeros_like(bottom_x), [2.0, 2.0, 7.0, 7.0])) + 0.0004,
        layer=7,
    )
    assert polygon.npts == npts
    small_polygon = Poly([20, 22, 22, 20], [1, 1, 3, 3], layer=8)
    path = GeomPath([24.0, 25.0], [1.0, 3.0], width=0.5, layer=9)
    group = GeomGroup()
    for geom in (polygon, small_polygon, path):
        group.add(geom)

    output_file = tmp_path / "large_polygon.gds"
    gds_writer.open_library(str(output_file))
    gds_writer.write_structure("TOP", group)
    gds_writer.close_library()

    reader = GDSReader()
    reader.quick_read(str(output_file))
    result = reader.get_cell("TOP")

    large_output = result.select_layer(7)
    if npts <= 8000:
        assert len(large_output.group) == 1
    else:
        assert len(large_output.group) > 1
    for geom in large_output.group:
        assert isinstance(geom, Poly)
        assert geom.npts <= 8000
    assert large_output.get_area() == pytest.approx(62.0)

    expected = GeomGroup()
    expected.add(Poly([0, 16, 16, 6, 6, 0], [0, 0, 2, 2, 7, 7], layer=7))
    expected.boolean_xor(large_output, layer_a=7, layer_b=7)
    assert not expected.group

    if npts <= 8000:
        unsplit = large_output.group[0]
        assert isinstance(unsplit, Poly)
        np.testing.assert_array_equal(unsplit.int_data(), polygon.int_data())
    small_output = result.select_layer(8).group
    assert len(small_output) == 1
    assert isinstance(small_output[0], Poly)
    np.testing.assert_array_equal(small_output[0].int_data(), small_polygon.int_data())
    path_output = result.select_layer(9).group
    assert len(path_output) == 1
    assert isinstance(path_output[0], GeomPath)
    assert path_output[0].xpts == pytest.approx(path.xpts)
    assert path_output[0].ypts == pytest.approx(path.ypts)
    assert path_output[0].width == pytest.approx(path.width)


def test_write_structure_calls_open_write_close_in_order(
    gds_writer: GDSWriter,
    sample_geometry: GeomGroup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_open_structure(structure_name: str) -> None:
        calls.append(f"open:{structure_name}")

    def fake_write_geomgroup(group: GeomGroup) -> None:
        assert group is sample_geometry
        calls.append("write")

    def fake_close_structure() -> None:
        calls.append("close")

    monkeypatch.setattr(gds_writer, "open_structure", fake_open_structure)
    monkeypatch.setattr(gds_writer, "write_geomgroup", fake_write_geomgroup)
    monkeypatch.setattr(gds_writer, "close_structure", fake_close_structure)

    gds_writer.write_structure("ORDERED", sample_geometry)

    assert calls == ["open:ORDERED", "write", "close"]


def test_write_pool_writes_all_named_groups(
    gds_writer: GDSWriter,
    layout_pool: dict[str, GeomGroup],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, GeomGroup]] = []

    def fake_write_structure(structure_name: str, group: GeomGroup) -> None:
        calls.append((structure_name, group))

    monkeypatch.setattr(gds_writer, "write_structure", fake_write_structure)
    gds_writer.write_pool(layout_pool)

    assert len(calls) == len(layout_pool)
    assert {name for name, _ in calls} == set(layout_pool.keys())


def test_write_pool_use_cache_writes_cached_binary_and_non_cached_structures(
    gds_writer: GDSWriter,
    layout_pool: dict[str, GeomGroup],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_file = tmp_path / "cache.gds"
    cached_name = next(iter(layout_pool))
    cache_data = b"CACHED_STRUCTURE_BINARY"
    cache = {cached_name: cache_data}
    fallback_calls: list[str] = []

    def fake_write_structure(structure_name: str, group: GeomGroup) -> None:
        _ = group
        fallback_calls.append(structure_name)

    monkeypatch.setattr(gds_writer, "write_structure", fake_write_structure)

    gds_writer.open_library(str(output_file))
    gds_writer.write_pool_use_cache(layout_pool, cache)
    gds_writer.close_library()

    written_data = output_file.read_bytes()
    assert cache_data in written_data
    assert fallback_calls == [name for name in layout_pool if name not in cache]

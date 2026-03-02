# Used automatically by pytest to reset state before each test:
from collections.abc import Generator
from pathlib import Path

import pytest
from fixtures import reset_samplemaker  # noqa: F401

import samplemaker.makers as sm
from samplemaker import LayoutPool
from samplemaker.baselib.devices import CrossMark
from samplemaker.devices import Device
from samplemaker.gdswriter import GDSWriter
from samplemaker.shapes import GeomGroup


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
        rX=2.0,
        rY=1.0,
        rot=15.0,
        layer=6,
        to_poly=False,
    )
    ring = sm.make_ring(
        50.0,
        10.0,
        rX=2.5,
        rY=1.5,
        rot=0.0,
        w=0.4,
        layer=7,
        to_poly=False,
    )
    arc = sm.make_arc(
        60.0,
        10.0,
        rX=2.5,
        rY=1.5,
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
    assert len(layout_pool) == 3
    assert "CELL00" in layout_pool
    assert "BASELIB_CMARK_0001" in layout_pool
    assert "SUBCELL" in layout_pool
    assert all(isinstance(group, GeomGroup) for group in layout_pool.values())
    srefs = layout_pool["CELL00"].get_sref_list()
    assert srefs == {"BASELIB_CMARK_0001", "SUBCELL"}


def test_init_populates_circle_lookup_tables(gds_writer: GDSWriter) -> None:
    assert gds_writer.circleres == 12
    assert gds_writer.arcres == 32
    assert gds_writer.xc.shape == (12,)
    assert gds_writer.yc.shape == (12,)
    assert gds_writer.xc[0] == pytest.approx(1.0)
    assert gds_writer.yc[0] == pytest.approx(0.0)


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
    gds_writer: GDSWriter, layout_pool, tmp_path: Path
) -> None:
    output_file = tmp_path / "mixed.gds"

    gds_writer.open_library(str(output_file))
    for cellname, group in layout_pool.items():
        gds_writer.write_structure(cellname, group)
    gds_writer.close_library()

    data = output_file.read_bytes()
    for cellname in layout_pool:
        assert cellname.encode() in data


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

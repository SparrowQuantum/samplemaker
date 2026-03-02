# Used automatically by pytest to reset state before each test:
from collections.abc import Generator
from pathlib import Path

import pytest
from fixtures import reset_samplemaker  # noqa: F401

import samplemaker.makers as sm
import samplemaker.shapes as sp
from samplemaker import LayoutPool
from samplemaker.baselib.devices import CrossMark
from samplemaker.devices import Device
from samplemaker.gdsreader import GDSReader, GDSRecord
from samplemaker.gdswriter import GDSWriter


@pytest.fixture
def gds_reader() -> GDSReader:
    return GDSReader()


@pytest.fixture
def cell_name() -> str:
    return "CELL00"


@pytest.fixture
def sample_device() -> Device:
    return CrossMark.build()


@pytest.fixture
def sample_geometry(sample_device: Device) -> sp.GeomGroup:
    dev_geom: sp.GeomGroup = sample_device.run()
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
    sample_geometry: sp.GeomGroup,
) -> Generator[dict[str, sp.GeomGroup], None, None]:
    original_layout_pool = LayoutPool.copy()
    LayoutPool[cell_name] = sample_geometry
    yield LayoutPool

    LayoutPool.clear()
    LayoutPool.update(original_layout_pool)


@pytest.fixture
def sample_gds_file(tmp_path: Path, layout_pool: dict[str, sp.GeomGroup]) -> Path:
    output_file = tmp_path / "reader_sample.gds"

    writer = GDSWriter()
    writer.open_library(str(output_file))
    writer.write_structure("CELL00", layout_pool["CELL00"])
    writer.write_structure("SUBCELL", layout_pool["SUBCELL"])
    writer.write_structure("BASELIB_CMARK_0001", LayoutPool["BASELIB_CMARK_0001"])
    writer.close_library()

    return output_file


def test_gdsrecord_attributes() -> None:
    record = GDSRecord(
        size=10, rectype=1, datatype=2, bheader=b"\x00\x0a\x01\x02", data=b"payload"
    )
    assert record.size == 10
    assert record.rectype == 1
    assert record.datatype == 2
    assert record.bheader == b"\x00\x0a\x01\x02"
    assert record.data == b"payload"


def test_gdsrecord_attributes_default_data() -> None:
    record = GDSRecord(size=10, rectype=1, datatype=2, bheader=b"\x00\x0a\x01\x02")
    assert record.size == 10
    assert record.rectype == 1
    assert record.datatype == 2
    assert record.bheader == b"\x00\x0a\x01\x02"
    assert record.data == ""


def test_gdsrecord_to_binary_size_4_returns_header_only() -> None:
    header = b"\x00\x04\x00\x00"
    record = GDSRecord(size=4, rectype=0, datatype=0, bheader=header)

    assert record.to_binary() == header


def test_gdsrecord_to_binary_non_empty_record_returns_header_plus_data() -> None:
    header = b"\x00\x06\x01\x02"
    payload = b"\xaa\xbb"
    record = GDSRecord(size=6, rectype=1, datatype=2, bheader=header, data=payload)

    assert record.to_binary() == header + payload


def test_gdsreader_init_sets_default_state(gds_reader: GDSReader) -> None:
    assert gds_reader.buf == ""
    assert gds_reader.ptr == 0
    assert gds_reader.celldata == {}


def test_quick_read_extracts_cell_binary_data(
    gds_reader: GDSReader, sample_gds_file: Path
) -> None:
    gds_reader.quick_read(str(sample_gds_file))

    assert set(gds_reader.celldata.keys()) == {
        "BASELIB_CMARK_0001",
        "SUBCELL",
        "CELL00",
    }
    assert all(
        isinstance(cell_data, bytes) for cell_data in gds_reader.celldata.values()
    )
    assert all(len(cell_data) > 0 for cell_data in gds_reader.celldata.values())


def test_get_cell_raises_for_unknown_cellname(gds_reader: GDSReader) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        gds_reader.get_cell("UNKNOWN")


def test_get_cell_decodes_mixed_structure(
    gds_reader: GDSReader, sample_gds_file: Path
) -> None:
    gds_reader.quick_read(str(sample_gds_file))

    geom_group = gds_reader.get_cell("CELL00")

    assert isinstance(geom_group, sp.GeomGroup)
    assert len(geom_group.group) == 11

    by_type = {}
    for geom in geom_group.group:
        by_type[type(geom)] = by_type.get(type(geom), 0) + 1

    assert by_type.get(sp.Poly, 0) == 6
    assert by_type.get(sp.Path, 0) == 1
    assert by_type.get(sp.Text, 0) == 1
    assert by_type.get(sp.SRef, 0) == 2
    assert by_type.get(sp.ARef, 0) == 1

    decoded_text = next(geom for geom in geom_group.group if isinstance(geom, sp.Text))
    assert decoded_text.text == "GDS"

    decoded_sref = next(geom for geom in geom_group.group if isinstance(geom, sp.SRef))
    assert decoded_sref.cellname == "SUBCELL"

    decoded_aref = next(geom for geom in geom_group.group if isinstance(geom, sp.ARef))
    assert decoded_aref.cellname == "SUBCELL"
    assert decoded_aref.ncols == 2
    assert decoded_aref.nrows == 2


def test_get_cell_decodes_subcell_boundary(
    gds_reader: GDSReader, sample_gds_file: Path
) -> None:
    gds_reader.quick_read(str(sample_gds_file))

    geom_group = gds_reader.get_cell("SUBCELL")

    assert isinstance(geom_group, sp.GeomGroup)
    assert len(geom_group.group) == 1
    assert isinstance(geom_group.group[0], sp.Poly)

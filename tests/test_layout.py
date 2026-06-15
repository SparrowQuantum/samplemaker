"""Unit tests for the samplemaker.layout module."""

import os
from pathlib import Path
from typing import Generator

import pytest

import samplemaker.devices as smdev
import samplemaker.layout as smlay
from samplemaker import (
    LayoutPool,
    _BoundingBoxPool,
    _DeviceCountPool,
    _DeviceLocalParamPool,
    _DevicePool,
)
from samplemaker.baselib.devices import CrossMark
from samplemaker.makers import make_rect, make_sref
from samplemaker.shapes import ARef, Box, GeomGroup, Poly, SRef, Text
from tests import dummy as dm
from tests.fakes import FakeGDSReader, FakeGDSWriter


@pytest.fixture
def simple_rect_geometry() -> GeomGroup:
    """Small helper geometry used by several layout tests."""
    return make_rect(0, 0, 10, 6, numkey=5)


@pytest.fixture
def tmp_cwd_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Temporarily change the current working directory to a temporary path."""
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        yield tmp_path
    finally:
        os.chdir(original_cwd)


class TestMarker:
    def test_marker_inits_correct_attributes(self) -> None:
        name = "TestMarker"
        dev = CrossMark.build()
        x0 = 10
        y0 = 20
        marker = smlay.Marker(name, dev, x0, y0)
        assert marker.name == name
        assert marker.dev == dev
        assert marker.x0 == x0
        assert marker.y0 == y0

    def test_marker_inits_default_attributes(self) -> None:
        name = "TestMarker"
        dev = CrossMark.build()
        marker = smlay.Marker(name, dev)
        assert marker.name == name
        assert marker.dev == dev
        assert marker.x0 == 0
        assert marker.y0 == 0

    def test_marker_get_geom(self) -> None:
        name = "TestMarker"
        dev = CrossMark.build()
        dev.use_references = False
        x0 = 10
        y0 = 20
        marker = smlay.Marker(name, dev, x0=x0, y0=y0)
        g = marker.get_geom()
        assert dev.use_references is True
        assert isinstance(g, GeomGroup)
        assert len(g.group) == 1
        assert isinstance(g.group[0], SRef)

        bb = g.bounding_box()
        assert bb.cx() == pytest.approx(x0)
        assert bb.cy() == pytest.approx(y0)


class TestMarkerSet:
    def test_markerset_inits_correct_attributes(self) -> None:
        name = "TestMarkerSet"
        dev = CrossMark.build()
        x0 = 10
        y0 = 20
        mset = 2
        xdist = 200
        ydist = 300
        markerset = smlay.MarkerSet(name, dev, x0, y0, mset, xdist, ydist)
        assert issubclass(smlay.MarkerSet, smlay.Marker)
        assert markerset.name == name
        assert markerset.dev == dev
        assert markerset.x0 == x0
        assert markerset.y0 == y0
        assert markerset.mset == mset
        assert markerset.xdist == xdist
        assert markerset.ydist == ydist

    def test_markerset_inits_default_attributes(self) -> None:
        name = "TestMarkerSet"
        dev = CrossMark.build()

        markerset = smlay.MarkerSet(name, dev)
        assert markerset.name == name
        assert markerset.dev == dev
        assert markerset.y0 == 0
        assert markerset.x0 == 0
        assert markerset.mset == 4
        assert markerset.xdist == 1000
        assert markerset.ydist == 1000

    @pytest.mark.xfail(reason="Invalid mset silently ignored", strict=True)
    def test_markerset_init_raises_on_invalid_mset(self) -> None:
        name = "TestMarkerSet"
        dev = CrossMark.build()
        with pytest.raises(ValueError):
            smlay.MarkerSet(name, dev, mset=0)

    @pytest.mark.xfail(
        reason="Geometry is not translated correctly for mset==1", strict=True
    )
    def test_markerset_get_geom_mset1(self) -> None:
        name = "TestMarkerSet"
        dev = CrossMark.build()
        dev.use_references = False
        x0 = 10
        y0 = 20
        mset = 1
        xdist = 200
        ydist = 300
        markerset = smlay.MarkerSet(name, dev, x0, y0, mset, xdist, ydist)
        g = markerset.get_geom()
        assert dev.use_references is True
        assert isinstance(g, GeomGroup)
        assert len(g.group) == 1
        assert isinstance(g.group[0], SRef)
        assert dev._name in g.group[0].cellname

        bb = g.bounding_box()
        assert bb.cx() == pytest.approx(x0)
        assert bb.cy() == pytest.approx(y0)

    def test_markerset_get_geom_mset2(self) -> None:
        name = "TestMarkerSet"
        dev = CrossMark.build()
        dev.use_references = False
        x0 = 10
        y0 = 20
        mset = 2
        xdist = 200
        ydist = 300
        markerset = smlay.MarkerSet(name, dev, x0, y0, mset, xdist, ydist)
        g = markerset.get_geom()
        assert dev.use_references is True
        assert isinstance(g, GeomGroup)
        assert len(g.group) == 1
        assert isinstance(g.group[0], ARef)
        aref = g.group[0]
        assert aref.x0 == pytest.approx(x0)
        assert aref.y0 == pytest.approx(y0)
        assert dev._name in aref.cellname
        assert aref.ncols == 2
        assert aref.nrows == 1
        assert aref.ax == pytest.approx(xdist)
        assert aref.ay == 0
        assert aref.bx == 0
        assert aref.by == pytest.approx(ydist)

    def test_markerset_get_geom_mset4(self) -> None:
        name = "TestMarkerSet"
        dev = CrossMark.build()
        dev.use_references = False
        x0 = 10
        y0 = 20
        mset = 4
        xdist = 200
        ydist = 300
        markerset = smlay.MarkerSet(name, dev, x0, y0, mset, xdist, ydist)
        g = markerset.get_geom()
        assert dev.use_references is True
        assert isinstance(g, GeomGroup)
        assert len(g.group) == 1
        assert isinstance(g.group[0], ARef)
        aref = g.group[0]
        assert aref.x0 == pytest.approx(x0)
        assert aref.y0 == pytest.approx(y0)
        assert dev._name in aref.cellname
        assert aref.ncols == 2
        assert aref.nrows == 2
        assert aref.ax == pytest.approx(xdist)
        assert aref.ay == 0
        assert aref.bx == 0
        assert aref.by == pytest.approx(ydist)


class TestDeviceTableAnnotations:
    def test_annotations_init_defaults(self) -> None:
        ann = smlay.DeviceTableAnnotations(
            rowfmt="row %I %J",
            colfmt="col %I %J",
            xoff=11,
            yoff=22,
            rowvars=("rv",),
            colvars=("cv",),
        )

        assert ann.rowfmt == "row %I %J"
        assert ann.colfmt == "col %I %J"
        assert ann.xoff == 11
        assert ann.yoff == 22
        assert ann.rowvars == ("rv",)
        assert ann.colvars == ("cv",)
        assert ann.text_width == 1
        assert ann.text_height == 10
        assert ann.left is True
        assert ann.right is True
        assert ann.above is True
        assert ann.below is True
        assert ann.to_poly is True

    def test_annotations_init_sets_all_attributes(self) -> None:
        ann = smlay.DeviceTableAnnotations(
            rowfmt="row %I %J",
            colfmt="col %I %J",
            xoff=11,
            yoff=22,
            rowvars=("rv",),
            colvars=("cv",),
            text_width=2,
            text_height=20,
            left=False,
            right=False,
            above=False,
            below=False,
        )

        assert ann.rowfmt == "row %I %J"
        assert ann.colfmt == "col %I %J"
        assert ann.xoff == 11
        assert ann.yoff == 22
        assert ann.rowvars == ("rv",)
        assert ann.colvars == ("cv",)
        assert ann.text_width == 2
        assert ann.text_height == 20
        assert ann.left is False
        assert ann.right is False
        assert ann.above is False
        assert ann.below is False
        assert ann.to_poly is True

    def test_set_poly_text_toggles_output_type(self) -> None:
        ann = smlay.DeviceTableAnnotations(
            rowfmt="R",
            colfmt="C",
            xoff=2,
            yoff=3,
            rowvars=("rv",),
            colvars=("cv",),
        )
        rowdict = {"rv": [1.0]}
        coldict = {"cv": [2.0]}

        geom_poly = ann.render(0, 0, 1, 1, 0, 0, rowdict, coldict)
        assert isinstance(geom_poly, GeomGroup)
        assert not any(isinstance(elem, Text) for elem in geom_poly.group)
        assert any(isinstance(elem, Poly) for elem in geom_poly.group)

        ann.set_poly_text(False)
        assert ann.to_poly is False
        geom_text = ann.render(0, 0, 1, 1, 0, 0, rowdict, coldict)
        assert isinstance(geom_text, GeomGroup)
        assert all(isinstance(elem, Text) for elem in geom_text.group)

    def test_render_correctly_formats_and_places_text(self) -> None:
        ann = smlay.DeviceTableAnnotations(
            rowfmt="ROW i=%I j=%J c=%C0 r=%R0",
            colfmt="COL i=%I j=%J c=%C0 r=%R0",
            xoff=1,
            yoff=2,
            rowvars=("rv",),
            colvars=("cv",),
        )
        ann.set_poly_text(False)

        rowdict = {"rv": [1.23456, 5.0]}
        coldict = {"cv": [9.87654, 8.0]}
        g = ann.render(0, 0, 2, 2, 10, 20, rowdict, coldict)
        assert isinstance(g, GeomGroup)
        assert len(g.group) == 2
        texts = [elem for elem in g.group if isinstance(elem, Text)]
        assert len(texts) == 2

        left = [
            t for t in texts if t.x0 == pytest.approx(9) and t.y0 == pytest.approx(20)
        ]
        below = [
            t for t in texts if t.x0 == pytest.approx(10) and t.y0 == pytest.approx(18)
        ]
        assert len(left) == 1
        assert len(below) == 1

        assert left[0].text == "ROW i=0 j=0 c=9.877 r=1.235"
        assert below[0].text == "COL i=0 j=0 c=9.877 r=1.235"

    def test_render_returns_annotations_on_configured_edges(self) -> None:
        ann = smlay.DeviceTableAnnotations(
            rowfmt="R",
            colfmt="C",
            xoff=1,
            yoff=1,
            rowvars=("rv",),
            colvars=("cv",),
        )
        ann.set_poly_text(False)
        rowdict = {"rv": [1.0, 2.0, 3.0]}
        coldict = {"cv": [10.0, 20.0, 30.0]}

        geom_interior = ann.render(1, 1, 3, 3, 0, 0, rowdict, coldict)
        assert isinstance(geom_interior, GeomGroup)
        assert len(geom_interior.group) == 0

        geom_top_right = ann.render(2, 2, 3, 3, 0, 0, rowdict, coldict)
        assert isinstance(geom_top_right, GeomGroup)
        assert len(geom_top_right.group) == 2

        ann.left = False
        ann.right = False
        ann.above = True
        ann.below = False
        geom_top_right_above_only = ann.render(2, 2, 3, 3, 0, 0, rowdict, coldict)
        assert isinstance(geom_top_right_above_only, GeomGroup)
        assert len(geom_top_right_above_only.group) == 1


class TestDeviceTable:
    def test_mutators_update_state_and_invalidate_cached_arrays(
        self, dummy_device: smdev.Device
    ) -> None:
        rowvars = {"height": [2.0, 4.0]}
        colvars = {}
        tab = smlay.DeviceTable(dummy_device, 2, 2, rowvars, colvars)
        tab._geometries = [[GeomGroup()]]
        tab._portmap = [[{"in": object()}]]

        positions = (((1.0, 2.0), (3.0, 4.0)), ((5.0, 6.0), (7.0, 8.0)))
        tab.set_table_positions(positions)
        assert tab.pos_xy == positions
        assert tab._geometries == []
        assert tab._portmap == []

        tab._geometries = [[GeomGroup()]]
        tab._portmap = [[{"in": object()}]]
        tab.set_device_rotation(90)
        assert tab.device_rotation == 90
        assert tab._geometries == []
        assert tab._portmap == []

        tab.set_linked_ports(row_linkports=(("a", "b"),), col_linkports=(("c", "d"),))
        assert tab.row_linkports == (("a", "b"),)
        assert tab.col_linkports == (("c", "d"),)

        tab.set_aligned_ports(align_rows=True, align_columns=True)
        assert tab.row_alignports is True
        assert tab.col_alignports is True

        ann = smlay.DeviceTableAnnotations("r", "c", 1, 1, (), ())
        tab.set_annotations(ann)
        assert tab.annotations is ann

    def test_shift_table_origin_offsets_all_positions(
        self, dummy_device: smdev.Device
    ) -> None:
        tab = smlay.DeviceTable(dummy_device, 2, 2, {}, {})
        tab.set_table_positions((((0.0, 0.0), (1.0, 1.0)), ((2.0, 2.0), (3.0, 3.0))))
        tab.shift_table_origin(10.0, -5.0)
        assert tab.pos_xy == (
            ((10.0, -5.0), (11.0, -4.0)),
            ((12.0, -3.0), (13.0, -2.0)),
        )

    def test_regular_returns_expected_coordinate_grid(self) -> None:
        reg = smlay.DeviceTable.Regular(2, 3, 5.0, 1.0, -2.0, 7.0, x0=100.0, y0=200.0)
        assert reg == (
            ((100.0, 200.0), (105.0, 201.0), (110.0, 202.0)),
            ((98.0, 207.0), (103.0, 208.0), (108.0, 209.0)),
        )

    def test_get_geometries_builds_table_with_param_sweep(
        self, dummy_device: smdev.Device
    ) -> None:
        rowvars = {"height": [2.0, 4.0]}
        colvars = {"width": [3.0]}
        tab = smlay.DeviceTable(dummy_device, 2, 2, rowvars, colvars)
        tab.use_references = False
        tab.set_table_positions(
            (((0.0, 0.0), (20.0, 0.0)), ((0.0, 20.0), (20.0, 20.0)))
        )

        g = tab.get_geometries()
        assert isinstance(g, GeomGroup)
        assert len(g.group) == 4
        assert tab._getgeom_ran is True
        assert all(isinstance(elem, Poly) for elem in g.group)

        ports = tab.get_external_ports()
        assert set(ports.keys()) == {"in_0_0", "in_0_1", "in_1_0", "in_1_1"}

    def test_get_geometries_sets_device_parameters(
        self, dummy_device: smdev.Device, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_params: dict[str, list[float]] = {}

        def _capture_set_param(self: smdev.Device, param: str, value: float) -> None:
            if param not in captured_params:
                captured_params[param] = []
            captured_params[param].append(value)

        rowvars = {"height": [2.0, 4.0]}
        colvars = {"width": [3.0, 5.0]}
        tab = smlay.DeviceTable(dummy_device, 2, 2, rowvars, colvars)
        monkeypatch.setattr(smdev.Device, "set_param", _capture_set_param)
        tab.set_table_positions((((0.0, 0.0), (1.0, 1.0)), ((2.0, 2.0), (3.0, 3.0))))
        tab.get_geometries()
        assert captured_params == {
            "height": [2.0, 4.0, 2.0, 4.0],  # Set in inner loop: ncol calls per value
            "width": [3.0, 5.0],  # Set in outer loop: 1 call per value
        }

    def test_get_geometries_repeated_call_resets_and_returns_stable_output(
        self, dummy_device: smdev.Device
    ) -> None:
        tab = smlay.DeviceTable(dummy_device, 1, 2, {}, {"width": [5.0, 7.0]})
        tab.set_table_positions((((0.0, 0.0), (12.0, 0.0)),))

        g1 = tab.get_geometries()
        g2 = tab.get_geometries()

        assert isinstance(g1, GeomGroup)
        assert isinstance(g2, GeomGroup)
        assert len(g1.group) == len(g2.group) == 2

    def test_get_external_ports_returns_copy_with_indexed_names(
        self, dummy_device: smdev.Device
    ) -> None:
        tab = smlay.DeviceTable(dummy_device, 2, 2, {}, {})
        tab.get_geometries()

        ports1 = tab.get_external_ports()
        ports2 = tab.get_external_ports()

        assert set(ports1.keys()) == {"in_0_0", "in_0_1", "in_1_0", "in_1_1"}
        assert ports1["in_0_0"] is not ports2["in_0_0"]

    def test_get_geometries_calls_connectors_for_row_and_column_links(
        self,
        dummy_connector_device: smdev.Device,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: list[tuple[float, float, float, float]] = []

        def _capture_connector(
            port1: smdev.DevicePort, port2: smdev.DevicePort
        ) -> GeomGroup:
            captured.append((port1.x0, port1.y0, port2.x0, port2.y0))
            return GeomGroup()

        monkeypatch.setattr(dm, "_dummy_connector", _capture_connector)
        tab = smlay.DeviceTable(dummy_connector_device, 2, 2, {}, {})
        tab.set_table_positions(
            (((0.0, 0.0), (20.0, 0.0)), ((0.0, 20.0), (20.0, 20.0)))
        )
        tab.set_linked_ports(
            row_linkports=(("io", "io"),), col_linkports=(("io", "io"),)
        )

        tab.get_geometries()
        assert len(captured) == 4

    def test_get_geometries_raises_on_incompatible_connector_functions(
        self,
        dummy_device: smdev.Device,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tab = smlay.DeviceTable(dummy_device, 1, 2, {}, {})

        original_build = getattr(tab, "_DeviceTable__build_geomarray")

        def _build_with_incompatible_ports() -> None:
            original_build()
            tab._portmap[0][0]["in"].connector_function = lambda p1, p2: GeomGroup()
            tab._portmap[0][1]["in"].connector_function = lambda p1, p2: GeomGroup()

        monkeypatch.setattr(
            tab, "_DeviceTable__build_geomarray", _build_with_incompatible_ports
        )
        tab.set_linked_ports(col_linkports=(("in", "in"),))

        with pytest.raises(smdev.IncompatiblePortError, match="Incompatible ports"):
            tab.get_geometries()

    @pytest.mark.xfail(reason="Row alignment uses p2.yx typo", strict=True)
    def test_get_geometries_row_alignment_bug_path(
        self,
        dummy_connector_device: smdev.Device,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tab = smlay.DeviceTable(dummy_connector_device, 2, 1, {}, {})
        tab.set_table_positions((((0.0, 0.0),), ((10.0, 20.0),)))
        tab.set_linked_ports(row_linkports=(("io", "io"),))
        tab.set_aligned_ports(align_rows=True, align_columns=False)

        original_build = getattr(tab, "_DeviceTable__build_geomarray")

        def _build_with_vertical_ports() -> None:
            original_build()
            for row in tab._portmap:
                for pmap in row:
                    pmap["io"].hv = False
                    pmap["io"].bf = True

        monkeypatch.setattr(
            tab, "_DeviceTable__build_geomarray", _build_with_vertical_ports
        )
        tab.get_geometries()

    def test_get_geometries_renders_annotations(
        self, dummy_device: smdev.Device, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        indices = []
        coords = []

        def _capture_render(
            self: smlay.DeviceTableAnnotations,
            i: int,
            j: int,
            rows: int,
            cols: int,
            x0: int,
            y0: int,
            rowdict: dict[str, list[float]],
            coldict: dict[str, list[float]],
        ) -> GeomGroup:
            indices.append((i, j))
            coords.append((x0, y0))
            return GeomGroup()

        monkeypatch.setattr(smlay.DeviceTableAnnotations, "render", _capture_render)
        ann = smlay.DeviceTableAnnotations("R", "C", 1, 1, ("width",), ("height",))
        rowvars = {"width": [1.0, 2.0]}
        colvars = {"height": [10.0, 20.0]}
        tab = smlay.DeviceTable(dummy_device, 2, 2, rowvars, colvars)
        tab.set_annotations(ann)
        tab.get_geometries()
        assert indices == [(0, 0), (1, 0), (0, 1), (1, 1)]
        assert coords == [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]

    def test_auto_align_produces_non_overlapping_positions(
        self, dummy_device: smdev.Device
    ) -> None:
        rowvars = {"height": [2.0, 6.0]}
        colvars = {"width": [3.0, 9.0]}
        tab = smlay.DeviceTable(dummy_device, 2, 2, rowvars=rowvars, colvars=colvars)
        tab.auto_align(min_dist_x=5.0, min_dist_y=7.0)

        assert tab.pos_xy[0][1][0] > tab.pos_xy[0][0][0]
        assert tab.pos_xy[1][0][1] > tab.pos_xy[0][0][1]


class TestMask:
    def test_mask_init_and_clear_reset_pools_and_basic_elements(self) -> None:
        LayoutPool["JUNK"] = GeomGroup()
        _DevicePool["hash"] = "JUNK"
        _DeviceLocalParamPool["hash"] = {}
        _DeviceCountPool["JUNK"] = 1
        _BoundingBoxPool["JUNK"] = object()

        mask = smlay.Mask("test")
        assert mask.name == "test"
        assert mask.mainsymbol == "CELL00"
        assert mask.cache is False
        assert mask.writefields == []
        assert "_CIRCLE" in LayoutPool

        mask.writefields.append((10, 0, 0, 1, 0))
        LayoutPool["ANOTHER"] = GeomGroup()
        mask.clear()

        assert "_CIRCLE" in LayoutPool
        assert "ANOTHER" not in LayoutPool
        assert mask.writefields == []
        assert set(_DevicePool.keys()) == set()
        assert set(_DeviceLocalParamPool.keys()) == set()
        assert set(_DeviceCountPool.keys()) == set()
        assert set(_BoundingBoxPool.keys()) == {"_CIRCLE"}

    def test_add_to_main_cell_add_cell_and_get_cell(
        self, simple_rect_geometry: GeomGroup
    ) -> None:
        themask = smlay.Mask("test_cells")
        g1 = simple_rect_geometry.copy()
        g2 = simple_rect_geometry.copy()

        themask.addToMainCell(g1)
        themask.addToMainCell(g2)
        assert themask.mainsymbol in LayoutPool
        assert len(LayoutPool[themask.mainsymbol].group) == 2

        themask.addCell("EXTRA", simple_rect_geometry.copy())
        cell = themask.getCell("EXTRA")
        assert isinstance(cell, GeomGroup)
        assert cell is LayoutPool["EXTRA"]

        with pytest.raises(ValueError, match="does not exist"):
            themask.getCell("MISSING")

    def test_add_markers_add_writefield_and_writefield_grid(self) -> None:
        mask1 = smlay.Mask("test_wf")
        markerset1 = smlay.MarkerSet("M1", CrossMark.build(), x0=2, y0=3, mset=2)
        mask1.addMarkers(markerset1)
        assert mask1.mainsymbol in LayoutPool
        assert len(LayoutPool[mask1.mainsymbol].group) == 1
        assert isinstance(LayoutPool[mask1.mainsymbol].group[0], ARef)

        markerset2 = smlay.MarkerSet("M2", CrossMark.build(), x0=5, y0=6, mset=4)
        mask1.addMarkers(markerset2)
        assert mask1.mainsymbol in LayoutPool
        assert len(LayoutPool[mask1.mainsymbol].group) == 2
        assert isinstance(LayoutPool[mask1.mainsymbol].group[1], ARef)

        mask1.addWriteField(500, 10, 20, passes=2, shift=0.5)
        assert mask1.writefields[-1] == (500, 10, 20, 2, 0.5)

        mask2 = smlay.Mask("test_wf_grid")
        mask2.addWriteFieldGrid(50, 0, 0, 2, 3, passes=3, shift=1.5)
        assert len(mask2.writefields) == 6
        assert mask2.mainsymbol in LayoutPool
        assert len(LayoutPool[mask2.mainsymbol].group) == 6

    def test_add_device_table_centers_geometry_in_main_and_named_cell(
        self, dummy_device: smdev.Device
    ) -> None:
        tab1 = smlay.DeviceTable(dummy_device, 1, 1, {}, {})
        themask = smlay.Mask("test_table")

        themask.addDeviceTable(tab1, x0=100, y0=200)
        bb_main = LayoutPool[themask.mainsymbol].bounding_box()
        assert bb_main.cx() == pytest.approx(100)
        assert bb_main.cy() == pytest.approx(200)

        tab2 = smlay.DeviceTable(dummy_device, 1, 1, {}, {})
        themask.addDeviceTable(tab2, x0=10, y0=20, cell="TABLE_CELL")
        assert "TABLE_CELL" in LayoutPool
        bb_cell = LayoutPool["TABLE_CELL"].bounding_box()
        assert bb_cell.cx() == pytest.approx(10)
        assert bb_cell.cy() == pytest.approx(20)

    def test_set_cache_triggers_import_only_when_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        themask = smlay.Mask("test_cache")
        calls: list[bool] = []

        def _capture_import() -> None:
            calls.append(True)

        monkeypatch.setattr(themask, "_Mask__import_cache", _capture_import)

        themask.set_cache(False)
        assert themask.cache is False
        assert calls == []

        themask.set_cache(True)
        assert themask.cache is True
        assert calls == [True]

    def test_export_gds_non_cache_cleans_unreferenced_and_writes_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created_writers: list[FakeGDSWriter] = []

        def _writer_factory() -> FakeGDSWriter:
            w = FakeGDSWriter()
            created_writers.append(w)
            return w

        monkeypatch.setattr(smlay, "GDSWriter", _writer_factory)

        themask = smlay.Mask("test_export")
        child = make_rect(0, 0, 5, 5)
        unref = make_rect(0, 0, 1, 1)
        themask.addCell("CHILD", child)
        themask.addCell("UNREF", unref)
        main = make_sref(0, 0, "CHILD", LayoutPool["CHILD"])
        themask.addToMainCell(main)

        _DevicePool["h_keep"] = "CHILD"
        _DevicePool["h_drop"] = "UNREF"
        _DeviceLocalParamPool["h_keep"] = {}
        _DeviceLocalParamPool["h_drop"] = {}

        themask.exportGDS()

        assert "UNREF" not in LayoutPool
        assert "CHILD" in LayoutPool
        assert "h_drop" not in _DevicePool
        assert "h_drop" not in _DeviceLocalParamPool
        assert "h_keep" in _DevicePool

        writer = created_writers[0]
        assert writer.calls[0] == ("open_library", "test_export.gds")
        assert writer.calls[1][0] == "write_pool"
        assert writer.calls[2] == ("close_library", None)

    def test_export_gds_cache_success_uses_reader_cache_and_exports_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created_writers: list[FakeGDSWriter] = []
        fake_reader = FakeGDSReader(celldata={"CELL00": b"x", "OLD": b"y"})
        export_cache_calls: list[bool] = []

        def _writer_factory() -> FakeGDSWriter:
            w = FakeGDSWriter()
            created_writers.append(w)
            return w

        monkeypatch.setattr(smlay, "GDSWriter", _writer_factory)
        monkeypatch.setattr(smlay, "GDSReader", lambda: fake_reader)

        themask = smlay.Mask("test_cache_export")
        themask.addToMainCell(make_rect(0, 0, 2, 2))
        themask.cache = True

        def _capture_export_cache() -> None:
            export_cache_calls.append(True)

        monkeypatch.setattr(themask, "_Mask__export_cache", _capture_export_cache)

        themask.exportGDS()

        assert fake_reader.quick_read_calls == ["test_cache_export.gds"]
        writer = created_writers[0]
        assert writer.calls[1][0] == "write_pool_use_cache"
        cached_cells = writer.calls[1][2]
        assert isinstance(cached_cells, list)
        assert "CELL00" not in cached_cells
        assert export_cache_calls == [True]

    def test_import_gds_single_top_candidate_sets_mainsymbol_and_sref_groups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_reader = FakeGDSReader(celldata={"TOP": b"", "SUB": b""})
        sub = make_rect(0, 0, 5, 5)
        top = make_sref(0, 0, "SUB", GeomGroup())
        fake_reader.cell_geometries["TOP"] = top
        fake_reader.cell_geometries["SUB"] = sub
        monkeypatch.setattr(smlay, "GDSReader", lambda: fake_reader)

        themask = smlay.Mask("test_import_single")
        themask.importGDS("fake.gds")

        assert themask.mainsymbol == "TOP"
        sref = LayoutPool["TOP"].group[0]
        assert isinstance(sref, SRef)
        assert sref.group is LayoutPool["SUB"]

    def test_import_gds_multiple_candidates_selects_one_with_most_subrefs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_reader = FakeGDSReader(
            celldata={"A": b"", "B": b"", "C": b"", "D": b"", "E": b""}
        )
        fake_reader.cell_geometries["B"] = make_rect(0, 0, 1, 1)
        fake_reader.cell_geometries["C"] = make_rect(0, 0, 1, 1)
        fake_reader.cell_geometries["E"] = make_rect(0, 0, 1, 1)
        fake_reader.cell_geometries["A"] = make_sref(
            0, 0, "B", GeomGroup()
        ) + make_sref(0, 0, "C", GeomGroup())
        fake_reader.cell_geometries["D"] = make_sref(0, 0, "E", GeomGroup())

        monkeypatch.setattr(smlay, "GDSReader", lambda: fake_reader)

        themask = smlay.Mask("test_import_multi")
        themask.importGDS("fake.gds")

        assert themask.mainsymbol == "A"

    def test_export_import_cache_restores_state(self, tmp_cwd_dir: Path) -> None:
        _ = tmp_cwd_dir
        name = "test_cache_restore"
        themask = smlay.Mask(name)
        themask.addToMainCell(make_rect(0, 0, 5, 5))
        themask.addCell("EXTRA", make_rect(0, 0, 1, 1))
        _DevicePool["h_extra"] = "EXTRA"
        _DeviceLocalParamPool["h_extra"] = {"param": 42}
        _DeviceCountPool["EXTRA"] = 1
        _BoundingBoxPool["EXTRA"] = Box(0, 0, 1, 1)

        themask._Mask__export_cache()
        expected_filepath = tmp_cwd_dir / f"{name}.cache"
        assert os.path.isfile(expected_filepath)

        themask.clear()
        assert "EXTRA" not in LayoutPool
        assert "h_extra" not in _DevicePool
        assert "h_extra" not in _DeviceLocalParamPool
        assert "EXTRA" not in _DeviceCountPool
        assert "EXTRA" not in _BoundingBoxPool

        themask._Mask__import_cache()
        assert "EXTRA" in LayoutPool
        assert "_CIRCLE" in LayoutPool
        assert _DevicePool["h_extra"] == "EXTRA"
        assert _DeviceLocalParamPool["h_extra"] == {"param": 42}
        assert _DeviceCountPool["EXTRA"] == 1
        assert isinstance(_BoundingBoxPool["EXTRA"], Box)
        bbox = _BoundingBoxPool["EXTRA"]
        assert bbox.llx == 0
        assert bbox.lly == 0
        assert bbox.width == 1
        assert bbox.height == 1

        os.remove(expected_filepath)

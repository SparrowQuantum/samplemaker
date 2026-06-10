"""Unit tests for the samplemaker.layout module."""

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
from samplemaker.shapes import ARef, GeomGroup, Poly, SRef, Text
from tests import dummy as dm
from tests.fakes import FakeGDSReader, FakeGDSWriter


@pytest.fixture
def simple_rect_geometry() -> GeomGroup:
    """Small helper geometry used by several layout tests."""
    return make_rect(0, 0, 10, 6, numkey=5)


@pytest.fixture
def dummy_connector_device(
    dummy_device_list: dict[str, type[smdev.Device]],
) -> smdev.Device:
    """Build a connector-compatible device from the shared dummy registry."""
    _ = dummy_device_list
    return smdev.Device.build_registered("TESTLIB_DUMMY_CONNECTOR")


@pytest.fixture
def layout_pool_snapshot() -> dict[str, GeomGroup]:
    """Return a shallow copy of the current LayoutPool for comparison assertions."""
    return LayoutPool.copy()


def assert_pool_keys(expected_keys: set[str]) -> None:
    assert set(LayoutPool.keys()) == expected_keys


def assert_cache_pools(
    *,
    device_pool_keys: set[str],
    local_param_pool_keys: set[str],
    device_count_pool_keys: set[str],
    bbox_pool_keys: set[str],
) -> None:
    assert set(_DevicePool.keys()) == device_pool_keys
    assert set(_DeviceLocalParamPool.keys()) == local_param_pool_keys
    assert set(_DeviceCountPool.keys()) == device_count_pool_keys
    assert set(_BoundingBoxPool.keys()) == bbox_pool_keys


def assert_pool_references(cellname: str, expected: set[str]) -> None:
    assert cellname in LayoutPool
    assert LayoutPool[cellname].get_sref_list() == expected


class TestDeviceTableAnnotations:
    def test_annotations_init_defaults(self):
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

    def test_set_poly_text_toggles_output_type(self):
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
        assert not any(isinstance(elem, Text) for elem in geom_poly.group)
        assert any(isinstance(elem, Poly) for elem in geom_poly.group)

        ann.set_poly_text(False)
        geom_text = ann.render(0, 0, 1, 1, 0, 0, rowdict, coldict)
        assert all(isinstance(elem, Text) for elem in geom_text.group)

    def test_render_replaces_tokens_rounds_values_and_places_expected_edges(self):
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
        geom = ann.render(0, 0, 2, 2, 10, 20, rowdict, coldict)

        assert len(geom.group) == 2
        texts = [elem for elem in geom.group if isinstance(elem, Text)]
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

    def test_render_only_emits_annotations_on_configured_edges(self):
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

        interior = ann.render(1, 1, 3, 3, 0, 0, rowdict, coldict)
        assert len(interior.group) == 0

        top_right = ann.render(2, 2, 3, 3, 0, 0, rowdict, coldict)
        assert len(top_right.group) == 2

        ann.left = False
        ann.right = False
        ann.above = True
        ann.below = False
        top_right_above_only = ann.render(2, 2, 3, 3, 0, 0, rowdict, coldict)
        assert len(top_right_above_only.group) == 1


class TestDeviceTable:
    @pytest.fixture
    def table_device(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
    ) -> smdev.Device:
        _ = dummy_device_list
        return smdev.Device.build_registered("TESTLIB_DUMMY")

    @pytest.fixture
    def connector_table_device(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
    ) -> smdev.Device:
        _ = dummy_device_list
        return smdev.Device.build_registered("TESTLIB_DUMMY_CONNECTOR")

    def test_mutators_update_state_and_invalidate_cached_arrays(self, table_device):
        dt = smlay.DeviceTable(table_device, 2, 2, {"height": [2.0, 4.0]}, {})
        dt._geometries = [[GeomGroup()]]
        dt._portmap = [[{"in": object()}]]

        positions = (((1.0, 2.0), (3.0, 4.0)), ((5.0, 6.0), (7.0, 8.0)))
        dt.set_table_positions(positions)
        assert dt.pos_xy == positions
        assert dt._geometries == []
        assert dt._portmap == []

        dt._geometries = [[GeomGroup()]]
        dt._portmap = [[{"in": object()}]]
        dt.set_device_rotation(90)
        assert dt.device_rotation == 90
        assert dt._geometries == []
        assert dt._portmap == []

        dt.set_linked_ports(row_linkports=(("a", "b"),), col_linkports=(("c", "d"),))
        assert dt.row_linkports == (("a", "b"),)
        assert dt.col_linkports == (("c", "d"),)

        dt.set_aligned_ports(align_rows=True, align_columns=True)
        assert dt.row_alignports is True
        assert dt.col_alignports is True

        ann = smlay.DeviceTableAnnotations("r", "c", 1, 1, (), ())
        dt.set_annotations(ann)
        assert dt.annotations is ann

    def test_shift_table_origin_offsets_all_positions(self, table_device):
        dt = smlay.DeviceTable(table_device, 2, 2, {}, {})
        dt.set_table_positions((((0.0, 0.0), (1.0, 1.0)), ((2.0, 2.0), (3.0, 3.0))))
        dt.shift_table_origin(10.0, -5.0)
        assert dt.pos_xy == (
            ((10.0, -5.0), (11.0, -4.0)),
            ((12.0, -3.0), (13.0, -2.0)),
        )

    def test_regular_returns_expected_coordinate_grid(self):
        reg = smlay.DeviceTable.Regular(2, 3, 5.0, 1.0, -2.0, 7.0, x0=100.0, y0=200.0)
        assert reg == (
            ((100.0, 200.0), (105.0, 201.0), (110.0, 202.0)),
            ((98.0, 207.0), (103.0, 208.0), (108.0, 209.0)),
        )

    def test_get_geometries_builds_table_with_param_sweep_and_fallback_lists(
        self, table_device
    ):
        dt = smlay.DeviceTable(
            table_device,
            2,
            2,
            rowvars={"height": [2.0, 4.0]},
            colvars={"width": [3.0]},
        )
        dt.use_references = False
        dt.set_table_positions((((0.0, 0.0), (20.0, 0.0)), ((0.0, 20.0), (20.0, 20.0))))

        g = dt.get_geometries()
        assert isinstance(g, GeomGroup)
        assert len(g.group) == 4
        assert dt._getgeom_ran is True
        assert all(isinstance(elem, Poly) for elem in g.group)

        ports = dt.get_external_ports()
        assert set(ports.keys()) == {"in_0_0", "in_0_1", "in_1_0", "in_1_1"}

    def test_get_geometries_repeated_call_resets_and_returns_stable_output(
        self, table_device
    ):
        dt = smlay.DeviceTable(table_device, 1, 2, {}, {"width": [5.0, 7.0]})
        dt.use_references = False
        dt.set_table_positions((((0.0, 0.0), (12.0, 0.0)),))

        g1 = dt.get_geometries()
        g2 = dt.get_geometries()

        assert isinstance(g1, GeomGroup)
        assert isinstance(g2, GeomGroup)
        assert len(g1.group) == len(g2.group) == 2

    def test_get_external_ports_returns_copy_with_indexed_names(self, table_device):
        dt = smlay.DeviceTable(table_device, 2, 2, {}, {})
        dt.use_references = False
        _ = dt.get_geometries()

        ports1 = dt.get_external_ports()
        ports2 = dt.get_external_ports()

        assert set(ports1.keys()) == {"in_0_0", "in_0_1", "in_1_0", "in_1_1"}
        assert ports1["in_0_0"] is not ports2["in_0_0"]

    def test_get_geometries_calls_connectors_for_row_and_column_links(
        self,
        connector_table_device,
        monkeypatch: pytest.MonkeyPatch,
    ):
        captured: list[tuple[float, float, float, float]] = []

        def _capture_connector(
            port1: smdev.DevicePort, port2: smdev.DevicePort
        ) -> GeomGroup:
            captured.append((port1.x0, port1.y0, port2.x0, port2.y0))
            return GeomGroup()

        monkeypatch.setattr(dm, "_dummy_connector", _capture_connector)
        dt = smlay.DeviceTable(connector_table_device, 2, 2, {}, {})
        dt.use_references = False
        dt.set_table_positions((((0.0, 0.0), (20.0, 0.0)), ((0.0, 20.0), (20.0, 20.0))))
        dt.set_linked_ports(
            row_linkports=(("io", "io"),), col_linkports=(("io", "io"),)
        )

        _ = dt.get_geometries()
        assert len(captured) == 4

    def test_get_geometries_raises_on_incompatible_connector_functions(
        self,
        table_device,
        monkeypatch: pytest.MonkeyPatch,
    ):
        dt = smlay.DeviceTable(table_device, 1, 2, {}, {})
        dt.use_references = False

        original_build = getattr(dt, "_DeviceTable__build_geomarray")

        def _build_with_incompatible_ports() -> None:
            original_build()
            dt._portmap[0][0]["in"].connector_function = lambda p1, p2: GeomGroup()
            dt._portmap[0][1]["in"].connector_function = lambda p1, p2: GeomGroup()

        monkeypatch.setattr(
            dt, "_DeviceTable__build_geomarray", _build_with_incompatible_ports
        )
        dt.set_linked_ports(col_linkports=(("in", "in"),))

        with pytest.raises(smdev.IncompatiblePortError, match="Incompatible ports"):
            dt.get_geometries()

    @pytest.mark.xfail(reason="Row alignment uses p2.yx typo", strict=True)
    def test_get_geometries_row_alignment_bug_path(
        self,
        connector_table_device,
        monkeypatch: pytest.MonkeyPatch,
    ):
        dt = smlay.DeviceTable(connector_table_device, 2, 1, {}, {})
        dt.use_references = False
        dt.set_table_positions((((0.0, 0.0),), ((10.0, 20.0),)))
        dt.set_linked_ports(row_linkports=(("io", "io"),))
        dt.set_aligned_ports(align_rows=True, align_columns=False)

        original_build = getattr(dt, "_DeviceTable__build_geomarray")

        def _build_with_vertical_ports() -> None:
            original_build()
            for row in dt._portmap:
                for pmap in row:
                    pmap["io"].hv = False
                    pmap["io"].bf = True

        monkeypatch.setattr(
            dt, "_DeviceTable__build_geomarray", _build_with_vertical_ports
        )

        _ = dt.get_geometries()

    def test_auto_align_produces_non_overlapping_positions(self, table_device):
        dt = smlay.DeviceTable(
            table_device,
            2,
            2,
            rowvars={"height": [2.0, 6.0]},
            colvars={"width": [3.0, 9.0]},
        )
        dt.use_references = False
        dt.auto_align(min_dist_x=5.0, min_dist_y=7.0)

        assert dt.pos_xy[0][1][0] > dt.pos_xy[0][0][0]
        assert dt.pos_xy[1][0][1] > dt.pos_xy[0][0][1]


class TestMask:
    @pytest.fixture
    def table_device(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
    ) -> smdev.Device:
        _ = dummy_device_list
        return smdev.Device.build_registered("TESTLIB_DUMMY")

    def test_mask_init_and_clear_reset_pools_and_basic_elements(self):
        LayoutPool["JUNK"] = GeomGroup()
        _DevicePool["hash"] = "JUNK"
        _DeviceLocalParamPool["hash"] = {}
        _DeviceCountPool["JUNK"] = 1
        _BoundingBoxPool["JUNK"] = object()

        mask = smlay.Mask("phase4")
        assert mask.name == "phase4"
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
        assert_cache_pools(
            device_pool_keys=set(),
            local_param_pool_keys=set(),
            device_count_pool_keys=set(),
            bbox_pool_keys={"_CIRCLE"},
        )

    def test_add_to_main_cell_add_cell_and_get_cell(self, simple_rect_geometry):
        mask = smlay.Mask("phase4_cells")
        g1 = simple_rect_geometry.copy()
        g2 = simple_rect_geometry.copy()

        mask.addToMainCell(g1)
        mask.addToMainCell(g2)
        assert mask.mainsymbol in LayoutPool
        assert len(LayoutPool[mask.mainsymbol].group) == 2

        mask.addCell("EXTRA", simple_rect_geometry.copy())
        got = mask.getCell("EXTRA")
        assert isinstance(got, GeomGroup)
        assert got is LayoutPool["EXTRA"]

        with pytest.raises(ValueError, match="does not exist"):
            mask.getCell("MISSING")

    def test_add_markers_add_writefield_and_writefield_grid(self):
        mask = smlay.Mask("phase4_wf")
        marker = smlay.MarkerSet("M1", CrossMark.build(), x0=2, y0=3, mset=2)
        mask.addMarkers(marker)
        assert mask.mainsymbol in LayoutPool
        assert len(LayoutPool[mask.mainsymbol].group) == 1
        assert isinstance(LayoutPool[mask.mainsymbol].group[0], ARef)

        mask.addWriteField(100, 10, 20, passes=2, shift=0.5)
        assert mask.writefields[-1] == (100, 10, 20, 2, 0.5)

        mask2 = smlay.Mask("phase4_wf_grid")
        mask2.addWriteFieldGrid(50, 0, 0, 2, 3, passes=3, shift=1.5)
        assert len(mask2.writefields) == 6
        assert mask2.mainsymbol in LayoutPool
        assert len(LayoutPool[mask2.mainsymbol].group) == 6

    def test_add_device_table_centers_geometry_in_main_and_named_cell(
        self,
        table_device,
    ):
        dt = smlay.DeviceTable(table_device, 1, 1, {}, {})
        dt.use_references = False
        mask = smlay.Mask("phase4_table")

        mask.addDeviceTable(dt, x0=100, y0=200)
        bb_main = LayoutPool[mask.mainsymbol].bounding_box()
        assert bb_main.cx() == pytest.approx(100)
        assert bb_main.cy() == pytest.approx(200)

        dt2 = smlay.DeviceTable(table_device, 1, 1, {}, {})
        dt2.use_references = False
        mask.addDeviceTable(dt2, x0=10, y0=20, cell="TABLE_CELL")
        assert "TABLE_CELL" in LayoutPool
        bb_cell = LayoutPool["TABLE_CELL"].bounding_box()
        assert bb_cell.cx() == pytest.approx(10)
        assert bb_cell.cy() == pytest.approx(20)

    def test_set_cache_triggers_import_only_when_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        mask = smlay.Mask("phase4_cache")
        calls: list[bool] = []

        def _capture_import() -> None:
            calls.append(True)

        monkeypatch.setattr(mask, "_Mask__import_cache", _capture_import)

        mask.set_cache(False)
        assert mask.cache is False
        assert calls == []

        mask.set_cache(True)
        assert mask.cache is True
        assert calls == [True]

    def test_export_gds_non_cache_cleans_unreferenced_and_writes_pool(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        created_writers: list[FakeGDSWriter] = []

        def _writer_factory() -> FakeGDSWriter:
            w = FakeGDSWriter()
            created_writers.append(w)
            return w

        monkeypatch.setattr(smlay, "GDSWriter", _writer_factory)

        mask = smlay.Mask("phase4_export")
        child = make_rect(0, 0, 5, 5)
        unref = make_rect(0, 0, 1, 1)
        mask.addCell("CHILD", child)
        mask.addCell("UNREF", unref)
        main = make_sref(0, 0, "CHILD", LayoutPool["CHILD"])
        mask.addToMainCell(main)

        _DevicePool["h_keep"] = "CHILD"
        _DevicePool["h_drop"] = "UNREF"
        _DeviceLocalParamPool["h_keep"] = {}
        _DeviceLocalParamPool["h_drop"] = {}

        mask.exportGDS()

        assert "UNREF" not in LayoutPool
        assert "CHILD" in LayoutPool
        assert "h_drop" not in _DevicePool
        assert "h_drop" not in _DeviceLocalParamPool
        assert "h_keep" in _DevicePool

        writer = created_writers[0]
        assert writer.calls[0] == ("open_library", "phase4_export.gds")
        assert writer.calls[1][0] == "write_pool"
        assert writer.calls[2] == ("close_library", None)

    def test_export_gds_cache_success_uses_reader_cache_and_exports_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        created_writers: list[FakeGDSWriter] = []
        fake_reader = FakeGDSReader(celldata={"CELL00": b"x", "OLD": b"y"})
        export_cache_calls: list[bool] = []

        def _writer_factory() -> FakeGDSWriter:
            w = FakeGDSWriter()
            created_writers.append(w)
            return w

        monkeypatch.setattr(smlay, "GDSWriter", _writer_factory)
        monkeypatch.setattr(smlay, "GDSReader", lambda: fake_reader)

        mask = smlay.Mask("phase4_cache_export")
        mask.addToMainCell(make_rect(0, 0, 2, 2))
        mask.cache = True

        def _capture_export_cache() -> None:
            export_cache_calls.append(True)

        monkeypatch.setattr(mask, "_Mask__export_cache", _capture_export_cache)

        mask.exportGDS()

        assert fake_reader.quick_read_calls == ["phase4_cache_export.gds"]
        writer = created_writers[0]
        assert writer.calls[1][0] == "write_pool_use_cache"
        cached_cells = writer.calls[1][2]
        assert isinstance(cached_cells, list)
        assert "CELL00" not in cached_cells
        assert export_cache_calls == [True]

    def test_import_gds_single_top_candidate_sets_mainsymbol_and_sref_groups(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        fake_reader = FakeGDSReader(celldata={"TOP": b"", "SUB": b""})
        sub = make_rect(0, 0, 5, 5)
        top = make_sref(0, 0, "SUB", GeomGroup())
        fake_reader.cell_geometries["TOP"] = top
        fake_reader.cell_geometries["SUB"] = sub
        monkeypatch.setattr(smlay, "GDSReader", lambda: fake_reader)

        mask = smlay.Mask("phase4_import_single")
        mask.importGDS("fake.gds")

        assert mask.mainsymbol == "TOP"
        sref = LayoutPool["TOP"].group[0]
        assert isinstance(sref, SRef)
        assert sref.group is LayoutPool["SUB"]

    def test_import_gds_multiple_candidates_selects_one_with_most_subrefs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
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

        mask = smlay.Mask("phase4_import_multi")
        mask.importGDS("fake.gds")

        assert mask.mainsymbol == "A"


class TestMarker:
    def test_marker_inits_correct_attributes(self):
        name = "TestMarker"
        dev = CrossMark.build()
        x0 = 10
        y0 = 20
        marker = smlay.Marker(name, dev, x0, y0)
        assert marker.name == name
        assert marker.dev == dev
        assert marker.x0 == x0
        assert marker.y0 == y0

    def test_marker_inits_default_attributes(self):
        name = "TestMarker"
        dev = CrossMark.build()
        marker = smlay.Marker(name, dev)
        assert marker.name == name
        assert marker.dev == dev
        assert marker.x0 == 0
        assert marker.y0 == 0

    def test_marker_get_geom(self):
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
    def test_markerset_inits_correct_attributes(self):
        name = "TestMarkerSet"
        dev = CrossMark.build()
        x0 = 10
        y0 = 20
        mset = 2
        xdist = 200
        ydist = 300
        marker_set = smlay.MarkerSet(name, dev, x0, y0, mset, xdist, ydist)
        assert issubclass(smlay.MarkerSet, smlay.Marker)
        assert marker_set.name == name
        assert marker_set.dev == dev
        assert marker_set.x0 == x0
        assert marker_set.y0 == y0
        assert marker_set.mset == mset
        assert marker_set.xdist == xdist
        assert marker_set.ydist == ydist

    def test_markerset_inits_default_attributes(self):
        name = "TestMarkerSet"
        dev = CrossMark.build()

        marker_set = smlay.MarkerSet(name, dev)
        assert marker_set.name == name
        assert marker_set.dev == dev
        assert marker_set.y0 == 0
        assert marker_set.x0 == 0
        assert marker_set.mset == 4
        assert marker_set.xdist == 1000
        assert marker_set.ydist == 1000

    @pytest.mark.xfail(reason="Invalid mset silently ignored", strict=True)
    def test_markerset_init_raises_on_invalid_mset(self):
        name = "TestMarkerSet"
        dev = CrossMark.build()
        with pytest.raises(ValueError):
            smlay.MarkerSet(name, dev, mset=0)

    @pytest.mark.xfail(
        reason="Geometry is not translated correctly for mset==1", strict=True
    )
    def test_markerset_get_geom_mset1(self):
        name = "TestMarkerSet"
        dev = CrossMark.build()
        dev.use_references = False
        x0 = 10
        y0 = 20
        mset = 1
        xdist = 200
        ydist = 300
        marker_set = smlay.MarkerSet(name, dev, x0, y0, mset, xdist, ydist)
        g = marker_set.get_geom()
        assert dev.use_references is True
        assert isinstance(g, GeomGroup)
        assert len(g.group) == 1
        assert isinstance(g.group[0], SRef)
        assert dev._name in g.group[0].cellname

        bb = g.bounding_box()
        assert bb.cx() == pytest.approx(x0)
        assert bb.cy() == pytest.approx(y0)

    def test_markerset_get_geom_mset2(self):
        name = "TestMarkerSet"
        dev = CrossMark.build()
        dev.use_references = False
        x0 = 10
        y0 = 20
        mset = 2
        xdist = 200
        ydist = 300
        marker_set = smlay.MarkerSet(name, dev, x0, y0, mset, xdist, ydist)
        g = marker_set.get_geom()
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

    def test_markerset_get_geom_mset4(self):
        name = "TestMarkerSet"
        dev = CrossMark.build()
        dev.use_references = False
        x0 = 10
        y0 = 20
        mset = 4
        xdist = 200
        ydist = 300
        marker_set = smlay.MarkerSet(name, dev, x0, y0, mset, xdist, ydist)
        g = marker_set.get_geom()
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

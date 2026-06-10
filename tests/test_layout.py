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
from samplemaker.makers import make_rect
from samplemaker.shapes import ARef, GeomGroup, Poly, SRef, Text


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


class FakeGDSWriter:
    """Simple writer double used to assert Mask export call sequencing."""

    def __init__(self):
        self.calls: list[tuple[object, ...]] = []

    def open_library(self, filename: str) -> None:
        self.calls.append(("open_library", filename))

    def write_pool(self, pool: dict[str, GeomGroup]) -> None:
        self.calls.append(("write_pool", sorted(pool.keys())))

    def write_pool_use_cache(
        self, pool: dict[str, GeomGroup], celldata: dict[str, bytes]
    ) -> None:
        self.calls.append(
            ("write_pool_use_cache", sorted(pool.keys()), sorted(celldata.keys()))
        )

    def close_library(self) -> None:
        self.calls.append(("close_library", None))


class FakeGDSReader:
    """Simple reader double used to control Mask cache import/export paths."""

    def __init__(
        self,
        celldata: dict[str, bytes] | None = None,
        quick_read_exception: Exception | None = None,
    ):
        self.celldata = {} if celldata is None else dict(celldata)
        self.quick_read_exception = quick_read_exception
        self.quick_read_calls: list[str] = []
        self.cell_geometries: dict[str, GeomGroup] = {}

    def quick_read(self, filename: str) -> None:
        self.quick_read_calls.append(filename)
        if self.quick_read_exception is not None:
            raise self.quick_read_exception

    def get_cell(self, cellname: str) -> GeomGroup:
        return self.cell_geometries[cellname]


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

        left = [t for t in texts if t.x0 == pytest.approx(9) and t.y0 == pytest.approx(20)]
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

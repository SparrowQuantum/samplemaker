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
from samplemaker.shapes import ARef, GeomGroup, SRef


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

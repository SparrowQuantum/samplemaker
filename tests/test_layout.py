"""Unit tests for the samplemaker.layout module."""

import pytest

import samplemaker.layout as smlay
from samplemaker.baselib.devices import CrossMark
from samplemaker.shapes import ARef, GeomGroup, SRef


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

"""Unit tests for routing helpers."""

import pytest

import samplemaker.routers as rt
from samplemaker.devices import DevicePort

CONNECTABLE_FACING = getattr(rt, "__connectable_facing")
CONNECTABLE_BEND = getattr(rt, "__connectable_bend")
CONNECT_STEP = getattr(rt, "__connect_step")


def _make_port(x0: float, y0: float, direction: str) -> DevicePort:
    direction_to_flags = {
        "E": (True, True),
        "W": (True, False),
        "N": (False, True),
        "S": (False, False),
    }
    hv, bf = direction_to_flags[direction]
    return DevicePort(x0, y0, hv, bf)


class TestConnectableFacing:
    def test_horizontal_straight(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(10.0, 0.0, "W")

        ok, seq = CONNECTABLE_FACING(p1, p2, rad=3)

        assert ok is True
        assert seq == [["S", 10.0]]

    def test_horizontal_c_bend_with_straights(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(10.0, 2.0, "W")

        ok, seq = CONNECTABLE_FACING(p1, p2, rad=3)

        assert ok is True
        assert seq == [["S", 2.0], ["C", 2.0, 3], ["S", 2.0]]

    def test_horizontal_c_bend_without_straights_when_tight(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(4.0, 2.0, "W")

        ok, seq = CONNECTABLE_FACING(p1, p2, rad=3)

        assert ok is True
        assert seq == [["C", 2.0, 2.0]]

    def test_vertical_straight(self) -> None:
        p1 = _make_port(0.0, 0.0, "N")
        p2 = _make_port(0.0, 10.0, "S")

        ok, seq = CONNECTABLE_FACING(p1, p2, rad=3)

        assert ok is True
        assert seq == [["S", 10.0]]

    def test_vertical_c_bend(self) -> None:
        p1 = _make_port(0.0, 0.0, "N")
        p2 = _make_port(2.0, 10.0, "S")

        ok, seq = CONNECTABLE_FACING(p1, p2, rad=3)

        assert ok is True
        assert seq == [["S", 2.0], ["C", -2.0, 3], ["S", 2.0]]

    def test_vertical_c_bend_without_straights_when_tight(self) -> None:
        p1 = _make_port(0.0, 0.0, "N")
        p2 = _make_port(2.0, 4.0, "S")

        ok, seq = CONNECTABLE_FACING(p1, p2, rad=3)

        assert ok is True
        assert seq == [["C", -2.0, 2.0]]

    def test_not_connectable_when_not_facing(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(10.0, 0.0, "E")

        ok, seq = CONNECTABLE_FACING(p1, p2, rad=3)

        assert ok is False
        assert seq == []


class TestConnectableBend:
    def test_connectable_bend_left(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(10.0, 10.0, "S")

        ok, seq = CONNECTABLE_BEND(p1, p2, rad=3)

        assert ok is True
        assert seq == [["S", 7.0], ["B", 90, 3], ["S", 7.0]]

    def test_connectable_bend_right(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(10.0, -10.0, "N")

        ok, seq = CONNECTABLE_BEND(p1, p2, rad=3)

        assert ok is True
        assert seq == [["S", 7.0], ["B", -90, 3], ["S", 7.0]]

    def test_not_connectable_when_parallel(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(5.0, 2.0, "E")

        ok, seq = CONNECTABLE_BEND(p1, p2, rad=3)

        assert ok is False
        assert seq == []


class TestConnectStep:
    def test_connect_step_returns_true_from_left_bend_path(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(-14.0, 4.0, "E")

        ok, seq = CONNECT_STEP(p1, p2, rad=3)

        assert ok is True
        assert seq == [
            ["B", 90, 3],
            ["S", 2.0],
            ["B", 90, 3],
            ["S", 4.0],
            ["C", 4.0, 3],
            ["S", 4.0],
        ]

    def test_connect_step_returns_true_from_right_bend_path(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(-12.0, -12.0, "E")

        ok, seq = CONNECT_STEP(p1, p2, rad=3)

        assert ok is True
        assert seq == [
            ["B", -90, 3],
            ["S", 6.0],
            ["B", -90, 3],
            ["S", 12.0],
        ]

    def test_connect_step_returns_false_and_chooses_left_fallback(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(-12.0, 2.0, "E")

        ok, seq = CONNECT_STEP(p1, p2, rad=3)

        assert ok is False
        assert seq == [["B", 90, 3]]
        assert p1.angle_to_text() == "N"


class TestWaveguideConnect:
    def test_waveguide_connect_trivial_facing(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(10.0, 0.0, "W")

        ok, seq = rt.WaveguideConnect(p1, p2, rad=3)

        assert ok is True
        assert seq == [["S", 10.0]]

    def test_waveguide_connect_single_bend(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(10.0, 10.0, "S")

        ok, seq = rt.WaveguideConnect(p1, p2, rad=3)

        assert ok is True
        assert seq == [["S", 7.0], ["B", 90, 3], ["S", 7.0]]

    def test_waveguide_connect_returns_failure_when_unreachable(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(0.0, 0.0, "E")

        ok, seq = rt.WaveguideConnect(p1, p2, rad=3)

        assert ok is False
        assert seq == []


class TestElbowRouter:
    def test_elbow_router_straight_when_aligned(self) -> None:
        p1 = _make_port(0.0, 0.0, "E")
        p2 = _make_port(10.0, 0.0, "W")

        xpts, ypts = rt.ElbowRouter(p1, p2, offset=5)

        assert xpts == pytest.approx([0.0, 10.0])
        assert ypts == pytest.approx([0.0, 0.0])

    def test_elbow_router_curve_preserves_endpoints(self) -> None:
        p1 = _make_port(1.0, 2.0, "E")
        p2 = _make_port(11.0, 5.0, "N")

        xpts, ypts = rt.ElbowRouter(p1, p2, offset=2)

        assert len(xpts) == 7
        assert len(ypts) == 7
        assert xpts[0] == pytest.approx(p1.x0)
        assert ypts[0] == pytest.approx(p1.y0)
        assert xpts[-1] == pytest.approx(p2.x0)
        assert ypts[-1] == pytest.approx(p2.y0)

    def test_elbow_router_rotated_frame_preserves_endpoints(self) -> None:
        p1 = _make_port(0.0, 0.0, "N")
        p2 = _make_port(-3.0, 8.0, "E")

        xpts, ypts = rt.ElbowRouter(p1, p2, offset=1.5)

        assert xpts[0] == pytest.approx(p1.x0)
        assert ypts[0] == pytest.approx(p1.y0)
        assert xpts[-1] == pytest.approx(p2.x0)
        assert ypts[-1] == pytest.approx(p2.y0)

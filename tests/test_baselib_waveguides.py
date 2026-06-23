"""Tests for the base waveguide library in samplemaker.baselib.waveguides."""

import math

import pytest

import samplemaker.baselib.waveguides as smwvg
import samplemaker.sequencer as smseq
from samplemaker.devices import DevicePort
from samplemaker.shapes import GeomGroup, Poly


def test_base_waveguide_options(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []

    def mock_default_options() -> dict:
        called.append(True)
        return {}

    monkeypatch.setattr(smseq, "default_options", mock_default_options)
    opts = smwvg.create_base_waveguide_options()
    assert called == [True]
    assert isinstance(opts, dict)
    assert "wgLayer" in opts
    assert "bendResolution" in opts
    assert "defaultWidth" in opts
    assert opts["wgLayer"] == 1
    assert opts["bendResolution"] == 30
    assert opts["defaultWidth"] == 0.3


def test_base_waveguide_state() -> None:
    state = smwvg.BaseWaveguideState()
    assert isinstance(state, smseq.SequencerState)
    assert "w" in state.state
    assert state.state["w"] == 0


def test_base_waveguide_init(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []

    def mock_init_state(args: list, state: dict, options: dict) -> None:
        called.append(state.copy())

    monkeypatch.setattr(smseq, "_init_state", mock_init_state)
    state = smseq.SequencerState()
    options = {"defaultWidth": 0.5, "__no_init__": False}
    smwvg.base_waveguide_init([], state.state, options)
    assert len(called) == 1
    assert "w" not in called[0]  # init first
    assert state.state["w"] == 0.5


def test_base_waveguide_init_no_init(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []

    def mock_init_state(args: list, state: dict, options: dict) -> None:
        called.append(state.copy())

    monkeypatch.setattr(smseq, "_init_state", mock_init_state)
    state = smseq.SequencerState()
    options = {"defaultWidth": 0.5, "__no_init__": True}
    smwvg.base_waveguide_init([], state.state, options)
    assert len(called) == 1
    assert "w" not in called[0]  # init first
    assert "w" not in state.state  # No init, so w should not be set


def test_base_waveguide_straight_horizontal() -> None:
    args = [10]
    state = {"w": 0.4, "x": 0, "y": 0, "a": 0, "__OL__": 0}
    options = {"wgLayer": 1, "bendResolution": 30}
    g = smwvg.base_waveguide_straight(args, state, options)

    assert isinstance(g, GeomGroup)
    assert len(g.group) == 1
    assert isinstance(g.group[0], Poly)
    assert g.group[0].layer == 1
    bb = g.bounding_box()
    assert bb.llx == pytest.approx(0)
    assert bb.lly == pytest.approx(-state["w"] / 2)
    assert bb.urx == pytest.approx(args[0])
    assert bb.ury == pytest.approx(state["w"] / 2)
    assert state["x"] == pytest.approx(args[0])
    assert state["y"] == pytest.approx(0)
    assert state["a"] == pytest.approx(0)
    assert state["w"] == 0.4
    assert state["__OL__"] == pytest.approx(args[0])


def test_base_waveguide_straight_vertical() -> None:
    args = [10]
    state = {"w": 0.4, "x": 0, "y": 0, "a": 90, "__OL__": 0}
    options = {"wgLayer": 1, "bendResolution": 30}
    g = smwvg.base_waveguide_straight(args, state, options)

    assert isinstance(g, GeomGroup)
    assert len(g.group) == 1
    assert isinstance(g.group[0], Poly)
    assert g.group[0].layer == 1
    bb = g.bounding_box()
    assert bb.llx == pytest.approx(-state["w"] / 2)
    assert bb.lly == pytest.approx(0)
    assert bb.urx == pytest.approx(state["w"] / 2)
    assert bb.ury == pytest.approx(args[0])
    assert state["x"] == pytest.approx(0)
    assert state["y"] == pytest.approx(args[0])
    assert state["a"] == pytest.approx(90)
    assert state["w"] == 0.4
    assert state["__OL__"] == pytest.approx(args[0])


def test_base_waveguide_straight_zero_length() -> None:
    args = [0]
    state = {"w": 0.4, "x": 0, "y": 0, "a": 0, "__OL__": 0}
    options = {"wgLayer": 1, "bendResolution": 30}
    g = smwvg.base_waveguide_straight(args, state, options)

    assert isinstance(g, GeomGroup)
    assert len(g.group) == 0
    assert state["x"] == pytest.approx(0)
    assert state["y"] == pytest.approx(0)
    assert state["a"] == pytest.approx(0)
    assert state["w"] == pytest.approx(0.4)
    assert state["__OL__"] == pytest.approx(0)


def test_base_waveguide_bend_positive_angle() -> None:
    args = [90, 5]
    state = {"w": 0.4, "x": 0, "y": 0, "a": 0, "__OL__": 0}
    options = {"wgLayer": 2, "bendResolution": 30}

    g = smwvg.base_waveguide_bend(args, state, options)

    assert isinstance(g, GeomGroup)
    assert len(g.group) == 1
    assert isinstance(g.group[0], Poly)
    assert g.group[0].layer == 2
    assert state["x"] == pytest.approx(5)
    assert state["y"] == pytest.approx(5)
    assert state["a"] == pytest.approx(90)
    assert state["__OL__"] == pytest.approx(5 * math.pi / 2)


def test_base_waveguide_bend_negative_angle() -> None:
    args = [-90, 4]
    state = {"w": 0.3, "x": 1, "y": 2, "a": 0, "__OL__": 0}
    options = {"wgLayer": 1, "bendResolution": 20}

    smwvg.base_waveguide_bend(args, state, options)

    assert state["x"] == pytest.approx(5)
    assert state["y"] == pytest.approx(-2)
    assert state["a"] == pytest.approx(-90)
    assert state["__OL__"] == pytest.approx(2 * math.pi)


def test_base_waveguide_bend_zero_angle() -> None:
    args = [0, 3]
    state = {"w": 0.4, "x": 2, "y": 3, "a": 45, "__OL__": 1}
    options = {"wgLayer": 1, "bendResolution": 30}

    g = smwvg.base_waveguide_bend(args, state, options)

    assert isinstance(g, GeomGroup)
    assert len(g.group) == 0
    assert state == {"w": 0.4, "x": 2, "y": 3, "a": 45, "__OL__": 1}


def test_base_waveguide_cosine_bend_changes_state() -> None:
    args = [2, 6]
    state = {"w": 0.35, "x": 0, "y": 0, "a": 0, "__OL__": 0}
    options = {"wgLayer": 7, "bendResolution": 40}

    g = smwvg.base_waveguide_cosine_bend(args, state, options)

    assert isinstance(g, GeomGroup)
    assert len(g.group) == 1
    assert state["x"] == pytest.approx(12)
    assert state["y"] == pytest.approx(2)
    assert state["a"] == pytest.approx(0)
    assert state["__OL__"] > 12


def test_base_waveguide_cosine_bend_zero_radius() -> None:
    args = [1, 0.01]  # radius is reduced by internal delta
    state = {"w": 0.3, "x": 0, "y": 0, "a": 0, "__OL__": 0}
    options = {"wgLayer": 1, "bendResolution": 10}

    g = smwvg.base_waveguide_cosine_bend(args, state, options)

    assert isinstance(g, GeomGroup)
    assert len(g.group) == 0
    assert state["x"] == pytest.approx(0)
    assert state["y"] == pytest.approx(0)
    assert state["a"] == pytest.approx(0)
    assert state["__OL__"] == pytest.approx(0)


def test_base_waveguide_taper_uses_default_width() -> None:
    args = [10, -1]
    state = {"w": 0.4, "x": 0, "y": 0, "a": 0, "__OL__": 0}
    options = {"wgLayer": 3, "bendResolution": 30, "defaultWidth": 0.8}

    g = smwvg.base_waveguide_taper(args, state, options)

    assert isinstance(g, GeomGroup)
    assert len(g.group) == 1
    assert state["x"] == pytest.approx(10)
    assert state["y"] == pytest.approx(0)
    assert state["w"] == pytest.approx(0.8)
    assert state["__OL__"] == pytest.approx(10)


def test_base_waveguide_taper_zero_length() -> None:
    args = [0, 1.2]
    state = {"w": 0.4, "x": 4, "y": 5, "a": 90, "__OL__": 3}
    options = {"wgLayer": 1, "bendResolution": 30, "defaultWidth": 0.7}

    g = smwvg.base_waveguide_taper(args, state, options)

    assert isinstance(g, GeomGroup)
    assert len(g.group) == 0
    assert state == {"w": 0.4, "x": 4, "y": 5, "a": 90, "__OL__": 3}


def test_base_waveguide_offset() -> None:
    args = [2]
    state = {"w": 0.3, "x": 0, "y": 0, "a": 0, "__OL__": 1}
    options = {"wgLayer": 1, "bendResolution": 30}

    g = smwvg.base_waveguide_offset(args, state, options)

    assert isinstance(g, GeomGroup)
    assert len(g.group) == 0
    assert state["x"] == pytest.approx(0)
    assert state["y"] == pytest.approx(2)
    assert state["__OL__"] == pytest.approx(1)


def test_base_waveguide_commands() -> None:
    commands = smwvg.create_base_waveguide_commands()

    assert commands["INIT"][0] == 0
    assert commands["S"][0] == 1
    assert commands["B"][0] == 2
    assert commands["C"][0] == 2
    assert commands["T"][0] == 2
    assert commands["OFF"][0] == 1
    assert commands["S"][1] is smwvg.base_waveguide_straight
    assert commands["B"][1] is smwvg.base_waveguide_bend


def test_base_waveguide_sequencer_runs() -> None:
    seq = [["S", 5], ["T", 2, 0.5]]
    s = smwvg.BaseWaveguideSequencer(seq)

    g = s.run()

    assert isinstance(s, smseq.Sequencer)
    assert isinstance(g, GeomGroup)
    assert len(g.group) == 2
    assert s.state["x"] == pytest.approx(7)
    assert s.state["y"] == pytest.approx(0)
    assert s.state["w"] == pytest.approx(0.5)


def test_base_waveguide_connector_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_connect(
        port1: DevicePort, port2: DevicePort, radius: float
    ) -> tuple[bool, list[list[object]]]:
        captured["radius"] = radius
        captured["ports"] = (port1, port2)
        return True, [["S", 1]]

    class FakeSequencer:
        def __init__(self, seq: list[list[object]]) -> None:
            captured["seq"] = seq
            self.options = {"wgLayer": 99}

        def run(self) -> GeomGroup:
            captured["run_called"] = True
            return GeomGroup()

    monkeypatch.setattr(smwvg, "connect_waveguide_ports", fake_connect)
    monkeypatch.setattr(smwvg, "BaseWaveguideSequencer", FakeSequencer)

    p1 = smwvg.BaseWaveguidePort(0, 0, "E")
    p2 = smwvg.BaseWaveguidePort(10, 0, "W")
    g = smwvg.connect_base_waveguide_ports(p1, p2)

    assert isinstance(g, GeomGroup)
    assert captured["radius"] == smwvg.BaseWaveguideConnectorOptions["bending_radius"]
    assert captured["seq"] == [["S", 1]]
    assert captured["run_called"] is True


def test_base_waveguide_connector_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_connect(
        port1: DevicePort, port2: DevicePort, radius: float
    ) -> tuple[bool, list[list[object]]]:
        _ = (port1, port2, radius)
        return False, []

    monkeypatch.setattr(smwvg, "connect_waveguide_ports", fake_connect)

    p1 = smwvg.BaseWaveguidePort(0, 0, "E")
    p2 = smwvg.BaseWaveguidePort(1, 1, "N")
    g = smwvg.connect_base_waveguide_ports(p1, p2)

    assert isinstance(g, GeomGroup)
    assert len(g.group) == 0


@pytest.mark.parametrize(
    ("orient", "expected"),
    [
        ("East", "E"),
        ("W", "W"),
        ("north", "N"),
        ("s", "S"),
    ],
)
def test_base_waveguide_port_orientation(orient: str, expected: str) -> None:
    port = smwvg.BaseWaveguidePort(3, 4, orient=orient, width=0.45, name="io")

    assert port.angle_to_text() == expected
    assert port.width == pytest.approx(0.45)
    assert port.name == "io"
    assert port.connector_function is smwvg.connect_base_waveguide_ports

import pytest

import samplemaker.baselib.waveguides as smwvg
import samplemaker.sequencer as smseq
from samplemaker.shapes import Poly


def test_base_waveguide_options(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []

    def mock_default_options() -> dict:
        called.append(True)
        return {}

    monkeypatch.setattr(smseq, "default_options", mock_default_options)
    opts = smwvg.BaseWaveguideOptions()
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

    def mock_init_state(state: dict, options: dict) -> None:
        called.append(state.copy())

    monkeypatch.setattr(smseq, "__initState", mock_init_state)
    state = smseq.SequencerState()
    options = {"defaultWidth": 0.5, "__no_init__": False}
    smwvg.BaseWaveguideINIT(state.state, options)
    assert len(called) == 1
    assert "w" not in called[0]  # init first
    assert state.state["w"] == 0.5


def test_base_waveguide_init_no_init(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []

    def mock_init_state(state: dict, options: dict) -> None:
        called.append(state.copy())

    monkeypatch.setattr(smseq, "__initState", mock_init_state)
    state = smseq.SequencerState()
    options = {"defaultWidth": 0.5, "__no_init__": True}
    smwvg.BaseWaveguideINIT(state.state, options)
    assert len(called) == 1
    assert "w" not in called[0]  # init first
    assert "w" not in state.state  # No init, so w should not be set


def test_base_waveguide_s_horizontal() -> None:
    args = [10]
    state = {"w": 0.4, "x": 0, "y": 0, "a": 0, "__OL__": 0}
    options = {"wgLayer": 1, "bendResolution": 30}
    g = smwvg.BaseWaveguideS(args, state, options)

    assert isinstance(g, smseq.GeomGroup)
    assert len(g.group) == 1
    assert isinstance(g.group[0], Poly)
    assert g.group[0].layer == 1
    bb = g.bounding_box()
    assert bb.llx == pytest.approx(0)
    assert bb.lly == pytest.approx(-state["w"] / 2)
    assert bb.urx() == pytest.approx(args[0])
    assert bb.ury() == pytest.approx(state["w"] / 2)

    assert state["x"] == pytest.approx(args[0])
    assert state["y"] == pytest.approx(0)
    assert state["a"] == pytest.approx(0)
    assert state["w"] == 0.4
    assert state["__OL__"] == pytest.approx(args[0])


def test_base_waveguide_s_vertical() -> None:
    args = [10]
    state = {"w": 0.4, "x": 0, "y": 0, "a": 90, "__OL__": 0}
    options = {"wgLayer": 1, "bendResolution": 30}
    g = smwvg.BaseWaveguideS(args, state, options)

    assert isinstance(g, smseq.GeomGroup)
    assert len(g.group) == 1
    assert isinstance(g.group[0], Poly)
    assert g.group[0].layer == 1
    bb = g.bounding_box()
    assert bb.llx == pytest.approx(-state["w"] / 2)
    assert bb.lly == pytest.approx(0)
    assert bb.urx() == pytest.approx(state["w"] / 2)
    assert bb.ury() == pytest.approx(args[0])

    assert state["x"] == pytest.approx(0)
    assert state["y"] == pytest.approx(args[0])
    assert state["a"] == pytest.approx(90)
    assert state["w"] == 0.4
    assert state["__OL__"] == pytest.approx(args[0])


def test_base_waveguide_s_zero_length() -> None:
    args = [0]
    state = {"w": 0.4, "x": 0, "y": 0, "a": 0, "__OL__": 0}
    options = {"wgLayer": 1, "bendResolution": 30}
    g = smwvg.BaseWaveguideS(args, state, options)

    assert isinstance(g, smseq.GeomGroup)
    assert len(g.group) == 0

    assert state["x"] == pytest.approx(0)
    assert state["y"] == pytest.approx(0)
    assert state["a"] == pytest.approx(0)
    assert state["w"] == pytest.approx(0.4)
    assert state["__OL__"] == pytest.approx(0)
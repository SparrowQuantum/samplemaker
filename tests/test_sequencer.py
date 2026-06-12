"""Unit tests for the sequencer module."""

import inspect
from collections.abc import Callable

import pytest

from samplemaker import devices as smdev
from samplemaker import sequencer as smseq
from samplemaker.shapes import GeomGroup, SRef

_DCMD = dict[str, Callable]


@pytest.fixture
def default_command_list() -> _DCMD:
    """Fixture for the default command list."""
    return smseq.default_command_list()


@pytest.fixture
def sequencer_test_state() -> dict:
    """Fixture for a sample sequencer state."""
    return {
        "x": 1.3,
        "y": 2.7,
        "a": 0,
        "__OL__": 5,
        "__XC__": 1,
        "__YC__": 2,
        "STORED": [[1, 2]],
    }


class TestDefaultCommandList:
    def _eval_signature(
        self, value: tuple[int, Callable], num_args: int, expected_params: set[str]
    ) -> None:
        assert isinstance(value, tuple) and len(value) == 2

        num_args_value, func = value
        assert num_args_value == num_args
        assert callable(func)
        params = set(inspect.signature(func).parameters)
        assert params == expected_params

    def test_default_command_list_keys_are_correct(
        self, default_command_list: _DCMD
    ) -> None:
        expected_keys = {"INIT", "STATE", "CENTER", "STORE", "DEV"}
        assert set(default_command_list) == expected_keys

    def test_init_command_has_correct_signature(
        self, default_command_list: _DCMD
    ) -> None:
        value = default_command_list["INIT"]
        num_args = 0
        expected_params = {"state", "options"}
        self._eval_signature(value, num_args, expected_params)

    def test_init_command_correctly_modifies_state(
        self, default_command_list: _DCMD, sequencer_test_state: dict
    ) -> None:
        _, fun = default_command_list["INIT"]
        expected_state = {
            "x": 0,
            "y": 0,
            "a": 0,
            "__OL__": 0,
            "__XC__": 0,
            "__YC__": 0,
            "STORED": [],
        }

        options = {"__no_init__": False}
        res = fun(sequencer_test_state, options)
        assert sequencer_test_state == expected_state
        assert options == {"__no_init__": False}  # Ensure options is not modified
        assert res is None

    def test_init_no_init_option_does_nothing(
        self, default_command_list: _DCMD, sequencer_test_state: dict
    ) -> None:
        _, fun = default_command_list["INIT"]
        expected_state = sequencer_test_state.copy()

        options = {"__no_init__": True}
        res = fun(sequencer_test_state, options)
        assert sequencer_test_state == expected_state
        assert options == {"__no_init__": True}  # Ensure options is not modified
        assert res is None

    def test_state_command_has_correct_signature(
        self, default_command_list: _DCMD
    ) -> None:
        value = default_command_list["STATE"]
        num_args = 2
        expected_params = {"args", "state", "options"}
        self._eval_signature(value, num_args, expected_params)

    def test_state_command_changes_state(self, default_command_list: _DCMD) -> None:
        _, fun = default_command_list["STATE"]
        state = {"x": 0}
        args = ["x", 5]
        options = {}
        res = fun(args, state, options)
        assert state["x"] == 5
        assert args == ["x", 5]  # Ensure args is not modified
        assert options == {}  # Ensure options is not modified
        assert isinstance(res, GeomGroup)
        assert len(res.group) == 0

    def test_center_command_has_correct_signature(
        self, default_command_list: _DCMD
    ) -> None:
        value = default_command_list["CENTER"]
        num_args = 2
        expected_params = {"args", "state", "options"}
        self._eval_signature(value, num_args, expected_params)

    def test_center_command_correctly_modifies_state(
        self, default_command_list: _DCMD, sequencer_test_state: dict
    ) -> None:
        _, fun = default_command_list["CENTER"]
        test_state = sequencer_test_state.copy()
        x0 = test_state["x"]
        y0 = test_state["y"]
        args = [3.2, 4.1]
        expected_xc = 3.2 - x0
        expected_yc = 4.1 - y0
        options = {}
        res = fun(args, test_state, options)
        assert test_state.keys() == sequencer_test_state.keys()
        assert test_state["__XC__"] == expected_xc
        assert test_state["__YC__"] == expected_yc
        for key, value in sequencer_test_state.items():
            if key not in {"__XC__", "__YC__"}:
                assert test_state[key] == value
        assert args == [3.2, 4.1]  # Ensure args is not modified
        assert options == {}  # Ensure options is not modified
        assert isinstance(res, GeomGroup)
        assert len(res.group) == 0

    def test_store_command_has_correct_signature(
        self, default_command_list: _DCMD
    ) -> None:
        value = default_command_list["STORE"]
        num_args = 0
        expected_params = {"args", "state", "options"}
        self._eval_signature(value, num_args, expected_params)

    def test_store_command_saves_coordinates(
        self, default_command_list: _DCMD, sequencer_test_state: dict
    ) -> None:
        _, fun = default_command_list["STORE"]
        test_state = sequencer_test_state.copy()
        x0 = test_state["x"]
        y0 = test_state["y"]
        expected_stored = test_state["STORED"] + [[x0, y0]]
        args = []
        options = {}
        res = fun(args, test_state, options)
        assert test_state.keys() == sequencer_test_state.keys()
        assert test_state["STORED"] == expected_stored
        for key, value in sequencer_test_state.items():
            if key != "STORED":
                assert test_state[key] == value
        assert args == []  # Ensure args is not modified
        assert options == {}  # Ensure options is not modified
        assert isinstance(res, GeomGroup)
        assert len(res.group) == 0

    def test_dev_command_has_correct_signature(
        self, default_command_list: _DCMD
    ) -> None:
        value = default_command_list["DEV"]
        num_args = 3
        expected_params = {"args", "state", "options"}
        self._eval_signature(value, num_args, expected_params)

    def test_dev_command_propagates_state_and_returns_sref(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        default_command_list: _DCMD,
        sequencer_test_state: dict,
    ) -> None:
        _ = dummy_device_list
        _, fun = default_command_list["DEV"]
        test_state = sequencer_test_state.copy()
        length_param = 5
        x0 = test_state["x"]
        y0 = test_state["y"]
        expected_x = x0 + length_param
        expected_y = y0 + length_param
        expected_a = 90  # Device has north facing port

        devname = "TESTLIB_TWO_PORT"
        inport = "p1"
        outport = "p2"
        args = [devname, inport, outport]
        options = {f"dev_{devname}": {"length": length_param}}
        res = fun(args, test_state, options)

        assert test_state.keys() == sequencer_test_state.keys()
        assert test_state["x"] == pytest.approx(expected_x)
        assert test_state["y"] == pytest.approx(expected_y)
        assert test_state["a"] == pytest.approx(expected_a)
        for key, value in sequencer_test_state.items():
            if key not in {"x", "y", "a"}:
                assert test_state[key] == value
        assert args == [devname, inport, outport]
        assert options == {f"dev_{devname}": {"length": length_param}}
        assert isinstance(res, GeomGroup)
        assert len(res.group) == 1
        assert isinstance(res.group[0], SRef)
        assert devname in res.group[0].cellname

    def test_dev_command_rotates_translates_geometry_correctly(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        default_command_list: _DCMD,
        sequencer_test_state: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ = dummy_device_list
        _, fun = default_command_list["DEV"]

        # Rotate around input port (0, 0)
        expected_x0 = 0
        expected_y0 = 0

        # Translate input port to state (x, y)
        expected_dx = sequencer_test_state["x"]
        expected_dy = sequencer_test_state["y"]
        expected_angle = sequencer_test_state["a"] - 360

        rotations = []
        translations = []

        def capture_rotate(self: GeomGroup, x: float, y: float, angle: float) -> None:
            rotations.append((x, y, angle))

        def capture_translate(self: GeomGroup, dx: float, dy: float) -> None:
            translations.append((dx, dy))

        monkeypatch.setattr(GeomGroup, "rotate", capture_rotate)
        monkeypatch.setattr(GeomGroup, "translate", capture_translate)

        devname = "TESTLIB_TWO_PORT"
        args = [devname, "p1", "p2"]
        options = {f"dev_{devname}": {"length": 5}}
        fun(args, sequencer_test_state, options)

        rx, ry, rangle = rotations[-1]
        tx, ty = translations[-1]
        assert rx == pytest.approx(expected_x0)
        assert ry == pytest.approx(expected_y0)
        assert rangle == pytest.approx(expected_angle)
        assert tx == pytest.approx(expected_dx)
        assert ty == pytest.approx(expected_dy)

    def test_dev_command_raises_for_invalid_device(
        self, default_command_list: _DCMD, sequencer_test_state: dict
    ) -> None:
        _, fun = default_command_list["DEV"]
        devname = "NONEXISTENT_DEVICE"
        args = [devname, "p1", "p2"]
        options = {}
        with pytest.raises(ValueError, match=f"No device found with name {devname}."):
            fun(args, sequencer_test_state, options)

    def test_dev_command_raises_for_invalid_ports(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        default_command_list: _DCMD,
        sequencer_test_state: dict,
    ) -> None:
        _ = dummy_device_list
        _, fun = default_command_list["DEV"]
        devname = "TESTLIB_TWO_PORT"
        args = [devname, "invalid_inport", "invalid_outport"]
        options = {f"dev_{devname}": {"length": 5}}
        match = (
            f"Device {devname} has no port called invalid_inport or invalid_outport."
        )
        with pytest.raises(ValueError, match=match):
            fun(args, sequencer_test_state.copy(), options)


def test_default_options_configuration(
    dummy_device_list: dict[str, type[smdev.Device]],
) -> None:
    expected_device_params = {}
    for devname, devclass in dummy_device_list.items():
        dev = devclass()
        dev.parameters()
        expected_device_params[f"dev_{devname}"] = dev._p

    expected_keys = set(expected_device_params) | {"__no_init__"}
    options = smseq.default_options()
    assert set(options) == expected_keys
    assert options["__no_init__"] is False
    for key, expected_params in expected_device_params.items():
        assert options[key] == expected_params


def test_init_sequencer_state() -> None:
    state_obj = smseq.SequencerState()
    expected_state = {
        "x": 0,
        "y": 0,
        "a": 0,
        "__OL__": 0,
        "__XC__": 0,
        "__YC__": 0,
        "STORED": [],
    }
    assert hasattr(state_obj, "state")
    assert isinstance(state_obj.state, dict)
    assert state_obj.state == expected_state

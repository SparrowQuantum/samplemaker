"""Unit tests for the sequencer module."""

import inspect
from collections.abc import Callable

import pytest

from samplemaker import devices as smdev
from samplemaker import sequencer as smseq
from samplemaker.makers import make_rect
from samplemaker.shapes import GeomGroup, SRef

_DCMD = dict[str, tuple[int, Callable]]


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


class FakeSequencer(smseq.Sequencer):
    def __init__(self, seq: list) -> None:
        self.calls = []
        super().__init__(
            seq=seq,
            seq_options={},
            seq_state=smseq.SequencerState(),
            seq_dictionary={
                "INIT": (0, self._capture_init),
                "STATE": (2, self._capture_state),
                "ADD_RECT": (2, self._add_rect),
            },
        )

    def _capture_init(self, state: dict, options: dict) -> None:
        self.calls.append(("INIT", state.copy(), options.copy()))

    def _capture_state(self, args: list, state: dict, options: dict) -> GeomGroup:
        self.calls.append(("STATE", args.copy(), state.copy(), options.copy()))
        state[args[0]] = args[1]
        return GeomGroup()

    def _add_rect(self, args: list, state: dict, options: dict) -> GeomGroup:
        self.calls.append(("ADD_RECT", args.copy(), state.copy(), options.copy()))
        return make_rect(state["x"], state["y"], args[0], args[1], numkey=5)


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


class TestSequencer:
    def test_init_sets_correct_attributes(self) -> None:
        seq = [["S", 10], ["CENTER", 1, 2]]
        seq_options = {"hello": "world"}
        seq_state = smseq.SequencerState()
        seq_dict = {"a": "b"}
        sequencer = smseq.Sequencer(
            seq=seq,
            seq_options=seq_options,
            seq_state=seq_state,
            seq_dictionary=seq_dict,
        )

        assert sequencer.seq == seq
        assert sequencer.options == seq_options
        assert sequencer.dic == seq_dict
        assert sequencer.state == seq_state.state
        assert sequencer.debug_state is False

    def test_get_state_returns_state_deepcopy(self) -> None:
        seq = []
        seq_state = smseq.SequencerState()
        seq_state.state["x"] = 1.3
        sequencer = smseq.Sequencer(
            seq=seq,
            seq_options={},
            seq_state=seq_state,
            seq_dictionary={},
        )

        state_copy = sequencer.get_state()
        assert state_copy == sequencer.state
        assert state_copy is not sequencer.state  # Ensure it's a deepcopy

    def test_reset_returns_state_to_origin(self) -> None:
        seq_state = smseq.SequencerState()
        seq_state.state["x"] = 1.3
        seq_state.state["y"] = 2.7
        seq_state.state["a"] = 45
        seq_state.state["__OL__"] = 5
        seq_state.state["__XC__"] = 1
        seq_state.state["__YC__"] = 2
        seq_state.state["STORED"] = [[1, 2]]
        sequencer = smseq.Sequencer(
            seq=[],
            seq_options={},
            seq_state=seq_state,
            seq_dictionary={},
        )

        sequencer.reset()

        assert sequencer.state["x"] == 0
        assert sequencer.state["y"] == 0
        assert sequencer.state["a"] == 0
        assert sequencer.state["__OL__"] == 5
        assert sequencer.state["__XC__"] == 1
        assert sequencer.state["__YC__"] == 2
        assert sequencer.state["STORED"] == [[1, 2]]

    def test_run_calls_sequence(self) -> None:
        # sequencer skips empty instructions
        seq = [["STATE", "x", 5], [], ["STATE", "y", 10]]
        sequencer = FakeSequencer(seq)
        g = sequencer.run()

        assert isinstance(g, GeomGroup)
        assert len(g.group) == 0
        assert sequencer.calls[0][0] == "INIT"
        assert sequencer.calls[1][0] == "STATE"
        assert sequencer.calls[1][1] == ["x", 5]
        assert sequencer.calls[1][2]["x"] == 0  # State before modification
        assert sequencer.calls[1][3] == {}  # Options
        assert sequencer.calls[2][0] == "STATE"
        assert sequencer.calls[2][1] == ["y", 10]
        assert sequencer.calls[2][2]["x"] == 5  # State after modification
        assert sequencer.calls[2][3] == {}  # Options
        assert sequencer.state["x"] == 5  # State after modification
        assert sequencer.state["y"] == 10

    def test_run_raises_for_invalid_command(self) -> None:
        seq = [["INVALID_CMD", 1, 2]]
        sequencer = FakeSequencer(seq)
        with pytest.raises(ValueError, match="Command INVALID_CMD does not exist."):
            sequencer.run()

    def test_run_raises_for_wrong_number_of_arguments(self) -> None:
        seq = [["STATE", "x"]]  # STATE expects 2 arguments
        sequencer = FakeSequencer(seq)
        with pytest.raises(
            ValueError, match="Wrong number of arguments for command STATE."
        ):
            sequencer.run()

    def test_run_centers_geom_and_coords_after_sequence(self) -> None:
        seq = [["ADD_RECT", 2, 3]]
        sequencer = FakeSequencer(seq)
        x0, y0 = 1, 2
        xc, yc = 5, 10
        stored_x0, stored_y0 = 3, 4
        sequencer.state["x"] = x0
        sequencer.state["y"] = y0
        sequencer.state["STORED"] = [[stored_x0, stored_y0]]
        sequencer.state["__XC__"] = xc
        sequencer.state["__YC__"] = yc

        g = sequencer.run()
        assert isinstance(g, GeomGroup)
        assert len(g.group) == 1
        bb = g.bounding_box()
        assert bb.cx() == pytest.approx(x0 + xc)
        assert bb.cy() == pytest.approx(y0 + yc)
        assert sequencer.state["x"] == pytest.approx(x0 + xc)
        assert sequencer.state["y"] == pytest.approx(y0 + yc)
        stored = sequencer.state["STORED"]
        assert len(stored) == 1
        assert stored[0][0] == pytest.approx(stored_x0 + xc)
        assert stored[0][1] == pytest.approx(stored_y0 + yc)

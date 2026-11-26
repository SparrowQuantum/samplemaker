import math
from pathlib import Path
from unittest.mock import Mock

import pytest

import samplemaker.devices as smdev
import samplemaker.shapes as sp
from samplemaker.baselib.devices import DirectionalCoupler, FocusingGratingCoupler
from samplemaker.baselib.waveguides import BaseWaveguidePort, BaseWaveguideSequencer
from samplemaker.shapes import GeomGroup
from tests import dummy as dm
from tests.fakes import FakeGDSWriter


@pytest.fixture
def simple_netlist(
    dummy_device_list: dict[str, type[smdev.Device]],
) -> smdev.NetList:
    _ = dummy_device_list
    netlist_entries = [
        smdev.NetListEntry(
            "TESTLIB_DUMMY_CONNECTOR", 0.0, 0.0, "E", {"io": "wire1"}, {}
        ),
        smdev.NetListEntry(
            "TESTLIB_DUMMY_CONNECTOR", 20.0, 0.0, "W", {"io": "wire1"}, {}
        ),
    ]
    return smdev.NetList("simple", netlist_entries)


def test_incompatible_port_error_subclass() -> None:
    assert issubclass(smdev.IncompatiblePortError, RuntimeError)


class TestDevicePort:
    def test_initialize(self) -> None:
        port = smdev.DevicePort(1.0, 2.0, True, False)
        assert port.x0 == pytest.approx(1.0)
        assert port.y0 == pytest.approx(2.0)
        assert port.hv is True
        assert port.bf is False
        assert port.name == ""

    def test_set_name(self) -> None:
        port = smdev.DevicePort(0, 0, True, True)
        port.set_name("world")
        assert port.name == "world"

    def test_angles(self) -> None:
        east = smdev.DevicePort(0, 0, True, True)
        west = smdev.DevicePort(0, 0, True, False)
        north = smdev.DevicePort(0, 0, False, True)
        south = smdev.DevicePort(0, 0, False, False)

        assert east.angle_to_text() == "E"
        assert east.dx() == 1
        assert east.dy() == 0
        assert east.angle() == 0.0

        assert west.angle_to_text() == "W"
        assert west.dx() == -1
        assert west.dy() == 0
        assert west.angle() == pytest.approx(math.pi)

        assert north.angle_to_text() == "N"
        assert north.dx() == 0
        assert north.dy() == 1
        assert north.angle() == pytest.approx(math.pi / 2)

        assert south.angle_to_text() == "S"
        assert south.dx() == 0
        assert south.dy() == -1
        assert south.angle() == pytest.approx(3 * math.pi / 2)

    def test_set_angle(self) -> None:
        port = smdev.DevicePort(0, 0, True, True)
        port.set_angle(math.pi / 2)
        assert port.angle_to_text() == "N"
        assert port.dx() == 0
        assert port.dy() == 1

    @pytest.mark.parametrize("angle", [0, math.pi / 2, math.pi, 3 * math.pi / 2])
    def test_set_angle_and_get_angle_consistent(self, angle: float) -> None:
        port = smdev.DevicePort(0, 0, True, True)
        port.set_angle(angle)  # set_angle expects degrees
        assert port.angle() == pytest.approx(angle)  # angle() returns radians

    def test_rotate_and_reset(self) -> None:
        port = smdev.DevicePort(1.0, 1.0, True, True)

        port.rotate(0.0, 0.0, 90.0)
        assert port.x0 == pytest.approx(-1.0)
        assert port.y0 == pytest.approx(1.0)
        assert port.angle_to_text() == "N"

        port.reset()
        assert port.x0 == pytest.approx(1.0)
        assert port.y0 == pytest.approx(1.0)
        assert port.angle_to_text() == "E"

    def test_move_commands(self) -> None:
        port = smdev.DevicePort(0.0, 0.0, True, True)
        port.bend_left(5.0)
        assert port.x0 == pytest.approx(5.0)
        assert port.y0 == pytest.approx(5.0)
        assert port.angle_to_text() == "N"

        port.bend_left(3.0)
        assert port.x0 == pytest.approx(2.0)
        assert port.y0 == pytest.approx(8.0)
        assert port.angle_to_text() == "W"

        port.move_straight(6.0)
        assert port.x0 == pytest.approx(-4.0)
        assert port.y0 == pytest.approx(8.0)
        assert port.angle_to_text() == "W"

        port.bend_right(2.0)
        assert port.x0 == pytest.approx(-6.0)
        assert port.y0 == pytest.approx(10.0)
        assert port.angle_to_text() == "N"

    def test_fix_updates_reset_anchor(self) -> None:
        port = smdev.DevicePort(0.0, 0.0, True, True)
        port.move_straight(5.0)
        port.fix()
        port.move_straight(2.0)
        assert port.x0 == pytest.approx(7.0)
        assert port.y0 == pytest.approx(0.0)

        port.reset()
        assert port.x0 == pytest.approx(5.0)
        assert port.y0 == pytest.approx(0.0)

    def test_dist(self) -> None:
        p1 = smdev.DevicePort(0.0, 0.0, True, True)
        p2 = smdev.DevicePort(3.0, 4.0, True, True)
        assert p1.dist(p2) == pytest.approx(5.0)


class TestDevice:
    def test_initialize_sets_default_attributes(self) -> None:
        dev = dm.DummyDevice()
        assert dev._p == {}
        assert dev._pdescr == {}
        assert dev._ptype == {}
        assert dev._prange == {}
        assert dev._localp == {}
        assert dev._x0 == 0
        assert dev._y0 == 0
        assert dev._hv is True
        assert dev._bf is True
        assert dev._ports == {}
        assert dev._localports == {}
        assert dev._name == ""
        assert dev._description == "No description yet"
        assert dev.use_references is True

    def test_build_initializes_device(self) -> None:
        dev = dm.DummyDevice.build()

        assert dev._name == "TESTLIB_DUMMY"
        assert dev._description == "A dummy device for testing purposes."
        assert set(dev._p.keys()) == {"width", "height"}

    def test_name_and_params_affect_hash(self, dummy_device: smdev.Device) -> None:
        h1 = hash(dummy_device)
        dummy_device.set_param("width", 20.0)
        h2 = hash(dummy_device)
        dummy_device.set_name("TESTLIB_DUMMY_MODIFIED")
        h3 = hash(dummy_device)
        assert h1 != h2 != h3

    def test_sequencer_options_affect_hash(self, dummy_device: smdev.Device) -> None:
        h1 = hash(dummy_device)
        dummy_device._seq = BaseWaveguideSequencer([])
        h2 = hash(dummy_device)
        dummy_device._seq.options["defaultWidth"] += 1.0
        h3 = hash(dummy_device)
        assert h1 != h2 != h3

    @pytest.mark.parametrize(
        ("angle", "expected_hv", "expected_bf"),
        [
            (0, True, True),
            (math.pi / 2, False, True),
            (math.pi, True, False),
            (3 * math.pi / 2, False, False),
        ],
    )
    def test_get_set_angle(
        self,
        angle: float,
        expected_hv: bool,
        expected_bf: bool,
        dummy_device: smdev.Device,
    ) -> None:
        dummy_device.set_angle(angle)
        dummy_device._hv = expected_hv
        dummy_device._bf = expected_bf
        assert dummy_device.angle() == pytest.approx(angle)

    def test_set_position(self, dummy_device: smdev.Device) -> None:
        dummy_device.set_position(10.0, 20.0)
        assert dummy_device._x0 == pytest.approx(10.0)
        assert dummy_device._y0 == pytest.approx(20.0)

        g = dummy_device.run()
        bb = g.bounding_box()
        assert bb.llx == pytest.approx(10.0)
        assert bb.cy == pytest.approx(20.0)

    def test_addparameter_rejects_colon(self, dummy_device: smdev.Device) -> None:
        with pytest.raises(ValueError, match="containing ':'"):
            dummy_device.addparameter("bad:name", 1.0, "invalid")

    def test_addlocalparameter_rejects_colon(self, dummy_device: smdev.Device) -> None:
        with pytest.raises(ValueError, match="containing ':'"):
            dummy_device.addlocalparameter("bad:name", 1.0, "invalid")

    def test_set_param_unknown_raises(self, dummy_device: smdev.Device) -> None:
        with pytest.raises(ValueError, match="Could not set parameter"):
            dummy_device.set_param("does_not_exist", 1)

    def test_set_param_does_not_traverse_device_parameters(
        self, dummy_device: smdev.Device
    ) -> None:
        dummy_device.addparameter("options", {"width": 1}, "Options")

        with pytest.raises(ValueError, match="Could not set parameter"):
            dummy_device.set_param("options::width", 2)

        assert dummy_device._p["options"] == {"width": 1}

    def test_get_params_casts_and_clips(self, dummy_device: smdev.Device) -> None:
        dummy_device.set_param("width", "1234")
        dummy_device.set_param("height", -23)
        p = dummy_device.get_params(cast_types=True, clip_in_range=True)
        assert isinstance(p["width"], float)
        assert p["width"] == pytest.approx(1000.0)
        assert p["height"] == pytest.approx(1.0)

        dummy_device.set_param("width", 5)
        p = dummy_device.get_params(cast_types=True, clip_in_range=True)
        assert p["width"] == pytest.approx(5.0)
        assert isinstance(p["width"], float)

        dummy_device.set_param("height", "not a number")
        with pytest.raises(ValueError, match="could not convert"):
            p = dummy_device.get_params(cast_types=True, clip_in_range=True)

    def test_get_localport_raises_for_missing_port(
        self, dummy_device: smdev.Device
    ) -> None:
        with pytest.raises(ValueError, match="Could not find port"):
            dummy_device.get_localport("missing")

    def test_remove_localport(self, dummy_device: smdev.Device) -> None:
        port = BaseWaveguidePort(0, 0, name="test")
        dummy_device.addlocalport(port)
        assert "test" in dummy_device._localports
        dummy_device.remove_localport("test")
        assert "test" not in dummy_device._localports

    def test_remove_localport_unknown_raises(self, dummy_device: smdev.Device) -> None:
        with pytest.raises(ValueError, match="Could not find port"):
            dummy_device.remove_localport("missing")

    def test_get_port_raises_for_missing_port(self, dummy_device: smdev.Device) -> None:
        with pytest.raises(ValueError, match="Could not find port"):
            dummy_device.get_port("missing")

    def test_build_registered_returns_named_device(
        self, dummy_device_list: dict[str, type[smdev.Device]]
    ) -> None:
        _ = dummy_device_list
        dev = smdev.Device.build_registered("TESTLIB_DUMMY")
        assert isinstance(dev, dm.DummyDevice)

    def test_build_registered_unknown_name_raises(
        self, default_device_list: dict[str, type[smdev.Device]]
    ) -> None:
        _ = default_device_list
        with pytest.raises(ValueError, match="No device named"):
            smdev.Device.build_registered("UNKNOWN")

    def test_run_use_reference(self, dummy_device: smdev.Device) -> None:
        g = dummy_device.run()
        assert "in" in dummy_device._ports

        lport = dummy_device.get_localport("in")
        port = dummy_device._ports["in"]
        assert isinstance(lport, BaseWaveguidePort)
        assert isinstance(port, BaseWaveguidePort)
        assert lport is not port  # deepcopy
        assert lport.x0 == port.x0
        assert lport.y0 == port.y0
        assert lport.hv == port.hv
        assert lport.bf == port.bf
        assert lport.width == port.width
        assert lport.name == port.name

        assert len(g.group) == 1
        assert isinstance(g.group[0], sp.SRef)

    def test_run_no_reference(self, dummy_device: smdev.Device) -> None:
        dummy_device.use_references = False
        g = dummy_device.run()
        assert "in" in dummy_device._ports

        lport = dummy_device.get_localport("in")
        port = dummy_device._ports["in"]
        assert isinstance(lport, BaseWaveguidePort)
        assert isinstance(port, BaseWaveguidePort)
        assert lport is not port  # deepcopy
        assert lport.x0 == port.x0
        assert lport.y0 == port.y0
        assert lport.hv == port.hv
        assert lport.bf == port.bf
        assert lport.width == port.width
        assert lport.name == port.name

        assert len(g.group) == 1
        assert isinstance(g.group[0], sp.Poly)


class TestNetListEntry:
    def test_rotation_mapping(self) -> None:
        assert smdev.NetListEntry("A", 0, 0, "E", {}, {}).rot == 0
        assert smdev.NetListEntry("A", 0, 0, "N", {}, {}).rot == 90
        assert smdev.NetListEntry("A", 0, 0, "W", {}, {}).rot == 180
        assert smdev.NetListEntry("A", 0, 0, "S", {}, {}).rot == 270

    def test_hash_changes_with_parameters(self) -> None:
        e1 = smdev.NetListEntry("A", 0, 0, "E", {"p": "w"}, {"x": 1})
        e2 = smdev.NetListEntry("A", 0, 0, "E", {"p": "w"}, {"x": 2})
        assert hash(e1) != hash(e2)


class TestNetList:
    def test_setters_update_internal_state(self) -> None:
        netlist = smdev.NetList("n", [])
        netlist.set_external_ports(["wire_out"])
        netlist.set_aligned_ports(["wire_aligned"])
        netlist.set_path("wire_out", [0, 0, 0, 10, 0, 0])

        assert netlist.external_ports == ["wire_out"]
        assert netlist.aligned_ports == ["wire_aligned"]
        assert netlist.paths["wire_out"] == [0, 0, 0, 10, 0, 0]

    def test_import_circuit_parses_subcircuits(self, tmp_path: Path) -> None:
        circuit_file = tmp_path / "test_circuit.txt"
        lines = [
            ".CIRCUIT CHILD out",
            "TESTLIB_DUMMY_CONNECTOR 0 0 E io out .",
            ".END",
            ".CIRCUIT TOP ext",
            "X CHILD 10 0 E io ext .",
            ".END",
        ]
        circuit_file.write_text("\n".join(lines))
        all_circuits = smdev.NetList.import_circuit(str(circuit_file))
        assert isinstance(all_circuits, dict)
        assert set(all_circuits.keys()) == {"CHILD", "TOP"}
        top = all_circuits["TOP"]
        assert top.entry_list[0].devname == "X"
        assert top.entry_list[0].params["NETLIST"].name == "CHILD"

    def test_import_circuit_returns_named_circuit(self, tmp_path: Path) -> None:
        circuit_file = tmp_path / "single_circuit.txt"
        lines = [
            ".CIRCUIT MAIN wire",
            "TESTLIB_DUMMY_CONNECTOR 0 0 E io wire .",
            ".END",
        ]
        circuit_file.write_text("\n".join(lines))
        netlist = smdev.NetList.import_circuit(str(circuit_file), "MAIN")

        assert isinstance(netlist, smdev.NetList)
        assert netlist.name == "MAIN"

    def test_import_circuit_parses_align_and_path_directives(
        self, tmp_path: Path
    ) -> None:
        circuit_file = tmp_path / "align_path_circuit.txt"
        lines = [
            ".CIRCUIT MAIN ext",
            ".ALIGN wire1 wire2",
            ".PATH wire1 0 0 E 5 5 N 10 5 W",
            "TESTLIB_DUMMY_CONNECTOR 0 0 E io wire1 .",
            ".END",
        ]
        circuit_file.write_text("\n".join(lines))
        netlist = smdev.NetList.import_circuit(str(circuit_file), "MAIN")
        assert isinstance(netlist, smdev.NetList)
        assert netlist.aligned_ports == ["wire1", "wire2"]
        assert netlist.paths["wire1"] == [0.0, 0.0, 0, 5.0, 5.0, 90, 10.0, 5.0, 180]
        assert len(netlist.entry_list) == 1


class TestCircuit:
    def test_build_device_geometry_instantiates_before_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dummy_device_list: dict[str, type[smdev.Device]],
    ) -> None:
        _ = dummy_device_list
        entry = smdev.NetListEntry(
            "TESTLIB_DUMMY_CONNECTOR",
            12.0,
            34.0,
            "N",
            {"io": "wire"},
            {"length": 25.0},
        )
        circuit = smdev.Circuit.build()
        circuit.use_references = False
        circuit.set_netlist(smdev.NetList("single", [entry]))

        original_run = dm.DummyConnectorDevice.run

        def assert_initialized_before_run(device: smdev.Device) -> GeomGroup:
            assert device._name == "TESTLIB_DUMMY_CONNECTOR"
            assert device._p == {"length": 25.0}
            assert device._x0 == pytest.approx(12.0)
            assert device._y0 == pytest.approx(34.0)
            assert device.angle() == pytest.approx(math.pi / 2)
            assert device.use_references is False
            return original_run(device)

        monkeypatch.setattr(
            dm.DummyConnectorDevice, "run", assert_initialized_before_run
        )

        geom, ports = circuit._build_device_geometry(entry, 1)

        assert isinstance(geom, GeomGroup)
        assert set(ports) == {"io"}

    def test_set_param_netlist_updates_hierarchical_parameters(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        simple_netlist: smdev.NetList,
    ) -> None:
        _ = dummy_device_list
        circuit = smdev.Circuit.build()
        circuit.set_netlist(simple_netlist)

        assert "dev_TESTLIB_DUMMY_CONNECTOR_1" in circuit._p
        assert "dev_TESTLIB_DUMMY_CONNECTOR_2" in circuit._p

    def test_set_param_updates_nested_circuit_parameters(
        self, dummy_device_list: dict[str, type[smdev.Device]]
    ) -> None:
        child_entry = smdev.NetListEntry(
            "TESTLIB_DUMMY_CONNECTOR", 0, 0, "E", {}, {"length": 10.0}
        )
        child = smdev.NetList("child", [child_entry])
        outer_entry = smdev.NetListEntry("X", 0, 0, "E", {}, {"NETLIST": child})
        circuit = smdev.Circuit.build()
        circuit.set_netlist(smdev.NetList("outer", [outer_entry]))

        circuit.set_param("dev_X_1::dev_TESTLIB_DUMMY_CONNECTOR_1::length", 25.0)

        assert circuit._p["dev_X_1"]["dev_TESTLIB_DUMMY_CONNECTOR_1"]["length"] == 25.0

    def test_set_param_updates_rendered_device(
        self, simple_netlist: smdev.NetList
    ) -> None:
        circuit = smdev.Circuit.build()
        circuit.use_references = False
        circuit.set_netlist(simple_netlist)
        circuit.set_param("dev_TESTLIB_DUMMY_CONNECTOR_1::length", 25.0)

        geom, _ = circuit._build_device_geometry(simple_netlist.entry_list[0], 1)

        assert geom.bounding_box().width == pytest.approx(25.0)
        assert circuit._p["dev_TESTLIB_DUMMY_CONNECTOR_2"]["length"] == 10.0

    @pytest.mark.parametrize(
        "name",
        [
            "missing::length",
            "dev_TESTLIB_DUMMY_CONNECTOR_1::missing",
            "dev_TESTLIB_DUMMY_CONNECTOR_1::length::extra",
        ],
    )
    def test_set_param_rejects_invalid_hierarchical_paths(
        self, simple_netlist: smdev.NetList, name: str
    ) -> None:
        circuit = smdev.Circuit.build()
        circuit.set_netlist(simple_netlist)

        with pytest.raises(ValueError, match="Could not set parameter"):
            circuit.set_param(name, 25.0)

    def test_run_connects_two_compatible_ports(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        simple_netlist: smdev.NetList,
    ) -> None:
        _ = dummy_device_list
        circuit = smdev.Circuit.build()
        circuit.set_netlist(simple_netlist)

        g = circuit.run()
        assert isinstance(g, GeomGroup)

    @pytest.mark.parametrize("use_references", [False, True])
    def test_nested_circuits_connect_and_place_geometry(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        monkeypatch: pytest.MonkeyPatch,
        use_references: bool,
    ) -> None:
        _ = dummy_device_list
        connections: list[tuple[float, float, float, float]] = []

        def connector(port1: smdev.DevicePort, port2: smdev.DevicePort) -> GeomGroup:
            connections.append((port1.x0, port1.y0, port2.x0, port2.y0))
            poly = sp.Box(
                min(port1.x0, port2.x0),
                min(port1.y0, port2.y0),
                abs(port1.x0 - port2.x0),
                abs(port1.y0 - port2.y0),
            ).to_poly()
            poly.layer = 9
            group = GeomGroup()
            group.add(poly)
            return group

        monkeypatch.setattr(dm, "_dummy_connector", connector)
        child_entries = [
            smdev.NetListEntry(
                devname="TESTLIB_TWO_PORT",
                x0=0,
                y0=0,
                rot="E",
                portmap={"p1": "in", "p2": "out"},
                params={},
            )
        ]
        child = smdev.NetList("child", child_entries)
        child.set_external_ports(["in", "out"])
        parent_entries = [
            smdev.NetListEntry(
                devname="X",
                x0=0,
                y0=0,
                rot="E",
                portmap={"in": "input", "out": "link"},
                params={"NETLIST": child},
            ),
            smdev.NetListEntry(
                devname="X",
                x0=30,
                y0=0,
                rot="E",
                portmap={"in": "link", "out": "output"},
                params={"NETLIST": child},
            ),
        ]
        parent = smdev.NetList("parent", parent_entries)
        parent.set_external_ports(["input", "output"])
        circuit = smdev.Circuit.build()
        circuit.use_references = use_references
        circuit.set_param("NETLIST", parent)
        circuit.set_param("dev_X_2::dev_TESTLIB_TWO_PORT_1::length", 4)

        geometry = circuit.run().flatten()

        assert connections == [(10, 10, 30, 0)]
        assert set(circuit._ports) == {"input", "output"}
        assert circuit.get_port("input").x0 == 0
        assert circuit.get_port("input").y0 == 0
        assert circuit.get_port("output").x0 == 34
        assert circuit.get_port("output").y0 == 4
        assert len(geometry.group) == 3
        boxes = [
            (shape.layer, box.llx, box.lly, box.width, box.height)
            for shape in geometry.group
            for box in [shape.bounding_box()]
        ]
        for box, expected in zip(
            boxes,
            [(1, 0, 0, 10, 10), (1, 30, 0, 4, 4), (9, 10, 0, 20, 10)],
            strict=True,
        ):
            assert box == pytest.approx(expected)

    @pytest.mark.parametrize("modified_instance", [1, 2])
    def test_tutorial_nested_circuits_use_independent_parameters(
        self,
        default_device_list: dict[str, type[smdev.Device]],
        modified_instance: int,
    ) -> None:
        default_device_list["BASELIB_DCPL"] = DirectionalCoupler
        default_device_list["BASELIB_FGC"] = FocusingGratingCoupler
        child = smdev.NetList(
            "my_circuit",
            [
                smdev.NetListEntry(
                    "BASELIB_DCPL",
                    0,
                    0,
                    "E",
                    {"p1": "inA", "p4": "inB", "p3": "AAA", "p2": "BBB"},
                    {},
                ),
                smdev.NetListEntry("BASELIB_FGC", -30, 0, "W", {"p1": "inA"}, {}),
                smdev.NetListEntry("BASELIB_FGC", 30, -15, "S", {"p1": "inB"}, {}),
            ],
        )
        child.set_aligned_ports(["inA"])
        child.set_external_ports(["AAA", "BBB"])
        parent = smdev.NetList(
            "bigger",
            [
                smdev.NetListEntry(
                    "X", 0, 0, "E", {"AAA": "input", "BBB": "link"}, {"NETLIST": child}
                ),
                smdev.NetListEntry(
                    "X",
                    40,
                    12,
                    "E",
                    {"AAA": "link", "BBB": "output"},
                    {"NETLIST": child},
                ),
            ],
        )
        parent.set_external_ports(["input", "output"])
        circuit = smdev.Circuit.build()
        circuit.set_param("NETLIST", parent)
        circuit.set_param(f"dev_X_{modified_instance}::dev_BASELIB_DCPL_1::gap", 0.1)

        result = circuit.run()

        assert len(result.group) == 1
        parent_ref = result.group[0]
        assert isinstance(parent_ref, sp.SRef)
        first, second, *connector = parent_ref.group.group
        assert isinstance(first, sp.SRef)
        assert isinstance(second, sp.SRef)
        assert (first.x0, first.y0, second.x0, second.y0) == (0, 0, 40, 12)
        assert connector
        assert all(isinstance(shape, sp.Poly) for shape in connector)
        first_coupler = first.group.group[0]
        second_coupler = second.group.group[0]
        assert isinstance(first_coupler, sp.SRef)
        assert isinstance(second_coupler, sp.SRef)
        expected_heights = (5.7, 6.1) if modified_instance == 1 else (6.1, 5.7)
        assert (
            first_coupler.bounding_box().height,
            second_coupler.bounding_box().height,
        ) == pytest.approx(expected_heights)
        assert result.get_layer_list() == {1, 3}
        assert set(circuit._ports) == {"input", "output"}
        assert circuit.get_port("input").x0 == pytest.approx(-17)
        assert circuit.get_port("output").x0 == pytest.approx(57)
        assert circuit.get_port("input").y0 == pytest.approx(
            -2.7 if modified_instance == 1 else -2.9
        )
        assert circuit.get_port("output").y0 == pytest.approx(
            14.7 if modified_instance == 2 else 14.9
        )

    def test_run_warns_for_unconnected_non_external_port(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
    ) -> None:
        _ = dummy_device_list
        netlist_entries = [
            smdev.NetListEntry(
                "TESTLIB_DUMMY_CONNECTOR", 0, 0, "E", {"io": "dangling"}, {}
            )
        ]
        netlist = smdev.NetList("warn", netlist_entries)
        circuit = smdev.Circuit.build()
        circuit.set_netlist(netlist)

        with pytest.warns(UserWarning, match="unconnected"):
            _ = circuit.run()

    def test_run_exports_external_ports(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
    ) -> None:
        _ = dummy_device_list
        netlist_entries = [
            smdev.NetListEntry(
                "TESTLIB_DUMMY_CONNECTOR", 0, 0, "E", {"io": "wire_out"}, {}
            )
        ]
        netlist = smdev.NetList("ext", netlist_entries)
        netlist.set_external_ports(["wire_out"])

        circuit = smdev.Circuit.build()
        circuit.set_netlist(netlist)
        circuit.run()

        assert "wire_out" in circuit._ports

    def test_cached_nested_circuits_restore_external_ports(
        self, dummy_device_list: dict[str, type[smdev.Device]]
    ) -> None:
        child_entries = [
            smdev.NetListEntry(
                devname="TESTLIB_TWO_PORT",
                x0=0,
                y0=0,
                rot="E",
                portmap={"p1": "in", "p2": "out"},
                params={},
            )
        ]
        child = smdev.NetList("child", child_entries)
        child.set_external_ports(["in", "out"])
        parent_entries = [
            smdev.NetListEntry(
                devname="X",
                x0=0,
                y0=0,
                rot="E",
                portmap={"in": "first_in", "out": "first_out"},
                params={"NETLIST": child},
            ),
            smdev.NetListEntry(
                devname="X",
                x0=30,
                y0=0,
                rot="E",
                portmap={"in": "second_in", "out": "second_out"},
                params={"NETLIST": child},
            ),
        ]
        parent = smdev.NetList("parent", parent_entries)
        parent.set_external_ports(["first_in", "first_out", "second_in", "second_out"])
        circuit = smdev.Circuit.build()
        circuit.set_netlist(parent)

        first_geometry = circuit.run()
        second = smdev.Circuit.build()
        second.set_netlist(parent)
        second.set_position(100, 20)
        second_geometry = second.run()

        first_ref = first_geometry.group[0]
        second_ref = second_geometry.group[0]
        assert isinstance(first_ref, sp.SRef)
        assert isinstance(second_ref, sp.SRef)
        assert first_ref.cellname == second_ref.cellname
        first_child, second_child = first_ref.group.group
        assert isinstance(first_child, sp.SRef)
        assert isinstance(second_child, sp.SRef)
        assert first_child.cellname == second_child.cellname
        assert len(first_geometry.flatten().group) == 2
        assert set(second._ports) == set(parent.external_ports)
        assert second.get_port("first_out").x0 == 110
        assert second.get_port("first_out").y0 == 30
        assert second.get_port("second_out").x0 == 140
        assert second.get_port("second_out").y0 == 30

    @pytest.mark.parametrize("use_references", [False, True])
    def test_changed_netlist_replaces_exported_ports(
        self, dummy_device_list: dict[str, type[smdev.Device]], use_references: bool
    ) -> None:
        def netlist(name: str, port: str) -> smdev.NetList:
            netlist_entries = [
                smdev.NetListEntry(
                    "TESTLIB_DUMMY_CONNECTOR", 0, 0, "E", {"io": port}, {}
                )
            ]
            result = smdev.NetList(name, netlist_entries)
            result.set_external_ports([port])
            return result

        circuit = smdev.Circuit.build()
        circuit.use_references = use_references
        first_netlist = netlist("first", "old")
        circuit.set_netlist(first_netlist)
        circuit.run()
        circuit.set_netlist(netlist("second", "new"))
        circuit.run()

        assert set(circuit._ports) == {"new"}
        assert circuit.get_localport("new").name == "new"

        original = smdev.Circuit.build()
        original.use_references = use_references
        original.set_netlist(first_netlist)
        original.run()
        assert set(original._ports) == {"old"}
        assert original.get_localport("old").name == "old"

    def test_run_aligns_ports_on_y_when_aligned_wire_exists(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ = dummy_device_list
        captured: list[tuple[float, float, float, float]] = []

        def _capture_connector(
            port1: smdev.DevicePort, port2: smdev.DevicePort
        ) -> GeomGroup:
            captured.append((port1.x0, port1.y0, port2.x0, port2.y0))
            return GeomGroup()

        monkeypatch.setattr(dm, "_dummy_connector", _capture_connector)
        netlist_entries = [
            smdev.NetListEntry(
                "TESTLIB_DUMMY_CONNECTOR", 0.0, 0.0, "E", {"io": "w1"}, {}
            ),
            smdev.NetListEntry(
                "TESTLIB_DUMMY_CONNECTOR", 20.0, 5.0, "W", {"io": "w1"}, {}
            ),
        ]
        netlist = smdev.NetList("align_y", netlist_entries)
        netlist.set_aligned_ports(["w1"])

        circuit = smdev.Circuit.build()
        circuit.set_netlist(netlist)
        circuit.run()

        assert captured
        assert captured[-1][1] == pytest.approx(captured[-1][3])

    def test_run_aligns_ports_on_x_for_vertical_ports(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ = dummy_device_list
        captured: list[tuple[float, float, float, float]] = []

        def _capture_connector(
            port1: smdev.DevicePort, port2: smdev.DevicePort
        ) -> GeomGroup:
            captured.append((port1.x0, port1.y0, port2.x0, port2.y0))
            return GeomGroup()

        monkeypatch.setattr(dm, "_dummy_connector", _capture_connector)

        netlist_entries = [
            smdev.NetListEntry(
                "TESTLIB_DUMMY_CONNECTOR", 0.0, 0.0, "N", {"io": "w1"}, {}
            ),
            smdev.NetListEntry(
                "TESTLIB_DUMMY_CONNECTOR", 5.0, 20.0, "S", {"io": "w1"}, {}
            ),
        ]
        netlist = smdev.NetList("align_x", netlist_entries)
        netlist.set_aligned_ports(["w1"])

        circuit = smdev.Circuit.build()
        circuit.set_netlist(netlist)
        circuit.run()

        assert captured
        assert captured[-1][0] == pytest.approx(captured[-1][2])

    def test_run_routes_connector_through_virtual_path_points(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ = dummy_device_list
        captured: list[tuple[float, float, float, float]] = []

        def _capture_connector(
            port1: smdev.DevicePort, port2: smdev.DevicePort
        ) -> GeomGroup:
            captured.append((port1.x0, port1.y0, port2.x0, port2.y0))
            return GeomGroup()

        monkeypatch.setattr(dm, "_dummy_connector", _capture_connector)

        netlist_entries = [
            smdev.NetListEntry(
                "TESTLIB_DUMMY_CONNECTOR", 0.0, 0.0, "E", {"io": "w1"}, {}
            ),
            smdev.NetListEntry(
                "TESTLIB_DUMMY_CONNECTOR", 20.0, 0.0, "W", {"io": "w1"}, {}
            ),
        ]
        netlist = smdev.NetList("path_route", netlist_entries)
        netlist.set_path("w1", [10.0, 0.0, 0.0])

        circuit = smdev.Circuit.build()
        circuit.set_netlist(netlist)
        _ = circuit.run()

        # One call for the virtual point, one final call to the destination port.
        assert len(captured) == 2
        assert captured[0][2] == pytest.approx(10.0)
        assert captured[0][3] == pytest.approx(0.0)
        assert captured[1][0] == pytest.approx(10.0)
        assert captured[1][1] == pytest.approx(0.0)

    def test_run_routes_connector_through_multiple_virtual_path_points(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ = dummy_device_list
        captured: list[tuple[float, float, float, float]] = []

        def _capture_connector(
            port1: smdev.DevicePort, port2: smdev.DevicePort
        ) -> GeomGroup:
            captured.append((port1.x0, port1.y0, port2.x0, port2.y0))
            return GeomGroup()

        monkeypatch.setattr(dm, "_dummy_connector", _capture_connector)

        netlist_entries = [
            smdev.NetListEntry(
                "TESTLIB_DUMMY_CONNECTOR", 0.0, 0.0, "E", {"io": "w1"}, {}
            ),
            smdev.NetListEntry(
                "TESTLIB_DUMMY_CONNECTOR", 20.0, 0.0, "W", {"io": "w1"}, {}
            ),
        ]
        netlist = smdev.NetList("multi_path_route", netlist_entries)
        netlist.set_path("w1", [10.0, 0.0, 0.0, 10.0, 5.0, 90])

        circuit = smdev.Circuit.build()
        circuit.set_netlist(netlist)
        _ = circuit.run()

        assert len(captured) == 3
        assert captured[0][2] == pytest.approx(10.0)
        assert captured[0][3] == pytest.approx(0.0)
        assert captured[1][0] == pytest.approx(10.0)
        assert captured[1][1] == pytest.approx(0.0)
        assert captured[1][2] == pytest.approx(10.0)
        assert captured[1][3] == pytest.approx(5.0)
        assert captured[2][0] == pytest.approx(10.0)
        assert captured[2][1] == pytest.approx(5.0)

    def test_run_resets_sequencer(
        self,
        simple_netlist: smdev.NetList,
        dummy_device_list: dict[str, type[smdev.Device]],
    ) -> None:
        captured: list[bool] = []

        def _capture_reset() -> None:
            captured.append(True)

        seq = BaseWaveguideSequencer([])
        seq.reset = _capture_reset
        dev = dummy_device_list["TESTLIB_DUMMY_CONNECTOR"]
        dev._seq = seq

        circuit = smdev.Circuit.build()
        circuit.set_netlist(simple_netlist)
        circuit.run()

        # Two devices in the netlist, so should be reset twice
        assert captured == [True, True]

    def test_set_netlist_parameter_updates_netlist(
        self,
        monkeypatch: pytest.MonkeyPatch,
        simple_netlist: smdev.NetList,
        dummy_device_list: dict[str, type[smdev.Device]],
    ) -> None:
        circuit = smdev.Circuit.build()
        update_parameters = Mock()
        monkeypatch.setattr(smdev.Circuit, "_update_parameters", update_parameters)

        circuit.set_param("NETLIST", simple_netlist)
        assert circuit._netlist == simple_netlist
        assert "NETLIST" not in circuit._p
        update_parameters.assert_called_once()


class TestDeviceLibraryExports:
    def test_create_device_library(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ = dummy_device_list
        output_file = tmp_path / "library.gds"

        created_writers: list[FakeGDSWriter] = []

        def _writer_factory() -> FakeGDSWriter:
            writer = FakeGDSWriter()
            created_writers.append(writer)
            return writer

        monkeypatch.setattr(smdev, "GDSWriter", _writer_factory)

        smdev.create_device_library(
            "TESTLIB_DUMMY_CONNECTOR", {"length": 7.0}, str(output_file)
        )

        writer = created_writers[0]
        geom = writer.calls[1][2]

        assert writer.calls[0] == ("open_library", str(output_file))
        assert writer.calls[1][0] == "write_structure"
        assert writer.calls[1][1] == "TESTLIB_DUMMY_CONNECTOR"
        assert writer.calls[2] == ("close_library", None)
        assert isinstance(geom, GeomGroup)
        assert isinstance(geom.group[-1], sp.Text)

        txt = geom.group[-1]
        assert txt.x0 == pytest.approx(0)
        assert txt.y0 == pytest.approx(0)
        assert "__PORT__ io E" in txt.text
        assert "DummyConnectorPort" in txt.text

    def test_export_device_schematics(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        tmp_path: Path,
    ) -> None:
        _ = dummy_device_list
        output_file = tmp_path / "SampleMakerLibrary.lel"

        smdev.export_device_schematics(str(output_file))
        content = output_file.read_text()

        assert "<Component DummyDevice>" in content
        assert "<Component DummyConnectorDevice>" in content
        assert "<Description>" in content
        assert "<Parameter>" in content
        assert "<Prefix TESTLIB_DUMMY>" in content
        assert "<Prefix TESTLIB_DUMMY_CONNECTOR>" in content
        assert "<Netlist spice>" in content
        assert "<Prefix X>" not in content


class TestDeviceRegistration:
    def test_register_devices_in_module(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
    ) -> None:
        _ = dummy_device_list
        smdev.register_devices_in_module(__name__)
        assert "TESTLIB_DUMMY" in smdev._DeviceList
        assert "TESTLIB_DUMMY_CONNECTOR" in smdev._DeviceList

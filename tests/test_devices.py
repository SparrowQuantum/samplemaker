from collections.abc import Generator
import math
from pathlib import Path

from samplemaker.baselib.waveguides import BaseWaveguidePort, BaseWaveguideSequencer
import samplemaker.devices as smdev
import samplemaker.shapes as sp
import pytest
from fixtures import reset_samplemaker  # noqa: F401

from samplemaker.makers import make_rect
from samplemaker.shapes import GeomGroup


class DummyDevice(smdev.Device):
    """Simple test device.

    Consists of a rectangle with a single port on the left side.
    The width and height of the rectangle can be set via parameters.
    """

    def initialize(self):
        self.set_name("TESTLIB_DUMMY")
        self.set_description("A dummy device for testing purposes.")

    def parameters(self):
        self.addparameter(
            param_name="width",
            default_value=10.0,
            param_description="Width of the sample.",
            param_type=float,
            param_range=(1.0, 1000.0),
        )
        self.addparameter(
            param_name="height",
            default_value=5.0,
            param_description="Height of the sample.",
            param_type=float,
            param_range=(1.0, 1000.0),
        )

    def geom(self):
        p = self.get_params()
        width = p["width"]
        height = p["height"]

        rectangle = make_rect(0, 0, width, height, numkey=4)
        port = BaseWaveguidePort(0, 0, orient="West", name="in")
        self.addlocalport(port)

        return rectangle


def _dummy_connector(_port1: smdev.DevicePort, _port2: smdev.DevicePort) -> GeomGroup:
    return GeomGroup()


class ConnectorPort(smdev.DevicePort):
    def __init__(
        self, x0: float, y0: float, horizontal: bool, forward: bool, name: str
    ):
        super().__init__(x0, y0, horizontal, forward)
        self.name = name
        self.connector_function = _dummy_connector


class ConnectorDevice(smdev.Device):
    def initialize(self):
        self.set_name("TESTLIB_CONNECTOR")
        self.set_description("Simple connector-compatible test device")

    def parameters(self):
        self.addparameter(
            param_name="length",
            default_value=10.0,
            param_description="Length of test geometry",
            param_type=float,
        )

    def geom(self):
        p = self.get_params()
        rect = make_rect(0, 0, p["length"], 1, numkey=4)
        self.addlocalport(ConnectorPort(0, 0, True, True, name="io"))
        return rect


@pytest.fixture
def default_device_list() -> Generator[dict[str, type[smdev.Device]], None, None]:
    old_devlist = smdev._DeviceList.copy()
    yield smdev._DeviceList
    smdev._DeviceList.clear()
    smdev._DeviceList.update(old_devlist)


@pytest.fixture
def device_list(
    default_device_list: dict[str, type[smdev.Device]],
) -> Generator[dict[str, type[smdev.Device]], None, None]:
    dev = DummyDevice()
    dev.initialize()
    default_device_list[dev._name] = DummyDevice
    yield default_device_list


@pytest.fixture
def connector_device_list(
    default_device_list: dict[str, type[smdev.Device]],
) -> Generator[dict[str, type[smdev.Device]], None, None]:
    dev = ConnectorDevice()
    dev.initialize()
    default_device_list[dev._name] = ConnectorDevice
    yield default_device_list


@pytest.fixture
def simple_netlist(
    connector_device_list: dict[str, type[smdev.Device]],
) -> smdev.NetList:
    _ = connector_device_list
    return smdev.NetList(
        "simple",
        [
            smdev.NetListEntry("TESTLIB_CONNECTOR", 0.0, 0.0, "E", {"io": "wire1"}, {}),
            smdev.NetListEntry(
                "TESTLIB_CONNECTOR", 20.0, 0.0, "W", {"io": "wire1"}, {}
            ),
        ],
    )


def test_incompatible_port_error_subclass():
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

    @pytest.mark.xfail(
        strict=True,
        reason="Radian/degree inconsistency between angle() and set_angle()",
    )
    def test_set_angle_and_get_angle_consistent() -> None:
        port = smdev.DevicePort(0, 0, True, True)
        port.set_angle(math.pi / 4)  # set_angle expects degrees
        assert port.angle() == pytest.approx(math.pi / 4)  # angle() returns radians

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
        port.BL(5.0)
        assert port.x0 == pytest.approx(5.0)
        assert port.y0 == pytest.approx(5.0)
        assert port.angle_to_text() == "N"

        port.BL(3.0)
        assert port.x0 == pytest.approx(2.0)
        assert port.y0 == pytest.approx(8.0)
        assert port.angle_to_text() == "W"

        port.S(6.0)
        assert port.x0 == pytest.approx(-4.0)
        assert port.y0 == pytest.approx(8.0)
        assert port.angle_to_text() == "W"

        port.BR(2.0)
        assert port.x0 == pytest.approx(-6.0)
        assert port.y0 == pytest.approx(10.0)
        assert port.angle_to_text() == "N"

    def test_fix_updates_reset_anchor(self) -> None:
        port = smdev.DevicePort(0.0, 0.0, True, True)
        port.S(5.0)
        port.fix()
        port.S(2.0)
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
        dev = DummyDevice()
        assert dev._p == {}
        assert dev._pdescr == {"_ports_": "Ports calculated by geom"}
        assert "_ports_" in dev._ptype  # Is really float...
        assert "_ports_" in dev._prange  # Range not applicable to ports.
        assert dev._localp == {"_ports_": {}}
        assert dev._x0 == 0
        assert dev._y0 == 0
        assert dev._hv is True
        assert dev._bf is True
        assert dev._ports == {}
        assert dev._name == ""
        assert dev._description == "No description yet"
        assert dev.use_references is True

    def test_build_initializes_device(
        self, device_list: dict[str, type[smdev.Device]]
    ) -> None:
        _ = device_list
        dev = DummyDevice.build()

        assert dev._name == "TESTLIB_DUMMY"
        assert dev._description == "A dummy device for testing purposes."
        assert set(dev._p.keys()) == {"width", "height"}

    def test_name_and_params_affect_hash(self) -> None:
        dev = DummyDevice.build()
        h1 = hash(dev)
        dev.set_param("width", 20.0)
        h2 = hash(dev)
        dev.set_name("TESTLIB_DUMMY_MODIFIED")
        h3 = hash(dev)
        assert h1 != h2 != h3

    def test_sequencer_options_affect_hash(self) -> None:
        dev = DummyDevice.build()
        h1 = hash(dev)
        dev._seq = BaseWaveguideSequencer([])
        h2 = hash(dev)
        dev._seq.options["defaultWidth"] += 1.0
        h3 = hash(dev)
        assert h1 != h2 != h3

    @pytest.mark.xfail(
        strict=True,
        reason="set_angle() currently accepts radians while angle() "
        "returns degrees, causing inconsistency.",
    )
    def test_get_set_angle(self) -> None:
        dev = DummyDevice.build()
        dev.set_angle(90)
        assert dev._hv is False
        assert dev._bf is True
        assert dev.angle() == pytest.approx(90.0)

    def test_set_position(self) -> None:
        dev = DummyDevice.build()
        dev.set_position(10.0, 20.0)
        assert dev._x0 == pytest.approx(10.0)
        assert dev._y0 == pytest.approx(20.0)

        g = dev.run()
        bb = g.bounding_box()
        assert bb.llx == pytest.approx(10.0)
        assert bb.cy() == pytest.approx(20.0)


    def test_addparameter_rejects_colon(self) -> None:
        dev = DummyDevice.build()
        with pytest.raises(ValueError, match="containing ':'"):
            dev.addparameter("bad:name", 1.0, "invalid")

    def test_addlocalparameter_rejects_colon(self) -> None:
        dev = DummyDevice.build()
        with pytest.raises(ValueError, match="containing ':'"):
            dev.addlocalparameter("bad:name", 1.0, "invalid")

    def test_set_param_unknown_raises(self) -> None:
        dev = DummyDevice.build()
        with pytest.raises(ValueError, match="Could not set parameter"):
            dev.set_param("does_not_exist", 1)

    def test_get_params_casts_and_clips(self) -> None:
        dev = DummyDevice.build()
        dev.set_param("width", "1234")
        dev.set_param("height", -23)
        p = dev.get_params(cast_types=True, clip_in_range=True)
        assert isinstance(p["width"], float)
        assert p["width"] == pytest.approx(1000.0)
        assert p["height"] == pytest.approx(1.0)

        dev.set_param("width", 5)
        p = dev.get_params(cast_types=True, clip_in_range=True)
        assert p["width"] == pytest.approx(5.0)
        assert isinstance(p["width"], float)

        dev.set_param("height", "not a number")
        with pytest.raises(ValueError, match="could not convert"):
            p = dev.get_params(cast_types=True, clip_in_range=True)

    def test_get_localport_raises_for_missing_port(self) -> None:
        dev = DummyDevice.build()
        with pytest.raises(ValueError, match="Could not find port"):
            dev.get_localport("missing")

    def test_remove_localport(self) -> None:
        dev = DummyDevice.build()
        port = BaseWaveguidePort(0, 0, name="test")
        dev.addlocalport(port)
        assert "test" in dev._localp["_ports_"]
        dev.remove_localport("test")
        assert "test" not in dev._localp["_ports_"]

    @pytest.mark.xfail(
        strict=True,
        reason="remove_localport should raise for missing port but currently does not",
    )
    def test_remove_localport_unknown_raises(self) -> None:
        dev = DummyDevice.build()
        with pytest.raises(ValueError, match="Could not find local port"):
            dev.remove_localport("missing")

    def test_get_port_raises_for_missing_port(self) -> None:
        dev = DummyDevice.build()
        with pytest.raises(ValueError, match="Could not find port"):
            dev.get_port("missing")

    def test_build_registered_returns_named_device(
        self, device_list: dict[str, type[smdev.Device]]
    ) -> None:
        _ = device_list
        dev = smdev.Device.build_registered("TESTLIB_DUMMY")
        assert isinstance(dev, DummyDevice)

    def test_build_registered_unknown_name_raises(
        self, default_device_list: dict[str, type[smdev.Device]]
    ) -> None:
        _ = default_device_list
        with pytest.raises(ValueError, match="No device named"):
            smdev.Device.build_registered("UNKNOWN")

    def test_run_use_reference(self) -> None:
        dev = DummyDevice.build()
        g = dev.run()
        assert "in" in dev._ports

        lport = dev.get_localport("in")
        port = dev._ports["in"]
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

    def test_run_no_reference(self) -> None:
        dev = DummyDevice.build()
        dev.use_references = False
        g = dev.run()
        assert "in" in dev._ports

        lport = dev.get_localport("in")
        port = dev._ports["in"]
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
        net = smdev.NetList("n", [])
        net.set_external_ports(["wire_out"])
        net.set_aligned_ports(["wire_aligned"])
        net.set_path("wire_out", [0, 0, 0, 10, 0, 0])

        assert net.external_ports == ["wire_out"]
        assert net.aligned_ports == ["wire_aligned"]
        assert net.paths["wire_out"] == [0, 0, 0, 10, 0, 0]

    def test_import_circuit_parses_subcircuits(self, tmp_path: Path) -> None:
        circuit_file = tmp_path / "test_circuit.txt"
        circuit_file.write_text(
            "\n".join(
                [
                    ".CIRCUIT CHILD out",
                    "TESTLIB_CONNECTOR 0 0 E io out .",
                    ".END",
                    ".CIRCUIT TOP ext",
                    "X CHILD 10 0 E io ext .",
                    ".END",
                ]
            )
        )

        all_circuits = smdev.NetList.ImportCircuit(str(circuit_file))

        assert set(all_circuits.keys()) == {"CHILD", "TOP"}
        top = all_circuits["TOP"]
        assert top.entry_list[0].devname == "X"
        assert top.entry_list[0].params["NETLIST"].name == "CHILD"

    def test_import_circuit_returns_named_circuit(self, tmp_path: Path) -> None:
        circuit_file = tmp_path / "single_circuit.txt"
        circuit_file.write_text(
            "\n".join(
                [
                    ".CIRCUIT MAIN wire",
                    "TESTLIB_CONNECTOR 0 0 E io wire .",
                    ".END",
                ]
            )
        )

        net = smdev.NetList.ImportCircuit(str(circuit_file), "MAIN")
        assert isinstance(net, smdev.NetList)
        assert net.name == "MAIN"


class TestCircuit:
    def test_set_param_netlist_updates_hierarchical_parameters(
        self,
        connector_device_list: dict[str, type[smdev.Device]],
        simple_netlist: smdev.NetList,
    ) -> None:
        _ = connector_device_list
        circuit = smdev.Circuit.build()
        circuit.set_param("NETLIST", simple_netlist)

        assert "dev_TESTLIB_CONNECTOR_1" in circuit._p
        assert "dev_TESTLIB_CONNECTOR_2" in circuit._p

    def test_run_connects_two_compatible_ports(
        self,
        connector_device_list: dict[str, type[smdev.Device]],
        simple_netlist: smdev.NetList,
    ) -> None:
        _ = connector_device_list
        circuit = smdev.Circuit.build()
        circuit.set_param("NETLIST", simple_netlist)

        g = circuit.run()
        assert isinstance(g, GeomGroup)

    def test_run_warns_for_unconnected_non_external_port(
        self,
        connector_device_list: dict[str, type[smdev.Device]],
    ) -> None:
        _ = connector_device_list
        net = smdev.NetList(
            "warn",
            [
                smdev.NetListEntry(
                    "TESTLIB_CONNECTOR", 0, 0, "E", {"io": "dangling"}, {}
                )
            ],
        )
        circuit = smdev.Circuit.build()
        circuit.set_param("NETLIST", net)

        with pytest.warns(UserWarning, match="unconnected"):
            _ = circuit.run()

    def test_run_exports_external_ports(
        self,
        connector_device_list: dict[str, type[smdev.Device]],
    ) -> None:
        _ = connector_device_list
        net = smdev.NetList(
            "ext",
            [
                smdev.NetListEntry(
                    "TESTLIB_CONNECTOR", 0, 0, "E", {"io": "wire_out"}, {}
                )
            ],
        )
        net.set_external_ports(["wire_out"])

        circuit = smdev.Circuit.build()
        circuit.set_param("NETLIST", net)
        _ = circuit.run()

        assert "wire_out" in circuit._ports


class TestDeviceRegistration:
    def test_register_devices_in_module(
        self,
        default_device_list: dict[str, type[smdev.Device]],
    ) -> None:
        _ = default_device_list
        smdev.registerDevicesInModule(__name__)
        assert "TESTLIB_DUMMY" in smdev._DeviceList
        assert "TESTLIB_CONNECTOR" in smdev._DeviceList

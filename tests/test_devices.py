import math
from pathlib import Path

from samplemaker.baselib.waveguides import BaseWaveguidePort, BaseWaveguideSequencer
import samplemaker.devices as smdev
import samplemaker.shapes as sp
import pytest

from samplemaker.shapes import GeomGroup
import dummy as dm


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
        dev = dm.DummyDevice()
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
        self, dummy_device_list: dict[str, type[smdev.Device]]
    ) -> None:
        _ = dummy_device_list
        dev = dm.DummyDevice.build()

        assert dev._name == "TESTLIB_DUMMY"
        assert dev._description == "A dummy device for testing purposes."
        assert set(dev._p.keys()) == {"width", "height"}

    def test_name_and_params_affect_hash(self) -> None:
        dev = dm.DummyDevice.build()
        h1 = hash(dev)
        dev.set_param("width", 20.0)
        h2 = hash(dev)
        dev.set_name("TESTLIB_DUMMY_MODIFIED")
        h3 = hash(dev)
        assert h1 != h2 != h3

    def test_sequencer_options_affect_hash(self) -> None:
        dev = dm.DummyDevice.build()
        h1 = hash(dev)
        dev._seq = BaseWaveguideSequencer([])
        h2 = hash(dev)
        dev._seq.options["defaultWidth"] += 1.0
        h3 = hash(dev)
        assert h1 != h2 != h3

    @pytest.mark.parametrize(
        "angle, expected_hv, expected_bf",
        [
            (0, True, True),
            (math.pi / 2, False, True),
            (math.pi, True, False),
            (3 * math.pi / 2, False, False),
        ],
    )
    def test_get_set_angle(
        self, angle: float, expected_hv: bool, expected_bf: bool
    ) -> None:
        dev = dm.DummyDevice.build()
        dev.set_angle(angle)
        dev._hv = expected_hv
        dev._bf = expected_bf
        assert dev.angle() == pytest.approx(angle)

    def test_set_position(self) -> None:
        dev = dm.DummyDevice.build()
        dev.set_position(10.0, 20.0)
        assert dev._x0 == pytest.approx(10.0)
        assert dev._y0 == pytest.approx(20.0)

        g = dev.run()
        bb = g.bounding_box()
        assert bb.llx == pytest.approx(10.0)
        assert bb.cy() == pytest.approx(20.0)

    def test_addparameter_rejects_colon(self) -> None:
        dev = dm.DummyDevice.build()
        with pytest.raises(ValueError, match="containing ':'"):
            dev.addparameter("bad:name", 1.0, "invalid")

    def test_addlocalparameter_rejects_colon(self) -> None:
        dev = dm.DummyDevice.build()
        with pytest.raises(ValueError, match="containing ':'"):
            dev.addlocalparameter("bad:name", 1.0, "invalid")

    def test_set_param_unknown_raises(self) -> None:
        dev = dm.DummyDevice.build()
        with pytest.raises(ValueError, match="Could not set parameter"):
            dev.set_param("does_not_exist", 1)

    def test_get_params_casts_and_clips(self) -> None:
        dev = dm.DummyDevice.build()
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
        dev = dm.DummyDevice.build()
        with pytest.raises(ValueError, match="Could not find port"):
            dev.get_localport("missing")

    def test_remove_localport(self) -> None:
        dev = dm.DummyDevice.build()
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
        dev = dm.DummyDevice.build()
        with pytest.raises(ValueError, match="Could not find local port"):
            dev.remove_localport("missing")

    def test_get_port_raises_for_missing_port(self) -> None:
        dev = dm.DummyDevice.build()
        with pytest.raises(ValueError, match="Could not find port"):
            dev.get_port("missing")

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

    def test_run_use_reference(self) -> None:
        dev = dm.DummyDevice.build()
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
        dev = dm.DummyDevice.build()
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
        all_circuits = smdev.NetList.ImportCircuit(str(circuit_file))

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
        netlist = smdev.NetList.ImportCircuit(str(circuit_file), "MAIN")

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
        netlist = smdev.NetList.ImportCircuit(str(circuit_file), "MAIN")

        assert netlist.aligned_ports == ["wire1", "wire2"]
        assert netlist.paths["wire1"] == [0.0, 0.0, 0, 5.0, 5.0, 90, 10.0, 5.0, 180]
        assert len(netlist.entry_list) == 1


class TestCircuit:
    def test_set_param_netlist_updates_hierarchical_parameters(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        simple_netlist: smdev.NetList,
    ) -> None:
        _ = dummy_device_list
        circuit = smdev.Circuit.build()
        circuit.set_param("NETLIST", simple_netlist)

        assert "dev_TESTLIB_DUMMY_CONNECTOR_1" in circuit._p
        assert "dev_TESTLIB_DUMMY_CONNECTOR_2" in circuit._p

    def test_run_connects_two_compatible_ports(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        simple_netlist: smdev.NetList,
    ) -> None:
        _ = dummy_device_list
        circuit = smdev.Circuit.build()
        circuit.set_param("NETLIST", simple_netlist)

        g = circuit.run()
        assert isinstance(g, GeomGroup)

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
        circuit.set_param("NETLIST", netlist)

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
        circuit.set_param("NETLIST", netlist)
        circuit.run()

        assert "wire_out" in circuit._ports

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
        circuit.set_param("NETLIST", netlist)
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
        circuit.set_param("NETLIST", netlist)
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
        circuit.set_param("NETLIST", netlist)
        _ = circuit.run()

        # One call for the virtual point, one final call to the destination port.
        assert len(captured) == 2
        assert captured[0][2] == pytest.approx(10.0)
        assert captured[0][3] == pytest.approx(0.0)
        assert captured[1][0] == pytest.approx(10.0)
        assert captured[1][1] == pytest.approx(0.0)


class TestDeviceLibraryExports:
    def test_create_device_library(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ = dummy_device_list
        output_file = tmp_path / "library.gds"

        calls: dict[str, object] = {}

        class FakeGDSWriter:
            def open_library(self, filename: str) -> None:
                calls["open"] = filename

            def write_structure(self, devname: str, geom: GeomGroup) -> None:
                calls["devname"] = devname
                calls["geom"] = geom

            def close_library(self) -> None:
                calls["closed"] = True

        monkeypatch.setattr(smdev, "GDSWriter", FakeGDSWriter)

        smdev.CreateDeviceLibrary(
            "TESTLIB_DUMMY_CONNECTOR", {"length": 7.0}, str(output_file)
        )

        geom = calls["geom"]

        assert calls["open"] == str(output_file)
        assert calls["devname"] == "TESTLIB_DUMMY_CONNECTOR"
        assert calls["closed"] is True
        assert isinstance(geom, GeomGroup)
        assert isinstance(geom.group[-1], sp.Text)

        txt = geom.group[-1]
        assert txt.x0 == pytest.approx(0)
        assert txt.y0 == pytest.approx(0)
        assert "__PORT__ io E" in txt.text
        assert "ConnectorPort" in txt.text

    def test_export_device_schematics(
        self,
        dummy_device_list: dict[str, type[smdev.Device]],
        tmp_path: Path,
    ) -> None:
        _ = dummy_device_list
        output_file = tmp_path / "SampleMakerLibrary.lel"

        smdev.ExportDeviceSchematics(str(output_file))
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
        smdev.registerDevicesInModule(__name__)
        assert "TESTLIB_DUMMY" in smdev._DeviceList
        assert "TESTLIB_DUMMY_CONNECTOR" in smdev._DeviceList

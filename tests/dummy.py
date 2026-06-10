from samplemaker.baselib.waveguides import BaseWaveguidePort
import samplemaker.devices as smdev
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


class DummyConnectorDevice(smdev.Device):
    def initialize(self):
        self.set_name("TESTLIB_DUMMY_CONNECTOR")
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

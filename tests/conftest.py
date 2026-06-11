"""Common fixtures for tests."""

from collections.abc import Generator

import pytest

import samplemaker
import samplemaker.devices as smdev
from tests import dummy as dm


@pytest.fixture(autouse=True)
def reset_samplemaker() -> None:
    """Reset samplemaker global variables before each test.

    This fixture should always be used to ensure that the layout pool is empty.
    Not doing so may lead to unexpected behavior in tests that rely on a clean state.
    """
    samplemaker.LayoutPool.clear()
    samplemaker._DevicePool.clear()
    samplemaker._DeviceLocalParamPool.clear()
    samplemaker._DeviceCountPool.clear()
    samplemaker._BoundingBoxPool.clear()


@pytest.fixture
def dummy_device() -> smdev.Device:
    return dm.DummyDevice.build()


@pytest.fixture
def dummy_connector_device() -> smdev.Device:
    return dm.DummyConnectorDevice.build()


@pytest.fixture
def default_device_list() -> Generator[dict[str, type[smdev.Device]], None, None]:
    old_devlist = smdev._DeviceList.copy()
    yield smdev._DeviceList
    smdev._DeviceList.clear()
    smdev._DeviceList.update(old_devlist)


@pytest.fixture
def dummy_device_list(
    default_device_list: dict[str, type[smdev.Device]],
) -> dict[str, type[smdev.Device]]:
    dev = dm.DummyDevice()
    dev.initialize()
    default_device_list[dev._name] = dm.DummyDevice

    dev = dm.DummyConnectorDevice()
    dev.initialize()
    default_device_list[dev._name] = dm.DummyConnectorDevice

    dummy_two_port_dev = dm.DummyTwoPortDevice()
    dummy_two_port_dev.initialize()
    default_device_list[dummy_two_port_dev._name] = dm.DummyTwoPortDevice

    return default_device_list

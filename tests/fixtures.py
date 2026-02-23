"""Common fixtures for tests."""

import pytest
import samplemaker


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

"""Shared test doubles used across test modules."""

from samplemaker.shapes import GeomGroup


class FakeGDSWriter:
    """Flexible GDS writer fake used by multiple test modules."""

    def __init__(self):
        self.calls: list[tuple[object, ...]] = []

    def open_library(self, filename: str) -> None:
        self.calls.append(("open_library", filename))

    def write_structure(self, devname: str, geom: GeomGroup) -> None:
        self.calls.append(("write_structure", devname, geom))

    def write_pool(self, pool: dict[str, GeomGroup]) -> None:
        self.calls.append(("write_pool", sorted(pool.keys())))

    def write_pool_use_cache(
        self, pool: dict[str, GeomGroup], celldata: dict[str, bytes]
    ) -> None:
        self.calls.append(
            ("write_pool_use_cache", sorted(pool.keys()), sorted(celldata.keys()))
        )

    def close_library(self) -> None:
        self.calls.append(("close_library", None))


class FakeGDSReader:
    """Simple reader double used to control Mask cache import/export paths."""

    def __init__(
        self,
        celldata: dict[str, bytes] | None = None,
        quick_read_exception: Exception | None = None,
    ):
        self.celldata = {} if celldata is None else dict(celldata)
        self.quick_read_exception = quick_read_exception
        self.quick_read_calls: list[str] = []
        self.cell_geometries: dict[str, GeomGroup] = {}

    def quick_read(self, filename: str) -> None:
        self.quick_read_calls.append(filename)
        if self.quick_read_exception is not None:
            raise self.quick_read_exception

    def get_cell(self, cellname: str) -> GeomGroup:
        return self.cell_geometries[cellname]

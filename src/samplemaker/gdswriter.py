"""Binary export to GDS files.

The `GDSWriter` class should not be used directly but via the `samplemaker.layout.Mask`
object in the `samplemaker.layout` submodule.

"""

import math
import struct
import time
from pathlib import Path as _Path
from typing import TYPE_CHECKING

import numpy as np

import samplemaker.shapes as smsh

if TYPE_CHECKING:
    from io import BufferedWriter


class GDSWriter:
    """GDS output class."""

    def __init__(self, circleres: int = 12, arcres: int = 32) -> None:
        """Initialize the GDSWriter class.

        Parameters
        ----------
        circleres : int, optional
            Number of points to use for circles, by default 12.
        arcres : int, optional
            Number of points to use for round elements (ellipses, rings, arcs). The
            default is 32.

        """
        self.circleres = circleres
        self.arcres = arcres
        self.xc = np.array([0.0] * circleres)
        self.yc = np.array([0.0] * circleres)
        self.fid: BufferedWriter | None = None
        for i in range(circleres):
            self.xc[i] = math.cos(i * 2 * math.pi / circleres)
            self.yc[i] = math.sin(i * 2 * math.pi / circleres)

    def __write_string(self, text: str, tag: int) -> None:
        text_len = len(text)
        self.__write_data(struct.pack(">2H", text_len + text_len % 2 + 4, tag))
        self.__write_data(text.encode())
        if text_len % 2 == 1:
            self.__write_data(struct.pack("b", 0))

    def __write_real8(self, value: float) -> None:
        num = value if value >= 0 else -value
        exponent = math.floor(-math.log2(num) / 4)
        mantissa = num * math.pow(2, 4 * exponent) * math.pow(2, 56)
        real = [0] * 8
        real[0] = (64 - exponent) | (128 if value < 0 else 0)
        for i in range(6, -1, -1):
            real[6 - i + 1] = math.floor(mantissa / math.pow(2, 8 * i))
            mantissa -= real[6 - i + 1] * math.pow(2, 8 * i)
        self.__write_data(struct.pack("8B", *real))

    def __write_data(self, data: bytes) -> None:
        if self.fid is None:
            msg = "File stream is not open."
            raise ValueError(msg)
        self.fid.write(data)

    def __write_polygon(self, poly: smsh.Poly) -> None:
        if poly.layer < 0:
            return
        pdata = poly.int_data()
        buf = np.array(
            [4, 0x0800, 6, 0x0D02, poly.layer, 6, 0x0E02, 0, 4 * len(pdata) + 4, 0x1003]
        )
        self.__write_data(struct.pack(f">{buf.size}H", *buf))
        self.__write_data(struct.pack(f">{pdata.size}i", *pdata))
        self.__write_data(struct.pack(">2H", 4, 0x1100))

    def __write_circle(self, circ: smsh.Circle) -> None:
        self.__write_polygon(
            smsh.Poly(
                circ.r * self.xc + circ.x0, circ.r * self.yc + circ.y0, circ.layer
            )
        )

    def __write_path(self, path: smsh.Path) -> None:
        buf = np.array(
            [4, 0x0900, 6, 0x0D02, path.layer, 6, 0x0E02, 0, 6, 0x2102, 1, 8, 0x0F03]
        )
        self.__write_data(struct.pack(f">{buf.size}H", *buf))
        self.__write_data(struct.pack(">i", math.floor(path.width * 1000)))
        self.__write_data(struct.pack(">2H", 8 * len(path.xpts) + 4, 0x1003))
        data = np.transpose(
            np.round(np.array([path.xpts, path.ypts]) * 1000).astype(int)
        ).reshape(-1)
        self.__write_data(struct.pack(f">{data.size}i", *data))
        self.__write_data(struct.pack(">2H", 4, 0x1100))

    def __write_text(self, text: smsh.Text) -> None:
        if text.text.replace(" ", "") == "":
            return
        buf = np.array(
            [
                4,
                0x0C00,
                6,
                0x0D02,
                text.layer,
                6,
                0x1602,
                0,
                6,
                0x1701,
                text.posu + text.posv * 4 + 16,
                8,
                0x0F03,
            ]
        )
        self.__write_data(struct.pack(f">{buf.size}H", *buf))
        self.__write_data(struct.pack(">i", math.floor(text.width * 1000)))
        self.__write_data(struct.pack(">2H", 12, 0x1003))
        self.__write_data(
            struct.pack(">2i", math.floor(text.x0 * 1000), math.floor(text.y0 * 1000))
        )
        text_len = len(text.text)
        self.__write_data(struct.pack(">2H", text_len + 4, 0x1906))
        self.__write_data(text.text.encode())
        self.__write_data(struct.pack(">2H", 4, 0x1100))

    def __write_strans(self, mag: float, angle: float, mirror: bool) -> None:
        if mag == 1 and angle == 0 and not mirror:
            return
        strans = 0
        if mirror:
            strans = 1 << 15
        # if(mag!=1):
        #    strans+=4
        # if(angle!=0):
        #    strans+=2
        buf = np.array([6, 0x1A01, strans])
        self.__write_data(struct.pack(f">{buf.size}H", *buf))
        if mag != 1:
            self.__write_data(struct.pack(">2H", 12, 0x1B05))
            self.__write_real8(mag)
        if angle != 0:
            self.__write_data(struct.pack(">2H", 12, 0x1C05))
            self.__write_real8(angle)

    def __write_sref(self, sref: smsh.SRef) -> None:
        self.__write_data(struct.pack(">2H", 4, 0x0A00))
        self.__write_string(sref.cellname, 0x1206)
        self.__write_strans(sref.mag, sref.angle, sref.mirror)
        self.__write_data(struct.pack(">2H", 12, 0x1003))
        rounded_x0 = round(float(sref.x0) * 1000)
        rounded_y0 = round(float(sref.y0) * 1000)
        self.__write_data(struct.pack(">2i", rounded_x0, rounded_y0))
        self.__write_data(struct.pack(">2H", 4, 0x1100))

    def __write_aref(self, aref: smsh.ARef) -> None:
        self.__write_data(struct.pack(">2H", 4, 0x0B00))
        self.__write_string(aref.cellname, 0x1206)
        self.__write_strans(aref.mag, aref.angle, aref.mirror)
        self.__write_data(
            struct.pack(
                ">4H", 8, 0x1302, math.floor(aref.ncols), math.floor(aref.nrows)
            )
        )
        self.__write_data(struct.pack(">2H", 28, 0x1003))
        rounded_x0 = round(float(aref.x0) * 1000)
        rounded_y0 = round(float(aref.y0) * 1000)
        self.__write_data(struct.pack(">2i", rounded_x0, rounded_y0))
        self.__write_data(
            struct.pack(
                ">2i",
                math.floor((aref.x0 + aref.ax * aref.ncols) * 1000),
                math.floor((aref.y0 + aref.ay * aref.ncols) * 1000),
            )
        )
        self.__write_data(
            struct.pack(
                ">2i",
                math.floor((aref.x0 + aref.bx * aref.nrows) * 1000),
                math.floor((aref.y0 + aref.by * aref.nrows) * 1000),
            )
        )
        self.__write_data(struct.pack(">2H", 4, 0x1100))

    def __large_polygons(self, gg: smsh.GeomGroup) -> smsh.GeomGroup:
        group = []
        for geom in gg.group:
            geomtype = type(geom)
            if geomtype == smsh.Poly and geom.Npts > 8000:
                newgrp = smsh.GeomGroup()
                newgrp.add(geom)
                newgrp.trapezoids(geom.layer)
                group += newgrp.group
                continue
            group += [geom]
        gg.group = group
        return gg

    def open_library(self, filename: str) -> None:
        """Open a new GDS file for writing.

        To close, call close_library().

        Parameters
        ----------
        filename : str
            The name of the file to write into.

        Returns
        -------
        None

        """
        if self.fid is not None:
            msg = (
                "A file stream is already open. "
                "Please close it before opening a new one."
            )
            raise ValueError(msg)
        self.fid = _Path(filename).open("wb")  # noqa: SIM115
        # Write header
        lt = time.localtime(time.time())
        buf = np.array(
            [
                6,
                2,
                3,
                28,
                258,
                lt.tm_year,
                lt.tm_mon,
                lt.tm_mday,
                lt.tm_hour,
                lt.tm_min,
                lt.tm_sec,
                lt.tm_year,
                lt.tm_mon,
                lt.tm_mday,
                lt.tm_hour,
                lt.tm_min,
                lt.tm_sec,
            ]
        )
        self.__write_data(struct.pack(f">{buf.size}H", *buf))
        # Library name
        self.__write_string(filename, 518)
        # Units
        self.__write_data(struct.pack(">2H", 20, 0x0305))
        self.__write_real8(1e-3)
        self.__write_real8(1e-9)
        print(f"Opened {filename}")

    def open_structure(self, structure_name: str) -> None:
        """Open a new structure (or cell) in the existing GDS stream.

        The file should be already open using open_library().

        This function can be used to write multiple objects in a single cell.

        The `close_structure()` should be called after writing all objects in the cell.

        Parameters
        ----------
        structure_name : str
            A string with a valid GDS cell/structure name.

        Returns
        -------
        None

        """
        print(f"Writing structure: {structure_name}")
        lt = time.localtime(time.time())
        buf = np.array(
            [
                28,
                1282,
                lt.tm_year,
                lt.tm_mon,
                lt.tm_mday,
                lt.tm_hour,
                lt.tm_min,
                lt.tm_sec,
                lt.tm_year,
                lt.tm_mon,
                lt.tm_mday,
                lt.tm_hour,
                lt.tm_min,
                lt.tm_sec,
            ]
        )
        self.__write_data(struct.pack(f">{buf.size}H", *buf))
        self.__write_string(structure_name, 1542)

    def write_geomgroup(self, geom_group: smsh.GeomGroup) -> None:
        """Write a GeomGroup to GDS stream.

        The file should be first opened with open_library() followed by
        open_structure().

        To be used for interactive writing only. See write_structure() for
        direct writing (recommended).

        Parameters
        ----------
        geom_group : GeomGroup
            The geometry to be written into GDS format.

        Returns
        -------
        None

        """
        geom_group = self.__large_polygons(geom_group)
        for geom in geom_group.group:
            geomtype = type(geom)
            if geomtype == smsh.Poly:
                self.__write_polygon(geom)
                continue
            if geomtype == smsh.Circle:
                self.__write_circle(geom)
                continue
            if geomtype == smsh.Path:
                self.__write_path(geom)
                continue
            if geomtype == smsh.Text:
                self.__write_text(geom)
                continue
            if geomtype == smsh.SRef:
                self.__write_sref(geom)
                continue
            if geomtype == smsh.ARef:
                self.__write_aref(geom)
                continue
            if geomtype == smsh.Ellipse:
                g = geom.to_polygon(self.arcres)
                self.__write_polygon(g.group[0])  # produces one geometry only
                continue
            if geomtype == smsh.Ring:
                g = geom.to_polygon(self.arcres)
                self.__write_polygon(g.group[0])  # produces one geometry only
                continue
            if geomtype == smsh.Arc:
                g = geom.to_polygon(self.arcres)
                self.__write_polygon(g.group[0])  # produces one geometry only
                continue

    def close_structure(self) -> None:
        """Close the open structure.

        Should be called after open_structure().

        Returns
        -------
        None

        """
        self.__write_data(struct.pack(">2H", 4, 1792))

    def write_structure(self, structure_name: str, geom_group: smsh.GeomGroup) -> None:
        """Write a GeomGroup into a named structure/cell.

        The GeomGroup is written into the cell once and then the GDS cell is closed.

        Parameters
        ----------
        structure_name : str
            A string with a valid GDS structure/cell name.
        geom_group : GeomGroup
            The GeomGroup to be written into GDS format.

        Returns
        -------
        None

        """
        self.open_structure(structure_name)
        self.write_geomgroup(geom_group)
        self.close_structure()

    def write_pool(self, pool: dict[str, smsh.GeomGroup]) -> None:
        """Write all the structures in the pool dictionary into GDS cells.

        Parameters
        ----------
        pool : dict
            A dictionary containing structure names as keys and GeomGroup as values.

        Returns
        -------
        None

        """
        for sname, group in pool.items():
            self.write_structure(sname, group)

    def write_pool_use_cache(
        self, pool: dict[str, smsh.GeomGroup], cache: dict[str, bytes]
    ) -> None:
        """Write all the structures in the pool dictionary into GDS cells.

        Uses GDS cache data when available.

        Parameters
        ----------
        pool : dict[str, GeomGroup]
            A dictionary containing structure names as keys and GeomGroup as values.
        cache: dict[str, bytes]
            A dictionary containing structure names as keys and binary GDS data as
            values.

        Returns
        -------
        None

        """
        for sname, group in pool.items():
            if sname in cache:
                print("Writing cached", sname)
                self.__write_data(cache[sname])
            else:
                self.write_structure(sname, group)

    def close_library(self) -> None:
        """Close the GDS library and the file stream.

        Returns
        -------
        None

        """
        if self.fid is None:
            return
        self.__write_data(struct.pack(">2H", 4, 1024))
        pos = self.fid.tell()
        buf = np.zeros(2048 - pos % 2048, dtype=int)
        self.__write_data(struct.pack(f"{buf.size}b", *buf))
        print("Writing to GDS complete.")
        self.fid.close()
        self.fid = None

    def __del__(self) -> None:
        """Destructor to ensure the file stream is closed."""
        self.close_library()

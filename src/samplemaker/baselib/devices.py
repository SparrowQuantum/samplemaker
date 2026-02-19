"""
Base device library.

This is a collection of some simple demo devices distributed with the base version of
`samplemaker`. It can be used as template for creating new libraries or to learn how to
design them.

Note that individual device methods are not documented but should be readable and
self-explanatory.
"""

import math

import numpy as np

import samplemaker.makers as sm
from samplemaker.baselib.waveguides import BaseWaveguidePort, BaseWaveguideSequencer
from samplemaker.devices import Device, registerDevicesInModule


class CrossMark(Device):
    def initialize(self):
        self.set_name("BASELIB_CMARK")
        self.set_description("Generic cross marker for mask alignment.")

    def parameters(self):
        self.addparameter(
            param_name="length1",
            default_value=20,
            param_description="Length of inner cross",
            param_type=float,
        )
        self.addparameter(
            param_name="length2",
            default_value=10,
            param_description="Length of outer cross",
            param_type=float,
        )
        self.addparameter(
            param_name="width1",
            default_value=0.5,
            param_description="Width of inner cross",
            param_type=float,
        )
        self.addparameter(
            param_name="width2",
            default_value=2,
            param_description="width of outer cross",
            param_type=float,
        )
        self.addparameter(
            param_name="layer",
            default_value=4,
            param_description="Layer to use for cross",
            param_type=int,
            param_range=(0, 255),
        )
        self.addparameter(
            param_name="mark_number",
            default_value=0,
            param_description="Places a square in the corner, use 0 to remove",
            param_type=float,
            param_range=(0, 4),
        )
        self.addparameter(
            param_name="square_size",
            default_value=10,
            param_description="Size of the square in the corner",
            param_type=float,
        )

    def geom(self):
        p = self.get_params()
        cross = sm.make_rect(0, 0, p["length1"], p["width1"], layer=1)
        cross += sm.make_rect(0, 0, p["width1"], p["length1"], layer=1)
        cross.boolean_union(1)
        ocross = sm.make_rect(p["length1"] / 2, 0, p["length2"], p["width2"], numkey=4)
        for i in range(4):
            c = ocross.copy()
            c.rotate(0, 0, 90 * i)
            cross += c
        if p["mark_number"] > 0:
            rot = 90 * (p["mark_number"] - 1)
            sq_dim = p["length1"] / 2 + p["length2"]
            sq_size = p["square_size"]
            square = sm.make_rect(sq_dim, sq_dim, sq_size, sq_size, numkey=1)
            square.rotate(0, 0, rot)
            cross += square

        cross.set_layer(p["layer"])
        return cross


class DirectionalCoupler(Device):
    def initialize(self):
        self.set_name("BASELIB_DCPL")
        self.set_description("Simple symmetric directional coupler")

    def parameters(self):
        self.addparameter("length", 20, "Coupling length", float)
        self.addparameter(
            param_name="width",
            default_value=0.3,
            param_description="Width of the waveguides in the coupling section",
            param_type=float,
            param_range=(0.01, 1),
        )
        self.addparameter(
            param_name="gap",
            default_value=0.5,
            param_description="Distance between waveguides in the coupling section",
            param_type=float,
        )
        self.addparameter(
            param_name="input_dist",
            default_value=5,
            param_description="Distance between waveguides at input",
            param_type=float,
            param_range=(0.01, np.inf),
        )
        self.addparameter(
            param_name="input_len",
            default_value=7,
            param_description="Length of the input section from input to coupling",
            param_type=float,
            param_range=(3, np.inf),
        )

    def geom(self):
        p = self.get_params()
        # Draw the upper arm, then mirror
        off = p["input_dist"] / 2
        clen = (p["input_len"] - 1) / 2
        ltot = p["length"] + p["input_len"] * 2
        seq = [["T", 1, p["width"]], ["C", -off, clen], ["S", p["length"] / 2]]
        ss = BaseWaveguideSequencer(seq)
        dc = ss.run()
        dc2 = dc.copy()
        dc2.mirrorX(ltot / 2)
        dc += dc2
        dc.translate(-ltot / 2, off + p["gap"] / 2 + p["width"] / 2)
        dc3 = dc.copy()
        dc3.mirrorY(0)
        dc += dc3

        # Add ports
        xp = ltot / 2
        yp = off + p["gap"] / 2 + p["width"] / 2
        nw_port = BaseWaveguidePort(-xp, yp, "west", ss.options["defaultWidth"], "p1")
        ne_port = BaseWaveguidePort(xp, yp, "east", ss.options["defaultWidth"], "p2")
        sw_port = BaseWaveguidePort(-xp, -yp, "west", ss.options["defaultWidth"], "p3")
        se_port = BaseWaveguidePort(xp, -yp, "east", ss.options["defaultWidth"], "p4")
        for port in (nw_port, ne_port, sw_port, se_port):
            self.addlocalport(port)

        return dc


class FocusingGratingCoupler(Device):
    def initialize(self):
        self.set_name("BASELIB_FGC")
        self.set_description("Grating coupler demo.")

    def parameters(self):
        self.addparameter(
            param_name="w0",
            default_value=0.3,
            param_description="Width of the waveguide at the start",
            param_type=float,
        )
        self.addparameter(
            param_name="pitch",
            default_value=0.355,
            param_description="Grating default pitch",
            param_type=float,
        )
        self.addparameter(
            param_name="ff",
            default_value=0.5,
            param_description="Fill factor",
            param_type=float,
        )
        self.addparameter(
            param_name="theta",
            default_value=10,
            param_description="Emission angle at central wavelength",
            param_type=float,
        )
        self.addparameter(
            param_name="lambda0",
            default_value=0.94,
            param_description="Central wavelength",
            param_type=float,
        )
        self.addparameter(
            param_name="nr_Apo",
            default_value=11,
            param_description="nr of the 1st arc with pitch and ff",
            param_type=int,
        )
        self.addparameter(
            param_name="ff_coef",
            default_value=0.5,
            param_description="min ff_apod = ff_coef*ff",
            param_type=float,
        )
        self.addparameter(
            param_name="order_start",
            default_value=10,
            param_description="Starting period",
            param_type=int,
        )
        self.addparameter(
            param_name="order",
            default_value=15,
            param_description="Number of periods",
            param_type=int,
        )
        self.addparameter(
            param_name="diverg_angle",
            default_value=20,
            param_description="GRT divergence angle/2, deg",
            param_type=float,
        )
        self.addparameter(
            param_name="pre_split",
            default_value=True,
            param_description="Split in quads = false",
            param_type=bool,
        )

    def geom(self):
        # Grating first
        p = self.get_params()
        theta = math.radians(p["theta"])
        div_angle = p["diverg_angle"]
        q0 = p["order_start"]
        qn = q0 + p["order"] + 1
        lambda0 = p["lambda0"]
        pitch = p["pitch"]
        n = math.sin(theta) + lambda0 / pitch  # Effective refractive index
        p0 = lambda0 / math.sqrt(n * n - np.power(math.sin(theta), 2))
        ff = p["ff"]
        nr_apo = p["nr_Apo"]
        ff_coef = p["ff_coef"]

        g = sm.GeomGroup()
        for q in range(q0, qn):
            b = q * p0
            x0 = b * b * math.sin(theta) / (q * lambda0)
            a = b * b * n / (q * lambda0)
            if q <= q0 + nr_apo - 1:
                ff_chi = ff - (1 - ff_coef) * ff / (nr_apo - 2) * (q0 + nr_apo - q)
            else:
                ff_chi = ff

            w = ff_chi * pitch
            g += sm.make_arc(
                x0=x0,
                y0=0,
                rX=a,
                rY=b,
                rot=0,
                w=w,
                a1=-div_angle - 5,
                a2=div_angle + 5,
                layer=3,
                to_poly=True,
                vertices=40,
                split=p["pre_split"],
            )

        # waveguide
        l_taper = 1
        g_taper = qn * pitch
        w_taper = g_taper * math.tan(math.radians(div_angle)) * 2

        seq = [
            ["T", l_taper, p["w0"]],
            ["CENTER", 0, 0],
            ["T", g_taper, w_taper],
            ["STATE", "w", 2.5],
            ["S", 1],
        ]

        ss = BaseWaveguideSequencer(seq)
        g += ss.run()

        self.addlocalport(BaseWaveguidePort(-l_taper, 0, "west", p["w0"], "p1"))

        return g


# Register all devices here in this module
registerDevicesInModule(__name__)

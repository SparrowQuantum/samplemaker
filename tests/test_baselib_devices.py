"""Tests for samplemaker.baselib.devices."""

import pytest

from samplemaker.baselib.devices import (
    CrossMark,
    DirectionalCoupler,
    FocusingGratingCoupler,
)


def test_crossmark_layer_and_bbox_change_with_parameters() -> None:
    base = CrossMark.build()
    base.use_references = False
    g_base = base.run()
    bb_base = g_base.bounding_box()

    assert g_base.get_layer_list() == {4}
    assert bb_base.width == pytest.approx(40.0)
    assert bb_base.height == pytest.approx(40.0)

    changed = CrossMark.build()
    changed.use_references = False
    changed.set_param("length2", 20.0)
    changed.set_param("layer", 9)
    g_changed = changed.run()
    bb_changed = g_changed.bounding_box()

    assert g_changed.get_layer_list() == {9}
    assert bb_changed.width > bb_base.width + 19.0
    assert bb_changed.height > bb_base.height + 19.0


def test_crossmark_mark_number_adds_corner_square() -> None:
    no_mark = CrossMark.build()
    no_mark.use_references = False
    bb_no_mark = no_mark.run().bounding_box()

    with_mark = CrossMark.build()
    with_mark.use_references = False
    with_mark.set_param("mark_number", 1)
    with_mark.set_param("square_size", 12.0)
    bb_with_mark = with_mark.run().bounding_box()

    assert bb_with_mark.width > bb_no_mark.width
    assert bb_with_mark.height > bb_no_mark.height


def test_directional_coupler_ports_follow_parameters() -> None:
    dev = DirectionalCoupler.build()
    dev.use_references = False
    dev.set_param("length", 30.0)
    dev.set_param("input_len", 9.0)
    dev.set_param("input_dist", 8.0)
    dev.set_param("gap", 0.6)
    dev.set_param("width", 0.4)
    dev.run()

    assert set(dev._ports) == {"p1", "p2", "p3", "p4"}

    xp = (30.0 + 2 * 9.0) / 2
    yp = 8.0 / 2 + 0.6 / 2 + 0.4 / 2
    assert dev.get_port("p1").x0 == pytest.approx(-xp)
    assert dev.get_port("p1").y0 == pytest.approx(yp)
    assert dev.get_port("p1").angle_to_text() == "W"
    assert dev.get_port("p2").x0 == pytest.approx(xp)
    assert dev.get_port("p2").y0 == pytest.approx(yp)
    assert dev.get_port("p2").angle_to_text() == "E"
    assert dev.get_port("p3").x0 == pytest.approx(-xp)
    assert dev.get_port("p3").y0 == pytest.approx(-yp)
    assert dev.get_port("p3").angle_to_text() == "W"
    assert dev.get_port("p4").x0 == pytest.approx(xp)
    assert dev.get_port("p4").y0 == pytest.approx(-yp)
    assert dev.get_port("p4").angle_to_text() == "E"


def test_directional_coupler_geometry_changes_with_length_and_gap() -> None:
    base = DirectionalCoupler.build()
    base.use_references = False
    bb_base = base.run().bounding_box()

    longer = DirectionalCoupler.build()
    longer.use_references = False
    longer.set_param("length", 40.0)
    bb_longer = longer.run().bounding_box()

    wider_gap = DirectionalCoupler.build()
    wider_gap.use_references = False
    wider_gap.set_param("gap", 2.5)
    bb_wider_gap = wider_gap.run().bounding_box()

    assert bb_longer.width > bb_base.width + 15.0
    assert bb_wider_gap.height > bb_base.height + 1.0


def test_focusing_grating_coupler_has_expected_layers_and_port() -> None:
    dev = FocusingGratingCoupler.build()
    dev.use_references = False
    geom = dev.run()

    layers = geom.get_layer_list()
    assert 1 in layers
    assert 3 in layers

    p1 = dev.get_port("p1")
    assert p1.x0 == pytest.approx(-1.0)
    assert p1.y0 == pytest.approx(0.0)
    assert p1.angle_to_text() == "W"
    assert p1.width == pytest.approx(dev.get_params()["w0"])


def test_focusing_grating_coupler_bbox_changes_with_order_and_divergence() -> None:
    base = FocusingGratingCoupler.build()
    base.use_references = False
    bb_base = base.run().bounding_box()

    higher_order = FocusingGratingCoupler.build()
    higher_order.use_references = False
    higher_order.set_param("order", 25)
    bb_higher_order = higher_order.run().bounding_box()

    larger_divergence = FocusingGratingCoupler.build()
    larger_divergence.use_references = False
    larger_divergence.set_param("diverg_angle", 35.0)
    bb_larger_divergence = larger_divergence.run().bounding_box()

    assert bb_higher_order.width > bb_base.width
    assert bb_larger_divergence.height > bb_base.height

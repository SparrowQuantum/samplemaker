"""13_tutorial_geometry_validation.

Use structural fingerprints as regression checks and dictionaries to investigate
unexpected geometry changes. Run this tutorial from a writable output directory.
"""

import json

import samplemaker.layout as smlay
import samplemaker.makers as sm

# Start with a small drawing whose dimensions and layers we have checked.
base = sm.make_rect(0, 0, 2, 10)
base += sm.make_circle(4, 5, 1, layer=3)

# In a regression test, compare the result against a fingerprint saved after
# reviewing the geometry. Only replace this baseline after reviewing a change;
# calculating the expected value from the current result would not detect a bug.
expected_fingerprint = (
    "48a6d5090f71b6a8cb778db3a729c8492b1010056510760dc7361b6407cec2c2"
)
assert base.fingerprint() == expected_fingerprint

# Copies and element reordering do not change the result. Duplicate elements,
# layers, coordinates, primitive types, and reference hierarchy do matter.
reordered = base.copy()
reordered.group.reverse()
assert reordered.fingerprint() == expected_fingerprint

# A dictionary snapshot can be saved as JSON alongside a regression baseline.
# Unlike the digest alone, it records the geometry for investigating a mismatch.
# Version 1 stores groups once, keyed by their content digest; root identifies
# the top group, and references use {"$ref": digest} instead of repeating groups.
snapshot = base.to_dict()
print("Geometry snapshot:")
print(json.dumps(snapshot, sort_keys=True, indent=2))

# Changing a layer is an intentional example of a regression.
changed = base.copy()
changed.set_layer(7)
assert changed.fingerprint() != expected_fingerprint
assert changed.to_dict() != snapshot

# Individual primitive dictionaries help pinpoint the changed fields without
# navigating the group table. Floats are exact hexadecimal strings, not rounded
# decimal coordinates (float.fromhex() converts them back to ordinary floats).
print("Original primitive:", base.group[0].to_dict())
print("Changed primitive: ", changed.group[0].to_dict())

# Validate the assembled mask too: its fingerprint covers the main cell and all
# reachable named cells, including reference positions and array parameters.
themask = smlay.Mask("13_tutorial_geometry_validation")
themask.add_cell("BASE", base)
geom = sm.make_sref(0, 0, "BASE", base)
geom += sm.make_sref(20, 0, "BASE", base)
themask.add_to_main_cell(geom)
mask_fingerprint = themask.fingerprint()
mask_snapshot = themask.to_dict()
print("Mask snapshot:")
print(json.dumps(mask_snapshot, sort_keys=True, indent=2))

# References share their target. Even a nested edit is detected on the next call;
# neither fingerprint() nor to_dict() retains a stale cache across calls.
base.set_layer(7)
assert themask.fingerprint() != mask_fingerprint
assert themask.to_dict() != mask_snapshot
# Restore the layer on each original primitive, preserving their different layers.
base.group[0].layer = 1
base.group[1].layer = 3
assert themask.fingerprint() == mask_fingerprint

# Unreferenced cells and the output filename are not part of mask validation.
themask.add_cell("UNUSED", sm.make_rect(0, 0, 100, 100))
assert themask.fingerprint() == mask_fingerprint
themask.export_gds()
assert themask.fingerprint() == mask_fingerprint

# These are exact structural checks, not physical-equivalence or fabrication
# checks. Flattening, polygon conversion, or tiny float changes can change a
# fingerprint even when a drawing looks identical. This validates the in-memory
# layout, not the exported GDS bytes or a geometric tolerance.
# Cycles, non-finite numbers, missing named cells, and references inconsistent
# with the mask's named cells are rejected rather than silently fingerprinted.

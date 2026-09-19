from road_defect.classes import CLASS_IDS, RDD_CODE_TO_CLASS
from road_defect.data.voc_to_yolo import convert_box


def test_centered_box():
    assert convert_box((100, 100, 300, 300), 400, 400) == (0.5, 0.5, 0.5, 0.5)


def test_offset_box():
    cx, cy, w, h = convert_box((0, 0, 100, 50), 200, 100)
    assert (cx, cy, w, h) == (0.25, 0.25, 0.5, 0.5)


def test_non_square_image_normalizes_per_axis():
    cx, cy, w, h = convert_box((0, 0, 600, 300), 600, 600)
    assert (cx, cy, w, h) == (0.5, 0.25, 1.0, 0.5)


def test_box_is_clamped_to_image_bounds():
    cx, cy, w, h = convert_box((-50, -50, 700, 700), 600, 600)
    assert (cx, cy, w, h) == (0.5, 0.5, 1.0, 1.0)


def test_inverted_coordinates_are_normalized():
    inverted = convert_box((300, 300, 100, 100), 400, 400)
    assert inverted == convert_box((100, 100, 300, 300), 400, 400)


def test_degenerate_boxes_are_rejected():
    assert convert_box((10, 10, 10, 10), 600, 600) is None
    assert convert_box((10, 10, 10.5, 400), 600, 600) is None


def test_boxes_fully_outside_image_are_rejected():
    assert convert_box((700, 700, 900, 900), 600, 600) is None


def test_zero_sized_image_is_rejected():
    assert convert_box((0, 0, 10, 10), 0, 0) is None


def test_crack_codes_collapse_to_one_class():
    assert {RDD_CODE_TO_CLASS[c] for c in ("D00", "D10", "D20")} == {"crack"}


def test_pothole_and_lane_marking_mapping():
    assert RDD_CODE_TO_CLASS["D40"] == "pothole"
    assert RDD_CODE_TO_CLASS["D44"] == "faded_lane_marking"


def test_non_defect_codes_are_unmapped():
    # D50 is a utility hole cover and D43 a crosswalk, neither is road damage.
    assert "D50" not in RDD_CODE_TO_CLASS
    assert "D43" not in RDD_CODE_TO_CLASS


def test_class_ids_are_contiguous_from_zero():
    assert sorted(CLASS_IDS.values()) == list(range(len(CLASS_IDS)))

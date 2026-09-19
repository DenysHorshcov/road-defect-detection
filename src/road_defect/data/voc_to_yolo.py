"""Convert RDD2022 Pascal VOC annotations into YOLO label files.

RDD2022 ships one XML per image with absolute xmin/ymin/xmax/ymax boxes tagged
by damage code; Ultralytics wants `class cx cy w h` normalised to [0, 1].
Codes outside RDD_CODE_TO_CLASS are dropped -- notably D50 (utility hole cover),
which is not road damage, and D43 (crosswalk blur), see classes.py.

Usage:
    uv run python -m road_defect.data.voc_to_yolo
"""

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from road_defect.classes import CLASS_IDS, CLASS_NAMES, RDD_CODE_TO_CLASS


def convert_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Absolute (xmin, ymin, xmax, ymax) -> normalised (cx, cy, w, h).

    Returns None for boxes that are degenerate or fall outside the image, which
    do occur in RDD2022 and would otherwise poison training.
    """
    xmin, ymin, xmax, ymax = box
    if width <= 0 or height <= 0:
        return None

    xmin, xmax = sorted((xmin, xmax))
    ymin, ymax = sorted((ymin, ymax))

    xmin = max(0.0, min(xmin, width))
    xmax = max(0.0, min(xmax, width))
    ymin = max(0.0, min(ymin, height))
    ymax = max(0.0, min(ymax, height))

    box_w = xmax - xmin
    box_h = ymax - ymin
    if box_w <= 1 or box_h <= 1:
        return None

    return (
        (xmin + box_w / 2) / width,
        (ymin + box_h / 2) / height,
        box_w / width,
        box_h / height,
    )


def convert_annotation(xml_path: Path) -> tuple[list[str], Counter]:
    """Parse one VOC file into YOLO label lines plus per-class counts."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))

    lines: list[str] = []
    counts: Counter = Counter()

    for obj in root.findall("object"):
        code = (obj.findtext("name") or "").strip()
        class_name = RDD_CODE_TO_CLASS.get(code)
        if class_name is None:
            counts[f"skipped:{code}"] += 1
            continue

        bnd = obj.find("bndbox")
        raw = tuple(float(bnd.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        converted = convert_box(raw, width, height)
        if converted is None:
            counts["skipped:degenerate"] += 1
            continue

        cx, cy, w, h = converted
        lines.append(f"{CLASS_IDS[class_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        counts[class_name] += 1

    return lines, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xmls", type=Path, default=Path("data/raw/Japan/train/annotations/xmls"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/labels_all"))
    args = parser.parse_args()

    xml_files = sorted(args.xmls.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No XML files under {args.xmls}")

    args.out.mkdir(parents=True, exist_ok=True)
    totals: Counter = Counter()
    written = 0
    empty = 0

    for xml_path in xml_files:
        lines, counts = convert_annotation(xml_path)
        totals.update(counts)
        if not lines:
            # No mapped defect in this image; excluded from the dataset.
            empty += 1
            continue
        (args.out / f"{xml_path.stem}.txt").write_text("\n".join(lines) + "\n")
        written += 1

    print(f"Parsed {len(xml_files)} annotations -> {written} label files ({empty} had no defect)")
    print()
    print(f"{'class':<22}{'instances':>10}")
    for name in CLASS_NAMES:
        print(f"{name:<22}{totals[name]:>10}")
    print()
    skipped = {k: v for k, v in totals.items() if k.startswith("skipped:")}
    for key, value in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<20}{value:>10}")


if __name__ == "__main__":
    main()

"""Arrange converted RDD2022 labels into an Ultralytics dataset directory.

RDD2022 images come from continuous road surveys, so `Japan_000123` and
`Japan_000124` are often metres apart and nearly identical. Splitting at random
would leak near-duplicates from train into test and inflate every metric, so we
split on contiguous blocks of frames instead of individual images.

Images are hardlinked rather than copied where the filesystem allows it, so the
dataset directory costs almost no extra disk.

Usage:
    uv run python -m road_defect.data.split
"""

import argparse
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

from road_defect.classes import CLASS_NAMES

SPLITS = ("train", "val", "test")


def assign_blocks(
    stems: list[str], block_size: int, ratios: tuple[float, float, float], seed: int
) -> dict[str, list[str]]:
    """Chunk consecutive frames into blocks, then deal whole blocks to splits."""
    blocks = [stems[i : i + block_size] for i in range(0, len(stems), block_size)]
    random.Random(seed).shuffle(blocks)

    n_train = int(len(blocks) * ratios[0])
    n_val = int(len(blocks) * ratios[1])
    grouped = {
        "train": blocks[:n_train],
        "val": blocks[n_train : n_train + n_val],
        "test": blocks[n_train + n_val :],
    }
    return {split: [stem for block in bs for stem in block] for split, bs in grouped.items()}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/processed/labels_all"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/Japan/train/images"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/rdd_japan"))
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    label_files = sorted(args.labels.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"No label files under {args.labels}. Run voc_to_yolo first.")

    stems = [p.stem for p in label_files]
    assignment = assign_blocks(stems, args.block_size, (0.7, 0.2, 0.1), args.seed)

    for split in SPLITS:
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {split: Counter() for split in SPLITS}
    missing = 0

    for split, split_stems in assignment.items():
        for stem in split_stems:
            image_src = args.images / f"{stem}.jpg"
            if not image_src.exists():
                missing += 1
                continue

            link_or_copy(image_src, args.out / "images" / split / f"{stem}.jpg")
            label_text = (args.labels / f"{stem}.txt").read_text()
            (args.out / "labels" / split / f"{stem}.txt").write_text(label_text)

            stats[split]["images"] += 1
            for line in label_text.strip().splitlines():
                stats[split][CLASS_NAMES[int(line.split()[0])]] += 1

    dataset_yaml = args.out / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(args.out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": dict(enumerate(CLASS_NAMES)),
            },
            sort_keys=False,
        )
    )

    header = f"{'split':<8}{'images':>8}" + "".join(f"{n:>20}" for n in CLASS_NAMES)
    print(header)
    for split in SPLITS:
        row = f"{split:<8}{stats[split]['images']:>8}"
        row += "".join(f"{stats[split][n]:>20}" for n in CLASS_NAMES)
        print(row)
    if missing:
        print(f"\nWARNING: {missing} labels had no matching image")
    print(f"\nWrote {dataset_yaml}")


if __name__ == "__main__":
    main()

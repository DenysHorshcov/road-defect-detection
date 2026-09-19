"""Collect street-level imagery from the Mapillary Graph API.

The /images endpoint rejects bounding boxes of 0.01 degrees or larger and
returns at most 2000 images per query, so an area of interest is swept as a
grid of small tiles rather than fetched in one call.

Mapillary imagery is CC BY-SA 4.0. The manifest keeps each image's id and URL
so the dataset can be published as annotations plus references, which avoids
the ShareAlike obligations that redistributing the images in bulk would carry.

Usage:
    uv run python -m road_defect.data.mapillary --limit 500
    uv run python -m road_defect.data.mapillary --download
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

SEARCH_URL = "https://graph.mapillary.com/images"
FIELDS = "id,thumb_2048_url,geometry,captured_at,compass_angle"
MAX_TILE_DEG = 0.01
Bbox = tuple[float, float, float, float]


def iter_tiles(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float, tile_size: float
) -> list[Bbox]:
    """Split an area into bboxes the API will accept, clipped to the area."""
    if not 0 < tile_size < MAX_TILE_DEG:
        raise ValueError(f"tile_size must be in (0, {MAX_TILE_DEG}), got {tile_size}")
    if max_lon <= min_lon or max_lat <= min_lat:
        raise ValueError("max_lon/max_lat must exceed min_lon/min_lat")

    n_lon = math.ceil((max_lon - min_lon) / tile_size)
    n_lat = math.ceil((max_lat - min_lat) / tile_size)

    tiles = []
    for i in range(n_lon):
        for j in range(n_lat):
            lon0 = min_lon + i * tile_size
            lat0 = min_lat + j * tile_size
            tiles.append(
                (lon0, lat0, min(lon0 + tile_size, max_lon), min(lat0 + tile_size, max_lat))
            )
    return tiles


def search_tile(session: requests.Session, token: str, bbox: Bbox) -> list[dict]:
    params = {
        "access_token": token,
        "fields": FIELDS,
        "bbox": ",".join(f"{v:.6f}" for v in bbox),
    }
    for attempt in range(5):
        resp = session.get(SEARCH_URL, params=params, timeout=60)
        if resp.status_code == 429:
            time.sleep(2**attempt)
            continue
        resp.raise_for_status()
        return resp.json().get("data", [])
    raise RuntimeError(f"Rate limited repeatedly on bbox {bbox}")


def scrape(areas_config: Path, token: str, manifest_path: Path, limit: int | None) -> dict:
    config = yaml.safe_load(areas_config.read_text())
    tile_size = config.get("tile_size_deg", 0.009)

    images: dict[str, dict] = {}
    if manifest_path.exists():
        images = json.loads(manifest_path.read_text())
        print(f"Resuming from {len(images)} images already in the manifest")

    session = requests.Session()
    for area in config["areas"]:
        tiles = iter_tiles(
            area["min_lon"], area["min_lat"], area["max_lon"], area["max_lat"], tile_size
        )
        print(f"{area['name']}: {len(tiles)} tiles")

        for n, bbox in enumerate(tiles, 1):
            for image in search_tile(session, token, bbox):
                image["area"] = area["name"]
                images[image["id"]] = image
            print(f"\r  tile {n}/{len(tiles)} -> {len(images)} images", end="", flush=True)
            if limit and len(images) >= limit:
                break
        print()
        if limit and len(images) >= limit:
            print(f"Reached limit of {limit}")
            break

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(images, indent=2))
    return images


def download(manifest_path: Path, out_dir: Path) -> int:
    images = json.loads(manifest_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    saved = 0
    for n, (image_id, meta) in enumerate(images.items(), 1):
        target = out_dir / f"{image_id}.jpg"
        if target.exists():
            continue
        url = meta.get("thumb_2048_url")
        if not url:
            continue
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        saved += 1
        print(f"\r  {n}/{len(images)} ({saved} new)", end="", flush=True)
    print()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--areas", type=Path, default=Path("configs/mapillary_areas.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/mapillary/manifest.json"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/mapillary/images"))
    parser.add_argument("--limit", type=int, default=None, help="stop after this many images")
    parser.add_argument("--download", action="store_true", help="download thumbs in the manifest")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise SystemExit("MAPILLARY_TOKEN is not set. See .env.example.")

    if not args.download:
        images = scrape(args.areas, token, args.manifest, args.limit)
        print(f"Manifest: {len(images)} images -> {args.manifest}")
    else:
        saved = download(args.manifest, args.images)
        print(f"Downloaded {saved} new images -> {args.images}")


if __name__ == "__main__":
    main()

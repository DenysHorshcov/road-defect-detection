"""Download a country subset of RDD2022 from figshare.

RDD2022 is published as a single 13.3 GB zip whose top level holds one zip per
country. figshare's storage honours HTTP range requests, so we mount the remote
archive as a seekable file, read its central directory, and stream out only the
country we want -- 1.07 GB for Japan instead of 13.3 GB for everything.

Usage:
    uv run python -m road_defect.data.download_rdd --country Japan
"""

import argparse
import io
import shutil
import zipfile
from pathlib import Path

import requests

FIGSHARE_URL = "https://ndownloader.figshare.com/files/38030910"
BLOCK_SIZE = 8 * 1024 * 1024
MAX_CACHED_BLOCKS = 8


class HttpRangeReader(io.RawIOBase):
    """A seekable read-only file backed by HTTP range requests.

    Reads are served in fixed-size cached blocks, because zipfile issues many
    small seeks and an unbuffered mapping would mean one request per read.
    """

    def __init__(self, url: str, block_size: int = BLOCK_SIZE):
        self._url = url
        self._block_size = block_size
        self._pos = 0
        self._blocks: dict[int, bytes] = {}
        self._resolved_url, self.size = self._resolve()

    def _resolve(self) -> tuple[str, int]:
        resp = requests.get(self._url, headers={"Range": "bytes=0-0"}, timeout=60)
        resp.raise_for_status()
        content_range = resp.headers.get("Content-Range")
        if not content_range:
            raise RuntimeError("Server did not honour a range request; cannot stream the zip.")
        return resp.url, int(content_range.split("/")[-1])

    def _fetch_block(self, index: int) -> bytes:
        cached = self._blocks.get(index)
        if cached is not None:
            return cached

        start = index * self._block_size
        end = min(start + self._block_size, self.size) - 1
        headers = {"Range": f"bytes={start}-{end}"}

        resp = requests.get(self._resolved_url, headers=headers, timeout=300)
        if resp.status_code == 403:
            # figshare hands out short-lived presigned URLs; re-resolve and retry.
            self._resolved_url, self.size = self._resolve()
            resp = requests.get(self._resolved_url, headers=headers, timeout=300)
        resp.raise_for_status()

        data = resp.content
        if len(self._blocks) >= MAX_CACHED_BLOCKS:
            self._blocks.pop(next(iter(self._blocks)))
        self._blocks[index] = data
        return data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        return self._pos

    def readinto(self, buffer) -> int:
        requested = min(len(buffer), max(0, self.size - self._pos))
        written = 0
        while written < requested:
            index, offset = divmod(self._pos, self._block_size)
            block = self._fetch_block(index)
            chunk = block[offset : offset + (requested - written)]
            buffer[written : written + len(chunk)] = chunk
            written += len(chunk)
            self._pos += len(chunk)
        return written


def fetch_country_zip(country: str, dest_dir: Path) -> Path:
    """Stream <country>.zip out of the remote combined archive."""
    local_zip = dest_dir / f"{country}.zip"
    if local_zip.exists():
        print(f"{local_zip} already present, skipping download")
        return local_zip

    reader = HttpRangeReader(FIGSHARE_URL)
    print(f"Remote archive: {reader.size / 1e9:.2f} GB")

    with zipfile.ZipFile(reader) as archive:
        try:
            member = archive.getinfo(f"RDD2022/{country}.zip")
        except KeyError:
            available = sorted(Path(n).stem for n in archive.namelist())
            raise SystemExit(f"Unknown country {country!r}. Available: {available}") from None

        total = member.file_size
        print(f"Downloading {country}.zip ({total / 1e9:.2f} GB)")

        dest_dir.mkdir(parents=True, exist_ok=True)
        partial = local_zip.with_suffix(".zip.part")
        copied = 0
        with archive.open(member) as src, partial.open("wb") as dst:
            while chunk := src.read(4 * 1024 * 1024):
                dst.write(chunk)
                copied += len(chunk)
                print(f"\r  {copied / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
        print()
        partial.replace(local_zip)

    return local_zip


def extract(country_zip: Path, dest_dir: Path) -> int:
    with zipfile.ZipFile(country_zip) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        print(f"Extracting {len(members)} files")
        archive.extractall(dest_dir)
    return len(members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="Japan")
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="keep the intermediate country zip so re-extraction needs no re-download",
    )
    args = parser.parse_args()

    country_zip = fetch_country_zip(args.country, args.dest)
    count = extract(country_zip, args.dest)
    if not args.keep_zip:
        country_zip.unlink()

    root = args.dest / args.country
    if not root.exists():
        root = args.dest / "RDD2022" / args.country
    print(f"Done: {count} files under {root}")
    print(f"Disk usage: {shutil.disk_usage(args.dest).free / 1e9:.1f} GB free")


if __name__ == "__main__":
    main()

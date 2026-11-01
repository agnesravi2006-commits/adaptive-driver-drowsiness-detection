"""UTA-RLDD dataset discovery, inspection, and optional download helpers.

Official labels (from the UTA-RLDD project page):
    0  = alert
    5  = low vigilant
    10 = drowsy

Expected layout after extracting the 10 public ZIP parts:

    dataset/raw/
        Fold1_part1/
            <subject>/
                0.mov | 0.mp4 | 0.avi
                5.mov
                10.mov
        Fold1_part2/
        ...
        Fold5_part2/

The Drive folder is publicly listed, but each ZIP is 7-13 GB. This module
never downloads those archives unless you pass --download-zip with an
explicit file name.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".MP4", ".MOV"}
LABEL_MAP = {
    0: "alert",
    5: "low_vigilant",
    10: "drowsy",
}
DRIVE_FOLDER_ID = "1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
OFFICIAL_PAGE = "https://sites.google.com/view/utarldd/home"

ZIP_PARTS = [
    "Fold1_part1.zip",
    "Fold1_part2.zip",
    "Fold2_part1.zip",
    "Fold2_part2.zip",
    "Fold3_part1.zip",
    "Fold3_part2.zip",
    "Fold4_part1.zip",
    "Fold4_part2.zip",
    "Fold5_part1.zip",
    "Fold5_part2.zip",
]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_raw_dir() -> Path:
    return project_root() / "dataset" / "raw"


def default_processed_dir() -> Path:
    return project_root() / "dataset" / "processed"


@dataclass
class VideoSample:
    path: str
    fold: str | None
    subject: str | None
    label_id: int | None
    label_name: str | None


def _infer_label(path: Path) -> tuple[int | None, str | None]:
    stem = path.stem
    for token in (stem, path.parent.name):
        if token.isdigit():
            value = int(token)
            if value in LABEL_MAP:
                return value, LABEL_MAP[value]
    for value, name in LABEL_MAP.items():
        if name in stem.lower() or name in path.parent.name.lower():
            return value, name
    return None, None


def _infer_fold(path: Path, raw_dir: Path) -> str | None:
    try:
        relative = path.relative_to(raw_dir)
    except ValueError:
        return None
    for part in relative.parts:
        if part.lower().startswith("fold"):
            return part
    return relative.parts[0] if relative.parts else None


def discover_videos(raw_dir: Path | None = None) -> list[VideoSample]:
    raw_dir = raw_dir or default_raw_dir()
    if not raw_dir.exists():
        return []
    samples: list[VideoSample] = []
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix not in VIDEO_EXTENSIONS:
            continue
        label_id, label_name = _infer_label(path)
        fold = _infer_fold(path, raw_dir)
        subject = None
        try:
            rel = path.relative_to(raw_dir)
            if len(rel.parts) >= 2:
                subject = rel.parts[-2]
        except ValueError:
            subject = path.parent.name
        samples.append(
            VideoSample(
                path=str(path),
                fold=fold,
                subject=subject,
                label_id=label_id,
                label_name=label_name,
            )
        )
    samples.sort(key=lambda s: s.path)
    return samples


def inspect_dataset(raw_dir: Path | None = None) -> dict:
    raw_dir = raw_dir or default_raw_dir()
    samples = discover_videos(raw_dir)
    counts: dict[str, int] = {}
    unlabeled = 0
    for sample in samples:
        if sample.label_name is None:
            unlabeled += 1
        else:
            counts[sample.label_name] = counts.get(sample.label_name, 0) + 1
    return {
        "official_page": OFFICIAL_PAGE,
        "drive_folder": DRIVE_FOLDER_URL,
        "raw_dir": str(raw_dir),
        "raw_dir_exists": raw_dir.exists(),
        "video_count": len(samples),
        "label_counts": counts,
        "unlabeled_videos": unlabeled,
        "zip_parts": ZIP_PARTS,
        "samples_preview": [asdict(s) for s in samples[:12]],
        "ready_for_training": len(samples) > 0 and unlabeled < len(samples),
    }


def print_download_instructions() -> str:
    return f"""
UTA-RLDD access
===============
Official page: {OFFICIAL_PAGE}
Google Drive folder (public listing): {DRIVE_FOLDER_URL}

The folder lists 10 ZIP files (about 7.5 GB to 13.4 GB each, 111 GB total).
Direct programmatic download of a 12 GB Drive file is possible with `gdown`
if the file stays public, but it is not started automatically.

Manual steps (recommended):
  1. Open the Drive folder while signed into Google.
  2. Start with the smallest archive: Fold3_part2.zip (~7.55 GB).
  3. Download it into:
       {default_raw_dir().parent}
  4. Extract it into:
       {default_raw_dir()}
  5. Run:
       python backend/dataset_loader.py --inspect

Optional Drive download of ONE zip (still several GB):
       python backend/dataset_loader.py --download-zip Fold3_part2.zip

Do not download all 10 archives unless you have enough disk space and a
stable connection.
""".strip()


def try_download_zip(file_name: str, dest_dir: Path | None = None) -> Path:
    if file_name not in ZIP_PARTS:
        raise ValueError(f"Unknown zip '{file_name}'. Choose one of: {', '.join(ZIP_PARTS)}")
    dest_dir = dest_dir or (project_root() / "dataset")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_name
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is not installed. Install it with `pip install gdown` "
            "or download the ZIP from Google Drive in your browser."
        ) from exc
    url = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
    print(f"Attempting folder-aware download of {file_name} from {url}")
    # gdown can fetch a named file from a public folder listing.
    gdown.download_folder(url=url, output=str(dest_dir), quiet=False, file_types=["zip"])
    if not dest.exists():
        raise FileNotFoundError(
            f"{dest} was not downloaded. The Drive file may require a browser login. "
            "Download it manually from the official folder."
        )
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or download UTA-RLDD assets")
    parser.add_argument("--inspect", action="store_true", help="Scan dataset/raw and print label stats")
    parser.add_argument("--download-info", action="store_true", help="Print manual download steps")
    parser.add_argument("--download-zip", metavar="NAME", help="Download one named ZIP via gdown")
    parser.add_argument("--raw-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.download_info or not any([args.inspect, args.download_zip]):
        print(print_download_instructions())
        if not args.inspect and not args.download_zip:
            return 0

    if args.download_zip:
        try:
            path = try_download_zip(args.download_zip)
            print(f"Downloaded: {path}")
        except Exception as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            print(print_download_instructions())
            return 1

    if args.inspect:
        info = inspect_dataset(args.raw_dir)
        print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

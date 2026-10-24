"""Copy all 1,000 brand photos from their multilevel subfolders into a single flat master folder.

Retains the exact descriptive filename (e.g. gucci_person_001.jpg, chanel_dress_025.jpg)
without renaming, preserving brand and bifurcation level identification.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("copy_to_master")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def copy_to_master(source_dir: str | Path, master_dir: str | Path) -> int:
    source_path = Path(source_dir).resolve()
    master_path = Path(master_dir).resolve()
    master_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Source directory: {source_path}")
    logger.info(f"Master folder: {master_path}")

    copied_count = 0
    all_files = sorted(
        p
        for p in source_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VALID_EXTENSIONS
        and not str(p).startswith(str(master_path))
        and "_temp" not in str(p)
    )

    logger.info(f"Found {len(all_files)} files to copy.")

    for file_path in all_files:
        dest_path = master_path / file_path.name
        shutil.copy2(file_path, dest_path)
        copied_count += 1

    logger.info(f"Successfully copied {copied_count} photos to master folder: {master_path}")
    return copied_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy all brand photos into a single master folder with the same name.")
    parser.add_argument(
        "--source-dir",
        default="datasetdownload/dataset_10_brands_multilevel",
        help="Source directory with brand subfolders",
    )
    parser.add_argument(
        "--master-dir",
        default="datasetdownload/master_folder",
        help="Destination master folder",
    )
    args = parser.parse_args()

    copy_to_master(args.source_dir, args.master_dir)

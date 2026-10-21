"""Scan faces across all photos in classification_results.csv using AuraFace."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
import pandas as pd

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "face_scan.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("face_scanner")

sys.path.insert(0, str(Path(__file__).parent))
import face_identity


def main() -> None:
    csv_path = Path("classification_results.csv")
    if not csv_path.is_file():
        logger.error("classification_results.csv not found!")
        return

    logger.info("Loading classification_results.csv...")
    df = pd.read_csv(csv_path)
    subset = df[["Photo", "Photo ID", "Source path"]].drop_duplicates()
    
    items = []
    for _, row in subset.iterrows():
        p_path = Path(str(row["Source path"]))
        if p_path.is_file():
            items.append((str(row["Photo"]), str(row["Photo ID"]), str(p_path)))

    logger.info("Found %d valid photos to check/scan for faces.", len(items))
    if not items:
        logger.warning("No valid photo files found.")
        return

    logger.info("Loading AuraFace analyzer...")
    t0 = time.time()
    analyzer = face_identity.load_face_analyzer()
    logger.info("Analyzer loaded in %.2f seconds.", time.time() - t0)

    def progress_callback(completed: int, total: int, photo_name: str, counts: dict, error: str) -> None:
        if completed % 25 == 0 or completed == total:
            logger.info("Progress: %d/%d photos (faces found: %d, matched: %d, new: %d)",
                        completed, total, counts["faces"], counts["matched"], counts["new"])
        if error:
            logger.error("Error processing %s: %s", photo_name, error)

    logger.info("Starting face scanning...")
    t1 = time.time()
    result = face_identity.process_face_photos(
        items,
        analyzer,
        force_rerun=False,
        progress_callback=progress_callback,
    )
    elapsed = time.time() - t1
    logger.info(
        "Face scanning finished in %.2f seconds! Summary: %d processed, %d faces, %d matched, %d new IDs, %d review, %d skipped, %d errors",
        elapsed,
        result["processed"],
        result["faces"],
        result["matched"],
        result["new"],
        result["review"],
        result["skipped"],
        len(result["errors"]),
    )
    if result["errors"]:
        for err in result["errors"]:
            logger.error("Face scan error: %s", err)


if __name__ == "__main__":
    main()

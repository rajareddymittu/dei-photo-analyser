"""Script to download and bifurcate a 1,000-photo dataset across 10 fashion brands.

Brands selected (5 Luxury, 5 Other fashion/lifestyle):
- Luxury: Gucci, Louis Vuitton, Prada, Chanel, Dior
- Others: Zara, Levi's, The North Face, Calvin Klein, Ralph Lauren

Bifurcation:
- Person-level (50 photos per brand): full person, outfit styling, candid street shots
- Dress-level (50 photos per brand): dresses, gowns, jackets, garments, apparel details
Total: 10 brands * 100 photos = 1,000 photos
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageOps, UnidentifiedImageError
from icrawler.builtin import BingImageCrawler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("brand_downloader")

# Suppress overly verbose icrawler debug logs
logging.getLogger("icrawler").setLevel(logging.WARNING)

# 10 Distinct Brands (5 Luxury, 5 Others)
BRANDS_CONFIG: list[dict[str, Any]] = [
    # --- 5 LUXURY BRANDS ---
    {
        "id": "gucci",
        "name": "Gucci",
        "tier": "luxury",
        "description": "Italian luxury fashion house known for eclectic, high-fashion ready-to-wear",
        "person_queries": [
            "person wearing gucci outfit street style candid",
            "model wearing gucci dress fashion street",
            "candid street photo woman wearing gucci outfit",
        ],
        "dress_queries": [
            "gucci dress runway garment fashion lookbook",
            "gucci evening gown dress garment detail",
            "gucci suit jacket dress collection editorial",
        ],
    },
    {
        "id": "louis_vuitton",
        "name": "Louis Vuitton",
        "tier": "luxury",
        "description": "French luxury fashion house renowned for high-end ready-to-wear and apparel",
        "person_queries": [
            "model wearing louis vuitton clothing runway street",
            "person wearing louis vuitton outfit street style candid",
            "celebrity wearing louis vuitton ready to wear street",
        ],
        "dress_queries": [
            "louis vuitton dress jacket garment fashion lookbook",
            "louis vuitton runway dress collection garment",
            "louis vuitton coat dress apparel editorial detail",
        ],
    },
    {
        "id": "prada",
        "name": "Prada",
        "tier": "luxury",
        "description": "Italian luxury fashion house known for minimalist modern elegance and clean tailoring",
        "person_queries": [
            "person wearing prada coat outfit city street style",
            "candid model wearing prada dress fashion week",
            "person wearing prada outfit milan street",
        ],
        "dress_queries": [
            "prada dress coat garment detail collection lookbook",
            "prada runway dress gown garment apparel",
            "prada skirt dress tailored garment fashion",
        ],
    },
    {
        "id": "chanel",
        "name": "Chanel",
        "tier": "luxury",
        "description": "French luxury fashion house famed for haute couture, tweed suits, and iconic dresses",
        "person_queries": [
            "street style woman wearing chanel suit outfit paris",
            "person wearing chanel tweed jacket dress candid street",
            "model wearing chanel outfit street style",
        ],
        "dress_queries": [
            "chanel tweed dress jacket garment haute couture",
            "chanel evening dress gown runway garment detail",
            "chanel haute couture dress garment lookbook",
        ],
    },
    {
        "id": "dior",
        "name": "Dior",
        "tier": "luxury",
        "description": "French luxury fashion house celebrated for haute couture gowns and modern elegance",
        "person_queries": [
            "fashion person wearing dior outfit street style paris",
            "candid model wearing christian dior dress street",
            "woman wearing dior outfit street candid",
        ],
        "dress_queries": [
            "dior evening gown dress garment runway couture",
            "christian dior dress garment collection lookbook",
            "dior haute couture dress garment apparel detail",
        ],
    },
    # --- 5 OTHER POPULAR / LIFESTYLE / CASUAL / OUTDOOR BRANDS ---
    {
        "id": "zara",
        "name": "Zara",
        "tier": "fast_fashion",
        "description": "Global fashion brand known for on-trend contemporary clothing and dresses",
        "person_queries": [
            "candid person wearing zara outfit street style",
            "woman wearing zara dress street fashion summer",
            "model wearing zara clothing street candid",
        ],
        "dress_queries": [
            "zara dress apparel garment collection lookbook",
            "zara maxi dress midi dress garment catalog",
            "zara linen dress suit garment fashion",
        ],
    },
    {
        "id": "levis",
        "name": "Levi's",
        "tier": "denim_heritage",
        "description": "Historic denim and casualwear brand known for jeans, denim jackets, and casual dresses",
        "person_queries": [
            "person wearing levis denim jacket jeans street style",
            "candid person wearing levis outfit city street",
            "woman wearing levis denim dress street style",
        ],
        "dress_queries": [
            "levis denim jacket dress garment collection lookbook",
            "levis trucker jacket jeans apparel garment",
            "levis denim shirt dress garment catalog",
        ],
    },
    {
        "id": "the_north_face",
        "name": "The North Face",
        "tier": "outdoor_lifestyle",
        "description": "Premier outdoor apparel brand famous for technical jackets, parkas, and fleeces",
        "person_queries": [
            "street photo person wearing the north face jacket outdoor",
            "candid person wearing the north face nuptse puffer city",
            "person wearing the north face fleece outfit outdoor",
        ],
        "dress_queries": [
            "the north face jacket parka coat garment technical",
            "the north face nuptse down jacket garment lookbook",
            "the north face winter coat garment catalog",
        ],
    },
    {
        "id": "calvin_klein",
        "name": "Calvin Klein",
        "tier": "contemporary_lifestyle",
        "description": "American minimalist brand known for slip dresses, modern casuals, and denim",
        "person_queries": [
            "person wearing calvin klein clothes street style candid",
            "model wearing calvin klein dress street",
            "candid person wearing calvin klein denim outfit",
        ],
        "dress_queries": [
            "calvin klein slip dress garment apparel collection",
            "calvin klein minimalist dress garment lookbook",
            "calvin klein blazer dress garment catalog",
        ],
    },
    {
        "id": "ralph_lauren",
        "name": "Ralph Lauren",
        "tier": "classic_heritage",
        "description": "Iconic American brand spanning preppy casualwear, polo dresses, and formal evening wear",
        "person_queries": [
            "candid person wearing ralph lauren sweater outfit street",
            "woman wearing ralph lauren polo dress street style",
            "person wearing ralph lauren blazer street style",
        ],
        "dress_queries": [
            "ralph lauren evening gown dress garment collection lookbook",
            "ralph lauren polo dress shirt dress garment catalog",
            "ralph lauren silk dress blazer garment fashion",
        ],
    },
]

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def compute_sha256(file_path: Path) -> str:
    """Compute sha256 checksum of file contents."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_and_sanitize_image(
    file_path: Path, min_dimension: int = 300
) -> Optional[tuple[int, int, str]]:
    """Verify image is valid, non-corrupt, and meets minimum dimensions.

    Returns (width, height, format) if valid, None if invalid.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            transposed = ImageOps.exif_transpose(img)
            w, h = transposed.size
            if w < min_dimension or h < min_dimension:
                return None
            fmt = img.format or "JPEG"
            return w, h, fmt
    except (OSError, UnidentifiedImageError, ValueError):
        return None


def crawl_category(
    queries: list[str],
    target_dir: Path,
    target_count: int,
    brand_prefix: str,
    level: str,
    threads: int = 4,
    min_size: tuple[int, int] = (500, 500),
) -> int:
    """Crawl images for a list of queries until target_count valid images are collected."""
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = target_dir / "_temp_crawl"
    temp_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    valid_count = 0

    # Scan existing valid files first (for resuming)
    for existing in target_dir.glob("*"):
        if existing.is_file() and existing.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            meta = validate_and_sanitize_image(existing, min_dimension=min_size[0])
            if meta:
                file_hash = compute_sha256(existing)
                if file_hash not in seen_hashes:
                    seen_hashes.add(file_hash)
                    valid_count += 1
                else:
                    existing.unlink(missing_ok=True)
            else:
                existing.unlink(missing_ok=True)

    if valid_count >= target_count:
        logger.info(
            f"  -> Already have {valid_count}/{target_count} valid images for {brand_prefix} [{level}]"
        )
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return valid_count

    logger.info(
        f"  -> Need {target_count - valid_count} more images for {brand_prefix} [{level}] (currently {valid_count})"
    )

    for query in queries:
        if valid_count >= target_count:
            break

        crawler = BingImageCrawler(
            downloader_threads=threads,
            storage={"root_dir": str(temp_dir)},
            log_level=logging.WARNING,
        )

        needed = (target_count - valid_count) + 15  # Buffer for invalid/duplicate images
        logger.info(f"     Crawling Bing for query: '{query}' (requesting up to {needed})...")

        try:
            crawler.crawl(
                keyword=query,
                filters=dict(type="photo", size="large"),
                max_num=needed,
                min_size=min_size,
            )
        except Exception as exc:
            logger.warning(f"     Crawl error for query '{query}': {exc}")

        # Process downloaded temp images
        for temp_file in sorted(temp_dir.glob("*")):
            if not temp_file.is_file():
                continue
            ext = temp_file.suffix.lower()
            if ext not in VALID_IMAGE_EXTENSIONS:
                temp_file.unlink(missing_ok=True)
                continue

            meta = validate_and_sanitize_image(temp_file, min_dimension=min_size[0])
            if not meta:
                temp_file.unlink(missing_ok=True)
                continue

            file_hash = compute_sha256(temp_file)
            if file_hash in seen_hashes:
                temp_file.unlink(missing_ok=True)
                continue

            # Move and standardize filename
            valid_count += 1
            seen_hashes.add(file_hash)
            dest_name = f"{brand_prefix}_{level}_{valid_count:03d}{ext}"
            dest_path = target_dir / dest_name
            shutil.move(str(temp_file), str(dest_path))

            if valid_count >= target_count:
                break

    # Clean up temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info(
        f"  -> Completed {brand_prefix} [{level}]: {valid_count}/{target_count} images downloaded."
    )
    return valid_count


def generate_manifest(base_dir: Path) -> tuple[Path, Path]:
    """Generate comprehensive CSV and JSON manifests for the downloaded dataset."""
    manifest_rows = []

    for brand in BRANDS_CONFIG:
        brand_id = brand["id"]
        brand_dir = base_dir / brand_id

        for level in ["person_level", "dress_level"]:
            level_dir = brand_dir / level
            if not level_dir.exists():
                continue

            files = sorted(
                f
                for f in level_dir.glob("*")
                if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
            )

            for file_path in files:
                meta = validate_and_sanitize_image(file_path, min_dimension=100)
                if not meta:
                    continue
                w, h, fmt = meta
                file_hash = compute_sha256(file_path)
                size_bytes = file_path.stat().st_size

                manifest_rows.append(
                    {
                        "brand_id": brand_id,
                        "brand_name": brand["name"],
                        "brand_tier": brand["tier"],
                        "bifurcation_level": level,
                        "file_name": file_path.name,
                        "relative_path": str(file_path.relative_to(base_dir)),
                        "width": w,
                        "height": h,
                        "aspect_ratio": round(w / h, 3),
                        "format": fmt,
                        "size_kb": round(size_bytes / 1024, 1),
                        "sha256": file_hash,
                    }
                )

    csv_path = base_dir / "dataset_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if manifest_rows:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    json_path = base_dir / "dataset_manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_images": len(manifest_rows),
                "brands_count": len(BRANDS_CONFIG),
                "levels": ["person_level", "dress_level"],
                "records": manifest_rows,
            },
            f,
            indent=2,
        )

    logger.info(f"Manifests generated: {csv_path} and {json_path} ({len(manifest_rows)} photos)")
    return csv_path, json_path


def run_pipeline(
    target_dir: str | Path = "datasetdownload/dataset_10_brands_multilevel",
    target_per_level: int = 50,
    threads: int = 4,
) -> None:
    """Run full download and bifurcation pipeline for all 10 brands."""
    base_dir = Path(target_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    logger.info("=" * 70)
    logger.info(f"STARTING 1,000 PHOTO DATASET PIPELINE (10 BRANDS)")
    logger.info(f"Target directory: {base_dir}")
    logger.info(
        f"Goal: {len(BRANDS_CONFIG)} brands x 2 levels x {target_per_level} photos = {len(BRANDS_CONFIG) * 2 * target_per_level} total photos"
    )
    logger.info("=" * 70)

    summary: dict[str, dict[str, int]] = {}

    for idx, brand in enumerate(BRANDS_CONFIG, 1):
        brand_id = brand["id"]
        brand_name = brand["name"]
        tier = brand["tier"]

        logger.info(f"\n[{idx}/10] Processing Brand: {brand_name} ({tier.upper()})")

        brand_dir = base_dir / brand_id
        summary[brand_name] = {}

        # 1. Person Level
        person_dir = brand_dir / "person_level"
        person_count = crawl_category(
            queries=brand["person_queries"],
            target_dir=person_dir,
            target_count=target_per_level,
            brand_prefix=brand_id,
            level="person",
            threads=threads,
        )
        summary[brand_name]["person_level"] = person_count

        # 2. Dress Level
        dress_dir = brand_dir / "dress_level"
        dress_count = crawl_category(
            queries=brand["dress_queries"],
            target_dir=dress_dir,
            target_count=target_per_level,
            brand_prefix=brand_id,
            level="dress",
            threads=threads,
        )
        summary[brand_name]["dress_level"] = dress_count

    elapsed = round(time.time() - start_time, 1)
    logger.info("\n" + "=" * 70)
    logger.info(f"DOWNLOAD AND BIFURCATION COMPLETED in {elapsed}s")
    logger.info("=" * 70)

    # Print summary table
    total_downloaded = 0
    print("\nSUMMARY TABLE:")
    print("-" * 65)
    print(f"{'Brand Name':<20} | {'Person Level':<15} | {'Dress Level':<15} | {'Total':<10}")
    print("-" * 65)
    for brand_name, counts in summary.items():
        p_c = counts.get("person_level", 0)
        d_c = counts.get("dress_level", 0)
        b_total = p_c + d_c
        total_downloaded += b_total
        print(f"{brand_name:<20} | {p_c:<15} | {d_c:<15} | {b_total:<10}")
    print("-" * 65)
    print(f"{'TOTAL':<20} | {'':<15} | {'':<15} | {total_downloaded:<10}")
    print("-" * 65)

    # Generate Manifests
    generate_manifest(base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and bifurcate fashion brand photos at person and dress levels."
    )
    parser.add_argument(
        "--output-dir",
        default="datasetdownload/dataset_10_brands_multilevel",
        help="Destination directory for the dataset",
    )
    parser.add_argument(
        "--per-level",
        type=int,
        default=50,
        help="Target photos per bifurcation level (default: 50 -> 100/brand -> 1000 total)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Crawler downloader threads per category",
    )
    args = parser.parse_args()

    run_pipeline(
        target_dir=args.output_dir,
        target_per_level=args.per_level,
        threads=args.threads,
    )

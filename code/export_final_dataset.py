"""Utility script to extract and copy all branded images from classification results.

Generates:
1. `final_data_set/`: Flat directory of all branded images named by their canonical Photo names.
2. `final_data_set/by_brand/`: Categorized subdirectories for each brand.
3. `final_data_set/manifest.csv`: Detailed dataset manifest.
4. `final_data_set/README.md`: Summary and statistics.
5. Symlink `final data set` pointing to `final_data_set`.
"""

import sys
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from analytics_assistant.brands import TARGET_BRANDS

CSV_PATH = ROOT / "classification_results.csv"
OUTPUT_DIR = ROOT / "final_data_set"
SYMLINK_PATH = ROOT / "final data set"


def export_branded_dataset() -> None:
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    if df.empty or "Brands" not in df.columns:
        print("Error: No data or Brands column found in classification results.")
        return

    # Filter strictly for currently configured target brands
    branded = df[df["Brands"].isin(TARGET_BRANDS)].copy()

    total_sightings = len(branded)
    unique_photos = branded["Photo"].nunique()
    print(f"Found {total_sightings} branded person sightings across {unique_photos} unique photos.")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SYMLINK_PATH.exists() or SYMLINK_PATH.is_symlink():
        SYMLINK_PATH.unlink()
    try:
        SYMLINK_PATH.symlink_to("final_data_set", target_is_directory=True)
    except Exception as exc:
        print(f"Notice regarding symlink: {exc}")

    copied_count = 0
    records = []
    brand_counts = {}

    for _, row in branded.iterrows():
        photo_name = str(row["Photo"])
        source_path = Path(str(row["Source path"]))
        brand = str(row["Brands"])

        if not source_path.is_file():
            print(f"Warning: Source file {source_path} does not exist.")
            continue

        dest_file = OUTPUT_DIR / photo_name
        if not dest_file.exists():
            shutil.copy2(source_path, dest_file)
            copied_count += 1
            brand_counts[brand] = brand_counts.get(brand, 0) + 1

        # Categorized by brand
        safe_brand = brand.replace("/", "_").replace(" ", "_").lower()
        brand_folder = OUTPUT_DIR / "by_brand" / safe_brand
        brand_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, brand_folder / photo_name)

        records.append({
            "Photo": photo_name,
            "Brand": brand,
            "Person": row.get("Person", ""),
            "Person bbox": row.get("Person bbox", ""),
            "Wardrobe Breakdown": row.get("Wardrobe Breakdown", ""),
            "Top (Upper)": row.get("Top (Upper)", ""),
            "Bottom (Pants)": row.get("Bottom (Pants)", ""),
            "Dress / One-Piece": row.get("Dress / One-Piece", ""),
            "Clothing types": row.get("Clothing types", ""),
            "Colors": row.get("Colors", ""),
            "Style / occasion": row.get("Style / occasion", ""),
            "Source path": str(source_path),
            "Final path": str(dest_file),
        })

    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(OUTPUT_DIR / "manifest.csv", index=False)

    # Generate README documentation
    brand_table_rows = [
        f"| **{b}** | {brand_counts.get(b, 0)} | {len(manifest_df[manifest_df['Brand'] == b])} |"
        for b in sorted(brand_counts.keys())
    ]
    brand_table = "\n".join(brand_table_rows)

    readme_content = f"""# Final Branded Dataset

Extracted from verified zero-shot and probe classification results.

## Overview
- **Total Unique Branded Photos**: {copied_count}
- **Total Branded Person Sightings**: {len(manifest_df)}
- **Destination Folder**: `final_data_set/` (also symlinked as `final data set/`)

## Distribution by Brand

| Brand | Unique Photos | Total Sightings |
| :--- | :---: | :---: |
{brand_table}

## Directory Structure
- `final_data_set/` : Flat collection of all {copied_count} branded photos with their canonical names.
- `final_data_set/by_brand/<brand_name>/` : Subdirectories sorted by brand.
- `final_data_set/manifest.csv` : Complete metadata including bounding boxes, wardrobe breakdowns, and clothing types.
"""
    (OUTPUT_DIR / "README.md").write_text(readme_content)
    print(f"Export complete: {copied_count} photos copied to {OUTPUT_DIR}")


if __name__ == "__main__":
    export_branded_dataset()

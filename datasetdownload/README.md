# Brand Dataset Download & Multilevel Bifurcation

This directory contains pipelines for downloading, validating, and bifurcating fashion brand photo datasets across luxury and contemporary brands.

## Structure & Pipeline

1. **`download_10_brands_multilevel.py`**:
   Downloads 1,000 photos across 10 distinct fashion brands (5 Luxury, 5 Others):
   - **5 Luxury Brands**: Gucci, Louis Vuitton, Prada, Chanel, Dior
   - **5 Other Brands**: Zara, Levi's, The North Face, Calvin Klein, Ralph Lauren
   
   **Two-Level Bifurcation**:
   - **Person Level** (50 photos per brand): Full-person outfits, street candid, model styling.
   - **Dress Level** (50 photos per brand): Dresses, gowns, jackets, garments, apparel closeups.
   
   **Quality Controls**:
   - Automatic image integrity validation using PIL
   - Corrupt / truncated file removal
   - SHA-256 duplicate image detection
   - Dimension check (minimum 500x500 resolution)
   - Auto-generates `dataset_manifest.csv` and `dataset_manifest.json`

2. **`bifurcate_crops.py`**:
   - Uses PyTorch Faster R-CNN person detector and body/clothing geometry to extract bounding crops at person and dress levels.
   - Saves crops into `person_crops/` and `dress_crops/` with a detailed CSV manifest.

3. **`copy_to_master.py`**:
   - Copies all 1,000 photos from brand subfolders into a single flat `master_folder/` preserving the original descriptive filenames (e.g. `gucci_person_001.jpg`, `chanel_dress_025.jpg`).

## Running the Pipeline

```bash
# Download 1000 images bifurcated at person and dress levels
.venv/bin/python datasetdownload/download_10_brands_multilevel.py --output-dir datasetdownload/dataset_10_brands_multilevel --per-level 50

# Copy all photos into a single flat master folder with the same filenames
.venv/bin/python datasetdownload/copy_to_master.py --source-dir datasetdownload/dataset_10_brands_multilevel --master-dir datasetdownload/master_folder

# Optional: Extract cropped person and garment patches
.venv/bin/python datasetdownload/bifurcate_crops.py --dataset-dir datasetdownload/dataset_10_brands_multilevel
```

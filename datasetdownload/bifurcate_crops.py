"""Bifurcation tool to extract person-level and dress-level crops from brand dataset photos.

Uses Faster R-CNN person detector and body/garment geometry to bifurcate each photo into:
- Person Crop: full body / upper-to-lower outfit crop
- Dress / Garment Crop: focused clothing / dress area crop
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bifurcate_crops")


def get_detector() -> tuple[Any, Any, torch.device]:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    detector = fasterrcnn_resnet50_fpn(weights=weights).eval()
    try:
        detector.to(device)
    except (RuntimeError, NotImplementedError):
        device = torch.device("cpu")
        detector.to(device)
    return detector, weights, device


def detect_people(
    image: Image.Image, detector: Any, weights: Any, device: torch.device, threshold: float = 0.6
) -> list[tuple[int, int, int, int]]:
    tensor = to_tensor(image.convert("RGB")).to(device)
    try:
        with torch.inference_mode():
            prediction = detector([tensor])[0]
    except (RuntimeError, NotImplementedError):
        device = torch.device("cpu")
        detector.to(device)
        tensor = tensor.to(device)
        with torch.inference_mode():
            prediction = detector([tensor])[0]

    labels = weights.meta["categories"]
    person_label = labels.index("person")
    boxes = prediction["boxes"].tolist()
    scores = prediction["scores"].tolist()
    classes = prediction["labels"].tolist()

    people = [
        tuple(int(round(v)) for v in box)
        for box, score, class_id in zip(boxes, scores, classes)
        if class_id == person_label and score >= threshold
    ]
    return sorted(people, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)


def extract_dress_crop(person_crop: Image.Image) -> Image.Image:
    """Extract clothing / torso / dress region from person crop (from chest to knees/thighs)."""
    w, h = person_crop.size
    # Focus on clothing region (approximately 20% down from head to 85% down)
    top = int(h * 0.18)
    bottom = int(h * 0.88)
    left = int(w * 0.05)
    right = int(w * 0.95)
    return person_crop.crop((left, top, right, bottom))


def process_dataset(
    dataset_dir: str | Path,
    output_crops_dir: str | Path,
    confidence: float = 0.6,
) -> None:
    dataset_path = Path(dataset_dir).resolve()
    out_path = Path(output_crops_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    person_crops_dir = out_path / "person_crops"
    dress_crops_dir = out_path / "dress_crops"
    person_crops_dir.mkdir(parents=True, exist_ok=True)
    dress_crops_dir.mkdir(parents=True, exist_ok=True)

    detector, weights, device = get_detector()
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    all_images = [
        p
        for p in dataset_path.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions and "_temp" not in str(p)
    ]
    logger.info(f"Found {len(all_images)} images to process for crop bifurcation.")

    processed_records = []

    for idx, img_path in enumerate(all_images, 1):
        rel_path = img_path.relative_to(dataset_path)
        parts = rel_path.parts
        brand = parts[0] if len(parts) > 0 else "unknown"
        level = parts[1] if len(parts) > 1 else "general"

        try:
            with Image.open(img_path) as src_img:
                img = ImageOps.exif_transpose(src_img).convert("RGB")
        except (OSError, UnidentifiedImageError):
            continue

        boxes = detect_people(img, detector, weights, device, threshold=confidence)

        if boxes:
            # Primary detected person
            bx = boxes[0]
            w, h = img.size
            pad_x = int((bx[2] - bx[0]) * 0.08)
            pad_y = int((bx[3] - bx[1]) * 0.04)
            p_crop = img.crop((
                max(0, bx[0] - pad_x),
                max(0, bx[1] - pad_y),
                min(w, bx[2] + pad_x),
                min(h, bx[3] + pad_y),
            ))
            d_crop = extract_dress_crop(p_crop)
        else:
            # Fallback when whole image is clothing / flat-lay
            p_crop = img
            d_crop = img

        stem = img_path.stem
        p_name = f"{brand}_{stem}_person.jpg"
        d_name = f"{brand}_{stem}_dress.jpg"

        p_dest = person_crops_dir / p_name
        d_dest = dress_crops_dir / d_name

        p_crop.save(p_dest, quality=90)
        d_crop.save(d_dest, quality=90)

        processed_records.append({
            "brand": brand,
            "source_level": level,
            "original_file": str(rel_path),
            "person_crop_file": p_name,
            "dress_crop_file": d_name,
            "people_detected": len(boxes),
        })

        if idx % 100 == 0 or idx == len(all_images):
            logger.info(f"Processed {idx}/{len(all_images)} images for bifurcation crops.")

    # Save manifest
    manifest_file = out_path / "crop_bifurcation_manifest.csv"
    if processed_records:
        with open(manifest_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(processed_records[0].keys()))
            writer.writeheader()
            writer.writerows(processed_records)
    logger.info(f"Done! Crops saved to {out_path} with manifest at {manifest_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract person and dress crops from brand photos.")
    parser.add_argument(
        "--dataset-dir",
        default="datasetdownload/dataset_10_brands_multilevel",
        help="Input brand dataset folder",
    )
    parser.add_argument(
        "--output-dir",
        default="datasetdownload/dataset_10_brands_multilevel/crops",
        help="Output folder for extracted crops",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Person detection confidence threshold",
    )
    args = parser.parse_args()

    process_dataset(args.dataset_dir, args.output_dir, args.confidence)

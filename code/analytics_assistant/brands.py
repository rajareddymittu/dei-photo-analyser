"""Brand definitions, categorization, dedicated classifier models, and analytics helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

TARGET_BRANDS: list[str] = [
    "Nike",
    "Puma",
    "Adidas",
    "Under Armour",
    "New Balance",
]

ALL_BRAND_LABELS: list[str] = [
    *TARGET_BRANDS,
    "Other / unbranded",
]

BRAND_COLORS: dict[str, str] = {
    "Nike": "#E74C3C",
    "Puma": "#E67E22",
    "Adidas": "#2980B9",
    "Under Armour": "#8E44AD",
    "New Balance": "#27AE60",
    "Other / unbranded": "#95A5A6",
}

WEIGHTS_PATH = Path(__file__).resolve().parent / "brand_detector_weights.pt"


class BrandClassifier:
    """Trained multi-head brand classifier ensemble for person and logo crops."""

    def __init__(self, checkpoint: dict[str, Any], device: torch.device) -> None:
        self.device = device
        self.classes: list[str] = checkpoint["classes"]
        self.target_brands: list[str] = checkpoint["target_brands"]
        self.multiclass_weight = checkpoint["multiclass_weight"].to(device)
        self.multiclass_bias = checkpoint["multiclass_bias"].to(device)
        self.binary_weight = checkpoint["binary_weight"].to(device)
        self.binary_bias = checkpoint["binary_bias"].to(device)

    def predict_crops(
        self,
        features: torch.Tensor,
        crop_counts: list[int],
        threshold: float = 0.55,
        text_features: torch.Tensor | None = None,
    ) -> list[str]:
        """Classify each person using multi-scale crop features with unbranded calibration.

        Args:
            features: Tensor of shape (total_crops, 512), normalized CLIP features.
            crop_counts: Number of crops per person (e.g. [3, 3, 3]).
            threshold: Confidence threshold for confirming a target brand.
            text_features: Optional Tensor of shape (6, 512) with zero-shot text prototypes.

        Returns:
            List of predicted brand strings per person.
        """
        with torch.no_grad():
            multi_logits = features @ self.multiclass_weight.T + self.multiclass_bias
            probe_probs = torch.softmax(multi_logits, dim=-1)

            if text_features is not None:
                zs_sims = features @ text_features.T
                zs_probs = torch.softmax(zs_sims * 25.0, dim=-1)
                # Geometric ensemble of probe logits and zero-shot priors
                combined_probs = (probe_probs * zs_probs) ** 0.5
                combined_probs = combined_probs / combined_probs.sum(dim=-1, keepdim=True)
            else:
                combined_probs = probe_probs

        predictions: list[str] = []
        offset = 0

        unbranded_idx = self.classes.index("Other / unbranded") if "Other / unbranded" in self.classes else 5

        for count in crop_counts:
            person_probs = combined_probs[offset : offset + count]
            full_unbranded = person_probs[0, unbranded_idx].item()
            offset += count

            best_brand = "Other / unbranded"
            best_confidence = 0.0

            # Evaluate crops: 0=Full body, 1=Upper torso/chest, 2=Footwear
            for c_idx in range(count):
                probs = person_probs[c_idx]
                c_unbranded = probs[unbranded_idx].item()

                # Get top candidate among target brands
                target_probs = [probs[self.classes.index(b)].item() for b in self.target_brands]
                max_target_prob = max(target_probs)
                top_class = self.target_brands[target_probs.index(max_target_prob)]

                # Calibrated criteria:
                # 1. Candidate brand probability must exceed unbranded baseline with margin
                # 2. Candidate brand must meet sensitivity threshold
                # 3. Footwear crops require extra confidence and cannot overturn a full-body unbranded outfit
                min_threshold = threshold + 0.15 if c_idx == 2 else threshold
                if c_idx == 2 and full_unbranded > 0.65:
                    continue

                if max_target_prob > c_unbranded + 0.05 and max_target_prob >= min_threshold:
                    if max_target_prob > best_confidence:
                        best_confidence = max_target_prob
                        best_brand = top_class

            predictions.append(best_brand)

        return predictions


_LOADED_CLASSIFIER: BrandClassifier | None = None


def load_brand_classifier(device: torch.device) -> BrandClassifier | None:
    """Load the trained brand detector weights onto the given device."""
    global _LOADED_CLASSIFIER
    if _LOADED_CLASSIFIER is not None and _LOADED_CLASSIFIER.device == device:
        return _LOADED_CLASSIFIER

    if not WEIGHTS_PATH.exists():
        return None

    try:
        ckpt = torch.load(WEIGHTS_PATH, map_location=device)
        _LOADED_CLASSIFIER = BrandClassifier(ckpt, device)
        return _LOADED_CLASSIFIER
    except Exception:
        return None


def generate_anatomical_crops(image: Image.Image) -> list[Image.Image]:
    """Generate multi-scale crops (full body, upper chest/torso, footwear) for logo detection."""
    w, h = image.size
    # 1. Full person crop
    crops = [image]

    # 2. Upper body / chest zoom (where 85% of shirt/hoodie/jacket logos reside)
    if h > 40 and w > 20:
        chest_crop = image.crop((
            int(w * 0.05),
            int(h * 0.08),
            int(w * 0.95),
            int(h * 0.60),
        ))
        crops.append(chest_crop)

    # 3. Lower body / footwear zoom (where sneaker logos and pant cuffs reside)
    if h > 40 and w > 20:
        shoes_crop = image.crop((
            int(w * 0.05),
            int(h * 0.65),
            int(w * 0.95),
            h,
        ))
        crops.append(shoes_crop)

    return crops


_BRAND_TEXT_CACHE: torch.Tensor | None = None


def get_brand_text_features(model: Any, device: torch.device) -> torch.Tensor:
    """Precompute and cache zero-shot text prototype embeddings for target brands and unbranded class."""
    global _BRAND_TEXT_CACHE
    if _BRAND_TEXT_CACHE is not None and _BRAND_TEXT_CACHE.device == device:
        return _BRAND_TEXT_CACHE

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    brand_prompts: dict[str, list[str]] = {
        "Nike": [
            "a photo of a person wearing Nike sportswear with the visible Nike swoosh logo",
            "Nike athletic apparel with iconic Nike swoosh",
            "Nike sneakers with the Nike swoosh logo",
        ],
        "Puma": [
            "a photo of a person wearing Puma athletic apparel with the leaping puma cat logo",
            "Puma brand sportswear with leaping puma logo or Puma text",
            "Puma sneakers with the leaping puma cat",
        ],
        "Adidas": [
            "a photo of a person wearing Adidas apparel with iconic three white stripes",
            "Adidas sportswear with the three stripes or trefoil logo",
            "Adidas athletic sneakers with three stripes",
        ],
        "Under Armour": [
            "a photo of a person wearing Under Armour athletic gear with interlocking UA logo",
            "Under Armour sportswear with visible UA logo",
            "Under Armour compression athletic clothing",
        ],
        "New Balance": [
            "a photo of a person wearing New Balance sneakers with prominent letter N logo",
            "New Balance shoes with bold N logo",
            "New Balance branded athletic apparel",
        ],
        "Other / unbranded": [
            "plain unbranded clothing without any commercial brand logo",
            "traditional cultural attire with no sportswear branding",
            "casual everyday clothes with no visible logos",
            "plain formal wear with no commercial logos or emblems",
            "casual clothing with no visible sportswear logo",
        ],
    }

    class_order = [*TARGET_BRANDS, "Other / unbranded"]
    embeds = []
    with torch.no_grad():
        for label in class_order:
            prompts = brand_prompts[label]
            tokens = tokenizer(prompts).to(device)
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean_f = feats.mean(dim=0, keepdim=True)
            embeds.append(mean_f / mean_f.norm(dim=-1, keepdim=True))
        _BRAND_TEXT_CACHE = torch.cat(embeds, dim=0)

    return _BRAND_TEXT_CACHE


def predict_brands(
    images: list[Image.Image],
    model: Any,
    preprocess: Any,
    device: torch.device,
    threshold: float = 0.55,
) -> list[str]:
    """High-accuracy brand detection using multi-crop zoom, zero-shot priors, and probe calibration."""
    classifier = load_brand_classifier(device)
    if classifier is None or not images:
        return ["Other / unbranded"] * len(images)

    text_features = get_brand_text_features(model, device)

    all_crops: list[Image.Image] = []
    crop_counts: list[int] = []

    for img in images:
        crops = generate_anatomical_crops(img)
        all_crops.extend(crops)
        crop_counts.append(len(crops))

    # Batch encode crops
    batch_tensors = [preprocess(crop) for crop in all_crops]
    all_features: list[torch.Tensor] = []
    batch_size = 16

    with torch.no_grad():
        for start in range(0, len(batch_tensors), batch_size):
            batch = torch.stack(batch_tensors[start : start + batch_size]).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_features.append(feats)

        feature_tensor = torch.cat(all_features, dim=0)
        return classifier.predict_crops(
            feature_tensor, crop_counts, threshold=threshold, text_features=text_features
        )


def _split_labels(series: pd.Series) -> list[list[str]]:
    return [
        [item.strip() for item in str(val).split(";") if item.strip()]
        for val in series.fillna("").astype(str)
    ]


def summarize_brands(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute brand counts, penetration, rankings, co-occurrences, and insights."""
    if frame.empty or "Brands" not in frame.columns:
        return {
            "has_brand_data": False,
            "target_brand_counts": {brand: 0 for brand in TARGET_BRANDS},
            "unbranded_count": 0,
            "total_brand_sightings": 0,
            "total_target_brand_sightings": 0,
            "branded_person_rows": 0,
            "branded_penetration_percent": 0.0,
            "unique_target_brands_detected": 0,
            "top_brand": None,
            "top_brand_count": 0,
            "rankings": [],
            "brand_co_occurrences": {},
            "insights": ["No brand classification rows are available yet."],
        }

    photo_col = frame.get("Photo ID", pd.Series(frame.index.astype(str), index=frame.index))
    brand_lists = _split_labels(frame["Brands"])
    all_brand_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter({b: 0 for b in TARGET_BRANDS})
    unbranded_count = 0
    branded_row_flags = []

    for labels in brand_lists:
        has_target = False
        for lbl in labels:
            all_brand_counts[lbl] += 1
            norm = lbl.lower()
            matched = False
            for target in TARGET_BRANDS:
                if norm == target.lower():
                    target_counts[target] += 1
                    has_target = True
                    matched = True
                    break
            if not matched and ("unbranded" in norm or "other" in norm):
                unbranded_count += 1
        branded_row_flags.append(has_target)

    total_target_sightings = sum(target_counts.values())
    total_brand_sightings = sum(all_brand_counts.values())
    branded_rows = sum(branded_row_flags)
    total_rows = len(frame)
    penetration = round((branded_rows / total_rows * 100), 1) if total_rows else 0.0

    detected_targets = [b for b, c in target_counts.items() if c > 0]
    ranked_targets = sorted(target_counts.items(), key=lambda item: item[1], reverse=True)
    top_brand, top_brand_count = ranked_targets[0] if ranked_targets and ranked_targets[0][1] > 0 else (None, 0)

    rankings = []
    for brand, count in ranked_targets:
        share = round((count / total_target_sightings * 100), 1) if total_target_sightings else 0.0
        rankings.append({
            "brand": brand,
            "count": count,
            "share_of_target_brands_percent": share,
        })

    # Co-occurrences with garments, colors, and styles
    co_occurrences: dict[str, dict[str, Any]] = {}
    garment_lists = _split_labels(frame["Clothing types"]) if "Clothing types" in frame.columns else [[]] * len(frame)
    color_lists = _split_labels(frame["Colors"]) if "Colors" in frame.columns else [[]] * len(frame)
    style_lists = _split_labels(frame["Style / occasion"]) if "Style / occasion" in frame.columns else [[]] * len(frame)

    for target in TARGET_BRANDS:
        brand_garments: Counter[str] = Counter()
        brand_colors: Counter[str] = Counter()
        brand_styles: Counter[str] = Counter()
        matching_photos: set[str] = set()

        for idx, (b_labels, photo_id) in enumerate(zip(brand_lists, photo_col)):
            if any(b.lower() == target.lower() for b in b_labels):
                matching_photos.add(str(photo_id))
                brand_garments.update(garment_lists[idx])
                brand_colors.update(color_lists[idx])
                brand_styles.update(style_lists[idx])

        co_occurrences[target] = {
            "detections": target_counts[target],
            "unique_photos": len(matching_photos),
            "top_garments": dict(brand_garments.most_common(5)),
            "top_colors": dict(brand_colors.most_common(5)),
            "top_styles": dict(brand_styles.most_common(5)),
        }

    # Generate insights summary bullets
    insights: list[str] = []
    if total_target_sightings > 0:
        insights.append(
            f"Detected {total_target_sightings} target brand sightings across {len(frame)} person rows "
            f"({penetration}% brand detection rate)."
        )
        if top_brand and top_brand_count > 0:
            top_share = round((top_brand_count / total_target_sightings * 100), 1)
            insights.append(
                f"**{top_brand}** leads with {top_brand_count} appearances "
                f"({top_share}% of target brand sightings)."
            )
        if len(ranked_targets) > 1 and ranked_targets[1][1] > 0:
            second_brand, second_count = ranked_targets[1]
            insights.append(
                f"**{second_brand}** is second with {second_count} appearances."
            )
        if top_brand and co_occurrences.get(top_brand, {}).get("top_garments"):
            top_g = list(co_occurrences[top_brand]["top_garments"].keys())[:2]
            insights.append(
                f"{top_brand} is most frequently paired with {', '.join(top_g)}."
            )
    else:
        insights.append(
            f"All {total_rows} evaluated person rows currently have unbranded or other brand labels."
        )

    return {
        "has_brand_data": True,
        "target_brand_counts": dict(target_counts),
        "unbranded_count": unbranded_count,
        "total_brand_sightings": total_brand_sightings,
        "total_target_brand_sightings": total_target_sightings,
        "branded_person_rows": branded_rows,
        "branded_penetration_percent": penetration,
        "unique_target_brands_detected": len(detected_targets),
        "top_brand": top_brand,
        "top_brand_count": top_brand_count,
        "rankings": rankings,
        "brand_co_occurrences": co_occurrences,
        "insights": insights,
    }

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

LUXURY_BRANDS: list[str] = [
    "Gucci",
    "Louis Vuitton",
    "Prada",
    "Chanel",
    "Dior",
]

SPORTSWEAR_BRANDS: list[str] = [
    "Nike",
    "Puma",
    "Adidas",
    "Under Armour",
    "New Balance",
]

LIFESTYLE_BRANDS: list[str] = [
    "Zara",
    "Levi's",
    "The North Face",
    "Calvin Klein",
    "Ralph Lauren",
]

TARGET_BRANDS: list[str] = [
    *LUXURY_BRANDS,
    *SPORTSWEAR_BRANDS,
    *LIFESTYLE_BRANDS,
]

ALL_BRAND_LABELS: list[str] = [
    *TARGET_BRANDS,
    "Other / unbranded",
]

BRAND_COLORS: dict[str, str] = {
    # Luxury
    "Gucci": "#10B981",          # Emerald Green
    "Louis Vuitton": "#B45309",  # Monogram Cognac / Amber
    "Prada": "#334155",          # Saffiano Slate Charcoal
    "Chanel": "#EC4899",         # Parisian Pink
    "Dior": "#6366F1",           # Dior Blue / Violet
    # Sportswear
    "Nike": "#E74C3C",           # Crimson Red
    "Puma": "#E67E22",           # Amber Orange
    "Adidas": "#2980B9",         # Athletic Cyan / Blue
    "Under Armour": "#8E44AD",   # Royal Violet Purple
    "New Balance": "#27AE60",    # Forest Emerald Green
    # Lifestyle & Others
    "Zara": "#0EA5E9",           # Sky Cyan
    "Levi's": "#DC2626",         # Denim Red Tab
    "The North Face": "#F59E0B", # Summit Yellow
    "Calvin Klein": "#64748B",   # Slate Gray
    "Ralph Lauren": "#1D4ED8",   # Classic Navy
    # Other / Unbranded
    "Other / unbranded": "#95A5A6",
}

WEIGHTS_PATH = Path(__file__).resolve().parent / "brand_detector_weights.pt"


class BrandClassifier:
    """Multi-head brand classifier ensemble for person and logo crops across all target brands."""

    def __init__(self, checkpoint: dict[str, Any] | None, device: torch.device) -> None:
        self.device = device
        self.all_classes: list[str] = ALL_BRAND_LABELS
        self.target_brands: list[str] = TARGET_BRANDS
        if checkpoint:
            self.probe_classes: list[str] = checkpoint.get("classes", [])
            self.multiclass_weight = checkpoint.get("multiclass_weight")
            if self.multiclass_weight is not None:
                self.multiclass_weight = self.multiclass_weight.to(device)
            self.multiclass_bias = checkpoint.get("multiclass_bias")
            if self.multiclass_bias is not None:
                self.multiclass_bias = self.multiclass_bias.to(device)
        else:
            self.probe_classes = []
            self.multiclass_weight = None
            self.multiclass_bias = None

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
            text_features: Optional Tensor of shape (len(ALL_BRAND_LABELS), 512) with zero-shot text prototypes.

        Returns:
            List of predicted brand strings per person.
        """
        all_labels = self.all_classes
        all_targets = self.target_brands
        unbranded_idx = all_labels.index("Other / unbranded")

        with torch.no_grad():
            if text_features is not None:
                zs_sims = features @ text_features.T
                combined_probs = torch.softmax(zs_sims * 25.0, dim=-1)
            else:
                combined_probs = None

            # Blend probe probabilities for legacy classes if trained weights exist
            if self.multiclass_weight is not None and combined_probs is not None and self.probe_classes:
                multi_logits = features @ self.multiclass_weight.T + self.multiclass_bias
                probe_probs = torch.softmax(multi_logits, dim=-1)
                for legacy_brand in self.probe_classes:
                    if legacy_brand in all_labels and legacy_brand in self.probe_classes:
                        l_idx = all_labels.index(legacy_brand)
                        p_idx = self.probe_classes.index(legacy_brand)
                        combined_probs[:, l_idx] = (combined_probs[:, l_idx] * probe_probs[:, p_idx]) ** 0.5
                combined_probs = combined_probs / combined_probs.sum(dim=-1, keepdim=True)

        predictions: list[str] = []
        offset = 0

        for count in crop_counts:
            person_probs = combined_probs[offset : offset + count] if combined_probs is not None else None
            offset += count

            best_brand = "Other / unbranded"
            best_confidence = 0.0

            if person_probs is None:
                predictions.append(best_brand)
                continue

            full_unbranded = person_probs[0, unbranded_idx].item()

            # Evaluate crops: 0=Full body, 1=Upper torso/chest, 2=Footwear
            for c_idx in range(count):
                probs = person_probs[c_idx]
                c_unbranded = probs[unbranded_idx].item()

                target_probs = [probs[all_labels.index(b)].item() for b in all_targets]
                max_target_prob = max(target_probs)
                top_class = all_targets[target_probs.index(max_target_prob)]

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


def load_brand_classifier(device: torch.device) -> BrandClassifier:
    """Load the trained brand detector weights onto the given device, or fall back to zero-shot."""
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
        # Luxury Brands
        "Gucci": [
            "a photo of a person wearing Gucci luxury designer outfit with signature Gucci styling or double G logo",
            "Gucci brand designer dress or high-fashion clothing",
            "model wearing Gucci ready-to-wear runway fashion",
        ],
        "Louis Vuitton": [
            "a photo of a person wearing Louis Vuitton luxury clothing with LV monogram or logo",
            "Louis Vuitton designer jacket or dress with visible LV emblem",
            "model wearing Louis Vuitton ready to wear fashion apparel",
        ],
        "Prada": [
            "a photo of a person wearing Prada luxury coat, dress, or outfit with triangle logo",
            "Prada designer minimalist luxury fashion apparel",
            "model wearing Prada tailored clothing or ready to wear",
        ],
        "Chanel": [
            "a photo of a person wearing Chanel haute couture tweed suit, jacket, or dress",
            "Chanel luxury designer fashion clothing with interlocking CC logo",
            "woman wearing Chanel iconic tweed outfit or evening dress",
        ],
        "Dior": [
            "a photo of a person wearing Christian Dior couture dress or outfit with CD emblem",
            "Dior luxury designer evening gown or tailored clothing",
            "model wearing Christian Dior high fashion apparel",
        ],
        # Sportswear Brands
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
        # Lifestyle & Heritage Brands
        "Zara": [
            "a photo of a person wearing Zara trendy contemporary street clothing or dress",
            "candid street photo of person wearing Zara stylish outfit",
            "Zara fast fashion everyday apparel",
        ],
        "Levi's": [
            "a photo of a person wearing Levi's denim jacket or jeans with red tab",
            "candid person wearing Levi's authentic denim clothing",
            "Levi's denim apparel and trucker jacket",
        ],
        "The North Face": [
            "a photo of a person wearing The North Face jacket with half dome logo",
            "The North Face nuptse puffer coat or fleece outerwear",
            "person wearing The North Face outdoor winter clothing",
        ],
        "Calvin Klein": [
            "a photo of a person wearing Calvin Klein minimalist clothing with CK logo",
            "model wearing Calvin Klein slip dress or denim apparel",
            "Calvin Klein casual contemporary clothes",
        ],
        "Ralph Lauren": [
            "a photo of a person wearing Ralph Lauren polo shirt or cable-knit sweater with polo player logo",
            "Ralph Lauren preppy blazer or formal evening dress",
            "model wearing Ralph Lauren collection classic clothing",
        ],
        # Unbranded / baseline
        "Other / unbranded": [
            "plain unbranded clothing without any commercial brand logo",
            "traditional cultural attire with no commercial branding",
            "casual everyday clothes with no visible logos",
            "plain formal wear with no commercial logos or emblems",
            "casual clothing with no visible sportswear or designer logo",
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
            "top_garments": dict(brand_garments.most_common(8)),
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

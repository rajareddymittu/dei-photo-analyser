"""Streamlit dashboard for zero-shot classification of clothing in photos.

Run from the project root with:
    python -m pip install -r requirements.txt
    streamlit run code/classifier.py

Dependencies: streamlit, torch, torchvision, open_clip_torch, pillow, pandas,
             insightface, onnxruntime, huggingface_hub
"""

from __future__ import annotations

import hashlib
import base64
import json
from datetime import datetime
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch
import open_clip
import face_identity
import altair as alt
from analytics_assistant import (
    ALL_BRAND_LABELS,
    BRAND_COLORS,
    TARGET_BRANDS,
    build_analytics_context,
    predict_brands,
    summarize_brands,
)
from analytics_assistant.ui import render_analytics_chat
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config(
    page_title="Wardrobe & Brand Intelligence",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "train"
RESULTS_FILE = ROOT / "classification_results.csv"
UPLOADED_IMAGE_DIR = ROOT / "classification_images"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
PAGE_SIZE = 100#size of the page for train photos
RESULTS_PAGE_SIZE = 100
CLIP_IMAGE_BATCH_SIZE = 16

# Categories are intentionally about visible clothing. The model is zero-shot,
# so treat results as suggestions and review uncertain labels.
ATTRIBUTES: dict[str, list[str]] = {
    "Clothing types": [
        "t-shirt", "polo shirt", "tank top", "camisole", "crop top", "blouse",
        "button-down shirt", "tunic", "sweater", "cardigan", "hoodie", "sweatshirt",
        "vest", "corset top", "bodysuit", "blazer", "suit jacket", "leather jacket",
        "denim jacket", "coat", "trench coat", "cape", "poncho", "shrug or bolero",
        "jeans", "trousers", "dress pants", "leggings", "shorts", "bermuda shorts",
        "mini skirt", "midi skirt", "maxi skirt", "pencil skirt", "pleated skirt",
        "wrap skirt", "a-line skirt", "a-line dress", "wrap dress", "shirt dress",
        "shift dress", "sheath dress", "bodycon dress", "fit-and-flare dress",
        "empire-waist dress", "slip dress", "sundress", "sweater dress", "cocktail dress",
        "evening gown", "ball gown", "wedding dress", "bridesmaid dress", "maxi dress",
        "midi dress", "mini dress", "jumpsuit", "romper", "playsuit", "pantsuit",
        "skirt suit", "saree", "lehenga", "salwar kameez", "kurta", "sherwani",
        "kaftan", "abaya", "kimono", "traditional robe", "overalls", "uniform",
        "swimwear", "athletic wear", "scarf or shawl", "other clothing",
    ],
    "Colors": [
        "black", "jet black", "charcoal", "gray", "silver", "white", "ivory", "off-white",
        "cream", "beige", "sand", "taupe", "tan", "camel", "khaki", "brown",
        "chocolate brown", "rust", "terracotta", "copper", "gold", "mustard yellow",
        "yellow", "lemon yellow", "orange", "burnt orange", "coral", "peach", "salmon",
        "red", "scarlet", "crimson", "burgundy", "maroon", "wine red", "pink", "blush pink",
        "rose pink", "hot pink", "fuchsia", "magenta", "purple", "violet", "lilac",
        "lavender", "indigo", "navy blue", "dark blue", "royal blue", "cobalt blue",
        "sky blue", "light blue", "teal", "turquoise", "aqua", "cyan", "mint green",
        "sage green", "olive green", "khaki green", "lime green", "emerald green",
        "forest green", "dark green", "bright green", "multicolored", "pastel colors",
        "metallic colors",
    ],
    "Pattern / detail": [
        "solid color", "color block", "floral print", "rose print", "tropical print",
        "botanical print", "polka dots", "pinstripes", "vertical stripes", "horizontal stripes",
        "chevron stripes", "gingham checks", "plaid", "tartan", "houndstooth", "windowpane checks",
        "geometric print", "abstract print", "animal print", "leopard print", "zebra print",
        "snake print", "camouflage", "paisley", "tie-dye", "ombre gradient", "polka dot mesh",
        "lace", "embroidered detail", "sequins", "beaded detail", "rhinestone embellishment",
        "applique detail", "brocade", "jacquard", "quilting", "denim texture", "ribbed texture",
        "pleated detail", "ruffles", "fringe", "cutout detail", "graphic print", "logo print",
        "text print", "patchwork", "pattern not visible",
    ],
    "Sleeve / neckline": [
        "sleeveless", "spaghetti straps", "thin straps", "halter neck", "strapless",
        "one shoulder", "cap sleeves", "short sleeves", "elbow-length sleeves",
        "three-quarter sleeves", "long sleeves", "bell sleeves", "puff sleeves", "bishop sleeves",
        "batwing sleeves", "off-the-shoulder neckline", "crew neckline", "v-neckline",
        "scoop neckline", "square neckline", "sweetheart neckline", "high neckline",
        "collared neckline", "asymmetric neckline", "neckline not visible",
    ],
    "Garment length": [
        "cropped length", "waist length", "hip length", "upper-thigh length", "mini length",
        "above-knee length", "knee length", "below-knee length", "midi length",
        "mid-calf length", "ankle length", "maxi length", "floor length", "length not visible",
    ],
    "Style / occasion": [
        "casual everyday style", "smart casual style", "business casual style", "office wear",
        "formal wear", "black-tie evening wear", "cocktail party wear", "party wear",
        "wedding guest outfit", "bridal wear", "bridesmaid outfit", "prom dress",
        "traditional cultural clothing", "festive celebration clothing", "religious ceremony clothing",
        "vacation resort wear", "beachwear", "streetwear", "vintage style", "bohemian style",
        "minimalist style", "romantic style", "preppy style", "athletic or activewear",
        "uniform or workwear", "occasion not clear",
    ],
    "Brands": ALL_BRAND_LABELS,
}

RESULT_COLUMNS = [
    "Photo", "Classified date", "Photo ID", "Source path", "Person bbox", "People in photo", "Person",
    *ATTRIBUTES.keys(), "Error",
]


@st.cache_resource(show_spinner="Loading the image model (first run may download weights)…")
def load_model() -> tuple[Any, Any, Any, torch.device]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer, device


@st.cache_resource(show_spinner="Loading the person detector…")
def load_person_detector() -> dict[str, Any]:
    # Torchvision's COCO detector has an explicit "person" class. Use MPS
    # when PyTorch exposes it, while retaining a CPU fallback for unsupported ops.
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
    return {"detector": detector, "weights": weights, "device": device}


@st.cache_data
def image_paths(folder: str) -> list[str]:
    root = Path(folder)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


@st.cache_data(max_entries=1)
def load_saved_results() -> pd.DataFrame:
    if not RESULTS_FILE.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    try:
        saved = pd.read_csv(RESULTS_FILE, dtype=str, keep_default_na=False)
        for column in RESULT_COLUMNS:
            if column not in saved:
                saved[column] = ""
        return saved[RESULT_COLUMNS]
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame(columns=RESULT_COLUMNS)


@st.cache_data(max_entries=5000)
def thumbnail_data_uri(source_path: str, modified_ns: int, size: int = 20) -> str:
    """Create a compact preview URI for result rows and the face library."""
    del modified_ns  # Included in the cache key so changed source files refresh.
    try:
        with Image.open(source_path) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
            image.thumbnail((size, size))
            canvas = Image.new("RGB", (size, size), "white")
            x = (size - image.width) // 2
            y = (size - image.height) // 2
            canvas.paste(image, (x, y))
            buffer = BytesIO()
            canvas.save(buffer, format="JPEG", quality=50, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except (OSError, UnidentifiedImageError, ValueError):
        return ""


def prepare_source(name: str, source: Any) -> tuple[str, str]:
    """Return a stable photo ID and durable path for a train or uploaded image."""
    if isinstance(source, (str, Path)):
        path = Path(source).resolve()
        identity = str(path)
        contents = path.read_bytes()
    else:
        identity = name
        contents = source.getvalue()
        suffix = Path(name).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            suffix = ".jpg"
        UPLOADED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        path = UPLOADED_IMAGE_DIR / f"{hashlib.sha256(contents).hexdigest()}{suffix}"
        if not path.exists():
            path.write_bytes(contents)
    content_hash = hashlib.sha256(contents).hexdigest()
    photo_id = hashlib.sha256(f"{identity}\0{content_hash}".encode("utf-8")).hexdigest()
    return photo_id, str(path)


@st.cache_resource
def attribute_text_features(_model: Any, _tokenizer: Any, device_name: str) -> dict[str, torch.Tensor]:
    device = torch.device(device_name)
    result: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for attribute, labels in ATTRIBUTES.items():
            if attribute == "Colors":
                prompts = [f"a photo of clothing in {label}." for label in labels]
            elif attribute == "Pattern / detail":
                prompts = [f"a photo of clothing with {label}." for label in labels]
            elif attribute == "Brands":
                prompts = [
                    "a photo of unbranded plain clothing with no logo."
                    if "unbranded" in label.lower() or "other" in label.lower()
                    else f"a photo of a person wearing {label} clothing or sportswear."
                    for label in labels
                ]
            else:
                prompts = [f"a photo of a person wearing {label}." for label in labels]
            tokens = _tokenizer(prompts).to(device)
            features = _model.encode_text(tokens)
            result[attribute] = features / features.norm(dim=-1, keepdim=True)
    return result


def classify_images(
    images: list[Image.Image],
    model: Any,
    preprocess: Any,
    text_features: dict[str, torch.Tensor],
    device: torch.device,
    max_labels: int,
    brand_threshold: float = 0.55,
) -> list[dict[str, str]]:
    """Classify person crops using zero-shot attributes and dedicated brand models."""
    results: list[dict[str, str]] = []
    # Dedicated high-accuracy multi-crop brand prediction
    brand_predictions = predict_brands(
        images, model, preprocess, device, threshold=brand_threshold
    )
    for start in range(0, len(images), CLIP_IMAGE_BATCH_SIZE):
        batch_images = [
            preprocess(ImageOps.exif_transpose(image).convert("RGB"))
            for image in images[start : start + CLIP_IMAGE_BATCH_SIZE]
        ]
        batch = torch.stack(batch_images).to(device)
        with torch.inference_mode():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            batch_results: list[dict[str, str]] = [{} for _ in batch_images]
            for attribute, labels in ATTRIBUTES.items():
                if attribute == "Brands":
                    continue
                scores = features @ text_features[attribute].T
                indices_by_image = scores.topk(min(max_labels, len(labels))).indices.cpu().tolist()
                for output, indices in zip(batch_results, indices_by_image):
                    output[attribute] = "; ".join(labels[index].capitalize() for index in indices)

            # Assign calibrated brand predictions
            for idx_in_batch, output in enumerate(batch_results):
                global_idx = start + idx_in_batch
                output["Brands"] = (
                    brand_predictions[global_idx]
                    if global_idx < len(brand_predictions)
                    else "Other / unbranded"
                )

            results.extend(batch_results)
    return results


def detect_people(image: Image.Image, threshold: float) -> list[tuple[int, int, int, int]]:
    runtime = load_person_detector()
    detector, weights, device = runtime["detector"], runtime["weights"], runtime["device"]
    tensor = to_tensor(image.convert("RGB")).to(device)
    try:
        with torch.inference_mode():
            prediction = detector([tensor])[0]
    except (RuntimeError, NotImplementedError):
        if device.type != "mps":
            raise
        # Some torchvision detection operators may not have an MPS kernel.
        # Move the cached detector once and continue on CPU without changing the model.
        device = torch.device("cpu")
        detector.to(device)
        runtime["device"] = device
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
    return sorted(people, key=lambda box: (box[1], box[0]))


def person_crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * 0.12)
    pad_y = int((y2 - y1) * 0.06)
    return image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)))


def read_image(source: Any) -> Image.Image:
    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    source.seek(0)
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def set_page_selection(paths: list[str], selector_key: str) -> None:
    """Apply the page-level selection toggle to each visible photo checkbox."""
    selected = bool(st.session_state.get(selector_key, False))
    for path in paths:
        st.session_state[f"train-photo-{path}"] = selected


def run_classification(
    items: list[tuple[str, Any]],
    max_labels: int,
    person_threshold: float,
    force_rerun: bool,
    brand_threshold: float = 0.55,
) -> None:
    if not items:
        st.info("Choose at least one photo to classify.")
        return

    saved = load_saved_results()
    prepared = []
    for name, source in items:
        photo_id, source_path = prepare_source(name, source)
        prepared.append((name, photo_id, source_path))

    selected_ids = {photo_id for _, photo_id, _ in prepared}
    saved_ids = set(saved["Photo ID"].astype(str))
    pending_ids = selected_ids if force_rerun else selected_ids - saved_ids
    pending = [item for item in prepared if item[1] in pending_ids]
    if not pending:
        st.session_state["classification_results"] = saved
        st.info("Loaded saved results. No photos were classified again. Check **Force rerun** to reprocess the selected photos.")
        return

    model, preprocess, tokenizer, device = load_model()
    text_features = attribute_text_features(model, tokenizer, str(device))
    rows: list[dict[str, str]] = []
    progress = st.progress(0, text="Finding people and classifying clothing…")
    for i, (name, photo_id, source_path) in enumerate(pending):
        classified_date = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            image = read_image(source_path)
            boxes = detect_people(image, person_threshold)
            crops = [person_crop(image, box) for box in boxes]
            if not crops:
                crops = [image]
            crop_labels = classify_images(
                crops,
                model,
                preprocess,
                text_features,
                device,
                max_labels,
                brand_threshold=brand_threshold,
            )
            for person_number, labels in enumerate(crop_labels, start=1):
                person_label = f"Person {person_number}" if boxes else "No person detected (whole photo)"
                box_label = ",".join(str(value) for value in boxes[person_number - 1]) if boxes else ""
                rows.append({
                    "Photo": name,
                    "Classified date": classified_date,
                    "Photo ID": photo_id,
                    "Source path": source_path,
                    "Person bbox": box_label,
                    "People in photo": str(len(boxes)),
                    "Person": person_label,
                    "Error": "",
                    **labels,
                })
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            rows.append({
                "Photo": name,
                "Classified date": classified_date,
                "Photo ID": photo_id,
                "Source path": source_path,
                "Person bbox": "",
                "People in photo": "",
                "Person": "",
                "Error": str(exc),
            })
        progress.progress((i + 1) / len(pending), text=f"Finding people and classifying clothing… {i + 1}/{len(pending)}")
    progress.empty()

    new_results = pd.DataFrame(rows)
    for column in RESULT_COLUMNS:
        if column not in new_results:
            new_results[column] = ""
    new_results = new_results[RESULT_COLUMNS]
    if force_rerun:
        saved = saved[~saved["Photo ID"].isin(selected_ids)]
    results = pd.concat([saved, new_results], ignore_index=True)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_FILE, index=False)
    load_saved_results.clear()
    st.session_state["classification_results"] = results
    st.success(f"Saved results for {len(pending)} photo(s) to `{RESULTS_FILE.name}`.")


@st.cache_data(max_entries=128, show_spinner=False)
def attribute_counts(data: pd.DataFrame, column: str) -> pd.Series:
    """Count semicolon-separated attribute labels across classified people."""
    if column not in data:
        return pd.Series(dtype="int64")
    labels = data[column].fillna("").astype(str).str.split(";").explode().str.strip()
    labels = labels[labels.ne("")]
    return labels.value_counts()


def configure_chart_theme(chart: alt.Chart, is_dark: bool = False) -> alt.Chart:
    text_color = "#F1F5F9" if is_dark else "#1E293B"
    subtitle_color = "#94A3B8" if is_dark else "#64748B"
    grid_color = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.06)"
    return chart.configure(
        background="transparent",
        view=alt.ViewConfig(stroke=None),
        title=alt.TitleConfig(
            color=text_color,
            fontSize=15,
            fontWeight=600,
            anchor="start",
            subtitleColor=subtitle_color,
        ),
        axis=alt.AxisConfig(
            labelColor=text_color,
            titleColor=text_color,
            gridColor=grid_color,
            domainColor=grid_color,
            tickColor=grid_color,
        ),
        legend=alt.LegendConfig(
            labelColor=text_color,
            titleColor=text_color,
        ),
    )


def analysis_pie(counts: pd.Series, title: str, key: str, is_dark: bool = False) -> None:
    """Render the leading categories as a compact donut chart."""
    if counts.empty:
        st.caption(f"No {title.lower()} available yet.")
        return
    top = counts.head(7)
    remainder = int(counts.iloc[7:].sum())
    chart_data = pd.DataFrame({"Category": top.index.astype(str), "Count": top.values})
    if remainder:
        chart_data.loc[len(chart_data)] = ["Other", remainder]
    chart = (
        alt.Chart(chart_data)
        .mark_arc(
            innerRadius=58,
            outerRadius=112,
            stroke="#0B0F19" if is_dark else "white",
            strokeWidth=2,
        )
        .encode(
            theta=alt.Theta("Count:Q", stack=True),
            color=alt.Color("Category:N", title=None, scale=alt.Scale(scheme="tableau10")),
            tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Count:Q", format=",")],
        )
        .properties(title=title, height=290)
    )
    chart = configure_chart_theme(chart, is_dark)
    st.altair_chart(chart, width="stretch", key=key)


def analysis_bar(counts: pd.Series, title: str, key: str, limit: int = 12, is_dark: bool = False) -> None:
    if counts.empty:
        st.caption(f"No {title.lower()} available yet.")
        return
    chart_data = pd.DataFrame({"Label": counts.head(limit).index.astype(str), "Count": counts.head(limit).values})
    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=6, color="#6366F1" if is_dark else "#4666D5")
        .encode(
            x=alt.X("Count:Q", title="Tag appearances"),
            y=alt.Y("Label:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("Label:N"), alt.Tooltip("Count:Q", format=",")],
        )
        .properties(title=title, height=max(250, min(420, len(chart_data) * 27)))
    )
    chart = configure_chart_theme(chart, is_dark)
    st.altair_chart(chart, width="stretch", key=key)


@st.cache_resource(show_spinner="Loading face detection and recognition models (first run downloads model files)…")
def load_face_analyzer() -> Any:
    return face_identity.load_face_analyzer()


def run_face_matching(items: list[tuple[str, Any]], force_rerun: bool = False) -> None:
    if not items:
        st.info("Choose at least one photo for face matching.")
        return
    prepared: list[tuple[str, str, str]] = []
    preparation_progress = st.progress(0, text=f"Preparing face scan for 0/{len(items)} photos…")
    for index, (name, source) in enumerate(items, start=1):
        try:
            photo_id, source_path = prepare_source(name, source)
            prepared.append((name, photo_id, source_path))
        except OSError as exc:
            st.warning(f"Could not prepare {name}: {exc}")
        preparation_progress.progress(
            index / len(items), text=f"Preparing face scan: {index}/{len(items)} photos…"
        )
    preparation_progress.empty()
    if not prepared:
        return
    try:
        with st.spinner("Loading face detector and recognition models…"):
            analyzer = load_face_analyzer()
    except Exception as exc:
        st.error(
            "Face matching could not start. Install the face dependencies listed at the top of "
            f"`code/classifier.py`, then retry. Details: {exc}"
        )
        return
    scan_progress = st.progress(0, text=f"Face scan: 0/{len(prepared)} photos completed…")

    def update_face_progress(
        completed: int,
        total: int,
        photo_name: str,
        counts: dict[str, int],
        error: str,
    ) -> None:
        status = f"Face scan: {completed}/{total} photos · {counts['faces']} faces found"
        if error:
            status += f" · issue: {Path(photo_name).name}"
        else:
            status += f" · {Path(photo_name).name}"
        scan_progress.progress(completed / total, text=status)

    try:
        result = face_identity.process_face_photos(
            prepared,
            analyzer,
            force_rerun=force_rerun,
            progress_callback=update_face_progress,
        )
    except Exception as exc:
        scan_progress.empty()
        st.error(f"Face scan stopped before completion. Details: {exc}")
        return
    scan_progress.empty()
    st.success(
        f"Face scan finished: {result['processed']} photo(s) processed, {result['faces']} face(s), "
        f"{result['matched']} matched, {result['new']} new IDs, {result['review']} needing review, "
        f"{result['skipped']} already scanned."
    )
    for error in result["errors"]:
        st.warning(error)


@st.dialog("Photo & Wardrobe Intelligence", width="large")
def show_photo_preview(name: str, image: Image.Image, people: list[tuple[str, Image.Image, dict[str, str]]]) -> None:
    st.caption(f"📁 Source: {name}")
    full_col, details_main_col = st.columns([1, 1.4])
    with full_col:
        st.image(image, caption="Full Photo", width="stretch")
        st.caption(f"Resolution: {image.width} × {image.height} px")

    with details_main_col:
        if not people:
            st.info("No person was detected; classified as whole-photo context.", icon=":material/info:")
        else:
            detected_count = sum(not label.startswith("No person detected") for label, _, _ in people)
            if detected_count:
                st.markdown(f"#### 👥 Detected People ({detected_count})")
            else:
                st.info("Whole-photo classification (no individual person box detected)")

            for person_label, crop, labels in people:
                with st.container(border=True):
                    photo_col, details_col = st.columns([1, 2])
                    with photo_col:
                        st.markdown(f"**{person_label}**")
                        st.image(crop, caption="Detection Crop", width="stretch")
                    with details_col:
                        # Brand Pill
                        brand_val = labels.get("Brand", labels.get("Brands", "Other / unbranded"))
                        if not brand_val or brand_val == "nan":
                            brand_val = "Other / unbranded"
                        brand_color = BRAND_COLORS.get(brand_val, "#64748B")
                        st.markdown(
                            f"<div style='display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:12px; "
                            f"background:{brand_color}1a; border:1px solid {brand_color}; font-weight:700; color:{brand_color}; font-size:0.85rem; margin-bottom:10px;'>"
                            f"<span style='width:8px; height:8px; border-radius:50%; background:{brand_color}; box-shadow:0 0 5px {brand_color};'></span>"
                            f"Brand: {brand_val}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        # Category Badges
                        cat_mappings = [
                            ("👔 Garments", labels.get("Clothing types", "")),
                            ("🎨 Colors", labels.get("Colors", "")),
                            ("🪡 Patterns", labels.get("Pattern / detail", "")),
                            ("✂️ Cut / Sleeve", labels.get("Sleeve / neckline", "")),
                            ("👗 Length", labels.get("Garment length", "")),
                            ("✨ Style", labels.get("Style / occasion", "")),
                        ]
                        for cat_icon, val in cat_mappings:
                            if val and str(val).strip() and str(val).lower() not in {"none", "nan"}:
                                items = [it.strip() for it in str(val).split(";") if it.strip()]
                                tags_html = "".join([f"<span class='fashion-tag'>{it}</span>" for it in items])
                                st.markdown(
                                    f"<div class='fashion-attr-row'>"
                                    f"<span class='fashion-cat-title'>{cat_icon}</span>"
                                    f"<div style='display:flex; flex-wrap:wrap; gap:4px;'>{tags_html}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )



def capture_photo_click() -> None:
    """Save the clicked table row for this rerun so its photo dialog can open."""
    _capture_clicked_row("photo_click")


def capture_preview_click() -> None:
    """Open a result from the preview action beside its thumbnail."""
    _capture_clicked_row("preview_click")


def _capture_clicked_row(key: str) -> None:
    click = st.session_state.get(key)
    row_index = getattr(click, "row", None)
    if row_index is None and hasattr(click, "get"):
        row_index = click.get("row")
    st.session_state["pending_photo_preview_row"] = row_index


def inject_custom_theme(is_dark: bool = False) -> None:
    """Inject modern, responsive theme CSS supporting light & dark modes with micro-animations."""
    if is_dark:
        bg_app = "#0B0F19"
        bg_sidebar = "#0F172A"
        text_primary = "#F8FAFC"
        text_secondary = "#94A3B8"
        border_subtle = "rgba(255, 255, 255, 0.08)"
        card_bg = "linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.75) 100%)"
        card_border = "rgba(255, 255, 255, 0.08)"
        card_shadow = "0 8px 30px rgba(0, 0, 0, 0.35)"
        hero_bg = "linear-gradient(135deg, #111827 0%, #1E1B4B 50%, #0F172A 100%)"
        hero_border = "rgba(99, 102, 241, 0.35)"
        hero_shadow = "0 12px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08)"
        hero_title_gradient = "linear-gradient(135deg, #FFFFFF 0%, #C7D2FE 50%, #818CF8 100%)"
        hero_badge_bg = "rgba(15, 23, 42, 0.85)"
        hero_badge_border = "rgba(99, 102, 241, 0.35)"
        hero_badge_num = "#818CF8"
        metric_bg = "linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%)"
        metric_border = "rgba(255, 255, 255, 0.08)"
        metric_accent = "#6366F1"
        metric_hover_border = "rgba(99, 102, 241, 0.5)"
        metric_hover_shadow = "0 10px 28px rgba(99, 102, 241, 0.22)"
        tab_color = "#CBD5E1"
        tab_active_bg = "linear-gradient(180deg, rgba(99, 102, 241, 0.2) 0%, rgba(99, 102, 241, 0.04) 100%)"
        tab_active_color = "#818CF8"
        tab_active_border = "#6366F1"
        input_bg = "#1E293B"
        input_color = "#F8FAFC"
        input_border = "rgba(255, 255, 255, 0.12)"
        badge_bg = "rgba(255, 255, 255, 0.08)"
        badge_border = "rgba(255, 255, 255, 0.14)"
    else:
        bg_app = "#F8FAFC"
        bg_sidebar = "#FFFFFF"
        text_primary = "#0F172A"
        text_secondary = "#64748B"
        border_subtle = "#E2E8F0"
        card_bg = "#FFFFFF"
        card_border = "#E2E8F0"
        card_shadow = "0 4px 20px -2px rgba(15, 23, 42, 0.05)"
        hero_bg = "linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 40%, #EEF2FF 100%)"
        hero_border = "#E0E7FF"
        hero_shadow = "0 10px 30px -5px rgba(79, 70, 229, 0.08), 0 4px 6px -2px rgba(15, 23, 42, 0.02)"
        hero_title_gradient = "linear-gradient(135deg, #0F172A 0%, #312E81 50%, #4F46E5 100%)"
        hero_badge_bg = "#FFFFFF"
        hero_badge_border = "#E2E8F0"
        hero_badge_num = "#4F46E5"
        metric_bg = "#FFFFFF"
        metric_border = "#E2E8F0"
        metric_accent = "#4F46E5"
        metric_hover_border = "#C7D2FE"
        metric_hover_shadow = "0 8px 24px rgba(79, 70, 229, 0.12)"
        tab_color = "#64748B"
        tab_active_bg = "#EEF2FF"
        tab_active_color = "#4F46E5"
        tab_active_border = "#4F46E5"
        input_bg = "#FFFFFF"
        input_color = "#0F172A"
        input_border = "#E2E8F0"
        badge_bg = "#F1F5F9"
        badge_border = "#E2E8F0"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Typography & Ambient Glow Background Mesh */
    html, body, [class*="css"], .stApp {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        {'background: radial-gradient(circle at 12% 12%, rgba(99, 102, 241, 0.15) 0%, transparent 40%), radial-gradient(circle at 88% 85%, rgba(168, 85, 247, 0.12) 0%, transparent 45%), radial-gradient(circle at 50% 50%, rgba(14, 165, 233, 0.05) 0%, transparent 60%), #0B0F19 !important;' if is_dark else 'background: radial-gradient(circle at 10% 12%, rgba(99, 102, 241, 0.06) 0%, transparent 45%), radial-gradient(circle at 90% 88%, rgba(139, 92, 246, 0.05) 0%, transparent 45%), #F8FAFC !important;'}
        color: {text_primary} !important;
    }}

    /* Smooth Micro-transitions */
    div, button, input, select {{
        transition: background-color 0.2s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    header[data-testid="stHeader"] {{
        background-color: transparent !important;
    }}

    /* Sidebar Styling */
    div[data-testid="stSidebar"] {{
        background-color: {bg_sidebar} !important;
        border-right: 1px solid {border_subtle} !important;
    }}
    div[data-testid="stSidebar"] * {{
        color: {text_primary};
    }}

    /* Hero Banner Card */
    .hero-banner-card {{
        background: {hero_bg};
        border: 1px solid {hero_border};
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 18px;
        box-shadow: {hero_shadow};
        position: relative;
        overflow: hidden;
    }}
    .hero-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 4px 10px;
        border-radius: 9999px;
        margin-bottom: 8px;
        background: {'rgba(99, 102, 241, 0.15)' if is_dark else 'rgba(79, 70, 229, 0.08)'};
        color: {'#A5B4FC' if is_dark else '#4F46E5'};
        border: 1px solid {'rgba(99, 102, 241, 0.3)' if is_dark else 'rgba(79, 70, 229, 0.2)'};
    }}
    .hero-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: {'#10B981' if is_dark else '#059669'};
        animation: hero-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }}
    @keyframes hero-pulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.4; transform: scale(0.85); }}
    }}
    .hero-banner-title {{
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: {hero_title_gradient};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .hero-banner-desc {{
        font-size: 0.92rem;
        color: {text_secondary};
        margin: 0;
        line-height: 1.5;
        max-width: 680px;
    }}
    .hero-stat-badge {{
        background: {hero_badge_bg};
        border: 1px solid {hero_badge_border};
        border-radius: 12px;
        padding: 12px 22px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, {'0.25' if is_dark else '0.04'});
        text-align: right;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .hero-stat-badge:hover {{
        transform: translateY(-2px);
    }}
    .hero-stat-num {{
        font-size: 1.5rem;
        font-weight: 800;
        color: {hero_badge_num};
        line-height: 1.1;
    }}
    .hero-stat-lbl {{
        font-size: 0.72rem;
        font-weight: 700;
        color: {text_secondary};
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    /* Luxury Metric Cards */
    div[data-testid="stMetric"] {{
        background: {metric_bg} !important;
        border: 1px solid {metric_border} !important;
        border-top: 3px solid {metric_accent} !important;
        border-radius: 16px !important;
        padding: 16px 22px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, {'0.25' if is_dark else '0.04'}) !important;
        backdrop-filter: blur(12px) !important;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease !important;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-4px) !important;
        border-color: {metric_hover_border} !important;
        box-shadow: {metric_hover_shadow} !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {text_primary} !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        letter-spacing: -0.02em !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {text_secondary} !important;
        font-weight: 700 !important;
        font-size: 0.80rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }}
    div[data-testid="stMetricDelta"] {{
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }}

    /* Floating Segmented Pill Island Tabs */
    div[data-baseweb="tab-list"],
    div[data-testid="stTabs"] [role="tablist"] {{
        display: inline-flex !important;
        gap: 6px !important;
        background: {'rgba(15, 23, 42, 0.85)' if is_dark else '#F1F5F9'} !important;
        border: 1px solid {border_subtle} !important;
        padding: 6px !important;
        border-radius: 14px !important;
        margin-bottom: 26px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, {'0.3' if is_dark else '0.04'}) !important;
        backdrop-filter: blur(12px) !important;
    }}
    div[data-baseweb="tab"],
    div[data-testid="stTab"],
    div[role="tab"] {{
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        padding: 8px 20px !important;
        color: {tab_color} !important;
        background: transparent !important;
        border: none !important;
        border-bottom: none !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    div[data-baseweb="tab"]:hover,
    div[data-testid="stTab"]:hover,
    div[role="tab"]:hover {{
        color: {text_primary} !important;
        background: {'rgba(255, 255, 255, 0.08)' if is_dark else '#E2E8F0'} !important;
    }}
    div[data-baseweb="tab"][aria-selected="true"],
    div[data-testid="stTab"][aria-selected="true"],
    div[role="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-bottom: none !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    }}

    /* Border Container Polish */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid {card_border} !important;
        border-radius: 16px !important;
        background: {card_bg} !important;
        box-shadow: {card_shadow} !important;
        backdrop-filter: blur(12px) !important;
    }}

    /* Primary & Secondary Buttons */
    button[kind="primary"] {{
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #4338CA 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease !important;
    }}
    button[kind="primary"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
        filter: brightness(1.06) !important;
    }}
    button[kind="primary"]:active {{
        transform: translateY(1px) !important;
    }}

    button[kind="secondary"] {{
        border-radius: 10px !important;
        font-weight: 600 !important;
        {'background: rgba(30, 41, 59, 0.8) !important; color: #F1F5F9 !important; border: 1px solid rgba(255, 255, 255, 0.12) !important;' if is_dark else ''}
    }}

    /* Inputs & Select Boxes */
    div[data-baseweb="select"] > div,
    input[type="text"],
    input[type="number"],
    div[data-baseweb="input"] {{
        background-color: {input_bg} !important;
        color: {input_color} !important;
        border-color: {input_border} !important;
        border-radius: 10px !important;
    }}

    {'div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] { background-color: #1E293B !important; color: #F8FAFC !important; border: 1px solid rgba(255, 255, 255, 0.12) !important; }' if is_dark else ''}
    {'li[role="option"] { color: #F8FAFC !important; } li[role="option"]:hover, li[role="option"][aria-selected="true"] { background-color: #334155 !important; }' if is_dark else ''}

    /* Transparent Altair & Vega charts */
    div[data-testid="stVegaLiteChart"],
    .vega-embed,
    div[data-testid="stVegaLiteChart"] svg,
    div[data-testid="stVegaLiteChart"] canvas {{
        background: transparent !important;
        background-color: transparent !important;
    }}

    /* Headings and Captions */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_primary} !important;
        font-weight: 700 !important;
    }}
    .stCaption, p:has(.stCaption) {{
        color: {text_secondary} !important;
    }}

    /* Brand Card Mini-Badges */
    .brand-pill-card {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 12px;
        border-radius: 10px;
        margin-bottom: 6px;
        background: {'rgba(30, 41, 59, 0.6)' if is_dark else '#F8FAFC'};
        border: 1px solid {'rgba(255, 255, 255, 0.08)' if is_dark else '#E2E8F0'};
        transition: transform 0.15s ease;
    }}
    .brand-pill-card:hover {{
        transform: translateX(3px);
    }}

    /* Fashion attributes and preview badges */
    .fashion-attr-row {{
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }}
    .fashion-cat-title {{
        font-weight: 600;
        min-width: 105px;
        color: {text_secondary};
        flex-shrink: 0;
        padding-top: 3px;
    }}
    .fashion-tag {{
        display: inline-block;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        background: {badge_bg};
        border: 1px solid {badge_border};
        color: {text_primary};
    }}

    /* Workflow Pipeline Steps */
    .workflow-steps-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }}
    .workflow-step-card {{
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 14px 16px;
        border-radius: 12px;
        background: {card_bg};
        border: 1px solid {card_border};
        box-shadow: {card_shadow};
    }}
    .workflow-step-num {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        flex-shrink: 0;
    }}
    .workflow-step-title {{
        font-weight: 700;
        font-size: 0.92rem;
        color: {text_primary};
        margin-bottom: 2px;
    }}
    .workflow-step-desc {{
        font-size: 0.8rem;
        color: {text_secondary};
        line-height: 1.3;
    }}

    /* Pills & Segmented Controls Polish */
    div[data-testid="stPills"] button,
    div[data-testid="stSegmentedControl"] button {{
        border-radius: 20px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.15s ease !important;
    }}
    div[data-testid="stPills"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {{
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
    }}

    /* File Uploader Polish */
    section[data-testid="stFileUploaderDropzone"] {{
        background: {input_bg} !important;
        border: 2px dashed {input_border} !important;
        border-radius: 12px !important;
        color: {text_primary} !important;
    }}
    section[data-testid="stFileUploaderDropzone"] span,
    section[data-testid="stFileUploaderDropzone"] p,
    section[data-testid="stFileUploaderDropzone"] small,
    div[data-testid="stFileUploaderDropzoneInstructions"] span {{
        color: {text_secondary} !important;
    }}
    section[data-testid="stFileUploaderDropzone"] button {{
        background: {'rgba(30, 41, 59, 0.9)' if is_dark else '#F1F5F9'} !important;
        color: {text_primary} !important;
        border: 1px solid {border_subtle} !important;
    }}

    /* Expander Polish */
    div[data-testid="stExpander"],
    div[data-testid="stExpander"] details,
    div[data-testid="stExpanderDetails"] {{
        border: 1px solid {card_border} !important;
        border-radius: 12px !important;
        background: {card_bg} !important;
    }}
    div[data-testid="stExpander"] summary {{
        background: transparent !important;
        color: {text_primary} !important;
    }}
    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary p {{
        color: {text_primary} !important;
        font-weight: 600 !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


is_dark = bool(st.session_state.get("is_dark_mode", False))
inject_custom_theme(is_dark)

if "classification_results" not in st.session_state:
    st.session_state["classification_results"] = load_saved_results()
saved_df = st.session_state["classification_results"]
saved_count = len(saved_df)
brand_counts_map = saved_df["Brands"].value_counts().to_dict() if "Brands" in saved_df.columns else {}

st.markdown(
    f"""
    <div class="hero-banner-card">
        <div>
            <div class="hero-chip"><span class="hero-dot"></span> Active Zero-Shot CV Engine</div>
            <h1 class="hero-banner-title">👔 Wardrobe & Brand Intelligence</h1>
            <p class="hero-banner-desc">Zero-shot clothing classification, sportswear brand recognition (Nike, Puma, Adidas, Under Armour, New Balance), and AI analytics.</p>
        </div>
        <div class="hero-stat-badge">
            <div class="hero-stat-num">{saved_count:,}</div>
            <div class="hero-stat-lbl">Classified Records</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### :material/palette: Appearance")
    st.toggle("🌙 Dark Mode", value=is_dark, key="is_dark_mode")
    st.divider()

    st.markdown("### :material/loyalty: Target Brands")
    st.caption("Active zero-shot & probe recognition:")
    for b in TARGET_BRANDS:
        color = BRAND_COLORS.get(b, "#4F46E5")
        b_count = brand_counts_map.get(b, 0)
        st.markdown(
            f"<div class='brand-pill-card'>"
            f"<div style='display:flex; align-items:center; gap:8px; font-size:0.9rem;'>"
            f"<span style='display:inline-block; width:10px; height:10px; border-radius:50%; background-color:{color}; box-shadow:0 0 6px {color};'></span>"
            f"<b>{b}</b></div>"
            f"<span style='font-size:0.75rem; font-weight:700; color:{color}; background:{color}18; border:1px solid {color}33; padding:2px 8px; border-radius:10px;'>{b_count:,} items</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.divider()

    st.subheader("Model settings", icon=":material/tune:")
    max_labels = st.slider("Maximum matches per attribute and person", min_value=1, max_value=6, value=3)
    person_threshold = st.slider("Person detection sensitivity", min_value=0.20, max_value=0.90, value=0.50, step=0.05)
    brand_threshold = st.slider(
        "Brand detection sensitivity",
        min_value=0.20,
        max_value=0.85,
        value=0.55,
        step=0.05,
        help="Confidence threshold for confirming Nike, Adidas, Puma, Under Armour, or New Balance. 0.55+ ensures unbranded clothing is not misclassified.",
    )
    st.metric("Database records", f"{saved_count:,}", border=True)

# Feature toggle for legacy tabs (hidden by default)
SHOW_EXPERIMENTAL_TABS = False

# Primary 4 tabs with Analysis as the default landing view
analysis_tab, results_tab, uploaded_tab, ai_analytics_tab = st.tabs(
    [
        ":material/insights: Analysis",
        ":material/table_chart: Results",
        ":material/cloud_upload: Upload & Classify",
        ":material/auto_awesome: AI Assistant",
    ]
)

with uploaded_tab:
    st.markdown("### :material/cloud_upload: Upload & Classify Photos")
    st.caption("Upload images to detect people, classify wardrobe attributes (clothing types, colors, styles), and recognize sportswear logos.")

    st.markdown(
        f"""
        <div class="workflow-steps-container">
            <div class="workflow-step-card">
                <div class="workflow-step-num">1</div>
                <div>
                    <div class="workflow-step-title">Select Photos</div>
                    <div class="workflow-step-desc">Upload local photos or test instantly with a curated sample image below.</div>
                </div>
            </div>
            <div class="workflow-step-card">
                <div class="workflow-step-num">2</div>
                <div>
                    <div class="workflow-step-title">Dual Vision AI</div>
                    <div class="workflow-step-desc">Faster R-CNN localizes people; OpenCLIP zero-shot classifies wardrobe & logos.</div>
                </div>
            </div>
            <div class="workflow-step-card">
                <div class="workflow-step-num">3</div>
                <div>
                    <div class="workflow-step-title">Intelligence & Export</div>
                    <div class="workflow-step-desc">Inspect visual badges, analyze market share charts, and export CSV reports.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        uploads = st.file_uploader(
            "Select one or more photos to process",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            help="Supported image formats: JPG, JPEG, PNG, WEBP, BMP.",
        )
        if uploads:
            st.info(f"📁 **{len(uploads)}** photo(s) selected and ready for analysis.", icon=":material/image:")
            with st.container(horizontal=True, vertical_alignment="center"):
                classify_uploads = st.button("✨ Run Wardrobe Classification", type="primary", icon=":material/play_arrow:")
                force_uploads = st.checkbox("Force rerun (re-classify existing records)", key="force-upload-rerun")
        else:
            classify_uploads = False
            force_uploads = False

    if TRAIN_DIR.is_dir():
        with st.expander("✨ Need an image? Try a sample photo from the gallery", expanded=not bool(uploads)):
            st.caption("Select a sample image from the dataset to test person detection and brand classification immediately:")
            sample_candidates = sorted(list(TRAIN_DIR.glob("*.jpg")))[:4]
            if sample_candidates:
                sample_cols = st.columns(len(sample_candidates))
                for idx, sample_path in enumerate(sample_candidates):
                    with sample_cols[idx]:
                        try:
                            with Image.open(sample_path) as s_img:
                                st.image(s_img, caption=f"Sample #{idx+1}", width="stretch")
                            if st.button(f"🎯 Test #{idx+1}", key=f"run_sample_{idx}", type="secondary"):
                                run_classification(
                                    [(sample_path.name, sample_path)],
                                    max_labels,
                                    person_threshold,
                                    True,
                                    brand_threshold,
                                )
                        except (OSError, UnidentifiedImageError):
                            st.caption("Sample unavailable")

    if classify_uploads:
        run_classification(
            [(item.name, item) for item in (uploads or [])],
            max_labels,
            person_threshold,
            force_uploads,
            brand_threshold,
        )

# Photos in train/ tab is hidden for now
if SHOW_EXPERIMENTAL_TABS:
    st.subheader("Photos in train/")
    if not TRAIN_DIR.is_dir():
        st.warning(f"Photo folder not found: {TRAIN_DIR}")
    else:
        paths = image_paths(str(TRAIN_DIR))
        st.write(f"Found **{len(paths):,}** photos in `train/`. Select photos a page at a time; up to {PAGE_SIZE} are shown per page.")
        if paths:
            page_count = (len(paths) + PAGE_SIZE - 1) // PAGE_SIZE
            page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1, key="train_page")
            start = (int(page) - 1) * PAGE_SIZE
            page_paths = paths[start : start + PAGE_SIZE]
            selector_key = f"select-page-{page}"
            select_page = st.checkbox(
                "Select all photos on this page",
                key=selector_key,
                on_change=set_page_selection,
                args=(page_paths, selector_key),
            )
            chosen: list[str] = []
            cols = st.columns(4)
            for index, path in enumerate(page_paths):
                with cols[index % 4]:
                    try:
                        with Image.open(path) as im:
                            st.image(im, width="stretch")
                        selected = st.checkbox(
                            Path(path).name,
                            value=select_page,
                            key=f"train-photo-{path}",
                        )
                        if selected:
                            chosen.append(path)
                    except (OSError, UnidentifiedImageError):
                        st.warning(f"Could not preview {Path(path).name}")
            st.caption(f"Selected on this page: {len(chosen)}")
            with st.container(horizontal=True, vertical_alignment="bottom"):
                classify_train = st.button("Classify selected train photos", type="primary", disabled=not chosen)
                force_train = st.checkbox("Force rerun", key="force-train-rerun")
                match_train_faces = st.button(
                    "Detect and match faces", disabled=not chosen, key="match-train-faces"
                )
                force_train_faces = st.checkbox("Force rerun faces", key="force-train-faces-rerun")
            if classify_train:
                run_classification(
                    [(Path(path).name, path) for path in chosen],
                    max_labels,
                    person_threshold,
                    force_train,
                    brand_threshold,
                )
            if match_train_faces:
                run_face_matching([(Path(path).name, path) for path in chosen], force_train_faces)

with analysis_tab:
    st.markdown("### :material/insights: Wardrobe & Brand Analytics Dashboard")
    st.caption("Comprehensive breakdown of detected clothing attributes, sportswear brand market share, and demographic distributions.")
    analysis_results = st.session_state["classification_results"]
    analysis_face_registry: dict[str, Any] = {"identities": [], "processed_photos": {}}
    analysis_face_occurrences = pd.DataFrame()
    try:
        analysis_face_registry, analysis_face_occurrences, _ = face_identity.load_face_data()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.warning(f"Face tracking data is unavailable: {exc}")
    if analysis_results.empty:
        st.info("No saved classifications yet. Classify some photos first to populate this page.", icon=":material/info:")
    else:
        valid_results = analysis_results[
            analysis_results.get("Error", pd.Series("", index=analysis_results.index)).fillna("").eq("")
        ].copy()
        attribute_columns = [column for column in ATTRIBUTES if column in valid_results.columns]

        # Early brand summary for executive KPI bar
        brand_stats = summarize_brands(valid_results)

        photo_count = valid_results["Photo ID"].nunique() if "Photo ID" in valid_results else 0
        person_rows = len(valid_results)
        people_per_photo = (
            valid_results.groupby("Photo ID").size()
            if "Photo ID" in valid_results
            else pd.Series(dtype="int64")
        )
        average_people = float(people_per_photo.mean()) if not people_per_photo.empty else 0.0

        top_b_label = f"{brand_stats['top_brand']}" if brand_stats.get("top_brand") else "None"
        top_b_delta = f"{brand_stats['top_brand_count']:,} sightings" if brand_stats.get("top_brand_count") else None
        with st.container(horizontal=True):
            st.metric("Photos analyzed", f"{photo_count:,}", delta="100% Ingested", delta_color="normal", border=True)
            st.metric("Person detections", f"{person_rows:,}", delta=f"{average_people:.1f} avg / photo", delta_color="normal", border=True)
            st.metric("Branded outfit rate", f"{brand_stats['branded_penetration_percent']:.1f}%", delta=f"{brand_stats.get('total_branded_sightings', 0):,} logo hits", delta_color="normal", help="Percentage of detected people with a target sportswear logo", border=True)
            st.metric("Leading brand", top_b_label, delta=top_b_delta, delta_color="normal", border=True)

        filter_options = ["All results", *attribute_columns]
        filter_col, tags_col = st.columns([1, 2])
        with filter_col:
            filter_attribute = st.selectbox(
                "Filter by category",
                filter_options,
                key="analysis-attribute-filter",
            )
        selected_tags: list[str] = []
        if filter_attribute != "All results":
            tag_counts = attribute_counts(valid_results, filter_attribute)
            with tags_col:
                selected_tags = st.multiselect(
                    f"Select specific {filter_attribute.lower()} to focus",
                    options=tag_counts.index.tolist(),
                    placeholder="All labels",
                    key="analysis-tag-filter",
                )
            if selected_tags:
                tag_sets = valid_results[filter_attribute].fillna("").astype(str).str.split(";")
                keep_rows = tag_sets.map(lambda labels: bool(set(selected_tags).intersection(label.strip() for label in labels)))
                valid_results = valid_results[keep_rows].copy()

        if valid_results.empty:
            st.info("No saved rows match those filters. Clear the selected labels to see all results.")
        else:
            pie_left, pie_right = st.columns(2)
            with pie_left:
                analysis_pie(attribute_counts(valid_results, "Clothing types"), "Garment mix", "analysis-garment-pie", is_dark=is_dark)
            with pie_right:
                analysis_pie(attribute_counts(valid_results, "Style / occasion"), "Style and occasion mix", "analysis-occasion-pie", is_dark=is_dark)

            color_col, pattern_col = st.columns(2)
            with color_col:
                analysis_bar(attribute_counts(valid_results, "Colors"), "Most seen colors", "analysis-colors", is_dark=is_dark)
            with pattern_col:
                analysis_bar(attribute_counts(valid_results, "Pattern / detail"), "Patterns and details", "analysis-patterns", is_dark=is_dark)

            people_chart_col, garment_table_col = st.columns(2)
            with people_chart_col:
                if not people_per_photo.empty:
                    people_distribution = (
                        people_per_photo.value_counts()
                        .sort_index()
                        .rename_axis("Detected people in photo")
                        .reset_index(name="Photos")
                    )
                    st.subheader("Photo group size")
                    st.bar_chart(people_distribution, x="Detected people in photo", y="Photos")
                    st.caption("Counts use classified person rows per photo; detections are not unique identity counts.")
            with garment_table_col:
                top_garments = attribute_counts(valid_results, "Clothing types").head(10)
                st.subheader("Top garment labels")
                if top_garments.empty:
                    st.caption("No garment labels available yet.")
                else:
                    st.dataframe(
                        top_garments.rename_axis("Garment").reset_index(name="Tag appearances"),
                        hide_index=True,
                        width="stretch",
                    )

            st.divider()
            st.subheader("Brand analytics", icon=":material/loyalty:")
            st.caption(
                "Zero-shot brand detection across target sportswear brands: Nike, Puma, Adidas, Under Armour, New Balance."
            )

            brand_stats = summarize_brands(valid_results)
            with st.container(horizontal=True):
                top_b_label = f"{brand_stats['top_brand']} ({brand_stats['top_brand_count']})" if brand_stats["top_brand"] else "None"
                st.metric("Top detected brand", top_b_label, border=True)
                st.metric(
                    "Branded detection rate",
                    f"{brand_stats['branded_penetration_percent']:.1f}%",
                    help="Percentage of detected person rows with at least one target brand detected",
                    border=True,
                )
                st.metric(
                    "Target brands seen",
                    f"{brand_stats['unique_target_brands_detected']} / {len(TARGET_BRANDS)}",
                    border=True,
                )
                st.metric(
                    "Total brand sightings",
                    f"{brand_stats['total_target_brand_sightings']:,}",
                    border=True,
                )

            if brand_stats.get("insights"):
                with st.container(border=True):
                    st.markdown("**Key brand insights**")
                    for insight in brand_stats["insights"]:
                        st.markdown(f"- {insight}")

            brand_col_pie, brand_col_bar = st.columns(2)
            with brand_col_pie:
                brand_counts_series = pd.Series(brand_stats["target_brand_counts"])
                if brand_stats["unbranded_count"] > 0:
                    brand_counts_series["Other / unbranded"] = brand_stats["unbranded_count"]
                if not brand_counts_series.empty and brand_counts_series.sum() > 0:
                    brand_chart_df = pd.DataFrame({
                        "Brand": brand_counts_series.index.astype(str),
                        "Count": brand_counts_series.values,
                    })
                    brand_pie = (
                        alt.Chart(brand_chart_df)
                        .mark_arc(innerRadius=58, outerRadius=112, stroke="#0B0F19" if is_dark else "white", strokeWidth=2)
                        .encode(
                            theta=alt.Theta("Count:Q", stack=True),
                            color=alt.Color(
                                "Brand:N",
                                title=None,
                                scale=alt.Scale(
                                    domain=list(BRAND_COLORS.keys()),
                                    range=list(BRAND_COLORS.values()),
                                ),
                            ),
                            tooltip=[alt.Tooltip("Brand:N"), alt.Tooltip("Count:Q", format=",")],
                        )
                        .properties(title="Brand distribution", height=290)
                    )
                    brand_pie = configure_chart_theme(brand_pie, is_dark=is_dark)
                    st.altair_chart(brand_pie, width="stretch", key="analysis-brand-pie")
                else:
                    st.caption("No brand sightings recorded yet.")

            with brand_col_bar:
                target_series = pd.Series(brand_stats["target_brand_counts"]).sort_values(ascending=False)
                if not target_series.empty and target_series.sum() > 0:
                    target_df = pd.DataFrame({
                        "Brand": target_series.index.astype(str),
                        "Count": target_series.values,
                    })
                    target_bar = (
                        alt.Chart(target_df)
                        .mark_bar(cornerRadiusEnd=6)
                        .encode(
                            x=alt.X("Count:Q", title="Brand sightings"),
                            y=alt.Y("Brand:N", sort="-x", title=None),
                            color=alt.Color(
                                "Brand:N",
                                legend=None,
                                scale=alt.Scale(
                                    domain=list(BRAND_COLORS.keys()),
                                    range=list(BRAND_COLORS.values()),
                                ),
                            ),
                            tooltip=[alt.Tooltip("Brand:N"), alt.Tooltip("Count:Q", format=",")],
                        )
                        .properties(title="Target brands comparison", height=290)
                    )
                    target_bar = configure_chart_theme(target_bar, is_dark=is_dark)
                    st.altair_chart(target_bar, width="stretch", key="analysis-brand-bar")
                else:
                    st.caption("No target brands recorded yet.")

            if brand_stats.get("brand_co_occurrences"):
                with st.container(border=True):
                    st.markdown("#### 🎯 Brand Deep Dive & Garment Synergy")
                    st.caption("Select a brand to inspect what garments and colorways it co-occurs with in the classified dataset:")
                    selected_inspect_brand = st.pills(
                        "Choose brand",
                        options=TARGET_BRANDS,
                        default=TARGET_BRANDS[0],
                        key="inspect-brand-pills",
                        label_visibility="collapsed",
                    )
                    if not selected_inspect_brand:
                        selected_inspect_brand = TARGET_BRANDS[0]
                    brand_info = brand_stats["brand_co_occurrences"].get(selected_inspect_brand, {})
                    brand_color = BRAND_COLORS.get(selected_inspect_brand, "#4F46E5")

                    b_det_col, b_gar_col, b_col_col = st.columns([1, 1.3, 1.3])
                    with b_det_col:
                        st.markdown(
                            f"<div style='display:inline-flex; align-items:center; gap:8px; padding:6px 14px; border-radius:20px; "
                            f"background:{brand_color}18; border:1px solid {brand_color}; font-weight:700; color:{brand_color}; margin-bottom:12px; font-size:0.9rem;'>"
                            f"<span style='width:10px; height:10px; border-radius:50%; background:{brand_color}; box-shadow:0 0 6px {brand_color};'></span>"
                            f"{selected_inspect_brand} Dossier"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.metric("Appearances", f"{brand_info.get('detections', 0):,}")
                        st.metric("Photos seen", f"{brand_info.get('unique_photos', 0):,}")
                        top_g_list = list(brand_info.get("top_garments", {}).keys())
                        top_c_list = list(brand_info.get("top_colors", {}).keys())
                        fav_g = top_g_list[0] if top_g_list else "garments"
                        fav_c = top_c_list[0] if top_c_list else "tones"
                        insight_bg = "rgba(255, 255, 255, 0.04)" if is_dark else "rgba(79, 70, 229, 0.05)"
                        insight_border = "rgba(255, 255, 255, 0.08)" if is_dark else "#E2E8F0"
                        st.markdown(
                            f"<div style='margin-top:12px; padding:10px 12px; border-radius:10px; background:{insight_bg}; border:1px solid {insight_border}; font-size:0.80rem; line-height:1.45;'>"
                            f"✨ <b>Signature Look:</b> Commonly styled with <b style='color:{brand_color};'>{fav_g.lower()}</b> in <b style='color:{brand_color};'>{fav_c.lower()}</b> colorways."
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with b_gar_col:
                        st.markdown("**Top Associated Garments**")
                        top_g = brand_info.get("top_garments", {})
                        if top_g:
                            max_g = max(top_g.values()) if top_g.values() else 1
                            for rank_idx, (g, cnt) in enumerate(top_g.items()):
                                pct = int((cnt / max_g) * 100)
                                bg_color = "rgba(255, 255, 255, 0.05)" if is_dark else "#F1F5F9"
                                border_col = "rgba(255, 255, 255, 0.08)" if is_dark else "#E2E8F0"
                                st.markdown(
                                    f"<div style='position:relative; overflow:hidden; padding:6px 12px; margin-bottom:5px; border-radius:8px; background:{bg_color}; font-size:0.86rem; border:1px solid {border_col};'>"
                                    f"<div style='position:absolute; left:0; top:0; bottom:0; width:{pct}%; background:{brand_color}22; pointer-events:none;'></div>"
                                    f"<div style='display:flex; justify-content:space-between; align-items:center; position:relative; z-index:1;'>"
                                    f"<span><span style='font-size:0.75rem; font-weight:700; opacity:0.6; margin-right:6px;'>#{rank_idx+1}</span>{g}</span><b style='color:{brand_color};'>{cnt}</b>"
                                    f"</div></div>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.caption("No garment associations yet.")
                    with b_col_col:
                        st.markdown("**Top Associated Colors**")
                        top_c = brand_info.get("top_colors", {})
                        if top_c:
                            max_c = max(top_c.values()) if top_c.values() else 1
                            for rank_idx, (c, cnt) in enumerate(top_c.items()):
                                pct = int((cnt / max_c) * 100)
                                bg_color = "rgba(255, 255, 255, 0.05)" if is_dark else "#F1F5F9"
                                border_col = "rgba(255, 255, 255, 0.08)" if is_dark else "#E2E8F0"
                                st.markdown(
                                    f"<div style='position:relative; overflow:hidden; padding:6px 12px; margin-bottom:5px; border-radius:8px; background:{bg_color}; font-size:0.86rem; border:1px solid {border_col};'>"
                                    f"<div style='position:absolute; left:0; top:0; bottom:0; width:{pct}%; background:#6366F122; pointer-events:none;'></div>"
                                    f"<div style='display:flex; justify-content:space-between; align-items:center; position:relative; z-index:1;'>"
                                    f"<span><span style='font-size:0.75rem; font-weight:700; opacity:0.6; margin-right:6px;'>#{rank_idx+1}</span>{c}</span><b style='color:#6366F1;'>{cnt}</b>"
                                    f"</div></div>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.caption("No color associations yet.")

        st.subheader("Face tracking snapshot")
        try:
            face_counts = (
                analysis_face_occurrences.groupby("Face ID")["Photo ID"].nunique()
                if not analysis_face_occurrences.empty
                else pd.Series(dtype="int64")
            )
            tracked_faces = len(analysis_face_registry.get("identities", []))
            repeat_faces = int((face_counts > 1).sum()) if not face_counts.empty else 0
            face_scan_photos = (
                analysis_face_occurrences["Photo ID"].nunique()
                if not analysis_face_occurrences.empty
                else len(analysis_face_registry.get("processed_photos", {}))
            )
            with st.container(horizontal=True):
                st.metric("Face IDs mapped", f"{tracked_faces:,}", border=True)
                st.metric("Photos face-scanned", f"{face_scan_photos:,}", border=True)
                st.metric("IDs seen in multiple photos", f"{repeat_faces:,}", border=True)
            if repeat_faces:
                repeat_rows = face_counts[face_counts > 1].sort_values(ascending=False).rename_axis("Face ID").reset_index(name="Photos seen")
                st.dataframe(repeat_rows, hide_index=True, width="stretch")
            else:
                st.caption("No face ID has appeared in multiple scanned photos yet.")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.warning(f"Face tracking data is unavailable: {exc}")

        with st.expander("More client-ready views we can add"):
            st.markdown(
                """
                - **Style profiles:** most common looks and garment combinations per recurring face ID.
                - **Color stories:** color palettes by garment type, occasion, or person, including co-occurrence matrices.
                - **Photo explorer:** filter the gallery by outfit, color, pattern, occasion, group size, or mapped identity.
                - **Coverage and review:** show unclassified photos, face matches needing review, and tag coverage.
                - **Collection comparisons:** compare selected folders, events, or date ranges once those sources are recorded.

                Classification dates are missing on some older rows, so date trends cover only dated rows. Event metadata and calibrated confidence scores are not stored yet; those can be added to the result schema for future analysis.
                """
            )

with ai_analytics_tab:
    st.title("AI analytics assistant", icon=":material/auto_awesome:")
    st.caption("Ask questions in plain language about clothing, sportswear brands, and face metrics across your library.")

    ai_results = st.session_state["classification_results"]
    ai_registry: dict[str, Any] = {"identities": [], "processed_photos": {}}
    ai_occurrences = pd.DataFrame()
    try:
        ai_registry, ai_occurrences, _ = face_identity.load_face_data()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.warning(f"Face tracking metrics are unavailable: {exc}")

    ai_attribute_columns = [column for column in ATTRIBUTES if column in ai_results.columns]
    ai_context = build_analytics_context(
        ai_results,
        ai_registry,
        ai_occurrences,
        ai_attribute_columns or list(ATTRIBUTES),
    )

    # Metric Context Summary Cards
    saved_rows = len(ai_results)
    unique_photos = ai_results["Photo ID"].nunique() if not ai_results.empty and "Photo ID" in ai_results.columns else 0
    brand_data = ai_context.get("brand_analytics", {})
    top_b = brand_data.get("top_brand") or "None"
    top_b_count = brand_data.get("top_brand_count", 0)
    tracked_count = len(ai_registry.get("identities", []))

    with st.container(horizontal=True):
        st.metric("Total person rows", f"{saved_rows:,}", border=True)
        st.metric("Classified photos", f"{unique_photos:,}", border=True)
        st.metric("Leading brand", f"{top_b}", f"{top_b_count} sightings" if top_b_count else None, border=True)
        st.metric("Tracked face IDs", f"{tracked_count:,}", border=True)

    with st.expander("Explore sample questions & assistant capabilities", icon=":material/help:"):
        st.markdown(
            """
            - **Brand Intelligence:** "Which brand leads in sightings?", "Compare Nike and Adidas in our photos", "What garments are most often paired with Puma?"
            - **Wardrobe Demographics:** "What are the most popular colors?", "What styles and patterns appear most frequently?"
            - **Face Demographics:** "How many distinct people were identified?", "Are there repeat individuals across multiple photos?"
            - **Penetration:** "What percentage of detected outfits have recognizable sportswear logos vs unbranded attire?"
            """
        )

    if ai_results.empty:
        st.info("There are no saved classifications yet. Classify photos in 'Upload photos' or 'Photos in train/' first, then ask questions about their metrics.", icon=":material/info:")

    render_analytics_chat(ai_context, is_dark=is_dark)

# Face library tab is hidden for now
if SHOW_EXPERIMENTAL_TABS:
    st.subheader("Face library")
    st.caption(
        "Face IDs stay stable when display names change. Matching thresholds and model settings "
        "are in `code/face_identity.py`."
    )
    st.caption(
        f"Current match threshold: {face_identity.FACE_MATCH_THRESHOLD_PERCENT:.1f}% scaled cosine score; "
        f"review margin: {face_identity.FACE_AMBIGUOUS_MARGIN_PERCENT:.1f} points."
    )
    try:
        face_registry, face_occurrences, _ = face_identity.load_face_data()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not read the face library: {exc}")
        face_registry, face_occurrences = {"identities": []}, pd.DataFrame()

    identities = face_registry.get("identities", [])
    if not identities:
        st.info("No faces are mapped yet. Select photos in Upload photos or Photos in train/, then choose Detect and match faces.")
        st.caption(f"Registry and tracking files will be stored in `{face_identity.FACE_DATA_DIR.name}/`.")
    else:
        photo_counts = (
            face_occurrences.groupby("Face ID")["Photo ID"].nunique().to_dict()
            if not face_occurrences.empty
            else {}
        )
        face_rows = []
        for identity in identities:
            crop_path = ROOT / identity.get("representative_image", "")
            try:
                preview = thumbnail_data_uri(str(crop_path), crop_path.stat().st_mtime_ns, 64) if crop_path.is_file() else ""
            except OSError:
                preview = ""
            face_rows.append({
                "Face preview": preview,
                "Face ID": identity["face_id"],
                "Display name": identity.get("display_name", identity["face_id"]),
                "Review status": identity.get("review_status", "matched"),
                "Possible match ID": identity.get("possible_match_id", ""),
                "Photos seen": int(photo_counts.get(identity["face_id"], 0)),
            })

        with st.form("face-library-name-form"):
            edited_faces = st.data_editor(
                pd.DataFrame(face_rows),
                width="stretch",
                hide_index=True,
                disabled=["Face preview", "Face ID", "Review status", "Possible match ID", "Photos seen"],
                column_config={
                    "Face preview": st.column_config.ImageColumn("Face", width="small"),
                    "Face ID": st.column_config.TextColumn("Stable face ID", width="large"),
                    "Display name": st.column_config.TextColumn("Name", required=True),
                    "Review status": st.column_config.TextColumn("Review status", width="small"),
                    "Possible match ID": st.column_config.TextColumn("Possible match", width="large"),
                    "Photos seen": st.column_config.NumberColumn("Photos", width="small", format="%d"),
                },
                key="face-library-editor",
            )
            save_names = st.form_submit_button("Save face names", type="primary")
        if save_names:
            names = dict(zip(edited_faces["Face ID"], edited_faces["Display name"].fillna("")))
            changed = face_identity.rename_face_identities(names)
            st.success(f"Saved {changed} name change(s) to `face_library_data/face_registry.json`.")
            st.rerun()

        name_by_id = {
            identity["face_id"]: identity.get("display_name", identity["face_id"])
            for identity in identities
        }
        selected_face_id = st.selectbox(
            "Inspect a mapped face",
            options=list(name_by_id),
            format_func=lambda face_id: f"{name_by_id[face_id]} ({face_id})",
            key="face-library-selection",
        )
        selected_identity = next(person for person in identities if person["face_id"] == selected_face_id)
        selected_crop = ROOT / selected_identity.get("representative_image", "")
        occurrence_rows = face_occurrences[face_occurrences["Face ID"] == selected_face_id].copy()
        preview_col, info_col = st.columns([1, 2])
        with preview_col:
            if selected_crop.is_file():
                st.image(str(selected_crop), caption=name_by_id[selected_face_id], width=240)
        with info_col:
            st.metric("Photos containing this face", int(photo_counts.get(selected_face_id, 0)))
            st.write(f"First seen: {selected_identity.get('created_at_utc', 'unknown')}")
            st.write(f"Last seen: {selected_identity.get('last_seen_utc', 'unknown')}")
            st.write(f"Review status: {selected_identity.get('review_status', 'matched')}")
        candidate_id = selected_identity.get("possible_match_id", "")
        if selected_identity.get("review_status") == "needs_review" and candidate_id in name_by_id:
            st.warning(
                f"Possible duplicate of **{name_by_id[candidate_id]}** "
                f"({candidate_id}). Compare the face previews above and in the table before merging."
            )
            if st.button(
                f"Confirm same person and merge into {name_by_id[candidate_id]}",
                key=f"merge-face-{selected_face_id}",
                type="primary",
            ):
                face_identity.merge_face_identities(selected_face_id, candidate_id)
                st.success("Merged appearances and face templates into the selected identity.")
                st.rerun()
        st.subheader("Photo appearances")
        if occurrence_rows.empty:
            st.info("No photo appearances are recorded for this face yet.")
        else:
            st.dataframe(
                occurrence_rows.drop(columns=["Photo ID", "Photo path"], errors="ignore"),
                width="stretch",
                hide_index=True,
            )
        st.caption(
            "Files: `face_library_data/face_registry.json` (editable names), "
            "`face_embeddings.npz` (matching templates), and `face_occurrences.csv` (photo tracking)."
        )

with results_tab:
    st.markdown("### :material/table_chart: Classification Records & Previews")
    if "classification_results" in st.session_state:
        results = st.session_state["classification_results"]
        if results.empty:
            st.info("No saved classifications yet. Classify photos in 'Upload & Classify' first to populate this table.", icon=":material/info:")
        else:
            pill_options = ["All", "Nike", "Adidas", "New Balance", "Puma", "Under Armour", "Other / unbranded"]
            selected_brand_pill = st.pills(
                "Brand Filter",
                pill_options,
                default="All",
                key="results_brand_pill",
            )

            search_col, page_col = st.columns([3, 1])
            with search_col:
                search_query = st.text_input(
                    "Search records",
                    placeholder="🔍 Filter records by brand, garment, color, or photo name...",
                    key="results_quick_search",
                    label_visibility="collapsed",
                )

            filtered_results = results
            if selected_brand_pill and selected_brand_pill != "All":
                brand_col = "Brands" if "Brands" in results.columns else ("Brand" if "Brand" in results.columns else None)
                if brand_col:
                    filtered_results = filtered_results[filtered_results[brand_col].astype(str).str.lower() == selected_brand_pill.lower()]

            if search_query:
                mask = pd.Series(False, index=filtered_results.index)
                search_cols = ["Photo", "Person", *ATTRIBUTES.keys(), "Brand", "Brands"]
                for col in search_cols:
                    if col in filtered_results.columns:
                        mask |= filtered_results[col].astype(str).str.contains(search_query, case=False, na=False)
                filtered_results = filtered_results[mask]

            active_filters = []
            if selected_brand_pill and selected_brand_pill != "All":
                active_filters.append(f"Brand: **{selected_brand_pill}**")
            if search_query:
                active_filters.append(f"Search: `\"{search_query}\"`")
            filter_summary = " + ".join(active_filters) if active_filters else "All records"
            st.caption(f"Showing **{len(filtered_results):,}** record(s) matching ({filter_summary}) • Click **Open preview** or photo name to inspect details.")

            results_page_count = max(1, ceil(len(filtered_results) / RESULTS_PAGE_SIZE))
            if st.session_state.get("results-page", 1) > results_page_count:
                st.session_state["results-page"] = 1
            with page_col:
                results_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=results_page_count,
                    value=1,
                    step=1,
                    key="results-page",
                )
            results_start = (int(results_page) - 1) * RESULTS_PAGE_SIZE
            visible_results = filtered_results.iloc[results_start : results_start + RESULTS_PAGE_SIZE].copy().reset_index(drop=True)
            thumbnails = []
            for source_path in visible_results["Source path"]:
                try:
                    image_path = Path(str(source_path))
                    thumbnails.append(
                        thumbnail_data_uri(str(image_path), image_path.stat().st_mtime_ns, 20)
                        if image_path.is_file()
                        else ""
                    )
                except OSError:
                    thumbnails.append("")
            if not visible_results.empty:
                quick_prev_col1, quick_prev_col2 = st.columns([3, 1])
                with quick_prev_col1:
                    inspect_options = list(range(len(visible_results)))
                    selected_inspect_idx = st.selectbox(
                        "Quick Photo Inspector",
                        options=inspect_options,
                        format_func=lambda i: f"#{i+1}: {visible_results.iloc[i]['Photo']} • {visible_results.iloc[i].get('Brands', 'Unbranded')}",
                        key="quick_inspect_record_select",
                        label_visibility="collapsed",
                    )
                with quick_prev_col2:
                    open_inspected = st.button("🔍 Open Detailed Preview", type="secondary", icon=":material/visibility:")
            else:
                open_inspected = False
                selected_inspect_idx = None

            st.dataframe(
                visible_results,
                width="stretch",
                hide_index=True,
                column_config={
                    "Preview": st.column_config.ImageColumn("Preview", width="small"),
                    "Open preview": st.column_config.ButtonColumn(
                        "Open preview",
                        width="small",
                        on_click=capture_preview_click,
                        key="preview_click",
                        type="secondary",
                    ),
                    "Photo": st.column_config.ButtonColumn(
                        "Photo (click to preview)",
                        width="large",
                        on_click=capture_photo_click,
                        key="photo_click",
                        type="tertiary",
                    ),
                    "Photo ID": None,
                    "Source path": None,
                    "Person bbox": None,
                },
            )
            row_index = st.session_state.pop("pending_photo_preview_row", None)
            if open_inspected and selected_inspect_idx is not None:
                row_index = selected_inspect_idx
            if row_index is not None and 0 <= row_index < len(visible_results):
                selected_row = visible_results.iloc[row_index]
                source_path = Path(str(selected_row.get("Source path", "")))
                if source_path.is_file():
                    try:
                        image = read_image(source_path)
                        photo_id = str(selected_row.get("Photo ID", ""))
                        photo_rows = results[results["Photo ID"] == photo_id]
                        people = []
                        for _, person_row in photo_rows.iterrows():
                            person_label = str(person_row.get("Person", ""))
                            if not person_label:
                                continue
                            box_text = str(person_row.get("Person bbox", ""))
                            box = tuple(int(value) for value in box_text.split(",")) if box_text else None
                            crop = person_crop(image, box) if box and len(box) == 4 else image
                            labels = {key: str(person_row.get(key, "")) for key in ATTRIBUTES}
                            if "Brands" in person_row and str(person_row["Brands"]).strip():
                                labels["Brand"] = str(person_row["Brands"])
                            elif "Brand" in person_row and str(person_row["Brand"]).strip():
                                labels["Brand"] = str(person_row["Brand"])
                            people.append((person_label, crop, labels))
                        show_photo_preview(
                            str(selected_row["Photo"]),
                            image,
                            people,
                        )
                    except (OSError, UnidentifiedImageError, ValueError):
                        st.warning("Could not open the original photo for this result.")
                else:
                    st.warning("The original photo file for this result is no longer available.")
            st.download_button(
                f"📥 Download Filtered Records as CSV ({len(filtered_results):,} rows)",
                data=filtered_results.drop(columns=["Photo ID", "Source path", "Person bbox"], errors="ignore").to_csv(index=False).encode("utf-8"),
                file_name="wardrobe_classification_export.csv",
                mime="text/csv",
                type="secondary",
            )
    else:
        st.info("No saved classifications yet.")

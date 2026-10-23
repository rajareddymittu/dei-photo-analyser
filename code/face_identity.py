"""Persistent local face matching for the photo classifier.

The thresholds below are the main tuning controls. Match scores are scaled
cosine similarities, not calibrated probabilities.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from error_logger import log_error, log_warning



ROOT = Path(__file__).resolve().parent.parent
FACE_DATA_DIR = ROOT / "face_library_data"
FACE_MODEL_DIR = FACE_DATA_DIR / "models" / "auraface"
FACE_CROPS_DIR = FACE_DATA_DIR / "faces"
REGISTRY_FILE = FACE_DATA_DIR / "face_registry.json"
EMBEDDINGS_FILE = FACE_DATA_DIR / "face_embeddings.npz"
OCCURRENCES_FILE = FACE_DATA_DIR / "face_occurrences.csv"
MODEL_REPO = "fal/AuraFace-v1"
MODEL_REVISION = "af6d057c9b0ec4071d4c49c80e3539258798b609"

# Accuracy controls: tune against representative photos from this collection.
# Percent is cosine similarity mapped from [-1, 1] to [0, 100], not probability.
FACE_MATCH_THRESHOLD_PERCENT = 82.0
FACE_AMBIGUOUS_MARGIN_PERCENT = 4.0
FACE_REVIEW_BAND_PERCENT = 5.0
FACE_DETECTION_THRESHOLD = 0.55
FACE_DETECTION_SIZE = (1024, 1024)
FACE_MIN_TEMPLATE_QUALITY = 0.48
FACE_MAX_TEMPLATES_PER_ID = 12
FACE_AGGREGATE_TOP_TEMPLATES = 3
FACE_USE_COREML = True
FACE_COREML_COMPUTE_UNITS = "CPUAndNeuralEngine"

OCCURRENCE_COLUMNS = [
    "Photo ID", "Photo", "Photo path", "Face ID", "Bounding box",
    "Match score %", "Detector score", "Quality", "Decision", "Possible match ID",
    "Seen at UTC",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _empty_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": {
            "repository": MODEL_REPO,
            "recognizer": "AuraFace-v1 glintr100",
            "detector": "SCRFD 10G with 5-point landmarks",
        },
        "identities": [],
        "processed_photos": {},
    }


def load_face_data() -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    registry = _empty_registry()
    if REGISTRY_FILE.exists():
        try:
            registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log_error(f"Failed to read face registry file {REGISTRY_FILE}", exc)
            registry = _empty_registry()
    registry.setdefault("identities", [])
    registry.setdefault("processed_photos", {})
    if OCCURRENCES_FILE.exists():
        try:
            occurrences = pd.read_csv(OCCURRENCES_FILE, dtype=str, keep_default_na=False)
            for column in OCCURRENCE_COLUMNS:
                if column not in occurrences:
                    occurrences[column] = ""
            occurrences = occurrences[OCCURRENCE_COLUMNS].drop_duplicates(subset=["Photo", "Face ID", "Bounding box"])
        except Exception as exc:
            log_error(f"Failed to read face occurrences file {OCCURRENCES_FILE}", exc)
            occurrences = pd.DataFrame(columns=OCCURRENCE_COLUMNS)
    else:
        occurrences = pd.DataFrame(columns=OCCURRENCE_COLUMNS)
    templates: dict[str, np.ndarray] = {}
    if EMBEDDINGS_FILE.exists():
        try:
            with np.load(EMBEDDINGS_FILE, allow_pickle=False) as stored:
                templates = {key: stored[key].astype(np.float32) for key in stored.files}
        except Exception as exc:
            log_error(f"Failed to read face embeddings file {EMBEDDINGS_FILE}", exc)
            templates = {}
    return registry, occurrences, templates


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def _save_face_data(
    registry: dict[str, Any],
    occurrences: pd.DataFrame,
    templates: dict[str, np.ndarray],
) -> None:
    try:
        FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(REGISTRY_FILE, registry)
        embedding_temp = EMBEDDINGS_FILE.with_suffix(".npz.tmp")
        with embedding_temp.open("wb") as stream:
            np.savez_compressed(stream, **templates)
        embedding_temp.replace(EMBEDDINGS_FILE)
        occurrence_temp = OCCURRENCES_FILE.with_suffix(".csv.tmp")
        occurrences.to_csv(occurrence_temp, index=False)
        occurrence_temp.replace(OCCURRENCES_FILE)
    except Exception as exc:
        log_error("Failed to save face library data", exc)
        raise



def load_face_analyzer() -> Any:
    """Load the ONNX face models with CoreML acceleration when available."""
    from huggingface_hub import snapshot_download
    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    if not (FACE_MODEL_DIR / "glintr100.onnx").exists() or not (
        FACE_MODEL_DIR / "scrfd_10g_bnkps.onnx"
    ).exists():
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=str(FACE_MODEL_DIR),
            allow_patterns=[
                "glintr100.onnx",
                "scrfd_10g_bnkps.onnx",
                "LICENSE.md",
                "README.md",
            ],
        )
    available_providers = set(ort.get_available_providers())
    providers: list[Any] = []
    if FACE_USE_COREML and "CoreMLExecutionProvider" in available_providers:
        providers.append((
            "CoreMLExecutionProvider",
            {
                "MLComputeUnits": FACE_COREML_COMPUTE_UNITS,
                "ModelCacheDirectory": str(FACE_MODEL_DIR / "coreml_cache"),
            },
        ))
    providers.append("CPUExecutionProvider")

    analyzer = FaceAnalysis(
        name="auraface",
        root=str(FACE_DATA_DIR),
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    analyzer.prepare(
        ctx_id=-1,
        det_thresh=FACE_DETECTION_THRESHOLD,
        det_size=FACE_DETECTION_SIZE,
    )
    return analyzer


def _score_percent(cosine_similarity: float) -> float:
    return (float(cosine_similarity) + 1.0) * 50.0


def _match_identity(
    embedding: np.ndarray,
    identities: list[dict[str, Any]],
    templates: dict[str, np.ndarray],
) -> tuple[str | None, float, float]:
    scored: list[tuple[str, float]] = []
    for identity in identities:
        face_id = identity["face_id"]
        samples = templates.get(face_id)
        if samples is None or len(samples) == 0:
            continue
        similarities = np.asarray(samples, dtype=np.float32) @ embedding
        top_count = min(FACE_AGGREGATE_TOP_TEMPLATES, len(similarities))
        top_similarities = np.sort(similarities)[-top_count:]
        score = _score_percent(float(np.mean(top_similarities)))
        scored.append((face_id, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    best_id, best_score = scored[0] if scored else (None, 0.0)
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    return best_id, best_score, second_score


def _new_face_id(existing_ids: set[str]) -> str:
    while True:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"{secrets.randbelow(1_000_000):06d}"
        face_id = f"johndoe_{stamp}_{suffix}"
        if face_id not in existing_ids:
            return face_id


def _face_quality(face: Any) -> tuple[float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in face.bbox]
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    min_side = min(width, height)
    detector_score = float(face.det_score)
    size_quality = min(1.0, min_side / 180.0)
    quality = detector_score * (0.5 + 0.5 * size_quality)
    return quality, width, height


def _save_face_crop(image: Image.Image, face: Any, face_id: str) -> str:
    FACE_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in face.bbox]
    pad_x, pad_y = (x2 - x1) * 0.20, (y2 - y1) * 0.25
    bounds = (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)),
        min(height, int(y2 + pad_y)),
    )
    path = FACE_CROPS_DIR / f"{face_id}.jpg"
    image.crop(bounds).save(path, format="JPEG", quality=94, optimize=True)
    return str(path.relative_to(ROOT))


def _store_template(
    face_id: str,
    embedding: np.ndarray,
    quality: float,
    identity: dict[str, Any],
    templates: dict[str, np.ndarray],
) -> None:
    samples = templates.get(face_id, np.empty((0, embedding.shape[0]), dtype=np.float32))
    qualities = list(identity.get("template_qualities", []))
    if len(samples) < FACE_MAX_TEMPLATES_PER_ID:
        templates[face_id] = np.concatenate([samples, embedding[None, :]], axis=0)
        qualities.append(float(quality))
    elif quality > min(qualities):
        replace_at = qualities.index(min(qualities))
        samples[replace_at] = embedding
        qualities[replace_at] = float(quality)
        templates[face_id] = samples
    identity["template_qualities"] = qualities


def process_face_photos(
    items: list[tuple[str, str, str]],
    analyzer: Any,
    force_rerun: bool = False,
    progress_callback: Callable[[int, int, str, dict[str, int], str], None] | None = None,
) -> dict[str, Any]:
    """Detect, match, and persist photos, optionally replacing prior scan rows."""
    registry, occurrences, templates = load_face_data()
    identities: list[dict[str, Any]] = registry["identities"]
    processed: dict[str, Any] = registry["processed_photos"]
    unique_items = list({item[1]: item for item in items}.values())
    todo = unique_items if force_rerun else [item for item in unique_items if item[1] not in processed]
    if not todo:
        return {"processed": 0, "faces": 0, "new": 0, "matched": 0, "review": 0, "skipped": len(unique_items), "errors": []}

    existing_ids = {identity["face_id"] for identity in identities}
    occurrence_rows: list[dict[str, str]] = []
    successful_photo_ids: set[str] = set()
    counts = {"processed": 0, "faces": 0, "new": 0, "matched": 0, "review": 0, "skipped": len(unique_items) - len(todo)}
    errors: list[str] = []

    for photo_index, (photo_name, photo_id, photo_path) in enumerate(todo, start=1):
        photo_error = ""
        try:
            with Image.open(photo_path) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
            rgb = np.asarray(image)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            detections = analyzer.get(bgr)
            stamp = _utc_now()
            photo_occurrence_rows: list[dict[str, str]] = []
            for face in detections:
                embedding = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(embedding))
                if not norm:
                    continue
                embedding /= norm
                quality, _, _ = _face_quality(face)
                best_id, best_score, second_score = _match_identity(embedding, identities, templates)
                gap = best_score - second_score
                match_ok = best_id is not None and best_score >= FACE_MATCH_THRESHOLD_PERCENT and gap >= FACE_AMBIGUOUS_MARGIN_PERCENT
                uncertain = (
                    best_id is not None
                    and best_score >= FACE_MATCH_THRESHOLD_PERCENT - FACE_REVIEW_BAND_PERCENT
                    and not match_ok
                )

                if match_ok:
                    face_id = best_id
                    decision = "matched"
                    target = next(person for person in identities if person["face_id"] == face_id)
                    target["last_seen_utc"] = stamp
                    if quality >= FACE_MIN_TEMPLATE_QUALITY:
                        _store_template(face_id, embedding, quality, target, templates)
                    counts["matched"] += 1
                    possible_match = ""
                else:
                    face_id = _new_face_id(existing_ids)
                    existing_ids.add(face_id)
                    decision = "needs_review" if uncertain else "new"
                    possible_match = best_id or ""
                    relative_crop = _save_face_crop(image, face, face_id)
                    target = {
                        "face_id": face_id,
                        "display_name": face_id,
                        "created_at_utc": stamp,
                        "last_seen_utc": stamp,
                        "representative_image": relative_crop,
                        "representative_quality": float(quality),
                        "template_qualities": [],
                        "review_status": decision,
                        "possible_match_id": possible_match,
                    }
                    identities.append(target)
                    _store_template(face_id, embedding, quality, target, templates)
                    counts["new"] += 1
                    if uncertain:
                        counts["review"] += 1

                if match_ok and quality > float(target.get("representative_quality", 0.0)):
                    relative_crop = _save_face_crop(image, face, face_id)
                    target["representative_image"] = relative_crop
                    target["representative_quality"] = float(quality)

                x1, y1, x2, y2 = [int(round(value)) for value in face.bbox]
                photo_occurrence_rows.append({
                    "Photo ID": photo_id,
                    "Photo": photo_name,
                    "Photo path": photo_path,
                    "Face ID": face_id,
                    "Bounding box": f"{x1},{y1},{x2},{y2}",
                    "Match score %": f"{best_score:.2f}" if best_id else "",
                    "Detector score": f"{float(face.det_score):.4f}",
                    "Quality": f"{quality:.4f}",
                    "Decision": decision,
                    "Possible match ID": possible_match,
                    "Seen at UTC": stamp,
                })
                counts["faces"] += 1
            processed[photo_id] = {
                "photo": photo_name,
                "path": photo_path,
                "processed_at_utc": stamp,
                "face_count": len(detections),
                "match_threshold_percent": FACE_MATCH_THRESHOLD_PERCENT,
                "detection_size": list(FACE_DETECTION_SIZE),
            }
            successful_photo_ids.add(photo_id)
            occurrence_rows.extend(photo_occurrence_rows)
            counts["processed"] += 1
        except Exception as exc:  # record individual photo failures and continue the batch
            photo_error = f"{photo_name}: {exc}"
            log_error(f"Face processing error for photo {photo_name}", exc, {"photo_id": photo_id, "path": photo_path})
            errors.append(photo_error)

        if progress_callback is not None:
            progress_callback(photo_index, len(todo), photo_name, counts.copy(), photo_error)

    if successful_photo_ids:
        occurrences = occurrences[~occurrences["Photo ID"].isin(successful_photo_ids)].copy()
    if occurrence_rows:
        occurrences = pd.concat(
            [occurrences, pd.DataFrame(occurrence_rows, columns=OCCURRENCE_COLUMNS)],
            ignore_index=True,
        )
    _save_face_data(registry, occurrences, templates)
    return {**counts, "errors": errors}


def rename_face_identities(names: dict[str, str]) -> int:
    registry, _, _ = load_face_data()
    updated = 0
    for identity in registry["identities"]:
        candidate = names.get(identity["face_id"], "").strip()
        if candidate and candidate != identity.get("display_name", identity["face_id"]):
            identity["display_name"] = candidate
            identity["renamed_at_utc"] = _utc_now()
            updated += 1
    _atomic_write_json(REGISTRY_FILE, registry)
    return updated


def merge_face_identities(source_face_id: str, target_face_id: str) -> None:
    """Merge a manually reviewed duplicate into the confirmed target identity."""
    if source_face_id == target_face_id:
        raise ValueError("Choose two different face IDs to merge.")
    registry, occurrences, templates = load_face_data()
    identities: list[dict[str, Any]] = registry["identities"]
    source = next((person for person in identities if person["face_id"] == source_face_id), None)
    target = next((person for person in identities if person["face_id"] == target_face_id), None)
    if source is None or target is None:
        raise ValueError("The selected face ID is no longer in the face library.")

    source_samples = templates.pop(source_face_id, np.empty((0, 512), dtype=np.float32))
    target_samples = templates.get(target_face_id, np.empty((0, 512), dtype=np.float32))
    source_qualities = list(source.get("template_qualities", []))
    target_qualities = list(target.get("template_qualities", []))
    all_samples = np.concatenate([target_samples, source_samples], axis=0)
    all_qualities = (target_qualities + source_qualities)[:len(all_samples)]
    if len(all_qualities) < len(all_samples):
        all_qualities.extend([0.0] * (len(all_samples) - len(all_qualities)))
    if len(all_samples):
        order = np.argsort(all_qualities)[-FACE_MAX_TEMPLATES_PER_ID:]
        templates[target_face_id] = all_samples[order].astype(np.float32)
        target["template_qualities"] = [float(all_qualities[index]) for index in order]

    if float(source.get("representative_quality", 0.0)) > float(target.get("representative_quality", 0.0)):
        target["representative_image"] = source.get("representative_image", "")
        target["representative_quality"] = float(source.get("representative_quality", 0.0))
    target["created_at_utc"] = min(target.get("created_at_utc", _utc_now()), source.get("created_at_utc", _utc_now()))
    target["last_seen_utc"] = max(target.get("last_seen_utc", _utc_now()), source.get("last_seen_utc", _utc_now()))
    target["review_status"] = "matched_manual"
    target["possible_match_id"] = ""

    if not occurrences.empty:
        merged_rows = occurrences["Face ID"] == source_face_id
        occurrences.loc[occurrences["Face ID"] == source_face_id, "Face ID"] = target_face_id
        occurrences.loc[occurrences["Possible match ID"] == source_face_id, "Possible match ID"] = target_face_id
        occurrences.loc[merged_rows, "Decision"] = "matched_manual"
        occurrences.loc[merged_rows, "Possible match ID"] = ""
    for identity in identities:
        if identity.get("possible_match_id") == source_face_id:
            identity["possible_match_id"] = target_face_id
    registry["identities"] = [person for person in identities if person["face_id"] != source_face_id]
    _save_face_data(registry, occurrences, templates)


def clear_face_data() -> None:
    """Clear all face registry, occurrences, embeddings, and face crop images."""
    registry = _empty_registry()
    occurrences = pd.DataFrame(columns=OCCURRENCE_COLUMNS)
    templates: dict[str, np.ndarray] = {}
    _save_face_data(registry, occurrences, templates)
    if FACE_CROPS_DIR.exists():
        for file in FACE_CROPS_DIR.glob("*"):
            if file.is_file():
                try:
                    file.unlink()
                except OSError:
                    pass


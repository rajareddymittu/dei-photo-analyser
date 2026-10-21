"""Build privacy-conscious, deterministic analytics context for language models."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import pandas as pd

from .brands import summarize_brands


def _labels(values: Iterable[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for value in values:
        if value is None or pd.isna(value):
            continue
        counts.update(label.strip() for label in str(value).split(";") if label.strip())
    return counts


def summarize_classifications(
    frame: pd.DataFrame,
    attribute_columns: list[str],
) -> dict[str, Any]:
    """Aggregate classification rows without including filenames or source paths."""
    if frame.empty:
        return {
            "classification_rows": 0,
            "unique_photos": 0,
            "people_detections": 0,
            "people_per_photo": {},
            "attribute_coverage": {},
            "tag_appearances": {},
            "brand_analytics": summarize_brands(pd.DataFrame()),
            "monthly_activity": [],
            "date_coverage": {"dated_rows": 0, "undated_rows": 0, "earliest": None, "latest": None},
        }

    photo_column = frame.get("Photo ID", pd.Series(frame.index.astype(str), index=frame.index))
    people_count = (
        pd.to_numeric(frame.get("People in photo", pd.Series(index=frame.index, dtype="float64")), errors="coerce")
        .groupby(photo_column)
        .max()
        .dropna()
    )
    photo_count = int(photo_column.nunique())
    attr_coverage: dict[str, dict[str, Any]] = {}
    tag_appearances: dict[str, dict[str, int]] = {}
    for column in attribute_columns:
        if column not in frame:
            continue
        values = frame[column].fillna("").astype(str)
        covered = values.str.strip().ne("")
        counts = _labels(values)
        attr_coverage[column] = {
            "rows_with_value": int(covered.sum()),
            "coverage_percent": round(float(covered.mean() * 100), 1) if len(values) else 0.0,
        }
        tag_appearances[column] = dict(counts.most_common())

    dates = pd.to_datetime(frame.get("Classified date", pd.Series(index=frame.index, dtype="object")), errors="coerce", utc=True)
    dated = dates.notna()
    monthly: list[dict[str, Any]] = []
    if dated.any():
        monthly_frame = pd.DataFrame({"month": dates[dated].dt.strftime("%Y-%m"), "photo_id": photo_column[dated]})
        monthly = [
            {"month": str(month), "classification_rows": int(rows), "unique_photos": int(photos)}
            for month, rows, photos in (
                monthly_frame.groupby("month").agg(classification_rows=("photo_id", "size"), unique_photos=("photo_id", "nunique"))
                .reset_index().tail(36).itertuples(index=False, name=None)
            )
        ]

    return {
        "classification_rows": int(len(frame)),
        "unique_photos": photo_count,
        "person_classification_rows": int(len(frame)),
        "people_per_photo": {
            "photos_with_count": int(len(people_count)),
            "average": round(float(people_count.mean()), 2) if len(people_count) else None,
            "median": float(people_count.median()) if len(people_count) else None,
            "distribution": {str(int(n)): int(c) for n, c in people_count.value_counts().sort_index().items()},
        },
        "attribute_coverage": attr_coverage,
        "tag_appearances": tag_appearances,
        "brand_analytics": summarize_brands(frame),
        "monthly_activity": monthly,
        "date_coverage": {
            "dated_rows": int(dated.sum()),
            "undated_rows": int((~dated).sum()),
            "earliest": dates[dated].min().strftime("%Y-%m-%d") if dated.any() else None,
            "latest": dates[dated].max().strftime("%Y-%m-%d") if dated.any() else None,
        },
    }


def summarize_faces(registry: dict[str, Any] | None, occurrences: pd.DataFrame | None) -> dict[str, Any]:
    """Summarize face tracking without sending names, images, paths, or embeddings."""
    registry = registry or {}
    occurrences = occurrences if occurrences is not None else pd.DataFrame()
    identities = registry.get("identities", [])
    processed = registry.get("processed_photos", {})
    if occurrences.empty:
        return {
            "mapped_identity_count": len(identities),
            "processed_photo_count": len(processed),
            "face_occurrences": 0,
            "photos_with_face_occurrences": 0,
            "repeat_identity_count": 0,
            "decision_counts": {},
            "review_status_counts": {},
        }

    face_col = occurrences.get("Face ID", pd.Series("", index=occurrences.index)).fillna("")
    photo_col = occurrences.get("Photo ID", pd.Series("", index=occurrences.index)).fillna("")
    photos_per_identity = photo_col.groupby(face_col).nunique()
    decisions = occurrences.get("Decision", pd.Series(dtype=str)).fillna("").value_counts()
    return {
        "mapped_identity_count": len(identities),
        "processed_photo_count": len(processed),
        "face_occurrences": int(len(occurrences)),
        "photos_with_face_occurrences": int(photo_col.nunique()),
        "repeat_identity_count": int((photos_per_identity > 1).sum()),
        "decision_counts": {str(key): int(value) for key, value in decisions.items() if key},
        "review_status_counts": {},
    }


def build_analytics_context(
    results: pd.DataFrame,
    face_registry: dict[str, Any] | None,
    face_occurrences: pd.DataFrame | None,
    attribute_columns: list[str],
    *,
    filter_attribute: str | None = None,
    selected_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Create an all-data plus current-filter summary suitable for provider context."""
    errors = results.get("Error", pd.Series("", index=results.index)).fillna("").astype(str).str.strip()
    valid = results[errors.eq("")].copy()
    error_summary = int((~errors.eq("")).sum())
    filtered = valid
    if filter_attribute and filter_attribute in valid.columns and selected_labels:
        selected = set(selected_labels)
        keep = valid[filter_attribute].fillna("").astype(str).str.split(";").map(
            lambda labels: bool(selected.intersection(item.strip() for item in labels))
        )
        filtered = valid[keep].copy()
    context: dict[str, Any] = {
        "scope_note": "Counts are computed from saved classification rows. Tag appearances count labels on person rows, not unique garments or unique people.",
        "saved_classification_rows": int(len(results)),
        "failed_classification_rows": error_summary,
        "all_results": summarize_classifications(valid, attribute_columns),
        "brand_analytics": summarize_brands(valid),
        "face_tracking": summarize_faces(face_registry, face_occurrences),
    }
    if filter_attribute and filter_attribute != "All results" and selected_labels:
        context["current_analysis_filter"] = {
            "attribute": filter_attribute,
            "selected_labels": selected_labels,
            "summary": summarize_classifications(filtered, attribute_columns),
        }
    return context


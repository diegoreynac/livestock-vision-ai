"""Dataset identity audit for animal-level grouping validation.

This module is intentionally limited to auditing dataset identity, view pairing,
weight consistency, sex consistency, and cross-dataset identity risk. It does not
implement any splitting, model training, or data transformation logic.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.analysis.animal_analysis import AnimalAnalysis
from src.dataset.models import ImageRecord


def _normalize_animal_id(value: Any) -> str | None:
    """Normalize a raw animal_id while preserving the original evidence."""
    if value is None:
        return None

    normalized = str(value).strip()
    if normalized == "" or normalized.lower() in {"null", "none", "nan", "na"}:
        return None

    return normalized


class DatasetIdentityAudit:
    """Computes animal-level identity evidence for downstream grouping decisions."""

    def __init__(self, weight_tolerance: float = 0.01) -> None:
        self.animal_analysis = AnimalAnalysis(weight_tolerance=weight_tolerance)

    def audit(self, records: Iterable[ImageRecord]) -> dict[str, Any]:
        """Return the full identity audit report for a record collection."""
        record_list = list(records)
        animal_groups: dict[str, list[ImageRecord]] = defaultdict(list)
        invalid_records: list[dict[str, Any]] = []

        for record in record_list:
            normalized = _normalize_animal_id(record.animal_id)
            if normalized is None:
                invalid_records.append(
                    {
                        "animal_id": record.animal_id,
                        "filename": record.filename,
                        "filepath": str(record.filepath),
                        "dataset": getattr(getattr(record, "folder", None), "dataset", None).value
                        if getattr(getattr(record, "folder", None), "dataset", None) is not None
                        else None,
                        "view": getattr(getattr(record, "folder", None), "view", None).value
                        if getattr(getattr(record, "folder", None), "view", None) is not None
                        else None,
                    }
                )
                continue
            animal_groups[normalized].append(record)

        analysis = self.animal_analysis.analyze(record_list)
        animal_summaries = [self._summarize_animal(animal_id, group) for animal_id, group in sorted(animal_groups.items())]

        per_animal_images = [
            {"animal_id": animal["animal_id"], "image_count": animal["image_count"]}
            for animal in animal_summaries
        ]
        image_counts = [entry["image_count"] for entry in per_animal_images]
        high_threshold = self._high_count_threshold(image_counts)
        high_count_animals = [
            {"animal_id": entry["animal_id"], "image_count": entry["image_count"]}
            for entry in per_animal_images
            if entry["image_count"] > high_threshold
        ]

        dataset_membership = {
            "B2 only": 0,
            "B3 only": 0,
            "B4 only": 0,
            "B2 + B3": 0,
            "B2 + B4": 0,
            "B3 + B4": 0,
            "B2 + B3 + B4": 0,
        }

        for animal in animal_summaries:
            datasets = sorted({item for item in animal["datasets"] if item in {"B2", "B3", "B4"}})
            if not datasets:
                continue
            dataset_key = self._dataset_membership_key(datasets)
            if dataset_key is not None:
                dataset_membership[dataset_key] += 1

        view_membership = {
            "Side only": 0,
            "Rear only": 0,
            "Side + Rear": 0,
            "Unknown": 0,
        }

        for animal in animal_summaries:
            if animal["view_category"] in view_membership:
                view_membership[animal["view_category"]] += 1
            else:
                view_membership["Unknown"] += 1

        paired_animals = []
        single_view_animals = []
        for animal in animal_summaries:
            if animal["view_category"] == "Side + Rear":
                paired_animals.append(
                    {
                        "animal_id": animal["animal_id"],
                        "side_images": animal["side_images"],
                        "rear_images": animal["rear_images"],
                        "datasets": animal["datasets"],
                    }
                )
            elif animal["view_category"] in {"Side only", "Rear only"}:
                single_view_animals.append(
                    {
                        "animal_id": animal["animal_id"],
                        "view_category": animal["view_category"],
                        "datasets": animal["datasets"],
                        "image_count": animal["image_count"],
                    }
                )

        weight_summary = self._summarize_consistency(
            animal_summaries,
            key="weight",
        )
        sex_summary = self._summarize_consistency(
            animal_summaries,
            key="sex",
        )

        cross_dataset = []
        for animal in animal_summaries:
            if len(animal["datasets"]) > 1:
                cross_dataset.append(
                    {
                        "animal_id": animal["animal_id"],
                        "datasets": animal["datasets"],
                        "views": animal["views"],
                        "image_count": animal["image_count"],
                        "weight_values": animal["weight_values"],
                        "sex_values": animal["sex_values"],
                    }
                )

        return {
            "total_images": len(record_list),
            "total_valid_animal_ids": len(animal_groups),
            "unique_animals": len(animal_groups),
            "invalid_animal_id_records": len(invalid_records),
            "invalid_animal_id_examples": invalid_records,
            "images_per_animal": per_animal_images,
            "animals_with_unusually_high_image_count": high_count_animals,
            "high_image_count_threshold": high_threshold,
            "dataset_membership": dataset_membership,
            "view_membership": view_membership,
            "side_rear_pairing": {
                "animals_with_both_views": paired_animals,
                "animals_with_only_one_view": single_view_animals,
                "animals_with_both_views_count": len(paired_animals),
                "animals_with_only_one_view_count": len(single_view_animals),
            },
            "weight_consistency": {
                "consistent_animals": weight_summary["consistent"],
                "inconsistent_animals": weight_summary["inconsistent"],
                "missing_weights": weight_summary["missing"],
                "details": weight_summary["details"],
            },
            "sex_consistency": {
                "consistent_animals": sex_summary["consistent"],
                "inconsistent_animals": sex_summary["inconsistent"],
                "missing_sex": sex_summary["missing"],
                "details": sex_summary["details"],
            },
            "cross_dataset_identity_risk": {
                "same_animal_id_appears_in_multiple_datasets": len(cross_dataset) > 0,
                "cross_dataset_animals": cross_dataset,
                "count": len(cross_dataset),
            },
            "animal_summary": animal_summaries,
            "analysis_metadata": {
                "weight_tolerance": str(self.animal_analysis.weight_tolerance),
                "dataset_membership_semantics": "Counts animals once per dataset membership subset. This is not an exclusive partition.",
                "identity_risk_note": "A repeated animal_id across datasets is evidence only; it does not prove that the IDs refer to the same physical animal without additional dataset evidence.",
            },
        }

    def write_outputs(self, report: dict[str, Any], output_directory: Path) -> tuple[Path, Path]:
        """Write the audit JSON and a concise text summary to disk."""
        output_directory.mkdir(parents=True, exist_ok=True)

        json_path = output_directory / "dataset_identity_audit.json"
        text_path = output_directory / "dataset_identity_audit_summary.txt"

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, sort_keys=True)

        text_path.write_text(self._build_summary(report), encoding="utf-8")
        return json_path, text_path

    def _summarize_animal(self, animal_id: str, records: list[ImageRecord]) -> dict[str, Any]:
        view_types = sorted({self._view_name(record) for record in records})
        datasets = sorted({self._dataset_name(record) for record in records if self._dataset_name(record) is not None})
        weight_values = [self._normalize_float(record.weight_kg) for record in records]
        sex_values = [self._normalize_sex(record.sex) for record in records]
        side_images = sum(1 for record in records if self._view_name(record) == "Side")
        rear_images = sum(1 for record in records if self._view_name(record) == "Rear")

        return {
            "animal_id": animal_id,
            "image_count": len(records),
            "datasets": datasets,
            "views": view_types,
            "view_category": self._view_category(view_types),
            "weight_values": weight_values,
            "distinct_weight_values": sorted(set(weight_values), key=lambda value: (value is None, value)),
            "consistent_weight": self._is_consistent_weight(weight_values),
            "sex_values": sex_values,
            "distinct_sex_values": sorted(set(sex_values)),
            "consistent_sex": self._is_consistent_sex(sex_values),
            "side_images": side_images,
            "rear_images": rear_images,
        }

    def _summarize_consistency(self, animals: list[dict[str, Any]], key: str) -> dict[str, Any]:
        consistent = []
        inconsistent = []
        missing = []
        details = []

        for animal in animals:
            values = animal[f"{key}_values"]
            if key == "weight":
                normalized = [self._normalize_float(value) for value in values]
                if all(value is None for value in normalized):
                    missing.append(animal["animal_id"])
                    details.append({"animal_id": animal["animal_id"], "status": "missing", "values": values})
                    continue
                if self._is_consistent_weight(normalized):
                    consistent.append(animal["animal_id"])
                    details.append({"animal_id": animal["animal_id"], "status": "consistent", "values": normalized})
                else:
                    inconsistent.append(animal["animal_id"])
                    details.append({"animal_id": animal["animal_id"], "status": "inconsistent", "values": normalized})
            else:
                normalized = [self._normalize_sex(value) for value in values]
                if all(value == "Missing" for value in normalized):
                    missing.append(animal["animal_id"])
                    details.append({"animal_id": animal["animal_id"], "status": "missing", "values": normalized})
                    continue
                if self._is_consistent_sex(normalized):
                    consistent.append(animal["animal_id"])
                    details.append({"animal_id": animal["animal_id"], "status": "consistent", "values": normalized})
                else:
                    inconsistent.append(animal["animal_id"])
                    details.append({"animal_id": animal["animal_id"], "status": "inconsistent", "values": normalized})

        return {
            "consistent": sorted(consistent),
            "inconsistent": sorted(inconsistent),
            "missing": sorted(missing),
            "details": details,
        }

    def _dataset_membership_key(self, datasets: list[str]) -> str | None:
        ordered = ["B2", "B3", "B4"]
        dataset_set = set(datasets)
        if dataset_set == {"B2"}:
            return "B2 only"
        if dataset_set == {"B3"}:
            return "B3 only"
        if dataset_set == {"B4"}:
            return "B4 only"
        if dataset_set == {"B2", "B3"}:
            return "B2 + B3"
        if dataset_set == {"B2", "B4"}:
            return "B2 + B4"
        if dataset_set == {"B3", "B4"}:
            return "B3 + B4"
        if dataset_set == {"B2", "B3", "B4"}:
            return "B2 + B3 + B4"
        return None

    def _view_category(self, view_types: list[str]) -> str:
        view_set = set(view_types)
        if view_set == {"Side"}:
            return "Side only"
        if view_set == {"Rear"}:
            return "Rear only"
        if view_set == {"Side", "Rear"}:
            return "Side + Rear"
        return "Unknown"

    def _view_name(self, record: ImageRecord) -> str:
        if record.folder is None:
            return "Unknown"
        return record.folder.view.value

    def _dataset_name(self, record: ImageRecord) -> str | None:
        if record.folder is None:
            return None
        return record.folder.dataset.value

    def _normalize_float(self, value: Any) -> float | None:
        if value is None:
            return None
        if not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric

    def _normalize_sex(self, value: Any) -> str:
        if value is None:
            return "Missing"
        if hasattr(value, "value"):
            sex_value = str(value.value)
            if sex_value in {"M", "F"}:
                return sex_value
            return "Missing"
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in {"M", "F"}:
                return normalized
        return "Missing"

    def _is_consistent_weight(self, weights: list[float | None]) -> bool:
        valid = [weight for weight in weights if weight is not None]
        if not valid:
            return False
        if len(valid) != len(weights):
            return False
        return max(valid) - min(valid) <= self.animal_analysis.weight_tolerance

    def _is_consistent_sex(self, sexes: list[str]) -> bool:
        distinct = set(sexes)
        if not distinct:
            return False
        if distinct == {"Missing"}:
            return False
        return len(distinct) == 1 and next(iter(distinct)) in {"M", "F"}

    def _high_count_threshold(self, counts: list[int]) -> int:
        if not counts:
            return 0

        mean_value = statistics.mean(counts)
        std_dev = statistics.pstdev(counts) if len(counts) > 1 else 0.0
        threshold = math.ceil(mean_value + (3 * std_dev))
        return max(3, threshold)

    def _build_summary(self, report: dict[str, Any]) -> str:
        dataset_membership = report["dataset_membership"]
        view_membership = report["view_membership"]
        weight = report["weight_consistency"]
        sex = report["sex_consistency"]
        cross_dataset = report["cross_dataset_identity_risk"]

        lines = [
            "Dataset Identity Audit",
            "======================",
            f"Total images: {report['total_images']}",
            f"Total valid animal IDs: {report['total_valid_animal_ids']}",
            f"Unique animals: {report['unique_animals']}",
            f"Invalid animal ID records: {report['invalid_animal_id_records']}",
            "",
            "Dataset membership:",
            f"  B2 only: {dataset_membership['B2 only']}",
            f"  B3 only: {dataset_membership['B3 only']}",
            f"  B4 only: {dataset_membership['B4 only']}",
            f"  B2 + B3: {dataset_membership['B2 + B3']}",
            f"  B2 + B4: {dataset_membership['B2 + B4']}",
            f"  B3 + B4: {dataset_membership['B3 + B4']}",
            f"  B2 + B3 + B4: {dataset_membership['B2 + B3 + B4']}",
            "",
            "View membership:",
            f"  Side only: {view_membership['Side only']}",
            f"  Rear only: {view_membership['Rear only']}",
            f"  Side + Rear: {view_membership['Side + Rear']}",
            f"  Unknown: {view_membership['Unknown']}",
            "",
            "Weight:",
            f"  Consistent: {len(weight['consistent_animals'])}",
            f"  Inconsistent: {len(weight['inconsistent_animals'])}",
            f"  Missing: {len(weight['missing_weights'])}",
            "",
            "Sex:",
            f"  Consistent: {len(sex['consistent_animals'])}",
            f"  Inconsistent: {len(sex['inconsistent_animals'])}",
            f"  Missing: {len(sex['missing_sex'])}",
            "",
            "Cross-dataset animal IDs:",
        ]

        if cross_dataset["cross_dataset_animals"]:
            for item in cross_dataset["cross_dataset_animals"]:
                lines.append(
                    f"  - {item['animal_id']}: datasets={item['datasets']}, views={item['views']}, "
                    f"images={item['image_count']}, weights={item['weight_values']}, sex={item['sex_values']}"
                )
        else:
            lines.append("  None observed")

        if report["animals_with_unusually_high_image_count"]:
            lines.append("")
            lines.append("Animals with unusually high image counts:")
            for item in report["animals_with_unusually_high_image_count"]:
                lines.append(f"  - {item['animal_id']}: {item['image_count']} images")

        return "\n".join(lines) + "\n"


def run_identity_audit() -> dict[str, Any]:
    """Load the configured dataset and write the identity audit artifacts."""
    from src.core.config import ProjectConfig
    from src.core.context import ProjectContext
    from src.dataset.reader import DatasetReader

    config = ProjectConfig.default()
    config.validate()

    context = ProjectContext(config=config, experiment_name="dataset_identity_audit")
    context.config.create_output_directories()

    try:
        reader = DatasetReader(context)
        dataset = reader.load()
        audit = DatasetIdentityAudit()
        result = audit.audit(dataset.records)
        audit.write_outputs(result, context.reports_dir)
        context.logger.section("Dataset Identity Audit Complete")
        context.logger.info(f"Audit JSON: {context.reports_dir / 'dataset_identity_audit.json'}")
        context.logger.info(f"Audit summary: {context.reports_dir / 'dataset_identity_audit_summary.txt'}")
        return result
    finally:
        context.close()


if __name__ == "__main__":
    run_identity_audit()

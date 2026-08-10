# COCO Integration Progress

## Project

Livestock Weight Estimation using Computer Vision

Repository:

livestock-vision-ai

---

# Objective

Integrate the COCO annotations into the dataset in order to:

- validate annotation quality;
- extract body measurements;
- train weight estimation models.

---

# Sprint 0 — Baseline

Status

✅ Completed

Implemented

- COCOReader
- COCOValidator
- COCOStatistics
- DatasetVisualizer

Results

- Pipeline executed successfully.
- Initial COCO integration completed.

Known Issues

- Only 2,131 annotations loaded.
- Many missing annotations.
- BBox validation generated thousands of errors.

# Sprint 1 — COCO Audit

Status

✅ Completed

Goal

Understand the real structure and quality of the COCO annotations before modifying the processing pipeline.

Implemented

New module:

src/coco/audit.py

Capabilities

- JSON validation
- Physical dataset comparison
- Annotation coverage
- Keypoint distribution
- BBox coverage
- Segmentation coverage
- Duplicate detection
- Orphan image IDs
- Dataset vs JSON matching

Main Findings

- B2 contains incomplete annotations.
- B4 provides keypoints but not valid bounding boxes.
- Several orphan image IDs exist.
- Dataset quality differs significantly between folders.

Outcome

The audit demonstrated that a large portion of the missing annotations was caused by the Reader implementation rather than the dataset itself.

# Sprint 2 — COCO Reader

Status

🚧 In Progress

Goal

Make the Reader fully compliant with the dataset.

Completed

### 2.1 Keypoint Validation

Problem

The Reader validated the flat keypoint payload correctly.

After converting it into COCOKeypoint objects it repeated an invalid validation.

Result

Annotations loaded

Before

2131

After

4752

---

### 2.2 Contextual Matching

Problem

The Reader indexed images only by filename.

Images with the same filename located in different folders collided.

Solution

Contextual key

(dataset, folder, filename)

Result

Before

4752 assignments

4749 ImageRecords

After

4749 assignments

4749 ImageRecords

No annotation overwriting.

# Sprint 4 — COCO Validator

Status

✅ Completed

Goal

Transform the Validator from an error detector into a dataset quality analyzer.

Implemented

- Coverage metrics
- Partial annotation support
- Real validation of keypoints
- Distinction between missing and invalid data

Results

Annotations loaded

4749

Missing annotations

161

BBox

865 valid

3884 missing

0 invalid

Segmentation

0 available

4749 missing

0 invalid

Keypoints

4748 valid

1 invalid

Conclusion

The remaining issue belongs to the dataset itself, not to the processing pipeline.
# Thesis Project Plan

**Project:** Livestock Weight Estimation using Computer Vision and Deep Learning

**Student:** Diego Reyna

**Degree:** Master of Science

**Status:** In Progress

---

# 1. Objective

Develop a computer vision system capable of estimating cattle weight from images using deep learning.

The system will:

- Detect the animal.
- Predict the bounding box.
- Classify the animal sex.
- Estimate the animal weight.

The final deployment target is an Android application developed in Kotlin.

---

# 2. Research Question

Can deep learning estimate cattle weight from RGB images with sufficient accuracy for practical livestock applications?

---

# 3. General Objectives

- Build a complete dataset processing pipeline.
- Train and compare at least three deep learning models.
- Evaluate detection and weight estimation performance.
- Deploy the selected model on Android.

---

# 4. Current Project Status

## Phase 1 - Dataset Audit

Status:

✅ Completed

Deliverables

- Dataset Reader
- Filename Parser
- Dataset Statistics
- Reports
- Visualizations

---

## Phase 2 - COCO Integration

Status:

✅ Completed

Deliverables

- COCO Reader
- COCO Audit
- COCO Validator
- COCO Visualization

Current Metrics

Images indexed:

4910

Annotations loaded:

4749

Missing annotations:

161

Success rate:

96.72%

---

## Phase 3 - Model Development

Status:

🟡 Not Started

---

# 5. Advisor Requirements

The following requirements were defined during the thesis advisory meeting.

## R1

Justify the selected dataset.

Priority:

High

Status:

⬜ Pending

---

## R2

Network outputs:

- Bounding Box
- Sex
- Weight

Priority:

High

Status:

⬜ Pending

---

## R3

Deployment

Android Studio

Language:

Kotlin

Priority:

Medium

Status:

⬜ Pending

---

## R4

Unified training script

One training pipeline capable of training any supported model.

Priority:

High

Status:

⬜ Pending

---

## R5

Clustering analysis

Use clustering techniques during dataset analysis.

Priority:

Medium

Status:

⬜ Pending

---

## R6

Data augmentation

Apply augmentation during training.

Priority:

High

Status:

⬜ Pending

---

## R7

Train at least three neural network models.

Priority:

High

Status:

⬜ Pending

---

## R8

Compare models using:

- Number of parameters
- Latency
- Hardware
- mAP
- IoU

Priority:

High

Status:

⬜ Pending

---

## R9

Understand YOLO metrics.

Study:

- mAP
- IoU
- Precision
- Recall
- Loss functions

Priority:

High

Status:

⬜ Pending

---

# 6. Planned Roadmap

## Phase 3

Dataset justification

Expected Deliverable

DATASET_JUSTIFICATION.md

---

## Phase 4

Study YOLO architecture and metrics.

Expected Deliverable

YOLO_RESEARCH.md

---

## Phase 5

Training Pipeline

Expected Deliverable

train.py

---

## Phase 6

Model Comparison

Expected Deliverable

TRAINING_RESULTS.md

---

## Phase 7

Android Deployment

Expected Deliverable

ANDROID_DEPLOYMENT.md

---

# 7. Planned Experiments

Experiment 01

Baseline model

Status

⬜

Experiment 02

Model comparison

Status

⬜

Experiment 03

Data augmentation

Status

⬜

Experiment 04

Hyperparameter tuning

Status

⬜

---

# 8. Tentative Timeline

August

- Finish project planning.
- Justify the dataset.
- Study YOLO.

September

- Implement the training pipeline.
- Data augmentation.
- Clustering.
- Initial experiments.

October

- Training scripts completed.
- Train the three models.
- Begin Android deployment.

November

- Final experiments.
- Performance comparison.
- Deployment optimization.

December

- Thesis writing.
- Final evaluation.
- Defense preparation.

---

# 9. Current Milestone

✅ COCO Integration completed.

Next milestone:

Dataset justification and training pipeline design.
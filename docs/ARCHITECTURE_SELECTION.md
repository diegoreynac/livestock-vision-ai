# Architecture Selection

## 1. Purpose

This document records the technical reasoning and experimental strategy used to select neural network architectures for the livestock weight-estimation thesis. It centralizes the candidate architecture families, documents the criteria used to compare them, and records decisions that are already made versus those intentionally deferred.

The goal is not to choose the model with the highest single-task accuracy, but to identify the best trade-off between predictive performance (detection, sex classification, weight regression) and practical deployment feasibility for real-time on-device inference on Android (Kotlin).

This document references the common requirements defined in [MODEL_REQUIREMENTS.md](C:/Repos/livestock-vision-ai/docs/MODEL_REQUIREMENTS.md) and relies on the standardized training/evaluation protocol to be defined in [TRAINING_PROTOCOL.md](C:/Repos/livestock-vision-ai/docs/TRAINING_PROTOCOL.md).


## 2. Architecture Selection Criteria

The candidate architectures will be compared using a multi-objective set of criteria. Experimental evidence must support the final selection. The following criteria will be measured and reported for every candidate under the common protocol:

- Detection performance (IoU, mAP)
- Sex classification performance (Accuracy, Precision, Recall, F1)
- Weight regression performance (MAE, RMSE, R²)
- Number of trainable parameters
- Computational complexity (FLOPs where available)
- Model size (serialized file size)
- Inference latency (ms per image pair)
- Frames per second (FPS) when applicable
- Memory requirements during inference (peak RAM)
- Android compatibility and export feasibility
- Real-time feasibility on representative target hardware

The final selection will be justified by experimental results that consider predictive performance together with resource and latency constraints relevant to on-device Android deployment.


## 3. Common Architecture Design

This section briefly summarizes the design characteristics that all three candidate architectures share. These commonalities are required by the project and described in detail in [MODEL_REQUIREMENTS.md](C:/Repos/livestock-vision-ai/docs/MODEL_REQUIREMENTS.md).

### 3.1 Dual-view input

All candidates operate on two input images for the same animal:

- Side image
- Rear image

The thesis investigates whether complementary information from these two views can improve weight estimation. This is a hypothesis to be tested and not an assumption that two views will necessarily improve performance.

### 3.2 Feature extraction

Each architecture extracts visual representations from the Side and Rear images using architecture-specific feature extractors. The feature extractor design (backbone, layer choices, parameterization) is architecture-dependent and will be selected per family.

### 3.3 Feature fusion

Side and Rear feature representations are combined into a fused representation prior to multi-task prediction (bounding box, sex, weight):

Side Image
    |
    v
Feature Extraction
    |
Side Features
    \
     \
      --> Fusion --> Prediction Heads
     /
    /
Rear Features
    ^
    |
Feature Extraction
    ^
    |
Rear Image

The exact fusion mechanism (concatenation, attention, element-wise operations, learned fusion layers, etc.) is intentionally undecided and will be determined as part of architecture selection and documented in this file when chosen.

### 3.4 Multi-task prediction

All architectures share the same multi-task prediction objectives:

- Bounding box prediction
- Sex classification (F / M)
- Weight regression (kilograms)

These tasks share information from the fused representation but will be implemented as task-specific output heads. Whether multi-task training benefits the problem is an empirical question to be verified.


## 4. Architecture Comparison Strategy

To ensure a scientifically meaningful comparison, the three architecture families will be trained and evaluated under a standardized experimental framework. The following elements will remain constant across families wherever scientifically reasonable:

- Dataset and train/validation/test split
- Target variables and ground-truth labels
- Preprocessing and data-augmentation policy (as defined in TRAINING_PROTOCOL.md)
- Evaluation metrics and reporting format
- Training protocol and checkpointing
- Test set (held-out, single final evaluation)

Architecture-specific differences should be limited to model design choices (backbone, fusion mechanism, prediction heads). This isolates the effect of architecture from other confounders and enables fair comparison.


## 5. Architecture 1 — YOLO-based

### 5.1 Why YOLO

Conventional YOLO (You Only Look Once) is an object-detection family that maps an image to localized object predictions:

Image
  |
  v
YOLO
  |
  +--> Bounding Box
  +--> Class
  +--> Confidence

The proposed thesis architecture does not reuse a standard single-image YOLO detector unchanged. Instead, it extends a detection-oriented representation into a dual-view, multi-task architecture where YOLO-style feature extraction forms the backbone of each view's feature extractor:

Side Image
    |
    v
YOLO-based Feature Extraction
    |
Side Features
    \
     \
      --> Fusion --> BBox
     /               --> Sex
    /                --> Weight
Rear Features
    ^
    |
YOLO-based Feature Extraction
    ^
    |
Rear Image

YOLO is attractive because:

- It is a well-established family for bounding-box prediction (a required output)
- YOLO feature representations can be repurposed or extended for multiple prediction heads
- Lightweight YOLO variants exist that are relevant to mobile deployment and real-time inference

Important: standard YOLO models are designed for single-image detection on standard benchmarks. The thesis architecture adapts the YOLO representation to a dual-view, multi-task setting; those adaptations must be validated experimentally.


### 5.2 YOLO26 Family

The YOLO26 family is the specific family under investigation. It provides scaled variants that trade off model capacity and computational cost. These official reference benchmark values are provided as contextual information only and must not be presented as thesis experimental results.


### 5.3 YOLO26n

**Primary candidate — TO BE EXPERIMENTALLY VALIDATED**

Official reference values (author-provided benchmarks on standard datasets/hardware):

- Parameters: approximately 2.4M
- FLOPs: approximately 5.4B
- COCO mAP50-95: approximately 40.9
- COCO end-to-end mAP50-95: approximately 40.1
- Official CPU ONNX benchmark: approximately 38.9 ms
- Official T4 TensorRT benchmark: approximately 1.7 ms

These are official reference benchmark values only. They are not results from our livestock dataset or Android measurements.

Why YOLO26n is the primary candidate:

- Low parameter count and lower computational cost compared to larger variants
- Better alignment with mobile deployment constraints and real-time inference goals
- Leaves computational budget for dual-view processing and additional prediction heads

Label: PRIMARY CANDIDATE — TO BE EXPERIMENTALLY VALIDATED


### 5.4 YOLO26s

**Alternative candidate — TO BE EXPERIMENTALLY VALIDATED**

Official reference values (author-provided benchmarks on standard datasets/hardware):

- Parameters: approximately 9.5M
- FLOPs: approximately 20.7B
- COCO mAP50-95: approximately 48.6
- COCO end-to-end mAP50-95: approximately 47.8
- Official CPU ONNX benchmark: approximately 87.2 ms
- Official T4 TensorRT benchmark: approximately 2.5 ms

These are official reference benchmark values only. They are not results from our livestock dataset or Android measurements.

YOLO26s provides substantially more capacity than YOLO26n, at a significantly higher computational cost. It may offer improved representation power at the expense of mobile deployment feasibility.

Label: ALTERNATIVE CANDIDATE — TO BE EXPERIMENTALLY VALIDATED


### 5.5 YOLO26n vs YOLO26s

| Property | YOLO26n | YOLO26s |
|----------|---------:|--------:|
| Parameters | ~2.4M | ~9.5M |
| FLOPs | ~5.4B | ~20.7B |
| COCO mAP50-95 | ~40.9 | ~48.6 |
| COCO end-to-end mAP50-95 | ~40.1 | ~47.8 |
| Official CPU ONNX | ~38.9 ms | ~87.2 ms |
| Official T4 TensorRT | ~1.7 ms | ~2.5 ms |

Trade-off summary:

- YOLO26n is smaller and computationally lighter, making it attractive for mobile deployment and low-latency inference.
- YOLO26s offers greater representational capacity and higher benchmark mAP, but at substantially higher computational cost and model size.

Correct conclusion at this stage: YOLO26n is the primary candidate because of its mobile-oriented profile; YOLO26s remains a valid alternative whose additional capacity must be evaluated experimentally.


### 5.6 Dual-view YOLO Architecture

Conceptual structure:

                    SIDE IMAGE
                        |
                        v
                 YOLO Feature
                  Extraction
                        |
                        v
                  Side Features
                        |
                        |
                        v
                    FUSION
                        ^
                        |
                        |
                  Rear Features
                        ^
                        |
                 YOLO Feature
                  Extraction
                        ^
                        |
                    REAR IMAGE

After fusion:

                     FUSED FEATURES
                           |
              +------------+------------+
              |            |            |
              v            v            v
             BBox         Sex         Weight

Notes:
- Both Side and Rear features are used and fused prior to prediction
- The exact YOLO feature layer used, fusion method, and prediction-head implementations remain TBD and will be selected and justified experimentally


### 5.7 Multi-task Outputs

The YOLO-based family must implement the three required outputs (bounding box, sex, weight) consistent with the common evaluation protocol. Bounding-box evaluation will use IoU and mAP measurements defined in the training protocol. Sex and weight evaluation metrics are defined in MODEL_REQUIREMENTS.md.


### 5.8 Mobile Deployment Considerations

Mobile deployment constraints strongly influence candidate selection. For YOLO variants, the following are primary concerns:

- Parameter count and model size
- FLOPs and computational cost
- Inference latency on target hardware
- Peak memory during inference
- Ability to export to a mobile runtime (TFLite, ONNX, or another validated runtime)
- Support for quantization and other compression strategies

The final Android runtime and quantization choices remain TBD.


## 6. Architecture 2 — TBD

**TBD — ARCHITECTURE NOT YET SELECTED**

- Architecture family: TBD
- Motivation: TBD
- Mobile considerations: TBD
- Expected strengths: TBD
- Expected limitations: TBD
- Experimental validation: TBD

This section is a placeholder for the second family that will be selected and documented here after initial exploration. Candidate families under consideration (but not decided) include MobileNet-family and EfficientNet-family variants, among others. No family selection or final decisions are recorded here.


## 7. Architecture 3 — TBD

**TBD — ARCHITECTURE NOT YET SELECTED**

- Architecture family: TBD
- Motivation: TBD
- Mobile considerations: TBD
- Expected strengths: TBD
- Expected limitations: TBD
- Experimental validation: TBD

This section is a placeholder for the third family to be selected and documented here after initial exploration.


## 8. Dual-view Experimental Baselines

The following ablation or baseline experiments may be evaluated to measure the contribution of each view:

### 8.1 Side-only

Train and evaluate models using only the Side image. Serve as a baseline to assess how much information the Side view provides alone.

### 8.2 Rear-only

Train and evaluate models using only the Rear image. Serve as a baseline to assess how much information the Rear view provides alone.

### 8.3 Side + Rear

Train and evaluate models using both views with the dual-branch + fusion architecture.

These baselines are designed to test the hypothesis that complementary information from both views improves weight estimation. They are NOT part of the three main candidate families but are useful ablation experiments.


## 9. Cross-Architecture Evaluation

All three architecture families will be compared using the same evaluation framework and reporting templates. A preliminary comparison table is provided and will be populated with experimental results after the training and evaluation pipeline runs.

| Criterion | Architecture 1 | Architecture 2 | Architecture 3 |
|-----------|----------------|----------------|----------------|
| Backbone | YOLO26 | TBD | TBD |
| Parameters | TBD | TBD | TBD |
| FLOPs | TBD | TBD | TBD |
| Model size | TBD | TBD | TBD |
| Detection mAP | TBD | TBD | TBD |
| Detection IoU | TBD | TBD | TBD |
| Sex Accuracy | TBD | TBD | TBD |
| Weight MAE | TBD | TBD | TBD |
| Weight RMSE | TBD | TBD | TBD |
| Weight R² | TBD | TBD | TBD |
| Latency | TBD | TBD | TBD |
| FPS | TBD | TBD | TBD |
| Android deployment | TBD | TBD | TBD |

This table will be filled with values measured on the common test set and reported with experimental details (hardware, input resolution, model precision).


## 10. Mobile Deployment Evaluation

Mobile deployment evaluation will include:

- Model export to one or more mobile-friendly formats
- Integration in an Android/Kotlin test harness
- Measurement of inference latency (ms) and FPS with representative input resolutions
- Reporting of model file size and peak memory usage during inference
- Documentation of hardware used for inference experiments (CPU, GPU, NNAPI availability)

The final mobile runtime and target smartphone hardware remain TBD and will be selected based on compatibility and experimental results.


## 11. Decisions Already Made

The following items are recorded as DECIDED for the project:

- Three neural network architectures will be compared. (DECIDED)
- All architectures solve the same livestock weight-estimation problem. (DECIDED)
- All architectures use Side + Rear information as input. (DECIDED)
- The required outputs are bounding box, sex, and weight. (DECIDED)
- Android/Kotlin is the final deployment target. (DECIDED)
- Real-time inference is a design constraint. (DECIDED)
- Parameter count must be reported. (DECIDED)
- Inference latency must be reported. (DECIDED)
- Hardware used for inference must be reported. (DECIDED)
- YOLO26 is the first architecture family being investigated. (DECIDED)
- Within YOLO26, YOLO26n is the PRIMARY CANDIDATE. (PRIMARY CANDIDATE)
- Within YOLO26, YOLO26s is an ALTERNATIVE CANDIDATE. (ALTERNATIVE)
- Final architecture selection must be experimentally validated. (DECIDED)


## 12. Decisions Intentionally Deferred

The following decisions remain TBD and will be determined during architecture selection and experimental validation:

- Exact YOLO feature layer and layer selection
- Exact feature extraction strategy per family
- Whether Side and Rear branches share weights or are independent
- Exact fusion mechanism and its parameterization
- Exact tensor dimensions at fusion and heads
- Detection-head configuration and anchor/parameterization choices
- Sex-classification head architecture
- Weight-regression head architecture
- Loss functions and task-weighting scheme
- Input resolution(s) to be used in final experiments
- Data-augmentation specifics and hyperparameters
- Training hyperparameters (optimizer, learning rate, batch size, number of epochs)
- Android inference runtime and quantization strategy
- Target smartphone hardware and minimum supported specs
- Pruning and compression strategies
- Final selection between YOLO26n and YOLO26s (if both retained)

These decisions will be made only after controlled experiments and cost/benefit analyses.


## 13. Experimental Validation

Architecture selection will be driven by experimental evaluation across the following categories.

Detection metrics:
- IoU
- mAP
- Precision
- Recall

Sex classification metrics:
- Accuracy
- Precision
- Recall
- F1

Weight regression metrics:
- MAE (kg)
- RMSE (kg)
- R²

Deployment metrics:
- Parameter count
- Model size
- FLOPs (when available)
- Inference latency (ms)
- FPS
- Hardware used
- Peak memory usage (when measurable)

The final selection will consider predictive performance together with model complexity and inference speed. There are currently NO experimental results for the proposed architectures; all candidate evaluations will be recorded and referenced here when available.


## 14. Related Documentation

- [MODEL_REQUIREMENTS.md](C:/Repos/livestock-vision-ai/docs/MODEL_REQUIREMENTS.md): Defines the common requirements for all architectures (inputs, outputs, evaluation metrics, data-leakage constraints, mobile constraints).
- [TRAINING_PROTOCOL.md](C:/Repos/livestock-vision-ai/docs/TRAINING_PROTOCOL.md): Will define the standardized training and evaluation protocol that all architectures must follow.

This document focuses on the technical rationale for candidate families and how they differ; it references common requirements rather than duplicating them.


## 15. Decision Status Summary

| Decision | Status |
|----------|--------|
| Dual-view Side + Rear | DECIDED |
| Multi-task outputs | DECIDED |
| Android/Kotlin deployment | DECIDED |
| Real-time requirement | DECIDED |
| YOLO as Architecture 1 family | DECIDED |
| YOLO26n | PRIMARY CANDIDATE |
| YOLO26s | ALTERNATIVE |
| Architecture 2 | TBD |
| Architecture 3 | TBD |
| Fusion mechanism | TBD |
| Weight sharing | TBD |
| Prediction heads | TBD |
| Training hyperparameters | TBD |
| Android runtime | TBD |
| Final architecture selection | TO BE EXPERIMENTALLY VALIDATED |


---

**Document status:** Architecture selection record (candidates and rationale).  
**Next steps:** populate Architecture 2 and Architecture 3 with selected families after initial exploration; perform controlled experiments per TRAINING_PROTOCOL.md and record results in this document.

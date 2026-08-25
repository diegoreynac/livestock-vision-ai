# Training Protocol

## 1. Purpose

This document defines the common experimental protocol for the livestock weight-estimation thesis. It specifies how the three candidate architecture families described in [ARCHITECTURE_SELECTION.md](ARCHITECTURE_SELECTION.md) will be trained, validated, compared, and eventually evaluated for deployment on Android/Kotlin hardware.

A common protocol is required because the project aims to compare distinct neural-network families under a fair and scientifically defensible experimental setup. Architecture-specific implementation details may differ, but dataset usage, split methodology, preprocessing, augmentation, loss design, evaluation metrics, and reporting standards must remain consistent unless a difference is explicitly documented as an experimental factor. Without this standardization, differences in performance would be confounded by differences in dataset handling, training choices, or evaluation logic.

This protocol therefore separates:

- decisions that are already established by the project,
- protocol decisions that can be set now, and
- hyperparameters that remain TBD and must be selected through controlled experiments.

The goal is not to claim that a particular model is already optimized, but to ensure that every comparison is reproducible, leakage-safe, and traceable.

---

## 2. Experimental Problem Definition

The common task addressed by all architecture families is multi-task prediction from a paired animal observation.

Input:

- Side-view image
- Rear-view image

Outputs:

- Bounding box
- Sex classification (F / M)
- Weight regression (kg)

Conceptual pipeline:

```text
Side + Rear
    |
    v
Architecture
    |
    v
Feature extraction / fusion
    |
    +---------------------------+
    |                           |
    v                           v
BBox                        Sex
                              |
                              v
                           Weight
```

The exact implementation of each branch and the fusion strategy are architecture-specific and are defined in [ARCHITECTURE_SELECTION.md](ARCHITECTURE_SELECTION.md). The common training protocol does not prescribe a single implementation; it prescribes the shared experimental conditions under which all implementations will be trained and compared.

---

## 3. Dataset Definition

The project uses a livestock image dataset that includes multiple images and views associated with individual animals. The dataset is the same for all architecture candidates and must be treated as the shared source of truth for training, validation, and final evaluation.

Relevant established project information from [DATASET_JUSTIFICATION.md](DATASET_JUSTIFICATION.md):

- Total dataset images: 4,910
- Images with valid COCO annotations: 4,749
- Images without annotation: 161
- Sex labels available for the dataset
- Side-view and rear-view image pairs are present in the collection
- B2 and B4 annotation groupings are present in the repository documentation
- Weight labels are available for the animals represented in the dataset

The dataset contains:

- Side images
- Rear images
- Associated metadata for sex and weight
- Annotation structure that may include B2/B3/B4 groups depending on the dataset subset and repository organization

The exact dataset location and repository loading mechanism are not treated as training decisions; they are a data-management concern. What matters for the protocol is that all evaluated architectures use the same underlying dataset and the same split logic.

Important dataset considerations:

- The data are not a collection of independent images only; they are associated with animals.
- Multiple views of the same animal are correlated and cannot be treated as independent samples.
- Weight is associated with the animal, not independently with each image.
- Bounding-box annotations may be incomplete across the full dataset, so the detection-related components must be handled consistently and transparently.

Therefore, the training pipeline must be designed around an animal-centric understanding of the dataset, even when image-level processing is performed during training.

---

## 4. Data Split Strategy

This section is one of the central safeguards against leakage.

### 4.1 Split principle

All images and views belonging to the same animal must belong to exactly one split.

This means the split must operate at the animal level whenever the animal identity can be reliably determined from the dataset structure or metadata.

Correct examples:

```text
Animal 001
  ├── Side ──> TRAIN
  └── Rear ──> TRAIN
```

```text
Animal 001
  ├── Side ──> TEST
  └── Rear ──> TEST
```

Incorrect example:

```text
Animal 001
  ├── Side ──> TRAIN
  └── Rear ──> TEST
```

This incorrect pattern creates leakage because the same animal appears in both training and evaluation data. In a dual-view setting, side and rear images are strongly related and are not independent observations in a statistical sense. If the same animal appears in multiple splits, the model can learn animal-specific appearance patterns that do not generalize to unseen animals.

### 4.2 Required splits

The protocol requires three disjoint subsets:

- Training set: used to fit model parameters
- Validation set: used for model selection, hyperparameter decisions, and early stopping criteria when applicable
- Test set: held out for final evaluation only

The exact train/validation/test percentages are currently TBD and must be defined before the formal experimental campaign. The tractable decision is not the exact numerical ratio alone, but the requirement that the split be leakage-safe and suitable for repeated architecture comparison.

### 4.3 Split constraints

The split procedure must enforce the following:

- No animal appears in more than one split
- Side and rear images for the same animal remain in the same split
- The validation set is used only for model selection and tuning
- The test set is completely isolated from architecture selection and hyperparameter tuning
- The final test set must not be used for augmentation decisions, threshold tuning, training-stop tuning, or model ranking

### 4.4 Data balancing and stratification

The protocol will consider dataset imbalance in sex distribution and annotation coverage. The project has already established that the dataset is imbalanced with respect to sex, with females outnumbering males in the audited data. Therefore, stratification or balancing strategies may be desirable when they are part of a scientifically valid training protocol, but they must be applied consistently and recorded transparently.

The split methodology should therefore be documented as:

- animal-level split, with sex-balanced or sex-aware stratification where feasible,
- annotation-aware handling when some samples lack valid bounding boxes,
- consistent application across all candidate architectures.

Exact rules for stratification are TBD and must be finalized before training experiments commence.

---

## 5. Reproducibility

Reproducibility is essential for a thesis comparing multiple architecture families.

The protocol must record, for every experiment:

- random seed (TBD at the time of experiment definition)
- Python version
- deep-learning framework version
- CUDA / device version when relevant
- operating system version
- hardware specification used for training
- exact model architecture and variant
- dataset split identifier
- image preprocessing configuration
- augmentation configuration
- optimizer configuration
- learning-rate schedule
- batch size
- epoch count
- loss configuration and weighting
- checkpointing policy
- validation metrics at each epoch
- final test metrics
- timestamp and experiment ID

Where practical, model training should be deterministic. However, some library behavior may still be nondeterministic under particular hardware and software combinations. The protocol should therefore state that full determinism is a target rather than a guaranteed property, and that all necessary settings must be recorded when deterministic training is attempted.

The final experiment log should make every reported result traceable to:

1. the trained model,
2. the dataset split,
3. the software stack,
4. the training configuration, and
5. the evaluation artifacts.

The exact seed value and reproducibility settings remain TBD until the final training configuration is defined.

---

## 6. Input Preprocessing

All candidate architectures must accept a shared input preprocessing pipeline to ensure comparability.

The processing pipeline conceptually follows this sequence:

```text
Original image
    |
    v
Load image
    |
    v
Resize / normalize / transform
    |
    v
Model input tensor
```

The following preprocessing steps are relevant and must be documented for every experiment:

- image loading format (e.g., RGB or BGR as used by the framework)
- image resolution and resizing strategy
- aspect-ratio handling (letterboxing, padding, or reshape)
- normalization strategy (e.g., ImageNet statistics or custom statistics)
- tensor conversion and dtype
- channel ordering
- bounding-box coordinate transformation if the image is resized or cropped

Important protocol requirement:

Any preprocessing that changes spatial geometry must also be applied consistently to all target annotations, especially bounding boxes. If an image is resized, cropped, padded, or otherwise transformed, the corresponding target coordinates must be updated to remain valid relative to the transformed image.

The exact input resolution is not finalized by this protocol. Architecture-specific native/reference resolutions may differ, but the preprocessing methodology used for each experiment must be clearly documented and consistent across all candidates within a given comparison experiment.

This means the final protocol can state:

- image resolution is a controlled experimental variable,
- all architecture families must be trained under the same documented resolution policy whenever the comparison is meant to be fair,
- deviations are allowed only when explicitly declared as an architecture-specific adaptation and are not hidden as a common-protocol result.

---

## 7. Data Augmentation

The project explicitly requires the use of data augmentation. In this task, augmentation is necessary because the dataset may contain limited pose variation, lighting variability, and appearance differences across animals, while the model must generalize to real-world smartphone capture conditions.

Augmentation should be viewed as a regularization and robustness tool, not as a replacement for clean data collection or proper labeling.

### 7.1 Augmentation categories likely to be considered

Potential augmentations include:

- horizontal flip where semantically valid
- small rotations
- translation
- scaling
- brightness and contrast adjustments
- mild crop/resize operations
- noise or blur where justified

### 7.2 Augmentation rules

Not every augmentation is appropriate for every task or every image type. Each augmentation must be evaluated in terms of:

- purpose,
- potential benefit,
- potential risk,
- semantic validity.

For side/rear livestock imagery, augmentation must not change the underlying meaning of the sample. For example:

- horizontal flipping may be acceptable only if the annotation semantics remain valid after transformation;
- excessive cropping may remove biologically relevant body structure;
- extreme rotations may produce unrealistic animal poses;
- augmentation that destroys the side/rear pairing relationship is not acceptable unless the experiment explicitly defines and documents such a change.

Therefore, the protocol requires that any augmentation affecting geometry must also update bounding boxes and related label geometry consistently.

### 7.3 Augmentation policy status

The exact augmentation list, probabilities, and magnitudes remain TBD. They must be defined before the final experimental campaign and then applied consistently across the architecture families. The protocol may allow a small set of controlled augmentation experiments, but the chosen configuration must be documented and not changed silently during final evaluation.

---

## 8. Target Preparation

The supervised training targets must be prepared consistently for all models.

### 8.1 Bounding box

A bounding-box target is required for the detection component. The protocol must standardize:

- coordinate representation,
- origin convention,
- normalized vs pixel-space target definition,
- handling of padded or resized images,
- conversion between image coordinates and model coordinates.

The exact coordinate convention is not finalized by this document and is therefore TBD unless already defined elsewhere in the project. What is required is a single consistent convention across all experiments.

### 8.2 Sex

Sex is a binary classification target.

The model must classify the animal as one of:

- F (Female)
- M (Male)

The exact encoding strategy (for example, one-hot vector vs single binary output) is an implementation choice and is not a protocol decision by itself. The representation must be documented in the experiment log.

### 8.3 Weight

Weight is a continuous regression target in kilograms.

The dataset contains validated weight values for the animals represented in the data, with a range documented in the dataset audit. The protocol requires:

- weight to be treated as a regression target,
- weight labels to remain unchanged by augmentation,
- weight to never be used as an input feature to the model,
- training labels and evaluation labels to come from the same recorded ground truth.

Important constraint: ground-truth weight must never be provided as an input feature to the model. It is a supervised target, not an observed model input.

---

## 9. Loss Functions

The prediction problem is multi-task. The total training objective is conceptually:

```text
Total Loss = λ_bbox * L_bbox + λ_sex * L_sex + λ_weight * L_weight
```

This formulation reflects the fact that the model must simultaneously optimize:

- bounding-box localization,
- sex classification,
- weight regression.

The exact loss functions and loss weights are not yet finalized. Candidate choices may include:

- bounding-box regression loss (for example IoU-based or coordinate-based variants),
- classification loss for sex,
- regression loss for weight.

The final selected losses remain TBD and must be justified by validation behavior rather than by test-set inspection. The loss weighting strategy must also be determined before final test-set evaluation.

This section thus distinguishes:

- candidate losses: TBD but allowed for exploratory experimentation,
- selected loss functions: TBD,
- final loss weights: TBD.

The exact configuration must be recorded in the experiment metadata for every run.

---

## 10. Training Strategy

The common training strategy must be defined at a level that is consistent across the three architecture families while allowing architecture-specific implementation details.

This protocol covers the following elements:

- training from scratch vs transfer learning
- pretrained backbone use where appropriate
- freezing versus unfreezing strategy
- fine-tuning schedule
- optimizer choice
- learning-rate policy
- scheduler choice
- batch size
- maximum epochs
- early stopping criteria
- checkpoint selection policy

### 10.1 Training initialization

Training may begin from:

- random initialization,
- pretrained weights from a relevant source, or
- a partially pretrained backbone if the architecture supports it.

The chosen strategy is not fixed by this protocol and must be documented explicitly for each experiment. If transfer learning is used, the experiment log must record which layers were frozen, which were fine-tuned, and how the training schedule evolved.

### 10.2 Optimization

Optimizer, learning rate, weight decay, and scheduler are all TBD at the protocol level and must be selected during controlled experimentation. The exact values used in the final thesis must be reported with the corresponding measurements and not hidden in external configuration files.

### 10.3 Epochs and early stopping

The number of training epochs and the early-stopping rule remain TBD. A practical protocol will generally rely on monitoring validation metrics and selecting the best checkpoint according to a documented criterion. However, the final rule must be defined before the final held-out test evaluation.

### 10.4 Training comparability

The exact training configuration must be standardized as much as possible within the experimental campaign. If an architecture-specific constraint requires a different optimizer or batch size, that must be documented as a justified variation and should be limited to the minimum required for that family.

---

## 11. Architecture-Specific Training Considerations

The shared protocol applies to all three architecture families, but implementation details may differ because each family has different inductive biases and practical training requirements.

### 11.1 YOLO-based architectures

YOLO-based models are detection-oriented and often rely on multi-scale detection heads and dense prediction structures. In this project, they are treated as a candidate family for dual-view, multi-task learning rather than as a direct off-the-shelf detector for the animal dataset.

Training considerations include:

- detection head integration,
- bounding-box prediction format,
- adaptation of YOLO-style feature extraction for dual-view fusion,
- fusion of detection representation with sex and weight heads,
- possible need for task-specific scaling or anchor-related design decisions.

The exact implementation is not stated here and remains a design decision captured in the architecture-selection process.

### 11.2 MobileNet-based architectures

MobileNet-based candidates will generally prioritize lightweight feature extraction and deployment efficiency. Their training protocol must still follow the common data split, evaluation standards, and logging requirements.

Relevant considerations include:

- backbone selection (V2, V3-Small, V3-Large),
- whether the two branches share weights or use independent feature extractors,
- fusion strategy between the side and rear branches,
- multi-task head design,
- tradeoff between model compactness and predictive quality.

### 11.3 EfficientNet-based architectures

EfficientNet-based candidates offer a capacity-efficiency tradeoff that is relevant to mobile deployment. Their training protocol must follow the same fairness constraints as the other families.

Relevant considerations include:

- EfficientNet variant selection (B0, B1, B2 as candidates in the architecture-selection phase),
- fusion design,
- multi-task output heads,
- shared vs independent branch weights,
- model size and latency compared to the required deployment constraints.

The protocol does not prescribe a single implementation for any family; it only establishes the common experimental rules under which they must be compared.

---

## 12. Evaluation Metrics

Evaluation must be performed using metrics that align with the three tasks and the thesis objective. The protocol must remain consistent across all candidate architectures.

### 12.1 Bounding-box evaluation

The bounding-box task is evaluated using detection metrics. At a conceptual level:

- IoU (Intersection over Union) measures overlap between predicted and ground-truth boxes:

```text
IoU = intersection area / union area
```

- AP (Average Precision) summarizes precision-recall behavior for a detection task.
- mAP (mean Average Precision) aggregates AP over one or more IoU thresholds and/or classes.

The exact IoU thresholds and AP/mAP configuration remain TBD and must be specified before the final evaluation run. The final configuration must be applied consistently across all architectures.

### 12.2 Sex classification

Sex prediction is evaluated with standard classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The sex distribution is imbalanced, so all metrics should be reported carefully. The imbalance itself is a known property of the dataset and must be acknowledged in the interpretation of results.

### 12.3 Weight regression

Weight is evaluated by regression metrics:

- MAE: mean absolute error in kilograms
- RMSE: root mean squared error, which penalizes large errors more strongly
- R²: coefficient of determination, indicating explained variance relative to a baseline

The final thesis should report these metrics with the same evaluation protocol for all architectures.

### 12.4 Multi-task evaluation

The project is not evaluating a single metric in isolation. A model is not considered better merely because it achieves a lower MAE or a higher mAP. The architecture comparison should consider the joint behavior across:

- detection quality,
- sex classification,
- weight estimation,
- computational cost,
- deployment efficiency.

This requirement ensures that the final selection is based on the multi-objective problem, not only on a single headline metric.

---

## 13. Model Complexity Reporting

This section directly addresses the requirement to report parameter counts and related model complexity information.

Every experiment must record:

- trainable parameters
- non-trainable parameters
- total parameters
- model file size
- optional FLOPs or MACs where available

This reporting is essential because the final architecture decision is not only a predictive-performance decision; it is also a mobile-deployment decision. A model with slightly better error metrics but substantially worse parameter count or inference cost may not be the best choice for a real-time Android application.

Important distinction:

- reference backbone parameter counts are not the final thesis result,
- the final thesis comparison must use the parameter counts measured from the actual implemented dual-view multi-task model.

Therefore, the final experiment table must report the parameter count for the actual model used in training and inference, not merely the published backbone statistics from an external benchmark.

---

## 14. Hardware and Latency Evaluation

The project must separate training hardware from deployment hardware.

### 14.1 Training hardware

The training environment must be documented for every experiment, including:

- CPU type
- GPU type
- RAM capacity
- software framework and version
- relevant driver/library versions
- training environment identifier

### 14.2 Deployment hardware

Deployment evaluation must be performed on Android hardware relevant to the target application:

- smartphone model
- SoC
- CPU/GPU/NPU details when available
- Android version
- runtime environment
- model precision state (FP32, FP16, INT8, etc.)
- quantization state if applicable

### 14.3 Latency and throughput reporting

Latency measurements must be performed on the target Android hardware, not only on a desktop workstation. At minimum, the evaluation should report:

- average latency
- median latency (optional but recommended)
- p95 latency (optional)
- number of runs
- warm-up behavior
- batch size
- FPS when relevant for real-time inference

The exact target hardware and the exact latency measurement procedure remain TBD until the deployment phase is executed. However, the methodology must be consistent and must be recorded for each architecture comparison.

---

## 15. Mobile Deployment Protocol

The final training protocol is not isolated from deployment. The models are intended for Android Studio/Kotlin inference on a smartphone with real-time requirements.

Conceptually, the deployed pipeline is:

```text
Python training
    |
    v
Trained model
    |
    v
Export / conversion
    |
    v
Mobile-compatible runtime format
    |
    v
Android Studio / Kotlin integration
    |
    v
Camera or input acquisition
    |
    v
Side + Rear inference
    |
    v
BBox + Sex + Weight
```

The exact runtime remains TBD. Potential choices include:

- TensorFlow Lite
- ONNX Runtime
- another compatible Android runtime

The final runtime must be selected based on:

- operator compatibility,
- conversion reliability,
- latency,
- memory footprint,
- accuracy preservation,
- Android integration feasibility.

This decision is a deployment concern and is not resolved by the training protocol alone. However, it must be aligned with the architecture-selection and final evaluation strategy.

---

## 16. Experiment Matrix

The thesis must compare the architecture families under a shared matrix of experimental conditions. The table below shows the required reporting structure; values are intentionally TBD until they are experimentally measured.

| Architecture | Variant | Weight Sharing | Pretraining | Input Resolution | Augmentation | Parameters | mAP | IoU | Sex Acc. | MAE | RMSE | R² | Latency | Hardware |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO | YOLO26n | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLO | YOLO26s | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MobileNet | V2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MobileNet | V3-Small | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MobileNet | V3-Large | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| EfficientNet | B0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| EfficientNet | B1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| EfficientNet | B2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

The B2 variant may be used as an optional sensitivity or ablation candidate, depending on the final experimental design. All entries above are examples of the required comparison structure; they are not results.

---

## 17. Model Selection Criteria

The final model will not be selected using a single arbitrary weighted score.

Instead, the architecture selection process will consider the joint tradeoff among:

1. weight estimation performance,
2. bounding-box performance,
3. sex classification performance,
4. parameter count,
5. model size,
6. inference latency,
7. memory usage,
8. Android deployment feasibility.

This is consistent with the project goal of balancing predictive performance with real-time on-device feasibility. A model that performs slightly better in one metric but fails the deployment constraints is not automatically the best final choice.

If a formal composite score is later desired, it must be justified and defined before the test-set evaluation begins. The score must not be engineered retroactively to favor one architecture after seeing the final metrics.

---

## 18. Test Set Protocol

The test set is the final held-out evaluation set. This is a critical safeguard against data leakage and overfitting.

Once the final architecture family, hyperparameters, and training regime have been selected:

- the test set is used for final evaluation only,
- the test set must not be used for architecture selection,
- the test set must not be used for hyperparameter tuning,
- the test set must not be used for augmentation tuning,
- the test set must not be used for threshold tuning,
- the test set must not be used for repeated model ranking.

All final reported thesis metrics must come from the held-out test evaluation under the pre-defined final evaluation procedure. If the final evaluation procedure requires multiple runs, that procedure must be specified in advance and then followed consistently.

---

## 19. Experiment Logging and Artifacts

Every experiment must produce a complete and traceable record of the configuration and outcomes. At minimum, the project should log:

- model name
- model variant
- random seed
- dataset split identifier
- training configuration
- augmentation configuration
- optimizer
- learning rate
- batch size
- epochs
- loss configuration
- parameter count
- validation metrics
- test metrics
- checkpoint path
- hardware summary
- software versions
- timestamp
- inference latency when available

The purpose of this logging is to ensure that every reported result is reproducible, explainable, and auditable. A thesis comparison is only scientifically credible if the evaluation pipeline is fully documented and traceable.

---

## 20. Current Decisions vs TBD

### 20.1 Decisions already established

The following decisions are already established by the project and are therefore fixed for the protocol:

- Side + Rear input pair
- Bounding box + sex + weight outputs
- Same underlying dataset for all architectures
- Leakage-safe split principle
- Data augmentation is required
- Three architecture families will be compared
- Android/Kotlin deployment target
- Real-time inference requirement
- Multi-objective evaluation strategy
- Test set remains isolated from model selection

### 20.2 Protocol decisions that can be defined now

The following decisions can be defined now within the common protocol without inventing results:

- The common pipeline will operate on paired side/rear inputs.
- All architectures must be evaluated under a shared dataset and split logic.
- All experiments must report the same core metrics and complexity measures.
- All experiments must use a leakage-safe animal-level split when possible.
- All experiments must maintain consistent annotation handling and transformation logic.
- The final evaluation procedure must remain test-set isolated.

### 20.3 Hyperparameters and protocol details still TBD

The following details remain TBD and must be finalized through controlled experiments before the final evaluation campaign:

- exact train/validation/test percentages
- exact input resolution
- exact augmentation list and probabilities
- exact loss functions
- exact loss weights
- optimizer
- learning rate
- scheduler
- batch size
- epochs
- early stopping rule
- pretrained vs from-scratch policy
- freezing and fine-tuning schedule
- exact IoU/mAP thresholds
- Android runtime
- target smartphone hardware
- quantization strategy
- final model-selection procedure
- exact random seed

These items are intentionally left open to avoid pretending that the protocol has already been experimentally validated.

---

## 21. Related Documentation

This document sits between the high-level requirements and the actual implementation.

```text
MODEL_REQUIREMENTS
    |
    v
Common problem definition, constraints, and deployment expectations

ARCHITECTURE_SELECTION
    |
    v
Candidate architecture families and variants

TRAINING_PROTOCOL
    |
    v
Shared experimental protocol for training, validation, and evaluation

Implementation
    |
    v
Model training and evaluation pipeline

Deployment
    |
    v
Android/Kotlin inference and mobile integration
```

The relationships are as follows:

- [MODEL_REQUIREMENTS.md](MODEL_REQUIREMENTS.md) defines what the system must do.
- [ARCHITECTURE_SELECTION.md](ARCHITECTURE_SELECTION.md) defines which architecture families are under consideration.
- This document defines how those architectures will be trained and compared under a fair protocol.
- The implementation step then turns the protocol into executable experiments and final model selection.

---

## Final Requirements

Before finishing this protocol, the project must confirm the following:

1. Only [docs/TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md) was created or modified.
2. No additional files were created.
3. No source code was modified.
4. No experiments were run.
5. No numerical results were invented.
6. Unresolved hyperparameters were not silently selected.
7. All unresolved decisions are explicitly marked as TBD.
8. The document remains consistent with [MODEL_REQUIREMENTS.md](MODEL_REQUIREMENTS.md) and [ARCHITECTURE_SELECTION.md](ARCHITECTURE_SELECTION.md).
9. The distinction between reference values and thesis measurements is clear.
10. The document is detailed enough that the author can later explain why each training and evaluation decision was made.

This protocol is intended to be a rigorous and fair foundation for the thesis experiment, not a final claim that any architecture has already been proven superior.

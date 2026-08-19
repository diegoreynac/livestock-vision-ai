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


## 6. Architecture 2 — MobileNet-based

### 6.1 Why MobileNet

MobileNet is relevant to this thesis because it was designed for computationally constrained and mobile-oriented environments. It provides a lightweight convolutional backbone that can be used to extract visual features with a substantially smaller computational footprint than heavier classification or detection backbones. This is important because the final application must eventually run on Android using Kotlin and must support practical real-time inference.

The comparison is therefore not simply whether a model is more accurate in a benchmark setting. The central question is which architecture offers the most suitable trade-off between:

- predictive performance
- parameter count
- computational complexity
- model size
- memory footprint
- inference latency
- mobile deployment feasibility

MobileNet is therefore a meaningful contrast to YOLO. YOLO is primarily associated with detection-first pipelines and object localization, while MobileNet is a mobile-backbone-first feature extractor that can be adapted to dual-view, multi-task prediction. This makes MobileNet relevant for a thesis problem where detection is only one objective among several tasks, and deployment feasibility is a major constraint.

At this stage, MobileNet is a candidate architecture under consideration, not a proven winner.

### 6.2 MobileNet Family

The MobileNet family is a collection of lightweight CNN architectures designed for efficient feature extraction in resource-limited environments. In the context of this thesis, the family is attractive because it allows us to compare a lightweight mobile-oriented backbone against a detection-oriented architecture (YOLO) under the same dual-view multi-task setting.

The family includes several variants with different trade-offs. For this thesis, the primary comparison is between:

- MobileNetV2
- MobileNetV3-Small
- MobileNetV3-Large

These variants differ in depth, width, block structure, and computational cost. The purpose of comparing them is not to claim a final winner, but to identify a MobileNet architecture that best balances performance and deployment efficiency.

### 6.3 MobileNetV2

MobileNetV2 is an established mobile CNN architecture based on inverted residual blocks and linear bottlenecks. Its lightweight design and efficient depthwise separable structure make it a useful baseline/reference for mobile-oriented feature extraction.

Why it is relevant:

- established lightweight CNN backbone
- strong reference for efficient feature extraction
- useful baseline for comparing lightweight mobile design choices with more recent MobileNetV3 variants

MobileNetV2 remains a useful comparison point, but it is not automatically selected as the final architecture. It serves as a reference for what a mobile backbone can provide under constrained computational budgets.

### 6.4 MobileNetV3-Small

**Primary candidate — TO BE EXPERIMENTALLY VALIDATED**

MobileNetV3-Small is the primary candidate within Architecture 2 because it is optimized specifically for lightweight and mobile inference. It is designed to maintain low computational cost while providing competitive feature representation quality.

Why MobileNetV3-Small is attractive for this thesis:

- smaller capacity than V3-Large
- lower parameter count and memory burden
- lower computational cost than larger mobile backbones
- better alignment with the thesis requirement for real-time inference on Android hardware
- leaves more computational budget for dual-view fusion and multi-task heads

This does not mean MobileNetV3-Small is experimentally superior. It is the most mobile-oriented primary candidate under consideration and will be evaluated against the other architecture families under the common experimental protocol.

### 6.5 MobileNetV3-Large

MobileNetV3-Large is a higher-capacity variant than MobileNetV3-Small. It may provide stronger representational capacity and potentially better feature quality for complex visual tasks, but it also increases computational and memory demands.

Why it is considered:

- more representational capacity than V3-Small
- potentially stronger visual features for complex tasks
- useful as a stronger mobile-oriented comparison point

Why it is not automatically preferred:

- higher computational and memory cost
- potentially less favorable for real-time Android deployment
- may leave less headroom for the dual-view fusion and multi-task prediction pipeline

MobileNetV3-Large remains a relevant alternative, but its final usefulness must be determined experimentally.

### 6.6 MobileNet Comparison

The MobileNet comparison is designed to study an architectural trade-off rather than simply selecting the model with the largest capacity. The decision is not based only on maximum accuracy; it must also consider mobile suitability.

MobileNetV2:
- inverted residual blocks
- linear bottlenecks
- lightweight and efficient design
- established mobile CNN architecture
- useful baseline/reference

MobileNetV3-Small:
- optimized for lightweight/mobile inference
- smaller capacity than V3-Large
- strong candidate when latency and model size are important
- primary candidate for Architecture 2

MobileNetV3-Large:
- higher capacity than V3-Small
- potentially better representation power
- higher computational and memory cost
- useful as a stronger mobile-oriented comparison point

This comparison is intended to support a practical design decision: which MobileNet family offers the best balance between predictive quality and mobile deployment feasibility.

Conceptually, the difference between Architecture 1 and Architecture 2 is important:

Architecture 1: YOLO = detection-first architecture.

Image
  |
YOLO
  |
BBox / detection features
  |
additional task heads / fusion as required

Architecture 2: MobileNet = mobile-backbone-first architecture.

Side --> MobileNet --\
                      > Fusion --> BBox / Sex / Weight
Rear --> MobileNet --/

This creates a meaningful architectural comparison: the YOLO family is designed around localization and detection-oriented representations, while the MobileNet family is designed around efficient feature extraction and mobile deployment. Neither is assumed to be superior before experiments.

### 6.7 Dual-view MobileNet Architecture

The conceptual design for Architecture 2 is a dual-view, multi-task feature-extraction system built around MobileNet backbones.

Side Image
    |
    v
MobileNet Backbone
    |
    v
Side Features
    |
    |
    v
  Fusion
    ^
    |
Rear Features
    ^
    |
MobileNet Backbone
    ^
    |
Rear Image

After feature fusion:

Fused Features
    |
    +----> Bounding Box Head
    |
    +----> Sex Classification Head
    |
    +----> Weight Regression Head

This architecture is:

- dual-view
- multi-task
- feature-fusion based
- designed with mobile deployment in mind

The exact backbone variant, the exact fusion mechanism, and the final prediction-head implementations remain TBD. This section documents the architectural concept, not a finalized implementation.

### 6.8 Shared vs Independent Weights

A major design decision within Architecture 2 is whether the two views share the same MobileNet backbone or use separate backbones.

#### Variant A — Shared Weights

The same MobileNet backbone processes both Side and Rear images.

Conceptually:

Side ----\
           >---- SAME MobileNet weights
Rear ----/

The resulting features are then fused.

Expected advantages:

- approximately one backbone's worth of trainable parameters
- lower model size
- lower memory footprint
- potentially lower inference cost
- attractive for Android deployment

Possible limitation:

- Side and Rear views represent substantially different viewpoints
- a single shared feature extractor may not specialize optimally for both views

This does not mean shared weights are better; it only indicates a possible efficiency advantage worth testing.

#### Variant B — Independent Weights

Side and Rear each use their own MobileNet backbone.

Conceptually:

Side --> MobileNet A
Rear --> MobileNet B

The features are then fused.

Expected advantages:

- each branch can specialize for its viewpoint
- greater representational flexibility

Expected costs:

- approximately two backbone parameter sets
- larger model size
- higher memory requirements
- potentially higher inference latency

This does not mean independent weights are better; it only indicates a possible representational benefit worth testing.

### 6.9 Controlled Weight-Sharing Experiment

The Shared vs Independent question is intended as a controlled experiment within Architecture 2, not as two official architecture families. The experimental question is:

"Does the potential representational benefit of independent Side and Rear backbones justify the additional computational and memory cost compared with a shared-weight backbone?"

The two variants must use the same:

- dataset
- train/validation/test split
- preprocessing
- input resolution
- augmentation policy
- optimizer
- learning-rate policy
- training budget
- evaluation metrics
- hardware and software environment for benchmarking

The intended independent variable is the weight-sharing strategy. No experimental results are claimed here.

### 6.10 Multi-task Prediction Heads

Architecture 2 must support the same multi-task outputs as all architectures in the thesis.

Fused Features
    |
    +--> BBox Head
    +--> Sex Head
    +--> Weight Head

These outputs are:

1. Bounding box
2. Sex classification (F / M)
3. Weight regression in kilograms

The exact final head architecture remains TBD and will be defined in the implementation phase according to data annotations, task-specific supervision, and results of experimental validation.

### 6.11 Parameter and Computational Complexity

Architecture defines the model topology: layers, operations, connections, channel dimensions, and overall structure. Parameters are learned numerical values such as convolution weights, bias values, and linear-layer coefficients. Pre-trained weights are parameter values learned previously on another dataset, and a checkpoint is a saved set of learned parameter values from a training run.

This distinction is important: YOLO and MobileNet both contain trainable parameters and learned weights. The architecture itself is not the same as the learned parameter values stored inside a checkpoint. A model architecture can be the same while the checkpoint differs depending on training state or pretraining history.

Reference complexity values are useful for architecture analysis, but they are not direct measurements from the custom dual-view multi-task thesis model. The final parameter count and computational cost must be measured after the custom architecture is implemented and evaluated under the common protocol.

The following values are reference-only comparisons for context and are not thesis results:

| Reference model | Parameters | Computational reference |
|-----------------|-----------:|------------------------:|
| YOLO26n | ~2.4M | ~5.4 GFLOPs |
| YOLO26s | ~9.5M | ~20.7 GFLOPs |
| MobileNetV2 1.0 | ~3.4M | ~300M MAdds |
| MobileNetV3-Small 1.0 | ~2.5M | ~56M MAdds |
| MobileNetV3-Large 1.0 | ~5.4M | ~219M MAdds |

Important limitations:

- these figures come from reference/original model configurations
- they may differ in input resolution, implementation, task, and head design
- FLOPs and MAdds are not always directly comparable without matching conventions
- detection models and classification models have different computational profiles
- our custom dual-view multi-task architecture includes fusion layers, multiple heads, and the Side + Rear pipeline, which are not captured by a single original-reference backbone number

Therefore the final thesis comparison must measure parameter count and computational cost using our implemented architectures under a common experimental protocol.

### 6.12 Mobile Deployment Considerations

The final application target is Android Studio + Kotlin, and the model must eventually support practical real-time inference. This is central to the architecture-selection decision.

Architecture 2 is therefore evaluated in terms of:

- parameter count
- model size
- memory footprint
- computational cost
- inference latency
- hardware used for benchmarking
- eventual Android runtime compatibility

The exact Android runtime remains TBD. The architecture is being assessed for mobile feasibility, but no final runtime decision is made here.

### 6.13 Architecture 2 Status

Primary candidate:
MobileNetV3-Small

Weight-sharing variants:
- Shared weights
- Independent weights

Status:
Candidate architecture — not experimentally validated.

The final choice between:
- MobileNetV3-Small shared
- MobileNetV3-Small independent

will be based on experimental comparison of predictive performance and mobile-computational efficiency.


## 7. Architecture 3 — EfficientNet-based

### 7.1 Why EfficientNet

EfficientNet is a family of convolutional neural networks that emphasizes a principled balance of depth, width and input resolution via compound model scaling. For this thesis, EfficientNet is relevant because it provides a compact set of backbone variants that are explicitly designed to offer good accuracy–efficiency trade-offs. That design objective makes EfficientNet useful for studying whether a carefully scaled backbone can provide competitive predictive performance while remaining feasible for on-device Android deployment and real-time inference.

EfficientNet is therefore a meaningful third family to compare against the detection-oriented (YOLO) and mobile-backbone-first (MobileNet) approaches. The comparison addresses the multi-objective nature of the thesis: predictive performance (detection, sex, weight) versus deployment efficiency (parameters, size, latency, memory).

Important: references to EfficientNet variant capacities below are reference values from original model configurations and are not experimental results from the thesis dataset.


### 7.2 EfficientNet architecture (conceptual)

At a conceptual level, modern EfficientNet variants share several building blocks and design principles that are relevant to our selection:

- MBConv blocks: mobile inverted bottleneck convolutional blocks that combine depthwise separable convolutions and pointwise expansions. MBConv blocks are efficient at extracting features with lower computational cost than standard convolutions.
- Inverted residual structure: channel expansion followed by depthwise convolution and projection back to a lower-dimensional representation; supports efficient information flow and parameter efficiency.
- Depthwise separable convolution: separates spatial and channel mixing to reduce multiply-add cost.
- Squeeze-and-Excitation (SE): channel-wise recalibration that improves representational power with small overhead; many EfficientNet variants include SE modules inside MBConv blocks.
- Residual connections (where applicable): shortcut connections help gradient flow and stabilize training.

These components affect feature extraction quality and efficiency. For our dual-view multi-task problem, the backbone's ability to extract discriminative features for both shape (useful for weight) and local details (useful for sex and bounding-box cues) is important. EfficientNet's compound-scaling approach means that moving from B0 to B1 to B2 increases capacity and resolution in a controlled manner, allowing an experimental study of marginal gains versus computational cost.


### 7.3 EfficientNet-B0

**Primary candidate — TO BE EXPERIMENTALLY VALIDATED**

Rationale for B0 as primary candidate:

- B0 represents the baseline EfficientNet configuration with modest parameter count and computational cost, making it a practical starting point for mobile-oriented experiments.
- Reference input resolution for B0 is 224×224, which is amenable to mobile inference and helps control latency.
- As a smaller model, B0 leaves computational headroom for dual-view processing (two backbones or shared backbone), fusion layers, and multi-task heads while improving the chance of meeting real-time constraints on Android devices.

These properties make EfficientNet-B0 an appropriate primary candidate for Architecture 3, but this selection is provisional and must be validated experimentally on our livestock dataset.


### 7.4 EfficientNet-B1

**Secondary candidate — TO BE EXPERIMENTALLY VALIDATED**

Rationale for B1 as a controlled step up in capacity:

- EfficientNet-B1 increases model capacity and uses a reference input resolution of 240×240. This provides a controlled increase in both representational power and spatial resolution relative to B0.
- B1 is a useful secondary candidate to test whether modest increases in capacity and resolution yield meaningful improvements for the multi-task livestock problem that justify the additional computational and memory cost during deployment.

B1 should be evaluated experimentally as a controlled capacity step above B0, not as a presumption of superiority.


### 7.5 EfficientNet-B2

**Optional sensitivity/ablation candidate — TO BE EXPERIMENTALLY VALIDATED**

Rationale for considering B2 as an optional experiment:

- EfficientNet-B2 uses a reference input resolution of 260×260 and, in the original/reference models, has approximately 9.2M parameters.
- B2 offers greater capacity and higher-resolution inputs compared with B0 and B1, which may help tasks that benefit from spatial detail (for example, fine-grained sex cues or small-scale bounding-box precision).
- However, B2 is more computationally and memory expensive and therefore less immediately attractive for strict mobile real-time constraints.

Therefore B2 is documented as an OPTIONAL sensitivity or ablation experiment rather than a primary candidate. It can be used to probe whether further capacity/resolution improves performance enough to justify increased deployment cost.


### 7.6 Reference comparison table

The following table presents reference architectural values for context. These values are reference-only and must not be interpreted as measurements of our custom dual-view multi-task models.

| Criterion | EfficientNet-B0 | EfficientNet-B1 | EfficientNet-B2 |
|---|---:|---:|---:|
| Reference parameters | ~5.3M | ~7.8M | ~9.2M |
| Reference input resolution | 224×224 | 240×240 | 260×260 |
| Relative capacity | Low | Medium | Medium–High |
| Computational cost | Low | Medium | Medium–High |
| Mobile deployment expectation | Lower cost | Moderate cost | Higher cost |
| Primary role | PRIMARY | SECONDARY | OPTIONAL |
| Thesis experimental result | TBD | TBD | TBD |

Limitations of these reference values:

- They reflect the original single-image classification configurations and do not include dual-view fusion, multi-task heads, or any custom modifications.
- Input resolution, head design, and implementation details affect final parameter counts and computational cost.
- FLOPs/MAdds conventions may differ between sources and are not directly compared here.

Final parameter counts and computational costs for our dual-view multi-task models must be measured after implementation and will be reported as experimental results.


### 7.7 Implications for the livestock task (hypotheses)

Increasing model capacity and input resolution could plausibly help several aspects of the livestock problem:

- Animal body shape: higher-resolution features may capture finer shape cues correlated with mass
- Spatial information: larger input resolutions preserve more spatial detail useful for both localization and fine-grained appearance cues
- Bounding-box prediction: improved localization precision may benefit from higher-resolution feature maps
- Sex classification: subtle morphological cues may be easier to discriminate at higher resolution/capacity
- Weight regression: richer multi-scale features may better capture the visual correlates of weight

These are hypotheses to be tested. Increased capacity or resolution does not guarantee improved performance; empirical validation is required to quantify trade-offs between predictive gains and deployment cost.


### 7.8 Dual-view EfficientNet conceptual design

Conceptual dataflow (dual-backbone view):

Side image
    ↓
EfficientNet backbone
    ↓
Side feature representation
    ↓
      Fusion
    ↑
Rear feature representation
    ↑
EfficientNet backbone
    ↑
Rear image

After fusion:

Fusion
  ↓
┌───────────────┬───────────────┬───────────────┐
↓               ↓               ↓
BBox            Sex             Weight

This mirrors the dual-view patterns used in Architectures 1 and 2. The exact fusion mechanism remains TBD and will be selected and justified experimentally.


### 7.9 Shared vs Independent Weights (EfficientNet)

As with Architecture 2, two weight-sharing strategies are considered for EfficientNet:

- Shared weights: one EfficientNet backbone processes both Side and Rear inputs (weight sharing). This reduces parameter count and model size and is attractive for mobile deployment, but may limit view-specific specialization.

- Independent weights: separate EfficientNet backbone instances for Side and Rear. This allows per-view specialization at the cost of increased parameters, memory and inference cost.

The trade-offs are the same conceptual ones described for MobileNet: shared weights reduce resource usage; independent weights increase representational flexibility. This is a controlled experimental factor and must be evaluated empirically.


### 7.10 Parameter implications for dual-view deployment

Conceptually, the total model parameter count depends on whether backbones are shared:

- Shared B0: approximately one B0 backbone + fusion layers + BBox/Sex/Weight heads (conceptual)
- Independent B0: approximately two B0 backbones + fusion layers + heads

The same conceptual difference applies to B1 and B2. These are conceptual descriptions: final total parameter counts will depend on implementation choices (fusion architecture, head sizes, use of pre-trained layers) and must be measured after implementation.


### 7.11 Mobile deployment implications

When evaluating EfficientNet variants for on-device Android deployment, the following practical considerations apply:

- Model size and parameter count influence binary size and storage requirements
- Input resolution affects memory usage and per-inference compute cost (higher resolutions increase both)
- Peak RAM usage during inference constrains deployment on lower-end devices
- Quantization and pruning strategies may reduce size and latency but require separate validation
- Actual inference latency depends on runtime, hardware acceleration (CPU, GPU, NNAPI), and implementation details

Do not infer latency solely from parameter counts; latency must be measured on representative Android hardware and reported as part of experimental results.


### 7.12 Transfer learning and pretraining

EfficientNet backbones are commonly available with ImageNet-pretrained weights. A practical strategy for the thesis is to initialize convolutional feature extractors from ImageNet-pretrained checkpoints, remove or omit the original classifier head, and attach our fusion module and multi-task heads:

ImageNet pretrained EfficientNet
    ↓
remove/omit classifier
    ↓
feature extractor
    ↓
fusion module
    ↓
BBox / Sex / Weight heads

Whether to freeze early layers, fine-tune the full backbone, or adopt staged unfreezing is a training-protocol decision and belongs in TRAINING_PROTOCOL.md. Transfer learning can accelerate convergence and improve initial performance, but final strategies must be experimentally validated.


### 7.13 B0 vs B1 vs B2 decision rationale

Summary decision roles (preliminary):

- EfficientNet-B0: PRIMARY candidate — chosen for its parameter efficiency and mobile suitability; expected to be the most deployment-friendly starting point.
- EfficientNet-B1: SECONDARY candidate — a controlled upward step in capacity and resolution to measure marginal gains versus cost.
- EfficientNet-B2: OPTIONAL sensitivity/ablation candidate — higher capacity and resolution for probing whether further scaling yields practical benefits.

Rationale:

- The primary priority for the thesis is to find architectures that achieve acceptable predictive performance while meeting mobile deployment constraints. B0 addresses this priority directly.
- B1 provides a nearby capacity increase to test whether modest scaling yields meaningful improvements in accuracy that justify additional deployment cost.
- B2 is reserved for optional testing when the research question specifically targets sensitivity to further capacity/resolution.

All choices are provisional and subject to empirical validation.


### 7.14 Cross-architecture consistency

Selecting EfficientNet variants for Architecture 3 does not change the project's common experimental requirements. All three families (YOLO, MobileNet, EfficientNet) must comply with the same:

- Side + Rear dataset and annotations
- Train/validation/test split
- Target definitions and label formats
- Preprocessing and augmentation policy (TRAINING_PROTOCOL.md)
- Evaluation metrics and reporting format
- Deployment benchmarking methodology

This ensures that architecture-level differences are the primary source of performance variation in the final comparison.


### 7.15 Architecture 3 Status

Primary candidate:
EfficientNet-B0

Secondary candidate:
EfficientNet-B1

Optional sensitivity candidate:
EfficientNet-B2

Status:
Candidate architecture — not experimentally validated.

Final selection among B0/B1 (and optional use of B2) will be based on controlled experiments that balance predictive performance and mobile deployment feasibility.


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
- MobileNet is the second architecture family currently under investigation. (DECIDED)
- Within MobileNet, MobileNetV3-Small is the PRIMARY CANDIDATE. (PRIMARY CANDIDATE)
- MobileNetV2 and MobileNetV3-Large remain comparative alternatives within Architecture 2. (ALTERNATIVE)
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
| MobileNet as Architecture 2 family | DECIDED |
| MobileNetV3-Small | PRIMARY CANDIDATE |
| MobileNetV2 | ALTERNATIVE |
| MobileNetV3-Large | ALTERNATIVE |
| Architecture 3 | TBD |
| Fusion mechanism | TBD |
| Weight sharing | TBD |
| Prediction heads | TBD |
| Training hyperparameters | TBD |
| Android runtime | TBD |
| Final architecture selection | TO BE EXPERIMENTALLY VALIDATED |


---

**Document status:** Architecture selection record (candidates and rationale).  
**Next steps:** complete Architecture 3 after its family is selected, perform controlled experiments per TRAINING_PROTOCOL.md, and record results in this document.

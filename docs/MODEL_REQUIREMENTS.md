# Model Requirements

## 1. Purpose

This document defines the common requirements that all three neural network architectures must satisfy in the livestock weight estimation system. It specifies **what the models must accomplish** without prematurely prescribing **how** they will accomplish it.

All three proposed architectures must solve the same prediction problem and be evaluated under a comparable experimental methodology. This approach ensures that the architecture comparison is fair, reproducible, and scientifically valid. The goal is to enable an evidence-based selection of the best-performing architecture given the constraints of real-time inference on mobile hardware.

## 2. Problem Definition

The core task is to estimate physical and morphological characteristics of livestock from visual information captured using two complementary viewing angles.

The system receives two images of the same animal taken from different perspectives:

- **Side view**: captures longitudinal body profile, length, and body depth
- **Rear view**: captures width, rear body contour, and rear-view morphology

Information from both views must be integrated to make final predictions. Each view provides complementary visual information that helps resolve ambiguities and improve prediction accuracy.

### System Data Flow

```
                    SIDE IMAGE
                         |
                         v
                    SIDE BRANCH
                         |
                         |
                         +------+
                                |
                              FUSION
                                |
                         +------+
                         |
                         v
                  PREDICTION HEADS
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
            BBOX        SEX        WEIGHT


                    REAR IMAGE
                         |
                         v
                    REAR BRANCH
                         |
                         |
                         +------> FUSION
```

## 3. Input Requirements

### 3.1 Visual Inputs

The model receives exactly two image inputs:

1. **Side-view image**: A photographic or video frame showing the animal's lateral profile
2. **Rear-view image**: A photographic or video frame showing the animal's posterior profile

Both images must represent the same individual animal in the same session or time period.

### 3.2 Dataset

All three candidate architectures must:

- Use the same underlying dataset
- Apply a comparable train/validation/test split across all three experiments
- Ensure that no individual animal appears in more than one split

This standardization enables fair comparison between architectures and prevents dataset leakage.

### 3.3 Input Consistency

The spatial and temporal relationship between the side-view and rear-view images must be preserved throughout the pipeline. The model must treat the pair as a semantically related unit representing a single animal at a single point in time.

**Critical constraint:** The ground-truth weight value must **never** be provided as an input feature to the model. Weight is exclusively a supervised learning target, not a model input. The architecture must learn to infer weight from visual features alone.

## 4. Output Requirements

The model must produce three distinct predictions for each animal:

### 4.1 Bounding Box

Bounding-box prediction for the Side and Rear views is an architecture-selection decision and must be defined consistently for all three candidate models in **ARCHITECTURE_SELECTION.md**. The three architectures must adopt the same bounding-box prediction strategy for experimental comparability. Possible strategies include (but are not limited to): both views predicting bounding boxes, only a single view predicting bounding boxes, or an alternative detection strategy that associates detections across views.

The important requirement is not which strategy is chosen, but that the chosen strategy is identical in scope and application across all three architectures and is documented in ARCHITECTURE_SELECTION.md.

**Evaluation concepts:**
- **Intersection over Union (IoU)**: measures the spatial overlap between predicted and ground-truth bounding boxes
- **Mean Average Precision (mAP)**: aggregates precision-recall performance across IoU thresholds

Exact IoU thresholds and mAP configuration details will be specified in TRAINING_PROTOCOL.md.

### 4.2 Sex Classification

The model must classify the animal's biological sex into one of two categories:

- **F** (Female)
- **M** (Male)

The internal representation of sex (e.g., as a single binary output neuron, as two-class softmax, or embedded within a multi-task head) is an implementation detail and will be chosen during architecture design.

### 4.3 Weight Regression

The model must estimate the animal's weight as a continuous numerical value in **kilograms**. This is a regression task, not a classification task.

The predicted weight corresponds to the biological mass of the animal represented by the side-view and rear-view image pair provided as input.

## 5. Architecture Requirements

Three distinct neural network architectures will be evaluated for this task. Although the specific backbone implementations and fusion mechanisms are not yet selected, the overall conceptual architecture is fixed.

### 5.1 Dual-Branch Design Pattern

All three architectures must follow a dual-branch design pattern:

```
SIDE IMAGE          REAR IMAGE
    |                   |
    v                   v
SIDE BRANCH         REAR BRANCH
    |                   |
    |                   |
    +------- FUSION ----+
             |
             v
     PREDICTION HEADS
             |
     +-------+-------+
     |       |       |
     v       v       v
   BBOX    SEX    WEIGHT
```

- **Side Branch**: Processes the side-view image and extracts visual features relevant to the animal's body structure and profile
- **Rear Branch**: Processes the rear-view image independently and extracts visual features from the rear perspective
- **Feature Fusion**: Combines or integrates the feature representations from both branches before making predictions
- **Prediction Heads**: Three separate output heads that generate predictions for bounding box, sex, and weight

### 5.2 Architectural Flexibility

The following design decisions are **intentionally open** and will be determined during architecture selection:

- Exact backbone architecture family (e.g., YOLO variant, MobileNet variant, EfficientNet variant)
- Whether candidate families currently under consideration will ultimately be selected
- Specific backbone implementation details
- Feature fusion mechanism (e.g., concatenation, element-wise operations, attention-based fusion)
- Prediction head architecture and structure
- Whether the two branches share weights (Siamese-style) or have independent weights

Architecture selection and justification will be documented separately in **ARCHITECTURE_SELECTION.md**.

## 6. Training Requirements

All three architectures must be trained using the same dataset with a comparable training methodology to ensure experimental validity.

### 6.1 Data Split

The training process must support:

- **Training data**: used to update model parameters during training
- **Validation data**: used to monitor generalization and guide hyperparameter selection during training
- **Test data**: used for final performance evaluation, applied only after training is complete

### 6.2 Data Augmentation

The training pipeline must support data augmentation techniques applied to the input images. Augmentation must be applied consistently to maintain semantic correctness:

- Bounding box coordinates must be transformed to remain valid for augmented images
- Sex labels must remain unchanged (augmentation must not alter the animal's sex)
- Weight labels must remain unchanged (augmentation must not alter the animal's weight)
- The side-view and rear-view pair must maintain their correspondence relationship

Exact augmentation techniques and parameters will be specified in TRAINING_PROTOCOL.md.

### 6.3 Training Infrastructure Requirements

The training process must support:

- Reproducibility through random seed management
- Model checkpointing at regular intervals
- Validation during training
- Evaluation metrics logging
- Recovery from interrupted training when applicable

### 6.4 Intentionally Deferred Training Decisions

The following training hyperparameters and techniques are **intentionally deferred** and will be documented in TRAINING_PROTOCOL.md:

- Optimizer algorithm and configuration
- Learning rate and learning rate schedule
- Batch size
- Number of training epochs
- Loss functions for each task (bounding box, sex, weight)
- Loss weighting scheme (relative importance of the three tasks)
- Early stopping criteria
- Exact data augmentation parameters and techniques
- Input image resolution
- Model initialization strategy

These decisions will be justified based on preliminary experiments and domain considerations.

## 7. Evaluation Requirements

Model performance must be quantified using standard metrics appropriate to each prediction task.

### 7.1 Bounding Box Evaluation

The bounding box predictions must be evaluated using object-detection metrics:

- **Intersection over Union (IoU)**: computed for each prediction, measuring spatial overlap
- **Mean Average Precision (mAP)**: computed across IoU thresholds, aggregating precision and recall

Additional details such as:
- IoU threshold(s) used for evaluation
- Average Precision computation method
- Precision-recall curve analysis

will be specified in TRAINING_PROTOCOL.md.

### 7.2 Sex Classification Evaluation

Sex classification must be evaluated using classification metrics:

- **Accuracy**: overall proportion of correct predictions
- **Precision**: proportion of predicted females/males that are actually correct
- **Recall**: proportion of actual females/males that were correctly predicted
- **F1-score**: harmonic mean of precision and recall

Confusion matrices and per-class breakdowns should be reported to detect any systematic bias.

### 7.3 Weight Regression Evaluation

Weight estimation must be evaluated using regression metrics:

- **Mean Absolute Error (MAE)**: average absolute difference between predicted and ground-truth weight, in kilograms
- **Root Mean Squared Error (RMSE)**: penalizes larger errors more heavily, in kilograms
- **R² Score (Coefficient of Determination)**: measures the proportion of variance in weight explained by the model

Weight values must be reported in kilograms with consistent precision.

## 8. Mobile Deployment Requirements

The final application is intended for deployment on mobile devices using the Android platform.

### 8.1 Development Environment

- **IDE**: Android Studio
- **Programming Language**: Kotlin
- **Inference Target**: on-device execution (no remote server required)

### 8.2 Real-Time Performance Requirement

Real-time inference on Android hardware is a **major requirement**. The model must be evaluated not only on prediction accuracy but also on its feasibility for practical deployment.

### 8.3 Mobile Efficiency Metrics

Each candidate architecture must be evaluated for:

- **Model size**: total file size of the saved model weights and architecture
- **Number of trainable parameters**: indicator of model complexity
- **Inference latency**: time required to process one complete prediction (both images)
- **Memory requirements**: peak memory usage during inference (RAM)
- **Mobile compatibility**: ability to run on typical Android smartphones without specialized hardware
- **Export/deployment feasibility**: ease of converting the trained model to a mobile inference format

### 8.4 Intentionally Deferred Deployment Decisions

The following mobile deployment aspects are **intentionally deferred** to later project phases:

- Specific Android inference runtime (e.g., TensorFlow Lite, ONNX Runtime, MediaPipe)
- Model quantization strategy (e.g., int8, float16)
- Target smartphone hardware (processor, RAM, storage)
- Specific Android SDK version
- User interface design

These decisions will be made based on architecture selection and resource constraints.

## 9. Performance Reporting

For each of the three candidate architectures, the final experimental report must include comprehensive performance data to enable fair comparison and reproducibility.

### 9.1 Model Complexity

- Total number of trainable parameters
- Total model size (serialized model file size in MB)
- Breakdown by component when meaningful (e.g., side branch parameters, rear branch parameters, fusion layer parameters)

### 9.2 Prediction Performance

- **Bounding box metrics**: IoU values, mAP score
- **Sex classification metrics**: Accuracy, Precision, Recall, F1-score (per class and macro-averaged)
- **Weight regression metrics**: MAE (kg), RMSE (kg), R² score

Performance should be reported separately for training, validation, and test sets.

### 9.3 Inference Performance

- **Inference latency**: average time (milliseconds) to produce predictions from a single image pair
- **Hardware used**: specifications of the machine used for inference testing
- **Runtime/inference backend**: which inference framework was used (if different from training framework)
- **Input resolution**: dimensions of input images (height, width in pixels)
- **Model precision**: data type of model weights (e.g., float32, float16, int8)

### 9.4 Reproducibility

When possible, record:

- **Random seed(s)**: values used for NumPy, TensorFlow, PyTorch, or other random number generators
- **Dataset split**: specification of train/validation/test split (e.g., 70%/10%/20%, or specific file lists)
- **Model configuration**: complete specification of all architecture hyperparameters
- **Training configuration**: learning rate, batch size, number of epochs, optimizer, loss functions
- **Software environment**: Python version, framework versions (TensorFlow/PyTorch version)
- **Training hardware**: GPU model, CPU, RAM, and other relevant specifications
- **Deployment hardware**: smartphone model or Android emulator configuration (if applicable)
- **Training time**: total wall-clock time required to train the model

## 10. Data Leakage Constraints

Data leakage—inadvertently providing the model with information about the target variable during training—would invalidate the experiment. The project must maintain strict separation between features and targets.

### 10.1 Weight as Target Only

**Critical requirement:** The ground-truth weight value must **never** be used as an input feature to the model. Weight is exclusively a supervised learning target.

The model must learn to estimate weight from visual information alone (side-view and rear-view images).

### 10.2 Leakage Prevention

Any preprocessing, feature engineering, clustering analysis, or dataset construction must be reviewed to ensure:

- Individual ground-truth weight values are not exposed to unsupervised feature-extraction pipelines or used as input features (weight remains a supervised target only)
- Weight values are not used to guide image selection, preprocessing, or normalization in a way that leaks target information into training inputs

Training-set weight statistics (for example: mean, variance, distributional shape) may be used for dataset characterization and to inform training-design decisions such as loss scaling, normalization strategies, or sampling schemes. Validation data may be used for model and hyperparameter selection following the defined protocol in TRAINING_PROTOCOL.md.

**Critical prohibition:** Test-set information (including test-set weight statistics or labels) must not be used for architecture selection, hyperparameter tuning, or any training decisions.

The project's purpose is to learn the fundamental visual relationship between animal morphology and weight; this requires genuine separation of individual visual features from individual ground-truth weight targets.

## 11. Clustering

Clustering is an exploratory unsupervised analysis technique that is used in this project to investigate the structure of the dataset and visual similarity.

### 11.1 Clustering Use Cases

Clustering analysis can be used to investigate:

- Dataset structure and composition
- Visual similarity patterns across images
- Potential subgroups (e.g., breed-specific clusters, age-related groups)
- Outliers and anomalies in the dataset
- Whether side-view and rear-view images form coherent pairs
- Dataset-specific patterns that may inform training strategy

### 11.2 Clustering Constraints

Clustering may be used for exploratory dataset analysis and hypothesis generation. However, clustering results must never be used to:

- create supervised labels
- modify ground-truth targets
- generate weight labels
- leak target information into model inputs
- selectively manipulate the test set

Clustering analyses must remain independent from supervised target generation. Clustering outputs are permitted for understanding dataset structure, guiding exploratory hypotheses, or informing non-targeting quality-control checks, but they must not alter the labeled dataset used for training or evaluation.

## 12. Data Augmentation

Data augmentation is a critical technique for improving model generalization when training data is limited.

### 12.1 Augmentation Principles

During training, input images will be augmented using transformations such as rotations, translations, brightness adjustments, contrast adjustments, and other techniques. Augmentation must:

- Preserve semantic correctness of all labels:
  - Bounding box coordinates must be transformed correctly to remain valid for augmented images
  - Sex labels must remain unchanged (augmentation does not alter animal sex)
  - Weight labels must remain unchanged (augmentation does not alter animal weight)
- Maintain the correspondence between side-view and rear-view image pairs
- Apply transformations consistently to both images in a pair when appropriate

### 12.2 Augmentation Configuration

The specific augmentation techniques, parameters, probability of application, and magnitude of transformations will be specified in TRAINING_PROTOCOL.md. Augmentation decisions will be based on:

- Dataset characteristics
- Baseline model performance
- Preliminary experiments

## 13. Architecture Comparison Requirements

The core objective of this project is to compare three neural network architectures on the same prediction task and select the architecture that best balances accuracy and deployment feasibility.

### 13.1 Fair Comparison Criteria

The three architectures must solve the same task and be evaluated fairly:

- All use the same dataset and split
- All are trained with comparable methodology
- All are evaluated with the same metrics
- All are tested on the same test set without modification

### 13.2 Multi-Objective Evaluation

**A model should not be considered superior based only on prediction accuracy.**

The comparison must consider four interrelated dimensions:

1. **Prediction Performance** (accuracy, precision, recall, F1-score for sex; IoU, mAP for bounding box; MAE, RMSE, R² for weight)
2. **Model Complexity** (number of parameters, model size in MB)
3. **Inference Latency** (milliseconds per prediction)
4. **Mobile Deployment Feasibility** (compatibility with Android, memory requirements, quantization support)

### 13.3 Trade-off Analysis

The final architecture selection must explicitly consider trade-offs:

- A highly accurate model that requires 5 seconds per inference is not practical for real-time Android deployment
- A lightweight model that sacrifices accuracy by 10% may be preferable if it enables smooth real-time inference on mid-range smartphones
- Model complexity should scale appropriately to the available computational resources on the target hardware

This multi-objective perspective is essential because the application is specifically intended for real-time inference on Android hardware with typical computational and memory constraints.

## 14. Decisions Intentionally Deferred

The following design decisions have **not** been finalized and will be addressed in subsequent project phases. Premature decisions would limit the experimental scope without corresponding benefits.

### 14.1 Architecture Selection (ARCHITECTURE_SELECTION.md)

- Exact YOLO variant (if YOLO is selected)
- Exact MobileNet variant (e.g., MobileNetV2, MobileNetV3, if selected)
- Exact EfficientNet variant (e.g., EfficientNet-B0 through B7, if selected)
- Whether YOLO, MobileNet, and EfficientNet candidate families will ultimately be selected as the final three architectures
- Alternative backbone architectures not yet considered
- Backbone implementation details and modifications
- Feature fusion mechanism (concatenation, element-wise operations, attention-based fusion, others)
- Prediction head implementation and structure
- Whether branches share weights or have independent weights
- Input image resolution
- Specific backbone normalization and activation function choices

### 14.2 Training and Optimization (TRAINING_PROTOCOL.md)

- Optimizer algorithm (SGD, Adam, RMSprop, others) and configuration
- Learning rate value and learning rate schedule (constant, exponential decay, cosine annealing, others)
- Batch size
- Total number of training epochs
- Loss functions for bounding box, sex, and weight predictions
- Loss weighting scheme (relative importance of the three tasks)
- Early stopping criteria and patience
- Data augmentation techniques and parameters
- Model initialization strategy
- Regularization techniques (dropout, batch normalization parameters, L1/L2 regularization)
- Warm-up learning phase (if applicable)

### 14.3 Evaluation and Deployment (later phases)

- Exact mAP configuration (e.g., which IoU thresholds, how mAP is averaged)
- Exact IoU thresholds for bounding box evaluation
- Model quantization strategy (int8, float16, mixed precision)
- Android inference runtime (TensorFlow Lite, ONNX Runtime, MediaPipe, others)
- Target smartphone hardware (processor family, RAM constraints)
- Specific Android SDK version
- User interface design and workflows

### 14.4 Rationale

These decisions are deferred because they:

- Require experimental validation (no decision should be made without evidence)
- Are interdependent on architecture selection (some choices only make sense after the backbone is chosen)
- May require preliminary experiments or surveys of current best practices
- Should be justified explicitly rather than assumed to be obvious

## 15. Related Documentation

This document defines the common requirements that all three architectures must satisfy. It establishes the foundation for a fair, reproducible comparison.

Two additional documents will define decisions that are currently deferred:

### 15.1 ARCHITECTURE_SELECTION.md

Documents the selection and justification of the three neural network architectures. This document will specify:

- Why specific backbones are chosen
- How backbone selection supports the performance requirements
- Justification for the dual-branch design and fusion approach
- Trade-offs between different architectural choices
- Preliminary experiments or evidence supporting the selection

This document will be created after initial architecture exploration and candidate evaluation.

### 15.2 TRAINING_PROTOCOL.md

Documents the standardized training procedure that all three architectures must follow. This document will specify:

- Training hyperparameters (optimizer, learning rate, batch size, epochs)
- Loss functions and their weighting
- Data augmentation strategy
- Validation and evaluation procedures
- Reproducibility checklist
- Expected computational requirements
- Training time estimates

This document will be created after pilot training experiments establish reasonable hyperparameter ranges.

---

**Document Version:** 1.0  
**Status:** Requirements specification (architecture and training protocols pending)  
**Last Updated:** August 2026

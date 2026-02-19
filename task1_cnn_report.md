# Task 1: CNN Classification with Comprehensive Analysis

**Using Convolutional Neural Networks (PneumoniaMNIST)**

Abeer | February 2026

---

## 1. Introduction

This report presents a comprehensive analysis of a Convolutional Neural Network (CNN) developed for binary classification of pneumonia from chest X-ray images using the PneumoniaMNIST dataset. The implemented model achieved 88.14% accuracy with an AUC of 94.67%, demonstrating strong discriminative capability for pneumonia detection. The analysis examines the complete machine learning pipeline including data preprocessing, model architecture, training methodology, evaluation metrics, and failure case analysis.

## 2. Dataset

The PneumoniaMNIST dataset exhibits the following properties: the training set contains 4,708 images (1,214 Normal and 3,494 Pneumonia), the validation set contains 524 images (135 Normal and 389 Pneumonia), and the test set contains 624 images (234 Normal and 390 Pneumonia). The dataset demonstrates significant class imbalance (74.2% pneumonia vs. 25.8% normal in training), which is representative of real-world medical scenarios where positive cases often outnumber healthy controls in clinical datasets. This imbalance necessitates careful consideration in model evaluation beyond simple accuracy metrics.

### 2.1 Normalisation Strategy

The use of `transforms.Normalize(mean=[0.5], std=[0.5])` is justified as a normalisation strategy that performs zero-centering and variance scaling to improve optimisation stability during neural network training. For grayscale medical images with pixel values in the range [0, 1], this transformation maps the input distribution to [−1, 1], aligning it with a symmetric range that improves gradient flow and reduces the risk of activation saturation, particularly when using ReLU activations.

From a theoretical perspective, normalisation mitigates internal covariate shift, enabling the use of higher learning rates and promoting faster convergence. In the context of medical imaging, dataset-level standardisation further ensures consistent feature scaling despite variations in acquisition protocols, thereby improving model robustness and generalisation.

### 2.2 Data Augmentation

Data preprocessing involved augmentation and normalisation tailored for medical imaging. The training set was subjected to random rotations (±10°) and horizontal flips to increase robustness to orientation variability, followed by normalisation to a mean of 0.5 and a standard deviation of 0.5. Validation and test sets were normalised identically but not augmented to prevent bias in performance assessment. This approach improves generalisation while ensuring that key radiographic features, such as lung opacities, remain clinically interpretable. Batch processing using a size of 128 ensured efficient training and evaluation.

## 3. Model Architecture

The proposed CNN was intentionally designed as a compact architecture tailored to the characteristics of the PneumoniaMNIST dataset, which consists of very low-resolution (28×28) grayscale chest X-ray images. Unlike high-resolution medical imaging tasks that benefit from deep pretrained networks, this dataset imposes a representational constraint where excessive model depth can lead to overfitting without providing additional diagnostic information. A lightweight CNN was therefore selected to balance feature learning capacity with statistical efficiency.

The network comprises three convolutional blocks with channel depths of 32, 64, and 128, each followed by ReLU activation and spatial downsampling through max pooling. This progressive expansion of feature channels enables hierarchical representation learning: early layers capture fundamental intensity transitions and anatomical boundaries, intermediate layers aggregate these into localised radiographic textures, and deeper layers encode coarse pathological structures such as diffuse opacities.

Instead of flattening feature maps into a large fully-connected representation, the architecture employs `AdaptiveAvgPool2d(1)` to perform global average pooling. This design dramatically reduces the number of parameters and acts as a structural regulariser, encouraging the network to respond to the presence of learned features rather than memorising their spatial configuration — particularly important for small medical datasets where dense layers can easily overfit.

The pooled 128-dimensional representation is finally mapped to a single logit through a linear layer, producing a probabilistic prediction optimised using binary cross-entropy with logits. By concentrating model capacity within the convolutional feature extractor rather than dense classification layers, the architecture remains computationally efficient while preserving the ability to learn clinically meaningful global patterns.

## 4. Model Training

The network was trained using the binary cross-entropy loss with logits (`BCEWithLogitsLoss`) and optimised with the Adam optimiser at a learning rate of 1e-3. A step learning rate scheduler reduced the rate by 50% every five epochs to stabilise convergence. Early stopping with a patience of three epochs prevented overfitting, with the best model saved at the lowest validation loss. Training and validation loss curves showed consistent reduction, with validation loss decreasing from 0.5202 in the first epoch to 0.1676 by the fifteenth epoch, indicating effective convergence and generalisation.

| Hyperparameter | Value | Purpose / Justification |
|---|---|---|
| Batch Size | 128 | Stable gradient estimation with efficient memory usage for 28×28 images |
| Max Epochs | 15 | Sufficient for convergence; controlled via early stopping |
| Early Stopping | Patience 3 | Stops training when validation loss does not improve, preventing overfitting |
| Optimizer | Adam | Adaptive learning rate optimization for fast and stable convergence |
| Learning Rate | 1e-3 | Standard starting point enabling efficient early training |
| Loss Function | BCEWithLogitsLoss | Numerically stable binary classification loss (sigmoid + BCE combined) |
| Scheduler | StepLR | Gradual reduction of learning rate for refined convergence |
| Step Size | 5 epochs | Learning rate reduced every 5 epochs |
| Gamma (LR Decay) | 0.5 | Multiplies learning rate by 0.5 for smoother optimization |

*Table 1: Training Hyperparameters for PneumoniaCNN*

## 5. Evaluation Metrics and Results

On the test set, the CNN achieved accuracy = 88.14%, precision = 85.76%, recall = 97.18%, F1-score = 91.11%, and AUC = 94.67%, indicating strong overall performance. The confusion matrix quantifies classification outcomes as shown in Table 2 below.

| Metric | Value | Clinical Significance |
|---|---|---|
| Accuracy | 88.41% | Overall correctness |
| Precision | 85.75% | Of predicted pneumonia cases, 89.19% are correct |
| Recall (Sensitivity) | 97.18% | Detects 93.08% of actual pneumonia cases |
| F1-Score | 91.11% | Harmonic mean of precision and recall |
| AUC-ROC | 94.67% | Excellent discriminative ability |

*Table 2: Performance Metrics for PneumoniaCNN*

From the confusion matrix, true positives (pneumonia correctly detected) = 381, true negatives (normal correctly detected) = 171, false positives = 63, and false negatives = 11. These numbers reflect an engineering trade-off favouring sensitivity: the network prioritises detecting pneumonia and minimises false negatives (only 11 missed cases), which is clinically important. The 63 false positives primarily result from subtle shadows or low-contrast patterns in normal X-rays that the CNN misinterprets as pathological, highlighting the limitations imposed by low-resolution input.

The ROC curve rises sharply at first and forms a smooth, convex shape, getting close to the top-left corner, with an AUC of 0.9467. This means the model is very good at distinguishing pneumonia cases from normal ones. Across most thresholds, it achieves high true positive rates while keeping false positives low, indicating that it consistently identifies pneumonia correctly.

The training curves show that the model is learning well. Both the training and validation losses started around 0.53 and 0.38 but steadily dropped to about 0.20 by the fifteenth epoch. The validation loss closely follows the training loss and even dips slightly below it at times. Since the two curves move together without splitting apart, this suggests the model is not overfitting and generalises well, showing that data augmentation and early stopping worked effectively.

### 5.1 Failure Case Analysis

Visual inspection of the ten misclassified cases reveals systematic patterns in model errors. Nine false positives (True: 0, Pred: 1) show normal chest X-rays incorrectly classified as pneumonia, exhibiting relatively clear lung fields with well-defined cardiac borders and normal bronchovascular markings. These errors likely arise from the model misinterpreting normal anatomical structures — such as the cardiac silhouette overlap with the left lower lobe, prominent hilar vessels, or positioning artefacts — as pathological opacities after compression to 28×28 resolution.

The single false negative case (True: 1, Pred: 0) displays subtle bilateral infiltrates with preserved lung lucency, representing early-stage or atypical pneumonia where minimal consolidation becomes undetectable at low resolution. This case demonstrates the fundamental limitation of 28×28 pixel representation, where fine reticular patterns and ground-glass opacities characteristic of viral or interstitial pneumonia are lost during downsampling, preventing the model from recognising diffuse patterns that differ from the dense alveolar consolidation it predominantly learned from the training set.

## 6. Model Strengths and Limitations

**Strengths**

The model demonstrates several strengths that make it suitable as a prototype for an integrated medical AI system. First, the architecture is computationally efficient and can be trained on standard hardware without requiring GPUs, aligning well with the objective of building accessible AI solutions. Second, the use of domain-appropriate augmentation improves generalisation while preserving clinical realism. Third, the model achieves high recall and AUC, indicating strong sensitivity to pathological findings — a key requirement for diagnostic-support tools.

**Limitations**

The observed failure cases highlight not only technical limitations but also an interpretability gap. While the model achieves high sensitivity, it provides no explicit reasoning for its predictions, making it difficult to determine whether decisions are based on clinically meaningful patterns or dataset-specific artefacts. The model functions largely as a "black box", producing predictions without clinically interpretable justification. Such lack of interpretability limits its readiness for real-world deployment, where explainability is necessary for physician trust, validation, and regulatory approval.

Furthermore, class imbalance was not explicitly addressed through weighting strategies such as Focal Loss or cost-sensitive learning, which may affect robustness when applied to different populations. In medical AI, decision opacity raises concerns related to transparency, auditability, and accountability of automated systems. Without a clear decision pathway, the model cannot fully satisfy the principle of non-maleficence ("do no harm"), as clinicians cannot verify whether predictions rely on clinically relevant evidence or unintended correlations.

To address this gap, Explainable AI (XAI) techniques such as Grad-CAM or SHAP could be integrated into the pipeline. These methods would allow visualisation of the regions influencing model predictions, transforming the system from a purely predictive tool into an interpretable decision-support aid. Incorporating XAI would not only reduce diagnostic uncertainty but also align the system with emerging ethical and regulatory frameworks for trustworthy medical AI.

## References

Bates, M. (2024). Visualizing deep learning decisions: grad-cam-based explainable AI for medical image analysis. *Journal of Scalable Data Engineering and Intelligent Computing*, 26–33.

Brima, Y. and Atemkeng, M. (2024). Saliency-driven explainable deep learning in medical imaging: bridging visual explainability and statistical quantitative analysis. *BioData Mining*, 17(1), 18.

Chauhan, N., Gupta, P. K., and Doja, F. (2025). LightPneumoNet: Lightweight pneumonia classifier. *arXiv preprint arXiv:2510.11232*.

Houssein, E. H., Gamal, A. M., Younis, E. M., and Mohamed, E. (2025). Explainable artificial intelligence for medical imaging systems using deep learning: a comprehensive review. *Cluster Computing*, 28(7), 469.

Tong, R., Liu, J., Wang, T., Hu, X., Liu, S., Wang, L., and Xu, J. (2025). Does bigger mean better? Comparative analysis of CNNs and biomedical vision language models in medical diagnosis. *arXiv preprint arXiv:2510.00411*.

Ukwuoma, C. C., Cai, D., Eziefuna, E. O., Oluwasanmi, A., Abdi, S. F., Muoka, G. W., Thomas, D., and Sarpong, K. (2025). Enhancing histopathological medical image classification for early cancer diagnosis using deep learning and explainable AI–LIME & SHAP. *Biomedical Signal Processing and Control*, 100, 107014.

Yen, C.-T. and Tsao, C.-Y. (2024). Lightweight convolutional neural network for chest X-ray images classification. *Scientific Reports*, 14(1), 29759.

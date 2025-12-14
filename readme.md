# ELEC509 Final Project  
**Multimodal Seizure Detection via Teacher–Student Learning with Feature-Based DNNs and Raw TCN Models**

## 📌 Overview
This repository contains the code, experiments, and results for the ELEC509 final project on **multimodal seizure detection using wearable sensors**. We investigate a **teacher–student learning framework** that combines:

- A **feature-based deep neural network (DNN) teacher** trained on EEG + ECG + EMG + motion features  
- **Lightweight student models** trained without EEG using knowledge distillation  
- **Raw time-series Temporal Convolutional Networks (TCNs)** operating directly on physiological signals  
- Control baselines to isolate the contribution of knowledge distillation  

The goal is to evaluate whether **high-performing EEG-informed models can transfer predictive knowledge** to models operating under **reduced sensing assumptions**, which is critical for deployable wearable systems.

---

## 🧠 Key Contributions
- End-to-end **preprocessing (V5 pipeline)** for multimodal wearable data  
- Comprehensive **feature engineering** across EEG, ECG, EMG, and motion  
- **ADASYN-balanced DNN teacher** achieving strong AUROC and AUPR  
- **Knowledge-distilled DNN student** trained using soft teacher probabilities  
- **Raw TCN student and control models** for end-to-end learning analysis  
- Detailed training diagnostics to distinguish **optimization failure vs. overfitting**

---

## 📂 Repository Structure

```text
ELEC509-FINAL-PROJECT/
│
├── Models/
│   ├── teacher_dnn_model_lightweight_features_*.keras
│
├── Papers/
│   └── (Methods, results drafts, figures for report)
│
├── Plots/
│   ├── combined_training_histories.png
│   ├── dnn_control.png
│   ├── dnn_teacher.png
│   ├── dnn_student.png
│   ├── tcn_control.png
│   ├── tcn_student.png
│   └── plot_combiner.ipynb
│
├── Results/
│   ├── control_dnn_trial*.npz
│   ├── control_tcn_trial*.npz
│   ├── Lightweight DNN Teacher *.npz
│   ├── Lightweight DNN Student *.npz
│   ├── Lightweight Nerfed TCN Student *.npz
│
├── Preprocessing.ipynb
├── Feature Extraction.ipynb
├── Baseline Models.ipynb
├── DNN Control.ipynb
├── DNN Teacher Models.ipynb
├── DNN Student Models.ipynb
├── TCN Control.ipynb
├── TCN_Teacher_Model.ipynb
├── Dragonfly DNN Bonn EEG.ipynb
└── .gitattributes
```

## ⚙️ Preprocessing
Preprocessing was implemented in `preprocessing.ipynb` using the final **V5 pipeline** developed during iterative experimentation and validation. Raw multimodal wearable signals were first cleaned and standardized to ensure consistent downstream learning. Signals were segmented into fixed-length windows with modality-aware overlap strategies to address class imbalance at the data level. Background windows were extracted with 50% overlap, while seizure windows used 75% overlap to increase seizure sample density without artificial signal synthesis.

All modalities were temporally aligned prior to windowing. EEG signals were processed at 250 Hz, while motion signals (accelerometer and gyroscope) were processed at 25 Hz and upsampled or aligned as needed. Window-level labels were assigned using seizure annotations such that a window was labeled positive if any seizure activity occurred within its temporal bounds. This preprocessing strategy ensured consistent sample construction across all models while preserving physiological signal integrity.

---

## 🧬 Feature Engineering
Feature extraction was implemented in `feature_extraction.ipynb`. Hand-engineered features were computed independently for each modality and then concatenated to form multimodal feature vectors for feature-based models.

EEG features included time-domain statistics, frequency-domain power features, and time–frequency representations designed to capture seizure-related spectral signatures. ECG and EMG features focused on statistical descriptors and frequency-domain characteristics relevant to autonomic and muscular activity during seizures. Motion features were computed from both accelerometer and gyroscope signals, including per-axis statistics and triaxial vector magnitude features to capture gross motor activity.

All features were standardized prior to model training. Feature extraction was applied only to feature-based DNN models; raw TCN models operated directly on windowed time-series signals without feature engineering.

---

## 🏗️ Model Architectures

### DNN Teacher Model
The teacher model was a fully connected deep neural network trained on the full multimodal feature set derived from EEG, ECG, EMG, and motion signals. The network consisted of four hidden layers with decreasing width (512, 256, 128, and 64 units), each followed by batch normalization and dropout for regularization. Swish activations were used throughout, and L2 weight regularization was applied to all dense layers to mitigate overfitting. A final sigmoid output layer produced continuous seizure probabilities used for downstream knowledge distillation.

To address severe class imbalance, ADASYN oversampling was applied exclusively during teacher training, allowing the model to focus on hard-to-learn seizure samples while maintaining decision boundary smoothness.

### DNN Student Model
The DNN student model followed the same architectural family as the teacher but was trained using only student-available modalities (ECG, EMG, and motion). Instead of relying solely on binary ground truth labels, the student was supervised using soft probability outputs generated by the teacher model, enabling knowledge distillation. ADASYN was not applied during student training to avoid distribution mismatch between synthetic samples and teacher-generated probabilities.

### TCN Models (Raw Time-Series)
Temporal Convolutional Network (TCN) models were implemented to evaluate end-to-end learning directly from raw physiological signals. The TCN architecture consisted of stacked residual blocks with causal dilated convolutions, kernel sizes of 3, exponentially increasing dilation rates (1, 2, 4, 8, 16), batch normalization, and dropout. Residual connections were used to stabilize training. Global temporal aggregation was applied before the final sigmoid classification layer. Synthetic oversampling was not used for TCN models due to incompatibility with raw signal structure; instead, class imbalance was addressed via class-weighted loss functions.

Control TCN models and knowledge-distilled TCN students were trained to isolate the impact of architectural choice versus distillation.

---

## 🧪 Training Strategy

### Teacher Training
The DNN teacher model was trained using binary cross-entropy loss on ADASYN-balanced data. Optimization used the Adam optimizer with learning rate scheduling and early stopping based on validation loss to prevent overfitting. The trained teacher produced probabilistic outputs for all training samples, which were stored and later used for student supervision.

### Knowledge Distillation Loss
Student models were trained using a composite loss that combined hard ground truth supervision with soft teacher guidance:

\[
\mathcal{L}_{KD} = \alpha \cdot \text{BCE}(y, p_s) + (1-\alpha) \cdot \text{KL}(p_t \parallel p_s)
\]

where \( y \) is the binary ground truth label, \( p_t \) is the teacher-predicted probability, \( p_s \) is the student-predicted probability, and \( \alpha = 0.7 \). This formulation encourages the student to match both correct classification behavior and the teacher’s confidence structure.

### Control Models
Control DNN and TCN models were trained without knowledge distillation and without EEG inputs. These baselines served to quantify the performance gains attributable specifically to teacher–student learning rather than architectural or preprocessing effects alone.

---

## 📊 Training Behavior and Diagnostics
Training and validation loss and accuracy curves were monitored for all models to assess optimization stability and overfitting. Feature-based DNN models demonstrated smooth convergence with closely aligned training and validation curves, indicating good generalization. The TCN models showed limited learning under the shared preprocessing pipeline, suggesting sensitivity to raw signal preprocessing choices rather than overfitting. These diagnostics motivated planned future refinements to the raw TCN preprocessing strategy.


## ⚠️ Known Limitations
This study has several known limitations that should be considered when interpreting the results. First, the preprocessing and windowing pipeline was optimized primarily for feature-based learning and subsequently reused for raw time-series models. While this ensured experimental consistency across architectures, it likely disadvantaged the Temporal Convolutional Network (TCN), which is known to be sensitive to sampling rate, normalization strategy, and window overlap. In particular, the use of overlapping windows and z-score normalization via batch normalization layers differs from established TCN-based seizure detection pipelines that employ lower sampling rates, non-overlapping windows, and explicit min–max normalization.

Second, class imbalance was addressed using different strategies across models. The feature-based teacher model leveraged ADASYN oversampling, while raw TCN models relied on class-weighted loss functions due to the incompatibility of synthetic oversampling with physiological time-series data. Although this choice was methodologically justified, it introduces asymmetry in the training conditions that may affect direct performance comparisons.

Third, the TCN architecture implemented in this study did not include attention mechanisms or multiscale temporal aggregation beyond dilation, which have been shown to improve seizure localization and robustness in prior work. As a result, the raw TCN models likely underutilized long-range temporal dependencies present in the data.

Finally, due to course timeline constraints, hyperparameter optimization and preprocessing redesign for the TCN models were limited. As such, the reported TCN results should be interpreted as preliminary and diagnostic rather than definitive.

---

## 🔮 Future Work
Several directions for future work are planned to address these limitations and extend the current study. First, the raw TCN pipeline will be redesigned to more closely align with established best practices, including downsampling EEG signals, eliminating window overlap, and applying explicit signal-level normalization prior to model input. These changes are expected to significantly improve training stability and classification performance.

Second, future TCN models will incorporate self-attention mechanisms and multiscale temporal feature fusion to better capture seizure onset dynamics and long-range dependencies. This will enable a more faithful comparison to state-of-the-art sequence-based seizure detection models.

Third, additional knowledge distillation strategies will be explored, including temperature-scaled soft labels and intermediate feature matching, to further improve student model performance under reduced input assumptions.

Finally, this framework will be extended to patient-independent and cross-dataset evaluations to assess generalization and clinical robustness. These improvements will be incorporated prior to submission to a peer-reviewed venue, with the current study serving as a strong feasibility and systems-level validation of multimodal teacher–student learning for wearable seizure detection.

---

## 👥 Authors
**Ibrahim Samhar Al-Akash**  
Master of Bioengineering, Rice University  

**Jaehyun Nam**  
Master of Bioengineering, Rice University  

**Pavan Sastry**  
Master of Bioengineering, Rice University  

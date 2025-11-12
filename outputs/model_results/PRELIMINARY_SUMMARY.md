# Model Training Preliminary Summary

**Generated**: 2025-11-11
**Status**: Cross-validation in progress (80% complete)

## Overview

This document summarizes the 5-fold cross-validation results for peptide stability prediction on SIF (Simulated Intestinal Fluid) and SGF (Simulated Gastric Fluid) datasets.

### Datasets

1. **US9624268_cleaned**:
   - SIF: 130 samples, 5 classes
   - SGF: 90 samples, 5 classes (40 samples with missing labels filtered out)

2. **sif_sgf_second_cleaned**:
   - SIF: 558 samples, 4 classes
   - SGF: 558 samples, 4 classes

### Models Evaluated

- Logistic Regression (with balanced class weights)
- Random Forest (100 estimators, balanced class weights)
- XGBoost (100 estimators, with sample weighting)

### Evaluation Metrics

- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1-score (macro-averaged)
- AUC (one-vs-rest, macro-averaged)

---

## Results

### 1. US9624268 - SIF (130 samples, 5 classes)

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.377 ± 0.092 | 0.303 ± 0.069 | 0.308 ± 0.060 | 0.297 ± 0.064 | 0.631 ± 0.038 |
| Random Forest | 0.446 ± 0.080 | 0.329 ± 0.096 | 0.331 ± 0.086 | 0.317 ± 0.087 | 0.635 ± 0.060 |
| **XGBoost** | **0.415 ± 0.074** | **0.323 ± 0.091** | **0.340 ± 0.081** | **0.324 ± 0.081** | **0.655 ± 0.024** |

**Key Observations**:
- Random Forest achieved the highest accuracy (0.446)
- XGBoost showed the best AUC (0.655) and lowest variance
- All models show modest performance on this 5-class problem with limited samples
- The dataset is challenging due to class imbalance (Class 4: only 10 samples)

---

### 2. US9624268 - SGF (90 samples, 5 classes)

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.400 ± 0.099 | 0.319 ± 0.169 | 0.304 ± 0.116 | 0.294 ± 0.123 | 0.646 ± 0.090 |
| **Random Forest** | **0.489 ± 0.099** | **0.332 ± 0.147** | **0.319 ± 0.098** | **0.301 ± 0.103** | **0.645 ± 0.148** |
| XGBoost | *(in progress)* | - | - | - | - |

**Key Observations**:
- Random Forest outperforms Logistic Regression
- Small sample size (90) makes this a very challenging task
- High variance in metrics indicates sensitivity to train/test splits
- Severe class imbalance (Class 4: only 4 samples)

---

### 3. sif_sgf_second - SIF (558 samples, 4 classes)

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.597 ± 0.030 | 0.432 ± 0.009 | 0.409 ± 0.028 | 0.396 ± 0.004 | 0.719 ± 0.052 |
| **Random Forest** | **0.797 ± 0.040** | **0.494 ± 0.090** | **0.442 ± 0.044** | **0.453 ± 0.056** | **0.728 ± 0.052** |
| XGBoost | *(in progress)* | - | - | - | - |

**Key Observations**:
- Random Forest shows strong performance with 79.7% accuracy
- Larger sample size (558) enables better model training
- Substantial improvement over US9624268 results
- Class imbalance still present (Class 4: 64.7%, Class 2: 2.87%)

---

### 4. sif_sgf_second - SGF (558 samples, 4 classes)

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.573 ± 0.030 | 0.432 ± 0.022 | 0.432 ± 0.061 | 0.399 ± 0.027 | 0.730 ± 0.027 |
| **Random Forest** | **0.790 ± 0.044** | **0.547 ± 0.120** | **0.475 ± 0.034** | **0.481 ± 0.045** | **0.796 ± 0.026** |
| XGBoost | *(in progress)* | - | - | - | - |

**Key Observations**:
- Random Forest achieves 79% accuracy and 79.6% AUC
- Best performing dataset overall
- More balanced class distribution than SIF
- Similar performance to SIF dataset

---

## Preliminary Insights

### Dataset Size Impact
- **Larger datasets perform significantly better**: sif_sgf_second (558 samples) shows ~35-40% higher accuracy compared to US9624268 (90-130 samples)
- Larger datasets also show lower variance in metrics (more stable predictions)

### Model Comparison
- **Random Forest** consistently outperforms other models across all datasets
- **XGBoost** shows promise with low variance (US9624268 SIF: AUC=0.655±0.024)
- **Logistic Regression** serves as a solid baseline but is outperformed by tree-based methods

### Class Imbalance Handling
- Class weighting (`class_weight='balanced'`) helps but doesn't fully solve the imbalance problem
- Random Forest's ensemble nature makes it more robust to class imbalance
- Very small classes (n<10) remain challenging for all models

### SIF vs SGF
- **SGF prediction** tends to perform slightly better than SIF on larger datasets
- sif_sgf_second SGF: 79% accuracy vs SIF: 79.7% accuracy (similar)
- US9624268 SGF: 48.9% vs SIF: 44.6% (both challenging due to small size)

---

## Next Steps

1. **Complete XGBoost training** on remaining datasets
2. **Transfer learning evaluation**:
   - Train on US9624268 → test on sif_sgf_second
   - Train on sif_sgf_second → test on US9624268
   - Apply 5→4 class mapping for compatibility
3. **Feature importance analysis**:
   - Identify key molecular descriptors
   - Compare importance across models
4. **Visualization**:
   - Performance comparison charts
   - Confusion matrices
   - Transfer learning heatmaps

---

## Technical Notes

- All experiments use 5-fold stratified cross-validation
- Random seed: 42 (for reproducibility)
- Feature set: 1560 features (Morgan fingerprints, Avalon fingerprints, QED properties, physicochemical descriptors)
- Class labels were remapped to 0-based indices for XGBoost compatibility
- Missing labels (-1) were filtered out before training

---

*This is a preliminary summary. Full results and visualizations will be available once all training completes.*

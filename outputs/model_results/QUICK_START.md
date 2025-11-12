# Quick Start Guide - Model Analysis Results

## 📁 Results Location

All results are in: `outputs/model_results/`

## 📊 Key Files

### Summary Reports
- **`MODEL_SUMMARY.md`** - Complete analysis report with all insights
- **`PRELIMINARY_SUMMARY.md`** - Early results snapshot

### Performance Data
- **`cv_results/`** - Cross-validation results
  - `*_cv_summary.csv` - Performance metrics tables
  - `confusion_matrices/` - Confusion matrices for each model
  - `feature_importance/` - Feature rankings

- **`transfer_results/`** - Transfer learning results
  - `*_summary.csv` - Cross-dataset performance
  - `confusion_matrices/` - Transfer confusion matrices

### Visualizations
- **`figures/`** - All plots and charts
  - `cv_performance_comparison_*.png` - Model comparison bar charts
  - `transfer_performance_heatmap_*.png` - Transfer learning heatmaps
  - `fi_*.png` - Feature importance plots (12 plots)

## 🎯 Main Findings

### Best Models
- **sif_sgf_second SGF**: Random Forest (Acc: 79%, AUC: 81%)
- **sif_sgf_second SIF**: Random Forest (Acc: 80%, AUC: 79%)
- **US9624268 SIF**: XGBoost (AUC: 66%)
- **US9624268 SGF**: Random Forest (Acc: 49%)

### Transfer Learning
- ❌ Poor cross-dataset performance (5-24% accuracy)
- ✅ Need dataset-specific models

### Top Features
1. Morgan fingerprints (structural patterns)
2. LogP (lipophilicity) ⭐
3. Molecular Weight ⭐
4. HBA/HBD (hydrogen bonding)

## 📈 Quick View Commands

```bash
# View all CV results
cat outputs/model_results/cv_results/*_summary.csv

# View transfer results
cat outputs/model_results/transfer_results/*_summary.csv

# View visualizations
ls -lh outputs/model_results/figures/

# View feature importance (top 20)
head -21 outputs/model_results/cv_results/feature_importance/*_XGBoost*.csv
```

## 🔬 Reproducibility

Scripts used:
- `scripts/train_models.py` - 5-fold CV training
- `scripts/evaluate_transfer.py` - Transfer learning
- `scripts/visualize_model_results.py` - Visualization

Rerun with:
```bash
uv run python scripts/train_models.py --input_dir outputs/features --output_dir outputs/model_results/cv_results
uv run python scripts/evaluate_transfer.py --dataset1 outputs/features/US9624268_cleaned.npz --dataset2 outputs/features/sif_sgf_second_cleaned.npz
uv run python scripts/visualize_model_results.py --cv_dir outputs/model_results/cv_results --transfer_dir outputs/model_results/transfer_results
```

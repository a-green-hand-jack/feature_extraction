# 快速入门指南 - 模型分析结果

## 📁 结果位置

所有结果位于：`outputs/model_results/`

## 📊 关键文件

### 摘要报告
- **`MODEL_SUMMARY.md`** - 包含所有洞察的完整分析报告
- **`PRELIMINARY_SUMMARY.md`** - 早期结果快照

### 性能数据
- **`cv_results/`** - 交叉验证结果
  - `*_cv_summary.csv` - 性能指标表
  - `confusion_matrices/` - 每个模型的混淆矩阵
  - `feature_importance/` - 特征排名

- **`transfer_results/`** - 迁移学习结果
  - `*_summary.csv` - 跨数据集性能
  - `confusion_matrices/` - 迁移混淆矩阵

### 可视化
- **`figures/`** - 所有图表
  - `cv_performance_comparison_*.png` - 模型比较条形图
  - `transfer_performance_heatmap_*.png` - 迁移学习热图
  - `fi_*.png` - 特征重要性图（12 个图）

## 🎯 主要发现

### 最佳模型
- **sif_sgf_second SGF**：随机森林（准确率：79%，AUC：81%）
- **sif_sgf_second SIF**：随机森林（准确率：80%，AUC：79%）
- **US9624268 SIF**：XGBoost（AUC：66%）
- **US9624268 SGF**：随机森林（准确率：49%）

### 迁移学习
- ❌ 跨数据集性能差（5-24% 准确率）
- ✅ 需要数据集特定模型

### 顶级特征
1. Morgan 指纹（结构模式）
2. LogP（亲脂性）⭐
3. 分子量 ⭐
4. HBA/HBD（氢键）

## 📈 快速查看命令

```bash
# 查看所有交叉验证结果
cat outputs/model_results/cv_results/*_summary.csv

# 查看迁移结果
cat outputs/model_results/transfer_results/*_summary.csv

# 查看可视化
ls -lh outputs/model_results/figures/

# 查看特征重要性（前 20）
head -21 outputs/model_results/cv_results/feature_importance/*_XGBoost*.csv
```

## 🔬 可重现性

使用的脚本：
- `scripts/train_models.py` - 5 折交叉验证训练
- `scripts/evaluate_transfer.py` - 迁移学习
- `scripts/visualize_model_results.py` - 可视化

重新运行：
```bash
uv run python scripts/train_models.py --input_dir outputs/features --output_dir outputs/model_results/cv_results
uv run python scripts/evaluate_transfer.py --dataset1 outputs/features/US9624268_cleaned.npz --dataset2 outputs/features/sif_sgf_second_cleaned.npz
uv run python scripts/visualize_model_results.py --cv_dir outputs/model_results/cv_results --transfer_dir outputs/model_results/transfer_results
```


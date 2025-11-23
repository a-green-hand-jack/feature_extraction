#!/bin/bash
# 按顺序运行所有 Phase notebooks

set -e  # 遇到错误立即退出

echo "========================================="
echo "开始运行完整的三阶段工作流程"
echo "========================================="

# Phase 1: 数据转化
echo ""
echo "=== Phase 1: 数据转化 ==="
echo "处理原始数据，提取分子特征..."
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 \
    notebooks/Phase1_数据转化.ipynb \
    --output Phase1_executed

echo "✓ Phase 1 完成！"
echo "  - 生成: data/processed/*.csv"
echo "  - 生成: outputs/features/*.npz"

# Phase 2: 数据可视化
echo ""
echo "=== Phase 2: 数据可视化 ==="
echo "生成特征可视化和统计分析..."
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 \
    notebooks/Phase2_数据可视化.ipynb \
    --output Phase2_executed

echo "✓ Phase 2 完成！"
echo "  - 生成: outputs/figures/phase1/"
echo "  - 生成: outputs/figures/phase2/"

# Phase 3: 模型验证
echo ""
echo "=== Phase 3: 模型验证 ==="
echo "训练机器学习模型并评估性能..."
uv run jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=1800 \
    notebooks/Phase3_模型验证.ipynb \
    --output Phase3_executed

echo "✓ Phase 3 完成！"
echo "  - 生成: outputs/model_results/phase3_binary/"
echo "  - 生成: outputs/figures/phase3/"

echo ""
echo "========================================="
echo "所有阶段执行完成！"
echo "========================================="
echo ""
echo "执行结果已保存到:"
echo "  - notebooks/Phase1_executed.ipynb"
echo "  - notebooks/Phase2_executed.ipynb"
echo "  - notebooks/Phase3_executed.ipynb"
echo ""
echo "查看结果："
echo "  uv run jupyter notebook notebooks/"

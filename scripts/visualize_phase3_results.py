#!/usr/bin/env python3
"""
Phase 3 结果可视化

功能:
1. 交叉验证结果可视化
2. 特征重要性可视化
3. 模型迁移热力图
4. 混淆矩阵可视化

用法:
    uv run python scripts/visualize_phase3_results.py \
        --input_dir outputs/model_results/phase3_binary/ \
        --output_dir outputs/figures/phase3/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("visualize_phase3_results.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# 设置绘图样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_cv_results(cv_results: List[Dict], output_dir: Path, dpi: int = 300):
    """
    可视化交叉验证结果

    Args:
        cv_results: 交叉验证结果列表
        output_dir: 输出目录
        dpi: 图像分辨率
    """
    if not cv_results:
        logger.warning("没有交叉验证结果可视化")
        return

    # 转换为 DataFrame
    df = pd.DataFrame(cv_results)

    # 1. 模型性能对比（按数据集分组）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = ["mean_f1", "mean_accuracy", "mean_precision", "mean_recall"]
    titles = ["F1 Score", "Accuracy", "Precision", "Recall"]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        # 按数据集和目标分组
        for dataset in df['dataset'].unique():
            for target in df['target'].unique():
                subset = df[(df['dataset'] == dataset) & (df['target'] == target)]
                if len(subset) == 0:
                    continue

                x = np.arange(len(subset))
                y = subset[metric].values
                yerr = subset[metric.replace('mean_', 'std_')].values

                label = f"{dataset}_{target}"
                ax.bar(x + len(subset) * 0.1, y, yerr=yerr, label=label, alpha=0.7)

        ax.set_xlabel("Model")
        ax.set_ylabel(title)
        ax.set_title(f"Cross-Validation {title} Comparison")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "cv_performance_comparison.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"保存交叉验证性能对比图: {output_file}")

    # 2. F1 Score 热力图（数据集 x 模型）
    for target in df['target'].unique():
        subset = df[df['target'] == target]

        pivot_f1 = subset.pivot_table(
            values='mean_f1',
            index='dataset',
            columns='model',
            aggfunc='first'
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            pivot_f1,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={'label': 'F1 Score'}
        )
        ax.set_title(f"Cross-Validation F1 Score Heatmap ({target})")
        ax.set_xlabel("Model")
        ax.set_ylabel("Dataset")

        plt.tight_layout()
        output_file = output_dir / f"cv_f1_heatmap_{target}.png"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"保存 F1 热力图: {output_file}")


def plot_feature_importance(importance_dir: Path, output_dir: Path, top_n: int = 20, dpi: int = 300):
    """
    可视化特征重要性

    Args:
        importance_dir: 特征重要性文件目录
        output_dir: 输出目录
        top_n: 显示前 N 个特征
        dpi: 图像分辨率
    """
    csv_files = sorted(importance_dir.glob("*.csv"))

    if not csv_files:
        logger.warning("没有特征重要性文件可视化")
        return

    for csv_file in csv_files:
        logger.info(f"处理: {csv_file.name}")

        df = pd.read_csv(csv_file)
        df = df.head(top_n)

        fig, ax = plt.subplots(figsize=(12, 8))

        # 水平条形图
        y_pos = np.arange(len(df))
        ax.barh(y_pos, df['importance'], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['feature'], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importance\n{csv_file.stem}")
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        output_file = output_dir / f"{csv_file.stem}.png"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"保存特征重要性图: {output_file}")


def plot_transfer_learning_heatmap(transfer_results: List[Dict], output_dir: Path, dpi: int = 300):
    """
    可视化模型迁移热力图

    Args:
        transfer_results: 迁移学习结果列表
        output_dir: 输出目录
        dpi: 图像分辨率
    """
    if not transfer_results:
        logger.warning("没有迁移学习结果可视化")
        return

    # 转换为 DataFrame
    df = pd.DataFrame(transfer_results)

    # 按目标和模型分组
    for target in df['target'].unique():
        for model in df['model'].unique():
            subset = df[(df['target'] == target) & (df['model'] == model)]

            if len(subset) == 0:
                continue

            # 创建热力图矩阵
            datasets = sorted(list(set(subset['train_dataset'].tolist() + subset['test_dataset'].tolist())))
            n_datasets = len(datasets)

            heatmap_data = np.zeros((n_datasets, n_datasets))

            for _, row in subset.iterrows():
                train_idx = datasets.index(row['train_dataset'])
                test_idx = datasets.index(row['test_dataset'])
                heatmap_data[test_idx, train_idx] = row['f1']

            # 绘制热力图
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                xticklabels=datasets,
                yticklabels=datasets,
                ax=ax,
                cbar_kws={'label': 'F1 Score'}
            )
            ax.set_title(f"Transfer Learning F1 Score\n({target}, {model.upper()})")
            ax.set_xlabel("Train Dataset")
            ax.set_ylabel("Test Dataset")

            plt.tight_layout()
            output_file = output_dir / f"transfer_f1_heatmap_{target}_{model}.png"
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"保存迁移学习热力图: {output_file}")


def plot_confusion_matrices(transfer_results: List[Dict], output_dir: Path, dpi: int = 300):
    """
    可视化混淆矩阵（迁移学习）

    Args:
        transfer_results: 迁移学习结果列表
        output_dir: 输出目录
        dpi: 图像分辨率
    """
    if not transfer_results:
        logger.warning("没有迁移学习结果可视化")
        return

    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)

    for result in transfer_results:
        if 'confusion_matrix' not in result:
            continue

        cm = np.array(result['confusion_matrix'])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Unstable', 'Stable'],
            yticklabels=['Unstable', 'Stable'],
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        ax.set_title(f"Confusion Matrix\nTrain: {result['train_dataset']}, Test: {result['test_dataset']}\n({result['target']}, {result['model'].upper()})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        plt.tight_layout()
        output_file = cm_dir / f"cm_{result['train_dataset']}_to_{result['test_dataset']}_{result['target']}_{result['model']}.png"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()

    logger.info(f"保存 {len(transfer_results)} 个混淆矩阵到: {cm_dir}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 结果可视化")
    parser.add_argument("--input_dir", type=str, required=True, help="Phase 3 结果目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--dpi", type=int, default=300, help="图像分辨率")
    parser.add_argument("--top_n", type=int, default=20, help="特征重要性显示前 N 个特征")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Phase 3 结果可视化")
    logger.info("=" * 80)

    # 1. 加载交叉验证结果
    cv_summary_file = input_dir / "cv_results" / "cv_summary.json"
    if cv_summary_file.exists():
        logger.info("\n加载交叉验证结果...")
        with open(cv_summary_file) as f:
            cv_results = json.load(f)
        logger.info(f"找到 {len(cv_results)} 个交叉验证结果")

        plot_cv_results(cv_results, output_dir, dpi=args.dpi)
    else:
        logger.warning(f"交叉验证汇总文件不存在: {cv_summary_file}")

    # 2. 可视化特征重要性
    importance_dir = input_dir / "feature_importance"
    if importance_dir.exists():
        logger.info("\n可视化特征重要性...")
        plot_feature_importance(importance_dir, output_dir / "feature_importance", top_n=args.top_n, dpi=args.dpi)
    else:
        logger.warning(f"特征重要性目录不存在: {importance_dir}")

    # 3. 加载迁移学习结果
    transfer_summary_file = input_dir / "transfer_results" / "transfer_summary.json"
    if transfer_summary_file.exists():
        logger.info("\n加载迁移学习结果...")
        with open(transfer_summary_file) as f:
            transfer_results = json.load(f)
        logger.info(f"找到 {len(transfer_results)} 个迁移学习结果")

        plot_transfer_learning_heatmap(transfer_results, output_dir, dpi=args.dpi)
        plot_confusion_matrices(transfer_results, output_dir, dpi=args.dpi)
    else:
        logger.warning(f"迁移学习汇总文件不存在: {transfer_summary_file}")

    logger.info("\n" + "=" * 80)
    logger.info("可视化完成!")
    logger.info("=" * 80)
    logger.info(f"所有图表保存到: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型结果可视化脚本

从交叉验证和迁移学习结果生成可视化图表：
- 模型性能比较条形图
- 迁移学习性能热力图
- 特征重要性对比图
- 混淆矩阵可视化

用法：
    python scripts/visualize_model_results.py \\
        --cv_dir outputs/model_results/cv_results/ \\
        --transfer_dir outputs/model_results/transfer_results/ \\
        --output_dir outputs/model_results/figures/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def setup_logging(log_level: int = logging.INFO) -> None:
    """配置日志系统"""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("visualize_model_results.log"),
        ],
    )


logger = logging.getLogger(__name__)

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_cv_results(cv_dir: Path) -> pd.DataFrame:
    """
    加载所有交叉验证结果。

    Args:
        cv_dir: 交叉验证结果目录

    Returns:
        汇总结果的 DataFrame
    """
    csv_files = list(cv_dir.glob("*_summary.csv"))

    if not csv_files:
        logger.warning(f"在 {cv_dir} 中未找到交叉验证结果 CSV 文件")
        return pd.DataFrame()

    logger.info(f"找到 {len(csv_files)} 个交叉验证结果文件")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"加载了 {len(df_all)} 条交叉验证记录")

    return df_all


def load_transfer_results(transfer_dir: Path) -> pd.DataFrame:
    """
    加载所有迁移学习结果。

    Args:
        transfer_dir: 迁移学习结果目录

    Returns:
        汇总结果的 DataFrame
    """
    csv_files = list(transfer_dir.glob("*_summary.csv"))

    if not csv_files:
        logger.warning(f"在 {transfer_dir} 中未找到迁移学习结果 CSV 文件")
        return pd.DataFrame()

    logger.info(f"找到 {len(csv_files)} 个迁移学习结果文件")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"加载了 {len(df_all)} 条迁移学习记录")

    return df_all


def plot_cv_performance_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    metric: str = 'f1',
    dpi: int = 300
) -> Path:
    """
    绘制交叉验证性能比较条形图。

    Args:
        df: 交叉验证结果 DataFrame
        output_dir: 输出目录
        metric: 要可视化的指标
        dpi: 图像分辨率

    Returns:
        保存的图表路径
    """
    if df.empty:
        logger.warning("没有交叉验证结果可供可视化")
        return None

    logger.info(f"绘制交叉验证性能比较图 (指标: {metric})")

    # 为每个 dataset-target 组合创建子图
    datasets = df['dataset'].unique()
    targets = df['target'].unique()
    models = df['model'].unique()

    n_plots = len(datasets) * len(targets)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    axes = axes.flatten()

    plot_idx = 0

    for dataset in datasets:
        for target in targets:
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]

            # 筛选数据
            df_subset = df[(df['dataset'] == dataset) & (df['target'] == target)]

            if df_subset.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f"{dataset} - {target}")
                plot_idx += 1
                continue

            # 准备数据
            mean_col = f'mean_{metric}'
            std_col = f'std_{metric}'

            x_pos = np.arange(len(models))
            means = [df_subset[df_subset['model'] == m][mean_col].values[0]
                    if m in df_subset['model'].values else 0
                    for m in models]
            stds = [df_subset[df_subset['model'] == m][std_col].values[0]
                   if m in df_subset['model'].values else 0
                   for m in models]

            # 绘制条形图
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                         alpha=0.7, edgecolor='black')

            # 颜色编码
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            # 添加数值标签
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}',
                       ha='center', va='bottom', fontsize=9)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=15, ha='right')
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{dataset} - {target}")
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)

            plot_idx += 1

    # 隐藏多余的子图
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    fig.suptitle(
        f"Cross-Validation Performance Comparison ({metric.upper()})",
        fontsize=16, fontweight='bold', y=1.00
    )

    output_path = output_dir / f"cv_performance_comparison_{metric}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"保存图表: {output_path}")
    plt.close()

    return output_path


def plot_transfer_performance_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    metric: str = 'f1',
    dpi: int = 300
) -> Path:
    """
    绘制迁移学习性能热力图。

    Args:
        df: 迁移学习结果 DataFrame
        output_dir: 输出目录
        metric: 要可视化的指标
        dpi: 图像分辨率

    Returns:
        保存的图表路径
    """
    if df.empty:
        logger.warning("没有迁移学习结果可供可视化")
        return None

    logger.info(f"绘制迁移学习性能热力图 (指标: {metric})")

    targets = df['target'].unique()
    models = df['model'].unique()

    n_plots = len(targets) * len(models)
    n_cols = len(models)
    n_rows = len(targets)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, target in enumerate(targets):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]

            # 筛选数据
            df_subset = df[(df['target'] == target) & (df['model'] == model)]

            if df_subset.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f"{target} - {model}")
                continue

            # 创建热力图数据
            train_datasets = df_subset['train_dataset'].unique()
            test_datasets = df_subset['test_dataset'].unique()

            heatmap_data = np.zeros((len(train_datasets), len(test_datasets)))

            for i, train_ds in enumerate(train_datasets):
                for j, test_ds in enumerate(test_datasets):
                    mask = (df_subset['train_dataset'] == train_ds) & \
                           (df_subset['test_dataset'] == test_ds)
                    if mask.any():
                        heatmap_data[i, j] = df_subset[mask][metric].values[0]

            # 绘制热力图
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=test_datasets,
                yticklabels=train_datasets,
                cbar_kws={'label': metric.upper()},
                vmin=0,
                vmax=1,
                ax=ax
            )

            ax.set_xlabel('Test Dataset')
            ax.set_ylabel('Train Dataset')
            ax.set_title(f"{target} - {model}")

    plt.tight_layout()
    fig.suptitle(
        f"Transfer Learning Performance ({metric.upper()})",
        fontsize=16, fontweight='bold', y=1.00
    )

    output_path = output_dir / f"transfer_performance_heatmap_{metric}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"保存图表: {output_path}")
    plt.close()

    return output_path


def plot_feature_importance_comparison(
    cv_dir: Path,
    output_dir: Path,
    top_n: int = 20,
    dpi: int = 300
) -> List[Path]:
    """
    绘制特征重要性对比图。

    Args:
        cv_dir: 交叉验证结果目录
        output_dir: 输出目录
        top_n: 显示前 N 个重要特征
        dpi: 图像分辨率

    Returns:
        保存的图表路径列表
    """
    fi_dir = cv_dir / "feature_importance"

    if not fi_dir.exists():
        logger.warning(f"特征重要性目录不存在: {fi_dir}")
        return []

    fi_files = list(fi_dir.glob("*.csv"))

    if not fi_files:
        logger.warning("未找到特征重要性文件")
        return []

    logger.info(f"找到 {len(fi_files)} 个特征重要性文件")

    output_paths = []

    for fi_file in fi_files:
        try:
            df_fi = pd.read_csv(fi_file)

            # 只保留前 N 个特征
            df_fi = df_fi.head(top_n)

            # 绘制条形图
            fig, ax = plt.subplots(figsize=(10, 8))

            ax.barh(df_fi['feature'], df_fi['importance'], alpha=0.7,
                   edgecolor='black', color='steelblue')

            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_title(f"Top {top_n} Feature Importance - {fi_file.stem}")
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()

            output_path = output_dir / f"fi_{fi_file.stem}.png"
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"保存特征重要性图: {output_path}")
            plt.close()

            output_paths.append(output_path)

        except Exception as e:
            logger.error(f"处理 {fi_file} 时出错: {e}")
            continue

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="模型结果可视化"
    )
    parser.add_argument(
        '--cv_dir',
        type=Path,
        default=Path('outputs/model_results/cv_results'),
        help='交叉验证结果目录'
    )
    parser.add_argument(
        '--transfer_dir',
        type=Path,
        default=Path('outputs/model_results/transfer_results'),
        help='迁移学习结果目录'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('outputs/model_results/figures'),
        help='输出目录'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='图像分辨率'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['accuracy', 'f1', 'auc'],
        help='要可视化的指标'
    )

    args = parser.parse_args()

    setup_logging()
    logger.info("开始模型结果可视化")
    logger.info(f"交叉验证结果目录: {args.cv_dir}")
    logger.info(f"迁移学习结果目录: {args.transfer_dir}")
    logger.info(f"输出目录: {args.output_dir}")

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 加载交叉验证结果
    df_cv = load_cv_results(args.cv_dir)

    # 加载迁移学习结果
    df_transfer = load_transfer_results(args.transfer_dir)

    # 绘制交叉验证性能比较图
    if not df_cv.empty:
        for metric in args.metrics:
            if f'mean_{metric}' in df_cv.columns:
                try:
                    plot_cv_performance_comparison(
                        df_cv, args.output_dir, metric, args.dpi
                    )
                except Exception as e:
                    logger.error(f"绘制 CV 性能图 ({metric}) 时出错: {e}")
            else:
                logger.warning(f"指标 {metric} 不在交叉验证结果中")

    # 绘制迁移学习性能热力图
    if not df_transfer.empty:
        for metric in args.metrics:
            if metric in df_transfer.columns:
                try:
                    plot_transfer_performance_heatmap(
                        df_transfer, args.output_dir, metric, args.dpi
                    )
                except Exception as e:
                    logger.error(f"绘制迁移性能热力图 ({metric}) 时出错: {e}")
            else:
                logger.warning(f"指标 {metric} 不在迁移学习结果中")

    # 绘制特征重要性对比图
    try:
        plot_feature_importance_comparison(
            args.cv_dir, args.output_dir, top_n=20, dpi=args.dpi
        )
    except Exception as e:
        logger.error(f"绘制特征重要性图时出错: {e}")

    logger.info("\n所有可视化完成！")
    logger.info(f"图表保存在: {args.output_dir}")


if __name__ == "__main__":
    main()

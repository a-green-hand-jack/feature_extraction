#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 特征提取质量可视化脚本

验证和展示 Phase 1 中分子特征提取的质量和统计特性。
包括：
1. 数据过滤流程可视化（原始 → 处理后）
2. 结构特征共现分析（热力图和 Upset Plot）
3. 多数据集雷达图对比
4. 标签转换验证

用法：
    uv run python scripts/phase1_visualize.py \\
        --raw_dir data/raw/ \\
        --processed_dir data/processed/ \\
        --output_dir outputs/figures/phase1/ \\
        --dpi 300 \\
        --format png
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # 使用非 GUI 后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from upsetplot import UpSet, from_indicators

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def setup_logging(log_level: int = logging.INFO) -> None:
    """
    配置日志系统。

    Args:
        log_level (int): 日志级别。默认 logging.INFO。
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("phase1_visualization.log"),
        ],
    )


def load_dataset_pair(
    raw_path: Path,
    processed_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载对应的原始和处理后数据集。

    Args:
        raw_path (Path): 原始 CSV 文件路径。
        processed_path (Path): 处理后 CSV 文件路径。

    Returns:
        tuple: (raw_df, processed_df) 元组。
    """
    logger = logging.getLogger(__name__)

    raw_df = pd.read_csv(raw_path)
    processed_df = pd.read_csv(processed_path)

    logger.info(
        f"加载数据集: {raw_path.stem} "
        f"(原始: {len(raw_df)}, 处理后: {len(processed_df)})"
    )

    return raw_df, processed_df


def match_datasets(
    raw_dir: Path,
    processed_dir: Path
) -> Dict[str, Tuple[Path, Path]]:
    """
    匹配原始和处理后的数据集文件。

    Args:
        raw_dir (Path): 原始数据目录。
        processed_dir (Path): 处理后数据目录。

    Returns:
        dict: {dataset_name: (raw_path, processed_path)} 字典。
    """
    logger = logging.getLogger(__name__)

    matched = {}

    for processed_file in sorted(processed_dir.glob("*_processed.csv")):
        # 提取数据集名称（去除 _processed 后缀）
        dataset_name = processed_file.stem.replace("_processed", "")

        # 查找对应的原始文件
        raw_file = raw_dir / f"{dataset_name}.csv"

        if raw_file.exists():
            matched[dataset_name] = (raw_file, processed_file)
            logger.info(f"匹配数据集: {dataset_name}")
        else:
            logger.warning(f"未找到原始文件: {raw_file}")

    logger.info(f"共匹配 {len(matched)} 个数据集")

    return matched


def plot_data_filtering_overview(
    dataset_stats: Dict[str, Dict],
    output_dir: Path,
    dpi: int = 300,
    format: str = "png"
) -> Path:
    """
    绘制数据过滤流程概览图（堆叠条形图）。

    Args:
        dataset_stats (dict): 数据集统计信息。
        output_dir (Path): 输出目录。
        dpi (int): 图像分辨率。
        format (str): 输出格式。

    Returns:
        Path: 保存的图表路径。
    """
    logger = logging.getLogger(__name__)

    fig, ax = plt.subplots(figsize=(12, 8))

    datasets = list(dataset_stats.keys())
    retained = [dataset_stats[d]['retained'] for d in datasets]
    filtered = [dataset_stats[d]['filtered'] for d in datasets]

    # 绘制堆叠条形图
    ax.barh(datasets, retained, label='Retained', color='#2ecc71', alpha=0.8)
    ax.barh(datasets, filtered, left=retained, label='Filtered', color='#e74c3c', alpha=0.8)

    # 添加数据标签
    for i, dataset in enumerate(datasets):
        total = dataset_stats[dataset]['total']
        retention_rate = dataset_stats[dataset]['retention_rate']

        # 保留样本数
        ax.text(
            retained[i] / 2, i, f"{retained[i]}",
            ha='center', va='center', color='white',
            fontweight='bold', fontsize=10
        )

        # 过滤样本数
        if filtered[i] > 0:
            ax.text(
                retained[i] + filtered[i] / 2, i, f"{filtered[i]}",
                ha='center', va='center', color='white',
                fontweight='bold', fontsize=10
            )

        # 保留率
        ax.text(
            total + total * 0.02, i, f"{retention_rate:.1f}%",
            ha='left', va='center', fontsize=10
        )

    ax.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title(
        'Data Filtering Overview: Raw → Processed',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"data_filtering_overview.{format}"
    plt.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    logger.info(f"保存数据过滤概览图: {output_path}")
    plt.close()

    return output_path


def plot_quality_statistics_table(
    dataset_stats: Dict[str, Dict],
    output_dir: Path,
    dpi: int = 300,
    format: str = "png"
) -> Path:
    """
    绘制质量统计表格图。

    Args:
        dataset_stats (dict): 数据集统计信息。
        output_dir (Path): 输出目录。
        dpi (int): 图像分辨率。
        format (str): 输出格式。

    Returns:
        Path: 保存的图表路径。
    """
    logger = logging.getLogger(__name__)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据
    table_data = []
    for dataset, stats in dataset_stats.items():
        table_data.append([
            dataset,
            stats['total'],
            stats['retained'],
            stats['filtered'],
            f"{stats['retention_rate']:.1f}%",
            f"{stats['sif_valid_rate']:.1f}%",
            f"{stats['sgf_valid_rate']:.1f}%"
        ])

    col_labels = [
        'Dataset', 'Raw Samples', 'Retained', 'Filtered',
        'Retention Rate', 'SIF Valid Rate', 'SGF Valid Rate'
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')

    plt.title(
        'Phase 1: Data Quality Statistics',
        fontsize=14, fontweight='bold', pad=20
    )

    plt.tight_layout()
    output_path = output_dir / f"quality_statistics_table.{format}"
    plt.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    logger.info(f"保存质量统计表: {output_path}")
    plt.close()

    return output_path


def plot_feature_cooccurrence_heatmap(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    dpi: int = 300,
    format: str = "png"
) -> Path:
    """
    绘制结构特征共现热力图（4×4 矩阵）。

    Args:
        df (pd.DataFrame): 处理后的数据框。
        dataset_name (str): 数据集名称。
        output_dir (Path): 输出目录。
        dpi (int): 图像分辨率。
        format (str): 输出格式。

    Returns:
        Path: 保存的图表路径。
    """
    logger = logging.getLogger(__name__)

    # 准备特征列表
    features = ['is_monomer', 'is_dimer', 'is_cyclic', 'has_disulfide_bond']
    feature_labels = ['Monomer', 'Dimer', 'Cyclic', 'Disulfide Bond']

    # 确保布尔值类型正确
    for feat in features:
        if df[feat].dtype == 'object':
            df[feat] = df[feat].map({'True': True, 'False': False, True: True, False: False})
        df[feat] = df[feat].astype(bool)

    # 计算共现矩阵
    cooccurrence = np.zeros((4, 4))
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            cooccurrence[i, j] = ((df[feat1] == True) & (df[feat2] == True)).sum()

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cooccurrence,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        xticklabels=feature_labels,
        yticklabels=feature_labels,
        cbar_kws={'label': 'Number of Samples'},
        ax=ax,
        square=True
    )

    ax.set_title(
        f'Structural Feature Co-occurrence Matrix\n{dataset_name}',
        fontsize=14, fontweight='bold', pad=20
    )

    plt.tight_layout()
    output_path = output_dir / f"{dataset_name}_cooccurrence_heatmap.{format}"
    plt.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    logger.info(f"保存特征共现热力图: {output_path}")
    plt.close()

    return output_path


def plot_upset_plot(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    dpi: int = 300,
    format: str = "png"
) -> Path:
    """
    绘制 Upset Plot（集合关系图）。

    Args:
        df (pd.DataFrame): 处理后的数据框。
        dataset_name (str): 数据集名称。
        output_dir (Path): 输出目录。
        dpi (int): 图像分辨率。
        format (str): 输出格式。

    Returns:
        Path: 保存的图表路径。
    """
    logger = logging.getLogger(__name__)

    # 准备特征列表
    features = ['is_monomer', 'is_dimer', 'is_cyclic', 'has_disulfide_bond']
    feature_labels = ['is_monomer', 'is_dimer', 'is_cyclic', 'has_disulfide_bond']

    # 创建只包含布尔特征的数据框，确保都是布尔类型
    df_bool = df[features].copy()
    for feat in features:
        # 如果是object类型的字符串，转换为布尔值
        if df_bool[feat].dtype == 'object':
            df_bool[feat] = df_bool[feat].map({'True': True, 'False': False, True: True, False: False})
        # 确保是布尔类型
        df_bool[feat] = df_bool[feat].astype(bool)

    # 验证所有列都是布尔类型
    logger.debug(f"df_bool dtypes: {df_bool.dtypes.to_dict()}")
    logger.debug(f"df_bool shape: {df_bool.shape}")

    # 创建 Upset 数据
    upset_data = from_indicators(feature_labels, data=df_bool)

    # 绘制 Upset Plot
    upset = UpSet(upset_data, subset_size='count', show_counts=True)
    fig = plt.figure(figsize=(12, 6))
    upset.plot(fig=fig)

    plt.suptitle(
        f'Structural Feature Combinations\n{dataset_name}',
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout()
    output_path = output_dir / f"{dataset_name}_upset_plot.{format}"
    plt.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    logger.info(f"保存 Upset Plot: {output_path}")
    plt.close()

    return output_path


def plot_radar_chart(
    dataset_features: Dict[str, Dict],
    output_dir: Path,
    dpi: int = 300,
    format: str = "png"
) -> Path:
    """
    绘制多数据集雷达图对比。

    Args:
        dataset_features (dict): 数据集特征字典。
        output_dir (Path): 输出目录。
        dpi (int): 图像分辨率。
        format (str): 输出格式。

    Returns:
        Path: 保存的图表路径。
    """
    logger = logging.getLogger(__name__)

    categories = [
        'Monomer %', 'Cyclic %', 'Disulfide %',
        'Retention %', 'SIF Valid %', 'SGF Valid %'
    ]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合多边形

    colors = plt.cm.Set2(np.linspace(0, 1, len(dataset_features)))

    for idx, (dataset, features) in enumerate(dataset_features.items()):
        values = [
            features['monomer_ratio'],
            features['cyclic_ratio'],
            features['disulfide_ratio'],
            features['retention_rate'],
            features['sif_valid_rate'],
            features['sgf_valid_rate']
        ]
        values += values[:1]  # 闭合多边形

        ax.plot(angles, values, 'o-', linewidth=2, label=dataset, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.set_title(
        'Multi-Dataset Feature Comparison',
        fontsize=14, fontweight='bold', pad=30
    )
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    output_path = output_dir / f"all_datasets_radar_chart.{format}"
    plt.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    logger.info(f"保存雷达图: {output_path}")
    plt.close()

    return output_path


def plot_label_conversion_validation(
    all_processed_data: Dict[str, pd.DataFrame],
    output_dir: Path,
    dpi: int = 300,
    format: str = "png"
) -> Tuple[Path, Path]:
    """
    绘制标签转换验证图表。

    Args:
        all_processed_data (dict): 所有处理后的数据。
        output_dir (Path): 输出目录。
        dpi (int): 图像分辨率。
        format (str): 输出格式。

    Returns:
        tuple: (映射表路径, 分布图路径)
    """
    logger = logging.getLogger(__name__)

    # 1. 绘制标签映射表
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # 收集所有唯一的原始标签和转换值
    sif_mapping = {}
    sgf_mapping = {}

    for df in all_processed_data.values():
        for _, row in df.iterrows():
            sif_orig = str(row.get('SIF_class', ''))
            sif_conv = row.get('SIF_class_min', -1)
            if sif_orig and sif_orig != 'nan':
                sif_mapping[sif_orig] = sif_conv

            sgf_orig = str(row.get('SGF_class', ''))
            sgf_conv = row.get('SGF_class_min', -1)
            if sgf_orig and sgf_orig != 'nan':
                sgf_mapping[sgf_orig] = sgf_conv

    # 合并映射
    all_mappings = {}
    for orig, conv in sif_mapping.items():
        all_mappings[orig] = conv
    for orig, conv in sgf_mapping.items():
        if orig not in all_mappings:
            all_mappings[orig] = conv

    # 排序
    sorted_mappings = sorted(all_mappings.items(), key=lambda x: x[1] if x[1] != -1 else 999)

    table_data = [[orig, f"{conv} min" if conv != -1 else "Missing"]
                  for orig, conv in sorted_mappings]

    table = ax.table(
        cellText=table_data,
        colLabels=['Original Label', 'Converted (minutes)'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # 设置表头样式
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')

    plt.title(
        'Label Conversion Mapping',
        fontsize=14, fontweight='bold', pad=20
    )

    plt.tight_layout()
    mapping_path = output_dir / f"label_mapping_table.{format}"
    plt.savefig(mapping_path, dpi=dpi, format=format, bbox_inches='tight')
    logger.info(f"保存标签映射表: {mapping_path}")
    plt.close()

    # 2. 绘制标签分布对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 收集所有SIF和SGF值
    all_sif = []
    all_sgf = []
    for df in all_processed_data.values():
        sif_vals = df['SIF_class_min'].values
        sgf_vals = df['SGF_class_min'].values
        all_sif.extend(sif_vals[sif_vals != -1])
        all_sgf.extend(sgf_vals[sgf_vals != -1])

    # 绘制SIF分布
    axes[0].hist(all_sif, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('SIF Stability (minutes)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    axes[0].set_title('SIF Label Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # 绘制SGF分布
    axes[1].hist(all_sgf, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('SGF Stability (minutes)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    axes[1].set_title('SGF Label Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle(
        'Converted Label Distribution Comparison',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    dist_path = output_dir / f"label_distribution_comparison.{format}"
    plt.savefig(dist_path, dpi=dpi, format=format, bbox_inches='tight')
    logger.info(f"保存标签分布对比图: {dist_path}")
    plt.close()

    return mapping_path, dist_path


def main() -> int:
    """
    主函数：生成 Phase 1 特征提取质量可视化。

    Returns:
        int: 程序退出码（0 表示成功）。
    """
    parser = argparse.ArgumentParser(
        description="生成 Phase 1 特征提取质量可视化图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  uv run python scripts/phase1_visualize.py "
            "--raw_dir data/raw/ "
            "--processed_dir data/processed/ "
            "--output_dir outputs/figures/phase1/\n"
        ),
    )

    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
        help="原始数据目录（默认: data/raw）",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed"),
        help="处理后数据目录（默认: data/processed）",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/figures/phase1"),
        help="输出可视化图表的目录（默认: outputs/figures/phase1）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="图像分辨率，DPI（默认: 300）",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="输出图像格式（默认: png）",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别（默认: INFO）",
    )

    args = parser.parse_args()

    # 配置日志
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("开始 Phase 1 特征提取质量可视化")
    logger.info("=" * 70)
    logger.info(f"原始数据目录: {args.raw_dir.resolve()}")
    logger.info(f"处理后数据目录: {args.processed_dir.resolve()}")
    logger.info(f"输出目录: {args.output_dir.resolve()}")
    logger.info(f"图像分辨率: {args.dpi} DPI")
    logger.info(f"输出格式: {args.format}")

    # 验证输入目录
    if not args.raw_dir.exists():
        logger.error(f"原始数据目录不存在: {args.raw_dir}")
        return 2

    if not args.processed_dir.exists():
        logger.error(f"处理后数据目录不存在: {args.processed_dir}")
        return 2

    # 创建输出目录
    subdirs = ['quality_reports', 'feature_cooccurrence',
               'multi_dataset_comparison', 'label_conversion']

    for subdir in subdirs:
        subdir_path = args.output_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建输出子目录: {subdir_path.resolve()}")

    try:
        # 匹配数据集
        matched_datasets = match_datasets(args.raw_dir, args.processed_dir)

        if not matched_datasets:
            logger.error("未找到匹配的数据集")
            return 2

        # 收集统计信息
        dataset_stats = {}
        dataset_features = {}
        all_processed_data = {}

        for dataset_name, (raw_path, processed_path) in matched_datasets.items():
            raw_df, processed_df = load_dataset_pair(raw_path, processed_path)
            all_processed_data[dataset_name] = processed_df

            # 确保布尔特征类型正确
            for feat in ['is_monomer', 'is_dimer', 'is_cyclic', 'has_disulfide_bond']:
                if processed_df[feat].dtype == 'object':
                    processed_df[feat] = processed_df[feat].map({
                        'True': True, 'False': False, True: True, False: False
                    })
                processed_df[feat] = processed_df[feat].astype(bool)

            # 统计信息
            total = len(raw_df)
            retained = len(processed_df)
            filtered = total - retained
            retention_rate = (retained / total) * 100 if total > 0 else 0

            sif_valid = (processed_df['SIF_class_min'] != -1).sum()
            sgf_valid = (processed_df['SGF_class_min'] != -1).sum()
            sif_valid_rate = (sif_valid / retained) * 100 if retained > 0 else 0
            sgf_valid_rate = (sgf_valid / retained) * 100 if retained > 0 else 0

            dataset_stats[dataset_name] = {
                'total': total,
                'retained': retained,
                'filtered': filtered,
                'retention_rate': retention_rate,
                'sif_valid_rate': sif_valid_rate,
                'sgf_valid_rate': sgf_valid_rate
            }

            # 特征比例
            monomer_ratio = (processed_df['is_monomer'].sum() / retained) * 100
            cyclic_ratio = (processed_df['is_cyclic'].sum() / retained) * 100
            disulfide_ratio = (processed_df['has_disulfide_bond'].sum() / retained) * 100

            dataset_features[dataset_name] = {
                'monomer_ratio': monomer_ratio,
                'cyclic_ratio': cyclic_ratio,
                'disulfide_ratio': disulfide_ratio,
                'retention_rate': retention_rate,
                'sif_valid_rate': sif_valid_rate,
                'sgf_valid_rate': sgf_valid_rate
            }

        # 1. 绘制数据过滤概览
        logger.info("绘制数据过滤流程概览...")
        plot_data_filtering_overview(
            dataset_stats,
            args.output_dir / 'quality_reports',
            args.dpi,
            args.format
        )

        # 2. 绘制质量统计表
        logger.info("绘制质量统计表...")
        plot_quality_statistics_table(
            dataset_stats,
            args.output_dir / 'quality_reports',
            args.dpi,
            args.format
        )

        # 3. 绘制特征共现热力图和 Upset Plot
        logger.info("绘制特征共现分析...")
        for dataset_name, processed_df in all_processed_data.items():
            plot_feature_cooccurrence_heatmap(
                processed_df,
                dataset_name,
                args.output_dir / 'feature_cooccurrence',
                args.dpi,
                args.format
            )

            plot_upset_plot(
                processed_df,
                dataset_name,
                args.output_dir / 'feature_cooccurrence',
                args.dpi,
                args.format
            )

        # 4. 绘制雷达图
        logger.info("绘制多数据集雷达图...")
        plot_radar_chart(
            dataset_features,
            args.output_dir / 'multi_dataset_comparison',
            args.dpi,
            args.format
        )

        # 5. 绘制标签转换验证
        logger.info("绘制标签转换验证图表...")
        plot_label_conversion_validation(
            all_processed_data,
            args.output_dir / 'label_conversion',
            args.dpi,
            args.format
        )

        logger.info("=" * 70)
        logger.info("Phase 1 可视化生成完成")
        logger.info("=" * 70)
        logger.info(f"输出目录: {args.output_dir.resolve()}")
        logger.info(f"生成图表数: {len(matched_datasets) * 2 + 5} 张")
        logger.info("所有可视化图表生成成功！")

        return 0

    except Exception as e:
        logger.error(f"生成可视化时出错: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

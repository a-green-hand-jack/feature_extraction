#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
添加分子特征脚本

为原始 CSV 数据添加分子结构特征列，并将标签统一转换为分钟单位。

新增特征列：
- is_monomer: 是否为单体 (bool)
- is_dimer: 是否为二聚体 (bool)
- is_cyclic: 是否含环状结构 (bool)
- has_disulfide_bond: 是否含二硫键 (bool)

新增标签列：
- SIF_class_min: SIF 标签（分钟）
- SGF_class_min: SGF 标签（分钟）

样本过滤规则：
- 如果 SIF 和 SGF 两个标签都缺失（都是 -1），则跳过该样本
- 只要有一个标签有效，保留样本，另一个用 -1 占位

用法：
    python scripts/add_molecular_features.py \\
        --input_dir data/raw/ \\
        --output_dir data/processed/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from feature_extraction.utils import (
    get_csv_files,
    load_csv_safely,
    extract_molecular_features,
    convert_label_to_minutes,
)


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
            logging.FileHandler("add_molecular_features.log"),
        ],
    )


logger = logging.getLogger(__name__)


def process_single_csv(
    csv_path: Path, output_dir: Path
) -> Tuple[bool, Dict[str, int]]:
    """
    处理单个 CSV 文件，添加分子特征并转换标签。

    Args:
        csv_path (Path): 输入 CSV 文件路径
        output_dir (Path): 输出目录

    Returns:
        Tuple[bool, Dict[str, int]]:
            - 是否处理成功
            - 统计信息字典
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"处理文件: {csv_path.name}")
    logger.info(f"{'='*60}")

    # 加载 CSV
    df, status = load_csv_safely(csv_path, required_columns=["id", "SMILES"])
    if df is None:
        logger.error(f"加载失败: {status}")
        return False, {}

    original_count = len(df)
    logger.info(f"原始样本数: {original_count}")

    # 统计信息
    stats = {
        "original_count": original_count,
        "filtered_count": 0,
        "monomer_count": 0,
        "dimer_count": 0,
        "cyclic_count": 0,
        "disulfide_count": 0,
        "sif_valid_count": 0,
        "sgf_valid_count": 0,
        "both_valid_count": 0,
        "both_missing_count": 0,
    }

    # 提取分子特征
    logger.info("提取分子特征...")
    feature_records = []
    for idx, row in df.iterrows():
        smiles = row["SMILES"]
        features = extract_molecular_features(smiles)
        feature_records.append(features)
        if (idx + 1) % 100 == 0:
            logger.info(f"  已处理 {idx + 1}/{len(df)} 条样本...")

    # 添加特征列到数据框
    feature_df = pd.DataFrame(feature_records)
    df = pd.concat([df, feature_df], axis=1)

    # 转换标签
    logger.info("转换标签到分钟单位...")
    sif_col = "SIF_class" if "SIF_class" in df.columns else None
    sgf_col = "SGF_class" if "SGF_class" in df.columns else None

    if sif_col:
        df["SIF_class_min"] = df[sif_col].apply(convert_label_to_minutes)
    else:
        df["SIF_class_min"] = -1

    if sgf_col:
        df["SGF_class_min"] = df[sgf_col].apply(convert_label_to_minutes)
    else:
        df["SGF_class_min"] = -1

    # 过滤样本：如果两个标签都缺失，则跳过
    logger.info("过滤缺失标签样本...")
    mask_both_missing = (df["SIF_class_min"] == -1) & (df["SGF_class_min"] == -1)
    stats["both_missing_count"] = mask_both_missing.sum()

    df_filtered = df[~mask_both_missing].copy()
    stats["filtered_count"] = len(df_filtered)

    logger.info(f"过滤掉双标签缺失样本: {stats['both_missing_count']} 个")
    logger.info(f"保留样本数: {stats['filtered_count']}")

    # 统计特征分布
    stats["monomer_count"] = df_filtered["is_monomer"].sum()
    stats["dimer_count"] = df_filtered["is_dimer"].sum()
    stats["cyclic_count"] = df_filtered["is_cyclic"].sum()
    stats["disulfide_count"] = df_filtered["has_disulfide_bond"].sum()
    stats["sif_valid_count"] = (df_filtered["SIF_class_min"] != -1).sum()
    stats["sgf_valid_count"] = (df_filtered["SGF_class_min"] != -1).sum()
    stats["both_valid_count"] = (
        (df_filtered["SIF_class_min"] != -1) & (df_filtered["SGF_class_min"] != -1)
    ).sum()

    # 调整列顺序
    column_order = [
        "id",
        "SMILES",
        "is_monomer",
        "is_dimer",
        "is_cyclic",
        "has_disulfide_bond",
    ]

    # 添加原始标签列（如果存在）
    if sif_col:
        column_order.append(sif_col)
    if sgf_col:
        column_order.append(sgf_col)

    # 添加转换后的标签列
    column_order.extend(["SIF_class_min", "SGF_class_min"])

    # 添加其他可能存在的列
    for col in df_filtered.columns:
        if col not in column_order:
            column_order.append(col)

    df_filtered = df_filtered[column_order]

    # 保存处理后的 CSV
    output_path = output_dir / f"{csv_path.stem}_processed.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)
    logger.info(f"已保存处理后的文件: {output_path}")

    # 输出统计摘要
    logger.info(f"\n{'='*60}")
    logger.info("统计摘要:")
    logger.info(f"{'='*60}")
    logger.info(f"原始样本数: {stats['original_count']}")
    logger.info(f"过滤后样本数: {stats['filtered_count']}")
    logger.info(f"过滤掉样本数: {stats['both_missing_count']}")
    logger.info(f"\n分子特征分布:")
    logger.info(f"  - 单体: {stats['monomer_count']} ({stats['monomer_count']/stats['filtered_count']*100:.1f}%)")
    logger.info(f"  - 二聚体: {stats['dimer_count']} ({stats['dimer_count']/stats['filtered_count']*100:.1f}%)")
    logger.info(f"  - 环化: {stats['cyclic_count']} ({stats['cyclic_count']/stats['filtered_count']*100:.1f}%)")
    logger.info(f"  - 二硫键: {stats['disulfide_count']} ({stats['disulfide_count']/stats['filtered_count']*100:.1f}%)")
    logger.info(f"\n标签有效性:")
    logger.info(f"  - SIF 有效: {stats['sif_valid_count']} ({stats['sif_valid_count']/stats['filtered_count']*100:.1f}%)")
    logger.info(f"  - SGF 有效: {stats['sgf_valid_count']} ({stats['sgf_valid_count']/stats['filtered_count']*100:.1f}%)")
    logger.info(f"  - 双标签有效: {stats['both_valid_count']} ({stats['both_valid_count']/stats['filtered_count']*100:.1f}%)")
    logger.info(f"{'='*60}\n")

    return True, stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="为原始 CSV 数据添加分子特征并转换标签",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw"),
        help="输入目录路径（包含原始 CSV 文件）",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="输出目录路径",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()

    # 配置日志
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)

    logger.info("="*60)
    logger.info("分子特征提取脚本")
    logger.info("="*60)
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("="*60)

    # 获取所有 CSV 文件
    csv_files = get_csv_files(args.input_dir)
    if not csv_files:
        logger.error("未找到 CSV 文件，退出")
        return

    # 处理每个文件
    total_stats = {
        "total_files": len(csv_files),
        "successful_files": 0,
        "total_original": 0,
        "total_filtered": 0,
        "total_monomer": 0,
        "total_dimer": 0,
        "total_cyclic": 0,
        "total_disulfide": 0,
    }

    for csv_file in csv_files:
        success, stats = process_single_csv(csv_file, args.output_dir)
        if success:
            total_stats["successful_files"] += 1
            total_stats["total_original"] += stats["original_count"]
            total_stats["total_filtered"] += stats["filtered_count"]
            total_stats["total_monomer"] += stats["monomer_count"]
            total_stats["total_dimer"] += stats["dimer_count"]
            total_stats["total_cyclic"] += stats["cyclic_count"]
            total_stats["total_disulfide"] += stats["disulfide_count"]

    # 输出总体统计
    logger.info("\n" + "="*60)
    logger.info("总体统计摘要")
    logger.info("="*60)
    logger.info(f"处理文件数: {total_stats['successful_files']}/{total_stats['total_files']}")
    logger.info(f"总原始样本数: {total_stats['total_original']}")
    logger.info(f"总过滤后样本数: {total_stats['total_filtered']}")
    logger.info(f"总过滤掉样本数: {total_stats['total_original'] - total_stats['total_filtered']}")
    logger.info(f"\n总体分子特征分布:")
    logger.info(f"  - 单体: {total_stats['total_monomer']}")
    logger.info(f"  - 二聚体: {total_stats['total_dimer']}")
    logger.info(f"  - 环化: {total_stats['total_cyclic']}")
    logger.info(f"  - 二硫键: {total_stats['total_disulfide']}")
    logger.info("="*60)
    logger.info("处理完成！")


if __name__ == "__main__":
    main()

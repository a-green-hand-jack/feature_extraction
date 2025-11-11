#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIF/SGF 肽类稳定性特征提取脚本

批量提取 cleaned 文件夹下所有 CSV 文件的分子特征，
并将结果保存为 NumPy 压缩格式（.npz）。

特征包括：
- QED 属性（分子量、LogP、HBA、HBD、PSA、旋转键数、芳香性、警报）
- 物理化学描述符（脂溶性、刚性、分子大小等）
- Gasteiger 电荷统计
- Morgan 指纹
- Avalon 指纹（可选）

用法：
    python scripts/extract_features.py \\
        --input_dir data/cleaned/ \\
        --output_dir outputs/features/ \\
        --morgan_bits 1024 \\
        --avalon_bits 512
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# 添加项目根目录到路径，使脚本能导入 src 目录下的模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from feature_extraction import PeptideFeaturizer
from feature_extraction.utils import (
    get_csv_files, load_csv_safely, build_output_path,
    save_features_to_npz, format_batch_summary
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
            logging.FileHandler("feature_extraction.log"),
        ],
    )


def process_single_file(
    csv_path: Path,
    output_dir: Path,
    featurizer: PeptideFeaturizer,
) -> tuple[bool, dict]:
    """
    处理单个 CSV 文件并提取特征。

    Args:
        csv_path (Path): 输入 CSV 文件路径。
        output_dir (Path): 输出特征文件的目录。
        featurizer (PeptideFeaturizer): 特征提取器实例。

    Returns:
        tuple[bool, dict]: 
            - 第一个元素：处理是否成功（bool）
            - 第二个元素：包含处理统计信息的字典
                {
                    'total_rows': int,
                    'valid_rows': int,
                    'invalid_rows': int,
                    'output_path': Path,
                }
    """
    logger = logging.getLogger(__name__)
    
    # 加载 CSV 文件
    required_columns = ["id", "SMILES", "SIF_class", "SGF_class"]
    df, load_msg = load_csv_safely(csv_path, required_columns)
    
    if df is None:
        return False, {"error": load_msg}
    
    logger.info(f"开始处理文件: {csv_path.name}")
    
    # 初始化特征和标签容器
    X = []
    y_sif = []
    y_sgf = []
    ids = []
    mask_valid = []
    
    # 遍历数据框，逐行提取特征
    for idx, row in df.iterrows():
        row_id = str(row["id"])
        smiles = str(row["SMILES"])
        sif_class = row["SIF_class"]
        sgf_class = row["SGF_class"]
        
        # 检查标签是否为空（NaN）或为特殊字符串
        if pd.isna(sif_class) or pd.isna(sgf_class):
            logger.debug(
                f"行 {idx}: 缺失标签 (SIF: {sif_class}, SGF: {sgf_class})，跳过"
            )
            mask_valid.append(False)
            continue
        
        # 处理特殊字符串（如 '----'）
        sif_str = str(sif_class).strip()
        sgf_str = str(sgf_class).strip()
        
        if sif_str == '' or sif_str == '----' or sgf_str == '' or sgf_str == '----':
            logger.debug(
                f"行 {idx}: 无效的标签字符串 (SIF: {sif_str}, SGF: {sgf_str})，跳过"
            )
            mask_valid.append(False)
            continue
        
        # 尝试转换为整数（先转浮点再转整数以支持 1.0 这样的值）
        try:
            sif_int = int(float(sif_str))
            sgf_int = int(float(sgf_str))
        except (ValueError, OverflowError):
            logger.debug(
                f"行 {idx}: 无法将标签转换为整数 (SIF: {sif_str}, SGF: {sgf_str})，跳过"
            )
            mask_valid.append(False)
            continue
        
        # 提取特征
        features, success = featurizer.featurize(smiles)
        
        if success and features is not None:
            X.append(features)
            y_sif.append(sif_int)
            y_sgf.append(sgf_int)
            ids.append(row_id)
            mask_valid.append(True)
        else:
            logger.debug(f"行 {idx}: 提取特征失败 (SMILES: {smiles})")
            mask_valid.append(False)
    
    # 检查是否有有效样本
    if len(X) == 0:
        error_msg = f"文件 {csv_path.name} 中没有有效样本"
        logger.error(error_msg)
        return False, {"error": error_msg}
    
    # 转换为 NumPy 数组
    X = np.asarray(X, dtype=np.float32)
    y_sif = np.asarray(y_sif, dtype=np.int32)
    y_sgf = np.asarray(y_sgf, dtype=np.int32)
    ids = np.asarray(ids, dtype=object)
    mask_valid = np.asarray(mask_valid, dtype=bool)
    feature_names = np.asarray(featurizer.get_feature_names(), dtype=object)
    
    # 构建元数据
    metadata = {
        "morgan_bits": featurizer.morgan_bits,
        "avalon_bits": featurizer.avalon_bits,
        "avalon_included": featurizer.use_avalon,
        "input_file": csv_path.name,
    }
    
    # 构建输出路径并保存
    output_path = build_output_path(csv_path, output_dir)
    save_success, save_msg = save_features_to_npz(
        output_path=output_path,
        X=X,
        y_sif=y_sif,
        y_sgf=y_sgf,
        ids=ids,
        feature_names=feature_names,
        mask_valid=mask_valid,
        metadata=metadata,
    )
    
    if not save_success:
        logger.error(f"文件 {csv_path.name} 保存失败")
        return False, {"error": save_msg}
    
    # 返回统计信息
    stats = {
        "total_rows": len(df),
        "valid_rows": len(X),
        "invalid_rows": len(df) - len(X),
        "output_path": output_path,
    }
    
    logger.info(f"文件处理完成: {csv_path.name}")
    logger.info(
        f"  - 总行数: {stats['total_rows']}, "
        f"有效行: {stats['valid_rows']}, "
        f"无效行: {stats['invalid_rows']}"
    )
    
    return True, stats


def main() -> int:
    """
    主函数：批量处理 CSV 文件并提取特征。

    Returns:
        int: 程序退出码（0 表示成功）。
    """
    parser = argparse.ArgumentParser(
        description="提取 SIF/SGF 肽类稳定性特征",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/extract_features.py "
            "--input_dir data/cleaned/ --output_dir outputs/features/\n"
        ),
    )
    
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/cleaned"),
        help="输入 CSV 文件所在的目录（默认: data/cleaned）",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/features"),
        help="输出特征文件的目录（默认: outputs/features）",
    )
    parser.add_argument(
        "--morgan_bits",
        type=int,
        default=1024,
        help="Morgan 指纹位数（默认: 1024）",
    )
    parser.add_argument(
        "--avalon_bits",
        type=int,
        default=512,
        help="Avalon 指纹位数（默认: 512）",
    )
    parser.add_argument(
        "--no_avalon",
        action="store_true",
        help="禁用 Avalon 指纹",
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
    
    logger.info("="*60)
    logger.info("开始 SIF/SGF 特征提取")
    logger.info("="*60)
    logger.info(f"输入目录: {args.input_dir.resolve()}")
    logger.info(f"输出目录: {args.output_dir.resolve()}")
    logger.info(f"Morgan 位数: {args.morgan_bits}")
    logger.info(f"Avalon 位数: {args.avalon_bits}")
    logger.info(f"启用 Avalon: {not args.no_avalon}")
    
    # 验证输入目录
    if not args.input_dir.exists():
        logger.error(f"输入目录不存在: {args.input_dir}")
        return 2
    
    if not args.input_dir.is_dir():
        logger.error(f"输入路径不是目录: {args.input_dir}")
        return 2
    
    # 创建输出目录
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录已创建: {args.output_dir.resolve()}")
    except OSError as e:
        logger.error(f"创建输出目录失败: {e}")
        return 2
    
    # 初始化特征提取器
    featurizer = PeptideFeaturizer(
        morgan_bits=args.morgan_bits,
        avalon_bits=args.avalon_bits,
        use_avalon=not args.no_avalon,
    )
    
    logger.info(f"特征维度: {featurizer.n_features}")
    
    # 获取所有 CSV 文件
    csv_files = get_csv_files(args.input_dir)
    
    if len(csv_files) == 0:
        logger.warning(f"在 {args.input_dir} 中未找到 CSV 文件")
        return 1
    
    logger.info(f"找到 {len(csv_files)} 个 CSV 文件待处理")
    
    # 批量处理 CSV 文件
    successful_files = 0
    total_samples = 0
    valid_samples = 0
    invalid_samples = 0
    
    for csv_path in csv_files:
        success, stats = process_single_file(
            csv_path, args.output_dir, featurizer
        )
        
        if success:
            successful_files += 1
            total_samples += stats["total_rows"]
            valid_samples += stats["valid_rows"]
            invalid_samples += stats["invalid_rows"]
        else:
            logger.error(f"处理文件失败: {csv_path.name}")
    
    # 输出摘要
    logger.info(
        format_batch_summary(
            total_files=len(csv_files),
            successful_files=successful_files,
            total_samples=total_samples,
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
        )
    )
    
    if successful_files == 0:
        logger.error("没有文件成功处理")
        return 1
    
    logger.info("特征提取完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())


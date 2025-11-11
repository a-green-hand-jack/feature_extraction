"""
辅助函数模块

提供特征提取过程中的辅助工具函数。
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def validate_csv_columns(
    df: pd.DataFrame, required_columns: List[str]
) -> Tuple[bool, List[str]]:
    """
    验证数据框是否包含所有必需列。

    Args:
        df (pd.DataFrame): 待验证的数据框。
        required_columns (List[str]): 必需的列名列表。

    Returns:
        Tuple[bool, List[str]]: 
            - 第一个元素：所有必需列是否都存在（bool）
            - 第二个元素：缺失的列名列表（若所有列都存在，则为空列表）
    """
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


def get_csv_files(directory: Path) -> List[Path]:
    """
    获取目录下所有 CSV 文件。

    Args:
        directory (Path): 输入目录路径。

    Returns:
        List[Path]: CSV 文件路径列表（按修改时间排序）。
    """
    if not directory.exists():
        logger.error(f"目录不存在: {directory}")
        return []
    
    if not directory.is_dir():
        logger.error(f"路径不是目录: {directory}")
        return []
    
    csv_files = sorted(directory.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    logger.info(f"在 {directory} 中找到 {len(csv_files)} 个 CSV 文件")
    
    return csv_files


def load_csv_safely(
    csv_path: Path, required_columns: Optional[List[str]] = None
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    安全地加载 CSV 文件。

    Args:
        csv_path (Path): CSV 文件路径。
        required_columns (Optional[List[str]]): 必需的列名列表。
            若为 None，则不进行列验证。

    Returns:
        Tuple[Optional[pd.DataFrame], str]:
            - 第一个元素：加载的数据框，若加载失败则为 None
            - 第二个元素：状态信息字符串
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"成功加载 CSV 文件: {csv_path} ({len(df)} 行)")
        
        if required_columns is not None:
            valid, missing = validate_csv_columns(df, required_columns)
            if not valid:
                msg = f"缺失必需列: {', '.join(missing)}"
                logger.error(msg)
                return None, msg
        
        return df, "ok"
    
    except FileNotFoundError:
        msg = f"文件不存在: {csv_path}"
        logger.error(msg)
        return None, msg
    
    except pd.errors.EmptyDataError:
        msg = f"CSV 文件为空: {csv_path}"
        logger.error(msg)
        return None, msg
    
    except Exception as e:
        msg = f"加载 CSV 文件时出错 ({csv_path}): {e}"
        logger.error(msg)
        return None, msg


def build_output_path(
    input_path: Path, output_dir: Path, suffix: str = ".npz"
) -> Path:
    """
    根据输入文件路径和输出目录构建输出文件路径。

    Args:
        input_path (Path): 输入文件路径。
        output_dir (Path): 输出目录。
        suffix (str): 输出文件后缀。默认 ".npz"。

    Returns:
        Path: 输出文件路径。
    """
    stem = input_path.stem  # 不包含后缀的文件名
    output_path = output_dir / f"{stem}{suffix}"
    return output_path


def save_features_to_npz(
    output_path: Path,
    X: np.ndarray,
    y_sif: np.ndarray,
    y_sgf: np.ndarray,
    ids: np.ndarray,
    feature_names: np.ndarray,
    mask_valid: np.ndarray,
    metadata: dict,
) -> Tuple[bool, str]:
    """
    将特征和标签保存为 NPZ 文件。

    Args:
        output_path (Path): 输出文件路径。
        X (np.ndarray): 特征矩阵，shape: (n_samples, n_features)。
        y_sif (np.ndarray): SIF 类标签，shape: (n_samples,)。
        y_sgf (np.ndarray): SGF 类标签，shape: (n_samples,)。
        ids (np.ndarray): 样本 ID，shape: (n_samples,)。
        feature_names (np.ndarray): 特征名称列表，shape: (n_features,)。
        mask_valid (np.ndarray): 有效样本掩码，shape: (n_rows,)。
        metadata (dict): 元数据字典。

    Returns:
        Tuple[bool, str]:
            - 第一个元素：是否保存成功（bool）
            - 第二个元素：状态信息字符串
    """
    try:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_path,
            X=X,
            y_sif=y_sif,
            y_sgf=y_sgf,
            ids=ids,
            feature_names=feature_names,
            mask_valid=mask_valid,
            metadata=np.asarray(metadata, dtype=object),
        )
        
        msg = f"成功保存特征文件: {output_path} ({X.shape[0]} 个有效样本, {X.shape[1]} 个特征)"
        logger.info(msg)
        return True, msg
    
    except Exception as e:
        msg = f"保存特征文件时出错 ({output_path}): {e}"
        logger.error(msg)
        return False, msg


def format_batch_summary(
    total_files: int,
    successful_files: int,
    total_samples: int,
    valid_samples: int,
    invalid_samples: int,
) -> str:
    """
    格式化批量处理摘要信息。

    Args:
        total_files (int): 处理的文件总数。
        successful_files (int): 成功处理的文件数。
        total_samples (int): 总样本数。
        valid_samples (int): 有效样本数。
        invalid_samples (int): 无效样本数。

    Returns:
        str: 格式化的摘要信息。
    """
    summary = (
        f"\n{'='*60}\n"
        f"批量特征提取摘要\n"
        f"{'='*60}\n"
        f"处理文件数: {successful_files}/{total_files}\n"
        f"总样本数: {total_samples}\n"
        f"  - 有效样本: {valid_samples}\n"
        f"  - 无效样本: {invalid_samples}\n"
        f"{'='*60}\n"
    )
    return summary


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迁移学习评估脚本

在两个数据集之间进行双向迁移学习测试：
- 在一个数据集上训练，在另一个数据集上测试
- 处理不同的类别定义（5类 vs 4类）通过类别映射

用法：
    python scripts/evaluate_transfer.py \\
        --dataset1 outputs/features/US9624268_cleaned.npz \\
        --dataset2 outputs/features/sif_sgf_second_cleaned.npz \\
        --output_dir outputs/model_results/transfer_results/
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def setup_logging(log_level: int = logging.INFO) -> None:
    """配置日志系统"""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("evaluate_transfer.log"),
        ],
    )


logger = logging.getLogger(__name__)


# US9624268 的 5 类映射到 4 类
# US9624268: 1(>360min), 2(180-360min), 3(120-180min), 4(60-120min), 5(<60min)
# sif_sgf_second: 1(<1h=60min), 2(60-90min), 3(90-120min), 4(>120min)
CLASS_MAPPING_5_TO_4 = {
    1: 4,  # >360 min → >120 min (Class 4: long stability)
    2: 4,  # 180-360 min → >120 min (Class 4)
    3: 4,  # 120-180 min → >120 min (Class 4)
    4: 2,  # 60-120 min → 60-90 min (Class 2, approximation)
    5: 1   # <60 min → <60 min (Class 1: short stability)
}


def map_labels(y: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """
    根据映射字典转换标签。

    Args:
        y: 原始标签数组
        mapping: 标签映射字典

    Returns:
        映射后的标签数组
    """
    y_mapped = np.array([mapping.get(int(label), label) for label in y])
    return y_mapped


def load_npz_data(
    npz_path: Path,
    target: str,
    apply_mapping: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    加载 NPZ 文件并提取特征和标签。

    Args:
        npz_path: NPZ 文件路径
        target: 目标变量 ('SIF' 或 'SGF')
        apply_mapping: 是否应用 5类→4类 映射

    Returns:
        X: 特征矩阵
        y: 标签向量 (0-based indices for XGBoost compatibility)
        ids: 样本 ID
        feature_names: 特征名称列表
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ 文件不存在: {npz_path}")

    npz = np.load(npz_path, allow_pickle=True)

    X = npz['X']
    feature_names = npz['feature_names'].tolist()
    ids = npz['ids'] if 'ids' in npz else np.arange(len(X), dtype=object)

    # 根据目标选择标签
    if target.upper() == 'SIF':
        y = npz['y_sif']
    elif target.upper() == 'SGF':
        y = npz['y_sgf']
    else:
        raise ValueError(f"无效的目标: {target}")

    # 过滤掉标签为 -1 的样本
    valid_mask = y != -1
    n_removed = np.sum(~valid_mask)

    if n_removed > 0:
        logger.info(f"过滤掉 {n_removed} 个缺失标签的样本")
        X = X[valid_mask]
        y = y[valid_mask]
        ids = ids[valid_mask]

    # 应用类别映射（如果需要）
    if apply_mapping:
        original_classes = np.unique(y)
        y = map_labels(y, CLASS_MAPPING_5_TO_4)
        mapped_classes = np.unique(y)
        logger.info(f"应用类别映射: {original_classes} → {mapped_classes}")

    # 转换为 0-based 标签 (XGBoost 需要)
    unique_labels = np.unique(y)
    label_mapping = {int(old_label): idx for idx, old_label in enumerate(unique_labels)}
    y_remapped = np.array([label_mapping[int(label)] for label in y])

    logger.info(
        f"加载数据: {npz_path.name}, 目标={target}, "
        f"样本数={len(X)}, 原始类别={unique_labels}, 映射为 0-based: {label_mapping}"
    )

    return X, y_remapped, ids, feature_names


def get_models() -> Dict[str, object]:
    """获取模型字典"""
    return {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs',
            multi_class='multinomial'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    }


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """计算类别权重用于 XGBoost"""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def evaluate_transfer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    model: object
) -> Dict[str, any]:
    """
    评估单个模型的迁移学习性能。

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        model_name: 模型名称
        model: 模型对象

    Returns:
        包含评估结果的字典
    """
    logger.info(f"训练 {model_name}...")

    # 训练模型
    if model_name == 'XGBoost':
        class_weights = compute_class_weights(y_train)
        sample_weights = np.array([class_weights[int(label)] for label in y_train])
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    else:
        model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except Exception as e:
        logger.warning(f"无法计算 AUC: {e}")
        auc = np.nan

    # 混淆矩阵和分类报告
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    result = {
        'model': model_name,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_classes': len(np.unique(y_test))
    }

    logger.info(
        f"{model_name} - Acc: {acc:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}"
    )

    return result


def run_transfer_experiment(
    train_npz: Path,
    test_npz: Path,
    target: str,
    output_dir: Path,
    direction: str
) -> None:
    """
    运行单向迁移学习实验。

    Args:
        train_npz: 训练数据集 NPZ 路径
        test_npz: 测试数据集 NPZ 路径
        target: 目标变量
        output_dir: 输出目录
        direction: 实验方向描述（用于日志）
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"迁移学习实验: {direction}")
    logger.info(f"目标: {target}")
    logger.info(f"{'='*80}")

    train_name = train_npz.stem
    test_name = test_npz.stem

    # 判断是否需要应用类别映射
    # US9624268 有 5 类，sif_sgf_second 有 4 类
    apply_train_mapping = 'US9624268' in train_name
    apply_test_mapping = 'US9624268' in test_name

    # 加载数据
    X_train, y_train, _, feature_names_train = load_npz_data(
        train_npz, target, apply_mapping=apply_train_mapping
    )
    X_test, y_test, _, feature_names_test = load_npz_data(
        test_npz, target, apply_mapping=apply_test_mapping
    )

    # 检查特征是否一致
    if feature_names_train != feature_names_test:
        logger.warning("训练集和测试集的特征名称不一致！")

    # 获取模型
    models = get_models()

    # 评估每个模型
    all_results = []

    for model_name, model in models.items():
        try:
            result = evaluate_transfer(
                X_train, y_train, X_test, y_test, model_name, model
            )
            result['train_dataset'] = train_name
            result['test_dataset'] = test_name
            result['target'] = target
            result['direction'] = direction
            all_results.append(result)
        except Exception as e:
            logger.error(f"评估 {model_name} 时出错: {e}", exc_info=True)
            continue

    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 详细结果
    output_file = output_dir / f"{train_name}_to_{test_name}_{target}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"保存详细结果: {output_file}")

    # CSV 汇总
    summary_data = []
    for result in all_results:
        summary_data.append({
            'train_dataset': result['train_dataset'],
            'test_dataset': result['test_dataset'],
            'target': result['target'],
            'direction': result['direction'],
            'model': result['model'],
            'n_train_samples': result['n_train_samples'],
            'n_test_samples': result['n_test_samples'],
            'n_classes': result['n_classes'],
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'auc': result['auc']
        })

    df_summary = pd.DataFrame(summary_data)
    csv_file = output_dir / f"{train_name}_to_{test_name}_{target}_summary.csv"
    df_summary.to_csv(csv_file, index=False)
    logger.info(f"保存汇总结果: {csv_file}")

    # 保存混淆矩阵
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    for result in all_results:
        cm = np.array(result['confusion_matrix'])
        cm_file = cm_dir / f"{train_name}_to_{test_name}_{target}_{result['model']}_cm.csv"
        pd.DataFrame(cm).to_csv(cm_file, index=False)

    logger.info(f"保存混淆矩阵: {cm_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="双向迁移学习评估"
    )
    parser.add_argument(
        '--dataset1',
        type=Path,
        default=Path('outputs/features/US9624268_cleaned.npz'),
        help='第一个数据集 NPZ 文件'
    )
    parser.add_argument(
        '--dataset2',
        type=Path,
        default=Path('outputs/features/sif_sgf_second_cleaned.npz'),
        help='第二个数据集 NPZ 文件'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('outputs/model_results/transfer_results'),
        help='输出目录'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        choices=['SIF', 'SGF', 'BOTH'],
        default=['BOTH'],
        help='目标变量'
    )

    args = parser.parse_args()

    setup_logging()
    logger.info("开始迁移学习评估")
    logger.info(f"数据集 1: {args.dataset1}")
    logger.info(f"数据集 2: {args.dataset2}")
    logger.info(f"输出目录: {args.output_dir}")

    # 确定目标变量
    targets = []
    if 'BOTH' in args.targets:
        targets = ['SIF', 'SGF']
    else:
        targets = args.targets

    # 双向迁移学习实验
    for target in targets:
        # 方向 1: dataset1 → dataset2
        try:
            run_transfer_experiment(
                args.dataset1,
                args.dataset2,
                target,
                args.output_dir,
                f"{args.dataset1.stem} → {args.dataset2.stem}"
            )
        except Exception as e:
            logger.error(f"实验失败: {e}", exc_info=True)

        # 方向 2: dataset2 → dataset1
        try:
            run_transfer_experiment(
                args.dataset2,
                args.dataset1,
                target,
                args.output_dir,
                f"{args.dataset2.stem} → {args.dataset1.stem}"
            )
        except Exception as e:
            logger.error(f"实验失败: {e}", exc_info=True)

    logger.info("\n所有迁移学习实验完成！")
    logger.info(f"结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()

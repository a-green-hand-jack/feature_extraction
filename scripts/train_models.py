#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练模型脚本 - 5折交叉验证

对每个数据集的 SIF 和 SGF 目标分别训练三种模型：
- Logistic Regression
- Random Forest
- XGBoost

使用 5 折分层交叉验证，使用 class_weight='balanced' 处理类别不平衡。

用法：
    python scripts/train_models.py \\
        --input_dir outputs/features/ \\
        --output_dir outputs/model_results/cv_results/ \\
        --n_folds 5
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
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
            logging.FileHandler("train_models.log"),
        ],
    )


logger = logging.getLogger(__name__)


def load_npz_data(npz_path: Path, target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[int, int]]:
    """
    加载 NPZ 文件并提取特征和标签。

    Args:
        npz_path: NPZ 文件路径
        target: 目标变量 ('SIF' 或 'SGF')

    Returns:
        X: 特征矩阵
        y: 标签向量 (0-based indices for XGBoost compatibility)
        ids: 样本 ID
        feature_names: 特征名称列表
        label_mapping: 原始标签到 0-based 标签的映射
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
        raise ValueError(f"无效的目标: {target}，必须是 'SIF' 或 'SGF'")

    # 过滤掉标签为 -1 的样本
    valid_mask = y != -1
    n_removed = np.sum(~valid_mask)

    if n_removed > 0:
        logger.info(f"过滤掉 {n_removed} 个缺失标签的样本 (标签为 -1)")
        X = X[valid_mask]
        y = y[valid_mask]
        ids = ids[valid_mask]

    logger.info(
        f"加载数据: {npz_path.name}, 目标={target}, "
        f"样本数={len(X)}, 特征数={X.shape[1]}"
    )
    logger.info(f"原始类别分布: {np.unique(y, return_counts=True)}")

    # 创建 0-based 标签映射 (XGBoost 需要)
    unique_labels = np.unique(y)
    label_mapping = {int(old_label): idx for idx, old_label in enumerate(unique_labels)}
    reverse_mapping = {idx: int(old_label) for old_label, idx in label_mapping.items()}

    # 将标签转换为 0-based
    y_remapped = np.array([label_mapping[int(label)] for label in y])

    logger.info(f"标签映射: {label_mapping}")

    return X, y_remapped, ids, feature_names, reverse_mapping


def get_models() -> Dict[str, object]:
    """
    获取模型字典。

    Returns:
        包含三个分类器的字典
    """
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
    """
    计算类别权重用于 XGBoost。

    Args:
        y: 标签向量

    Returns:
        类别权重字典
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model: object,
    n_folds: int = 5
) -> Dict[str, any]:
    """
    对单个模型执行分层 K 折交叉验证。

    Args:
        X: 特征矩阵
        y: 标签向量
        model_name: 模型名称
        model: 模型对象
        n_folds: 折数

    Returns:
        包含交叉验证结果的字典
    """
    logger.info(f"开始 {model_name} 的 {n_folds} 折交叉验证")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 对于 XGBoost，需要手动设置样本权重
        if model_name == 'XGBoost':
            class_weights = compute_class_weights(y_train)
            sample_weights = np.array([class_weights[int(label)] for label in y_train])
            model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # 计算指标
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # 计算 AUC (ovr: one-vs-rest)
        try:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        except Exception as e:
            logger.warning(f"无法计算 AUC: {e}")
            auc = np.nan

        fold_results.append({
            'fold': fold_idx,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.append(y_proba)

        logger.info(
            f"  Fold {fold_idx}: Acc={acc:.4f}, Precision={precision:.4f}, "
            f"Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
        )

    # 汇总结果
    df_folds = pd.DataFrame(fold_results)

    summary = {
        'model': model_name,
        'n_folds': n_folds,
        'mean_accuracy': df_folds['accuracy'].mean(),
        'std_accuracy': df_folds['accuracy'].std(),
        'mean_precision': df_folds['precision'].mean(),
        'std_precision': df_folds['precision'].std(),
        'mean_recall': df_folds['recall'].mean(),
        'std_recall': df_folds['recall'].std(),
        'mean_f1': df_folds['f1'].mean(),
        'std_f1': df_folds['f1'].std(),
        'mean_auc': df_folds['auc'].mean(),
        'std_auc': df_folds['auc'].std(),
        'fold_results': fold_results,
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred).tolist(),
        'classification_report': classification_report(
            all_y_true, all_y_pred, output_dict=True, zero_division=0
        )
    }

    # 提取特征重要性 (如果可用)
    if hasattr(model, 'feature_importances_'):
        summary['feature_importances'] = model.feature_importances_.tolist()
    elif hasattr(model, 'coef_'):
        # 对于 Logistic Regression，使用系数的绝对值平均
        summary['feature_importances'] = np.abs(model.coef_).mean(axis=0).tolist()

    logger.info(
        f"{model_name} 交叉验证完成: "
        f"Acc={summary['mean_accuracy']:.4f}±{summary['std_accuracy']:.4f}, "
        f"F1={summary['mean_f1']:.4f}±{summary['std_f1']:.4f}"
    )

    return summary


def train_on_dataset(
    npz_path: Path,
    target: str,
    output_dir: Path,
    n_folds: int = 5
) -> None:
    """
    对单个数据集的指定目标训练所有模型。

    Args:
        npz_path: NPZ 文件路径
        target: 目标变量 ('SIF' 或 'SGF')
        output_dir: 输出目录
        n_folds: 折数
    """
    dataset_name = npz_path.stem
    logger.info(f"\n{'='*80}")
    logger.info(f"数据集: {dataset_name}, 目标: {target}")
    logger.info(f"{'='*80}")

    # 加载数据
    X, y, ids, feature_names, reverse_mapping = load_npz_data(npz_path, target)

    # 检查样本数是否足够
    if len(X) < n_folds:
        logger.error(f"样本数 ({len(X)}) 少于折数 ({n_folds})，跳过此数据集")
        return

    # 获取模型
    models = get_models()

    # 训练每个模型
    all_results = []

    for model_name, model in models.items():
        try:
            result = cross_validate_model(X, y, model_name, model, n_folds)
            result['dataset'] = dataset_name
            result['target'] = target
            result['n_samples'] = len(X)
            result['n_features'] = X.shape[1]
            result['n_classes'] = len(np.unique(y))
            all_results.append(result)
        except Exception as e:
            logger.error(f"训练 {model_name} 时出错: {e}", exc_info=True)
            continue

    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 JSON 格式的详细结果
    output_file = output_dir / f"{dataset_name}_{target}_cv_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"保存详细结果: {output_file}")

    # 保存 CSV 格式的汇总结果
    summary_data = []
    for result in all_results:
        summary_data.append({
            'dataset': result['dataset'],
            'target': result['target'],
            'model': result['model'],
            'n_samples': result['n_samples'],
            'n_features': result['n_features'],
            'n_classes': result['n_classes'],
            'mean_accuracy': result['mean_accuracy'],
            'std_accuracy': result['std_accuracy'],
            'mean_precision': result['mean_precision'],
            'std_precision': result['std_precision'],
            'mean_recall': result['mean_recall'],
            'std_recall': result['std_recall'],
            'mean_f1': result['mean_f1'],
            'std_f1': result['std_f1'],
            'mean_auc': result['mean_auc'],
            'std_auc': result['std_auc']
        })

    df_summary = pd.DataFrame(summary_data)
    csv_file = output_dir / f"{dataset_name}_{target}_cv_summary.csv"
    df_summary.to_csv(csv_file, index=False)
    logger.info(f"保存汇总结果: {csv_file}")

    # 保存混淆矩阵
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    for result in all_results:
        cm = np.array(result['confusion_matrix'])
        cm_file = cm_dir / f"{dataset_name}_{target}_{result['model']}_confusion_matrix.csv"
        pd.DataFrame(cm).to_csv(cm_file, index=False)

    logger.info(f"保存混淆矩阵: {cm_dir}")

    # 保存特征重要性
    fi_dir = output_dir / "feature_importance"
    fi_dir.mkdir(parents=True, exist_ok=True)

    for result in all_results:
        if 'feature_importances' in result:
            fi_data = pd.DataFrame({
                'feature': feature_names,
                'importance': result['feature_importances']
            })
            fi_data = fi_data.sort_values('importance', ascending=False)
            fi_file = fi_dir / f"{dataset_name}_{target}_{result['model']}_feature_importance.csv"
            fi_data.to_csv(fi_file, index=False)

    logger.info(f"保存特征重要性: {fi_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="训练模型并执行 5 折交叉验证"
    )
    parser.add_argument(
        '--input_dir',
        type=Path,
        default=Path('outputs/features'),
        help='输入目录（包含 NPZ 文件）'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('outputs/model_results/cv_results'),
        help='输出目录'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='交叉验证折数'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='指定数据集名称（不含扩展名），如 US9624268_cleaned'
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
    logger.info("开始模型训练和交叉验证")
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"交叉验证折数: {args.n_folds}")

    # 获取所有 NPZ 文件
    if args.datasets:
        npz_files = [args.input_dir / f"{ds}.npz" for ds in args.datasets]
        npz_files = [f for f in npz_files if f.exists()]
    else:
        npz_files = sorted(args.input_dir.glob("*.npz"))

    if not npz_files:
        logger.error(f"在 {args.input_dir} 中未找到 NPZ 文件")
        return

    logger.info(f"找到 {len(npz_files)} 个 NPZ 文件")

    # 确定目标变量
    targets = []
    if 'BOTH' in args.targets:
        targets = ['SIF', 'SGF']
    else:
        targets = args.targets

    # 对每个数据集和目标进行训练
    for npz_file in npz_files:
        for target in targets:
            try:
                train_on_dataset(npz_file, target, args.output_dir, args.n_folds)
            except Exception as e:
                logger.error(
                    f"处理 {npz_file.name} 的 {target} 目标时出错: {e}",
                    exc_info=True
                )
                continue

    logger.info("\n所有模型训练完成！")
    logger.info(f"结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()

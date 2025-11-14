#!/usr/bin/env python3
"""
Phase 3: 二分类建模与验证

功能:
1. 将标签二值化（根据中位数）
2. 5-fold 分层交叉验证
3. 训练三种模型: Logistic Regression, Random Forest, XGBoost
4. 特征重要性分析
5. 模型迁移测试

用法:
    uv run python scripts/binary_classification.py \
        --input_dir data/processed/ \
        --features_dir outputs/features/ \
        --output_dir outputs/model_results/phase3_binary/ \
        --models lr rf xgb \
        --cv_folds 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("binary_classification.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_processed_data(csv_path: Path, npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    加载处理后的数据和特征

    Returns:
        X: 特征矩阵 (n_samples, n_samples)
        y_sif: SIF 标签（分钟） (n_samples,)
        y_sgf: SGF 标签（分钟） (n_samples,)
        feature_names: 特征名称列表
    """
    # 加载 NPZ（获取特征）
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    feature_names = data['feature_names'].tolist()
    ids = data['ids']

    # 加载 CSV（获取转换后的标签）
    df = pd.read_csv(csv_path)

    # 根据 IDs 匹配标签
    # 创建 ID 到行索引的映射（统一转换为字符串进行比较）
    id_to_idx = {str(id_val): idx for idx, id_val in enumerate(df['id'].values)}

    # 初始化标签数组
    y_sif = np.full(len(ids), -1, dtype=int)
    y_sgf = np.full(len(ids), -1, dtype=int)

    # 根据 IDs 填充标签（统一转换为字符串）
    for i, sample_id in enumerate(ids):
        sample_id_str = str(sample_id)
        if sample_id_str in id_to_idx:
            csv_idx = id_to_idx[sample_id_str]
            y_sif[i] = df.loc[csv_idx, 'SIF_class_min']
            y_sgf[i] = df.loc[csv_idx, 'SGF_class_min']

    return X, y_sif, y_sgf, feature_names


def binarize_labels(y: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, int]]:
    """
    根据中位数将标签二值化

    Args:
        y: 原始标签（分钟）

    Returns:
        y_binary: 二值标签（0=不稳定, 1=稳定）
        threshold: 中位数阈值
        stats: 统计信息
    """
    # 过滤缺失值 (-1)
    valid_mask = y != -1
    valid_y = y[valid_mask]

    if len(valid_y) == 0:
        logger.warning("所有标签都是缺失值，无法二值化")
        return np.array([]), -1.0, {}

    # 计算中位数
    threshold = np.median(valid_y)

    # 二值化: >= 中位数为稳定(1), < 中位数为不稳定(0)
    y_binary = np.full(len(y), -1, dtype=int)
    y_binary[valid_mask] = (y[valid_mask] >= threshold).astype(int)

    # 统计
    n_stable = np.sum(y_binary == 1)
    n_unstable = np.sum(y_binary == 0)
    n_missing = np.sum(y_binary == -1)

    stats = {
        "threshold": float(threshold),
        "n_stable": int(n_stable),
        "n_unstable": int(n_unstable),
        "n_missing": int(n_missing),
        "n_valid": int(n_stable + n_unstable),
    }

    return y_binary, threshold, stats


def get_model(model_name: str, use_gpu: bool = True) -> Any:
    """获取模型实例"""
    if model_name == "lr":
        return LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    elif model_name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    elif model_name == "xgb":
        if use_gpu:
            return XGBClassifier(
                n_estimators=100,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method='hist',
                device='cuda:0'  # 使用 cuda:0
            )
        else:
            return XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"未知模型: {model_name}")


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """
    交叉验证

    Returns:
        results: 包含所有fold的指标
    """
    # 过滤缺失值
    valid_mask = y != -1
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    if len(np.unique(y_valid)) < 2:
        logger.warning(f"类别数不足，跳过交叉验证")
        return {}

    # 检查每个类别的样本数
    unique, counts = np.unique(y_valid, return_counts=True)
    min_class_count = min(counts)

    # 调整 n_folds
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < n_folds:
        logger.warning(f"样本数不足，将 n_folds 从 {n_folds} 调整为 {actual_folds}")

    if actual_folds < 2:
        logger.warning(f"样本数过少，无法进行交叉验证")
        return {}

    # 分层交叉验证
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

    results = {
        "model": model_name,
        "n_folds": int(actual_folds),
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": [],
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_valid, y_valid)):
        X_train, X_test = X_valid[train_idx], X_valid[test_idx]
        y_train, y_test = y_valid[train_idx], y_valid[test_idx]

        # 训练模型
        model = get_model(model_name)
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        # 计算指标
        results["accuracy"].append(accuracy_score(y_test, y_pred))
        results["precision"].append(precision_score(y_test, y_pred, average='binary', zero_division=0))
        results["recall"].append(recall_score(y_test, y_pred, average='binary', zero_division=0))
        results["f1"].append(f1_score(y_test, y_pred, average='binary', zero_division=0))

        # AUC
        try:
            results["auc"].append(roc_auc_score(y_test, y_proba))
        except ValueError:
            results["auc"].append(0.0)

        logger.info(f"  Fold {fold+1}/{actual_folds} - Accuracy: {results['accuracy'][-1]:.4f}, F1: {results['f1'][-1]:.4f}")

    # 计算均值和标准差
    results["mean_accuracy"] = float(np.mean(results["accuracy"]))
    results["std_accuracy"] = float(np.std(results["accuracy"]))
    results["mean_precision"] = float(np.mean(results["precision"]))
    results["std_precision"] = float(np.std(results["precision"]))
    results["mean_recall"] = float(np.mean(results["recall"]))
    results["std_recall"] = float(np.std(results["recall"]))
    results["mean_f1"] = float(np.mean(results["f1"]))
    results["std_f1"] = float(np.std(results["f1"]))
    results["mean_auc"] = float(np.mean(results["auc"]))
    results["std_auc"] = float(np.std(results["auc"]))

    return results


def extract_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_name: str,
) -> pd.DataFrame:
    """
    提取特征重要性（RF 或 XGBoost）

    Returns:
        DataFrame with columns: feature, importance
    """
    valid_mask = y != -1
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    if model_name not in ["rf", "xgb"]:
        logger.warning(f"模型 {model_name} 不支持特征重要性分析")
        return pd.DataFrame()

    # 训练模型
    model = get_model(model_name)
    model.fit(X_valid, y_valid)

    # 提取重要性
    importances = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    df = df.sort_values("importance", ascending=False)

    return df


def transfer_learning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    train_name: str,
    test_name: str,
) -> Dict[str, Any]:
    """
    模型迁移: 在 train 数据上训练，在 test 数据上测试

    Returns:
        results: 迁移测试结果
    """
    # 过滤缺失值
    train_valid = y_train != -1
    test_valid = y_test != -1

    X_train_valid = X_train[train_valid]
    y_train_valid = y_train[train_valid]
    X_test_valid = X_test[test_valid]
    y_test_valid = y_test[test_valid]

    if len(X_train_valid) == 0 or len(X_test_valid) == 0:
        logger.warning(f"训练集或测试集为空，跳过迁移测试")
        return {}

    if len(np.unique(y_train_valid)) < 2 or len(np.unique(y_test_valid)) < 2:
        logger.warning(f"类别数不足，跳过迁移测试")
        return {}

    # 训练模型
    model = get_model(model_name)
    model.fit(X_train_valid, y_train_valid)

    # 测试
    y_pred = model.predict(X_test_valid)
    y_proba = model.predict_proba(X_test_valid)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    # 计算指标
    results = {
        "train_dataset": train_name,
        "test_dataset": test_name,
        "model": model_name,
        "n_train": len(X_train_valid),
        "n_test": len(X_test_valid),
        "accuracy": float(accuracy_score(y_test_valid, y_pred)),
        "precision": float(precision_score(y_test_valid, y_pred, average='binary', zero_division=0)),
        "recall": float(recall_score(y_test_valid, y_pred, average='binary', zero_division=0)),
        "f1": float(f1_score(y_test_valid, y_pred, average='binary', zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test_valid, y_pred).tolist(),
    }

    try:
        results["auc"] = float(roc_auc_score(y_test_valid, y_proba))
    except ValueError:
        results["auc"] = 0.0

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 3: 二分类建模与验证")
    parser.add_argument("--input_dir", type=str, required=True, help="处理后的 CSV 目录")
    parser.add_argument("--features_dir", type=str, required=True, help="特征 NPZ 目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--models", nargs="+", default=["lr", "rf", "xgb"], help="模型列表")
    parser.add_argument("--cv_folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--target", type=str, default="both", choices=["sif", "sgf", "both"], help="目标变量")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cv_results").mkdir(exist_ok=True)
    (output_dir / "feature_importance").mkdir(exist_ok=True)
    (output_dir / "transfer_results").mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("Phase 3: 二分类建模与验证")
    logger.info("=" * 80)

    # 查找所有数据集
    csv_files = sorted(input_dir.glob("*_processed.csv"))
    logger.info(f"找到 {len(csv_files)} 个数据集")

    # 加载所有数据集
    datasets = {}
    for csv_file in csv_files:
        dataset_name = csv_file.stem.replace("_processed", "")
        npz_file = features_dir / f"{csv_file.stem}.npz"

        if not npz_file.exists():
            logger.warning(f"特征文件不存在: {npz_file}, 跳过")
            continue

        logger.info(f"\n加载数据集: {dataset_name}")
        X, y_sif, y_sgf, feature_names = load_processed_data(csv_file, npz_file)

        # 二值化标签
        y_sif_binary, sif_threshold, sif_stats = binarize_labels(y_sif)
        y_sgf_binary, sgf_threshold, sgf_stats = binarize_labels(y_sgf)

        logger.info(f"  SIF 二值化: 阈值={sif_threshold:.1f}分钟, 稳定={sif_stats.get('n_stable', 0)}, 不稳定={sif_stats.get('n_unstable', 0)}")
        logger.info(f"  SGF 二值化: 阈值={sgf_threshold:.1f}分钟, 稳定={sgf_stats.get('n_stable', 0)}, 不稳定={sgf_stats.get('n_unstable', 0)}")

        datasets[dataset_name] = {
            "X": X,
            "y_sif": y_sif_binary,
            "y_sgf": y_sgf_binary,
            "feature_names": feature_names,
            "sif_threshold": sif_threshold,
            "sgf_threshold": sgf_threshold,
            "sif_stats": sif_stats,
            "sgf_stats": sgf_stats,
        }

    logger.info(f"\n成功加载 {len(datasets)} 个数据集")

    # ==========================================================================
    # 1. 交叉验证
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("任务 1: 交叉验证")
    logger.info("=" * 80)

    cv_results_all = []

    for dataset_name, data in datasets.items():
        logger.info(f"\n数据集: {dataset_name}")

        targets = []
        if args.target in ["sif", "both"]:
            targets.append(("SIF", data["y_sif"]))
        if args.target in ["sgf", "both"]:
            targets.append(("SGF", data["y_sgf"]))

        for target_name, y in targets:
            if len(y[y != -1]) == 0:
                logger.warning(f"  {target_name}: 无有效标签，跳过")
                continue

            logger.info(f"  目标: {target_name}")

            for model_name in args.models:
                logger.info(f"    模型: {model_name.upper()}")

                results = cross_validate_model(
                    data["X"],
                    y,
                    model_name,
                    n_folds=args.cv_folds,
                )

                if results:
                    results["dataset"] = dataset_name
                    results["target"] = target_name
                    cv_results_all.append(results)

                    # 保存结果
                    output_file = output_dir / "cv_results" / f"{dataset_name}_{target_name}_{model_name}_cv.json"
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2)

                    logger.info(f"      均值 F1: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
                    logger.info(f"      均值 AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")

    # 保存汇总
    summary_file = output_dir / "cv_results" / "cv_summary.json"
    with open(summary_file, "w") as f:
        json.dump(cv_results_all, f, indent=2)
    logger.info(f"\n交叉验证汇总保存到: {summary_file}")

    # ==========================================================================
    # 2. 特征重要性分析
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("任务 2: 特征重要性分析")
    logger.info("=" * 80)

    for dataset_name, data in datasets.items():
        logger.info(f"\n数据集: {dataset_name}")

        targets = []
        if args.target in ["sif", "both"]:
            targets.append(("SIF", data["y_sif"]))
        if args.target in ["sgf", "both"]:
            targets.append(("SGF", data["y_sgf"]))

        for target_name, y in targets:
            if len(y[y != -1]) == 0:
                continue

            logger.info(f"  目标: {target_name}")

            for model_name in ["rf", "xgb"]:
                if model_name not in args.models:
                    continue

                logger.info(f"    模型: {model_name.upper()}")

                df_importance = extract_feature_importance(
                    data["X"],
                    y,
                    data["feature_names"],
                    model_name,
                )

                if not df_importance.empty:
                    output_file = output_dir / "feature_importance" / f"{dataset_name}_{target_name}_{model_name}_importance.csv"
                    df_importance.to_csv(output_file, index=False)
                    logger.info(f"      Top 5 特征:")
                    for idx, row in df_importance.head(5).iterrows():
                        logger.info(f"        {row['feature']}: {row['importance']:.4f}")

    # ==========================================================================
    # 3. 模型迁移测试
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("任务 3: 模型迁移测试")
    logger.info("=" * 80)

    transfer_results_all = []
    dataset_names = list(datasets.keys())

    for i, train_name in enumerate(dataset_names):
        for j, test_name in enumerate(dataset_names):
            if i >= j:  # 跳过自身和重复对
                continue

            logger.info(f"\n训练: {train_name} → 测试: {test_name}")

            train_data = datasets[train_name]
            test_data = datasets[test_name]

            targets = []
            if args.target in ["sif", "both"]:
                targets.append(("SIF", train_data["y_sif"], test_data["y_sif"]))
            if args.target in ["sgf", "both"]:
                targets.append(("SGF", train_data["y_sgf"], test_data["y_sgf"]))

            for target_name, y_train, y_test in targets:
                logger.info(f"  目标: {target_name}")

                for model_name in args.models:
                    logger.info(f"    模型: {model_name.upper()}")

                    # 正向迁移
                    results_forward = transfer_learning(
                        train_data["X"],
                        y_train,
                        test_data["X"],
                        y_test,
                        model_name,
                        train_name,
                        test_name,
                    )

                    if results_forward:
                        results_forward["target"] = target_name
                        transfer_results_all.append(results_forward)

                        output_file = output_dir / "transfer_results" / f"{train_name}_to_{test_name}_{target_name}_{model_name}.json"
                        with open(output_file, "w") as f:
                            json.dump(results_forward, f, indent=2)

                        logger.info(f"      F1: {results_forward['f1']:.4f}, AUC: {results_forward['auc']:.4f}")

                    # 反向迁移
                    results_backward = transfer_learning(
                        test_data["X"],
                        y_test,
                        train_data["X"],
                        y_train,
                        model_name,
                        test_name,
                        train_name,
                    )

                    if results_backward:
                        results_backward["target"] = target_name
                        transfer_results_all.append(results_backward)

                        output_file = output_dir / "transfer_results" / f"{test_name}_to_{train_name}_{target_name}_{model_name}.json"
                        with open(output_file, "w") as f:
                            json.dump(results_backward, f, indent=2)

    # 保存迁移学习汇总
    summary_file = output_dir / "transfer_results" / "transfer_summary.json"
    with open(summary_file, "w") as f:
        json.dump(transfer_results_all, f, indent=2)
    logger.info(f"\n模型迁移汇总保存到: {summary_file}")

    logger.info("\n" + "=" * 80)
    logger.info("Phase 3 完成!")
    logger.info("=" * 80)
    logger.info(f"所有结果保存到: {output_dir}")


if __name__ == "__main__":
    main()

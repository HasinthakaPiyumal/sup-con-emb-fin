"""Evaluation metrics and reporting utilities."""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .config import set_seed
from .classifiers import build_centroids, predict_centroid, train_and_classify_knn
from .io_utils import load_saved_embeddings


def _report_method(
    name: str,
    fold_accuracies: List[float],
    fold_f1_scores: List[float],
    all_true: List[int],
    all_pred: List[int],
    class_names: List[str],
    wandb_key: str
) -> None:
    """
    Print and log evaluation metrics for a classification method.
    
    Args:
        name: Name of the method (for display).
        fold_accuracies: Accuracy for each fold.
        fold_f1_scores: Macro F1 for each fold.
        all_true: All ground truth labels.
        all_pred: All predicted labels.
        class_names: List of class names.
        wandb_key: Key prefix for wandb logging.
    """
    if not fold_accuracies:
        return
    
    # Print summary
    print(f"\n{'#' * 80}")
    print(f"{name}")
    print(f"{'#' * 80}")
    print(f"Avg Accuracy : {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Avg Macro F1 : {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
    
    # Print classification report
    print(classification_report(all_true, all_pred, target_names=class_names, digits=4))
    
    # Log to wandb
    report_dict = classification_report(
        all_true, all_pred, target_names=class_names, digits=4, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "class"})
    
    wandb.log({
        f"{wandb_key}_report": wandb.Table(dataframe=report_df),
        f"{wandb_key}_confusion": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_true,
            preds=all_pred,
            class_names=class_names
        )
    })


def evaluate_saved_embeddings_5fold(
    save_dir: str,
    class_names: List[str],
    num_folds: int = 5,
    seed: int = 42
) -> None:
    """
    Evaluate saved embeddings using 5-fold CV with centroid and KNN classifiers.
    
    This is Phase 2 evaluation: takes pre-computed embeddings and evaluates
    different classification strategies.
    
    Args:
        save_dir: Directory containing saved embeddings.
        class_names: List of class names for reporting.
        num_folds: Number of CV folds.
        seed: Random seed for reproducibility.
    """
    X, y = load_saved_embeddings(save_dir)
    
    if X is None:
        print(f"Phase 2 skipped: no embeddings in {save_dir}")
        return
    
    set_seed(seed)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    # Metrics storage
    acc_centroid, f1_centroid = [], []
    acc_knn, f1_knn = [], []
    all_true = []
    pred_centroid_all, pred_knn_all = [], []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Centroid classification
        centroids = build_centroids(torch.from_numpy(X_train).float(), y_train)
        pred_c = predict_centroid(torch.from_numpy(X_test).float(), centroids)
        acc_centroid.append(accuracy_score(y_test, pred_c))
        f1_centroid.append(f1_score(y_test, pred_c, average="macro"))
        all_true.extend(y_test)
        pred_centroid_all.extend(pred_c)
        
        # KNN classification
        pred_k = train_and_classify_knn(X_train, y_train, X_test, y_test)
        acc_knn.append(accuracy_score(y_test, pred_k))
        f1_knn.append(f1_score(y_test, pred_k, average="macro"))
        pred_knn_all.extend(pred_k)
        
        print(
            f"Phase 2 fold {fold}: "
            f"Centroid Acc {acc_centroid[-1]:.4f} F1 {f1_centroid[-1]:.4f} | "
            f"KNN Acc {acc_knn[-1]:.4f} F1 {f1_knn[-1]:.4f}"
        )
    
    # Report results
    _report_method(
        "PHASE 2 - CENTROID",
        acc_centroid, f1_centroid,
        all_true, pred_centroid_all,
        class_names, "phase2_centroid"
    )
    _report_method(
        "PHASE 2 - KNN",
        acc_knn, f1_knn,
        all_true, pred_knn_all,
        class_names, "phase2_knn"
    )

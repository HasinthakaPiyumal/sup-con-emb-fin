"""Main training pipeline with cross-validation."""

import os
from typing import List, Sequence

import numpy as np
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from .config import set_seed, clear_memory
from .data import prepare_fold_data
from .model import train_model, encode_in_batches
from .classifiers import build_centroids, predict_centroid
from .io_utils import save_fold_embeddings, save_all_embeddings
from .evaluation import evaluate_saved_embeddings_5fold


def init_wandb(model_name: str, config: dict) -> None:
    """Initialize Weights & Biases logging."""
    wandb.init(
        project="code-classification-super-cons-learn[AI Patterns]",
        name=f"{model_name.split('/')[-1]}-{config['max_seq_length']}",
        config=config,
    )


def run_5fold_cv(
    texts: Sequence[str],
    labels: Sequence[int],
    class_names: List[str],
    model_name: str = "nomic-ai/nomic-embed-text-v1",
    num_folds: int = 5,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 2e-5,
    warmup_steps: int = 10,
    max_pairs_per_class: int = 100,
    max_seq_length: int = 256,
    seed: int = 42,
    save_dir: str = "saved_test_embeddings",
    dense_dim: int = 8,
) -> None:
    """
    Run two-phase 5-fold cross-validation.
    
    Phase 1: Train embedding model and save test embeddings for each fold.
    Phase 2: Evaluate embeddings with centroid and KNN classifiers.
    
    Args:
        texts: Input text samples.
        labels: Encoded integer labels.
        class_names: Original class names for reporting.
        model_name: HuggingFace model identifier.
        num_folds: Number of CV folds.
        epochs: Training epochs per fold.
        batch_size: Training batch size.
        lr: Learning rate.
        warmup_steps: LR warmup steps.
        max_pairs_per_class: Maximum contrastive pairs per class.
        max_seq_length: Maximum token sequence length.
        seed: Random seed.
        save_dir: Directory to save embeddings.
        dense_dim: Output dimension for projection head.
    """
    set_seed(seed)
    
    # Build config for logging
    config = {
        "model_name": model_name,
        "num_folds": num_folds,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "max_pairs_per_class": max_pairs_per_class,
        "max_seq_length": max_seq_length,
        "seed": seed,
        "dense_dim": dense_dim,
    }
    init_wandb(model_name, config)
    
    # Convert to arrays
    texts = np.array(list(texts), dtype=object)
    labels = np.array(list(labels), dtype=int)
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    os.makedirs(save_dir, exist_ok=True)
    
    all_fold_data = []
    
    # Phase 1: Train and generate embeddings
    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), start=1):
        print(f"\n{'=' * 80}")
        print(f"PHASE 1 FOLD {fold}/{num_folds}")
        print(f"{'=' * 80}")
        
        clear_memory()
        
        # Prepare fold data
        prep = prepare_fold_data(texts, labels, train_idx, test_idx, max_pairs_per_class)
        if prep[0] is None:
            print("Skipping fold: not enough pairs.")
            continue
        
        train_examples, X_train, y_train, X_test, y_test = prep
        
        # Train model
        model = train_model(
            model_name, max_seq_length, train_examples,
            batch_size, epochs, warmup_steps, lr,
            dense_dim=dense_dim
        )
        
        clear_memory()
        model.eval()
        
        # Generate embeddings
        train_emb = encode_in_batches(model, X_train, batch_size=32)
        test_emb = encode_in_batches(model, X_test, batch_size=32)
        
        # Phase 1 evaluation with centroids
        centroids = build_centroids(train_emb, y_train)
        pred_c = predict_centroid(test_emb, centroids)
        acc_c = accuracy_score(y_test, pred_c)
        f1_c = f1_score(y_test, pred_c, average="macro")
        
        print(f"Phase 1 fold {fold} centroid: Acc {acc_c:.4f}, Macro F1 {f1_c:.4f}")
        wandb.log({
            f"phase1_fold{fold}_centroid_acc": acc_c,
            f"phase1_fold{fold}_centroid_f1": f1_c
        })
        
        # Save fold embeddings
        all_fold_data.append(save_fold_embeddings(fold, y_test, test_emb))
        
        # Cleanup
        del train_emb, centroids, pred_c
        del model, train_examples, test_emb, X_train, y_train, X_test, y_test
        clear_memory()
    
    # Save all embeddings
    if all_fold_data:
        out_path = save_all_embeddings(all_fold_data, save_dir)
        print(f"Saved embeddings -> {out_path}")
    
    # Phase 2: Evaluate with different classifiers
    evaluate_saved_embeddings_5fold(save_dir, class_names, num_folds=num_folds, seed=seed)
    
    wandb.finish()

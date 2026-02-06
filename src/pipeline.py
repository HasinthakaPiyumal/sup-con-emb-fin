"""Main training pipeline with cross-validation."""

import os
from typing import List, Sequence, Union

import numpy as np
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from .config import set_seed, clear_memory
from .data import prepare_fold_data, LossType
from .model import train_model, encode_in_batches
from .classifiers import build_centroids, predict_centroid
from .io_utils import save_fold_embeddings, save_all_embeddings
from .evaluation import evaluate_saved_embeddings_5fold


def build_wandb_tags(config: dict) -> List[str]:
    """Build tags for wandb run based on configuration."""
    tags = []
    
    # Model tag
    model_short = config["model_name"].split("/")[-1]
    tags.append(f"model:{model_short}")
    
    # Loss function tag
    loss_type = config.get("loss_type", "mnrl")
    tags.append(f"loss:{loss_type}")
    
    # Hard negative mining tag
    use_hn = config.get("use_hard_negatives", False)
    if use_hn:
        num_hn = config.get("num_hard_negatives", 3)
        tags.append("hard-negatives")
        tags.append(f"hn:{num_hn}")
    else:
        tags.append("no-hard-negatives")
    
    # Training config tags
    tags.append(f"epochs:{config.get('epochs', 1)}")
    tags.append(f"bs:{config.get('batch_size', 4)}")
    tags.append(f"seq:{config.get('max_seq_length', 256)}")
    
    # Folds tag
    tags.append(f"folds:{config.get('num_folds', 5)}")
    
    return tags


def init_wandb(model_name: str, config: dict) -> None:
    """Initialize Weights & Biases logging with tags."""
    tags = build_wandb_tags(config)
    
    # Build run name: model-loss-hn_status-seq_length
    loss_type = config.get("loss_type", "mnrl")
    hn_status = "hn" if config.get("use_hard_negatives", False) else "no-hn"
    run_name = f"{model_name.split('/')[-1]}-{loss_type}-{hn_status}-{config['max_seq_length']}"
    
    wandb.init(
        project="code-classification-super-cons-learn[AI Patterns]",
        name=run_name,
        config=config,
        tags=tags,
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
    # Loss function settings
    loss_type: Union[LossType, str] = LossType.MNRL,
    loss_margin: float = 0.5,
    # Hard negative mining settings
    use_hard_negatives: bool = False,
    num_hard_negatives: int = 3,
    hn_base_model: str = "all-MiniLM-L6-v2",
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
        loss_type: Loss function type ("contrastive", "mnrl", "triplet" or LossType enum).
        loss_margin: Margin for contrastive/triplet loss.
        use_hard_negatives: Whether to use hard negative mining.
        num_hard_negatives: Number of hard negatives per sample.
        hn_base_model: Base model for hard negative mining embeddings.
    """
    # Convert string loss_type to enum if needed
    if isinstance(loss_type, str):
        loss_type = LossType(loss_type)
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
        "loss_type": loss_type.value,
        "loss_margin": loss_margin,
        "use_hard_negatives": use_hard_negatives,
        "num_hard_negatives": num_hard_negatives,
        "hn_base_model": hn_base_model,
    }
    init_wandb(model_name, config)
    
    print(f"\n{'=' * 80}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"  Loss function: {loss_type.value}")
    if loss_type in (LossType.CONTRASTIVE, LossType.TRIPLET):
        print(f"  Loss margin: {loss_margin}")
    if use_hard_negatives:
        print(f"  Hard negative mining: {num_hard_negatives} negatives using {hn_base_model}")
    else:
        print(f"  Hard negative mining: disabled")
    
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
        
        # Prepare fold data with optional hard negative mining
        prep = prepare_fold_data(
            texts, labels, train_idx, test_idx, max_pairs_per_class,
            loss_type=loss_type,
            use_hard_negatives=use_hard_negatives,
            num_hard_negatives=num_hard_negatives,
            hn_base_model=hn_base_model,
        )
        if prep[0] is None:
            print("Skipping fold: not enough pairs.")
            continue
        
        train_examples, X_train, y_train, X_test, y_test = prep
        print(f"  Generated {len(train_examples)} training examples")
        
        # Train model
        model = train_model(
            model_name, max_seq_length, train_examples,
            batch_size, epochs, warmup_steps, lr,
            dense_dim=dense_dim,
            loss_type=loss_type,
            loss_margin=loss_margin,
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

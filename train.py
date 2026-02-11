#!/usr/bin/env python3
"""
Two-phase 5-fold CV training script.

Phase 1: Train embedding model with configurable loss, save test embeddings.
Phase 2: Evaluate embeddings with centroid and KNN classifiers.

Supported loss functions:
- "contrastive": ContrastiveLoss - pairs with similarity labels (0/1)
- "mnrl": MultipleNegativesRankingLoss - (anchor, positive, [negatives...])
- "triplet": TripletLoss - (anchor, positive, negative) triplets
"""

import os

from src.data import load_and_preprocess_data, LossType
from src.pipeline import run_5fold_cv, run_5fold_cv_no_finetuning


# =============================================================================
# Configuration
# =============================================================================

# If True: encode all samples with raw model (no training), then 5-fold CV with
# KNN and Centroid on embeddings only. Fine-tuning pipeline is skipped.
RUN_WITHOUT_FINETUNING = False

# Loss function: "contrastive", "mnrl", or "triplet"
LOSS_TYPE = "contrastive"  # Options: LossType.CONTRASTIVE, LossType.MNRL, LossType.TRIPLET
LOSS_MARGIN = 0.5  # Margin for contrastive/triplet loss

# Hard negative mining
USE_HARD_NEGATIVES = True
NUM_HARD_NEGATIVES = 10
HN_BASE_MODEL = "google-bert/bert-base-uncased"  # Fast model for mining

# Model and training
# MODEL_NAME = "BAAI/bge-reasoner-embed-qwen3-8b-0923"
MODEL_NAME = "google-bert/bert-base-uncased"
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WARMUP_STEPS = 10
MAX_PAIRS_PER_CLASS = 80
MAX_SEQ_LENGTH = 768
SEED = 42
DENSE_DIM = 8


def main():
    """Main entry point for training."""
    dataset_path = "./data/feb-10-2026-community-descriptions-concated.csv"
    
    try:
        dataset, label_encoder = load_and_preprocess_data(dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Suppress wandb output
    os.environ["WANDB_SILENT"] = "true"
    
    if RUN_WITHOUT_FINETUNING:
        run_5fold_cv_no_finetuning(
            texts=dataset["code_summary"],
            labels=dataset["label_enc"],
            class_names=list(label_encoder.classes_),
            model_name=MODEL_NAME,
            num_folds=5,
            batch_size=BATCH_SIZE,
            max_seq_length=MAX_SEQ_LENGTH,
            seed=SEED,
        )
        return
    
    # Run training with configurable loss and hard negative mining
    run_5fold_cv(
        texts=dataset["code_summary"],
        labels=dataset["label_enc"],
        class_names=list(label_encoder.classes_),
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_pairs_per_class=MAX_PAIRS_PER_CLASS,
        max_seq_length=MAX_SEQ_LENGTH,
        seed=SEED,
        dense_dim=DENSE_DIM,
        # Loss function settings
        loss_type=LOSS_TYPE,
        loss_margin=LOSS_MARGIN,
        # Hard negative mining settings
        use_hard_negatives=USE_HARD_NEGATIVES,
        num_hard_negatives=NUM_HARD_NEGATIVES,
        hn_base_model=HN_BASE_MODEL,
    )


if __name__ == "__main__":
    main()

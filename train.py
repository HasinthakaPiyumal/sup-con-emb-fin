#!/usr/bin/env python3
"""
Two-phase 5-fold CV training script.

Phase 1: Train embedding model with contrastive loss, save test embeddings.
Phase 2: Evaluate embeddings with centroid and KNN classifiers.
"""

import os

from src.data import load_and_preprocess_data
from src.pipeline import run_5fold_cv


def main():
    """Main entry point for training."""
    dataset_path = "./data/labeled_verified_data.csv"
    
    try:
        dataset, label_encoder = load_and_preprocess_data(dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Suppress wandb output
    os.environ["WANDB_SILENT"] = "true"
    
    # Run training
    run_5fold_cv(
        texts=dataset["code_summary"],
        labels=dataset["label_enc"],
        class_names=list(label_encoder.classes_),
        model_name="google-bert/bert-base-uncased",
        epochs=3,
        batch_size=32,
        lr=2e-5,
        warmup_steps=10,
        max_pairs_per_class=1000,
        max_seq_length=768,
        seed=42,
        dense_dim=8,
    )


if __name__ == "__main__":
    main()

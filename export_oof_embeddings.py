#!/usr/bin/env python3
"""
Standalone script to generate and export Out-Of-Fold (OOF) embeddings for external model training.

Format of saved CSV:
file, label, code_summary, dim_1, dim_2, ..., dim_D, fold

Usage:
    python export_oof_embeddings.py --output_path saved_test_embeddings/oof_embeddings.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from src.data import load_and_preprocess_data
from src.config import set_seed, clear_memory
from src.model import encode_in_batches
from src.io_utils import save_fold_embeddings, save_all_embeddings


def export_oof_embeddings(
    dataset_path: str = "./data/feb-10-2026-community-descriptions-concated.csv",
    model_name: str = "BAAI/bge-code-v1",
    output_dir: str = "saved_test_embeddings",
    min_samples_per_label: int = 120,
    batch_size: int = 32,
    max_seq_length: int = 768,
    seed: int = 42,
    num_folds: int = 5,
) -> str:
    """
    Generate and save Out-Of-Fold (OOF) embeddings in the exact requested format:
    file, label, code_summary, dim_1, dim_2, ..., dim_D, fold
    """
    set_seed(seed)

    print(f"Loading dataset from: {dataset_path}")
    dataset, label_encoder = load_and_preprocess_data(
        dataset_path, min_samples_per_label=min_samples_per_label
    )

    texts = np.array(dataset["code_summary"].tolist(), dtype=object)
    labels = np.array(dataset["label_enc"].tolist(), dtype=int)
    files = np.array(dataset["file"].tolist(), dtype=object) if "file" in dataset.columns else None

    print(f"Loaded {len(texts)} samples across {len(label_encoder.classes_)} classes.")
    if files is not None:
        print(f"Grouped CV by {len(np.unique(files))} unique code files.")

    print(f"Loading embedding model: {model_name}...")
    clear_memory()
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    model.eval()

    print("Encoding texts in batches...")
    embeddings = encode_in_batches(model, list(texts), batch_size=batch_size)
    del model
    clear_memory()

    # 5-fold Stratified Group split by unique files to ensure zero data leakage
    if files is not None:
        sgkf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        split_iter = sgkf.split(embeddings, labels, groups=files)
    else:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        split_iter = skf.split(embeddings, labels)

    all_fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
        X_test = embeddings[test_idx]
        y_test = labels[test_idx].tolist()
        test_files = files[test_idx].tolist() if files is not None else None
        test_descriptions = texts[test_idx].tolist()

        fold_df = save_fold_embeddings(
            fold=fold,
            y_test=y_test,
            test_emb=X_test,
            files=test_files,
            descriptions=test_descriptions,
        )
        all_fold_rows.append(fold_df)
        print(f"Fold {fold}/{num_folds} OOF samples collected: {len(fold_df)}")

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    saved_path = save_all_embeddings(all_fold_rows, output_dir)
    
    oof_df = pd.read_csv(saved_path)
    print(f"\nSuccessfully exported OOF embeddings to: {saved_path}")
    print(f"Total rows: {len(oof_df)}")
    print(f"Columns: {list(oof_df.columns[:5])} ... {list(oof_df.columns[-3:])}")
    
    return saved_path


def main():
    parser = argparse.ArgumentParser(description="Export Out-Of-Fold Embeddings to CSV")
    parser.add_argument("--dataset_path", type=str, default="./data/feb-10-2026-community-descriptions-concated.csv")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-code-v1")
    parser.add_argument("--output_dir", type=str, default="saved_test_embeddings")
    parser.add_argument("--min_samples_per_label", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    export_oof_embeddings(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        min_samples_per_label=args.min_samples_per_label,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

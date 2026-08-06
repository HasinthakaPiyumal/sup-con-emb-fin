"""I/O utilities for saving and loading embeddings."""

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import wandb


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor or array to numpy float32 array."""
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def save_fold_embeddings(
    fold: int,
    y_test: List[int],
    test_emb: Union[torch.Tensor, np.ndarray],
    files: Optional[Sequence[str]] = None,
    descriptions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Create a DataFrame with fold metadata and out-of-fold embeddings.

    Args:
        fold: Fold number.
        y_test: Test labels.
        test_emb: Test embeddings.
        files: Optional file identifiers (one per test sample). Added as column "file".
        descriptions: Optional descriptions (e.g. code_summary) per test sample.
            Added as column "code_summary".

    Returns:
        DataFrame with columns: file, label, code_summary, dim_1..dim_D, fold.
    """
    emb = _to_numpy(test_emb)

    data_dict = {}
    if files is not None:
        data_dict["file"] = list(files)
    
    data_dict["label"] = list(y_test)
    
    if descriptions is not None:
        data_dict["code_summary"] = list(descriptions)
    
    # 1-indexed dim_1, dim_2, ..., dim_D
    for i in range(emb.shape[1]):
        data_dict[f"dim_{i + 1}"] = emb[:, i]
        
    data_dict["fold"] = fold

    return pd.DataFrame(data_dict)


def load_saved_embeddings(
    save_dir: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load previously saved embeddings from disk.
    
    Args:
        save_dir: Directory containing saved embeddings.
    
    Returns:
        Tuple of (embeddings, labels, files) or (None, None, None) if not found.
    """
    path = os.path.join(save_dir, "oof_embeddings.csv")
    if not os.path.exists(path):
        path = os.path.join(save_dir, "all_folds_test_embeddings.csv")
    
    if not os.path.exists(path):
        return None, None, None
    
    df = pd.read_csv(path)
    
    # Extract embedding columns (supports both dim_ and emb_ prefixes)
    emb_cols = [c for c in df.columns if c.startswith("dim_") or c.startswith("emb_")]
    embeddings = df[emb_cols].values.astype(np.float32)
    labels = df["label"].values.astype(int)
    files = df["file"].values if "file" in df.columns else None
    
    return embeddings, labels, files


def save_all_embeddings(
    all_rows: List[pd.DataFrame],
    save_dir: str
) -> str:
    """
    Concatenate and save out-of-fold embeddings to disk.
    
    Args:
        all_rows: List of DataFrames from each fold.
        save_dir: Directory to save embeddings.
    
    Returns:
        Path to saved file.
    """
    if not all_rows:
        return ""
    
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "oof_embeddings.csv")
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out_path, index=False)
    
    # Save a copy as all_folds_test_embeddings.csv for backward compatibility
    legacy_path = os.path.join(save_dir, "all_folds_test_embeddings.csv")
    combined.to_csv(legacy_path, index=False)

    # Upload artifact to WandB if run is active
    if wandb.run is not None:
        try:
            artifact = wandb.Artifact(
                name="oof_embeddings",
                type="dataset",
                description="Out-of-fold embeddings generated from 5-fold Stratified Group CV",
            )
            artifact.add_file(out_path)
            wandb.log_artifact(artifact)
            print(f"Logged OOF embeddings artifact to WandB: oof_embeddings")
        except Exception as e:
            print(f"Warning: Failed to log WandB artifact: {e}")
    
    return out_path

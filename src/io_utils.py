"""I/O utilities for saving and loading embeddings."""

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch


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
    Create a DataFrame with fold metadata and embeddings.

    Args:
        fold: Fold number.
        y_test: Test labels.
        test_emb: Test embeddings.
        files: Optional file identifiers (one per test sample). When provided,
            added as column "file".
        descriptions: Optional descriptions (e.g. code_summary) per test sample.
            When provided, added as column "description".

    Returns:
        DataFrame with fold, label, optional file/description, and embedding columns.
    """
    emb = _to_numpy(test_emb)

    # Create metadata columns
    meta_dict = {"fold": fold, "label": y_test}
    if files is not None:
        meta_dict["file"] = list(files)
    if descriptions is not None:
        meta_dict["description"] = list(descriptions)
    meta = pd.DataFrame(meta_dict)

    # Create embedding columns
    emb_columns = [f"emb_{i}" for i in range(emb.shape[1])]
    features = pd.DataFrame(emb, columns=emb_columns)

    return pd.concat([meta, features], axis=1)


def load_saved_embeddings(
    save_dir: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load previously saved embeddings from disk.
    
    Args:
        save_dir: Directory containing saved embeddings.
    
    Returns:
        Tuple of (embeddings, labels) or (None, None) if not found.
    """
    path = os.path.join(save_dir, "all_folds_test_embeddings.csv")
    
    if not os.path.exists(path):
        return None, None
    
    df = pd.read_csv(path)
    
    # Extract embedding columns
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    embeddings = df[emb_cols].values.astype(np.float32)
    labels = df["label"].values.astype(int)
    
    return embeddings, labels


def save_all_embeddings(
    all_rows: List[pd.DataFrame],
    save_dir: str
) -> str:
    """
    Concatenate and save all fold embeddings to disk.
    
    Args:
        all_rows: List of DataFrames from each fold.
        save_dir: Directory to save embeddings.
    
    Returns:
        Path to saved file.
    """
    if not all_rows:
        return ""
    
    out_path = os.path.join(save_dir, "all_folds_test_embeddings.csv")
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out_path, index=False)
    
    return out_path

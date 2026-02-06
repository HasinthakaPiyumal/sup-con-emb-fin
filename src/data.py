"""Data loading, preprocessing, and pair generation for contrastive learning."""

import os
import random
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import InputExample


def load_and_preprocess_data(
    filepath: str,
    min_samples_per_label: int = 20
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Load dataset and filter labels with insufficient samples.
    
    Args:
        filepath: Path to the CSV file containing 'label' column.
        min_samples_per_label: Minimum samples required per label.
    
    Returns:
        Tuple of (preprocessed DataFrame, fitted LabelEncoder).
    """
    path = filepath if os.path.exists(filepath) else os.path.basename(filepath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(path)
    
    # Filter labels with enough samples
    valid_labels = df["label"].value_counts()
    valid_labels = valid_labels[valid_labels >= min_samples_per_label].index
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)
    
    # Encode labels
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    
    return df, le


def build_positive_pairs(
    texts: List[str],
    labels: List[int],
    max_pairs_per_class: Optional[int] = None
) -> List[InputExample]:
    """
    Build positive pairs for contrastive learning.
    
    Creates pairs of texts that share the same label for use with
    Multiple Negatives Ranking Loss.
    
    Args:
        texts: List of text samples.
        labels: List of corresponding labels.
        max_pairs_per_class: Maximum pairs to generate per class (None = all).
    
    Returns:
        List of InputExample objects for training.
    """
    # Group indices by label
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)
    
    examples = []
    for label, indices in label_to_indices.items():
        if len(indices) < 2:
            continue
        
        # Generate all possible pairs within the class
        pairs = [
            (indices[i], indices[j])
            for i in range(len(indices))
            for j in range(i + 1, len(indices))
        ]
        
        # Sample if too many pairs
        if max_pairs_per_class and len(pairs) > max_pairs_per_class:
            pairs = random.sample(pairs, max_pairs_per_class)
        
        # Create InputExamples
        for a, b in pairs:
            examples.append(InputExample(texts=[str(texts[a]), str(texts[b])]))
    
    random.shuffle(examples)
    return examples


def prepare_fold_data(
    texts: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    max_pairs_per_class: int
) -> Tuple[Optional[List[InputExample]], Optional[List[str]], Optional[List[int]], Optional[List[str]], Optional[List[int]]]:
    """
    Prepare training and test data for a single fold.
    
    Args:
        texts: Array of all texts.
        labels: Array of all labels.
        train_idx: Indices for training set.
        test_idx: Indices for test set.
        max_pairs_per_class: Maximum pairs per class for training.
    
    Returns:
        Tuple of (train_examples, X_train, y_train, X_test, y_test).
        Returns (None, None, None, None, None) if insufficient pairs.
    """
    X_train = texts[train_idx].tolist()
    y_train = labels[train_idx].tolist()
    X_test = texts[test_idx].tolist()
    y_test = labels[test_idx].tolist()
    
    train_examples = build_positive_pairs(
        X_train, y_train, max_pairs_per_class=max_pairs_per_class
    )
    
    if len(train_examples) < 4:
        return None, None, None, None, None
    
    return train_examples, X_train, y_train, X_test, y_test

"""Data loading, preprocessing, and pair generation for contrastive learning."""

import os
import random
from typing import List, Tuple, Optional, Dict, Literal
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import InputExample, SentenceTransformer
import torch


class LossType(str, Enum):
    """Supported loss functions for contrastive learning."""
    CONTRASTIVE = "contrastive"  # ContrastiveLoss: pairs with similarity labels
    MNRL = "mnrl"  # MultipleNegativesRankingLoss: (anchor, positive, [negatives...])
    TRIPLET = "triplet"  # TripletLoss: (anchor, positive, negative)


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


def mine_hard_negatives(
    texts: List[str],
    labels: List[int],
    embeddings: np.ndarray,
    num_hard_negatives: int = 3,
) -> Dict[int, List[int]]:
    """
    Mine hard negatives for each sample based on embedding similarity.
    
    Hard negatives are samples from different classes that are closest
    in embedding space to the anchor.
    
    Args:
        texts: List of text samples.
        labels: List of corresponding labels.
        embeddings: Pre-computed embeddings for all samples.
        num_hard_negatives: Number of hard negatives to mine per sample.
    
    Returns:
        Dictionary mapping sample index to list of hard negative indices.
    """
    n_samples = len(texts)
    labels_arr = np.array(labels)
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized_emb = embeddings / norms
    
    # Compute similarity matrix
    similarity_matrix = normalized_emb @ normalized_emb.T
    
    hard_negatives = {}
    
    for idx in range(n_samples):
        anchor_label = labels_arr[idx]
        
        # Get indices of samples with different labels
        different_class_mask = labels_arr != anchor_label
        different_class_indices = np.where(different_class_mask)[0]
        
        if len(different_class_indices) == 0:
            hard_negatives[idx] = []
            continue
        
        # Get similarities to samples from different classes
        sims = similarity_matrix[idx, different_class_indices]
        
        # Get top-k most similar (hardest) negatives
        k = min(num_hard_negatives, len(different_class_indices))
        top_k_local_indices = np.argsort(sims)[-k:][::-1]
        
        hard_negatives[idx] = different_class_indices[top_k_local_indices].tolist()
    
    return hard_negatives


def build_pairs_with_hard_negatives(
    texts: List[str],
    labels: List[int],
    embeddings: np.ndarray,
    max_pairs_per_class: Optional[int] = None,
    num_hard_negatives: int = 3,
) -> List[InputExample]:
    """
    Build training pairs with hard negative mining for contrastive learning.
    
    Creates triplets/tuples of (anchor, positive, hard_neg1, hard_neg2, ...)
    for use with MultipleNegativesRankingLoss.
    
    Args:
        texts: List of text samples.
        labels: List of corresponding labels.
        embeddings: Pre-computed embeddings for all samples.
        max_pairs_per_class: Maximum pairs to generate per class (None = all).
        num_hard_negatives: Number of hard negatives per sample.
    
    Returns:
        List of InputExample objects for training.
    """
    # Mine hard negatives for each sample
    hard_negatives = mine_hard_negatives(texts, labels, embeddings, num_hard_negatives)
    
    # Group indices by label
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)
    
    examples = []
    for label, indices in label_to_indices.items():
        if len(indices) < 2:
            continue
        
        # Generate pairs within the class
        pairs = [
            (indices[i], indices[j])
            for i in range(len(indices))
            for j in range(i + 1, len(indices))
        ]
        
        # Sample if too many pairs
        if max_pairs_per_class and len(pairs) > max_pairs_per_class:
            pairs = random.sample(pairs, max_pairs_per_class)
        
        # Create InputExamples with hard negatives
        for anchor_idx, positive_idx in pairs:
            anchor_text = str(texts[anchor_idx])
            positive_text = str(texts[positive_idx])
            
            # Get hard negatives for the anchor
            neg_indices = hard_negatives.get(anchor_idx, [])
            negative_texts = [str(texts[neg_idx]) for neg_idx in neg_indices]
            
            if negative_texts:
                # Format: [anchor, positive, neg1, neg2, ...]
                all_texts = [anchor_text, positive_text] + negative_texts
            else:
                # Fall back to just positive pair
                all_texts = [anchor_text, positive_text]
            
            examples.append(InputExample(texts=all_texts))
    
    random.shuffle(examples)
    return examples


def get_base_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generate embeddings using a base model for hard negative mining.
    
    Args:
        texts: List of text samples.
        model_name: Base model for generating initial embeddings.
        batch_size: Batch size for encoding.
    
    Returns:
        Numpy array of embeddings.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return embeddings


# =============================================================================
# Loss-specific data generation functions
# =============================================================================

def build_contrastive_pairs(
    texts: List[str],
    labels: List[int],
    embeddings: Optional[np.ndarray] = None,
    max_pairs_per_class: Optional[int] = None,
    num_hard_negatives: int = 3,
    use_hard_negatives: bool = False,
) -> List[InputExample]:
    """
    Build pairs for ContrastiveLoss with similarity labels.
    
    ContrastiveLoss expects pairs with labels:
    - label=1.0 for similar/same-class pairs
    - label=0.0 for dissimilar/different-class pairs
    
    Args:
        texts: List of text samples.
        labels: List of corresponding labels.
        embeddings: Pre-computed embeddings (for hard negative mining).
        max_pairs_per_class: Maximum pairs to generate per class.
        num_hard_negatives: Number of hard negatives per anchor (if enabled).
        use_hard_negatives: Whether to use hard negative mining for negatives.
    
    Returns:
        List of InputExample objects with label field set.
    """
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)
    
    examples = []
    
    # Generate positive pairs (same class, label=1.0)
    for label, indices in label_to_indices.items():
        if len(indices) < 2:
            continue
        
        pairs = [
            (indices[i], indices[j])
            for i in range(len(indices))
            for j in range(i + 1, len(indices))
        ]
        
        if max_pairs_per_class and len(pairs) > max_pairs_per_class:
            pairs = random.sample(pairs, max_pairs_per_class)
        
        for a, b in pairs:
            examples.append(InputExample(
                texts=[str(texts[a]), str(texts[b])],
                label=1.0  # Similar pair
            ))
    
    # Generate negative pairs (different class, label=0.0)
    if use_hard_negatives and embeddings is not None:
        # Use hard negatives
        hard_negs = mine_hard_negatives(texts, labels, embeddings, num_hard_negatives)
        for anchor_idx, neg_indices in hard_negs.items():
            for neg_idx in neg_indices:
                examples.append(InputExample(
                    texts=[str(texts[anchor_idx]), str(texts[neg_idx])],
                    label=0.0  # Dissimilar pair
                ))
    else:
        # Random negatives - sample similar number as positives
        all_indices = list(range(len(texts)))
        labels_arr = np.array(labels)
        num_neg_pairs = len(examples)  # Match number of positive pairs
        
        neg_count = 0
        attempts = 0
        max_attempts = num_neg_pairs * 10
        
        while neg_count < num_neg_pairs and attempts < max_attempts:
            i, j = random.sample(all_indices, 2)
            if labels_arr[i] != labels_arr[j]:
                examples.append(InputExample(
                    texts=[str(texts[i]), str(texts[j])],
                    label=0.0  # Dissimilar pair
                ))
                neg_count += 1
            attempts += 1
    
    random.shuffle(examples)
    return examples


def build_triplets(
    texts: List[str],
    labels: List[int],
    embeddings: Optional[np.ndarray] = None,
    max_pairs_per_class: Optional[int] = None,
    num_hard_negatives: int = 1,
    use_hard_negatives: bool = False,
) -> List[InputExample]:
    """
    Build triplets for TripletLoss.
    
    TripletLoss expects (anchor, positive, negative) triplets.
    
    Args:
        texts: List of text samples.
        labels: List of corresponding labels.
        embeddings: Pre-computed embeddings (for hard negative mining).
        max_pairs_per_class: Maximum triplets to generate per class.
        num_hard_negatives: Number of hard negatives per anchor (if enabled).
        use_hard_negatives: Whether to use hard negative mining.
    
    Returns:
        List of InputExample objects with 3 texts each.
    """
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)
    
    labels_arr = np.array(labels)
    all_indices = list(range(len(texts)))
    
    # Get hard negatives if enabled
    hard_negs = None
    if use_hard_negatives and embeddings is not None:
        hard_negs = mine_hard_negatives(texts, labels, embeddings, num_hard_negatives)
    
    examples = []
    
    for label, indices in label_to_indices.items():
        if len(indices) < 2:
            continue
        
        # Generate anchor-positive pairs within the class
        pairs = [
            (indices[i], indices[j])
            for i in range(len(indices))
            for j in range(i + 1, len(indices))
        ]
        
        if max_pairs_per_class and len(pairs) > max_pairs_per_class:
            pairs = random.sample(pairs, max_pairs_per_class)
        
        # For each anchor-positive pair, add a negative
        for anchor_idx, positive_idx in pairs:
            if hard_negs and anchor_idx in hard_negs and hard_negs[anchor_idx]:
                # Use hard negative
                neg_idx = hard_negs[anchor_idx][0]
            else:
                # Random negative from different class
                different_class_indices = [i for i in all_indices if labels_arr[i] != label]
                if not different_class_indices:
                    continue
                neg_idx = random.choice(different_class_indices)
            
            examples.append(InputExample(texts=[
                str(texts[anchor_idx]),
                str(texts[positive_idx]),
                str(texts[neg_idx])
            ]))
    
    random.shuffle(examples)
    return examples


def build_mnrl_examples(
    texts: List[str],
    labels: List[int],
    embeddings: Optional[np.ndarray] = None,
    max_pairs_per_class: Optional[int] = None,
    num_hard_negatives: int = 3,
    use_hard_negatives: bool = False,
) -> List[InputExample]:
    """
    Build examples for MultipleNegativesRankingLoss.
    
    MNRL expects (anchor, positive, [negative1, negative2, ...]) tuples.
    In-batch negatives are used automatically, explicit negatives are optional.
    
    Args:
        texts: List of text samples.
        labels: List of corresponding labels.
        embeddings: Pre-computed embeddings (for hard negative mining).
        max_pairs_per_class: Maximum pairs to generate per class.
        num_hard_negatives: Number of explicit hard negatives per sample.
        use_hard_negatives: Whether to include explicit hard negatives.
    
    Returns:
        List of InputExample objects.
    """
    if use_hard_negatives and embeddings is not None:
        return build_pairs_with_hard_negatives(
            texts, labels, embeddings,
            max_pairs_per_class=max_pairs_per_class,
            num_hard_negatives=num_hard_negatives,
        )
    else:
        return build_positive_pairs(texts, labels, max_pairs_per_class)


def generate_training_examples(
    texts: List[str],
    labels: List[int],
    loss_type: LossType,
    embeddings: Optional[np.ndarray] = None,
    max_pairs_per_class: Optional[int] = None,
    num_hard_negatives: int = 3,
    use_hard_negatives: bool = False,
) -> List[InputExample]:
    """
    Generate training examples appropriate for the specified loss function.
    
    Args:
        texts: List of text samples.
        labels: List of corresponding labels.
        loss_type: Type of loss function to generate data for.
        embeddings: Pre-computed embeddings (for hard negative mining).
        max_pairs_per_class: Maximum pairs to generate per class.
        num_hard_negatives: Number of hard negatives per sample.
        use_hard_negatives: Whether to use hard negative mining.
    
    Returns:
        List of InputExample objects formatted for the specified loss.
    """
    if loss_type == LossType.CONTRASTIVE:
        return build_contrastive_pairs(
            texts, labels, embeddings,
            max_pairs_per_class=max_pairs_per_class,
            num_hard_negatives=num_hard_negatives,
            use_hard_negatives=use_hard_negatives,
        )
    elif loss_type == LossType.TRIPLET:
        return build_triplets(
            texts, labels, embeddings,
            max_pairs_per_class=max_pairs_per_class,
            num_hard_negatives=num_hard_negatives,
            use_hard_negatives=use_hard_negatives,
        )
    elif loss_type == LossType.MNRL:
        return build_mnrl_examples(
            texts, labels, embeddings,
            max_pairs_per_class=max_pairs_per_class,
            num_hard_negatives=num_hard_negatives,
            use_hard_negatives=use_hard_negatives,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def prepare_fold_data(
    texts: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    max_pairs_per_class: int,
    loss_type: LossType = LossType.MNRL,
    use_hard_negatives: bool = False,
    num_hard_negatives: int = 3,
    hn_base_model: str = "all-MiniLM-L6-v2",
) -> Tuple[Optional[List[InputExample]], Optional[List[str]], Optional[List[int]], Optional[List[str]], Optional[List[int]]]:
    """
    Prepare training and test data for a single fold.
    
    Args:
        texts: Array of all texts.
        labels: Array of all labels.
        train_idx: Indices for training set.
        test_idx: Indices for test set.
        max_pairs_per_class: Maximum pairs per class for training.
        loss_type: Type of loss function (determines data format).
        use_hard_negatives: Whether to use hard negative mining.
        num_hard_negatives: Number of hard negatives per sample (if enabled).
        hn_base_model: Base model for generating embeddings for hard negative mining.
    
    Returns:
        Tuple of (train_examples, X_train, y_train, X_test, y_test).
        Returns (None, None, None, None, None) if insufficient pairs.
    """
    X_train = texts[train_idx].tolist()
    y_train = labels[train_idx].tolist()
    X_test = texts[test_idx].tolist()
    y_test = labels[test_idx].tolist()
    
    # Get embeddings for hard negative mining if needed
    train_embeddings = None
    if use_hard_negatives:
        print(f"  Mining hard negatives using {hn_base_model}...")
        train_embeddings = get_base_embeddings(X_train, model_name=hn_base_model)
    
    # Generate training examples based on loss type
    train_examples = generate_training_examples(
        X_train, y_train,
        loss_type=loss_type,
        embeddings=train_embeddings,
        max_pairs_per_class=max_pairs_per_class,
        num_hard_negatives=num_hard_negatives,
        use_hard_negatives=use_hard_negatives,
    )
    
    if train_embeddings is not None:
        del train_embeddings
    
    if len(train_examples) < 4:
        return None, None, None, None, None
    
    return train_examples, X_train, y_train, X_test, y_test

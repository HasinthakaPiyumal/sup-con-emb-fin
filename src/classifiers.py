"""Embedding-space classifiers: Centroid-based and KNN."""

from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor or array to numpy float32 array."""
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def build_centroids(
    embeddings: Union[torch.Tensor, np.ndarray],
    labels: Union[List[int], np.ndarray, torch.Tensor]
) -> Dict[int, torch.Tensor]:
    """
    Build normalized class centroids from embeddings.
    
    Args:
        embeddings: Tensor or array of shape (N, D).
        labels: Labels for each embedding.
    
    Returns:
        Dictionary mapping label -> normalized centroid tensor.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu()
    
    # Group embeddings by label
    label_to_vectors = {}
    for emb, label in zip(embeddings, labels):
        vec = emb.cpu() if isinstance(emb, torch.Tensor) else emb
        label_to_vectors.setdefault(label, []).append(vec)
    
    # Compute normalized centroids
    centroids = {}
    for label, vectors in label_to_vectors.items():
        stacked = torch.stack(vectors)
        centroid = stacked.mean(dim=0)
        centroids[label] = F.normalize(centroid, dim=-1)
    
    return centroids


def predict_centroid(
    embeddings: Union[torch.Tensor, np.ndarray],
    centroids: Dict[int, torch.Tensor]
) -> List[int]:
    """
    Predict labels using nearest centroid classification.
    
    Args:
        embeddings: Query embeddings of shape (N, D).
        centroids: Dictionary of class centroids.
    
    Returns:
        List of predicted labels.
    """
    classes = sorted(centroids.keys())
    if not classes:
        return []
    
    # Stack centroids into matrix
    centroid_matrix = torch.stack([centroids[c] for c in classes], dim=0).cpu()
    
    # Convert embeddings to tensor
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cpu() if embeddings.is_cuda else embeddings
    
    # Compute similarities and get predictions
    similarities = embeddings @ centroid_matrix.T
    pred_indices = similarities.argmax(dim=1).numpy()
    
    return [classes[i] for i in pred_indices]


def train_and_classify_knn(
    X_train: Union[torch.Tensor, np.ndarray],
    y_train: Union[List[int], np.ndarray],
    X_test: Union[torch.Tensor, np.ndarray],
    y_test: Union[List[int], np.ndarray],
    n_neighbors: int = 15
) -> np.ndarray:
    """
    Train KNN classifier and return predictions on test set.
    
    Args:
        X_train: Training embeddings.
        y_train: Training labels.
        X_test: Test embeddings.
        y_test: Test labels (unused, kept for API consistency).
        n_neighbors: Number of neighbors for KNN.
    
    Returns:
        Array of predicted labels for test set.
    """
    X_train = _to_numpy(X_train)
    X_test = _to_numpy(X_test)
    
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric="cosine",
        n_jobs=-1
    )
    knn.fit(X_train, y_train)
    
    return knn.predict(X_test)

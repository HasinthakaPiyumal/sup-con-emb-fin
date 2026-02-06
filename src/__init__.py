"""Supervised Contrastive Embedding for Financial Text Classification."""

from .config import set_seed, clear_memory
from .data import load_and_preprocess_data, build_positive_pairs, prepare_fold_data
from .classifiers import build_centroids, predict_centroid, train_and_classify_knn
from .model import train_model, encode_in_batches
from .io_utils import save_fold_embeddings, load_saved_embeddings
from .evaluation import evaluate_saved_embeddings_5fold
from .pipeline import run_5fold_cv

__all__ = [
    "set_seed",
    "clear_memory",
    "load_and_preprocess_data",
    "build_positive_pairs",
    "prepare_fold_data",
    "build_centroids",
    "predict_centroid",
    "train_and_classify_knn",
    "train_model",
    "encode_in_batches",
    "save_fold_embeddings",
    "load_saved_embeddings",
    "evaluate_saved_embeddings_5fold",
    "run_5fold_cv",
]

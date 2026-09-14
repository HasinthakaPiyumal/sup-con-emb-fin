#!/usr/bin/env python3
"""
Prediction / Inference script for fine-tuned code pattern embedding models.

Features:
- Dual-task single-call execution: classifies both AI Patterns and Microservice (MS) Patterns.
- Centroid-based classification matching train.py (normalized embedding cosine similarity).
- Skips already-labeled data (only unverified / missing labels are predicted).
- Generates detailed prediction CSV reports and an aggregated summary CSV report.
- WandB is completely disabled.
- Full support for GPU (CUDA FP16/BF16) and CPU inference with automatic memory management.

Usage:
    # Run both AI and MS pattern predictions on default datasets:
    python predict.py

    # Run only AI patterns or only MS patterns:
    python predict.py --task ai
    python predict.py --task ms

    # Quick test / dry-run on first 5 unlabeled samples:
    python predict.py --sample 5

    # Single text inference query:
    python predict.py --text "Implements JWT token verification middleware for API endpoints"
"""

import os
import sys
import gc
import json
import argparse
from typing import Dict, List, Tuple, Optional

# Strictly disable WandB
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download


# =============================================================================
# Default Configurations
# =============================================================================

DEFAULT_AI_MODEL = "hasinthakapiyumal/bge-reasoner-embed-ai-patterns"
DEFAULT_MS_MODEL = "hasinthakapiyumal/bge-reasoner-embed-ms-patterns"

DEFAULT_AI_DATASET = "data/ai-patterns-all.csv"
DEFAULT_MS_DATASET = "data/ms-patterns-all.csv"

DEFAULT_OUTPUT_DIR = "."


# =============================================================================
# Centroid & Model Loading
# =============================================================================

def get_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    """Detect optimal compute device and precision."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype


def load_model_and_centroids(
    model_name_or_path: str,
    hf_token: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> Tuple[SentenceTransformer, List[str], Dict[int, torch.Tensor]]:
    """
    Load SentenceTransformer model, class_names list, and class centroids dictionary.
    Supports both local directories and Hugging Face Hub repositories.
    """
    if device is None or dtype is None:
        device, dtype = get_device_and_dtype()

    print(f"\n[{model_name_or_path}] Loading model onto {device} ({dtype})...")
    
    # Model kwargs for precision and memory efficiency
    model_kwargs = {"torch_dtype": dtype} if device.type == "cuda" else {}
    model = SentenceTransformer(model_name_or_path, model_kwargs=model_kwargs, device=str(device))
    
    # 1. Load class_names.json
    class_names = None
    if os.path.isdir(model_name_or_path):
        cn_path = os.path.join(model_name_or_path, "class_names.json")
        if os.path.exists(cn_path):
            with open(cn_path, "r", encoding="utf-8") as f:
                class_names = json.load(f)
    if class_names is None:
        try:
            cn_file = hf_hub_download(repo_id=model_name_or_path, filename="class_names.json", token=hf_token)
            with open(cn_file, "r", encoding="utf-8") as f:
                class_names = json.load(f)
        except Exception as e:
            print(f"  Notice: Could not load class_names.json from Hub: {e}")

    # 2. Load centroids.pt
    centroids = None
    if os.path.isdir(model_name_or_path):
        cp_path = os.path.join(model_name_or_path, "centroids.pt")
        if os.path.exists(cp_path):
            centroids = torch.load(cp_path, map_location="cpu", weights_only=False)
    if centroids is None:
        try:
            cp_file = hf_hub_download(repo_id=model_name_or_path, filename="centroids.pt", token=hf_token)
            centroids = torch.load(cp_file, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  Notice: Could not load centroids.pt from Hub: {e}")

    # Normalize centroid dictionary keys to integer
    if centroids is not None:
        normalized_centroids = {}
        for k, v in centroids.items():
            int_k = int(k)
            vec = v.float().cpu()
            normalized_centroids[int_k] = F.normalize(vec, dim=-1)
        centroids = normalized_centroids

    return model, class_names, centroids


def build_centroids_from_labeled_data(
    model: SentenceTransformer,
    texts: List[str],
    labels: List[str],
    class_names: List[str],
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> Dict[int, torch.Tensor]:
    """
    Fallback: dynamically compute class centroids from labeled data matching train.py.
    """
    print("Computing class centroids from labeled dataset samples...")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    label_indices = [class_to_idx[lbl] for lbl in labels if lbl in class_to_idx]
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=str(device) if device else None
    ).cpu().float()

    label_to_vectors = {}
    for emb, l_idx in zip(embeddings, label_indices):
        label_to_vectors.setdefault(l_idx, []).append(emb)

    centroids = {}
    for l_idx, vectors in label_to_vectors.items():
        stacked = torch.stack(vectors)
        centroid = stacked.mean(dim=0)
        centroids[l_idx] = F.normalize(centroid, dim=-1)

    return centroids


# =============================================================================
# Dataset Classification
# =============================================================================

def run_centroid_inference(
    df: pd.DataFrame,
    model: SentenceTransformer,
    class_names: List[str],
    centroids: Dict[int, torch.Tensor],
    task_name: str,
    batch_size: int = 32,
    sample_limit: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify unlabeled rows in df using centroid cosine similarity.
    Skips rows that already have verified labels.
    """
    result_df = df.copy()
    
    # Standardize column presence
    if "label" not in result_df.columns:
        raise ValueError("DataFrame must contain a 'label' column.")
    if "code_summary" not in result_df.columns:
        raise ValueError("DataFrame must contain a 'code_summary' column.")
    
    # Identify labeled vs unlabeled
    is_labeled = result_df["label"].notna() & (result_df["label"].astype(str).str.strip() != "") & (result_df["label"].astype(str).str.lower() != "nan")
    is_unlabeled = ~is_labeled
    
    total_samples = len(result_df)
    labeled_count = int(is_labeled.sum())
    unlabeled_count = int(is_unlabeled.sum())
    
    print(f"\n{'='*70}")
    print(f"TASK: {task_name.upper()}")
    print(f"{'='*70}")
    print(f"Total Dataset Rows       : {total_samples}")
    print(f"Already Labeled (Skipping): {labeled_count} ({labeled_count / total_samples * 100:.1f}%)")
    print(f"Unlabeled (To Classify)  : {unlabeled_count} ({unlabeled_count / total_samples * 100:.1f}%)")
    print(f"Available Target Classes : {len(class_names)}")
    for i, c in enumerate(class_names):
        print(f"  [{i}] {c}")
    
    # Initialize output columns
    result_df["predicted_label"] = result_df["label"]
    result_df["confidence_score"] = np.where(is_labeled, 1.0, np.nan)
    result_df["top2_predicted_label"] = None
    result_df["top2_confidence_score"] = np.nan
    result_df["is_predicted"] = False

    if unlabeled_count == 0:
        print("\nNotice: No unlabeled samples found to classify. All rows already have labels.")
        empty_summary = pd.DataFrame(columns=["class", "already_labeled_count", "newly_predicted_count", "total_count", "mean_confidence"])
        return result_df, empty_summary

    # Get target rows to predict
    target_indices = result_df[is_unlabeled].index
    if sample_limit is not None and sample_limit > 0:
        target_indices = target_indices[:sample_limit]
        print(f"Note: Running with --sample limit of {len(target_indices)} samples for quick evaluation.")

    unlabeled_texts = result_df.loc[target_indices, "code_summary"].fillna("").astype(str).tolist()

    # Encode unlabeled samples
    print(f"\nEncoding {len(unlabeled_texts)} unlabeled code summaries in batches of {batch_size}...")
    embeddings = model.encode(
        unlabeled_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=str(device) if device else None
    ).float()

    # Build centroid matrix sorted by class index
    classes = sorted(centroids.keys())
    centroid_matrix = torch.stack([centroids[c] for c in classes], dim=0).to(embeddings.device).float()

    # Compute cosine similarities: (N, D) @ (num_classes, D).T -> (N, num_classes)
    similarities = embeddings @ centroid_matrix.T

    # Extract Top-1 and Top-2 predictions
    k = min(2, len(classes))
    top_scores, top_indices = torch.topk(similarities, k=k, dim=1)

    top1_indices = top_indices[:, 0].cpu().numpy()
    top1_scores = top_scores[:, 0].cpu().numpy()

    top1_classes = [class_names[classes[idx]] for idx in top1_indices]

    if k >= 2:
        top2_indices = top_indices[:, 1].cpu().numpy()
        top2_scores = top_scores[:, 1].cpu().numpy()
        top2_classes = [class_names[classes[idx]] for idx in top2_indices]
    else:
        top2_scores = [np.nan] * len(top1_indices)
        top2_classes = [None] * len(top1_indices)

    # Assign predictions
    result_df.loc[target_indices, "predicted_label"] = top1_classes
    result_df.loc[target_indices, "confidence_score"] = [round(float(s), 4) for s in top1_scores]
    result_df.loc[target_indices, "top2_predicted_label"] = top2_classes
    result_df.loc[target_indices, "top2_confidence_score"] = [round(float(s), 4) for s in top2_scores]
    result_df.loc[target_indices, "is_predicted"] = True

    # Generate summary report dataframe
    summary_rows = []
    for c_name in class_names:
        c_labeled = int(((result_df["label"] == c_name) & (~result_df["is_predicted"])).sum())
        c_pred = int(((result_df["predicted_label"] == c_name) & (result_df["is_predicted"])).sum())
        c_scores = result_df.loc[(result_df["predicted_label"] == c_name) & (result_df["is_predicted"]), "confidence_score"]
        mean_conf = round(float(c_scores.mean()), 4) if len(c_scores) > 0 else 0.0
        min_conf = round(float(c_scores.min()), 4) if len(c_scores) > 0 else 0.0
        max_conf = round(float(c_scores.max()), 4) if len(c_scores) > 0 else 0.0
        
        summary_rows.append({
            "task": task_name,
            "class": c_name,
            "already_labeled_count": c_labeled,
            "newly_predicted_count": c_pred,
            "total_count": c_labeled + c_pred,
            "mean_confidence": mean_conf,
            "min_confidence": min_conf,
            "max_confidence": max_conf,
        })
    
    summary_df = pd.DataFrame(summary_rows)

    print(f"\nClassification Summary for {task_name.upper()}:")
    print(summary_df[["class", "already_labeled_count", "newly_predicted_count", "total_count", "mean_confidence"]].to_string(index=False))

    return result_df, summary_df


# =============================================================================
# Main Pipeline
# =============================================================================

def run_single_text_prediction(text: str, model_name: str, top_k: int = 3):
    """Predict pattern for a single arbitrary text snippet."""
    device, dtype = get_device_and_dtype()
    model, class_names, centroids = load_model_and_centroids(model_name, device=device, dtype=dtype)
    
    embedding = model.encode([text], convert_to_tensor=True, normalize_embeddings=True).float()
    classes = sorted(centroids.keys())
    centroid_matrix = torch.stack([centroids[c] for c in classes], dim=0).to(embedding.device).float()
    
    similarities = (embedding @ centroid_matrix.T)[0]
    top_indices = torch.topk(similarities, k=min(top_k, len(classes))).indices.tolist()
    
    print(f"\nInput Text: \"{text[:100]}...\"")
    print("\n--- Predictions ---")
    for rank, idx in enumerate(top_indices, start=1):
        c_idx = classes[idx]
        c_name = class_names[c_idx] if c_idx < len(class_names) else f"Class_{c_idx}"
        score = similarities[idx].item()
        print(f"  {rank}. {c_name} (Cosine Similarity: {score:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Dual-Model Centroid-Based Code Pattern Predictor (No WandB)")
    parser.add_argument("--task", type=str, choices=["all", "ai", "ms"], default="all",
                        help="Task to run: 'all' (default, executes both), 'ai', or 'ms'")
    parser.add_argument("--ai_dataset", type=str, default=DEFAULT_AI_DATASET,
                        help="Path to AI patterns CSV dataset")
    parser.add_argument("--ms_dataset", type=str, default=DEFAULT_MS_DATASET,
                        help="Path to MS patterns CSV dataset")
    parser.add_argument("--ai_model", type=str, default=DEFAULT_AI_MODEL,
                        help="Hugging Face repo or local folder for AI patterns model")
    parser.add_argument("--ms_model", type=str, default=DEFAULT_MS_MODEL,
                        help="Hugging Face repo or local folder for MS patterns model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for embedding generation")
    parser.add_argument("--sample", type=int, default=None,
                        help="Optional: Run inference only on the first N unlabeled samples (useful for dry runs)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save prediction report CSV files")
    parser.add_argument("--text", type=str, default=None,
                        help="Optional: Single text query mode (predicts for given string)")
    args = parser.parse_args()

    # Quick single text prediction
    if args.text:
        model_name = args.ms_model if args.task == "ms" else args.ai_model
        run_single_text_prediction(args.text, model_name)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    device, dtype = get_device_and_dtype()
    print(f"Compute Hardware: {device} | Precision: {dtype}")

    all_summaries = []

    # =========================================================================
    # Task 1: AI Patterns
    # =========================================================================
    if args.task in ["all", "ai"]:
        if not os.path.exists(args.ai_dataset):
            print(f"Error: AI dataset not found at {args.ai_dataset}")
            if args.task == "ai":
                sys.exit(1)
        else:
            print(f"\n{'#'*75}")
            print(f"STARTING TASK 1: AI PATTERNS")
            print(f"{'#'*75}")
            
            df_ai = pd.read_csv(args.ai_dataset)
            model_ai, class_names_ai, centroids_ai = load_model_and_centroids(
                args.ai_model, device=device, dtype=dtype
            )
            
            # Fallback if centroids not found in model repo
            if centroids_ai is None:
                print("Building AI centroids from labeled rows...")
                labeled_ai = df_ai[df_ai["label"].notna() & (df_ai["label"] != "")]
                centroids_ai = build_centroids_from_labeled_data(
                    model_ai,
                    labeled_ai["code_summary"].tolist(),
                    labeled_ai["label"].tolist(),
                    class_names_ai,
                    batch_size=args.batch_size,
                    device=device
                )

            preds_ai, summary_ai = run_centroid_inference(
                df=df_ai,
                model=model_ai,
                class_names=class_names_ai,
                centroids=centroids_ai,
                task_name="ai_patterns",
                batch_size=args.batch_size,
                sample_limit=args.sample,
                device=device
            )
            
            ai_out_path = os.path.join(args.output_dir, "predictions_ai_patterns.csv")
            preds_ai.to_csv(ai_out_path, index=False)
            print(f"\n[AI Patterns] Saved prediction report to: {os.path.abspath(ai_out_path)}")
            all_summaries.append(summary_ai)

            # Cleanup AI model memory before MS model
            del model_ai, centroids_ai, class_names_ai, df_ai, preds_ai
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # =========================================================================
    # Task 2: Microservices Patterns
    # =========================================================================
    if args.task in ["all", "ms"]:
        if not os.path.exists(args.ms_dataset):
            print(f"Error: MS dataset not found at {args.ms_dataset}")
            if args.task == "ms":
                sys.exit(1)
        else:
            print(f"\n{'#'*75}")
            print(f"STARTING TASK 2: MICROSERVICES (MS) PATTERNS")
            print(f"{'#'*75}")
            
            df_ms = pd.read_csv(args.ms_dataset)
            model_ms, class_names_ms, centroids_ms = load_model_and_centroids(
                args.ms_model, device=device, dtype=dtype
            )
            
            # Fallback if centroids not found in model repo
            if centroids_ms is None:
                print("Building MS centroids from labeled rows...")
                labeled_ms = df_ms[df_ms["label"].notna() & (df_ms["label"] != "")]
                centroids_ms = build_centroids_from_labeled_data(
                    model_ms,
                    labeled_ms["code_summary"].tolist(),
                    labeled_ms["label"].tolist(),
                    class_names_ms,
                    batch_size=args.batch_size,
                    device=device
                )

            preds_ms, summary_ms = run_centroid_inference(
                df=df_ms,
                model=model_ms,
                class_names=class_names_ms,
                centroids=centroids_ms,
                task_name="ms_patterns",
                batch_size=args.batch_size,
                sample_limit=args.sample,
                device=device
            )
            
            ms_out_path = os.path.join(args.output_dir, "predictions_ms_patterns.csv")
            preds_ms.to_csv(ms_out_path, index=False)
            print(f"\n[MS Patterns] Saved prediction report to: {os.path.abspath(ms_out_path)}")
            all_summaries.append(summary_ms)

            # Cleanup MS model memory
            del model_ms, centroids_ms, class_names_ms, df_ms, preds_ms
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # =========================================================================
    # Aggregated Summary Report
    # =========================================================================
    if all_summaries:
        master_summary = pd.concat(all_summaries, ignore_index=True)
        summary_out_path = os.path.join(args.output_dir, "prediction_summary_report.csv")
        master_summary.to_csv(summary_out_path, index=False)
        print(f"\n{'='*75}")
        print(f"ALL PREDICTIONS COMPLETE")
        print(f"{'='*75}")
        print(f"Master Summary Report saved to: {os.path.abspath(summary_out_path)}")


if __name__ == "__main__":
    main()

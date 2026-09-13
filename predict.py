#!/usr/bin/env python3
"""
Prediction / Inference script for fine-tuned embedding models.

Usage:
    # Single prediction from command line:
    python predict.py --text "Implements a repository pattern with PostgreSQL database connector"

    # Specify custom saved model directory:
    python predict.py --model_dir saved_models/full-dataset-contrastive-hn10-ep3 --text "Your code snippet"

    # Interactive mode:
    python predict.py
"""

import argparse
import json
import os
import sys
import torch
from sentence_transformers import SentenceTransformer
from src.classifiers import predict_centroid


def find_latest_saved_model(base_dir: str = "saved_models") -> str:
    """Find the most recently created model folder in base_dir."""
    if not os.path.exists(base_dir):
        return None
    subdirs = [
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not subdirs:
        return None
    subdirs.sort(key=os.path.getmtime, reverse=True)
    return subdirs[0]


def load_model_and_artifacts(model_dir: str):
    """Load fine-tuned model, class names, and centroids."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading fine-tuned model from: {model_dir}")
    model = SentenceTransformer(model_dir)

    # Load class names
    classes_path = os.path.join(model_dir, "class_names.json")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Missing {classes_path}. Re-run training to generate class names.")
    with open(classes_path, "r") as f:
        class_names = json.load(f)

    # Load centroids
    centroids_path = os.path.join(model_dir, "centroids.pt")
    if not os.path.exists(centroids_path):
        raise FileNotFoundError(f"Missing {centroids_path}. Re-run training to generate centroids.")
    centroids = torch.load(centroids_path, map_location="cpu")

    return model, class_names, centroids


def predict(text: str, model: SentenceTransformer, class_names: list, centroids: dict, top_k: int = 3):
    """Predict top classes for a given text query."""
    embedding = model.encode([text], convert_to_tensor=True, normalize_embeddings=True)
    
    classes = sorted(centroids.keys())
    centroid_matrix = torch.stack([centroids[c] for c in classes], dim=0).cpu()
    
    # Cosine similarities
    similarities = (embedding.cpu() @ centroid_matrix.T)[0]
    
    # Top-K predictions
    top_indices = torch.topk(similarities, k=min(top_k, len(classes))).indices.tolist()
    
    results = []
    for idx in top_indices:
        class_idx = classes[idx]
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
        score = similarities[idx].item()
        results.append((class_name, score))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict pattern using fine-tuned embedding model")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to saved model directory")
    parser.add_argument("--text", type=str, default=None, help="Text/code description to classify")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display")
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir:
        model_dir = find_latest_saved_model("saved_models")
        if not model_dir:
            print("Error: No saved models found in 'saved_models/'. Please train a model first with train.py.")
            sys.exit(1)
        print(f"Using latest saved model: {model_dir}")

    model, class_names, centroids = load_model_and_artifacts(model_dir)

    if args.text:
        results = predict(args.text, model, class_names, centroids, top_k=args.top_k)
        print("\n--- Prediction Results ---")
        for rank, (name, score) in enumerate(results, start=1):
            print(f"{rank}. {name} (Similarity: {score:.4f})")
    else:
        print("\n--- Interactive Prediction Mode ---")
        print("Type your code summary or description (or 'exit' to quit):")
        while True:
            try:
                user_input = input("\nEnter text: ").strip()
                if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                    break
                results = predict(user_input, model, class_names, centroids, top_k=args.top_k)
                print("\nTop Predictions:")
                for rank, (name, score) in enumerate(results, start=1):
                    print(f"  {rank}. {name} (Cosine Similarity: {score:.4f})")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nDone.")


if __name__ == "__main__":
    main()

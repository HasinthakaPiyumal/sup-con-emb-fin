#!/usr/bin/env python3
"""
Two-phase 5-fold CV training script.

Phase 1: Train embedding model with configurable loss, save test embeddings.
Phase 2: Evaluate embeddings with centroid and KNN classifiers.

Supported loss functions:
- "contrastive": ContrastiveLoss - pairs with similarity labels (0/1)
- "mnrl": MultipleNegativesRankingLoss - (anchor, positive, [negatives...])
- "triplet": TripletLoss - (anchor, positive, negative) triplets
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.data import load_and_preprocess_data, LossType
from src.pipeline import run_5fold_cv, run_5fold_cv_no_finetuning, train_full_dataset_and_push_to_hub


# =============================================================================
# Configuration
# =============================================================================

# If True: encode all samples with raw model (no training), then 5-fold CV with
# KNN and Centroid on embeddings only. Fine-tuning pipeline is skipped.
RUN_WITHOUT_FINETUNING = False

# Loss function: "contrastive", "mnrl", or "triplet"
LOSS_TYPE = "contrastive"  # Options: LossType.CONTRASTIVE, LossType.MNRL, LossType.TRIPLET
LOSS_MARGIN = 0.5  # Margin for contrastive/triplet loss

# Hard negative mining
USE_HARD_NEGATIVES = True
NUM_HARD_NEGATIVES = 10
HN_BASE_MODEL = "google-bert/bert-base-uncased"  # Fast model for mining

# Model and training
MODEL_NAME = "BAAI/bge-reasoner-embed-qwen3-8b-0923"
# MODEL_NAME = "BAAI/bge-code-v1"
# MODEL_NAME = "google-bert/bert-base-uncased"
EPOCHS = 3
BATCH_SIZE = 32  # Recommended batch size for 1.5B LLM backbones on 20GB GPUs
LEARNING_RATE = 2e-5
WARMUP_STEPS = 10
MAX_PAIRS_PER_CLASS = 80
MAX_SEQ_LENGTH = 768
SEED = 42
DENSE_DIM = 8

# Training Strategy Mode:
# - "lora": Parameter-Efficient Fine-Tuning (PEFT) using LoRA adapters (Recommended)
# - "freeze_base": Freeze Transformer backbone parameters, train only projection head
# - "full": Full fine-tuning of all Transformer layers
TRAINING_MODE = "full"  # Options: "lora", "freeze_base", "full"

# LoRA / PEFT hyperparameters (active when TRAINING_MODE = "lora")
LORA_R = 16  # LoRA rank dimension (e.g. 8, 16, 32)
LORA_ALPHA = 32  # LoRA scaling factor alpha (e.g. 16, 32, 64)
LORA_DROPOUT = 0.05  # Dropout probability for LoRA layers
LORA_TARGET_MODULES = None  # None for auto-detection (works for Qwen2, BERT, RoBERTa, LLaMA)

# Minimum samples per label to keep (labels with fewer are dropped).
# For unique 413-sample dataset, use 5 or 20. (Use 120 only for 2478-sample concated dataset).
MIN_SAMPLES_PER_LABEL = 20

# 5-Fold Cross Validation:
# - If True: runs 5-fold CV evaluation, logs metrics and saves OOF embeddings/confusion matrix.
# - If False: skips 5-fold CV (saves massive time if you just want to train and save the final model).
RUN_5FOLD_CV = False

# Train on full dataset and optionally save/push model.
RUN_FULL_DATASET_TRAINING = True  # Set to True to train on full dataset and save model
SAVE_MODEL_LOCALLY = True  # Save trained model locally
LOCAL_MODEL_DIR = "saved_models"  # Directory to save models locally
PUSH_TO_HUB = False  # Set to True to push model to Hugging Face Hub (requires HUB_MODEL_ID)
HUB_MODEL_ID = "hasinthakapiyumal/bge-code-v1-ai-pattern-tuned"  # e.g. "your-org/code-embedding-model" (required if PUSH_TO_HUB=True)
HF_TOKEN = ""  # Optional; set or use `huggingface-cli login`


def main():
    """Main entry point for training."""
    # dataset_path = "./data/labeled_verified_data.csv"
    dataset_path = "./data/feb-10-2026-community-descriptions-unique.csv"
    
    try:
        dataset, label_encoder = load_and_preprocess_data(
            dataset_path, min_samples_per_label=MIN_SAMPLES_PER_LABEL
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Suppress wandb output
    os.environ["WANDB_SILENT"] = "true"
    
    if RUN_WITHOUT_FINETUNING:
        run_5fold_cv_no_finetuning(
            texts=dataset["code_summary"],
            labels=dataset["label_enc"],
            class_names=list(label_encoder.classes_),
            model_name=MODEL_NAME,
            num_folds=5,
            batch_size=BATCH_SIZE,
            max_seq_length=MAX_SEQ_LENGTH,
            seed=SEED,
            files=dataset["file"].tolist() if "file" in dataset.columns else None,
            descriptions=dataset["code_summary"].tolist(),
        )
    else:
        # Determine training mode flags
        mode = TRAINING_MODE.lower().strip()
        use_lora = (mode == "lora")
        freeze_base = (mode in ["freeze_base", "freeze", "frozen_backbone"])
        
        print(f"Executing Training Strategy: {mode.upper()} (use_lora={use_lora}, freeze_base={freeze_base})")

        # Run 5-fold CV evaluation if enabled
        if RUN_5FOLD_CV:
            run_5fold_cv(
                texts=dataset["code_summary"],
                labels=dataset["label_enc"],
                class_names=list(label_encoder.classes_),
                model_name=MODEL_NAME,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LEARNING_RATE,
                warmup_steps=WARMUP_STEPS,
                max_pairs_per_class=MAX_PAIRS_PER_CLASS,
                max_seq_length=MAX_SEQ_LENGTH,
                seed=SEED,
                dense_dim=DENSE_DIM,
                # Loss function settings
                loss_type=LOSS_TYPE,
                loss_margin=LOSS_MARGIN,
                # Hard negative mining settings
                use_hard_negatives=USE_HARD_NEGATIVES,
                num_hard_negatives=NUM_HARD_NEGATIVES,
                hn_base_model=HN_BASE_MODEL,
                # OOF CSV metadata (label, file, description + embeddings)
                files=dataset["file"].tolist() if "file" in dataset.columns else None,
                descriptions=dataset["code_summary"].tolist(),
                freeze_base_model=freeze_base,
                # LoRA / PEFT settings
                use_lora=use_lora,
                lora_r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                lora_target_modules=LORA_TARGET_MODULES,
            )
        else:
            print("Skipping 5-fold CV (RUN_5FOLD_CV=False).")

        # Train on full dataset and save/push model
        if RUN_FULL_DATASET_TRAINING:
            saved_path = train_full_dataset_and_push_to_hub(
                texts=dataset["code_summary"],
                labels=dataset["label_enc"],
                class_names=list(label_encoder.classes_),
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                warmup_steps=WARMUP_STEPS,
                max_pairs_per_class=MAX_PAIRS_PER_CLASS,
                seed=SEED,
                dense_dim=DENSE_DIM,
                loss_type=LOSS_TYPE,
                loss_margin=LOSS_MARGIN,
                use_hard_negatives=USE_HARD_NEGATIVES,
                num_hard_negatives=NUM_HARD_NEGATIVES,
                hn_base_model=HN_BASE_MODEL,
                hub_model_id=HUB_MODEL_ID if PUSH_TO_HUB else None,
                push_to_hub=PUSH_TO_HUB,
                hf_token=HF_TOKEN or None,
                save_local=SAVE_MODEL_LOCALLY,
                local_save_dir=LOCAL_MODEL_DIR,
                freeze_base_model=freeze_base,
                use_lora=use_lora,
                lora_r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                lora_target_modules=LORA_TARGET_MODULES,
            )
            print(f"\nModel saved at: {saved_path}")
            if SAVE_MODEL_LOCALLY and not PUSH_TO_HUB:
                print(f"\nTo upload later, run:")
                print(f"  python upload_model.py --model_path {saved_path} --hub_id {HUB_MODEL_ID}")


if __name__ == "__main__":
    main()

"""Main training pipeline with cross-validation."""

import os
from typing import List, Optional, Sequence, Union

import numpy as np
import wandb
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

from .config import set_seed, clear_memory
from .data import prepare_fold_data, LossType
from .model import train_model, encode_in_batches
from .classifiers import build_centroids, predict_centroid, train_and_classify_knn
from .io_utils import save_fold_embeddings, save_all_embeddings
from .evaluation import evaluate_saved_embeddings_5fold, _report_method


def build_wandb_tags(config: dict) -> List[str]:
    """Build tags for wandb run based on configuration."""
    tags = []
    
    # No-finetuning tag
    if config.get("run_without_finetuning", False):
        tags.append("no-finetuning")
    
    # Model tag
    model_short = config["model_name"].split("/")[-1]
    tags.append(f"model:{model_short}")
    
    # Loss function tag (skip when no finetuning)
    if not config.get("run_without_finetuning", False):
        loss_type = config.get("loss_type", "mnrl")
        tags.append(f"loss:{loss_type}")
    
    # Hard negative mining tag
    use_hn = config.get("use_hard_negatives", False)
    if use_hn:
        num_hn = config.get("num_hard_negatives", 3)
        tags.append("hard-negatives")
        tags.append(f"hn:{num_hn}")
    else:
        tags.append("no-hard-negatives")
    
    # Training config tags
    tags.append(f"epochs:{config.get('epochs', 1)}")
    tags.append(f"bs:{config.get('batch_size', 4)}")
    tags.append(f"seq:{config.get('max_seq_length', 256)}")
    
    # Folds tag
    tags.append(f"folds:{config.get('num_folds', 5)}")
    
    return tags


def init_wandb(model_name: str, config: dict, run_name_override: str = None) -> None:
    """Initialize Weights & Biases logging with tags."""
    tags = build_wandb_tags(config)
    
    if run_name_override is not None:
        run_name = run_name_override
    else:
        # Build run name: model-loss-hn_status-seq_length
        loss_type = config.get("loss_type", "mnrl")
        hn_status = "hn" if config.get("use_hard_negatives", False) else "no-hn"
        run_name = f"{model_name.split('/')[-1]}-{loss_type}-{hn_status}-{config['max_seq_length']}"
    
    wandb.init(
        project="code-classification-super-cons-learn[AI Patterns]",
        name=run_name,
        config=config,
        tags=tags,
    )


def run_5fold_cv_no_finetuning(
    texts: Sequence[str],
    labels: Sequence[int],
    class_names: List[str],
    model_name: str = "google-bert/bert-base-uncased",
    num_folds: int = 5,
    batch_size: int = 32,
    max_seq_length: int = 256,
    seed: int = 42,
    files: Optional[Sequence[str]] = None,
    descriptions: Optional[Sequence[str]] = None,
    save_dir: str = "saved_test_embeddings",
) -> None:
    """
    Encode all samples with the raw model (no training), then run 5-fold CV
    training KNN and Centroid on 4 folds and testing on 1 fold. Log results to
    console and wandb. Save out-of-fold test embeddings to disk.
    """
    from sentence_transformers import SentenceTransformer

    set_seed(seed)
    texts = np.array(list(texts), dtype=object)
    labels = np.array(list(labels), dtype=int)
    files_arr = np.array(list(files), dtype=object) if files is not None else None
    descriptions_arr = np.array(list(descriptions), dtype=object) if descriptions is not None else None

    if len(labels) == 0:
        raise ValueError(
            "No samples in dataset (0 texts/labels). Need at least 1 sample for 5-fold CV. "
            "Check that the data file exists, has rows, and preprocessing did not drop all rows."
        )

    config = {
        "run_without_finetuning": True,
        "model_name": model_name,
        "num_folds": num_folds,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "seed": seed,
    }
    run_name = f"no-ft-{model_name.split('/')[-1]}-{max_seq_length}"
    init_wandb(model_name, config, run_name_override=run_name)

    print(f"\n{'=' * 80}")
    print("RUN WITHOUT FINETUNING: encoding all samples with raw model")
    print(f"{'=' * 80}")
    print(f"  Model: {model_name}")
    print(f"  Samples: {len(texts)}, Folds: {num_folds}")
    if files_arr is not None:
        print(f"  Grouped by unique files: {len(np.unique(files_arr))}")

    clear_memory()
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    model.eval()

    embeddings = encode_in_batches(model, list(texts), batch_size=batch_size)
    del model
    clear_memory()

    acc_centroid, f1_centroid = [], []
    acc_knn, f1_knn = [], []
    all_true, pred_centroid_all, pred_knn_all = [], [], []
    all_fold_data = []

    if files_arr is not None:
        sgkf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        split_iter = sgkf.split(embeddings, labels, groups=files_arr)
    else:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        split_iter = skf.split(embeddings, labels)

    for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
        X_train = embeddings[train_idx]
        y_train = labels[train_idx]
        X_test = embeddings[test_idx]
        y_test = labels[test_idx]

        centroids = build_centroids(X_train, y_train)
        pred_c = predict_centroid(X_test, centroids)
        acc_c = accuracy_score(y_test, pred_c)
        f1_c = f1_score(y_test, pred_c, average="macro")
        acc_centroid.append(acc_c)
        f1_centroid.append(f1_c)
        all_true.extend(y_test)
        pred_centroid_all.extend(pred_c)

        pred_k = train_and_classify_knn(X_train, y_train, X_test, y_test)
        acc_k = accuracy_score(y_test, pred_k)
        f1_k = f1_score(y_test, pred_k, average="macro")
        acc_knn.append(acc_k)
        f1_knn.append(f1_k)
        pred_knn_all.extend(pred_k)

        # Save fold OOF embeddings
        test_files = files_arr[test_idx] if files_arr is not None else None
        test_descriptions = descriptions_arr[test_idx] if descriptions_arr is not None else None
        all_fold_data.append(save_fold_embeddings(
            fold, y_test, X_test,
            files=test_files, descriptions=test_descriptions,
        ))

        print(
            f"No-FT fold {fold}: "
            f"Centroid Acc {acc_c:.4f} F1 {f1_c:.4f} | "
            f"KNN Acc {acc_k:.4f} F1 {f1_k:.4f}"
        )
        wandb.log({
            f"no_ft_fold{fold}_centroid_acc": acc_c,
            f"no_ft_fold{fold}_centroid_f1": f1_c,
            f"no_ft_fold{fold}_knn_acc": acc_k,
            f"no_ft_fold{fold}_knn_f1": f1_k,
        })

    # Save all OOF embeddings
    if all_fold_data:
        out_path = save_all_embeddings(all_fold_data, save_dir)
        print(f"Saved out-of-fold embeddings -> {out_path}")

    _report_method(
        "NO FINETUNING - CENTROID",
        acc_centroid, f1_centroid,
        all_true, pred_centroid_all,
        class_names, "no_ft_centroid",
        save_dir=save_dir,
    )
    _report_method(
        "NO FINETUNING - KNN",
        acc_knn, f1_knn,
        all_true, pred_knn_all,
        class_names, "no_ft_knn",
        save_dir=save_dir,
    )
    wandb.finish()


def run_5fold_cv(
    texts: Sequence[str],
    labels: Sequence[int],
    class_names: List[str],
    model_name: str = "nomic-ai/nomic-embed-text-v1",
    num_folds: int = 5,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 2e-5,
    warmup_steps: int = 10,
    max_pairs_per_class: int = 100,
    max_seq_length: int = 256,
    seed: int = 42,
    save_dir: str = "saved_test_embeddings",
    dense_dim: int = 8,
    # Loss function settings
    loss_type: Union[LossType, str] = LossType.MNRL,
    loss_margin: float = 0.5,
    # Hard negative mining settings
    use_hard_negatives: bool = False,
    num_hard_negatives: int = 3,
    hn_base_model: str = "all-MiniLM-L6-v2",
    # Optional metadata for OOF CSV (must align with texts/labels)
    files: Optional[Sequence[str]] = None,
    descriptions: Optional[Sequence[str]] = None,
) -> None:
    """
    Run two-phase 5-fold cross-validation.
    
    Phase 1: Train embedding model and save test embeddings for each fold.
    Phase 2: Evaluate embeddings with centroid and KNN classifiers.
    
    Args:
        texts: Input text samples.
        labels: Encoded integer labels.
        class_names: Original class names for reporting.
        model_name: HuggingFace model identifier.
        num_folds: Number of CV folds.
        epochs: Training epochs per fold.
        batch_size: Training batch size.
        lr: Learning rate.
        warmup_steps: LR warmup steps.
        max_pairs_per_class: Maximum contrastive pairs per class.
        max_seq_length: Maximum token sequence length.
        seed: Random seed.
        save_dir: Directory to save embeddings.
        dense_dim: Output dimension for projection head.
        loss_type: Loss function type ("contrastive", "mnrl", "triplet" or LossType enum).
        loss_margin: Margin for contrastive/triplet loss.
        use_hard_negatives: Whether to use hard negative mining.
        num_hard_negatives: Number of hard negatives per sample.
        hn_base_model: Base model for hard negative mining embeddings.
        files: Optional file identifiers aligned with texts (for OOF CSV).
        descriptions: Optional descriptions (e.g. code_summary) aligned with texts (for OOF CSV).
    """
    # Convert string loss_type to enum if needed
    if isinstance(loss_type, str):
        loss_type = LossType(loss_type)
    set_seed(seed)
    
    # Build config for logging
    config = {
        "model_name": model_name,
        "num_folds": num_folds,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "max_pairs_per_class": max_pairs_per_class,
        "max_seq_length": max_seq_length,
        "seed": seed,
        "dense_dim": dense_dim,
        "loss_type": loss_type.value,
        "loss_margin": loss_margin,
        "use_hard_negatives": use_hard_negatives,
        "num_hard_negatives": num_hard_negatives,
        "hn_base_model": hn_base_model,
    }
    init_wandb(model_name, config)
    
    print(f"\n{'=' * 80}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"  Loss function: {loss_type.value}")
    if loss_type in (LossType.CONTRASTIVE, LossType.TRIPLET):
        print(f"  Loss margin: {loss_margin}")
    if use_hard_negatives:
        print(f"  Hard negative mining: {num_hard_negatives} negatives using {hn_base_model}")
    else:
        print(f"  Hard negative mining: disabled")
    
    # Convert to arrays
    texts = np.array(list(texts), dtype=object)
    labels = np.array(list(labels), dtype=int)
    files_arr = np.array(list(files), dtype=object) if files is not None else None
    descriptions_arr = np.array(list(descriptions), dtype=object) if descriptions is not None else None
    if files_arr is not None:
        print(f"  Grouped CV by unique files: {len(np.unique(files_arr))}")
        sgkf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        split_iter = sgkf.split(texts, labels, groups=files_arr)
    else:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        split_iter = skf.split(texts, labels)

    os.makedirs(save_dir, exist_ok=True)
    
    all_fold_data = []
    
    # Phase 1: Train and generate embeddings
    for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
        print(f"\n{'=' * 80}")
        print(f"PHASE 1 FOLD {fold}/{num_folds}")
        print(f"{'=' * 80}")
        
        clear_memory()
        
        # Prepare fold data with optional hard negative mining
        prep = prepare_fold_data(
            texts, labels, train_idx, test_idx, max_pairs_per_class,
            loss_type=loss_type,
            use_hard_negatives=use_hard_negatives,
            num_hard_negatives=num_hard_negatives,
            hn_base_model=hn_base_model,
        )
        if prep[0] is None:
            print("Skipping fold: not enough pairs.")
            continue
        
        train_examples, X_train, y_train, X_test, y_test = prep
        print(f"  Generated {len(train_examples)} training examples")
        
        # Train model
        model = train_model(
            model_name, max_seq_length, train_examples,
            batch_size, epochs, warmup_steps, lr,
            dense_dim=dense_dim,
            loss_type=loss_type,
            loss_margin=loss_margin,
        )
        
        clear_memory()
        model.eval()
        
        # Generate embeddings
        train_emb = encode_in_batches(model, X_train, batch_size=32)
        test_emb = encode_in_batches(model, X_test, batch_size=32)
        
        # Phase 1 evaluation with centroids
        centroids = build_centroids(train_emb, y_train)
        pred_c = predict_centroid(test_emb, centroids)
        acc_c = accuracy_score(y_test, pred_c)
        f1_c = f1_score(y_test, pred_c, average="macro")
        
        print(f"Phase 1 fold {fold} centroid: Acc {acc_c:.4f}, Macro F1 {f1_c:.4f}")
        wandb.log({
            f"phase1_fold{fold}_centroid_acc": acc_c,
            f"phase1_fold{fold}_centroid_f1": f1_c
        })
        
        # Save fold embeddings (with optional file/description for OOF CSV)
        test_files = files_arr[test_idx] if files_arr is not None else None
        test_descriptions = descriptions_arr[test_idx] if descriptions_arr is not None else None
        all_fold_data.append(save_fold_embeddings(
            fold, y_test, test_emb,
            files=test_files, descriptions=test_descriptions,
        ))
        
        # Cleanup
        del train_emb, centroids, pred_c
        del model, train_examples, test_emb, X_train, y_train, X_test, y_test
        clear_memory()
    
    # Save all embeddings
    if all_fold_data:
        out_path = save_all_embeddings(all_fold_data, save_dir)
        print(f"Saved embeddings -> {out_path}")
    
    # Phase 2: Evaluate with different classifiers
    evaluate_saved_embeddings_5fold(save_dir, class_names, num_folds=num_folds, seed=seed)

    wandb.finish()


def train_full_dataset_and_push_to_hub(
    texts: Sequence[str],
    labels: Sequence[int],
    model_name: str,
    max_seq_length: int,
    batch_size: int,
    epochs: int,
    lr: float,
    warmup_steps: int,
    max_pairs_per_class: int,
    seed: int,
    dense_dim: int = 8,
    loss_type: Union[LossType, str] = LossType.MNRL,
    loss_margin: float = 0.5,
    use_hard_negatives: bool = False,
    num_hard_negatives: int = 3,
    hn_base_model: str = "all-MiniLM-L6-v2",
    hub_model_id: Optional[str] = None,
    push_to_hub: bool = False,
    hf_token: Optional[str] = None,
    save_local: bool = True,
    local_save_dir: str = "saved_models",
) -> str:
    """
    Train contrastive embedding model on the full dataset, optionally save locally and push to Hugging Face Hub.

    Uses the same loss and hard-negative settings as the 5-fold pipeline. All samples
    are used for training (no train/test split). Call this after run_5fold_cv() when
    you want a single model trained on all data.

    Args:
        texts: Input text samples (full dataset).
        labels: Encoded integer labels.
        model_name: HuggingFace model identifier.
        max_seq_length: Maximum token sequence length.
        batch_size: Training batch size.
        epochs: Training epochs.
        lr: Learning rate.
        warmup_steps: LR warmup steps.
        max_pairs_per_class: Maximum contrastive pairs per class.
        seed: Random seed.
        dense_dim: Output dimension for projection head.
        loss_type: Loss function type.
        loss_margin: Margin for contrastive/triplet loss.
        use_hard_negatives: Whether to use hard negative mining.
        num_hard_negatives: Number of hard negatives per sample.
        hn_base_model: Base model for hard negative mining.
        hub_model_id: Hugging Face Hub repo id (e.g. "username/repo-name"). Required if push_to_hub.
        push_to_hub: Whether to push the trained model to the Hub.
        hf_token: Optional Hugging Face token for login (else use huggingface-cli login).
        save_local: Whether to save the model locally.
        local_save_dir: Directory to save the model locally.

    Returns:
        Path to saved model (local path if saved locally, hub_model_id if pushed to hub).
    """
    if isinstance(loss_type, str):
        loss_type = LossType(loss_type)
    set_seed(seed)

    texts_arr = np.array(list(texts), dtype=object)
    labels_arr = np.array(list(labels), dtype=int)
    n = len(texts_arr)
    train_idx = np.arange(n)
    test_idx = np.array([], dtype=np.int64)

    print(f"\n{'=' * 80}")
    print("FULL-DATASET CONTRASTIVE TRAINING")
    print(f"{'=' * 80}")
    print(f"  Samples: {n}, Loss: {loss_type.value}")

    clear_memory()
    prep = prepare_fold_data(
        texts_arr, labels_arr, train_idx, test_idx, max_pairs_per_class,
        loss_type=loss_type,
        use_hard_negatives=use_hard_negatives,
        num_hard_negatives=num_hard_negatives,
        hn_base_model=hn_base_model,
    )
    if prep[0] is None:
        print("Full-dataset training skipped: not enough pairs.")
        return
    train_examples, _, _, _, _ = prep
    print(f"  Generated {len(train_examples)} training examples")

    model = train_model(
        model_name, max_seq_length, train_examples,
        batch_size, epochs, warmup_steps, lr,
        dense_dim=dense_dim,
        loss_type=loss_type,
        loss_margin=loss_margin,
    )

    saved_path = None

    # Save locally if requested
    if save_local:
        import os
        os.makedirs(local_save_dir, exist_ok=True)
        # Create a descriptive model name based on config
        model_suffix = f"{loss_type.value}-hn{num_hard_negatives if use_hard_negatives else 0}-ep{epochs}"
        local_model_path = os.path.join(local_save_dir, f"full-dataset-{model_suffix}")
        print(f"Saving model locally to: {local_model_path}")
        model.save(local_model_path)
        saved_path = local_model_path
        print(f"Model saved locally at: {local_model_path}")

    # Push to Hub if requested
    if push_to_hub and hub_model_id:
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token)
            except Exception as e:
                print(f"Warning: HF login with token failed: {e}")
        
        print(f"Pushing model to Hugging Face Hub: {hub_model_id}")
        try:
            # Use create_repo with exist_ok=True to handle existing repos
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=hub_model_id, exist_ok=True, repo_type="model")
            
            # push_to_hub internally calls create_repo without exist_ok, so catch 409 errors
            try:
                model.push_to_hub(hub_model_id)
                print(f"Done. Model available at https://huggingface.co/{hub_model_id}")
            except Exception as push_error:
                error_str = str(push_error)
                # If repo already exists (409), use upload_folder directly
                if "409" in error_str or "already created" in error_str.lower() or "Conflict" in error_str:
                    print("Repository already exists. Uploading files directly...")
                    if save_local and saved_path:
                        # Upload from local path
                        api.upload_folder(
                            folder_path=saved_path,
                            repo_id=hub_model_id,
                            repo_type="model",
                            commit_message=f"Upload fine-tuned model: {loss_type.value}",
                        )
                    else:
                        # Save temporarily to upload
                        import tempfile
                        with tempfile.TemporaryDirectory() as tmpdir:
                            model.save(tmpdir)
                            api.upload_folder(
                                folder_path=tmpdir,
                                repo_id=hub_model_id,
                                repo_type="model",
                                commit_message=f"Upload fine-tuned model: {loss_type.value}",
                            )
                    print(f"Done. Model available at https://huggingface.co/{hub_model_id}")
                else:
                    raise
            
            saved_path = hub_model_id
        except Exception as e:
            print(f"Error pushing to Hub: {e}")
            print("Model was saved locally (if enabled) but not uploaded to Hub.")
    elif push_to_hub:
        print("Warning: push_to_hub=True but hub_model_id not set; skipping Hub upload.")

    del model
    clear_memory()
    return saved_path or "Model not saved"

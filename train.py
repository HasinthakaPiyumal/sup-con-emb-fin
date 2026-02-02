import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models

# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================
# Load dataset
# =========================
def load_and_preprocess_data(filepath, min_samples_per_label=20):
    if not os.path.exists(filepath):
        # Fallback to local file if absolute path doesn't exist
        local_path = os.path.basename(filepath)
        if os.path.exists(local_path):
            filepath = local_path
        else:
            raise FileNotFoundError(f"Dataset file not found at {filepath} or {local_path}")
    
    print(f"Loading dataset from: {filepath}")
    dataset = pd.read_csv(filepath)
    
    # Filter classes with few samples
    counts = dataset["label"].value_counts()
    valid_labels = counts[counts >= min_samples_per_label].index
    dataset = dataset[dataset["label"].isin(valid_labels)].reset_index(drop=True)
    
    le = LabelEncoder()
    dataset['label_enc'] = le.fit_transform(dataset['label'])
    
    return dataset, le

# =========================
# Build positive training pairs (same-class)
# =========================
def build_positive_pairs(texts, labels, max_pairs_per_class=None):
    label_to_idxs = {}
    for i, y in enumerate(labels):
        label_to_idxs.setdefault(y, []).append(i)

    examples = []
    for y, idxs in label_to_idxs.items():
        if len(idxs) < 2:
            continue

        pairs = []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                pairs.append((idxs[i], idxs[j]))

        if max_pairs_per_class is not None and len(pairs) > max_pairs_per_class:
            pairs = random.sample(pairs, max_pairs_per_class)

        for a, b in pairs:
            examples.append(InputExample(texts=[str(texts[a]), str(texts[b])]))

    random.shuffle(examples)
    return examples

# =========================
# Centroid classifier in embedding space
# =========================
def build_centroids(embeddings, labels):
    label_to_vecs = {}
    
    # Ensure labels are items if they are tensors
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
        
    for e, y in zip(embeddings, labels):
        label_to_vecs.setdefault(y, []).append(e)

    centroids = {}
    for y, vecs in label_to_vecs.items():
        c = torch.stack(vecs).mean(dim=0)
        centroids[y] = F.normalize(c, dim=-1)
    return centroids

def predict_centroid(embeddings, centroids):
    classes = sorted(centroids.keys())
    # Stack centroids handling potential device mismatches
    c_list = [centroids[c] for c in classes]
    if not c_list:
        return []
    
    first_device = c_list[0].device
    C = torch.stack(c_list, dim=0).to(embeddings.device)
    
    sims = embeddings @ C.T
    pred_idx = sims.argmax(dim=1).cpu().numpy()
    preds = [classes[i] for i in pred_idx]
    return preds

def init_wandb(model_name, config):
    wandb.init(
        project="code-classification-super-cons-learn[AI Patterns]",
        name=f"{model_name.split('/')[-1]}-{config['max_seq_length']}",
        config=config
    )

def prepare_fold_data(texts, labels, train_idx, test_idx, max_pairs_per_class, batch_size):
    X_train = texts[train_idx].tolist()
    y_train = labels[train_idx].tolist()
    X_test = texts[test_idx].tolist()
    y_test = labels[test_idx].tolist()

    train_examples = build_positive_pairs(
        X_train, y_train, max_pairs_per_class=max_pairs_per_class
    )

    if len(train_examples) < 4:
        return None, None, None, None

    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True
    )
    
    return train_loader, X_train, y_train, X_test, y_test

# def train_model(model_name, max_seq_length, train_loader, epochs, warmup_steps, lr):
#     # Load model (remote code needed for Nomic)
#     model = SentenceTransformer(model_name, trust_remote_code=True)
#     model.max_seq_length = max_seq_length

#     train_loss = losses.MultipleNegativesRankingLoss(model)

#     # Train
#     model.fit(
#         train_objectives=[(train_loader, train_loss)],
#         epochs=epochs,
#         warmup_steps=warmup_steps,
#         optimizer_params={"lr": lr},
#         show_progress_bar=True,
#         use_amp=True,
#     )
#     return model

def train_model(model_name, max_seq_length, train_loader, epochs, warmup_steps, lr):
    # Build SentenceTransformer from HF backbone
    word_emb = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling = models.Pooling(
        word_emb.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )

    model = SentenceTransformer(modules=[word_emb, pooling])

    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
        use_amp=True,
    )
    return model

def evaluate_fold(model, X_train, y_train, X_test, y_test):
    # Encode embeddings
    train_emb = model.encode(
        X_train, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False
    )
    test_emb = model.encode(
        X_test, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False
    )

    centroids = build_centroids(train_emb, y_train)
    y_pred = predict_centroid(test_emb, centroids)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    return acc, f1, y_pred, test_emb

def save_fold_embeddings(fold, X_test, y_test, y_pred, test_emb):
    # Convert embeddings to dataframe efficiently
    emb_np = test_emb.detach().cpu().numpy()
    df_meta = pd.DataFrame({
        "fold": fold,
        "label": y_test,
        "pred": y_pred
    })
    df_emb = pd.DataFrame(
        emb_np,
        columns=[f"emb_{i}" for i in range(emb_np.shape[1])]
    )
    df_fold = pd.concat([df_meta, df_emb], axis=1)
    return df_fold

def log_results(fold_acc, fold_f1, all_true, all_pred, class_names):
    print("\n" + "#" * 80)
    print("FINAL 5-FOLD RESULTS")
    print("#" * 80)
    
    if fold_acc:
        print(f"Avg Accuracy : {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
        print(f"Avg Macro F1 : {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")
        
        report_dict = classification_report(all_true, all_pred, target_names=class_names, digits=4, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "class"})

        wandb.log({"final_classification_report": wandb.Table(dataframe=df_report)})
        
        wandb.log({
            "final_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_true,
                preds=all_pred,
                class_names=class_names
            )
        })
    
    wandb.finish()

# =========================
# Main: 5-fold CV fine-tuning and evaluation
# =========================
def run_5fold_cv(
    texts,
    labels,
    class_names,
    model_name="nomic-ai/nomic-embed-text-v1",
    num_folds=5,
    epochs=1,
    batch_size=4,
    lr=2e-5,
    warmup_steps=10,
    max_pairs_per_class=200,
    max_seq_length=256,
    seed=42,
    save_dir="saved_test_embeddings",
    save_per_fold=False, 
):
    set_seed(seed)
    
    # Initialize wandb
    config = {
        "model_name": model_name,
        "num_folds": num_folds,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "max_pairs_per_class": max_pairs_per_class,
        "max_seq_length": max_seq_length,
        "seed": seed
    }
    init_wandb(model_name, config)

    texts = np.array(list(texts), dtype=object)
    labels = np.array(list(labels), dtype=int)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    fold_acc, fold_f1 = [], []
    all_test_rows = []
    
    # Aggregate true/preds for final report
    all_true = []
    all_pred = []

    os.makedirs(save_dir, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), start=1):
        print("\n" + "=" * 80)
        print(f"FOLD {fold}/{num_folds}")
        print("=" * 80)
        
        train_loader, X_train, y_train, X_test, y_test = prepare_fold_data(
            texts, labels, train_idx, test_idx, max_pairs_per_class, batch_size
        )

        if train_loader is None:
            print("Not enough same-class pairs to train. Skipping fold.")
            continue

        model = train_model(model_name, max_seq_length, train_loader, epochs, warmup_steps, lr)
        
        acc, f1, y_pred, test_emb = evaluate_fold(model, X_train, y_train, X_test, y_test)

        all_true.extend(y_test)
        all_pred.extend(y_pred)

        fold_acc.append(acc)
        fold_f1.append(f1)

        print(f"\nFold {fold} Accuracy   : {acc:.4f}")
        print(f"Fold {fold} Macro F1   : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))

        df_fold = save_fold_embeddings(fold, X_test, y_test, y_pred, test_emb)
        all_test_rows.append(df_fold)

        # cleanup
        del model
        torch.cuda.empty_cache()

    # Save all folds together
    if all_test_rows:
        all_test_df = pd.concat(all_test_rows, ignore_index=True).copy()
        out_all_path = os.path.join(save_dir, "all_folds_test_embeddings.csv")
        all_test_df.to_csv(out_all_path, index=False)
        print(f"\nSaved ALL folds test embeddings -> {out_all_path}")
        
    log_results(fold_acc, fold_f1, all_true, all_pred, class_names)

def main():
    dataset_path = './data/labeled_code_data.csv'
    
    try:
        dataset, le = load_and_preprocess_data(dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    texts = dataset["code"]
    labels = dataset["label_enc"]
    class_names = list(le.classes_)
    
    os.environ["WANDB_SILENT"] = "true"

    run_5fold_cv(
        texts=texts,
        labels=labels,
        class_names=class_names,
        epochs=3,
        batch_size=16,
        lr=2e-5,
        warmup_steps=10,
        max_pairs_per_class=400,
        max_seq_length=768,
        seed=42,
        model_name='microsoft/codebert-base'
    )

if __name__ == "__main__":
    main()

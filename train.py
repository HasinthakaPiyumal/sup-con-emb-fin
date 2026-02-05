"""Two-phase 5-fold CV: Phase 1 train emb + save test embeddings; Phase 2 evaluate with centroid + KNN."""
import os
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import models as sbert_models

# --- Config ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Data ---
def load_and_preprocess_data(filepath, min_samples_per_label=20):
    path = filepath if os.path.exists(filepath) else os.path.basename(filepath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    df = pd.read_csv(path)
    valid = df["label"].value_counts()
    valid = valid[valid >= min_samples_per_label].index
    df = df[df["label"].isin(valid)].reset_index(drop=True)
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    return df, le


def build_positive_pairs(texts, labels, max_pairs_per_class=None):
    label_to_idxs = {}
    for i, y in enumerate(labels):
        label_to_idxs.setdefault(y, []).append(i)
    examples = []
    for y, idxs in label_to_idxs.items():
        if len(idxs) < 2:
            continue
        pairs = [(idxs[i], idxs[j]) for i in range(len(idxs)) for j in range(i + 1, len(idxs))]
        if max_pairs_per_class and len(pairs) > max_pairs_per_class:
            pairs = random.sample(pairs, max_pairs_per_class)
        for a, b in pairs:
            examples.append(InputExample(texts=[str(texts[a]), str(texts[b])]))
    random.shuffle(examples)
    return examples


def prepare_fold_data(texts, labels, train_idx, test_idx, max_pairs_per_class):
    X_train, y_train = texts[train_idx].tolist(), labels[train_idx].tolist()
    X_test, y_test = texts[test_idx].tolist(), labels[test_idx].tolist()
    train_examples = build_positive_pairs(X_train, y_train, max_pairs_per_class=max_pairs_per_class)
    if len(train_examples) < 4:
        return None, None, None, None
    return train_examples, X_train, y_train, X_test, y_test


# --- Embedding-space classifiers ---
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def build_centroids(embeddings, labels):
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu()
    label_to_vecs = {}
    for e, y in zip(embeddings, labels):
        label_to_vecs.setdefault(y, []).append(e.cpu() if isinstance(e, torch.Tensor) else e)
    return {y: F.normalize(torch.stack(vecs).mean(dim=0), dim=-1) for y, vecs in label_to_vecs.items()}


def predict_centroid(embeddings, centroids):
    classes = sorted(centroids.keys())
    if not classes:
        return []
    C = torch.stack([centroids[c] for c in classes], dim=0).cpu()
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings).float()
    emb = embeddings.cpu() if embeddings.is_cuda else embeddings
    pred_idx = (emb @ C.T).argmax(dim=1).numpy()
    return [classes[i] for i in pred_idx]


def train_and_classify_knn(X_train, y_train, X_test, y_test):
    X_train, X_test = _to_numpy(X_train), _to_numpy(X_test)
    knn = KNeighborsClassifier(n_neighbors=15, metric="cosine", n_jobs=-1)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)


# --- Embedding model ---
def _add_dense_head(model, out_dim=8, activation=torch.nn.Tanh):
    """Append a single linear + activation projection to out_dim. Trained end-to-end with the rest."""
    in_dim = model.get_sentence_embedding_dimension()
    if in_dim is None or in_dim == out_dim:
        return model
    dense = sbert_models.Dense(
        in_features=in_dim,
        out_features=out_dim,
        bias=True,
        activation_function=activation() if activation else None,
    )
    model.add_module("dense", dense)
    return model


def train_model(model_name, max_seq_length, train_examples, batch_size, epochs, warmup_steps, lr, dense_dim=8):
    clear_memory()
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    if dense_dim and dense_dim > 0:
        model = _add_dense_head(model, out_dim=dense_dim)
    use_bf16 = torch.cuda.is_available()
    if use_bf16:
        model = model.to(torch.bfloat16)
    m = getattr(model, "_first_module", lambda: None)()
    if m and hasattr(m, "auto_model"):
        am = m.auto_model
        if hasattr(am, "gradient_checkpointing_enable"):
            am.gradient_checkpointing_enable()
        if getattr(am, "config", None):
            am.config.use_cache = False
            if hasattr(am.config, "attn_implementation"):
                am.config.attn_implementation = "sdpa"
    train_loader = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size, drop_last=False, collate_fn=model.smart_batching_collate
    )
    loss = losses.MultipleNegativesRankingLoss(model)
    opt_cls, opt_kw = torch.optim.AdamW, {"lr": lr, "weight_decay": 0.01}
    try:
        import bitsandbytes as bnb
        opt_cls = bnb.optim.AdamW8bit
    except Exception:
        pass
    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_class=opt_cls,
        optimizer_params=opt_kw,
        show_progress_bar=True,
        use_amp=not use_bf16,
    )
    return model


@torch.no_grad()
def encode_in_batches(model, texts, batch_size=32, normalize=True):
    out = []
    for i in range(0, len(texts), batch_size):
        emb = model.encode(
            texts[i : i + batch_size],
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        out.append(emb.cpu())
        if (i // batch_size) % 10 == 0:
            clear_memory()
    return torch.cat(out, dim=0)


# --- IO ---
def save_fold_embeddings(fold, y_test, test_emb):
    emb = _to_numpy(test_emb)
    meta = pd.DataFrame({"fold": fold, "label": y_test})
    feats = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    return pd.concat([meta, feats], axis=1)


def load_saved_embeddings(save_dir):
    path = os.path.join(save_dir, "all_folds_test_embeddings.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    return df[emb_cols].values.astype(np.float32), df["label"].values.astype(int)


# --- Evaluation ---
def _report_method(name, fold_acc, fold_f1, all_true, all_pred, class_names, wandb_key):
    if not fold_acc:
        return
    print(f"\n{'#' * 80}\n{name}\n{'#' * 80}")
    print(f"Avg Accuracy : {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
    print(f"Avg Macro F1 : {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")
    print(classification_report(all_true, all_pred, target_names=class_names, digits=4))
    report = classification_report(all_true, all_pred, target_names=class_names, digits=4, output_dict=True)
    wandb.log({f"{wandb_key}_report": wandb.Table(dataframe=pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "class"}))})
    wandb.log({f"{wandb_key}_confusion": wandb.plot.confusion_matrix(probs=None, y_true=all_true, preds=all_pred, class_names=class_names)})


def evaluate_saved_embeddings_5fold(save_dir, class_names, num_folds=5, seed=42):
    X, y = load_saved_embeddings(save_dir)
    if X is None:
        print(f"Phase 2 skipped: no embeddings in {save_dir}")
        return
    set_seed(seed)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    acc_c, f1_c, acc_k, f1_k = [], [], [], []
    all_true, pred_centroid, pred_knn = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        # Centroid
        centroids = build_centroids(torch.from_numpy(X_tr).float(), y_tr)
        pc = predict_centroid(torch.from_numpy(X_te).float(), centroids)
        acc_c.append(accuracy_score(y_te, pc))
        f1_c.append(f1_score(y_te, pc, average="macro"))
        all_true.extend(y_te)
        pred_centroid.extend(pc)
        # KNN
        pk = train_and_classify_knn(X_tr, y_tr, X_te, y_te)
        acc_k.append(accuracy_score(y_te, pk))
        f1_k.append(f1_score(y_te, pk, average="macro"))
        pred_knn.extend(pk)
        print(f"Phase 2 fold {fold}: Centroid Acc {acc_c[-1]:.4f} F1 {f1_c[-1]:.4f} | KNN Acc {acc_k[-1]:.4f} F1 {f1_k[-1]:.4f}")

    _report_method("PHASE 2 - CENTROID", acc_c, f1_c, all_true, pred_centroid, class_names, "phase2_centroid")
    _report_method("PHASE 2 - KNN", acc_k, f1_k, all_true, pred_knn, class_names, "phase2_knn")


# --- Pipeline ---
def init_wandb(model_name, config):
    wandb.init(
        project="code-classification-super-cons-learn[AI Patterns]",
        name=f"{model_name.split('/')[-1]}-{config['max_seq_length']}",
        config=config,
    )


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
    max_pairs_per_class=100,
    max_seq_length=256,
    seed=42,
    save_dir="saved_test_embeddings",
    dense_dim=8,
):
    set_seed(seed)
    config_keys = ("model_name", "num_folds", "epochs", "batch_size", "lr", "warmup_steps", "max_pairs_per_class", "max_seq_length", "seed", "dense_dim")
    config = {k: v for k, v in locals().items() if k in config_keys}
    init_wandb(model_name, config)
    texts = np.array(list(texts), dtype=object)
    labels = np.array(list(labels), dtype=int)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    os.makedirs(save_dir, exist_ok=True)
    all_rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), start=1):
        print(f"\n{'=' * 80}\nPHASE 1 FOLD {fold}/{num_folds}\n{'=' * 80}")
        clear_memory()
        prep = prepare_fold_data(texts, labels, train_idx, test_idx, max_pairs_per_class)
        if prep[0] is None:
            print("Skipping fold: not enough pairs.")
            continue
        train_examples, X_train, y_train, X_test, y_test = prep
        model = train_model(model_name, max_seq_length, train_examples, batch_size, epochs, warmup_steps, lr, dense_dim=dense_dim)
        clear_memory()
        model.eval()
        train_emb = encode_in_batches(model, X_train, batch_size=32)
        test_emb = encode_in_batches(model, X_test, batch_size=32)
        centroids = build_centroids(train_emb, y_train)
        pred_c = predict_centroid(test_emb, centroids)
        acc_c = accuracy_score(y_test, pred_c)
        f1_c = f1_score(y_test, pred_c, average="macro")
        print(f"Phase 1 fold {fold} centroid: Acc {acc_c:.4f}, Macro F1 {f1_c:.4f}")
        wandb.log({f"phase1_fold{fold}_centroid_acc": acc_c, f"phase1_fold{fold}_centroid_f1": f1_c})
        del train_emb, centroids, pred_c
        all_rows.append(save_fold_embeddings(fold, y_test, test_emb))
        del model, train_examples, test_emb, X_train, y_train, X_test, y_test
        clear_memory()

    if all_rows:
        out_path = os.path.join(save_dir, "all_folds_test_embeddings.csv")
        pd.concat(all_rows, ignore_index=True).to_csv(out_path, index=False)
        print(f"Saved embeddings -> {out_path}")
    evaluate_saved_embeddings_5fold(save_dir, class_names, num_folds=num_folds, seed=seed)
    wandb.finish()


def main():
    dataset_path = "./data/labeled_verified_data.csv"
    try:
        dataset, le = load_and_preprocess_data(dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    os.environ["WANDB_SILENT"] = "true"
    run_5fold_cv(
        texts=dataset["code_summary"],
        labels=dataset["label_enc"],
        class_names=list(le.classes_),
        epochs=3,
        batch_size=32,
        lr=2e-5,
        warmup_steps=10,
        max_pairs_per_class=1000,
        max_seq_length=768,
        seed=42,
        model_name="google-bert/bert-base-uncased",
        dense_dim=8,
    )


if __name__ == "__main__":
    main()

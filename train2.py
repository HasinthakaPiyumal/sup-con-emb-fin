import json
import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from datasets import Dataset
from huggingface_hub import login
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import torch.nn.functional as F
import numpy as np
import gc

# Configuration
PROCESSED_DATA_FILE = "data/processed_data_.jsonl"
MODEL_NAME = "google-bert/bert-base-uncased"
OUTPUT_DIR = "fine_tuned_ballerina_coderank"
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
HF_TOKEN = ""
HUB_MODEL_ID = "fine_tuned_ballerina_coderank"
NUM_FOLDS = 5
SEED = 42

# Classification settings
KNN_NEIGHBORS = 15  # Number of neighbors for KNN
CLASSIFICATION_METHOD = "both"  # Options: "centroid", "knn", "both"

def load_data(file_path):
    """Loads processed data and converts to format for MultipleNegativesRankingLoss."""
    examples = []
    label = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            query = item['query']
            positive = item['positive']
            negatives = item['negatives']
            
            # Format: [query, positive, neg1, neg2, ...]
            texts = [query, positive] + negatives
            examples.append(texts)
            label.append(item['label'])
            
    return examples,label

def train_model(raw_examples):
    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print(f"Loaded {len(raw_examples)} training triplets/tuples.")

    # Convert to HuggingFace Dataset
    # SentenceTransformerTrainer expects a Dataset object with specific columns or just returning lists
    # For MultipleNegativesRankingLoss, we can use a simple dictionary structure where we return "text" or similar, 
    # but typically it works best if we format it as a dataset where each column is a sentence.
    # However, since the number of negatives might vary (though my preprocess made it fixed ideally), 
    # let's assume fixed number of negatives from preprocess (which was top_k=3).
    
    # Let's verify consistency of length
    if not raw_examples:
        print("No data found!")
        return

    num_cols = len(raw_examples[0])
    # Column names: anchor, positive, negative_1, negative_2, ...
    column_names = ["anchor", "positive"] + [f"negative_{i+1}" for i in range(num_cols - 2)]
    
    # Transpose list of lists to dict of lists for Dataset creation
    data_dict = {name: [] for name in column_names}
    for ex in raw_examples:
        if len(ex) != num_cols:
            # Skip if length doesn't match (though it should)
            continue
        for i, text in enumerate(ex):
            data_dict[column_names[i]].append(text)
            
    train_dataset = Dataset.from_dict(data_dict)
    
    # 2. Load Model
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    
    # 3. Loss Function
    # MultipleNegativesRankingLoss is standard for (query, pos, negs)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 4. Training Arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        fp16=False, # Set to True if CUDA and supported
        bf16=False, # Set to True if supported (e.g. A100 or MPS on newer Macs?) - keep false for safety
        eval_strategy="no",
        save_strategy="epoch",
        logging_steps=10,
    )
    
    # 5. Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
    )
    
    print("Starting training...")
    trainer.train()
    return model

def build_centroids(embeddings, labels):
    """Build centroids on CPU to save GPU memory."""
    label_to_vecs = {}
    
    # Ensure labels are items if they are tensors
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    
    # Move embeddings to CPU for centroid computation
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu()
        
    for e, y in zip(embeddings, labels):
        if isinstance(e, torch.Tensor):
            e = e.cpu()
        label_to_vecs.setdefault(y, []).append(e)

    centroids = {}
    for y, vecs in label_to_vecs.items():
        c = torch.stack(vecs).mean(dim=0)
        centroids[y] = F.normalize(c, dim=-1)
    
    return centroids

def predict_centroid(embeddings, centroids):
    """Predict using centroids - all computation on CPU to save GPU memory."""
    classes = sorted(centroids.keys())
    c_list = [centroids[c] for c in classes]
    if not c_list:
        return []
    
    # Compute on CPU to save GPU memory
    C = torch.stack(c_list, dim=0).cpu()
    embeddings_cpu = embeddings.cpu() if embeddings.is_cuda else embeddings
    
    sims = embeddings_cpu @ C.T
    pred_idx = sims.argmax(dim=1).numpy()
    preds = [classes[i] for i in pred_idx]
    
    del C, embeddings_cpu, sims
    return preds


def _to_numpy(x):
    """Convert tensor or array to numpy float32 array."""
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def predict_knn(train_emb, train_labels, test_emb, n_neighbors=None):
    """Predict using KNN classifier with cosine similarity."""
    if n_neighbors is None:
        n_neighbors = KNN_NEIGHBORS
    
    X_train = _to_numpy(train_emb)
    X_test = _to_numpy(test_emb)
    
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric="cosine",
        n_jobs=-1
    )
    knn.fit(X_train, train_labels)
    return knn.predict(X_test)


def evaluate_model(model, train_texts, train_labels, test_texts, test_labels, method="both"):
    """
    Evaluate model using centroid and/or KNN classification.
    
    Args:
        method: "centroid", "knn", or "both"
    
    Returns:
        Dictionary with predictions and reports for each method.
    """
    train_emb = model.encode(train_texts, convert_to_tensor=True, show_progress_bar=True)
    test_emb = model.encode(test_texts, convert_to_tensor=True)
    
    results = {}
    
    if method in ("centroid", "both"):
        centroids = build_centroids(train_emb, train_labels)
        preds_centroid = predict_centroid(test_emb, centroids)
        report_centroid = classification_report(test_labels, preds_centroid)
        results["centroid"] = {
            "preds": preds_centroid,
            "report": report_centroid,
            "accuracy": accuracy_score(test_labels, preds_centroid),
            "f1": f1_score(test_labels, preds_centroid, average='weighted')
        }
    
    if method in ("knn", "both"):
        preds_knn = predict_knn(train_emb, train_labels, test_emb)
        report_knn = classification_report(test_labels, preds_knn)
        results["knn"] = {
            "preds": preds_knn,
            "report": report_knn,
            "accuracy": accuracy_score(test_labels, preds_knn),
            "f1": f1_score(test_labels, preds_knn, average='weighted')
        }
    
    # Cleanup
    del train_emb, test_emb
    
    return results, test_labels

    

def run_5fold_cv(raw_examples, labels, method="both"):
    """
    Run 5-fold cross-validation with centroid and/or KNN classification.
    
    Args:
        raw_examples: List of training examples.
        labels: List of labels.
        method: Classification method - "centroid", "knn", or "both".
    """
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Metrics storage for each method
    metrics = {
        "centroid": {"accuracies": [], "f1s": []},
        "knn": {"accuracies": [], "f1s": []}
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(raw_examples, labels), start=1):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold}/{NUM_FOLDS}")
        print(f"{'=' * 60}")

        train_examples = [raw_examples[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_examples = [raw_examples[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        model = train_model(train_examples)

        # Use anchor texts (queries) for evaluation
        train_texts = [ex[0] for ex in train_examples]
        test_texts = [ex[0] for ex in test_examples]

        results, test_labels = evaluate_model(
            model, train_texts, train_labels, test_texts, test_labels, method=method
        )
        
        # Print and store results for each method
        for method_name, result in results.items():
            print(f"\n--- {method_name.upper()} ---")
            print(result["report"])
            print(f"Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}")
            metrics[method_name]["accuracies"].append(result["accuracy"])
            metrics[method_name]["f1s"].append(result["f1"])

        del model, train_examples, train_labels, test_examples, test_labels
        torch.cuda.empty_cache()
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()

    # Print summary for each method
    print(f"\n{'#' * 60}")
    print("FINAL RESULTS")
    print(f"{'#' * 60}")
    
    for method_name in ["centroid", "knn"]:
        if metrics[method_name]["accuracies"]:
            acc = metrics[method_name]["accuracies"]
            f1s = metrics[method_name]["f1s"]
            print(f"\n{method_name.upper()}:")
            print(f"  Average Accuracy: {np.mean(acc):.4f} ± {np.std(acc):.4f}")
            print(f"  Average F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    


def main():
    print(f"Loading data from {PROCESSED_DATA_FILE}...")
    raw_examples, labels = load_data(PROCESSED_DATA_FILE)
    
    print(f"Classification method: {CLASSIFICATION_METHOD}")
    if CLASSIFICATION_METHOD in ("knn", "both"):
        print(f"KNN neighbors: {KNN_NEIGHBORS}")
    
    run_5fold_cv(raw_examples, labels, method=CLASSIFICATION_METHOD)
    
    # print(f"Pushing model to Hugging Face Hub: {HUB_MODEL_ID}...")
    # login(token=HF_TOKEN)
    # model.push_to_hub(HUB_MODEL_ID)
    # print("Done!")

if __name__ == "__main__":
    main()


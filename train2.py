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
import torch.nn.functional as F
import numpy as np
import gc

# Configuration
PROCESSED_DATA_FILE = "data/processed_data_.jsonl"
MODEL_NAME = "google-bert/bert-base-uncased"
OUTPUT_DIR = "fine_tuned_ballerina_coderank"
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
HF_TOKEN = ""
HUB_MODEL_ID = "fine_tuned_ballerina_coderank"
NUM_FOLDS = 5
SEED = 42

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

def evaluate_model(model,train_texts,train_labels, test_texts, test_labels):
    train_emb = model.encode(train_texts, convert_to_tensor=True, show_progress_bar=True)
    test_emb = model.encode(test_texts, convert_to_tensor=True)
    centroids = build_centroids(train_emb, train_labels)
    preds = predict_centroid(test_emb, centroids)
    report = classification_report(test_labels, preds)
    return preds, report, test_labels

    

def run_5fold_cv(raw_examples, labels):
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    all_accuracies = []
    all_f1s = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(raw_examples, labels), start=1):
        print(f"Fold {fold}")

        train_examples = [raw_examples[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_examples = [raw_examples[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        model = train_model(train_examples)

        # Use anchor texts (queries) for centroid evaluation to keep shapes consistent
        train_texts = [ex[0] for ex in train_examples]
        test_texts = [ex[0] for ex in test_examples]

        preds, report, test_labels = evaluate_model(model, train_texts, train_labels, test_texts, test_labels)
        print(report)
        accuracy = accuracy_score(test_labels, preds)
        f1 = f1_score(test_labels, preds, average='weighted')
        all_accuracies.append(accuracy)
        all_f1s.append(f1)

        del model, train_examples, train_labels, test_examples, test_labels
        torch.cuda.empty_cache()

        gc.collect()
        gc.collect()
        
        torch.cuda.empty_cache()

    print(f"Average Accuracy: {np.mean(all_accuracies):.4f}"
          f"\nAverage F1 Score: {np.mean(all_f1s):.4f}")
    


def main():
    print(f"Loading data from {PROCESSED_DATA_FILE}...")
    raw_examples,labels = load_data(PROCESSED_DATA_FILE)
    run_5fold_cv(raw_examples, labels)
    
    # print(f"Pushing model to Hugging Face Hub: {HUB_MODEL_ID}...")
    # login(token=HF_TOKEN)
    # model.push_to_hub(HUB_MODEL_ID)
    # print("Done!")

if __name__ == "__main__":
    main()


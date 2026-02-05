import json
import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from datasets import Dataset
from huggingface_hub import login

# Configuration
PROCESSED_DATA_FILE = "processed_data.jsonl"
MODEL_NAME = "nomic-ai/CodeRankEmbed"
OUTPUT_DIR = "fine_tuned_ballerina_coderank"
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
HF_TOKEN = ""
HUB_MODEL_ID = "fine_tuned_ballerina_coderank"

def load_data(file_path):
    """Loads processed data and converts to format for MultipleNegativesRankingLoss."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            query = item['query']
            positive = item['positive']
            negatives = item['negatives']
            
            # Format: [query, positive, neg1, neg2, ...]
            texts = [query, positive] + negatives
            examples.append(texts)
            
    return examples

def main():
    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # 1. Load Data
    print(f"Loading data from {PROCESSED_DATA_FILE}...")
    raw_examples = load_data(PROCESSED_DATA_FILE)
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
    
    print(f"Pushing model to Hugging Face Hub: {HUB_MODEL_ID}...")
    login(token=HF_TOKEN)
    model.push_to_hub(HUB_MODEL_ID)
    print("Done!")

if __name__ == "__main__":
    main()


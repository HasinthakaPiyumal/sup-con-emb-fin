import json
import re
import torch
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
import os
import pandas as pd
import random

# Configuration
DATA_FILE = "data/labeled_verified_data.csv"
MODEL_NAME = "google-bert/bert-base-uncased"
QUERY_PREFIX = "Represent this code summary for searching relevant code summaries"
TOP_K_FILTER = 2  # Consistency filtering: correct code must be in top-k
NUM_NEGATIVES = 3 # Number of hard negatives to mine per query
OUTPUT_FILE = "data/processed_data_.jsonl"


def main():
    print(f"Loading data from {DATA_FILE}...")
    
    dataset = pd.read_csv(DATA_FILE)
    counts = dataset["label"].value_counts()
    valid_labels = counts[counts >= 20].index
    dataset = dataset[dataset["label"].isin(valid_labels)].reset_index(drop=True)
    
    
    print(f"Found {len(dataset)} raw examples.")
    
    # 1. Extraction and Formatting
    extracted_data = []
    for _, item in dataset.iterrows():
        instruction = item['code_summary']
        class_label = item['label']
        other_samples_for_class = dataset[(dataset["label"] == class_label) & (dataset['code_summary'] != instruction)]
        answer = random.choice(other_samples_for_class['code_summary'].tolist()) if not other_samples_for_class.empty else ""
        code = answer
        if not instruction or not answer:
            continue

        if not code:
            continue
            
        # Format query with prefix
        query = f"{QUERY_PREFIX}: {instruction}"
        
        extracted_data.append({
            "query": query,
            "code": code,
            "original_instruction": instruction, # Keep for reference
            "file": item['file'] if 'file' in item else None,
            "label": item['label'] if 'label' in item else None

        })
        
    print(f"Extracted {len(extracted_data)} valid (query, code) pairs.")
    
    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # 2. Consistency Filtering & Hard Negative Mining
    print(f"Loading model {MODEL_NAME} for filtering and mining...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    
    # Encode all queries and codes
    queries = [d['query'] for d in extracted_data]
    codes = [d['code'] for d in extracted_data]
    
    print("Encoding queries...")
    query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
    
    print("Encoding codes...")
    # Note: CodeRankEmbed might require specific handling for code? 
    # Usually just passing text is fine for bi-encoders unless specified otherwise.
    # The usage example showed `model.encode(codes)` directly.
    code_embeddings = model.encode(codes, convert_to_tensor=True, show_progress_bar=True)
    
    print("Computing similarity matrix...")
    # Compute cosine similarity between all queries and all codes
    cos_scores = util.cos_sim(query_embeddings, code_embeddings)
    
    final_dataset = []
    
    print("Applying consistency filtering and mining hard negatives...")
    for i in range(len(extracted_data)):
        scores = cos_scores[i]
        
        # Get top results for this query
        # We need top_k + num_negatives roughly, but let's just get enough
        top_results = torch.topk(scores, k=max(TOP_K_FILTER + 5, NUM_NEGATIVES + 5))
        
        top_indices = top_results.indices.tolist()
        
        # Check consistency: is the correct code (index i) within top_k?
        # top_indices contains indices of codes. 
        # Ideally, top_indices[0] should be i, or at least within top_k.
        
        is_consistent = True
        if i in top_indices[:TOP_K_FILTER]:
            is_consistent = True
            
        if not is_consistent:
            continue
            
        # Hard Negative Mining
        # Find negatives: high score but not the correct index (i)
        negatives = []
        for idx in top_indices:
            if idx == i:
                continue # Skip positive
            negatives.append(extracted_data[idx]['code'])
            if len(negatives) >= NUM_NEGATIVES:
                break
                
        # Construct training example
        # Structure: query, positive, negative1, negative2, ...
        example = {
            "query": extracted_data[i]['query'],
            "positive": extracted_data[i]['code'],
            "file": extracted_data[i]['file'],
            "label": extracted_data[i]['label'],
            "negatives": negatives
        }
        final_dataset.append(example)
        
    print(f"Retained {len(final_dataset)} examples after filtering.")
    
    # Save to JSONL
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for item in final_dataset:
            f.write(json.dumps(item) + "\n")
            
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()


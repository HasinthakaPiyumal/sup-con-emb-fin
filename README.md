# Supervised Contrastive Code Embedding Fine-Tuning (`sup-con-emb-fin`)

Supervised contrastive fine-tuning pipeline for code embedding models (e.g., `BAAI/bge-code-v1`) designed for domain-specific code pattern classification. 

The pipeline uses **2-Phase 5-Fold Cross-Validation** with **Stratified Grouping** by unique code files to eliminate data leakage, paired with Hard Negative Mining, non-parametric evaluation (Nearest Centroid & KNN), and automated logging to Weights & Biases (WandB).

---

## 🌟 Key Features

- **Data-Leakage-Free Validation**: Uses `StratifiedGroupKFold` grouped by unique code file IDs so all augmented summaries of a code file remain strictly in either the training fold or the testing fold.
- **Hard Negative Mining**: Mines hard negatives using cosine similarity on base model embeddings to construct challenging positive/negative tuples.
- **Configurable Loss Functions**: Supports `contrastive` (ContrastiveLoss), `mnrl` (MultipleNegativesRankingLoss), and `triplet` (TripletLoss).
- **Two-Phase Evaluation**:
  - **Phase 1**: Trains fold models and generates out-of-fold (OOF) embeddings.
  - **Phase 2**: Evaluates out-of-fold embeddings using **Nearest Centroid** and **K-Nearest Neighbors (KNN)** classifiers.
- **WandB Integration**: Logs interactive confusion matrices, raw confusion count tables, classification reports, and scalar metrics directly to Weights & Biases.
- **Hugging Face Hub Export**: Includes scripts to retrain on the full dataset and push models directly to Hugging Face Hub.

---

## 📁 Repository Structure

```text
├── data/
│   ├── feb-10-2026-community-descriptions-concated.csv   # Main augmented dataset
│   └── labeled_verified_data.csv                        # Base file metadata dataset
├── src/
│   ├── config.py                                        # Seeds and memory cleanup
│   ├── data.py                                          # Data preprocessing & pair/triplet generators
│   ├── classifiers.py                                   # Nearest Centroid & KNN implementation
│   ├── evaluation.py                                    # Metrics report & WandB confusion matrix uploader
│   ├── io_utils.py                                      # Saving & loading embeddings CSVs
│   ├── model.py                                         # SentenceTransformer training loop
│   └── pipeline.py                                      # 5-fold CV & full training pipeline orchestration
├── install.sh                                           # Quick installation script
├── preprocess.py                                        # Dataset preprocessing utility
├── requirements.txt                                     # PyTorch & ML dependencies
├── train.py                                             # Main entry point for cross-validation & training
├── upload_model.py                                      # Export model to Hugging Face Hub
└── TRAINING_PIPELINE.md                                 # Technical architecture documentation
```

---

## 🚀 Environment Setup & Installation

### Option A: Local Setup (Linux / Windows)

```bash
# 1. Clone repository
git clone https://github.com/HasinthakaPiyumal/sup-con-emb-fin.git
cd sup-con-emb-fin

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate          # Windows PowerShell

# 3. Install PyTorch with CUDA support (e.g., CUDA 12.1/12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install project requirements
pip install -r requirements.txt
```

### Option B: RunPod / Cloud GPU Setup

```bash
# Move to persistent workspace disk
cd /workspace

# Clone repository
git clone https://github.com/HasinthakaPiyumal/sup-con-emb-fin.git
cd sup-con-emb-fin

# Create virtual environment & install requirements
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Verify GPU availability in PyTorch
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 💻 How to Run the Training & Evaluation Pipeline

### 1. Configure Options in `train.py`

Open `train.py` to adjust settings:

```python
# Enable raw baseline without fine-tuning:
RUN_WITHOUT_FINETUNING = True  

# Or configure fine-tuning settings:
RUN_WITHOUT_FINETUNING = False
MODEL_NAME = "BAAI/bge-code-v1"
LOSS_TYPE = "mnrl"              # Options: "contrastive", "mnrl", "triplet"
EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
USE_HARD_NEGATIVES = True
NUM_HARD_NEGATIVES = 10
MIN_SAMPLES_PER_LABEL = 120
```

---

### 2. Execution Commands

#### A. Interactive Foreground Run
```bash
python -u train.py
```

#### B. Foreground Run with Terminal Logging (`tee`)
```bash
python -u train.py 2>&1 | tee output.log
```

#### C. Background Run using `nohup` (Recommended for Cloud / RunPod SSH)
```bash
nohup python -u train.py > output.log 2>&1 &
```

#### D. Background Run using `tmux`
```bash
# Start a new named session
tmux new -s train

# Inside tmux, run:
python -u train.py 2>&1 | tee output.log

# Detach: Press Ctrl+B, then press D
# Reattach anytime:
tmux attach -t train
```

#### E. Background Run using `screen`
```bash
# Start detached screen session
screen -dmS train bash -c "python -u train.py 2>&1 | tee output.log"

# Reattach:
screen -r train
```

---

## 📊 Process & Log Monitoring Commands

```bash
# Watch real-time log output
tail -f output.log

# Monitor GPU utilization & memory
watch -n 1 nvidia-smi

# Check if Python training process is active
ps aux | grep python

# Kill a running training job if needed
pkill -f train.py
```

---

## 📤 Upload Model to Hugging Face Hub

After completing training, push your saved model weights (`saved_models/`) to Hugging Face:

```bash
# Login to Hugging Face CLI (one-time setup)
huggingface-cli login

# Run upload script
python upload_model.py \
  --model_path saved_models/bge-code-v1-ai-pattern-tuned \
  --hub_id username/bge-code-v1-ai-pattern-tuned
```

---

## 📈 WandB Dashboard Outputs

When training completes, your Weights & Biases dashboard will contain:
- **Interactive Confusion Matrix**: `phase2_centroid_confusion` and `phase2_knn_confusion`.
- **Confusion Matrix Data Table**: `phase2_centroid_confusion_matrix_table` and `phase2_knn_confusion_matrix_table` showing exact actual vs. predicted counts per class.
- **Classification Report Table**: Per-class Precision, Recall, Macro F1, and Accuracy.
- **Local Artifact Backups**: Saved in `saved_test_embeddings/confusion_matrix_*.csv` and `saved_test_embeddings/classification_report_*.csv`.

---

## 📄 License

Distributed under the MIT License.

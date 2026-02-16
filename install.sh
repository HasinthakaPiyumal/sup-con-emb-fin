#!/bin/bash
set -e

echo "🔹 Upgrading pip..."
python3 -m pip install --upgrade pip

echo "🔹 Installing requirements..."
pip install -r requirements.txt

echo "🔹 Logging into Weights & Biases..."

if [ -z "$RUNPOD_SECRET_wandb_hasinthaka" ]; then
  echo "❌ WANDB_API_KEY not found!"
  echo "👉 Make sure RunPod secret name is exactly: wandb-hasinthaka"
  exit 1
fi

pip install -U "huggingface_hub[cli]"
apt install screen -y

wandb login "$RUNPOD_SECRET_wandb_hasinthaka" --relogin
echo "🔹 Logging into Hugging Face..."

huggingface-cli login


echo "✅ Setup completed successfully!"

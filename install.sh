#!/bin/bash
set -e

echo "🔹 Upgrading pip..."
python3 -m pip install --upgrade pip

echo "🔹 Installing requirements..."
pip install -r requirements.txt

echo "🔹 Logging into Weights & Biases..."

if [ -z "$RUNPOD_SECRET_wandb-hasinthaka" ]; then
  echo "❌ WANDB_API_KEY not found!"
  echo "👉 Make sure RunPod secret name is exactly: wandb-hasinthaka"
  exit 1
fi

wandb login "$RUNPOD_SECRET_wandb-hasinthaka" --relogin

echo "✅ Setup completed successfully!"

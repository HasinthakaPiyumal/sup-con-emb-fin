#!/usr/bin/env python3
"""
Upload a locally saved SentenceTransformer model to Hugging Face Hub.

Usage:
    python upload_model.py --model_path saved_models/full-dataset-contrastive-hn10-ep3 --hub_id username/repo-name
"""

import argparse
import os
from typing import Optional
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi, login


def upload_model_to_hub(
    model_path: str,
    hub_model_id: str,
    hf_token: Optional[str] = None,
    commit_message: str = "Upload fine-tuned model",
    private: bool = False,
) -> None:
    """
    Upload a locally saved SentenceTransformer model to Hugging Face Hub.

    Args:
        model_path: Local path to the saved model directory.
        hub_model_id: Hugging Face Hub repo id (e.g. "username/repo-name").
        hf_token: Optional Hugging Face token for login (else use huggingface-cli login).
        commit_message: Commit message for the upload.
        private: Whether to create a private repo.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Login if token provided
    if hf_token:
        try:
            login(token=hf_token)
            print("Logged in to Hugging Face Hub using provided token.")
        except Exception as e:
            print(f"Warning: HF login with token failed: {e}")
            print("Trying to use existing login (huggingface-cli login)...")

    # Create repo if it doesn't exist (or use existing)
    api = HfApi()
    try:
        api.create_repo(
            repo_id=hub_model_id,
            exist_ok=True,
            repo_type="model",
            private=private,
        )
        print(f"Repository ready: {hub_model_id}")
    except Exception as e:
        print(f"Warning: Could not create/verify repo: {e}")
        print("Continuing anyway (repo may already exist)...")

    # Load and push model
    print(f"Loading model from: {model_path}")
    model = SentenceTransformer(model_path)

    print(f"Uploading model to Hugging Face Hub: {hub_model_id}")
    try:
        # push_to_hub internally calls create_repo without exist_ok, so we need to handle 409
        model.push_to_hub(
            repo_id=hub_model_id,
            commit_message=commit_message,
        )
        print(f"✓ Successfully uploaded model!")
        print(f"  Model available at: https://huggingface.co/{hub_model_id}")
    except Exception as e:
        # Check if it's a 409 Conflict (repo already exists)
        error_str = str(e)
        if "409" in error_str or "already created" in error_str.lower() or "Conflict" in error_str:
            print(f"Repository already exists. Attempting to upload files directly...")
            # Use HfApi to upload files directly instead
            try:
                api.upload_folder(
                    folder_path=model_path,
                    repo_id=hub_model_id,
                    repo_type="model",
                    commit_message=commit_message,
                )
                print(f"✓ Successfully uploaded model files!")
                print(f"  Model available at: https://huggingface.co/{hub_model_id}")
            except Exception as upload_error:
                print(f"✗ Error uploading files: {upload_error}")
                raise
        else:
            print(f"✗ Error uploading model: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload a locally saved SentenceTransformer model to Hugging Face Hub"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path to the saved model directory",
    )
    parser.add_argument(
        "--hub_id",
        type=str,
        required=True,
        help="Hugging Face Hub repo id (e.g. 'username/repo-name')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (optional, can use huggingface-cli login instead)",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload fine-tuned model",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )

    args = parser.parse_args()

    upload_model_to_hub(
        model_path=args.model_path,
        hub_model_id=args.hub_id,
        hf_token=args.token,
        commit_message=args.commit_message,
        private=args.private,
    )


if __name__ == "__main__":
    main()

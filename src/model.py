"""Embedding model training and inference."""

from typing import List, Optional, Type

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import models as sbert_models

from .config import clear_memory


def _add_dense_head(
    model: SentenceTransformer,
    out_dim: int = 8,
    activation: Optional[Type[torch.nn.Module]] = torch.nn.Tanh
) -> SentenceTransformer:
    """
    Append a dense projection layer to the model.
    
    Args:
        model: SentenceTransformer model.
        out_dim: Output dimension of dense layer.
        activation: Activation function class (or None).
    
    Returns:
        Model with dense head added.
    """
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


def _configure_model_for_training(model: SentenceTransformer) -> None:
    """Configure model internals for efficient training."""
    first_module = getattr(model, "_first_module", lambda: None)()
    if not first_module or not hasattr(first_module, "auto_model"):
        return
    
    auto_model = first_module.auto_model
    
    # Enable gradient checkpointing
    if hasattr(auto_model, "gradient_checkpointing_enable"):
        auto_model.gradient_checkpointing_enable()
    
    # Configure attention
    config = getattr(auto_model, "config", None)
    if config:
        config.use_cache = False
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "sdpa"


def _get_optimizer() -> tuple:
    """Get optimizer class and parameters, preferring 8-bit Adam if available."""
    opt_cls = torch.optim.AdamW
    opt_kwargs = {"lr": 2e-5, "weight_decay": 0.01}
    
    try:
        import bitsandbytes as bnb
        opt_cls = bnb.optim.AdamW8bit
    except ImportError:
        pass
    
    return opt_cls, opt_kwargs


def train_model(
    model_name: str,
    max_seq_length: int,
    train_examples: List[InputExample],
    batch_size: int,
    epochs: int,
    warmup_steps: int,
    lr: float,
    dense_dim: int = 8
) -> SentenceTransformer:
    """
    Train a SentenceTransformer model with contrastive loss.
    
    Args:
        model_name: HuggingFace model name or path.
        max_seq_length: Maximum sequence length.
        train_examples: List of InputExample pairs.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        warmup_steps: Learning rate warmup steps.
        lr: Learning rate.
        dense_dim: Output dimension for dense projection head.
    
    Returns:
        Trained SentenceTransformer model.
    """
    clear_memory()
    
    # Load and configure model
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    
    if dense_dim and dense_dim > 0:
        model = _add_dense_head(model, out_dim=dense_dim)
    
    # Use bfloat16 if available
    use_bf16 = torch.cuda.is_available()
    if use_bf16:
        model = model.to(torch.bfloat16)
    
    _configure_model_for_training(model)
    
    # Setup training
    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=model.smart_batching_collate
    )
    
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    opt_cls, opt_kwargs = _get_optimizer()
    opt_kwargs["lr"] = lr
    
    # Train
    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_class=opt_cls,
        optimizer_params=opt_kwargs,
        show_progress_bar=True,
        use_amp=not use_bf16,
    )
    
    return model


@torch.no_grad()
def encode_in_batches(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
    normalize: bool = True
) -> torch.Tensor:
    """
    Encode texts in batches to manage memory.
    
    Args:
        model: Trained SentenceTransformer.
        texts: List of texts to encode.
        batch_size: Encoding batch size.
        normalize: Whether to L2-normalize embeddings.
    
    Returns:
        Tensor of embeddings with shape (N, D).
    """
    outputs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        embeddings = model.encode(
            batch_texts,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        outputs.append(embeddings.cpu())
        
        # Periodically clear memory
        if (i // batch_size) % 10 == 0:
            clear_memory()
    
    return torch.cat(outputs, dim=0)

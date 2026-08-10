"""Embedding model training and inference."""

from typing import List, Optional, Type, Union

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import models as sbert_models

from .config import clear_memory
from .data import LossType


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


def _configure_model_for_training(model: SentenceTransformer, is_lora: bool = False) -> None:
    """Configure model internals for efficient training."""
    first_module = getattr(model, "_first_module", lambda: None)()
    if not first_module or not hasattr(first_module, "auto_model"):
        return
    
    auto_model = first_module.auto_model
    
    # Configure attention
    config = getattr(auto_model, "config", None)
    if config:
        config.use_cache = False
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "sdpa"

    # For LoRA PEFT: enable input require grads for autograd backward pass
    if is_lora:
        if hasattr(auto_model, "enable_input_require_grads"):
            auto_model.enable_input_require_grads()
        return

    # Enable gradient checkpointing for full fine-tuning
    if hasattr(auto_model, "gradient_checkpointing_enable"):
        auto_model.gradient_checkpointing_enable()


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


def get_loss_function(
    model: SentenceTransformer,
    loss_type: LossType,
    margin: float = 0.5,
) -> torch.nn.Module:
    """
    Get the appropriate loss function for the specified loss type.
    
    Args:
        model: SentenceTransformer model.
        loss_type: Type of loss function to use.
        margin: Margin for contrastive/triplet loss (default: 0.5).
    
    Returns:
        Loss function module.
    """
    if loss_type == LossType.CONTRASTIVE:
        # ContrastiveLoss: expects pairs with labels (0=dissimilar, 1=similar)
        return losses.ContrastiveLoss(model=model, margin=margin)
    
    elif loss_type == LossType.TRIPLET:
        # TripletLoss: expects (anchor, positive, negative) triplets
        return losses.TripletLoss(model=model, triplet_margin=margin)
    
    elif loss_type == LossType.MNRL:
        # MultipleNegativesRankingLoss: (anchor, positive, [negatives...])
        return losses.MultipleNegativesRankingLoss(model=model)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def apply_lora_to_model(
    model: SentenceTransformer,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> SentenceTransformer:
    """
    Apply LoRA (PEFT) adapters to the SentenceTransformer base backbone.
    
    Args:
        model: SentenceTransformer model.
        r: LoRA rank dimension.
        lora_alpha: LoRA scaling factor alpha.
        lora_dropout: LoRA dropout probability.
        target_modules: List of module names to apply LoRA to.

    Returns:
        SentenceTransformer model with LoRA adapters attached.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError(
            "PEFT library is required for LoRA training. Install via: pip install peft"
        )

    first_module = getattr(model, "_first_module", lambda: model[0])()
    if not hasattr(first_module, "auto_model"):
        print("  [LoRA Warning] Model backbone does not have auto_model attribute. LoRA skipped.")
        return model

    auto_model = first_module.auto_model

    # Extract all leaf layer module names
    all_module_names = set(name.split(".")[-1] for name, _ in auto_model.named_modules())

    # If user provided target_modules, filter only valid ones present in the model
    if target_modules:
        valid_targets = [t for t in target_modules if t in all_module_names]
        if not valid_targets:
            print(f"  [LoRA Warning] Target modules {target_modules} not found in model. Auto-detecting valid layers...")
            target_modules = None
        else:
            target_modules = valid_targets

    # Auto-detect target modules if not specified or invalid
    if not target_modules:
        if "q_proj" in all_module_names:
            target_modules = ["q_proj", "v_proj"]
        elif "query" in all_module_names:
            target_modules = ["query", "value"]
        else:
            target_modules = [t for t in ["q_proj", "v_proj", "query", "value", "k_proj", "o_proj"] if t in all_module_names]

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )

    peft_model = get_peft_model(auto_model, peft_config)
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    first_module.auto_model = peft_model
    print(f"  [LoRA Enabled] Configured PEFT LoRA (r={r}, alpha={lora_alpha}, dropout={lora_dropout}, targets={target_modules})")
    
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()
    elif hasattr(first_module.auto_model, "print_trainable_parameters"):
        first_module.auto_model.print_trainable_parameters()

    return model


def train_model(
    model_name: str,
    max_seq_length: int,
    train_examples: List[InputExample],
    batch_size: int,
    epochs: int,
    warmup_steps: int,
    lr: float,
    dense_dim: int = 8,
    loss_type: LossType = LossType.MNRL,
    loss_margin: float = 0.5,
    freeze_base_model: bool = False,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
) -> SentenceTransformer:
    """
    Train a SentenceTransformer model with configurable loss function and PEFT/LoRA.
    
    Args:
        model_name: HuggingFace model name or path.
        max_seq_length: Maximum sequence length.
        train_examples: List of InputExample pairs/triplets.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        warmup_steps: Learning rate warmup steps.
        lr: Learning rate.
        dense_dim: Output dimension for dense projection head.
        loss_type: Type of loss function to use.
        loss_margin: Margin for contrastive/triplet loss.
        freeze_base_model: If True, freeze base Transformer backbone parameters.
        use_lora: If True, attach PEFT LoRA adapters to the transformer backbone.
        lora_r: Rank for LoRA adaptation.
        lora_alpha: Alpha scaling factor for LoRA.
        lora_dropout: Dropout probability for LoRA layers.
        lora_target_modules: Target layer names for LoRA.
    
    Returns:
        Trained SentenceTransformer model.
    """
    clear_memory()
    
    # Load and configure model
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.max_seq_length = max_seq_length

    # Apply LoRA if requested
    if use_lora:
        model = apply_lora_to_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
    # Freeze base Transformer backbone if specified (when LoRA is off)
    elif freeze_base_model:
        first_module = getattr(model, "_first_module", lambda: model[0])()
        if hasattr(first_module, "auto_model"):
            for param in first_module.auto_model.parameters():
                param.requires_grad = False
            print("  [Freeze Base] Successfully froze Transformer backbone parameters.")

        # Ensure trainable parameters exist by attaching a projection head if needed
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            in_dim = model.get_sentence_embedding_dimension()
            out_dim = dense_dim if (dense_dim and dense_dim > 0) else in_dim
            dense_layer = sbert_models.Dense(
                in_features=in_dim,
                out_features=out_dim,
                bias=True,
                activation_function=torch.nn.Identity(),
            )
            model.add_module("dense_head", dense_layer)
            print(f"  [Freeze Base] Added trainable Dense head layer ({in_dim} -> {out_dim}).")
    elif dense_dim and dense_dim > 0:
        model = _add_dense_head(model, out_dim=dense_dim, activation=torch.nn.Linear)
    
    # Use bfloat16 if available
    use_bf16 = torch.cuda.is_available()
    if use_bf16:
        model = model.to(torch.bfloat16)
    
    _configure_model_for_training(model, is_lora=use_lora)
    
    # Setup training
    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=model.smart_batching_collate
    )
    
    # Get the appropriate loss function
    loss_fn = get_loss_function(model, loss_type, margin=loss_margin)
    print(f"  Using loss: {loss_type.value} (margin={loss_margin}) | Freeze Base: {freeze_base_model}")
    
    opt_cls, opt_kwargs = _get_optimizer()
    opt_kwargs["lr"] = lr
    
    # Filter optimizer parameters to only trainable ones
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("  Warning: No trainable parameters found! All layers are frozen.")
    
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
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return torch.empty(0, dim, dtype=torch.float32)

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

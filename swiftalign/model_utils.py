"""
swiftalign.model_utils
~~~~~~~~~~~~~~~~~~~~~~
Load a causal LM and tokenizer with optional LoRA / QLoRA adapters.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from .hardware import HardwareProfile

logger = logging.getLogger(__name__)


# ── Default LoRA hyperparameters ────────────────────────────────────────────
LORA_DEFAULTS = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)


def load_model_and_tokenizer(
    model_id: str,
    hw: HardwareProfile,
    use_lora: bool = False,
    use_qlora: bool = False,
    lora_config_overrides: Optional[dict] = None,
):
    """
    Load ``model_id`` from HuggingFace Hub.

    Parameters
    ----------
    model_id : str
        A valid HuggingFace model repository, e.g. ``Qwen/Qwen2.5-1.5B-Instruct``.
    hw : HardwareProfile
        Hardware profile returned by :func:`swiftalign.hardware.detect_hardware`.
    use_lora : bool
        Wrap the model with LoRA adapters (PEFT).
    use_qlora : bool
        Load in 4-bit (BitsAndBytes) and wrap with LoRA adapters.
        Implies ``use_lora=True``.
    lora_config_overrides : dict, optional
        Override keys in :data:`LORA_DEFAULTS`.

    Returns
    -------
    model : transformers.PreTrainedModel
    tokenizer : transformers.PreTrainedTokenizer
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(hw.torch_dtype, torch.float32)

    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # recommended for causal LM batching

    # ── Quantisation config (QLoRA path) ────────────────────────────────────
    bnb_config = None
    if use_qlora:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("QLoRA: 4-bit NF4 quantisation enabled")
        except Exception as exc:
            logger.warning("BitsAndBytes config failed (%s); falling back to full precision", exc)
            bnb_config = None

    # ── Model load ───────────────────────────────────────────────────────────
    logger.info("Loading model: %s  [dtype=%s, attn=%s]",
                model_id, hw.torch_dtype, hw.attn_implementation)

    load_kwargs = dict(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if hw.device == "cuda" else None,
    )

    # Inject attention implementation when not using quantisation
    # (BNB + flash-attention-2 can conflict on some builds)
    if bnb_config is None and hw.attn_implementation in ("flash_attention_2", "sdpa"):
        load_kwargs["attn_implementation"] = hw.attn_implementation

    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    # Move to device when not using device_map="auto"
    if hw.device != "cuda" and bnb_config is None:
        model = model.to(hw.device)

    # ── LoRA / QLoRA adapter ─────────────────────────────────────────────────
    if use_lora or use_qlora:
        model = _apply_lora(model, lora_config_overrides, use_qlora)

    model.config.use_cache = False  # required for gradient checkpointing

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info("Model loaded: %.1f M params total, %.1f M trainable", param_count, trainable)

    return model, tokenizer


def _apply_lora(model, overrides: Optional[dict], is_qlora: bool):
    """Attach a LoRA adapter via PEFT."""
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError(
            "peft is required for LoRA/QLoRA. Install with: pip install peft"
        ) from exc

    cfg = {**LORA_DEFAULTS, **(overrides or {})}
    lora_cfg = LoraConfig(**cfg)

    if is_qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model

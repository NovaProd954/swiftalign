"""
swiftalign.trainers.dpo
~~~~~~~~~~~~~~~~~~~~~~~
Thin wrapper around ``trl.DPOTrainer`` with SwiftAlign defaults.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..hardware import HardwareProfile

logger = logging.getLogger(__name__)


def run_dpo(
    model,
    tokenizer,
    dataset,
    hw: HardwareProfile,
    output_dir: str = "./output",
    epochs: int = 1,
    batch_size: int = 2,
    max_steps: int = -1,
    lr: float = 5e-5,
    seed: int = 42,
    dashboard=None,
) -> dict:
    """
    Run DPO training using ``trl.DPOTrainer``.

    Parameters
    ----------
    model, tokenizer
        Loaded model and tokenizer from :func:`swiftalign.model_utils.load_model_and_tokenizer`.
    dataset : datasets.Dataset
        Must contain columns: ``prompt``, ``chosen``, ``rejected``.
    hw : HardwareProfile
        Hardware profile for precision settings.
    output_dir : str
        Where to save the final checkpoint.
    epochs, batch_size, max_steps, lr, seed
        Standard training hyperparameters.
    dashboard : Dashboard, optional
        SwiftAlign dashboard for progress reporting.

    Returns
    -------
    dict
        Final training metrics.
    """
    try:
        from trl import DPOTrainer, DPOConfig
    except ImportError as exc:
        raise ImportError(
            "trl is required. Install with: pip install trl"
        ) from exc

    os.makedirs(output_dir, exist_ok=True)

    # ── Training arguments ───────────────────────────────────────────────────
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        max_steps=max_steps if max_steps > 0 else -1,
        learning_rate=lr,
        fp16=hw.fp16,
        bf16=hw.bf16,
        gradient_checkpointing=hw.recommend_gradient_checkpointing,
        gradient_accumulation_steps=_grad_accum(hw),
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        report_to="none",       # disable wandb / tensorboard by default
        seed=seed,
        beta=0.1,               # DPO KL penalty coefficient
        max_length=512,
        max_prompt_length=256,
        remove_unused_columns=False,
    )

    if dashboard:
        dashboard.section("DPO Training")
        dashboard.log(
            f"DPOConfig: epochs={epochs}, batch={batch_size}, "
            f"lr={lr}, beta=0.1, max_len=512"
        )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # when ref_model=None, TRL creates a frozen copy
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Patch the logging callback to pipe step metrics to the dashboard
    if dashboard:
        _patch_trainer_callback(trainer, dashboard)

    logger.info("Starting DPO training — %d examples", len(dataset))
    train_result = trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)

    metrics = train_result.metrics
    metrics["output_dir"] = output_dir
    return metrics


def _grad_accum(hw: HardwareProfile) -> int:
    """Return a sensible gradient accumulation factor based on VRAM."""
    if hw.gpu_mem_gb < 8:
        return 8
    if hw.gpu_mem_gb < 16:
        return 4
    return 2


def _patch_trainer_callback(trainer, dashboard):
    """
    Monkey-patch the trainer's log method to forward step metrics to the dashboard.
    This is a lightweight alternative to writing a full TrainerCallback.
    """
    original_log = trainer.log

    def _log(logs: dict):
        original_log(logs)
        loss_keys = [k for k in logs if "loss" in k.lower()]
        display = {k: round(logs[k], 4) for k in loss_keys if isinstance(logs.get(k), float)}
        if "epoch" in logs:
            display["epoch"] = round(float(logs["epoch"]), 2)
        if display:
            dashboard.update_progress(advance=0, metrics=display)

    trainer.log = _log

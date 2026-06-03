"""
swiftalign.trainers.grpo
~~~~~~~~~~~~~~~~~~~~~~~~
Thin wrapper around ``trl.GRPOTrainer`` with SwiftAlign reward functions.
"""

from __future__ import annotations

import logging
import os

from ..hardware import HardwareProfile
from ..rewards import combined_reward, REWARD_REGISTRY

logger = logging.getLogger(__name__)


def run_grpo(
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
    reward_fn_name: str = "combined",
    dashboard=None,
) -> dict:
    """
    Run GRPO training using ``trl.GRPOTrainer``.

    Parameters
    ----------
    model, tokenizer
        Loaded model and tokenizer.
    dataset : datasets.Dataset
        Must contain a ``prompt`` column.
    hw : HardwareProfile
        Hardware profile for precision settings.
    output_dir : str
        Where to save the final checkpoint.
    epochs, batch_size, max_steps, lr, seed
        Standard training hyperparameters.
    reward_fn_name : str
        Key in :data:`swiftalign.rewards.REWARD_REGISTRY`.
        Defaults to ``"combined"``.
    dashboard : Dashboard, optional
        SwiftAlign dashboard for progress reporting.

    Returns
    -------
    dict
        Final training metrics.
    """
    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError as exc:
        raise ImportError(
            "trl is required. Install with: pip install trl"
        ) from exc

    os.makedirs(output_dir, exist_ok=True)

    # ── Resolve reward function ───────────────────────────────────────────────
    reward_fn = REWARD_REGISTRY.get(reward_fn_name, combined_reward)
    logger.info("Using reward function: %s", reward_fn.__name__)

    # GRPO reward signature: fn(prompts, completions, **kwargs) -> list[float]
    def _reward_fn(prompts, completions, **kwargs):
        scores = reward_fn(prompts, completions, **kwargs)
        if dashboard:
            avg = sum(scores) / max(len(scores), 1)
            dashboard.update_progress(
                advance=0,
                metrics={"reward": round(avg, 4)},
            )
        return scores

    # ── Training arguments ───────────────────────────────────────────────────
    training_args = GRPOConfig(
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
        report_to="none",
        seed=seed,
        # GRPO-specific
        num_generations=4,          # rollouts per prompt
        max_new_tokens=128,
        temperature=0.9,
        top_p=0.95,
        kl_coef=0.05,
        cliprange=0.2,
        remove_unused_columns=False,
    )

    if dashboard:
        dashboard.section("GRPO Training")
        dashboard.log(
            f"GRPOConfig: epochs={epochs}, batch={batch_size}, "
            f"lr={lr}, reward_fn='{reward_fn_name}', "
            f"num_generations=4, max_new_tokens=128"
        )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_funcs=[_reward_fn],
    )

    if dashboard:
        _patch_trainer_callback(trainer, dashboard)

    logger.info("Starting GRPO training — %d examples", len(dataset))
    train_result = trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)

    metrics = train_result.metrics
    metrics["output_dir"] = output_dir
    return metrics


def _grad_accum(hw: HardwareProfile) -> int:
    if hw.gpu_mem_gb < 8:
        return 8
    if hw.gpu_mem_gb < 16:
        return 4
    return 2


def _patch_trainer_callback(trainer, dashboard):
    original_log = trainer.log

    def _log(logs: dict):
        original_log(logs)
        display = {}
        for k in ["loss", "reward", "kl", "epoch"]:
            if k in logs and isinstance(logs[k], (int, float)):
                display[k] = round(float(logs[k]), 4)
        if display:
            dashboard.update_progress(advance=0, metrics=display)

    trainer.log = _log

"""
swiftalign.runner
~~~~~~~~~~~~~~~~~
Orchestrates a full alignment training run:
  1. Detect hardware
  2. Load model + tokenizer
  3. Load dataset
  4. Run DPO or GRPO
  5. Report results
"""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)


def run(args) -> dict:
    """
    Execute a training run from parsed CLI arguments (or an argparse.Namespace-like object).

    Minimum required attributes on ``args``:
        method, model, lora, qlora, output_dir, epochs, batch_size,
        max_steps, lr, seed, no_dashboard
    """
    _setup_logging()

    from .hardware import detect_hardware
    from .model_utils import load_model_and_tokenizer
    from .data import get_dpo_dataset, get_grpo_dataset
    from .dashboard import Dashboard
    from .trainers import run_dpo, run_grpo

    dash = Dashboard(enabled=not getattr(args, "no_dashboard", False))

    # ── Banner ───────────────────────────────────────────────────────────────
    dash.banner()

    # ── Seed ─────────────────────────────────────────────────────────────────
    seed = getattr(args, "seed", 42)
    _set_seed(seed)

    # ── Hardware detection ────────────────────────────────────────────────────
    dash.section("Hardware Detection")
    hw = detect_hardware()
    for note in hw.extra_notes:
        dash.log(f"  {note}", style="dim cyan")
    dash.hardware_table(hw.summary())

    # ── Config summary ────────────────────────────────────────────────────────
    use_lora  = getattr(args, "lora", False)
    use_qlora = getattr(args, "qlora", False)

    # Auto-enable QLoRA when VRAM is tight and neither flag is explicitly set
    if hw.recommend_qlora and not use_lora and not use_qlora:
        dash.warn(
            f"Low VRAM detected ({hw.gpu_mem_gb:.1f} GB). "
            "Auto-enabling QLoRA. Pass --lora or --qlora explicitly to override."
        )
        use_qlora = True

    cfg = {
        "method":      args.method,
        "model":       args.model,
        "lora":        use_lora,
        "qlora":       use_qlora,
        "output_dir":  args.output_dir,
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "max_steps":   args.max_steps,
        "lr":          args.lr,
        "seed":        seed,
        "device":      hw.device,
        "dtype":       hw.torch_dtype,
    }
    dash.config_panel(cfg)

    # ── Model loading ─────────────────────────────────────────────────────────
    dash.section("Loading Model")
    dash.log(f"  Model : {args.model}")
    dash.log(f"  LoRA  : {use_lora}  |  QLoRA : {use_qlora}")

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model,
        hw=hw,
        use_lora=use_lora,
        use_qlora=use_qlora,
    )
    dash.log("  ✓ Model loaded", style="bold green")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dash.section("Dataset")
    if args.method == "dpo":
        dataset = get_dpo_dataset(tokenizer)
        dash.log(f"  DPO dataset: {len(dataset)} preference pairs", style="cyan")
    else:
        dataset = get_grpo_dataset(tokenizer)
        dash.log(f"  GRPO dataset: {len(dataset)} prompts", style="cyan")

    # ── Training ──────────────────────────────────────────────────────────────
    total_steps = _estimate_steps(
        dataset_size=len(dataset),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )
    dash.start_progress(total_steps)

    common_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        hw=hw,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        seed=seed,
        dashboard=dash,
    )

    if args.method == "dpo":
        metrics = run_dpo(**common_kwargs)
    else:
        metrics = run_grpo(**common_kwargs)

    dash.stop_progress()

    # ── Results ───────────────────────────────────────────────────────────────
    dash.section("Results")
    dash.results_table(metrics)
    dash.done(args.output_dir)

    return metrics


# ── Helpers ──────────────────────────────────────────────────────────────────

def _setup_logging():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    # Silence noisy third-party loggers
    for name in ("transformers", "datasets", "tokenizers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _estimate_steps(dataset_size: int, epochs: int, batch_size: int, max_steps: int) -> int:
    if max_steps > 0:
        return max_steps
    steps_per_epoch = max(1, dataset_size // max(batch_size, 1))
    return steps_per_epoch * epochs

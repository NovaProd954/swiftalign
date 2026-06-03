"""
swiftalign.hardware
~~~~~~~~~~~~~~~~~~~
Detects available GPU hardware and returns a HardwareProfile with
precision, attention, and quantisation recommendations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HardwareProfile:
    """Resolved hardware settings for a training run."""

    device: str                          # "cuda", "mps", or "cpu"
    gpu_name: Optional[str]              # e.g. "Tesla T4"
    gpu_mem_gb: float                    # total VRAM in GB (0.0 for CPU)
    torch_dtype: str                     # "bfloat16" | "float16" | "float32"
    attn_implementation: str             # "flash_attention_2" | "sdpa" | "eager"
    recommend_qlora: bool                # True when VRAM < 16 GB
    recommend_gradient_checkpointing: bool
    fp16: bool
    bf16: bool
    extra_notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "device": self.device,
            "gpu": self.gpu_name or "N/A",
            "vram_gb": f"{self.gpu_mem_gb:.1f}",
            "dtype": self.torch_dtype,
            "attention": self.attn_implementation,
            "qlora_rec": self.recommend_qlora,
            "grad_ckpt": self.recommend_gradient_checkpointing,
        }


def detect_hardware() -> HardwareProfile:
    """
    Probe the runtime environment and return a HardwareProfile.

    Priority order:
        1. CUDA GPU  → full feature set
        2. Apple MPS → fp16, eager attention
        3. CPU       → float32, minimal settings
    """
    try:
        import torch
    except ImportError:
        return _cpu_profile(note="torch not installed")

    # ── CUDA ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        mem_bytes = torch.cuda.get_device_properties(0).total_memory
        mem_gb = mem_bytes / (1024 ** 3)

        notes = []

        # Dtype: Ampere+ supports bfloat16 natively; T4/V100 prefer float16
        compute_cap = torch.cuda.get_device_properties(0).major
        if compute_cap >= 8:          # A100, A6000, RTX 3090+
            dtype = "bfloat16"
            bf16, fp16 = True, False
            notes.append("Ampere+ detected → bfloat16 enabled")
        else:                         # T4 (Turing), V100, P100
            dtype = "float16"
            bf16, fp16 = False, True
            notes.append("Pre-Ampere GPU → float16 enabled")

        # Attention: flash-attention-2 if installed, else SDPA
        attn = _best_attention(torch, notes)

        # QLoRA recommendation
        rec_qlora = mem_gb < 16.0
        rec_gc = mem_gb < 24.0
        if rec_qlora:
            notes.append(f"Low VRAM ({mem_gb:.1f} GB) → QLoRA recommended")
        if rec_gc:
            notes.append("gradient_checkpointing recommended")

        return HardwareProfile(
            device="cuda",
            gpu_name=gpu_name,
            gpu_mem_gb=mem_gb,
            torch_dtype=dtype,
            attn_implementation=attn,
            recommend_qlora=rec_qlora,
            recommend_gradient_checkpointing=rec_gc,
            fp16=fp16,
            bf16=bf16,
            extra_notes=notes,
        )

    # ── Apple MPS ────────────────────────────────────────────────────────────
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return HardwareProfile(
            device="mps",
            gpu_name="Apple Silicon MPS",
            gpu_mem_gb=0.0,
            torch_dtype="float16",
            attn_implementation="eager",
            recommend_qlora=False,
            recommend_gradient_checkpointing=False,
            fp16=True,
            bf16=False,
            extra_notes=["Apple MPS: flash-attention not supported, using eager"],
        )

    # ── CPU fallback ─────────────────────────────────────────────────────────
    return _cpu_profile()


def _best_attention(torch, notes: list) -> str:
    """Return the best available attention implementation."""
    try:
        import flash_attn  # noqa: F401
        notes.append("flash-attention-2 available")
        return "flash_attention_2"
    except ImportError:
        pass

    # PyTorch ≥ 2.0 ships scaled_dot_product_attention
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        notes.append("Using torch SDPA (scaled_dot_product_attention)")
        return "sdpa"

    notes.append("Falling back to eager attention")
    return "eager"


def _cpu_profile(note: str = "") -> HardwareProfile:
    notes = ["Running on CPU — training will be slow"]
    if note:
        notes.append(note)
    return HardwareProfile(
        device="cpu",
        gpu_name=None,
        gpu_mem_gb=0.0,
        torch_dtype="float32",
        attn_implementation="eager",
        recommend_qlora=False,
        recommend_gradient_checkpointing=False,
        fp16=False,
        bf16=False,
        extra_notes=notes,
    )

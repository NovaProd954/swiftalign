"""
SwiftAlign — compact alignment training scaffold
Supports DPO and GRPO with optional LoRA/QLoRA on causal language models.
"""

__version__ = "0.1.0"
__author__ = "NovatasticRoScript"

from .runner import run
from .hardware import detect_hardware, HardwareProfile
from .model_utils import load_model_and_tokenizer
from .data import get_dpo_dataset, get_grpo_dataset
from .rewards import (
    reward_length_penalty,
    reward_format_check,
    reward_keyword_presence,
    combined_reward,
)

__all__ = [
    "run",
    "detect_hardware",
    "HardwareProfile",
    "load_model_and_tokenizer",
    "get_dpo_dataset",
    "get_grpo_dataset",
    "reward_length_penalty",
    "reward_format_check",
    "reward_keyword_presence",
    "combined_reward",
]

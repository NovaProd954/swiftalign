"""
SwiftAlign — compact alignment training scaffold
Entry point: python train.py [--method dpo|grpo] [--model <hf_id>] [--lora] [--qlora]
"""

import argparse
import sys
from swiftalign.runner import run

def parse_args():
    p = argparse.ArgumentParser(
        description="SwiftAlign alignment training scaffold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --method dpo --lora
  python train.py --method grpo --qlora --model Qwen/Qwen2.5-1.5B-Instruct
  python train.py --method dpo --model Qwen/Qwen2.5-0.5B-Instruct --output_dir ./runs/dpo_test
        """
    )
    p.add_argument("--method", choices=["dpo", "grpo"], default="dpo",
                   help="Alignment method (default: dpo)")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct)")
    p.add_argument("--lora", action="store_true",
                   help="Apply LoRA fine-tuning")
    p.add_argument("--qlora", action="store_true",
                   help="Apply QLoRA (4-bit quantised + LoRA) fine-tuning")
    p.add_argument("--output_dir", type=str, default="./output",
                   help="Directory for checkpoints and logs (default: ./output)")
    p.add_argument("--epochs", type=int, default=1,
                   help="Number of training epochs (default: 1)")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Per-device training batch size (default: 2)")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Max training steps; overrides epochs if > 0")
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Learning rate (default: 5e-5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--no_dashboard", action="store_true",
                   help="Disable rich console dashboard")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)

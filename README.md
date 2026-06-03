# SwiftAlign

   _________       .__  _____  __         .__  .__               
 /   _____/_  _  _|__|/ ____\/  |______  |  | |__| ____   ____  
 \_____  \\ \/ \/ /  \   __\\   __\__  \ |  | |  |/ ___\ /    \ 
 /        \\     /|  ||  |   |  |  / __ \|  |_|  / /_/  >   |  \
/_______  / \/\_/ |__||__|   |__| (____  /____/__\___  /|___|  /
        \/                             \/       /_____/      \/

**Compact alignment training scaffold — DPO & GRPO with optional LoRA / QLoRA**

SwiftAlign is a demo-oriented training pipeline for experimenting with alignment methods
(Direct Preference Optimisation and Group Relative Policy Optimisation) on small causal
language models such as `Qwen2.5-1.5B-Instruct`. It is designed to run out-of-the-box in
constrained GPU environments like Google Colab T4 (15 GB VRAM).

```
┌──────────────────────────────────────────────────────────┐
│  SwiftAlign v0.1.0  —  compact alignment training scaffold│
│  ─────────────────────────────────────────────────────── │
│  Hardware detection  →  auto dtype / attention / QLoRA   │
│  Model loading       →  LoRA or QLoRA via PEFT           │
│  Built-in datasets   →  synthetic DPO pairs & GRPO tasks │
│  Training            →  DPO or GRPO via TRL              │
│  Dashboard           →  rich-text console output         │
└──────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Detail |
|---|---|
| **DPO training** | `trl.DPOTrainer` with KL-regularised preference learning |
| **GRPO training** | `trl.GRPOTrainer` with pluggable reward functions |
| **LoRA / QLoRA** | PEFT adapters; 4-bit NF4 quantisation via BitsAndBytes |
| **Hardware-aware** | Auto-selects dtype, attention impl, gradient checkpointing |
| **Built-in data** | 8 DPO pairs + 8 GRPO prompts, no external files needed |
| **Reward functions** | Length, format, keyword, safety, code, combined |
| **Rich dashboard** | Progress bars, metric tables, colour-coded panels |
| **Colab-ready** | Demo notebook included (`swiftalign_demo.ipynb`) |

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/NovatasticRoScript/swiftalign.git
cd swiftalign

# 2. Install
pip install -r requirements.txt
pip install bitsandbytes   # for QLoRA

# 3. Run DPO (auto QLoRA on T4)
python train.py --method dpo --qlora

# 4. Run GRPO with LoRA on a smaller model
python train.py --method grpo --lora --model Qwen/Qwen2.5-0.5B-Instruct
```

---

## CLI reference

```
python train.py [options]

Options:
  --method {dpo,grpo}       Alignment method (default: dpo)
  --model  MODEL_ID         HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct)
  --lora                    Apply LoRA adapters
  --qlora                   Apply QLoRA (4-bit + LoRA); recommended for T4
  --output_dir DIR          Checkpoint directory (default: ./output)
  --epochs  N               Training epochs (default: 1)
  --batch_size N            Per-device batch size (default: 2)
  --max_steps N             Max steps; overrides epochs if > 0 (default: -1)
  --lr FLOAT                Learning rate (default: 5e-5)
  --seed INT                Random seed (default: 42)
  --no_dashboard            Disable rich console output
```

---

## Project layout

```
swiftalign/
├── train.py                  ← Entry point
├── requirements.txt
├── pyproject.toml
├── swiftalign_demo.ipynb     ← Colab demo notebook
│
├── swiftalign/
│   ├── __init__.py
│   ├── runner.py             ← Orchestrates the full pipeline
│   ├── hardware.py           ← GPU detection → HardwareProfile
│   ├── model_utils.py        ← Model + tokenizer loading, LoRA/QLoRA
│   ├── data.py               ← Synthetic DPO & GRPO datasets
│   ├── rewards.py            ← GRPO reward functions + registry
│   ├── dashboard.py          ← Rich-text console dashboard
│   └── trainers/
│       ├── __init__.py
│       ├── dpo.py            ← DPOTrainer wrapper
│       └── grpo.py           ← GRPOTrainer wrapper
│
└── tests/
    └── test_swiftalign.py    ← Unit tests (no GPU required)
```

---

## Hardware detection

`swiftalign.hardware.detect_hardware()` returns a `HardwareProfile` with:

| Field | Example (T4) | Example (A100) |
|---|---|---|
| `device` | `"cuda"` | `"cuda"` |
| `gpu_name` | `"Tesla T4"` | `"A100-SXM4"` |
| `gpu_mem_gb` | `15.0` | `80.0` |
| `torch_dtype` | `"float16"` | `"bfloat16"` |
| `attn_implementation` | `"sdpa"` | `"flash_attention_2"` |
| `recommend_qlora` | `True` | `False` |
| `recommend_gradient_checkpointing` | `True` | `False` |

On T4 (pre-Ampere, 15 GB), QLoRA is automatically recommended.

---

## Reward functions (GRPO)

Five composable reward signals are provided in `swiftalign/rewards.py`:

| Name | Description |
|---|---|
| `length` | Penalises responses outside a target word-count window |
| `format` | Rewards sentence-ending punctuation and penalises truncation |
| `keyword` | Rewards responses that contain relevant prompt keywords |
| `safety` | Hard penalty for obvious harmful content patterns |
| `code` | Rewards fenced code blocks when the prompt requests code |
| `combined` | Weighted sum of all the above (default for GRPO) |

Custom reward functions follow this signature:

```python
def my_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    ...  # return a score in [-1.0, 1.0] for each completion
```

Register and use them:

```python
from swiftalign.rewards import REWARD_REGISTRY
REWARD_REGISTRY["my_reward"] = my_reward
```

Then pass `--reward_fn my_reward` to the CLI (or set `reward_fn_name` in `run_grpo`).

---

## Using as a library

```python
from swiftalign.hardware import detect_hardware
from swiftalign.model_utils import load_model_and_tokenizer
from swiftalign.data import get_dpo_dataset
from swiftalign.trainers import run_dpo

hw = detect_hardware()
model, tokenizer = load_model_and_tokenizer(
    "Qwen/Qwen2.5-0.5B-Instruct",
    hw,
    use_qlora=True,
)
dataset = get_dpo_dataset()
metrics = run_dpo(model, tokenizer, dataset, hw, output_dir="./my_run")
print(metrics)
```

---

## Running tests

No GPU required.

```bash
pip install pytest
pytest tests/ -v
```

---

## Requirements

| Package | Min version |
|---|---|
| torch | 2.0 |
| transformers | 4.40 |
| trl | 0.8 |
| peft | 0.10 |
| datasets | 2.18 |
| accelerate | 0.28 |
| rich | 13.0 |
| bitsandbytes *(optional)* | 0.43 |
| flash-attn *(optional)* | 2.5 |

---

## License

MIT — see [LICENSE](LICENSE)

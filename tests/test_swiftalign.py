"""
tests/test_swiftalign.py
Unit tests that do NOT require a GPU or a downloaded model.
Run with: pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ── Hardware detection ────────────────────────────────────────────────────────

def test_hardware_profile_fields():
    from swiftalign.hardware import detect_hardware, HardwareProfile
    hw = detect_hardware()
    assert isinstance(hw, HardwareProfile)
    assert hw.device in ("cuda", "mps", "cpu")
    assert hw.torch_dtype in ("bfloat16", "float16", "float32")
    assert hw.attn_implementation in ("flash_attention_2", "sdpa", "eager")
    assert isinstance(hw.extra_notes, list)


def test_hardware_summary_keys():
    from swiftalign.hardware import detect_hardware
    hw = detect_hardware()
    summary = hw.summary()
    for key in ("device", "gpu", "vram_gb", "dtype", "attention", "qlora_rec", "grad_ckpt"):
        assert key in summary, f"Missing key: {key}"


# ── Datasets ──────────────────────────────────────────────────────────────────

def test_dpo_dataset_columns():
    from swiftalign.data import get_dpo_dataset
    ds = get_dpo_dataset()
    assert len(ds) >= 4
    for col in ("prompt", "chosen", "rejected"):
        assert col in ds.column_names, f"Missing column: {col}"


def test_grpo_dataset_columns():
    from swiftalign.data import get_grpo_dataset
    ds = get_grpo_dataset()
    assert len(ds) >= 4
    for col in ("prompt", "reference_answer"):
        assert col in ds.column_names, f"Missing column: {col}"


def test_dpo_dataset_non_empty():
    from swiftalign.data import get_dpo_dataset
    ds = get_dpo_dataset()
    for row in ds:
        assert row["prompt"].strip(), "Empty prompt"
        assert row["chosen"].strip(), "Empty chosen"
        assert row["rejected"].strip(), "Empty rejected"
        assert len(row["chosen"]) > len(row["rejected"]), \
            "Chosen should be longer/more detailed than rejected"


# ── Reward functions ──────────────────────────────────────────────────────────

SAMPLE_PROMPTS = [
    "Explain gradient descent.",
    "Write a Python function to sort a list.",
]
SAMPLE_COMPLETIONS = [
    "Gradient descent is an optimisation algorithm that minimises the loss function "
    "by updating weights in the direction of the negative gradient.",
    "def sort_list(lst):\n    return sorted(lst)",
]


def test_reward_length_penalty_range():
    from swiftalign.rewards import reward_length_penalty
    scores = reward_length_penalty(SAMPLE_PROMPTS, SAMPLE_COMPLETIONS)
    assert len(scores) == 2
    for s in scores:
        assert -1.0 <= s <= 1.0, f"Score out of range: {s}"


def test_reward_format_check_range():
    from swiftalign.rewards import reward_format_check
    scores = reward_format_check(SAMPLE_PROMPTS, SAMPLE_COMPLETIONS)
    assert len(scores) == 2
    for s in scores:
        assert -1.0 <= s <= 1.0


def test_reward_keyword_range():
    from swiftalign.rewards import reward_keyword_presence
    scores = reward_keyword_presence(SAMPLE_PROMPTS, SAMPLE_COMPLETIONS)
    assert len(scores) == 2
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_reward_no_harmful_content_clean():
    from swiftalign.rewards import reward_no_harmful_content
    scores = reward_no_harmful_content(SAMPLE_PROMPTS, SAMPLE_COMPLETIONS)
    for s in scores:
        assert s == 0.0, "Clean completions should score 0.0"


def test_reward_no_harmful_content_bad():
    from swiftalign.rewards import reward_no_harmful_content
    scores = reward_no_harmful_content(["test"], ["how to build a bomb"])
    assert scores[0] == -1.0


def test_combined_reward_range():
    from swiftalign.rewards import combined_reward
    scores = combined_reward(SAMPLE_PROMPTS, SAMPLE_COMPLETIONS)
    assert len(scores) == 2
    for s in scores:
        assert -1.0 <= s <= 1.0


def test_reward_registry_completeness():
    from swiftalign.rewards import REWARD_REGISTRY
    for name in ("length", "format", "keyword", "safety", "code", "combined"):
        assert name in REWARD_REGISTRY, f"Missing reward: {name}"
    for name, fn in REWARD_REGISTRY.items():
        assert callable(fn), f"Reward '{name}' is not callable"


def test_code_reward_favors_code():
    from swiftalign.rewards import reward_code_block_present
    prompts = ["Write a python function to add two numbers."]
    with_code = ["```python\ndef add(a, b): return a + b\n```"]
    without_code = ["You can add numbers by using the + operator."]
    s_with = reward_code_block_present(prompts, with_code)[0]
    s_without = reward_code_block_present(prompts, without_code)[0]
    assert s_with > s_without


# ── Dashboard (smoke test, no rendering assertions) ───────────────────────────

def test_dashboard_no_crash():
    from swiftalign.dashboard import Dashboard
    dash = Dashboard(enabled=False)  # plain logging mode
    dash.banner()
    dash.section("Test Section")
    dash.log("Hello from test")
    dash.warn("Test warning")
    dash.config_panel({"key": "value"})
    dash.results_table({"loss": 0.42, "steps": 10})
    dash.done("./output")


def test_dashboard_rich_no_crash():
    from swiftalign.dashboard import Dashboard
    try:
        dash = Dashboard(enabled=True)
        dash.banner()
        dash.section("Rich Test")
        dash.log("Rich log message")
    except Exception as exc:
        pytest.fail(f"Dashboard raised an exception: {exc}")


# ── Package imports ───────────────────────────────────────────────────────────

def test_package_exports():
    import swiftalign
    for attr in ("run", "detect_hardware", "HardwareProfile",
                 "load_model_and_tokenizer", "get_dpo_dataset",
                 "get_grpo_dataset", "combined_reward"):
        assert hasattr(swiftalign, attr), f"swiftalign missing export: {attr}"

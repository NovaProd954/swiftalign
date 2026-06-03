"""
swiftalign.rewards
~~~~~~~~~~~~~~~~~~
Simple, composable reward functions for GRPO training.

Each function follows the TRL GRPO signature::

    reward_fn(prompts, completions, **kwargs) -> list[float]

where ``completions`` is a list of generated strings.
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── Individual reward signals ────────────────────────────────────────────────

def reward_length_penalty(
    prompts: List[str],
    completions: List[str],
    min_words: int = 10,
    max_words: int = 200,
    **kwargs,
) -> List[float]:
    """
    Penalise completions that are too short or excessively long.

    Returns a score in [-1.0, 1.0]:
      - Within [min_words, max_words]: scaled toward +1.0
      - Outside the range: negative penalty proportional to distance
    """
    scores = []
    for c in completions:
        n = len(c.split())
        if n < min_words:
            # Linearly scale from -1 at 0 words to 0 at min_words
            score = -1.0 + (n / max(min_words, 1))
        elif n > max_words:
            # Linearly penalise beyond max_words
            excess = n - max_words
            score = max(-1.0, -excess / max_words)
        else:
            # Within range: scale from 0 to 1 peaking at midpoint
            mid = (min_words + max_words) / 2
            score = 1.0 - abs(n - mid) / (mid - min_words + 1e-9) * 0.5
        scores.append(round(float(score), 4))
    return scores


def reward_format_check(
    prompts: List[str],
    completions: List[str],
    require_sentence_end: bool = True,
    penalise_truncation: bool = True,
    **kwargs,
) -> List[float]:
    """
    Check surface-level formatting quality.

    Rewards:
      +0.5 if the response ends with a sentence-ending punctuation (. ! ?)
      +0.5 if the response is not obviously truncated mid-sentence
      -0.5 if the response is a single word or obviously incomplete
    """
    scores = []
    for c in completions:
        text = c.strip()
        score = 0.0

        if require_sentence_end and text and text[-1] in ".!?":
            score += 0.5

        if penalise_truncation:
            # Heuristic: last "sentence" ends properly
            last_char = text[-1] if text else ""
            if last_char in ".!?":
                score += 0.5
            elif last_char in ",;:":
                score -= 0.25

        if len(text.split()) <= 1:
            score -= 0.5

        scores.append(round(float(max(-1.0, min(1.0, score))), 4))
    return scores


def reward_keyword_presence(
    prompts: List[str],
    completions: List[str],
    keywords: Optional[List[str]] = None,
    per_keyword_score: float = 0.2,
    **kwargs,
) -> List[float]:
    """
    Reward completions that contain specified keywords (case-insensitive).

    If ``keywords`` is None, attempts to extract domain keywords from the prompt
    using a simple heuristic (capitalised nouns / technical terms).

    Score = min(1.0, number_of_matches * per_keyword_score)
    """
    scores = []
    default_kws = keywords  # may be None

    for prompt, completion in zip(prompts, completions):
        kws = default_kws
        if kws is None:
            # Extract capitalised words from prompt as approximate keywords
            kws = re.findall(r'\b[A-Z][a-z]{2,}\b', prompt)
            kws = [k.lower() for k in kws[:5]]  # limit to 5

        if not kws:
            scores.append(0.0)
            continue

        comp_lower = completion.lower()
        matches = sum(1 for kw in kws if kw.lower() in comp_lower)
        score = min(1.0, matches * per_keyword_score)
        scores.append(round(float(score), 4))
    return scores


def reward_no_harmful_content(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Simple heuristic penalty for obviously harmful content.
    Returns -1.0 if a blocked pattern is detected, else 0.0.
    This is intentionally lightweight — real deployments should use a classifier.
    """
    _blocked = re.compile(
        r'\b(kill|murder|hack|exploit|malware|bomb|weapon|suicide)\b',
        re.IGNORECASE,
    )
    return [
        -1.0 if _blocked.search(c) else 0.0
        for c in completions
    ]


def reward_code_block_present(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Reward completions that include a fenced code block (``` ... ```) when
    the prompt appears to be asking for code.
    """
    code_prompt_pattern = re.compile(
        r'\b(function|code|implement|write|program|script|python|def|class)\b',
        re.IGNORECASE,
    )
    scores = []
    for prompt, completion in zip(prompts, completions):
        if code_prompt_pattern.search(prompt):
            has_code = "```" in completion or "def " in completion
            scores.append(1.0 if has_code else -0.5)
        else:
            scores.append(0.0)
    return scores


# ── Composite reward ─────────────────────────────────────────────────────────

def combined_reward(
    prompts: List[str],
    completions: List[str],
    weights: Optional[dict] = None,
    **kwargs,
) -> List[float]:
    """
    Weighted combination of all reward signals.

    Default weights::

        {
            "length": 0.25,
            "format": 0.25,
            "keyword": 0.25,
            "safety": 0.25,
        }

    Weights are automatically normalised to sum to 1.
    """
    default_weights = {
        "length": 0.25,
        "format": 0.25,
        "keyword": 0.25,
        "safety": 0.25,
    }
    w = {**default_weights, **(weights or {})}
    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}

    r_length  = reward_length_penalty(prompts, completions, **kwargs)
    r_format  = reward_format_check(prompts, completions, **kwargs)
    r_keyword = reward_keyword_presence(prompts, completions, **kwargs)
    r_safety  = reward_no_harmful_content(prompts, completions, **kwargs)

    combined = []
    for i in range(len(completions)):
        score = (
            w["length"]  * r_length[i]
            + w["format"]  * r_format[i]
            + w["keyword"] * r_keyword[i]
            + w["safety"]  * r_safety[i]
        )
        combined.append(round(float(score), 4))
    return combined


# ── Registry — used by runner to look up reward functions by name ─────────────

REWARD_REGISTRY = {
    "length":   reward_length_penalty,
    "format":   reward_format_check,
    "keyword":  reward_keyword_presence,
    "safety":   reward_no_harmful_content,
    "code":     reward_code_block_present,
    "combined": combined_reward,
}

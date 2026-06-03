"""
swiftalign.data
~~~~~~~~~~~~~~~
Built-in synthetic datasets so the scaffold runs out-of-the-box
without any external data files.

DPO dataset  → {prompt, chosen, rejected}
GRPO dataset → {prompt, reference_answer}
"""

from __future__ import annotations

from datasets import Dataset


# ────────────────────────────────────────────────────────────────────────────
# DPO synthetic data
# Chosen responses are helpful and safe; rejected responses are evasive or poor.
# ────────────────────────────────────────────────────────────────────────────

_DPO_EXAMPLES = [
    {
        "prompt": "Explain the difference between supervised and unsupervised learning.",
        "chosen": (
            "Supervised learning uses labelled data — the model learns from input-output "
            "pairs so it can predict labels on new inputs. Examples include image classification "
            "and spam detection. Unsupervised learning works on unlabelled data, discovering "
            "hidden structure such as clusters or latent features. Examples include k-means "
            "clustering and autoencoders. The key distinction is whether ground-truth labels "
            "guide the learning signal."
        ),
        "rejected": (
            "They are both types of machine learning. One uses labels and the other does not. "
            "I recommend looking it up for more details."
        ),
    },
    {
        "prompt": "What is gradient descent and why is it used in neural network training?",
        "chosen": (
            "Gradient descent is an iterative optimisation algorithm that minimises a loss "
            "function by updating parameters in the direction opposite to the gradient. "
            "In neural network training we compute the gradient of the loss with respect "
            "to every weight via backpropagation, then take a step proportional to the "
            "learning rate. This converges the network toward lower loss. Variants like "
            "Adam and SGD-with-momentum improve convergence speed and stability."
        ),
        "rejected": (
            "Gradient descent helps train neural networks. It changes the weights to make "
            "the network better. It is a common technique used in deep learning."
        ),
    },
    {
        "prompt": "How does the transformer attention mechanism work?",
        "chosen": (
            "Transformer attention computes compatibility scores between every query-key "
            "pair in a sequence. Given matrices Q (queries), K (keys), and V (values), "
            "attention weights are softmax(QK^T / sqrt(d_k)) and the output is those "
            "weights multiplied by V. Multi-head attention runs this in parallel across "
            "several projected subspaces, allowing the model to jointly attend to "
            "information from different representation positions."
        ),
        "rejected": (
            "Attention lets the model look at other words in the sentence. Transformers "
            "use attention to understand language. It was introduced in the paper "
            "'Attention is All You Need'."
        ),
    },
    {
        "prompt": "Describe the concept of overfitting and two ways to prevent it.",
        "chosen": (
            "Overfitting occurs when a model learns the training data too closely, including "
            "noise and spurious patterns, causing poor generalisation to new data. Two "
            "effective countermeasures are: (1) Regularisation — adding an L2 or L1 penalty "
            "to the loss function discourages large weights and reduces model complexity; "
            "(2) Dropout — randomly zeroing activations during training prevents co-adaptation "
            "of neurons, acting as an ensemble of many sub-networks."
        ),
        "rejected": (
            "Overfitting is bad. You can prevent it by getting more data or making the model "
            "smaller. Early stopping also helps."
        ),
    },
    {
        "prompt": "What is the role of the KL divergence term in RLHF / DPO training?",
        "chosen": (
            "The KL divergence penalty constrains how far the fine-tuned policy is allowed "
            "to drift from the reference model. Without it, unconstrained reward maximisation "
            "causes reward hacking — the model exploits loopholes to score high rewards while "
            "producing degenerate text. The KL term acts as a regulariser, preserving fluency "
            "and factual knowledge learned during pre-training. In DPO the reference model "
            "is implicit in the loss formulation, and the beta hyperparameter controls the "
            "strength of this constraint."
        ),
        "rejected": (
            "KL divergence keeps the model from changing too much. It is a penalty used in "
            "reinforcement learning. The beta parameter controls it."
        ),
    },
    {
        "prompt": "Explain what LoRA is and why it is useful for fine-tuning large models.",
        "chosen": (
            "LoRA (Low-Rank Adaptation) injects trainable low-rank matrices into the weight "
            "updates of a frozen pre-trained model. Instead of updating the full weight matrix "
            "W, LoRA decomposes the update as ΔW = A·B where A and B have rank r ≪ d. This "
            "reduces trainable parameters by orders of magnitude — a 7 B parameter model can "
            "be fine-tuned with fewer than 10 M parameters. The frozen backbone preserves "
            "pre-trained knowledge while the adapters specialise for the target task."
        ),
        "rejected": (
            "LoRA reduces the number of parameters to train. It is efficient and widely used. "
            "It works by adding small matrices to the model layers."
        ),
    },
    {
        "prompt": "What is the difference between DPO and PPO for alignment training?",
        "chosen": (
            "PPO (Proximal Policy Optimisation) trains an explicit reward model from human "
            "preference data and then uses RL to optimise the language model against it — "
            "requiring four models in memory simultaneously (policy, reference, reward, value). "
            "DPO (Direct Preference Optimisation) eliminates the reward model by re-framing "
            "the preference learning objective directly in terms of the policy, using a "
            "closed-form transformation of the Bradley-Terry model. DPO is simpler, more "
            "memory-efficient, and often competitive with PPO on alignment benchmarks."
        ),
        "rejected": (
            "PPO uses reinforcement learning and needs a reward model. DPO is simpler and "
            "does not need one. DPO is generally preferred for fine-tuning."
        ),
    },
    {
        "prompt": "How does quantisation reduce GPU memory usage during inference and training?",
        "chosen": (
            "Quantisation represents model weights in lower-bit formats (e.g. INT8 or NF4) "
            "instead of the default 32- or 16-bit floats. A 7 B model stored in float16 "
            "occupies ~14 GB; in 4-bit NF4 (QLoRA) it occupies roughly 3.5 GB. During "
            "forward passes, activations are computed in the original dtype (bfloat16) via "
            "dequantisation on the fly, preserving numerical quality while dramatically "
            "reducing the memory footprint. Double quantisation further compresses the "
            "quantisation constants themselves."
        ),
        "rejected": (
            "Quantisation uses fewer bits to store weights. This saves memory. INT8 and "
            "4-bit quantisation are common options."
        ),
    },
]


# ────────────────────────────────────────────────────────────────────────────
# GRPO synthetic data
# Prompts are reasoning / instruction-following tasks where reward functions
# can objectively score the output.
# ────────────────────────────────────────────────────────────────────────────

_GRPO_EXAMPLES = [
    {
        "prompt": "List exactly three advantages of using a transformer architecture over an RNN.",
        "reference_answer": (
            "1. Parallelisable training — all tokens processed simultaneously. "
            "2. Long-range dependencies captured via global attention. "
            "3. Better gradient flow — no vanishing gradient across long sequences."
        ),
    },
    {
        "prompt": "Explain in one sentence what a learning rate scheduler does.",
        "reference_answer": (
            "A learning rate scheduler adjusts the learning rate during training, "
            "typically reducing it over time to allow finer parameter updates as the "
            "model approaches convergence."
        ),
    },
    {
        "prompt": "Write a Python function that returns the nth Fibonacci number using memoisation.",
        "reference_answer": (
            "def fib(n, memo={}):\n"
            "    if n <= 1: return n\n"
            "    if n not in memo: memo[n] = fib(n-1, memo) + fib(n-2, memo)\n"
            "    return memo[n]"
        ),
    },
    {
        "prompt": "What does RLHF stand for, and what is its goal?",
        "reference_answer": (
            "RLHF stands for Reinforcement Learning from Human Feedback. "
            "Its goal is to align a language model's outputs with human preferences "
            "by training a reward model on human comparisons and optimising the LM "
            "against that reward signal."
        ),
    },
    {
        "prompt": "Give a one-sentence definition of entropy in information theory.",
        "reference_answer": (
            "Entropy quantifies the average amount of information (surprise) in a "
            "random variable, measured in bits, and equals minus the expected "
            "log-probability of outcomes."
        ),
    },
    {
        "prompt": "Describe the softmax function and when it is applied in a neural network.",
        "reference_answer": (
            "Softmax converts a vector of real-valued logits into a probability "
            "distribution by exponentiating each element and normalising by the sum; "
            "it is applied at the output layer of classification networks and language "
            "models to produce next-token probabilities."
        ),
    },
    {
        "prompt": "What problem does batch normalisation solve in deep networks?",
        "reference_answer": (
            "Batch normalisation addresses internal covariate shift — the change in the "
            "distribution of layer inputs during training — by normalising activations "
            "within each mini-batch, which stabilises and accelerates training."
        ),
    },
    {
        "prompt": "Summarise the purpose of the Adam optimiser in two sentences.",
        "reference_answer": (
            "Adam (Adaptive Moment Estimation) maintains per-parameter adaptive learning "
            "rates by combining first-moment (mean) and second-moment (uncentred variance) "
            "estimates of the gradients. This allows it to converge faster than vanilla "
            "SGD on sparse gradients and non-stationary objectives."
        ),
    },
]


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def get_dpo_dataset(tokenizer=None) -> Dataset:
    """
    Return a :class:`datasets.Dataset` with columns::

        prompt, chosen, rejected

    Suitable for ``trl.DPOTrainer``.
    """
    return Dataset.from_list(_DPO_EXAMPLES)


def get_grpo_dataset(tokenizer=None) -> Dataset:
    """
    Return a :class:`datasets.Dataset` with columns::

        prompt, reference_answer

    The ``prompt`` column is used directly as the input for generation.
    Reward functions in :mod:`swiftalign.rewards` score the completions.
    """
    return Dataset.from_list(_GRPO_EXAMPLES)

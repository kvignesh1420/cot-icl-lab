"""Configs for the experiments."""

from dataclasses import dataclass, field
from typing import List, Optional

IGNORE_INDEX = -100


@dataclass(frozen=True)
class Args:
    """Configs for the experiments.

    Example:
    - n_inputs: 2
    - n_parents: 2
    - chain_length: 2
    - n_examples: 5
    - vocab_size: 4

    * A sequence of 5 examples will be generated.
    * Each example has 2 input tokens.
    * Each example has 2 output tokens (i,e the chain length).
    * Total length of the sequence will be 5 * (2+2) = 20.
    * Example: [3, 3, 1, 0,   3, 2, 1, 0,   3, 0, 3, 1,   3, 0, 3, 1,   0, 2, 2, 3]
    """

    # Random seed for reproducibility
    seed: int = 42

    # Data
    n_inputs: int = 4  # Number of input tokens per example
    n_parents: int = 4  # Number of tokens to use for generating one output in the chain
    chain_length: int = 2  # Length of the chain in CoT
    n_examples: int = 40  # Number of examples per sequence.
    vocab_size: int = 1024  # Vocabulary size
    n_dims: int = 10  # Embedding dimension
    n_tasks: int = (
        64 * 50 * 1000
    )  # Number of tasks (prompts) to generate (calculate based on num of steps and batch size)
    batch_size: int = 64
    data_initializer_range: float = 1  # Initializer range for the data embeddings

    # Function class G
    dag_strategy: str = "random"  # Strategy to generate the DAG
    ablation_fixed_dag: bool = False  # If True, use a fixed DAG for all tasks (train and eval)

    # Function class H
    activation: str = "leaky_relu"  # Activation function for the linear layers
    H_num_layers: int = 1  # Number of layers in the function class H of FCNs
    ablation_fixed_H: bool = False  # If True, use a fixed H for all tasks (train and eval)
    # Number of unique H functions to use. -1 means use all unique H functions.
    # Set this only when ablation_fixed_H is True
    num_unique_H: int = -1

    # custom llama model configs
    model_type: str = "llama"
    num_attention_heads: int = 32
    num_hidden_layers: int = 4
    num_key_value_heads: int = 8
    hidden_size: int = 2048
    intermediate_size: int = 8192
    rope_scaling: dict = field(
        hash=False,
        default_factory=lambda: {
            "factor": 1.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 1000,
            "rope_type": "llama3",
        },
    )
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = True
    pad_token_id: Optional[int] = None
    # we don't need these in our custom chain generation so use large placeholder values
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    reserved_token_ids: List = field(default_factory=list)

    # training configs
    lr: float = 5e-5
    num_epochs: int = 1
    output_dir: str = "/opt/cot-icl-lab"
    log_every_n_steps: int = 100
    checkpoint_every_n_steps: int = 500

    # To CoT or not
    enable_cot: bool = False
    prompt_strategy: str = "standard"  # Strategy to generate the prompt

    # Evaluation (during training)
    temperature: float = 0.0
    eval_every_n_steps: int = 500  # Evaluate every n steps
    n_eval_tasks: int = 10000  # Number of tasks to evaluate on

    # Metric logging
    metric_logger: str = "stdout"

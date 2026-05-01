"""Configs for the experiments.

As a reference, This is the config.json of Llama-3.2-1B model

{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 16,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.0.dev0",
  "use_cache": true,
  "vocab_size": 128256
}

"""

from dataclasses import dataclass, field
from typing import Dict, Optional

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
    n_input_choices: tuple = (4,)  # Number of input tokens per example
    n_parent_choices: tuple = (4,)  # Number of tokens to use for generating one output in the chain
    chain_length_choices: tuple = (2,)  # Length of the chain in CoT
    n_example_choices: tuple = (40,)  # Number of examples per sequence.
    vocab_size: int = 1024  # Vocabulary size
    n_dims: int = 10  # Embedding dimension
    n_tasks: int = (
        64 * 50 * 1000
    )  # Number of tasks (sequences) to generate (calculate based on num of steps and batch size)
    batch_size: int = 64  # Batch size for training. Batch size of 128 leads to OOM error with 1B arch.
    data_initializer_range: float = 1  # Initializer range for the data embeddings

    # Function class G
    dag_strategy: str = "random"  # Strategy to generate the DAG ('random', 'markov')
    ablation_fixed_dag: bool = False  # If True, use a fixed DAG for all tasks (train and eval)

    # Function class H
    activation: str = "leaky_relu"  # Activation function for the linear layers
    H_num_layers: int = 1  # Number of layers in the function class H of FCNs
    ablation_fixed_H: bool = False  # If True, use a fixed H for all tasks (train and eval)
    # Number of unique H functions to use. 'None' means use all unique H functions.
    # Set this only when ablation_fixed_H is True
    num_unique_H: Optional[int] = None

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

    # training configs
    lr: float = 5e-5
    num_epochs: int = 1
    output_base_dir: str = "/opt/cot-icl-lab"
    log_every_n_steps: int = 100
    checkpoint_every_n_steps: int = 500

    # Select prompt strategy
    prompt_strategy: str = "standard"  # Strategy to generate the prompt
    ## Scheduler params when using `prompt_strategy="hybrid_special_token"`
    cot_example_prob_recipe_info: dict = field(
        hash=False,
        default_factory=lambda: {},
    )

    # Evaluation (during training)
    temperature: float = 0.0
    eval_every_n_steps: int = 500  # Evaluate every n steps
    n_eval_tasks: int = 10000  # Number of tasks to evaluate on

    # special tokens
    enable_special_tokens: bool = False
    # (will be set in the post_init based on enable_special_tokens)
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    input_start_token_id: Optional[int] = None
    input_end_token_id: Optional[int] = None
    think_start_token_id: Optional[int] = None
    think_end_token_id: Optional[int] = None
    answer_start_token_id: Optional[int] = None
    answer_end_token_id: Optional[int] = None
    reserved_token_ids: Optional[Dict[str, int]] = field(
        hash=False,
        default_factory=lambda: {},
    )
    # inference time params
    max_pred_tokens: int = 100

    metric_logger: str = "stdout"

    def __post_init__(self):
        pad_token_id = self.vocab_size - 1 if self.enable_special_tokens else None
        object.__setattr__(self, "pad_token_id", pad_token_id)
        reserved_token_ids = {"pad_token_id": pad_token_id} if self.enable_special_tokens else {}
        remaining_special_tokens = (
            "bos_token_id",
            "eos_token_id",
            "input_start_token_id",
            "input_end_token_id",
            "think_start_token_id",
            "think_end_token_id",
            "answer_start_token_id",
            "answer_end_token_id",
        )
        vocab_offset = 2  # offset 1 for the pad token
        for token_name in remaining_special_tokens:
            token_id = (
                self.vocab_size - vocab_offset if self.enable_special_tokens else 2 * self.vocab_size + vocab_offset
            )  # some large value

            reserved_token_ids.update({token_name: token_id} if self.enable_special_tokens else {})
            object.__setattr__(self, token_name, token_id)
            vocab_offset += 1

        object.__setattr__(self, "reserved_token_ids", reserved_token_ids)

    def __str__(self):
        """Prepare the run name for the experiment based on the arguments.
        Example : llama_L_4_V_16384_heads_8_M_4_N_4_C_2_n_ex_40_n_dims_10_hs_256_bs_128_act_leaky_relu_data_std_1_prompt_standard
        """
        # prepare a list of substrings and join them with `_`
        n_parent_choices = [str(i) for i in self.n_parent_choices]
        n_input_choices = [str(i) for i in self.n_input_choices]
        chain_length_choices = [str(i) for i in self.chain_length_choices]
        n_example_choices = [str(i) for i in self.n_example_choices]
        # convert all choices to strings
        # and join them with `-`
        # for example: (4, 8) -> "4-8"  and (4,) -> "4"
        recipe_str = "none"
        if self.cot_example_prob_recipe_info:
            type = self.cot_example_prob_recipe_info.get("type", "unknown")
            alpha = self.cot_example_prob_recipe_info.get("alpha", "unknown")
            recipe_str = f"type_{type}_alpha_{alpha}"

        substrings = [
            self.model_type,
            f"L_{self.num_hidden_layers}",
            f"V_{self.vocab_size}",
            f"heads_{self.num_attention_heads}",
            f"M_{'-'.join(n_parent_choices)}",
            f"N_{'-'.join(n_input_choices)}",
            f"C_{'-'.join(chain_length_choices)}",
            f"n_ex_{'-'.join(n_example_choices)}",
            f"n_dims_{self.n_dims}",
            f"hs_{self.hidden_size}",
            f"bs_{self.batch_size}",
            f"act_{self.activation}",
            f"prompt_{self.prompt_strategy}",
            f"recipe_{recipe_str}",
        ]
        return "_".join(substrings)


@dataclass
class EvalArgs:
    # model information
    model_path: str = None

    # inference information
    force_think_token: bool = False
    force_answer_token: bool = False
    num_output_seqs: int = 1
    temperature: float = 0.0
    max_pred_tokens: int = 100

    # data information
    # if data_path is None, then the data will be generated on the fly based on the parameters below.
    data_path: str = None
    ablation_fixed_H: bool = True
    num_unique_H: int = 1
    n_input: int = 4
    n_parent: int = 4
    chain_len: int = 4
    n_examples: int = 40
    n_eval_tasks: int = 1000

    # set during runtime
    output_dir: str = ""

    # Metric logging
    metric_logger: str = "stdout"

    def __post_init__(self):
        assert not (self.force_think_token and self.force_answer_token), (
            "Both force_think_token and force_answer_token cannot be True at the same time."
        )

    def __str__(self):
        substrings = [
            f"ft={self.force_think_token}",
            f"fa={self.force_answer_token}",
            f"n_input={self.n_input}",
            f"chain_len={self.chain_len}",
            f"n_parent={self.n_parent}",
            f"n_examples={self.n_examples}",
            f"n_eval_tasks={self.n_eval_tasks}",
            f"num_output_seqs={self.num_output_seqs}",
            f"temperature={self.temperature}",
            f"num_unique_H={self.num_unique_H}",
        ]
        return "_".join(substrings)

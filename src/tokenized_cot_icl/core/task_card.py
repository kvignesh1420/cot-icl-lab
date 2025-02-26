from itertools import product
from typing import Dict

from tokenized_cot_icl.core.args import Args


def default_task_card() -> Dict[int, Args]:
    """Default task card."""
    args = Args()
    return {0: args}


def vary_vocab_size() -> Dict[int, Args]:
    """Vary the vocab size."""
    num_layers = [4, 8, 12]
    cot_flag = [False, True]
    vocab_sizes = [64, 128, 256, 512, 1024]
    settings = product(num_layers, cot_flag, vocab_sizes)
    args_dict = {}
    for idx, (num_hidden_layers, cot, vocab_size) in enumerate(settings):
        args = Args(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            enable_cot=cot,
        )
        args_dict[idx] = args
    return args_dict


def vary_n_dims() -> Dict[int, Args]:
    """Vary the embedding dims."""
    num_layers = [4, 8, 12, 16]
    cot_flag = [False, True]
    # Adjust n_dims based on the vocab size
    n_dims_list = [20, 30, 40, 50]
    settings = product(num_layers, cot_flag, n_dims_list)
    args_dict = {}
    for idx, (num_hidden_layers, cot, n_dims) in enumerate(settings):
        args = Args(
            vocab_size=64,
            n_dims=n_dims,
            num_hidden_layers=num_hidden_layers,
            enable_cot=cot,
        )
        args_dict[idx] = args
    return args_dict


def vary_chain_length() -> Dict[int, Args]:
    """Vary the chain length."""
    num_layers = [4, 8, 12, 16]
    cot_flag = [False, True]
    chain_lengths = [3, 4, 5]
    settings = product(num_layers, cot_flag, chain_lengths)
    args_dict = {}
    for idx, (num_hidden_layers, cot, chain_length) in enumerate(settings):
        args = Args(
            vocab_size=1024,
            chain_length=chain_length,
            num_hidden_layers=num_hidden_layers,
            enable_cot=cot,
            batch_size=32,
            checkpoint_every_n_steps=1000,
            eval_every_n_steps=1000,
        )
        args_dict[idx] = args
    return args_dict


def vary_n_parents() -> Dict[int, Args]:
    """Vary n_parents for a noisy chain."""
    num_layers = [4, 8, 12]
    cot_flag = [False, True]
    n_parents_list = [1, 2, 3]
    settings = product(num_layers, cot_flag, n_parents_list)
    args_dict = {}
    for idx, (num_hidden_layers, cot, n_parents) in enumerate(settings):
        args = Args(
            vocab_size=1024,
            n_parents=n_parents,
            chain_length=4,
            num_hidden_layers=num_hidden_layers,
            enable_cot=cot,
            ablation_fixed_H=True,
            num_unique_H=100,
        )
        args_dict[idx] = args
    return args_dict


def vary_data_activation() -> Dict[int, Args]:
    """Vary the chain length."""
    num_layers = [4, 8, 12]
    cot_flag = [False, True]
    activations = ["relu", "silu", "identity"]
    settings = product(num_layers, cot_flag, activations)
    args_dict = {}
    for idx, (num_hidden_layers, cot, activation) in enumerate(settings):
        args = Args(
            vocab_size=64,
            activation=activation,
            num_hidden_layers=num_hidden_layers,
            enable_cot=cot,
        )
        args_dict[idx] = args
    return args_dict


def vary_num_examples() -> Dict[int, Args]:
    """Vary the num of examples in context"""
    num_layers = [4, 8, 12]
    cot_flag = [False, True]
    # Adjust n_dims based on the vocab size
    n_examples_list = [10, 20, 30]
    settings = product(num_layers, cot_flag, n_examples_list)
    args_dict = {}
    for idx, (num_hidden_layers, cot, n_examples) in enumerate(settings):
        args = Args(
            vocab_size=1024,
            n_examples=n_examples,
            num_hidden_layers=num_hidden_layers,
            enable_cot=cot,
        )
        args_dict[idx] = args
    return args_dict


def ablation_dag_strategy_markov():
    """Ablation study for Markov chain."""
    num_layers = [4, 8, 12]
    cot_flag = [False, True]
    vocab_size = 64
    n_parents = 1  # Markov chain
    chain_length = 4
    dag_strategy = "markov"
    settings = product(num_layers, cot_flag)
    args_dict = {}
    for idx, (num_hidden_layers, cot) in enumerate(settings):
        args = Args(
            vocab_size=vocab_size,
            n_inputs=1,
            n_parents=n_parents,
            chain_length=chain_length,
            num_hidden_layers=num_hidden_layers,
            enable_cot=cot,
            dag_strategy=dag_strategy,
        )
        args_dict[idx] = args
    return args_dict


# Create a TASK_CARD for bulk-launch of experiments
TASK_CARD = default_task_card()
# TASK_CARD = vary_vocab_size()
# TASK_CARD = vary_n_dims()
# TASK_CARD = vary_chain_length()
# TASK_CARD = vary_n_parents()
# TASK_CARD = vary_data_activation()
# TASK_CARD = vary_num_examples()

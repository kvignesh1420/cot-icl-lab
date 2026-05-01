import math
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
    prompt_strategies = ["standard", "cot"]
    vocab_sizes = [64, 128, 256, 512, 1024]
    settings = product(num_layers, prompt_strategies, vocab_sizes)
    args_dict = {}
    for idx, (num_hidden_layers, prompt_strategy, vocab_size) in enumerate(settings):
        args = Args(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            prompt_strategy=prompt_strategy,
            enable_special_tokens=True,
        )
        args_dict[idx] = args
    return args_dict


def vary_n_dims() -> Dict[int, Args]:
    """Vary the embedding dims."""
    num_layers = [4, 8, 12]
    prompt_strategies = ["standard", "cot"]
    # Adjust n_dims based on the vocab size
    n_dims_list = [20, 30, 40, 50]
    settings = product(num_layers, prompt_strategies, n_dims_list)
    args_dict = {}
    for idx, (num_hidden_layers, prompt_strategy, n_dims) in enumerate(settings):
        args = Args(
            vocab_size=64,
            n_dims=n_dims,
            num_hidden_layers=num_hidden_layers,
            prompt_strategy=prompt_strategy,
        )
        args_dict[idx] = args
    return args_dict


def vary_chain_length() -> Dict[int, Args]:
    """Vary the chain length."""
    num_layers = [4, 8, 12]
    prompt_strategies = ["standard", "cot"]
    chain_lengths = [3, 4, 5]
    settings = product(num_layers, prompt_strategies, chain_lengths)
    args_dict = {}
    for idx, (num_hidden_layers, prompt_strategy, chain_length) in enumerate(settings):
        args = Args(
            vocab_size=1024,
            chain_length_choices=(chain_length,),
            num_hidden_layers=num_hidden_layers,
            prompt_strategy=prompt_strategy,
            batch_size=32,
            checkpoint_every_n_steps=1000,
            eval_every_n_steps=1000,
        )
        args_dict[idx] = args
    return args_dict


def vary_n_parents() -> Dict[int, Args]:
    """Vary n_parents for a noisy chain."""
    num_layers = [4, 8, 12]
    prompt_strategies = ["standard", "cot"]
    n_parents_list = [1, 2, 3]
    settings = product(num_layers, prompt_strategies, n_parents_list)
    args_dict = {}
    for idx, (num_hidden_layers, prompt_strategy, n_parents) in enumerate(settings):
        args = Args(
            vocab_size=1024,
            n_parent_choices=(n_parents,),
            chain_length_choices=(4,),
            num_hidden_layers=num_hidden_layers,
            prompt_strategy=prompt_strategy,
            ablation_fixed_H=True,
            num_unique_H=100,
        )
        args_dict[idx] = args
    return args_dict


def vary_data_activation() -> Dict[int, Args]:
    """Vary the chain length."""
    num_layers = [4, 8, 12]
    prompt_strategies = ["standard", "cot"]
    activations = ["relu", "silu", "identity"]
    settings = product(num_layers, prompt_strategies, activations)
    args_dict = {}
    for idx, (num_hidden_layers, prompt_strategy, activation) in enumerate(settings):
        args = Args(
            vocab_size=64,
            activation=activation,
            num_hidden_layers=num_hidden_layers,
            prompt_strategy=prompt_strategy,
        )
        args_dict[idx] = args
    return args_dict


def vary_num_examples() -> Dict[int, Args]:
    """Vary the num of examples in context"""
    num_layers = [4, 8, 12]
    prompt_strategies = ["standard", "cot"]
    # Adjust n_dims based on the vocab size
    n_examples_list = [10, 20, 30]
    settings = product(num_layers, prompt_strategies, n_examples_list)
    args_dict = {}
    for idx, (num_hidden_layers, prompt_strategy, n_examples) in enumerate(settings):
        args = Args(
            vocab_size=1024,
            n_example_choices=(n_examples,),
            num_hidden_layers=num_hidden_layers,
            prompt_strategy=prompt_strategy,
        )
        args_dict[idx] = args
    return args_dict


# Create a TASK_CARD for bulk-launch of experiments
# TASK_CARD = default_task_card()
# TASK_CARD = vary_vocab_size()
# TASK_CARD = vary_n_dims()
# TASK_CARD = vary_chain_length()
# TASK_CARD = vary_n_parents()
# TASK_CARD = vary_data_activation()
# TASK_CARD = vary_num_examples()

##############################################################
#  With special tokens
##############################################################


def hybrid_sanity_check() -> Dict[int, Args]:
    """Probabilistically vary the chain length."""

    args = Args(
        vocab_size=1024,
        num_hidden_layers=4,
        prompt_strategy="hybrid_special_token",
        n_tasks=4 * 32 * 10,
        n_eval_tasks=10,
        batch_size=32,
        checkpoint_every_n_steps=10,
        eval_every_n_steps=10,
        enable_special_tokens=True,
        ablation_fixed_H=True,
        num_unique_H=1,
        cot_example_prob_recipe_info={
            "type": "power_law",
            "initial_prob": 0.0,
            "final_prob": 1.0,
            "alpha": 2,
            "scale": 1.0,
        },
    )
    return {0: args}


def baseline_fixed_chain_length():
    """Probabilistically vary the chain length."""
    num_layers = [4, 8, 12]
    prompt_strategies = ["hybrid_special_token"]
    chain_length_choices = (5,)
    settings = product(num_layers, prompt_strategies)
    args_dict = {}
    for idx, (num_hidden_layers, prompt_strategy) in enumerate(settings):
        args = Args(
            vocab_size=1024,
            chain_length_choices=chain_length_choices,
            num_hidden_layers=num_hidden_layers,
            prompt_strategy=prompt_strategy,
            batch_size=32,
            checkpoint_every_n_steps=1000,
            eval_every_n_steps=1000,
            enable_special_tokens=True,
        )
        args_dict[idx] = args
    return args_dict


def probabilistic_chain_length() -> Dict[int, Args]:
    """Probabilistically vary the chain length."""
    num_layers = [4, 8, 12]
    prompt_strategies = ["hybrid_special_token"]
    chain_length_choices = (2, 3, 4)
    settings = product(num_layers, prompt_strategies)
    args_dict = {}
    for idx, (num_hidden_layers, prompt_strategy) in enumerate(settings):
        args = Args(
            vocab_size=1024,
            chain_length_choices=chain_length_choices,
            num_hidden_layers=num_hidden_layers,
            prompt_strategy=prompt_strategy,
            batch_size=32,
            checkpoint_every_n_steps=1000,
            eval_every_n_steps=1000,
            enable_special_tokens=True,
            ablation_fixed_H=True,
            num_unique_H=1,
        )
        args_dict[idx] = args
    return args_dict


def hybrid_power_law_alpha() -> Dict[int, Args]:
    """Probabilistically vary the chain length."""
    num_layers = [4, 8, 12]
    alphas = [0, 0.5, 1, 2, math.inf]
    settings = product(num_layers, alphas)
    args_dict = {}
    for idx, (num_hidden_layers, alpha) in enumerate(settings):
        args = Args(
            vocab_size=1024,
            n_input_choices=(3, 4),
            n_parent_choices=(3, 4),
            chain_length_choices=(3, 4),
            num_hidden_layers=num_hidden_layers,
            prompt_strategy="hybrid_special_token",
            n_tasks=64 * 50 * 1000 * 2,
            n_eval_tasks=10000,
            batch_size=16,
            lr=5e-5,
            checkpoint_every_n_steps=4000,
            eval_every_n_steps=4000,
            enable_special_tokens=True,
            ablation_fixed_H=True,
            num_unique_H=1,
            cot_example_prob_recipe_info={
                "type": "power_law",
                "initial_prob": 0.0,
                "final_prob": 1.0,
                "alpha": alpha,
                "scale": 1.0,
            },
        )
        args_dict[idx] = args
    return args_dict


TASK_CARD = hybrid_sanity_check()
# TASK_CARD = baseline_fixed_chain_length()
# TASK_CARD = probabilistic_chain_length()
# TASK_CARD = hybrid_power_law_alpha()

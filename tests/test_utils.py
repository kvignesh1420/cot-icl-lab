import random

import numpy as np
import torch

from tokenized_cot_icl.core.args import Args, EvalArgs
from tokenized_cot_icl.core.utils import prepare_run_name, set_random_seed


def test_set_random_seed_is_reproducible():
    set_random_seed(123)
    py_a, np_a, torch_a = random.random(), np.random.rand(), torch.rand(3)
    set_random_seed(123)
    py_b, np_b, torch_b = random.random(), np.random.rand(), torch.rand(3)
    assert py_a == py_b
    assert np_a == np_b
    assert torch.equal(torch_a, torch_b)


def test_set_random_seed_different_seeds_diverge():
    set_random_seed(1)
    a = torch.rand(5)
    set_random_seed(2)
    b = torch.rand(5)
    assert not torch.equal(a, b)


def test_prepare_run_name_for_args_includes_str_and_timestamp():
    args = Args()
    name = prepare_run_name(args=args)
    base = str(args)
    assert name.startswith(base + "_")
    suffix = name[len(base) + 1 :]
    # timestamp format YYYYMMDD_HHMMSS -> 15 chars with one underscore
    assert len(suffix) == 15
    assert suffix[8] == "_"
    assert suffix.replace("_", "").isdigit()


def test_prepare_run_name_for_eval_args_includes_str_and_timestamp():
    eval_args = EvalArgs()
    name = prepare_run_name(args=eval_args)
    assert name.startswith(str(eval_args) + "_")

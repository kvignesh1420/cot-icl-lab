from functools import partial

import pytest
from torch.utils.data import DataLoader

from tokenized_cot_icl.core.args import Args
from tokenized_cot_icl.core.data.dag import RandomDAG
from tokenized_cot_icl.core.data.data import EvalTokenizedDataset, TokenizedDataset, special_token_collate_fn

N_TASKS = 1000
N_EVAL_TASKS = 100


def test_train_tokenized_dataset_size():
    args = Args(n_tasks=N_TASKS)
    train_ds = TokenizedDataset(args=args)
    assert len(train_ds) == args.n_tasks
    assert len(train_ds) == N_TASKS


def test_eval_tokenized_dataset_size():
    args = Args(n_eval_tasks=N_EVAL_TASKS)
    eval_ds = EvalTokenizedDataset(args=args)
    assert len(eval_ds) == args.n_eval_tasks
    assert len(eval_ds) == N_EVAL_TASKS
    for item in eval_ds:
        for key in [
            "input_ids",
            "attention_mask",
            "labels",
            "cot_eval",
            "num_cot_examples",
            "num_standard_examples",
        ]:
            assert key in item


def test_eval_tokenized_dataset_finite_H():
    args = Args(n_eval_tasks=N_EVAL_TASKS, num_unique_H=10)
    eval_ds = EvalTokenizedDataset(args=args)
    assert len(eval_ds) == args.n_eval_tasks
    assert len(eval_ds) == N_EVAL_TASKS
    assert len(eval_ds.token_processor_cache.storage) == 10


def test_create_dataloaders():
    args = Args(
        chain_length_choices=(2, 3, 4, 5),
        n_eval_tasks=N_EVAL_TASKS,
        enable_special_tokens=True,
        prompt_strategy="cot_special_token",
        cot_example_prob_recipe_info={
            "type": "power_law",
            "initial_prob": 0.0,
            "final_prob": 1.0,
            "alpha": 2,
            "scale": 1.0,
        },
    )
    eval_dataset = EvalTokenizedDataset(args=args)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=partial(special_token_collate_fn, pad_token_id=args.pad_token_id),
    )
    for batch in eval_loader:
        for key in [
            "input_ids",
            "attention_mask",
            "labels",
            "cot_eval",
            "num_cot_examples",
            "num_standard_examples",
        ]:
            assert key in batch


def test_sample_params_caps_n_parents_to_n_inputs():
    args = Args(
        n_input_choices=(2,),
        n_parent_choices=(8,),
        n_tasks=10,
    )
    ds = TokenizedDataset(args=args)
    for i in range(20):
        params = ds._sample_params(index=i)
        assert params["n_parents"] <= params["n_inputs"]
        assert params["n_inputs"] == 2
        assert params["n_parents"] == 2


def test_sample_params_threads_recipe_cot_example_prob():
    n_tasks = 100
    args = Args(
        n_tasks=n_tasks,
        enable_special_tokens=True,
        prompt_strategy="hybrid_special_token",
        cot_example_prob_recipe_info={
            "type": "power_law",
            "initial_prob": 0.0,
            "final_prob": 1.0,
            "alpha": 1,
            "scale": 1.0,
        },
    )
    ds = TokenizedDataset(args=args)
    p_first = ds._sample_params(index=0)["cot_example_prob"]
    p_last = ds._sample_params(index=n_tasks - 1)["cot_example_prob"]
    assert p_first == 0.0
    assert p_last > p_first


def test_sample_params_no_recipe_means_no_prob_key():
    args = Args(n_tasks=10)
    ds = TokenizedDataset(args=args)
    params = ds._sample_params(index=0)
    assert "cot_example_prob" not in params


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_chain_tokens_avoid_reserved_token_ids(seed):
    """Generated chain tokens must never collide with the reserved special tokens.

    Calls `_generate_example` directly to inspect raw chain tokens (without any
    prompt-strategy framing).
    """
    args = Args(
        n_tasks=10,
        enable_special_tokens=True,
        prompt_strategy="cot_special_token",
        seed=seed,
    )
    ds = TokenizedDataset(args=args)
    ds.create_token_processors(n_inputs=4, n_parents=4, chain_length=4)
    dag = RandomDAG(n_inputs=4, n_parents=4, chain_length=4)
    adj_list = dag.generate_adj_list()
    reserved = set(args.reserved_token_ids.values())
    for _ in range(20):
        example = ds._generate_example(adj_list=adj_list, n_inputs=4, chain_length=4)
        for chain_token in example["chain_tokens"]:
            assert chain_token not in reserved
        for input_token in example["input_tokens"]:
            assert input_token not in reserved


def test_special_token_collate_fn_left_pads_to_max():
    args = Args(
        n_eval_tasks=8,
        enable_special_tokens=True,
        prompt_strategy="cot_special_token",
        # variable chain lengths => variable sequence lengths => padding required
        chain_length_choices=(2, 3, 4),
    )
    ds = EvalTokenizedDataset(args=args)
    batch = [ds[i] for i in range(len(ds))]
    collated = special_token_collate_fn(batch=batch, pad_token_id=args.pad_token_id)
    expected_max = max(item["input_ids"].shape[0] for item in batch)
    assert collated["input_ids"].shape == (len(batch), expected_max)
    assert collated["attention_mask"].shape == collated["input_ids"].shape
    assert collated["labels"].shape == collated["input_ids"].shape
    # left-padding -> padding lives at the start
    for row, original in zip(collated["input_ids"], batch):
        pad_len = expected_max - original["input_ids"].shape[0]
        assert (row[:pad_len] == args.pad_token_id).all()

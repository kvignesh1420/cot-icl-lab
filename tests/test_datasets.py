from torch.utils.data import DataLoader

from tokenized_cot_icl.core.args import Args
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
        collate_fn=lambda batch: special_token_collate_fn(batch=batch, pad_token_id=args.pad_token_id),
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

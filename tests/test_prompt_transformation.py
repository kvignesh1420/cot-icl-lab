import pytest

from tokenized_cot_icl.core.args import Args
from tokenized_cot_icl.core.prompts.strategies.base import BasePrompt
from tokenized_cot_icl.core.prompts.strategies.registry import PROMPT_STRATEGY_REGISTRY
from tokenized_cot_icl.core.prompts.transformations.registry import PROMPT_TRANSFORMATION_REGISTRY

DATA = {
    "n_input_choices": (3,),
    "chain_length_choices": (3,),
    "examples": [
        {"input_tokens": [1, 2, 3], "chain_tokens": [4, 5, 6]},
        {"input_tokens": [7, 8, 9], "chain_tokens": [10, 11, 12]},
    ],
}

THREE_EXAMPLE_DATA = {
    "examples": [
        {"input_tokens": [1, 2, 3], "chain_tokens": [4, 5, 6]},
        {"input_tokens": [7, 8, 9], "chain_tokens": [10, 11, 12]},
        {"input_tokens": [13, 14, 15], "chain_tokens": [16, 17, 18]},
    ],
}


def create_cot_special_token_prompt(examples=None):
    examples = examples or DATA["examples"]
    args = Args(
        prompt_strategy="cot_special_token",
        enable_special_tokens=True,
        n_input_choices=DATA["n_input_choices"],
        chain_length_choices=DATA["chain_length_choices"],
        n_example_choices=(len(examples),),
    )
    prompt: BasePrompt = PROMPT_STRATEGY_REGISTRY[args.prompt_strategy](args=args)
    prompt_info = prompt.prepare(examples=examples)
    return prompt_info, args


def test_drop_k_cot_example_tokens():
    prompt_info, args = create_cot_special_token_prompt()

    expected_token_ids = [
        args.bos_token_id,
        args.input_start_token_id,
        1,
        2,
        3,
        args.input_end_token_id,
        args.answer_start_token_id,
        6,
        args.answer_end_token_id,
        args.eos_token_id,
        args.input_start_token_id,
        7,
        8,
        9,
        args.input_end_token_id,
    ]

    transform_fn = PROMPT_TRANSFORMATION_REGISTRY["drop_k_cot_tokens"]
    prompt_info = transform_fn(batch=[prompt_info], transformation="first_k", k=1)[0]
    assert prompt_info["cot_eval"]["input_ids"].tolist() == expected_token_ids
    assert prompt_info["cot_eval"]["attention_mask"].tolist() == [1] * len(expected_token_ids)

    prompt_info = transform_fn(batch=[prompt_info], transformation="last_k", k=1)[0]
    assert prompt_info["cot_eval"]["input_ids"].tolist() == expected_token_ids
    assert prompt_info["cot_eval"]["attention_mask"].tolist() == [1] * len(expected_token_ids)

    prompt_info = transform_fn(batch=[prompt_info], transformation="random", k=1)[0]
    assert prompt_info["cot_eval"]["input_ids"].tolist() == expected_token_ids
    assert prompt_info["cot_eval"]["attention_mask"].tolist() == [1] * len(expected_token_ids)


def test_offset_tokens():
    prompt_info, args = create_cot_special_token_prompt()
    offset = 128_000

    expected_token_ids = [
        args.bos_token_id,
        args.input_start_token_id,
        1,
        2,
        3,
        args.input_end_token_id,
        args.think_start_token_id,
        4,
        5,
        args.think_end_token_id,
        args.answer_start_token_id,
        6,
        args.answer_end_token_id,
        args.eos_token_id,
        args.input_start_token_id,
        7,
        8,
        9,
        args.input_end_token_id,
    ]
    expected_token_ids = [t_id + offset for t_id in expected_token_ids]

    transform_fn = PROMPT_TRANSFORMATION_REGISTRY["offset_tokens"]
    prompt_info = transform_fn(batch=[prompt_info], offset=offset)[0]
    assert prompt_info["cot_eval"]["input_ids"].tolist() == expected_token_ids
    assert prompt_info["cot_eval"]["attention_mask"].tolist() == [1] * len(expected_token_ids)


def test_offset_tokens_applies_to_train_and_eval_fields():
    prompt_info, _ = create_cot_special_token_prompt()
    offset = 1000
    original_input_ids = prompt_info["input_ids"].clone()
    original_labels = prompt_info["labels"].clone()
    original_cot_input_ids = prompt_info["cot_eval"]["input_ids"].clone()
    original_last_example_cot = prompt_info["cot_eval"]["last_example_cot"].clone()

    transform_fn = PROMPT_TRANSFORMATION_REGISTRY["offset_tokens"]
    transformed = transform_fn(batch=[prompt_info], offset=offset)[0]

    assert (transformed["input_ids"] == original_input_ids + offset).all()
    assert (transformed["labels"] == original_labels + offset).all()
    assert (transformed["cot_eval"]["input_ids"] == original_cot_input_ids + offset).all()
    assert (transformed["cot_eval"]["last_example_cot"] == original_last_example_cot + offset).all()


def test_drop_k_zero_is_no_op():
    prompt_info, args = create_cot_special_token_prompt(examples=THREE_EXAMPLE_DATA["examples"])
    original = prompt_info["cot_eval"]["input_ids"].clone()

    transform_fn = PROMPT_TRANSFORMATION_REGISTRY["drop_k_cot_tokens"]
    out = transform_fn(batch=[prompt_info], transformation="first_k", k=0)[0]
    assert out["cot_eval"]["input_ids"].tolist() == original.tolist()


@pytest.mark.parametrize("transformation", ["first_k", "last_k", "random"])
def test_drop_k_one_removes_one_think_span(transformation):
    """Across a 3-CoT prompt, dropping k=1 must remove exactly one
    <think_start>...<think_end> span — leaving 2 think_start markers."""
    prompt_info, args = create_cot_special_token_prompt(examples=THREE_EXAMPLE_DATA["examples"])
    original_starts = (prompt_info["cot_eval"]["input_ids"] == args.think_start_token_id).sum().item()
    original_ends = (prompt_info["cot_eval"]["input_ids"] == args.think_end_token_id).sum().item()
    # cot_eval drops the very last example's CoT, so a 3-example CoT prompt has
    # 2 think spans inside cot_eval (examples 1 and 2).
    assert original_starts == 2
    assert original_ends == 2

    transform_fn = PROMPT_TRANSFORMATION_REGISTRY["drop_k_cot_tokens"]
    out = transform_fn(batch=[prompt_info], transformation=transformation, k=1)[0]
    new_ids = out["cot_eval"]["input_ids"]
    assert (new_ids == args.think_start_token_id).sum().item() == 1
    assert (new_ids == args.think_end_token_id).sum().item() == 1
    # all non-think framing tokens should still be there
    assert (new_ids == args.input_start_token_id).sum().item() == (
        prompt_info["cot_eval"]["input_ids"] == args.input_start_token_id
    ).sum().item() == 3


def test_drop_first_k_removes_earliest_span():
    """`first_k` with k=1 must remove the FIRST think span specifically."""
    prompt_info, args = create_cot_special_token_prompt(examples=THREE_EXAMPLE_DATA["examples"])

    transform_fn = PROMPT_TRANSFORMATION_REGISTRY["drop_k_cot_tokens"]
    out = transform_fn(batch=[prompt_info], transformation="first_k", k=1)[0]
    new_ids = out["cot_eval"]["input_ids"].tolist()

    # The dropped span belonged to example 1 -> chain tokens 4, 5 must be gone;
    # chain tokens 10, 11 (example 2) must still be present.
    assert 4 not in new_ids
    assert 5 not in new_ids
    assert 10 in new_ids
    assert 11 in new_ids


def test_drop_last_k_removes_latest_span():
    prompt_info, args = create_cot_special_token_prompt(examples=THREE_EXAMPLE_DATA["examples"])

    transform_fn = PROMPT_TRANSFORMATION_REGISTRY["drop_k_cot_tokens"]
    out = transform_fn(batch=[prompt_info], transformation="last_k", k=1)[0]
    new_ids = out["cot_eval"]["input_ids"].tolist()

    # cot_eval has spans for examples 1 and 2 only. Dropping the last keeps
    # example 1's chain tokens (4, 5) and removes example 2's (10, 11).
    assert 4 in new_ids
    assert 5 in new_ids
    assert 10 not in new_ids
    assert 11 not in new_ids


def test_drop_k_too_large_raises():
    prompt_info, args = create_cot_special_token_prompt(examples=THREE_EXAMPLE_DATA["examples"])

    transform_fn = PROMPT_TRANSFORMATION_REGISTRY["drop_k_cot_tokens"]
    # cot_eval has 2 think spans; asking to drop 5 must raise.
    with pytest.raises(AssertionError):
        transform_fn(batch=[prompt_info], transformation="first_k", k=5)

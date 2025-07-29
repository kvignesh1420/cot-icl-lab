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


def create_cot_special_token_prompt():
    args = Args(
        prompt_strategy="cot_special_token",
        enable_special_tokens=True,
        n_input_choices=DATA["n_input_choices"],
        chain_length_choices=DATA["chain_length_choices"],
        n_example_choices=(len(DATA["examples"]),),
    )
    prompt: BasePrompt = PROMPT_STRATEGY_REGISTRY[args.prompt_strategy](args=args)
    prompt_info = prompt.prepare(examples=DATA["examples"])
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

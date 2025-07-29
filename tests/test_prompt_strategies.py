from tokenized_cot_icl.core.args import IGNORE_INDEX, Args
from tokenized_cot_icl.core.prompts.strategies.base import BasePrompt
from tokenized_cot_icl.core.prompts.strategies.registry import PROMPT_STRATEGY_REGISTRY

DATA = {
    "n_input_choices": (3,),
    "chain_length_choices": (3,),
    "examples": [
        {"input_tokens": [1, 2, 3], "chain_tokens": [4, 5, 6]},
        {"input_tokens": [7, 8, 9], "chain_tokens": [10, 11, 12]},
    ],
}


def test_standard_prompt():
    args = Args(
        prompt_strategy="standard",
        enable_special_tokens=False,
        n_input_choices=DATA["n_input_choices"],
        chain_length_choices=DATA["chain_length_choices"],
        n_example_choices=(len(DATA["examples"]),),
    )

    prompt: BasePrompt = PROMPT_STRATEGY_REGISTRY[args.prompt_strategy](args=args)
    prompt_info = prompt.prepare(examples=DATA["examples"])
    for key in ["input_ids", "attention_mask", "labels", "cot_eval"]:
        assert key in prompt_info
    assert prompt_info["input_ids"].tolist() == [1, 2, 3, 6, 7, 8, 9, 12]
    assert prompt_info["attention_mask"].tolist() == [1] * 8
    assert prompt_info["labels"].tolist() == [IGNORE_INDEX] * 3 + [6] + [IGNORE_INDEX] * 3 + [12]

    assert prompt_info["cot_eval"]["input_ids"].tolist() == [1, 2, 3, 6, 7, 8, 9]
    assert prompt_info["cot_eval"]["attention_mask"].tolist() == [1] * 7
    assert prompt_info["cot_eval"]["last_example_cot"].tolist() == [12]
    assert prompt_info["num_cot_examples"] == 0
    assert prompt_info["num_standard_examples"] == 2


def test_cot_prompt():
    args = Args(
        prompt_strategy="cot",
        enable_special_tokens=False,
        n_input_choices=DATA["n_input_choices"],
        chain_length_choices=DATA["chain_length_choices"],
        n_example_choices=(len(DATA["examples"]),),
    )

    prompt: BasePrompt = PROMPT_STRATEGY_REGISTRY[args.prompt_strategy](args=args)
    prompt_info = prompt.prepare(examples=DATA["examples"])
    for key in ["input_ids", "attention_mask", "labels", "cot_eval"]:
        assert key in prompt_info
    assert prompt_info["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert prompt_info["attention_mask"].tolist() == [1] * 12
    assert prompt_info["labels"].tolist() == [IGNORE_INDEX] * 3 + [4, 5, 6] + [IGNORE_INDEX] * 3 + [10, 11, 12]

    assert prompt_info["cot_eval"]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert prompt_info["cot_eval"]["attention_mask"].tolist() == [1] * 9
    assert prompt_info["cot_eval"]["last_example_cot"].tolist() == [10, 11, 12]
    assert prompt_info["num_cot_examples"] == 2
    assert prompt_info["num_standard_examples"] == 0


def test_standard_special_token_prompt():
    args = Args(
        prompt_strategy="standard_special_token",
        enable_special_tokens=True,
        n_input_choices=DATA["n_input_choices"],
        chain_length_choices=DATA["chain_length_choices"],
        n_example_choices=(len(DATA["examples"]),),
    )
    prompt: BasePrompt = PROMPT_STRATEGY_REGISTRY[args.prompt_strategy](args=args)
    prompt_info = prompt.prepare(examples=DATA["examples"])
    for key in ["input_ids", "attention_mask", "labels", "cot_eval"]:
        assert key in prompt_info
    assert prompt_info["input_ids"].tolist() == [
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
        args.answer_start_token_id,
        12,
        args.answer_end_token_id,
        args.eos_token_id,
    ]
    assert prompt_info["attention_mask"].tolist() == [1] * prompt_info["input_ids"].shape[0]
    assert prompt_info["labels"].tolist() == [
        args.bos_token_id,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        args.answer_start_token_id,
        6,
        args.answer_end_token_id,
        args.eos_token_id,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        args.answer_start_token_id,
        12,
        args.answer_end_token_id,
        args.eos_token_id,
    ]
    assert prompt_info["cot_eval"]["input_ids"].tolist() == [
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
    assert prompt_info["cot_eval"]["attention_mask"].tolist() == [1] * prompt_info["cot_eval"]["input_ids"].shape[0]
    assert prompt_info["cot_eval"]["last_example_cot"].tolist() == [
        args.answer_start_token_id,
        12,
        args.answer_end_token_id,
        args.eos_token_id,
    ]
    assert prompt_info["num_cot_examples"] == 0
    assert prompt_info["num_standard_examples"] == 2


def test_cot_special_token_prompt():
    args = Args(
        prompt_strategy="cot_special_token",
        enable_special_tokens=True,
        n_input_choices=DATA["n_input_choices"],
        chain_length_choices=DATA["chain_length_choices"],
        n_example_choices=(len(DATA["examples"]),),
    )
    prompt: BasePrompt = PROMPT_STRATEGY_REGISTRY[args.prompt_strategy](args=args)
    prompt_info = prompt.prepare(examples=DATA["examples"])
    for key in ["input_ids", "attention_mask", "labels", "cot_eval"]:
        assert key in prompt_info
    assert prompt_info["input_ids"].tolist() == [
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
        args.think_start_token_id,
        10,
        11,
        args.think_end_token_id,
        args.answer_start_token_id,
        12,
        args.answer_end_token_id,
        args.eos_token_id,
    ]
    assert prompt_info["attention_mask"].tolist() == [1] * prompt_info["input_ids"].shape[0]
    assert prompt_info["labels"].tolist() == [
        args.bos_token_id,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        args.think_start_token_id,
        4,
        5,
        args.think_end_token_id,
        args.answer_start_token_id,
        6,
        args.answer_end_token_id,
        args.eos_token_id,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        args.think_start_token_id,
        10,
        11,
        args.think_end_token_id,
        args.answer_start_token_id,
        12,
        args.answer_end_token_id,
        args.eos_token_id,
    ]
    assert prompt_info["cot_eval"]["input_ids"].tolist() == [
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
    assert prompt_info["cot_eval"]["attention_mask"].tolist() == [1] * prompt_info["cot_eval"]["input_ids"].shape[0]
    assert prompt_info["cot_eval"]["last_example_cot"].tolist() == [
        args.think_start_token_id,
        10,
        11,
        args.think_end_token_id,
        args.answer_start_token_id,
        12,
        args.answer_end_token_id,
        args.eos_token_id,
    ]
    assert prompt_info["num_cot_examples"] == 2
    assert prompt_info["num_standard_examples"] == 0


def test_hybrid_special_token_prompt():
    args = Args(
        prompt_strategy="hybrid_special_token",
        enable_special_tokens=True,
        n_input_choices=DATA["n_input_choices"],
        chain_length_choices=DATA["chain_length_choices"],
        n_example_choices=(len(DATA["examples"]),),
    )
    prompt: BasePrompt = PROMPT_STRATEGY_REGISTRY[args.prompt_strategy](args=args)
    prompt_info = prompt.prepare(examples=DATA["examples"], cot_example_prob=0.5)
    for key in [
        "input_ids",
        "attention_mask",
        "labels",
        "cot_eval",
        "num_cot_examples",
        "num_standard_examples",
    ]:
        assert key in prompt_info

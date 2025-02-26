from tokenized_cot_icl.core.args import IGNORE_INDEX, Args
from tokenized_cot_icl.core.prompts.base import BasePrompt
from tokenized_cot_icl.core.prompts.registry import PROMPT_REGISTRY

DATA = {
    "n_inputs": 3,
    "chain_length": 3,
    "examples": [
        {"input_tokens": [1, 2, 3], "chain_tokens": [4, 5, 6]},
        {"input_tokens": [7, 8, 9], "chain_tokens": [10, 11, 12]},
    ],
}


def test_standard_prompt():
    args = Args(
        enable_cot=False,
        n_inputs=DATA["n_inputs"],
        chain_length=DATA["chain_length"],
        n_examples=len(DATA["examples"]),
    )

    prompt: BasePrompt = PROMPT_REGISTRY["standard"](args=args)
    prompt_info = prompt.prepare(examples=DATA["examples"])
    for key in ["input_ids", "attention_mask", "labels", "cot_eval"]:
        assert key in prompt_info
    assert prompt_info["input_ids"].tolist() == [1, 2, 3, 6, 7, 8, 9, 12]
    assert prompt_info["attention_mask"].tolist() == [1] * 8
    assert prompt_info["labels"].tolist() == [IGNORE_INDEX] * 3 + [6] + [
        IGNORE_INDEX
    ] * 3 + [12]

    assert prompt_info["cot_eval"]["input_ids"].tolist() == [1, 2, 3, 6, 7, 8, 9]
    assert prompt_info["cot_eval"]["attention_mask"].tolist() == [1] * 7
    assert prompt_info["cot_eval"]["last_example_cot"].tolist() == [12]


def test_cot_prompt():
    args = Args(
        enable_cot=True,
        n_inputs=DATA["n_inputs"],
        chain_length=DATA["chain_length"],
        n_examples=len(DATA["examples"]),
    )

    prompt: BasePrompt = PROMPT_REGISTRY["cot"](args=args)
    prompt_info = prompt.prepare(examples=DATA["examples"])
    for key in ["input_ids", "attention_mask", "labels", "cot_eval"]:
        assert key in prompt_info
    assert prompt_info["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert prompt_info["attention_mask"].tolist() == [1] * 12
    assert prompt_info["labels"].tolist() == [IGNORE_INDEX] * 3 + [4, 5, 6] + [
        IGNORE_INDEX
    ] * 3 + [10, 11, 12]

    assert prompt_info["cot_eval"]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert prompt_info["cot_eval"]["attention_mask"].tolist() == [1] * 9
    assert prompt_info["cot_eval"]["last_example_cot"].tolist() == [10, 11, 12]

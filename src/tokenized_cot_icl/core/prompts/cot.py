from typing import List

from copy import deepcopy
from typing import Dict
from tokenized_cot_icl.core.args import Args, IGNORE_INDEX
from tokenized_cot_icl.core.prompts.standard import StandardPrompt


class CoTPrompt(StandardPrompt):
    """Prepare prompt with CoT"""

    def __init__(self, args: Args):
        self.args = args
        assert self.args.enable_cot

    def _get_intermediate_and_answer_tokens(self, chain_tokens: List[int]) -> List[int]:
        """Get the intermediate and answer tokens."""
        intermediate_tokens = chain_tokens[:-1]
        answer_tokens = chain_tokens[-1:]
        return intermediate_tokens, answer_tokens

    def get_example_info(self, example: Dict[str, int]) -> Dict[str, List[int]]:
        intermediate_tokens, answer_tokens = self._get_intermediate_and_answer_tokens(
            chain_tokens=example["chain_tokens"]
        )
        example_input_ids = [
            *example["input_tokens"],
            *intermediate_tokens,
            *answer_tokens,
        ]
        example_attention_mask = [1] * len(example_input_ids)
        example_labels = deepcopy(example_input_ids)
        example_labels[: len(example["input_tokens"])] = [IGNORE_INDEX] * len(
            example["input_tokens"]
        )

        return {
            "example_input_ids": example_input_ids,
            "example_attention_mask": example_attention_mask,
            "example_labels": example_labels,
            "cot_eval_input_mask_length": len(example["input_tokens"]),
        }

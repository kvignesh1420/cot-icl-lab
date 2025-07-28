from typing import Dict, List

from copy import deepcopy
from tokenized_cot_icl.core.args import Args, IGNORE_INDEX
from tokenized_cot_icl.core.prompts.strategies.base import BasePrompt


class StandardSpecialTokenPrompt(BasePrompt):
    """Prepare prompt:
    - Without CoT
    - With special tokens
    - Without rethinking
    """

    def __init__(self, args: Args):
        self.args = args
        assert self.args.enable_special_tokens
        assert self.args.reserved_token_ids

    def _get_intermediate_and_answer_tokens(self, chain_tokens: List[int]) -> List[int]:
        """Get the intermediate and answer tokens."""
        intermediate_tokens = []
        answer_tokens = chain_tokens[-1:]
        return intermediate_tokens, answer_tokens

    def get_example_info(
        self, example: Dict[str, int], **kwargs
    ) -> Dict[str, List[int]]:
        _, answer_tokens = self._get_intermediate_and_answer_tokens(
            chain_tokens=example["chain_tokens"]
        )
        example_input_ids = [
            self.args.input_start_token_id,
            *example["input_tokens"],
            self.args.input_end_token_id,
            self.args.answer_start_token_id,
            *answer_tokens,
            self.args.answer_end_token_id,
            self.args.eos_token_id,
        ]
        example_attention_mask = [1] * len(example_input_ids)
        example_labels = deepcopy(example_input_ids)
        input_tokens_len_with_special_tokens = len(example["input_tokens"]) + 2
        example_labels[:input_tokens_len_with_special_tokens] = [
            IGNORE_INDEX
        ] * input_tokens_len_with_special_tokens
        return {
            "example_input_ids": example_input_ids,
            "example_attention_mask": example_attention_mask,
            "example_labels": example_labels,
            "cot_eval_input_mask_length": input_tokens_len_with_special_tokens,
            "is_cot_example": False,
        }

    def get_prefix_info(self):
        return {
            "prefix_input_ids": [self.args.bos_token_id],
            "prefix_attention_mask": [1],
            "prefix_labels": [self.args.bos_token_id],
        }

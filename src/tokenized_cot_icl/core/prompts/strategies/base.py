import abc
from copy import deepcopy
from typing import Dict, List

import torch

from tokenized_cot_icl.core.args import IGNORE_INDEX


class BasePrompt(abc.ABC):
    @abc.abstractmethod
    def _get_intermediate_and_answer_tokens(self, chain_tokens: List[int]) -> List[int]:
        """Get the intermediate and answer tokens given the chain tokens."""
        ...

    @abc.abstractmethod
    def get_prefix_info(self) -> Dict[str, List[int]]:
        """Get the prefix information.

        The return dictionary should have the following keys:
        {
            "prefix_input_ids": prefix_input_ids,
            "prefix_attention_mask": prefix_attention_mask,
            "prefix_labels": prefix_labels,
        }
        """
        ...

    def get_special_token_ids(self) -> Dict[str, torch.Tensor]:
        """Get the special token ids if available"""
        if not hasattr(self, "args") or self.args is None:
            return {}

        if not hasattr(self.args, "enable_special_tokens") or not self.args.enable_special_tokens:
            return {}

        return self.args.reserved_token_ids

    @abc.abstractmethod
    def get_example_info(self, example: Dict[str, int], **kwargs) -> Dict[str, List[int]]:
        """Get the example information.

        The return dictionary should have the following keys:
        {
            "example_input_ids": example_input_ids,
            "example_attention_mask": example_attention_mask,
            "example_labels": example_labels,
            "cot_eval_input_mask_length": cot_eval_input_mask_length,
        }
        """
        ...

    def _prepare_prompt_info(self, examples: List[Dict[str, int]], **kwargs) -> List[int]:
        """Prepare the prompt using a list of examples.

        Each prompt will have the following format:
        <bos> [<example>]*K

        Each <example> will have the following format:
        <input_tokens> [ <leftover_tokens> <rethink_id> ] * (rethink_count-1) <chain_tokens> <eos>
        """
        prefix_info = self.get_prefix_info()
        # data used for training
        input_ids = deepcopy(prefix_info["prefix_input_ids"])
        attention_mask = deepcopy(prefix_info["prefix_attention_mask"])
        labels = deepcopy(prefix_info["prefix_labels"])
        # data used for CoT evaluation
        cot_eval_input_ids = deepcopy(prefix_info["prefix_input_ids"])
        cot_eval_attention_mask = deepcopy(prefix_info["prefix_attention_mask"])
        last_example_cot = []
        num_examples = len(examples)
        # stats about standard and CoT examples
        num_cot_examples = 0
        num_standard_examples = 0

        for index, example in enumerate(examples):
            # obtain the example information
            example_info = self.get_example_info(example=example, **kwargs)

            # prepare training specific info
            input_ids.extend(example_info["example_input_ids"])
            attention_mask.extend(example_info["example_attention_mask"])
            labels.extend(example_info["example_labels"])
            if example_info["is_cot_example"]:
                num_cot_examples += 1
            else:
                num_standard_examples += 1

            # prepare CoT eval specific info
            if index < num_examples - 1:
                cot_eval_example_input_ids = example_info["example_input_ids"]
                cot_eval_example_attention_mask = example_info["example_attention_mask"]
            else:
                cot_eval_example_input_ids = example_info["example_input_ids"][
                    : example_info["cot_eval_input_mask_length"]
                ]
                cot_eval_example_attention_mask = [1] * len(cot_eval_example_input_ids)
                last_example_cot = [label for label in example_info["example_labels"] if label != IGNORE_INDEX]
            cot_eval_input_ids.extend(cot_eval_example_input_ids)
            cot_eval_attention_mask.extend(cot_eval_example_attention_mask)

        assert len(input_ids) == len(labels), f"len(input_ids): {len(input_ids)} != len(labels): {len(labels)}"
        assert len(input_ids) == len(attention_mask)
        assert len(cot_eval_input_ids) == len(cot_eval_attention_mask)
        assert num_examples == num_cot_examples + num_standard_examples, (
            f"num_examples: {num_examples} != "
            f"num_cot_examples: {num_cot_examples} + num_standard_examples: {num_standard_examples}"
        )

        prompt_info = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
            "cot_eval": {
                "input_ids": torch.tensor(cot_eval_input_ids),
                "attention_mask": torch.tensor(cot_eval_attention_mask),
                "last_example_cot": torch.tensor(last_example_cot),
            },
            "num_cot_examples": torch.tensor(num_cot_examples),
            "num_standard_examples": torch.tensor(num_standard_examples),
        }

        special_token_ids = self.get_special_token_ids()
        prompt_info.update(special_token_ids)
        return prompt_info

    def prepare(self, examples: List[Dict[str, int]], **kwargs) -> List[int]:
        """Prepare the train info and cot eval info using a list of examples.

        Args:
            examples: List of examples

        Each example has the following keys:
            - input_tokens: List of input input ids
            - chain_tokens: List of chain token ids

        """
        return self._prepare_prompt_info(examples=examples, **kwargs)

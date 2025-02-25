from typing import List, Dict
import abc

import torch
from tokenized_cot_icl.core.args import IGNORE_INDEX


class BasePrompt(abc.ABC):
    @abc.abstractmethod
    def _get_intermediate_and_answer_tokens(self, chain_tokens: List[int]) -> List[int]:
        """Get the intermediate and answer tokens given the chain tokens."""
        ...

    @abc.abstractmethod
    def get_example_info(self, example: Dict[str, int]) -> Dict[str, List[int]]:
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

    def _prepare_prompt_info(self, examples: List[Dict[str, int]]) -> List[int]:
        """Prepare the prompt using a list of examples."""
        prefix_info = self.get_prefix_info()
        input_ids = prefix_info["prefix_input_ids"]
        attention_mask = prefix_info["prefix_attention_mask"]
        labels = prefix_info["prefix_labels"]
        for example in examples:
            example_info = self.get_example_info(example=example)
            input_ids.extend(example_info["example_input_ids"])
            attention_mask.extend(example_info["example_attention_mask"])
            labels.extend(example_info["example_labels"])

        assert len(input_ids) == len(labels)
        assert len(input_ids) == len(attention_mask)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    def _prepare_cot_eval_info(self, examples: List[Dict[str, int]]) -> List[int]:
        """Prepare the prompt using a list of examples for non teacher forcing based
        evaluation of the outputs.
        """
        prefix_info = self.get_prefix_info()
        input_ids = prefix_info["prefix_input_ids"]
        attention_mask = prefix_info["prefix_attention_mask"]
        last_example_cot = []
        num_examples = len(examples)
        for idx, example in enumerate(examples):
            example_info = self.get_example_info(example=example)
            if idx < num_examples - 1:
                example_input_ids = example_info["example_input_ids"]
                example_attention_mask = example_info["example_attention_mask"]
            else:
                example_input_ids = example_info["example_input_ids"][
                    : example_info["cot_eval_input_mask_length"]
                ]
                example_attention_mask = [1] * len(example_input_ids)
                last_example_cot = [
                    label
                    for label in example_info["example_labels"]
                    if label != IGNORE_INDEX
                ]

            input_ids.extend(example_input_ids)
            attention_mask.extend(example_attention_mask)

        assert len(input_ids) == len(attention_mask)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "last_example_cot": torch.tensor(last_example_cot),
        }

    def prepare(self, examples: List[Dict[str, int]]) -> List[int]:
        """Prepare the train info and cot eval info using a list of examples.

        Args:
            examples: List of examples

        Each example has the following keys:
            - input_tokens: List of input input ids
            - chain_tokens: List of chain token ids

        """
        return {
            **self._prepare_prompt_info(examples=examples),
            "cot_eval": self._prepare_cot_eval_info(examples=examples),
        }

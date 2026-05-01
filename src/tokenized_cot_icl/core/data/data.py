"""Token generator"""

import logging
from copy import deepcopy
from functools import lru_cache
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from tokenized_cot_icl.core.args import IGNORE_INDEX, Args
from tokenized_cot_icl.core.data.dag import DAG_REGISTRY
from tokenized_cot_icl.core.data.recipes import RECIPE_REGISTRY
from tokenized_cot_icl.core.data.token_processor import TokenProcessorCache, get_activation_fn
from tokenized_cot_icl.core.prompts.strategies.base import BasePrompt
from tokenized_cot_icl.core.prompts.strategies.registry import PROMPT_STRATEGY_REGISTRY
from tokenized_cot_icl.core.utils import set_random_seed


@lru_cache(maxsize=1)
def get_cached_embeddings(vocab_size: int, n_dims: int, std: float, seed: int) -> nn.Embedding:
    set_random_seed(seed)
    embedding = nn.Embedding(vocab_size, n_dims)
    embedding.weight.data.normal_(mean=0.0, std=std)
    return embedding


@lru_cache(maxsize=1)
def get_cached_adj_list(args: Args) -> list:
    assert args.ablation_fixed_dag, "This function is only for fixed DAG ablation"
    dag_cls = DAG_REGISTRY[args.dag_strategy]
    dag = dag_cls(n_inputs=args.n_inputs, n_parents=args.n_parents, chain_length=args.chain_length)
    return dag.generate_adj_list()


class TokenizedDataset(Dataset):
    def __init__(self, args: Args):
        self.args = args
        self.embeddings = get_cached_embeddings(
            vocab_size=self.args.vocab_size,
            n_dims=self.args.n_dims,
            std=self.args.data_initializer_range,
            seed=self.args.seed,
        )
        self.activation_fn = get_activation_fn(activation=self.args.activation)
        # warm up the caches across all ranks (since the same seed is being used at init)
        self.token_processor_cache = TokenProcessorCache(
            maxsize=self.args.num_unique_H,
            n_dims=self.args.n_dims,
            num_layers=self.args.H_num_layers,
            activation=self.args.activation,
        )
        # initialize DAG, prompt and cot mixer classes (if applicable)
        self.dag_cls = DAG_REGISTRY[self.args.dag_strategy]
        self.prompt: BasePrompt = PROMPT_STRATEGY_REGISTRY[self.args.prompt_strategy](args=self.args)
        self.set_cot_example_prob_recipe()
        if args.ablation_fixed_H:
            self.token_processor_cache.warmup(seed=args.seed)
        if args.ablation_fixed_dag:
            _ = get_cached_adj_list(self.args)

    def set_cot_example_prob_recipe(self) -> None:
        self.cot_example_prob_recipe = None
        if self.args.cot_example_prob_recipe_info:
            recipe_clz = RECIPE_REGISTRY[self.args.cot_example_prob_recipe_info["type"]]
            recipe_kwargs = deepcopy(self.args.cot_example_prob_recipe_info)
            recipe_kwargs["n_prompts"] = self.__len__()
            self.cot_example_prob_recipe = recipe_clz(**recipe_kwargs)

    def create_token_processors(self, n_inputs: int, n_parents: int, chain_length: int) -> None:
        assert n_parents <= n_inputs, "the number of parent tokens cannot be more than the input tokens"
        self.token_processors = nn.ModuleList([self.token_processor_cache.sample() for _ in range(chain_length)])

    def get_output_token(self, available_tokens: list, adj_list: list, chain_idx: int) -> int:
        parent_indices = adj_list[chain_idx]
        parent_tokens = [available_tokens[idx] for idx in parent_indices]
        parent_embeddings = self.embeddings(torch.tensor(parent_tokens))  # shape: (self.args.M, self.args.n_dims)

        # apply token processors to the parent embeddings
        token_processor = self.token_processors[chain_idx]  # shape: (self.args.n_dims, self.args.n_dims)
        output = token_processor(parent_embeddings)  # shape: (self.args.M, self.args.n_dims)
        output = output.mean(dim=0)  # shape: (self.args.n_dims)
        output = self.activation_fn(output)  # shape: (self.args.n_dims)
        # project the output to the vocabulary size
        logits = torch.matmul(output, self.embeddings.weight.T)  # shape: (self.args.V)
        output_token = torch.argmax(logits).item()
        # if the output token is one of the reserved tokens, then we choose the next best token
        while output_token in self.args.reserved_token_ids.values():
            logits[output_token] = float("-inf")
            output_token = torch.argmax(logits).item()
        return output_token

    def _generate_example(self, adj_list: list, n_inputs: int, chain_length: int) -> list:
        # add self.args.N integers (token ids) as input
        assert len(adj_list) == chain_length
        effective_vocab_size = self.args.vocab_size - len(self.args.reserved_token_ids)
        input_tokens = np.random.randint(effective_vocab_size, size=n_inputs).tolist()
        available_tokens = deepcopy(input_tokens)
        chain_tokens = []
        for chain_idx in range(chain_length):
            output_token = self.get_output_token(
                available_tokens=available_tokens,
                adj_list=adj_list,
                chain_idx=chain_idx,
            )
            assert output_token not in self.args.reserved_token_ids.values(), (
                f"output token: {output_token} is one of the reserved tokens: {self.args.reserved_token_ids}"
            )
            # add the output token to the input tokens
            available_tokens.append(output_token)
            chain_tokens.append(output_token)

        example = {"input_tokens": input_tokens, "chain_tokens": chain_tokens}

        return example

    def generate(
        self,
        n_inputs: int,
        n_parents: int,
        chain_length: int,
        n_examples: int,
        **prompt_kwargs,
    ) -> dict:
        self.create_token_processors(n_inputs=n_inputs, n_parents=n_parents, chain_length=chain_length)
        if self.args.ablation_fixed_dag:
            adj_list = get_cached_adj_list(self.args)
        else:
            dag = self.dag_cls(
                n_inputs=n_inputs,
                n_parents=n_parents,
                chain_length=chain_length,
            )
            adj_list = dag.generate_adj_list()
        examples = []
        for _ in range(n_examples):
            examples.append(self._generate_example(adj_list=adj_list, n_inputs=n_inputs, chain_length=chain_length))

        prompt_info = self.prompt.prepare(examples=examples, **prompt_kwargs)
        return {
            **prompt_info,
            "adj_list": torch.tensor(adj_list),
        }

    def __len__(self):
        return self.args.n_tasks

    def _sample_params(self, index: int):
        n_inputs = int(np.random.choice(self.args.n_input_choices))
        n_parents = int(np.random.choice(self.args.n_parent_choices))
        # limit n_parents to be less than or equal to n_inputs
        n_parents = min(n_parents, n_inputs)
        chain_length = int(np.random.choice(self.args.chain_length_choices))
        n_examples = int(np.random.choice(self.args.n_example_choices))
        params = {
            "n_inputs": n_inputs,
            "n_parents": n_parents,
            "chain_length": chain_length,
            "n_examples": n_examples,
        }
        if self.cot_example_prob_recipe is not None:
            params["cot_example_prob"] = self.cot_example_prob_recipe.get_value(prompt_index=index)
        return params

    def __getitem__(self, index):
        # generate a new example on the fly to avoid io bottleneck
        # the torch dataloader can prefetch the next example and save time

        # sample the n_inputs, n_parents, chain_length, n_examples for diversity

        return self.generate(**self._sample_params(index=index))


class EvalTokenizedDataset(TokenizedDataset):
    # since we want the eval dataset to be the same for every evaluation
    # we don't need to generate new examples every time.
    def __init__(self, args: Args):
        super().__init__(args=args)
        # create all records at once and store them in memory
        # use a seed offset to avoid the same examples as the training dataset
        set_random_seed(self.args.seed + 1000)
        logging.info(f"Generating evaluation dataset of size: {self.args.n_eval_tasks}")
        self.data = [self.generate(**self._sample_params(index=index)) for index in tqdm(range(self.args.n_eval_tasks))]

    def __len__(self):
        return self.args.n_eval_tasks

    def __getitem__(self, index):
        return self.data[index]


def special_token_collate_fn(batch: Dict, pad_token_id: int):
    """For a given batch of inputs:
    1. we need to pad the inputs to the maximum length. (use left padding)
    2. set attention mask to 1 for all non-padded tokens and 0 for padded tokens.
    """
    batch_input_ids = [item["input_ids"] for item in batch]
    batch_attention_masks = [item["attention_mask"] for item in batch]
    batch_labels = [item["labels"] for item in batch]
    batch_num_cot_examples = [item["num_cot_examples"] for item in batch]
    batch_num_standard_examples = [item["num_standard_examples"] for item in batch]

    batch_cot_eval_input_ids = [item["cot_eval"]["input_ids"] for item in batch]
    batch_cot_eval_attention_masks = [item["cot_eval"]["attention_mask"] for item in batch]
    batch_cot_eval_le_labels = [item["cot_eval"]["last_example_cot"] for item in batch]

    # no need to pad adj_list since they are of the same length
    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []

    # pad the training inputs
    max_input_length = max([len(input_ids) for input_ids in batch_input_ids])
    for input_ids, attention_mask, labels in zip(batch_input_ids, batch_attention_masks, batch_labels):
        pad_length = max_input_length - len(input_ids)
        padded_input_ids.append([pad_token_id] * pad_length + input_ids.tolist())
        padded_attention_masks.append([0] * pad_length + attention_mask.tolist())
        padded_labels.append([IGNORE_INDEX] * pad_length + labels.tolist())

    # pad the evaluation inputs
    padded_cot_eval_input_ids = []
    padded_cot_eval_attention_masks = []
    max_cot_eval_input_length = max([len(input_ids) for input_ids in batch_cot_eval_input_ids])
    for input_ids, attention_mask in zip(batch_cot_eval_input_ids, batch_cot_eval_attention_masks):
        pad_length = max_cot_eval_input_length - len(input_ids)
        padded_cot_eval_input_ids.append([pad_token_id] * pad_length + input_ids.tolist())
        padded_cot_eval_attention_masks.append([0] * pad_length + attention_mask.tolist())

    # pad the evaluation labels
    padded_cot_eval_le_labels = []
    max_cot_eval_le_length = max([len(labels) for labels in batch_cot_eval_le_labels])
    for labels in batch_cot_eval_le_labels:
        pad_length = max_cot_eval_le_length - len(labels)
        padded_cot_eval_le_labels.append([IGNORE_INDEX] * pad_length + labels.tolist())

    return {
        "input_ids": torch.tensor(padded_input_ids),
        "attention_mask": torch.tensor(padded_attention_masks),
        "labels": torch.tensor(padded_labels),
        "cot_eval": {
            "input_ids": torch.tensor(padded_cot_eval_input_ids),
            "attention_mask": torch.tensor(padded_cot_eval_attention_masks),
            "last_example_cot": torch.tensor(padded_cot_eval_le_labels),
        },
        "num_cot_examples": torch.tensor(batch_num_cot_examples),
        "num_standard_examples": torch.tensor(batch_num_standard_examples),
    }

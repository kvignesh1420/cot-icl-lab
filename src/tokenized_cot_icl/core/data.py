"""Token generator"""

import logging
from copy import deepcopy
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from tokenized_cot_icl.core.args import Args
from tokenized_cot_icl.core.dag import DAG_REGISTRY
from tokenized_cot_icl.core.prompts.base import BasePrompt
from tokenized_cot_icl.core.prompts.registry import PROMPT_REGISTRY
from tokenized_cot_icl.core.token_processor import TokenProcessor, get_activation_fn


@lru_cache(maxsize=1)
def get_cached_embeddings(vocab_size: int, n_dims: int, std: float) -> nn.Embedding:
    embedding = nn.Embedding(vocab_size, n_dims)
    embedding.weight.data.normal_(mean=0.0, std=std)
    return embedding


@lru_cache(maxsize=1)
def get_cached_token_processors(args: Args) -> nn.ModuleList:
    assert args.ablation_fixed_H, "This function is only for fixed H ablation"
    count = args.num_unique_H
    return [nn.ModuleList([TokenProcessor(args=args) for _ in range(args.chain_length)]) for _ in range(count)]


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
        )
        self.activation_fn = get_activation_fn(args=self.args)
        self.dag_cls = DAG_REGISTRY[self.args.dag_strategy]
        self.prompt: BasePrompt = PROMPT_REGISTRY[self.args.prompt_strategy](args=self.args)
        # warm up the caches across all ranks (since the same seed is being used at init)
        if args.ablation_fixed_H:
            _ = get_cached_token_processors(self.args)
        if args.ablation_fixed_dag:
            _ = get_cached_adj_list(self.args)

    def create_token_processors(self) -> None:
        assert self.args.n_parents <= self.args.n_inputs, (
            "the number of parent tokens cannot be more than the input tokens"
        )
        # Linear layers for each sampled input combination.
        # prepare the same token processors if ablation_fixed_H is enabled
        if self.args.ablation_fixed_H:
            token_processors_choices = get_cached_token_processors(self.args)
            self.token_processors = token_processors_choices[np.random.randint(self.args.num_unique_H)]
        else:
            self.token_processors = nn.ModuleList(
                [TokenProcessor(args=self.args) for _ in range(self.args.chain_length)]
            )

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
        while output_token in self.args.reserved_token_ids:
            logits[output_token] = float("-inf")
            output_token = torch.argmax(logits).item()
        return output_token

    def _generate_example(self, adj_list) -> list:
        # add self.args.N integers (token ids) as input
        assert len(adj_list) == self.args.chain_length
        effective_vocab_size = self.args.vocab_size - len(self.args.reserved_token_ids)
        input_tokens = np.random.randint(effective_vocab_size, size=self.args.n_inputs).tolist()
        available_tokens = deepcopy(input_tokens)
        chain_tokens = []
        for chain_idx in range(self.args.chain_length):
            output_token = self.get_output_token(
                available_tokens=available_tokens,
                adj_list=adj_list,
                chain_idx=chain_idx,
            )
            assert output_token not in self.args.reserved_token_ids, (
                f"output token: {output_token} is one of the reserved tokens: {self.args.reserved_token_ids}"
            )
            # add the output token to the input tokens
            available_tokens.append(output_token)
            chain_tokens.append(output_token)

        example = {"input_tokens": input_tokens, "chain_tokens": chain_tokens}

        return example

    def generate(self) -> dict:
        self.create_token_processors()
        if self.args.ablation_fixed_dag:
            adj_list = get_cached_adj_list(self.args)
        else:
            dag = self.dag_cls(
                n_inputs=self.args.n_inputs,
                n_parents=self.args.n_parents,
                chain_length=self.args.chain_length,
            )
            adj_list = dag.generate_adj_list()
        examples = []
        for _ in range(self.args.n_examples):
            examples.append(self._generate_example(adj_list=adj_list))

        prompt_info = self.prompt.prepare(examples=examples)
        return {
            **prompt_info,
            "adj_list": torch.tensor(adj_list),
        }

    def __len__(self):
        return self.args.n_tasks

    def __getitem__(self, index):
        # generate a new example on the fly to avoid io bottleneck
        # the torch dataloader can prefetch the next example and save time
        return self.generate()


class EvalTokenizedDataset(TokenizedDataset):
    # since we want the eval dataset to be the same for every evaluation
    # we don't need to generate new examples every time.
    def __init__(self, args: Args):
        super().__init__(args=args)
        # create all records at once and store them in memory
        logging.info(f"Generating evaluation dataset of size: {self.args.n_eval_tasks}")
        self.data = [self.generate() for _ in tqdm(range(self.args.n_eval_tasks))]

    def __len__(self):
        return self.args.n_eval_tasks

    def __getitem__(self, index):
        return self.data[index]

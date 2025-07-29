import abc

import torch

from tokenized_cot_icl.core.args import Args, EvalArgs
from tokenized_cot_icl.core.data.data import EvalTokenizedDataset


class InferenceEvaluator(abc.ABC):
    def __init__(self, eval_args: EvalArgs):
        self.eval_args = eval_args
        self.setup()

    def _load_eval_dataset(self):
        """Load the eval dataset. If data_path is not provided, generate the dataset on the fly.
        This base implementation can be used to first prepare the dataset and then can be transformed
        as needed by the derived classes.
        """
        if self.eval_args.data_path is not None:
            print(f"Loading eval dataset from {self.eval_args.data_path}")
            self.eval_dataset = torch.load(self.eval_args.data_path)
        else:
            print("Eval dataset path is not provided, generating data on the fly.")
            chain_length_choices = (self.eval_args.chain_len,)
            n_input_choices = (self.eval_args.n_input,)
            n_parent_choices = (self.eval_args.n_parent,)
            n_example_choices = (self.eval_args.n_examples,)
            args = Args(
                prompt_strategy="cot_special_token",
                enable_special_tokens=True,
                chain_length_choices=chain_length_choices,
                n_input_choices=n_input_choices,
                n_parent_choices=n_parent_choices,
                n_example_choices=n_example_choices,
                n_eval_tasks=self.eval_args.n_eval_tasks,
                ablation_fixed_H=self.eval_args.ablation_fixed_H,
                num_unique_H=self.eval_args.num_unique_H,
            )
            self.eval_dataset = EvalTokenizedDataset(args)

    def setup(self):
        self._load_eval_dataset()
        self._setup_model()

    @abc.abstractmethod
    def _setup_model(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

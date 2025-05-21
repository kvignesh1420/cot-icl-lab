import abc
import json
import os

import torch

from tokenized_cot_icl.core.args import Args


class InferenceEvaluator(abc.ABC):
    def __init__(self, model_base_dir: str, checkpoint: int):
        self.model_base_dir = model_base_dir
        self.checkpoint = checkpoint
        self.setup()

    def _load_args(self):
        args_path = os.path.join(self.model_base_dir, "args.json")
        with open(args_path, "r") as f:
            args_dict = json.load(f)
        self.args = Args(**args_dict)

    @abc.abstractmethod
    def _setup_model(self):
        pass

    def _load_eval_dataset(self):
        self.eval_dataset = torch.load(
            os.path.join(self.model_base_dir, "eval_dataset", "eval_data.pt")
        )

    def setup(self):
        self._load_args()
        self._setup_model()
        self._load_eval_dataset()

    @abc.abstractmethod
    def evaluate(self):
        pass

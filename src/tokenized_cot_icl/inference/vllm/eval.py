import argparse
import json
import os

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams, TokensPrompt

from tokenized_cot_icl.core.args import Args


class VLLMEvaluator:
    def __init__(self, output_dir: str, checkpoint: int):
        self.output_dir = output_dir
        self.checkpoint = checkpoint
        self.setup()

    def _load_args(self):
        args_path = os.path.join(self.output_dir, "args.json")
        with open(args_path, "r") as f:
            args_dict = json.load(f)
        self.args = Args(**args_dict)

    def _setup_model(self):
        model_path = os.path.join(self.output_dir, "checkpoints", str(self.checkpoint))
        self.model = LLM(
            model_path,
            dtype=torch.float32,
            skip_tokenizer_init=True,
            enable_chunked_prefill=False,
        )
        self.sampling_params = SamplingParams(
            n=1,
            max_tokens=self.args.chain_length,
            skip_special_tokens=False,
            temperature=0.0,
        )

    def _load_eval_dataset(self):
        self.eval_dataset = torch.load(os.path.join(self.output_dir, "eval_dataset", "eval_data.pt"))

    def setup(self):
        self._load_args()
        self._setup_model()
        self._load_eval_dataset()

    def evaluate(self):
        answer_pred_info = {"correct": 0.0, "total": len(self.eval_dataset)}

        for item in tqdm(self.eval_dataset):
            prompt = item["cot_eval"]["input_ids"].tolist()

            o = self.model.generate(
                TokensPrompt({"prompt_token_ids": prompt}),
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            pred_ids = o[0].outputs[0].token_ids
            pred_answer = pred_ids[-1]
            gt_answer = item["cot_eval"]["last_example_cot"].tolist()[-1]
            if pred_answer == gt_answer:
                answer_pred_info["correct"] += 1

        accuracy = answer_pred_info["correct"] / answer_pred_info["total"]
        return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser_args = parser.parse_args()

    evaluator = VLLMEvaluator(output_dir=parser_args.output_dir, checkpoint=parser_args.checkpoint)
    answer_accuracy = evaluator.evaluate()
    print(f"Answer token accuracy: {answer_accuracy}")

import argparse
import json
import os

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams, TokensPrompt
from sglang import Engine

from tokenized_cot_icl.core.args import Args


class InferenceEvaluator:
    def __init__(self, output_dir: str, checkpoint: int, inference_engine: str):
        self.output_dir = output_dir
        self.checkpoint = checkpoint
        self.inference_engine = inference_engine
        self.setup()

    def _load_args(self):
        args_path = os.path.join(self.output_dir, "args.json")
        with open(args_path, "r") as f:
            args_dict = json.load(f)
        self.args = Args(**args_dict)

    def _setup_model(self):
        if self.checkpoint == "final":
            model_path = os.path.join(self.output_dir, "final_model")
        else:
            model_path = os.path.join(self.output_dir, "checkpoints", self.checkpoint)
        if self.inference_engine == "vllm":
            self.__setup_vllm_model(model_path)
        elif self.inference_engine == "sglang":
            self.__setup_sglang_model(model_path)
        else:
            raise ValueError("Invalid inference engine.")

    def __setup_vllm_model(self, model_path):
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

    def __setup_sglang_model(self, model_path):
        self.model = Engine(
            model_path=model_path,
            tokenizer_path=model_path,
            attention_backend="torch_native",
            sampling_backend="torch_native",
            skip_tokenizer_init=True,
            chunked_prefill_size=-1,
        )
        self.sampling_params = {
            "max_new_tokens": self.args.chain_length,
            "skip_special_tokens": False,
            "temperature": 0.0,
        }


    def _load_eval_dataset(self):
        self.eval_dataset = torch.load(os.path.join(self.output_dir, "eval_dataset", "eval_data.pt"))

    def setup(self):
        self._load_args()
        self._setup_model()
        self._load_eval_dataset()

    def evaluate(self):
        answer_pred_info = {"correct": 0.0, "total": len(self.eval_dataset)}
        if self.inference_engine == "vllm":
            eval_func = self._evaluate_vllm_one_step
        elif self.inference_engine == "sglang":
            eval_func = self._evaluate_sglang_one_step
        else:
            raise ValueError("Invalid inference engine.")

        for item in tqdm(self.eval_dataset):
            prompt = item["cot_eval"]["input_ids"].tolist()

            pred_answer = eval_func(prompt)
            gt_answer = item["cot_eval"]["last_example_cot"].tolist()[-1]
            if pred_answer == gt_answer:
                answer_pred_info["correct"] += 1

        accuracy = answer_pred_info["correct"] / answer_pred_info["total"]
        return accuracy

    def _evaluate_vllm_one_step(self, prompt: list[int]):
        o = self.model.generate(
                TokensPrompt({"prompt_token_ids": prompt}),
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

        pred_ids = o[0].outputs[0].token_ids
        pred_answer = pred_ids[-1]
        return pred_answer

    def _evaluate_sglang_one_step(self, prompt: list[int]):
        o = self.model.generate(
                input_ids=prompt,
                sampling_params=self.sampling_params
            )

        pred_ids = o['token_ids']
        pred_answer = pred_ids[-1]
        return pred_answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--inference-engine", type=str, choices=["vllm", "sglang"], default="vllm")
    parser_args = parser.parse_args()

    evaluator = InferenceEvaluator(output_dir=parser_args.output_dir, checkpoint=parser_args.checkpoint, inference_engine=parser_args.inference_engine)
    answer_accuracy = evaluator.evaluate()
    print(f"Answer token accuracy: {answer_accuracy}")

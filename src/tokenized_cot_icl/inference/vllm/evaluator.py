import argparse

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams, TokensPrompt

from tokenized_cot_icl.core.args import EvalArgs
from tokenized_cot_icl.inference.base.base_evaluator import InferenceEvaluator


class VLLMEvaluator(InferenceEvaluator):
    def __init__(self, eval_args: EvalArgs):
        super().__init__(eval_args=eval_args)

    def _setup_model(self):
        self.model = LLM(
            self.eval_args.model_path,
            dtype=torch.float32,
            skip_tokenizer_init=True,
            enable_chunked_prefill=False,
        )
        self.sampling_params = SamplingParams(
            n=self.eval_args.num_output_seqs,
            max_tokens=self.eval_args.max_pred_tokens,
            temperature=self.eval_args.temperature,
            skip_special_tokens=False,
        )

    def evaluate(self):
        outputs = []
        answer_pred_info = {"correct": 0.0, "total": len(self.eval_dataset)}
        for item in tqdm(self.eval_dataset):
            prompt = item["cot_eval"]["input_ids"].tolist()

            o = self.model.generate(
                TokensPrompt({"prompt_token_ids": prompt}),
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            pred_ids = o[0].outputs[0].token_ids
            outputs.append(o[0])
            pred_answer = pred_ids[-1]
            gt_answer = item["cot_eval"]["last_example_cot"].tolist()[-1]
            if pred_answer == gt_answer:
                answer_pred_info["correct"] += 1

        accuracy = answer_pred_info["correct"] / answer_pred_info["total"]
        return accuracy, outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser_args = parser.parse_args()

    eval_args = EvalArgs(
        model_path=parser_args.model_path,
        data_path=parser_args.data_path,
    )

    evaluator = VLLMEvaluator(eval_args=eval_args)
    answer_accuracy = evaluator.evaluate()
    print(f"Answer token accuracy: {answer_accuracy}")

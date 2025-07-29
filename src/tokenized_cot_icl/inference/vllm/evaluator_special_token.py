import argparse
import os
from collections import Counter
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams, TokensPrompt

from tokenized_cot_icl.core.args import EvalArgs
from tokenized_cot_icl.core.prompts.transformations.drop_cot_tokens import drop_k_cot_tokens_fn
from tokenized_cot_icl.inference.base.base_evaluator import InferenceEvaluator

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"


def cot_inference_transform_fn(transformation: str, k: int):
    transform_fn = partial(drop_k_cot_tokens_fn, transformation=transformation, k=k)
    return transform_fn


class VLLMSpecialTokenEvaluator(InferenceEvaluator):
    def __init__(self, eval_args: EvalArgs):
        super().__init__(eval_args=eval_args)

    def _setup_model(self):
        self.model = LLM(
            self.eval_args.model_path,
            dtype=torch.float32,
            skip_tokenizer_init=False,
            enable_chunked_prefill=False,
        )

    def evaluate(self, **kwargs):
        if "drop_strategy" in kwargs and "drop_K" in kwargs:
            drop_strategy = kwargs["drop_strategy"]
            drop_K = kwargs["drop_K"]
            transform_fn = cot_inference_transform_fn(transformation=drop_strategy, k=drop_K)
            transformed_eval_dataset = DataLoader(
                self.eval_dataset,
                collate_fn=lambda batch: transform_fn(batch=batch),
            )
        else:
            transformed_eval_dataset = DataLoader(
                self.eval_dataset,
                collate_fn=lambda batch: batch,
            )

        outputs = []
        gt_last_example_cots = []
        answer_pred_info = {"correct": 0.0, "total": len(transformed_eval_dataset)}
        for item in tqdm(transformed_eval_dataset):
            assert len(item) == 1
            prompt = item[0]["cot_eval"]["input_ids"].tolist()
            # force think token or answer token
            if self.eval_args.force_think_token:
                prompt.append(item[0]["think_start_token_id"])
            elif self.eval_args.force_answer_token:
                prompt.append(item[0]["answer_start_token_id"])

            sampling_params = SamplingParams(
                n=self.eval_args.num_output_seqs,
                max_tokens=self.eval_args.max_pred_tokens,
                temperature=self.eval_args.temperature,
                stop_token_ids=[item[0]["eos_token_id"]],
                skip_special_tokens=False,
            )

            o = self.model.generate(
                TokensPrompt({"prompt_token_ids": prompt}),
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            outputs.append(o[0])
            pred_ids = []
            for i in range(self.eval_args.num_output_seqs):
                o_ids = o[0].outputs[i].token_ids
                # capture only the instruction followed response
                if len(o_ids) >= 3 and o_ids[-2] == item[0]["answer_end_token_id"]:
                    pred_id = o_ids[-3]
                else:
                    pred_id = -100
                pred_ids.append(pred_id)

            pred_answer = Counter(pred_ids).most_common(1)[0][0]
            gt_last_example_cot = item[0]["cot_eval"]["last_example_cot"].tolist()
            gt_last_example_cots.append(gt_last_example_cot)
            gt_answer = gt_last_example_cot[-3]
            if pred_answer == gt_answer:
                answer_pred_info["correct"] += 1

        accuracy = answer_pred_info["correct"] / answer_pred_info["total"]
        return accuracy, {"outputs": outputs, "gt_last_example_cots": gt_last_example_cots}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None, required=False)
    parser.add_argument(
        "--num_output_seqs",
        type=int,
        default=1,
        desc="Number of output sequences to generate. As of now, a value > 1 would result to majority voting.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--force_think_token",
        action="store_true",
        help="Force the model to think and generate the chain tokens before answering.",
    )
    parser.add_argument(
        "--force_answer_token",
        action="store_true",
        help="Force the model to directly generate an answer without CoT.",
    )
    parser.add_argument(
        "--num_unique_H",
        type=int,
        default=1,
        help="Number of unique H processors to use for generating the data.",
    )
    parser.add_argument(
        "--drop_strategy",
        type=str,
        default="first_k",
        help="Drop strategy to use for generating the data.",
    )
    parser.add_argument(
        "--drop_K",
        type=int,
        default=0,
        help="Number of examples to drop the think tokens (in each prompt).",
    )
    parser.add_argument(
        "--n_input",
        type=int,
        default=4,
        help="Number of input tokens to use for generating the example.",
    )
    parser.add_argument(
        "--n_parent",
        type=int,
        default=4,
        help="Number of parent tokens to use for generating the example.",
    )
    parser.add_argument(
        "--chain_len",
        type=int,
        default=4,
        help="Chain length to use for generating the example.",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=40,
        help="Number of examples to use for generating the prompt.",
    )
    parser_args = parser.parse_args()
    print(parser_args)

    eval_args = EvalArgs(
        model_path=parser_args.model_path,
        data_path=parser_args.data_path,
        num_output_seqs=parser_args.num_output_seqs,
        temperature=parser_args.temperature,
        force_think_token=parser_args.force_think_token,
        force_answer_token=parser_args.force_answer_token,
        num_unique_H=parser_args.num_unique_H,
        n_input=parser_args.n_input,
        n_parent=parser_args.n_parent,
        chain_len=parser_args.chain_len,
        n_examples=parser_args.n_examples,
    )

    # run the evaluator
    evaluator = VLLMSpecialTokenEvaluator(eval_args=eval_args)
    answer_accuracy = evaluator.evaluate(
        **{
            "drop_strategy": parser_args.drop_strategy,
            "drop_K": parser_args.drop_K,
        }
    )
    print(f"Answer token accuracy: {answer_accuracy}")

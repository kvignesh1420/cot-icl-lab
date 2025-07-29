import argparse
import os
from dataclasses import asdict

from tokenized_cot_icl.core.args import EvalArgs
from tokenized_cot_icl.core.utils import create_metric_logger
from tokenized_cot_icl.inference.task_card import EVAL_TASK_CARD


def bulk_run(eval_args: EvalArgs):
    import torch

    from tokenized_cot_icl.inference.vllm.evaluator_special_token import VLLMSpecialTokenEvaluator

    print(f"Running bulk eval with eval_args: {asdict(eval_args)}")
    if not os.path.exists(eval_args.output_dir):
        os.makedirs(eval_args.output_dir)

    metric_logger = create_metric_logger(eval_args)

    acc_info = {"config": asdict(eval_args)}
    acc_info["drop_strategy"] = {}
    inputs_and_predictions = {}

    for drop_strategy in ["random"]:
        acc_info["drop_strategy"][drop_strategy] = {}
        inputs_and_predictions[drop_strategy] = {}
        for drop_K in [i for i in range(0, eval_args.n_examples - 10 + 1, 10)] + [eval_args.n_examples - 1]:
            evaluator = VLLMSpecialTokenEvaluator(eval_args=eval_args)
            answer_accuracy, outputs = evaluator.evaluate(**{"drop_strategy": drop_strategy, "drop_K": drop_K})

            acc_info["drop_strategy"][drop_strategy][drop_K] = answer_accuracy
            inputs_and_predictions[drop_strategy][drop_K] = outputs
            print(f"drop_strategy={drop_strategy}, K={drop_K}, answer_accuracy={answer_accuracy}")
            metric_logger.log_metrics({f"drop_strategy_{drop_strategy}/K_{drop_K}": answer_accuracy}, step=drop_K)

            del evaluator

    acc_output_file = os.path.join(eval_args.output_dir, f"evals_{eval_args_str}.pt")
    inputs_and_predictions_file = os.path.join(eval_args.output_dir, f"inputs_and_predictions_{eval_args_str}.pt")
    torch.save(acc_info, acc_output_file)
    metric_logger.log_artifact(local_path=acc_output_file)
    torch.save(inputs_and_predictions, inputs_and_predictions_file)
    metric_logger.log_artifact(local_path=inputs_and_predictions_file)

    metric_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_card_idx", type=int, required=True)
    parser_args = parser.parse_args()

    task_card_idx = parser_args.task_card_idx

    eval_args = EVAL_TASK_CARD[task_card_idx]
    bulk_run(eval_args=eval_args)

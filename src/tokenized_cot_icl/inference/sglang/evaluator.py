import argparse
import os

from sglang import Engine
from tqdm import tqdm

from tokenized_cot_icl.inference.base_evaluator import InferenceEvaluator


class SGLangEvaluator(InferenceEvaluator):
    def __init__(self, model_base_dir: str, checkpoint: int):
        super().__init__(model_base_dir, checkpoint)

    def _setup_model(self):
        if self.checkpoint == "final":
            model_path = os.path.join(self.model_base_dir, "final_model")
        else:
            model_path = os.path.join(
                self.model_base_dir, "checkpoints", self.checkpoint
            )
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

    def evaluate(self):
        answer_pred_info = {"correct": 0.0, "total": len(self.eval_dataset)}
        for item in tqdm(self.eval_dataset):
            prompt = item["cot_eval"]["input_ids"].tolist()
            o = self.model.generate(
                input_ids=prompt, sampling_params=self.sampling_params
            )

            pred_ids = o["token_ids"]
            pred_answer = pred_ids[-1]
            gt_answer = item["cot_eval"]["last_example_cot"].tolist()[-1]
            if pred_answer == gt_answer:
                answer_pred_info["correct"] += 1

        accuracy = answer_pred_info["correct"] / answer_pred_info["total"]
        return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser_args = parser.parse_args()

    evaluator = SGLangEvaluator(
        model_base_dir=parser_args.model_base_dir, checkpoint=parser_args.checkpoint
    )
    answer_accuracy = evaluator.evaluate()
    print(f"Answer token accuracy: {answer_accuracy}")

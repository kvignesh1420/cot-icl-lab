import argparse

from sglang import Engine
from tqdm import tqdm

from tokenized_cot_icl.core.args import EvalArgs
from tokenized_cot_icl.inference.base.base_evaluator import InferenceEvaluator


class SGLangEvaluator(InferenceEvaluator):
    def __init__(self, eval_args: EvalArgs):
        super().__init__(eval_args=eval_args)

    def _setup_model(self):
        self.model = Engine(
            model_path=self.eval_args.model_path,
            tokenizer_path=self.eval_args.model_path,
            attention_backend="torch_native",
            sampling_backend="torch_native",
            skip_tokenizer_init=True,
            chunked_prefill_size=-1,
        )
        self.sampling_params = {
            "max_new_tokens": self.eval_args.max_pred_tokens,
            "temperature": self.eval_args.temperature,
            "skip_special_tokens": False,
        }

    def evaluate(self):
        outputs = []
        answer_pred_info = {"correct": 0.0, "total": len(self.eval_dataset)}
        for item in tqdm(self.eval_dataset):
            prompt = item["cot_eval"]["input_ids"].tolist()
            o = self.model.generate(input_ids=prompt, sampling_params=self.sampling_params)

            pred_ids = o["token_ids"]
            outputs.append(o)
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

    evaluator = SGLangEvaluator(eval_args=eval_args)
    answer_accuracy = evaluator.evaluate()
    print(f"Answer token accuracy: {answer_accuracy}")

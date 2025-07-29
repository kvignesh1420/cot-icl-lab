## Inference

In addition to using the `transformers.GenerationConfig` for small scale inference during the training runs, we also support [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) based evaluation of the trained model (or model checkpoints) to analyze the predictions.

To evaluate the model in different scenarios (e.g. length generalization and different thinking strategies) we provide task cards in `task_card.py`. One can extend the existing task cards to add a new evaluation. The evaluation task cards are defined as a dictionary called `EVAL_TASK_CARD` where the keys are integer values representing a single setup.

To run the evaluations, modify `EVAL_TASK_CARD` in `task_card.py` to reflect the intended setup.  Then invoke the evaluate script with task card key. For example, the following runs the evaluate script for the first task card:

```bash
(.venv) $ python -m tokenized_cot_icl.inference.evaluate --task_card_idx 0

```
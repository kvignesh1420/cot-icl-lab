## Inference with vLLM/SGLang

In addition to using the `transformers.GenerationConfig` for small scale inference during the training runs, we also support [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) based evaluation of the trained model (or model checkpoints) to analyze the predictions.

```bash
(.venv) $ pip install vllm # install suitable version
(.venv) $ pip install sglang # install suitable version
```

We provide an easy to extend example for calculating the answer token prediction accuracy as follows:

```bash
# for vllm
(.venv) $ cd src && python tokenized_cot_icl/inference/vllm/evaluator.py \
                        --output_dir /opt/cot-icl-lab/run_name \ # set the path
                        --checkpoint final  # either final or 1000, 2000 etc.

# for sglang
(.venv) $ cd src && python tokenized_cot_icl/inference/sglang/evaluator.py \
                        --output_dir /opt/cot-icl-lab/run_name \ # set the path
                        --checkpoint final  # either final or 1000, 2000 etc.
```

import argparse
import os

import torch
from vllm import LLM, SamplingParams


class VLLMEvaluator:
    def __init__(self, output_dir: str, checkpoint: int):
        self.output_dir = output_dir
        self.checkpoint = checkpoint

    def setup(self):
        model_path = os.path.join(self.output_dir, "checkpoints", str(self.checkpoint))
        self.model = LLM(
            model_path,
            dtype=torch.float32,
            skip_tokenizer_init=True,
            enable_chunked_prefill=False,
        )
        self.sampling_params = SamplingParams(
            n=1,
            max_tokens=100,
            skip_special_tokens=False,
            temperature=0.0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser_args = parser.parse_args()

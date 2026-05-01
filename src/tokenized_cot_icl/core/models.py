"""Models"""

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from tokenized_cot_icl.core.args import Args


def create_llama_model(args: Args):
    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        eos_token_id=args.eos_token_id,
        bos_token_id=args.bos_token_id,
        pad_token_id=args.pad_token_id,
        tie_word_embeddings=args.tie_word_embeddings,
        max_position_embeddings=args.max_position_embeddings,
        rope_scaling=args.rope_scaling,
        rope_theta=args.rope_theta,
    )
    model = LlamaForCausalLM(config)
    assert model.dtype == torch.float32
    return model


MODEL_REGISTRY = {"llama": create_llama_model}
